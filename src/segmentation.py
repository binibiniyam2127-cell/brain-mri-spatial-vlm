"""MONAI BraTS 3D brain tumor segmentation inference adapter.

Provides inference handling for pretrained MONAI BraTS 3D segmentation models.
Designed specifically for 4-channel multi-sequence inputs (T1, T1c, T2, FLAIR)
and outputs 3D segmentation masks comprising tumor-core (TC), whole-tumor (WT),
and enhancing-tumor (ET) subregions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.config import load_config
from src.mri_io import (
    load_nifti_volume,
    save_nifti_volume,
    save_processed_volume,
    validate_mri_volume,
)
from src.utils import ensure_directory, get_project_root, safe_filename, timestamped_filename


REQUIRED_CHANNELS = ["t1", "t1c", "t2", "flair"]

CHANNEL_ALIASES: Dict[str, str] = {
    "t1": "t1",
    "t1ce": "t1c",
    "t1c": "t1c",
    "t1gd": "t1c",
    "t2": "t2",
    "flair": "flair",
}


class MissingChannelError(ValueError):
    """Raised when the mandatory four MRI input modalities are not provided."""
    pass


class BraTSSegmenter:
    """Pretrained MONAI BraTS 3D brain tumor segmentation inference adapter.

    Documentation:
        The MONAI BraTS bundle architecture is designed for four co-registered,
        aligned structural MRI modalities:
          1. T1-weighted (T1)
          2. Post-contrast T1-weighted (T1c / T1ce)
          3. T2-weighted (T2)
          4. Fluid-Attenuated Inversion Recovery (FLAIR)

        The network outputs 3D multi-region segmentations corresponding to:
          - Label 1: Necrotic / Non-enhancing tumor core (NCR/NET)
          - Label 2: Peritumoral edema (ED)
          - Label 3: Enhancing tumor (ET)

        Subregions evaluated clinically/analytically include:
          - Tumor Core (TC = NCR + ET)
          - Whole Tumor (WT = NCR + ED + ET)
          - Enhancing Tumor (ET)
    """

    def __init__(
        self,
        model_dir: Optional[Union[str, Path]] = None,
        bundle_name: Optional[str] = None,
        device: Optional[str] = None,
        model: Optional[Any] = None,
        config_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize the BraTS segmentation model adapter.

        Args:
            model_dir: Directory containing pretrained weights or bundle. Defaults to
                'monai_brats.bundle_dir' in configs/project_config.yaml.
            bundle_name: Bundle name identifier. Defaults to 'monai_brats.bundle_name' in config.
            device: Compute device ('cuda' or 'cpu'). Defaults to automatic selection
                (uses CUDA if available, falling back to CPU).
            model: Optional pre-instantiated torch model (useful for unit testing/mocking).
            config_path: Optional path to project configuration YAML.
        """
        self.project_root = get_project_root()
        cfg = load_config(config_path)

        self.bundle_name = (
            bundle_name
            if bundle_name is not None
            else getattr(cfg.monai_brats, "bundle_name", "brats_mri_segmentation")
        )

        if model_dir is None:
            rel_bundle_dir = getattr(cfg.monai_brats, "bundle_dir", "models/monai_brats")
            self.model_dir = (self.project_root / rel_bundle_dir).resolve()
        else:
            self.model_dir = Path(model_dir).resolve()

        self.bundle_path = (self.model_dir / self.bundle_name).resolve()

        # Compute device selection: CPU by default, CUDA automatically when available
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except ImportError:
            has_cuda = False

        if device is not None:
            req_dev = device.lower()
            if "cuda" in req_dev and not has_cuda:
                self.device_str = "cpu"
            else:
                self.device_str = req_dev
        else:
            self.device_str = "cuda" if has_cuda else "cpu"

        self.model = model
        self.weights_path: Optional[Path] = None

        if self.model is None:
            self._init_model_from_weights()

    def _init_model_from_weights(self) -> None:
        """Locate weights and load model architecture if bundle or checkpoint is present."""
        if not self.model_dir.exists():
            return

        # Candidate paths within bundle directory or standalone checkpoint
        candidate_files = [
            self.bundle_path / "models" / "model.pt",
            self.bundle_path / "models" / "model.pth",
            self.model_dir / self.bundle_name / "models" / "model.pt",
            self.model_dir / self.bundle_name / "models" / "model.pth",
            self.model_dir / "models" / "model.pt",
            self.model_dir / "model.pt",
            self.model_dir / "model.pth",
            self.model_dir / "best_metric_model.pt",
        ]
        for cand in candidate_files:
            if cand.is_file():
                self.weights_path = cand
                break

        # Attempt bundle load via MONAI or standard PyTorch load
        if self.bundle_path.exists():
            try:
                import monai.bundle
                loaded = monai.bundle.load(
                    name=self.bundle_name,
                    bundle_dir=str(self.model_dir),
                    device=self.device_str,
                )
                if loaded is not None:
                    self.model = loaded
                    self.model.eval()
                    return
            except Exception:
                pass

        if self.weights_path is not None:
            try:
                import torch
                checkpoint = torch.load(str(self.weights_path), map_location=self.device_str)
                if isinstance(checkpoint, torch.nn.Module):
                    self.model = checkpoint
                    self.model.to(self.device_str)
                    self.model.eval()
            except Exception:
                pass

    def is_bundle_available(self) -> bool:
        """Check whether the MONAI BraTS bundle or weights are present locally."""
        return self.weights_path is not None and self.weights_path.is_file()

    def prepare_multichannel_input(
        self,
        inputs: Union[Dict[str, Union[str, Path, np.ndarray]], np.ndarray, Path, str],
        affine: Optional[np.ndarray] = None,
        save_to_processed: bool = False,
        processed_filename: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Validate and stack input modalities into a unified 4-channel volume (4, H, W, D).

        Enforces the presence of all 4 required BraTS channels (T1, T1c, T2, FLAIR).
        If any modality is missing, raises a clear non-medical MissingChannelError.

        Args:
            inputs: Multi-channel input in one of three forms:
                - Dict mapping modality name to file path or 3D NumPy array.
                - 4D NumPy array with shape (4, H, W, D) or (H, W, D, 4).
                - File path to an existing 4D NIfTI volume.
            affine: 4x4 spatial affine matrix (optional, inferred from files if available).
            save_to_processed: If True, saves the 4-channel input into data/processed/.
            processed_filename: Optional filename for processed volume.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (4D array of shape (4, H, W, D), affine matrix).

        Raises:
            MissingChannelError: If fewer than 4 channels are provided.
            ValueError: If input array shapes or dimensions are mismatched.
        """
        ref_affine = affine if affine is not None else np.eye(4, dtype=np.float64)

        # Case 1: Dictionary of individual modality paths or arrays
        if isinstance(inputs, dict):
            normalized_inputs: Dict[str, Union[str, Path, np.ndarray]] = {}
            for k, v in inputs.items():
                alias = CHANNEL_ALIASES.get(k.lower().strip())
                if alias:
                    normalized_inputs[alias] = v

            missing = [ch for ch in REQUIRED_CHANNELS if ch not in normalized_inputs]
            if missing:
                raise MissingChannelError(
                    f"MONAI BraTS segmentation requires 4 aligned input channels: {REQUIRED_CHANNELS}. "
                    f"Received channels: {list(normalized_inputs.keys())}. "
                    f"Missing channel(s): {missing}. Please provide all 4 aligned channels."
                )

            channel_arrays: List[np.ndarray] = []
            expected_shape: Optional[Tuple[int, ...]] = None

            for ch in REQUIRED_CHANNELS:
                val = normalized_inputs[ch]
                if isinstance(val, (str, Path)):
                    vol, file_aff, _ = load_nifti_volume(val)
                    ref_affine = file_aff
                elif isinstance(val, np.ndarray):
                    vol = val.astype(np.float32)
                    validate_mri_volume(vol)
                else:
                    raise TypeError(f"Invalid channel value type for '{ch}': {type(val).__name__}")

                if expected_shape is None:
                    expected_shape = vol.shape
                elif vol.shape != expected_shape:
                    raise ValueError(
                        f"Spatial dimension mismatch between modality channels: "
                        f"Channel '{ch}' has shape {vol.shape}, expected {expected_shape}."
                    )
                channel_arrays.append(vol)

            stacked = np.stack(channel_arrays, axis=0)  # Shape: (4, H, W, D)

        # Case 2: File path to 4D NIfTI or 3D NIfTI
        elif isinstance(inputs, (str, Path)):
            input_path = Path(inputs).resolve()
            try:
                vol, file_aff, meta = load_nifti_volume(input_path)
                ref_affine = file_aff
                # If loaded as 3D (single channel)
                if vol.ndim == 3:
                    raise MissingChannelError(
                        f"MONAI BraTS segmentation requires 4 aligned channels (T1, T1c, T2, FLAIR). "
                        f"The provided file '{input_path.name}' contains a single 3D volume (shape: {vol.shape}). "
                        f"Missing channels: ['t1c', 't2', 'flair']. Multi-channel alignment is required."
                    )
                stacked = vol
            except ValueError as e:
                # If load_nifti_volume rejected it because it was 4D, load directly with nibabel
                import nibabel as nib
                nii = nib.load(str(input_path))
                raw_data = np.asanyarray(nii.dataobj).astype(np.float32)
                ref_affine = nii.affine

                if raw_data.ndim == 4:
                    if raw_data.shape[0] == 4:
                        stacked = raw_data
                    elif raw_data.shape[-1] == 4:
                        stacked = np.moveaxis(raw_data, -1, 0)
                    else:
                        raise MissingChannelError(
                            f"MONAI BraTS segmentation requires 4 channels. "
                            f"Input has shape {raw_data.shape}. Channel count does not match 4."
                        )
                else:
                    raise MissingChannelError(
                        f"MONAI BraTS segmentation requires 4 channels (T1, T1c, T2, FLAIR). "
                        f"Input dimension is {raw_data.ndim}D with shape {raw_data.shape}. Missing channels: ['t1c', 't2', 'flair']."
                    )

        # Case 3: NumPy array directly passed
        elif isinstance(inputs, np.ndarray):
            arr = inputs.astype(np.float32)
            if arr.ndim == 3:
                raise MissingChannelError(
                    f"MONAI BraTS segmentation requires 4 aligned channels (T1, T1c, T2, FLAIR). "
                    f"Received a single 3D array of shape {arr.shape}. Missing 3 channels."
                )
            elif arr.ndim == 4:
                if arr.shape[0] == 4:
                    stacked = arr
                elif arr.shape[-1] == 4:
                    stacked = np.moveaxis(arr, -1, 0)
                else:
                    raise MissingChannelError(
                        f"MONAI BraTS segmentation requires 4 channels. "
                        f"Input 4D array has shape {arr.shape} with {arr.shape[0]} channels."
                    )
            else:
                raise ValueError(f"Expected 4D array with 4 channels, got array of ndim={arr.ndim}")
        else:
            raise TypeError(f"Unsupported input type for segmentation: {type(inputs).__name__}")

        # Check for non-finite values in stacked channels
        if not np.isfinite(stacked).all():
            raise ValueError("Prepared multichannel volume contains non-finite values (NaN or Inf).")

        # Save prepared input to data/processed/ if requested (preserving raw data)
        if save_to_processed:
            # We save the stacked volume as NIfTI with channel axis at end for NIfTI convention (H, W, D, 4)
            nifti_ready_volume = np.moveaxis(stacked, 0, -1)
            fname = processed_filename or timestamped_filename("prepared_4channel_input", "nii.gz")
            save_processed_volume(
                volume=nifti_ready_volume,
                affine=ref_affine,
                filename=fname,
            )

        return stacked, ref_affine

    def predict(
        self,
        multichannel_volume: np.ndarray,
    ) -> np.ndarray:
        """Run inference under torch.no_grad() and return a 3D segmentation mask.

        Args:
            multichannel_volume: NumPy array of shape (4, H, W, D).

        Returns:
            np.ndarray: 3D integer array of shape (H, W, D) containing label indices.

        Raises:
            RuntimeError: If model is not loaded and no mock is present.
            ValueError: If volume does not have 4 channels or fails shape checks.
        """
        if multichannel_volume.ndim != 4 or multichannel_volume.shape[0] != 4:
            raise ValueError(
                f"Expected volume shape (4, H, W, D), got {multichannel_volume.shape}."
            )

        # Handle mocked model or callable model
        if self.model is not None and callable(self.model):
            try:
                import torch
                has_torch = True
            except ImportError:
                has_torch = False

            # Normalize intensity per channel (standard BraTS non-zero normalization)
            norm_channels = []
            for c in range(4):
                ch_data = multichannel_volume[c]
                mask = ch_data > 0
                if np.any(mask):
                    mean = ch_data[mask].mean()
                    std = ch_data[mask].std()
                    norm_ch = np.zeros_like(ch_data)
                    norm_ch[mask] = (ch_data[mask] - mean) / (std + 1e-8)
                else:
                    norm_ch = ch_data
                norm_channels.append(norm_ch)

            norm_volume = np.stack(norm_channels, axis=0)

            # Spatial dimensions must be divisible by 16 for SegResNet downsampling/upsampling
            orig_h, orig_w, orig_d = norm_volume.shape[1], norm_volume.shape[2], norm_volume.shape[3]
            mult = 16
            pad_h = (mult - (orig_h % mult)) % mult
            pad_w = (mult - (orig_w % mult)) % mult
            pad_d = (mult - (orig_d % mult)) % mult

            if pad_h > 0 or pad_w > 0 or pad_d > 0:
                padded_volume = np.pad(
                    norm_volume,
                    ((0, 0), (0, pad_h), (0, pad_w), (0, pad_d)),
                    mode="constant",
                    constant_values=0,
                )
            else:
                padded_volume = norm_volume

            if has_torch:
                tensor_in = torch.from_numpy(padded_volume).unsqueeze(0).to(self.device_str)
                with torch.no_grad():
                    out = self.model(tensor_in)
                    if isinstance(out, torch.Tensor):
                        # MONAI BraTS SegResNet outputs 3 channels (TC, WT, ET)
                        if out.ndim == 5 and out.shape[1] == 3:
                            probs = torch.sigmoid(out[0]).cpu().numpy()
                            tc = probs[0] > 0.5
                            wt = probs[1] > 0.5
                            et = probs[2] > 0.5
                            mask_3d = np.zeros(padded_volume.shape[1:], dtype=np.int32)
                            mask_3d[wt] = 2  # Peritumoral Edema
                            mask_3d[tc] = 1  # Necrotic Core
                            mask_3d[et] = 3  # Enhancing Tumor
                        elif out.ndim == 5 and out.shape[1] > 3:
                            pred_tensor = torch.argmax(out, dim=1).squeeze(0)
                            mask_3d = pred_tensor.cpu().numpy().astype(np.int32)
                        elif out.ndim == 5 and out.shape[1] == 1:
                            pred_tensor = out.squeeze(0).squeeze(0)
                            mask_3d = pred_tensor.cpu().numpy().astype(np.int32)
                        elif out.ndim == 4:
                            pred_tensor = out.squeeze(0)
                            mask_3d = pred_tensor.cpu().numpy().astype(np.int32)
                        else:
                            mask_3d = out.cpu().numpy().astype(np.int32)
                    elif isinstance(out, np.ndarray):
                        if out.ndim == 5 and out.shape[1] == 3:
                            # Mock or sigmoid array
                            probs = 1.0 / (1.0 + np.exp(-out[0])) if np.any(out < 0) or np.any(out > 1) else out[0]
                            tc = probs[0] > 0.5
                            wt = probs[1] > 0.5
                            et = probs[2] > 0.5
                            mask_3d = np.zeros(padded_volume.shape[1:], dtype=np.int32)
                            mask_3d[wt] = 2
                            mask_3d[tc] = 1
                            mask_3d[et] = 3
                        elif out.ndim == 5 and out.shape[1] > 1:
                            mask_3d = np.argmax(out, axis=1)[0].astype(np.int32)
                        elif out.ndim == 4:
                            mask_3d = out[0].astype(np.int32)
                        else:
                            mask_3d = out.astype(np.int32)
                    else:
                        raise TypeError(f"Unexpected output type from model: {type(out)}")
            else:
                # NumPy fallback for mocked tests when torch is not yet installed
                batch_in = np.expand_dims(padded_volume, axis=0)
                out = self.model(batch_in)
                if isinstance(out, np.ndarray):
                    if out.ndim == 5 and out.shape[1] > 1:
                        mask_3d = np.argmax(out, axis=1)[0].astype(np.int32)
                    elif out.ndim == 4:
                        mask_3d = out[0].astype(np.int32)
                    else:
                        mask_3d = out.astype(np.int32)
                else:
                    raise TypeError(f"Unexpected output type from model: {type(out)}")

            # Crop back to original dimensions
            mask_3d = mask_3d[:orig_h, :orig_w, :orig_d]

            validate_mri_volume(mask_3d)
            return mask_3d

        # If model checkpoint is absent, inform clearly with download instructions
        raise RuntimeError(
            f"Pretrained MONAI BraTS segmentation bundle is not installed in '{self.model_dir}'.\n"
            f"Please run the download script to fetch the bundle:\n"
            f"    python scripts/download_monai_brats.py"
        )

    def segment_and_save(
        self,
        inputs: Union[Dict[str, Union[str, Path, np.ndarray]], np.ndarray, Path, str],
        output_dir: Optional[Union[str, Path]] = None,
        output_filename: Optional[str] = None,
        affine: Optional[np.ndarray] = None,
        save_processed_input: bool = False,
    ) -> Tuple[Path, Dict[str, Any]]:
        """Execute complete segmentation pipeline and save mask to outputs/masks/.

        Strictly follows data-flow rules:
        - Never writes to or overwrites data/raw/.
        - Saves predicted segmentation masks exclusively in outputs/masks/.
        - Does not write predictions to data/annotations/.

        Args:
            inputs: Multi-channel input (dictionary, 4D array, or file path).
            output_dir: Destination folder. Defaults to 'output_masks' in config.
            output_filename: Optional mask filename. Defaults to timestamped name.
            affine: Spatial affine matrix.
            save_processed_input: If True, saves intermediate preprocessed input in data/processed/.

        Returns:
            Tuple[Path, Dict[str, Any]]: (Saved mask file Path, metadata dictionary).
        """
        cfg = load_config()
        if output_dir is None:
            out_mask_dir = cfg.paths.resolve(self.project_root)["output_masks"]
        else:
            out_mask_dir = Path(output_dir).resolve()

        ensure_directory(out_mask_dir)

        # 1. Prepare multichannel inputs
        stacked, ref_affine = self.prepare_multichannel_input(
            inputs=inputs,
            affine=affine,
            save_to_processed=save_processed_input,
        )

        # 2. Predict 3D segmentation mask
        mask_3d = self.predict(stacked)

        # 3. Format output filename and destination in outputs/masks/
        fname = output_filename or timestamped_filename("brats_seg_mask", "nii.gz")
        if not fname.endswith(".nii") and not fname.endswith(".nii.gz"):
            fname = f"{fname}.nii.gz"

        destination = out_mask_dir / safe_filename(fname)

        # 4. Save mask
        saved_path = save_nifti_volume(mask_3d.astype(np.int32), ref_affine, destination)

        # 5. Extract summary metrics
        unique_labels, counts = np.unique(mask_3d, return_counts=True)
        label_distribution = {
            int(lbl): int(cnt) for lbl, cnt in zip(unique_labels, counts)
        }

        metadata = {
            "mask_path": str(saved_path),
            "mask_shape": mask_3d.shape,
            "unique_labels": [int(lbl) for lbl in unique_labels],
            "voxel_counts": label_distribution,
            "device": self.device_str,
            "channels_used": REQUIRED_CHANNELS,
        }

        return saved_path, metadata
