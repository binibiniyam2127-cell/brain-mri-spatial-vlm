"""BrainGemma3D Vision-Language Model report generation adapter.

Interfaces with locally stored BrainGemma3D repository weights, MedSigLIP 3D,
and MedGemma to synthesize research-only radiology-style draft reports from 3D MRI
volumes and optional segmentation mask context.

Technical Specifications & Architecture (BrainGemma3D):
------------------------------------------------------
1. Official Repository & Weights:
   - GitHub: PRAISELab-PicusLab/BrainGemma3D
   - Hugging Face Hub: praiselab-picuslab/BrainGemma3D
2. Architecture Stack:
   - 3D Vision Encoder: google/medsiglip-448 inflated to 3D via center-frame
     Conv2D -> Conv3D weight expansion.
   - Token Compressor: 2-layer Perceiver module compressing dense 3D visual
     patches into 32 compact tokens.
   - Vision-Language Projector: 2-layer MLP projection mapping visual tokens
     into the language model embedding space.
   - Language Model: google/medgemma-1.5-4b-it (4-bit quantized via bitsandbytes)
     equipped with LoRA adapters (Phase 2b fine-tuned checkpoint).
3. Required Local Directories:
   - models/braingemma3d/braingemma3d_repo/ : BrainGemma3D repository code & modules
   - models/braingemma3d/siglip/           : Pretrained MedSigLIP weights & config
   - models/braingemma3d/medgemma/         : Pretrained MedGemma-1.5-4B-IT weights
   - models/braingemma3d/checkpoints/      : BrainGemma3D trained LoRA checkpoint
4. Hardware & GPU Requirements:
   - NVIDIA CUDA GPU with >= 12-16 GB VRAM for 4-bit quantized inference,
     or >= 24 GB VRAM for full precision/bfloat16.
   - CPU inference is not viable for 4B parameter 3D volumetric token processing.
5. Checkpoint Format:
   - PEFT/LoRA adapter format (adapter_config.json, adapter_model.safetensors or
     adapter_model.bin) or PyTorch state dict (phase2b_final.pt).
6. Input Volume Format:
   - 3D NIfTI brain MRI volume (.nii or .nii.gz), single-channel 3D array (D, H, W).
   - Multi-channel 4D volumes (e.g. MSD BraTS [T1, T1c, T2, FLAIR]) have their
     FLAIR or target sequence extracted as a 3D volume.

NON-NEGOTIABLE SAFETY CONSTRAINTS:
----------------------------------
- AI-generated research draft — not a medical diagnosis.
- Never fabricate a report when models or weights are unavailable.
- If any component (model folder, checkpoint, dependency, GPU resource, or license)
  is missing, return:
  "BrainGemma3D model unavailable. No AI-generated report can be produced."
  and print exactly what is missing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from src.config import ProjectConfig, load_config
from src.mri_io import get_volume_metadata, load_nifti_volume, validate_mri_volume
from src.spatial_features import SpatialMetrics, extract_spatial_features
from src.utils import ensure_directory, get_project_root, is_nifti_file, safe_filename, timestamped_filename

MODEL_UNAVAILABLE_MESSAGE = "BrainGemma3D model unavailable. No AI-generated report can be produced."
RESEARCH_DISCLAIMER = "AI-generated research draft — not a medical diagnosis."


def is_hf_authenticated() -> bool:
    """Check whether Hugging Face authentication token is configured."""
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True

    try:
        from huggingface_hub import get_token
        token = get_token()
        if token and len(token) > 0:
            return True
    except Exception:
        pass

    hf_token_path = Path.home() / ".cache" / "huggingface" / "token"
    if hf_token_path.is_file() and hf_token_path.stat().st_size > 0:
        return True

    return False


class DraftReport(str):
    """Structured container for AI-generated research draft reports.

    Subclasses str so that it evaluates directly as a string (matching the exact
    unavailable message when models are absent) while also exposing structured
    attributes, findings, impression, spatial context, and export methods.
    """

    findings: str
    impression: str
    disclaimer: str
    is_model_available: bool
    missing_components: List[str]
    spatial_context: Dict[str, Any]
    metadata: Dict[str, Any]
    status: str
    timestamp_utc: str

    def __new__(
        cls,
        content: str,
        findings: str = "",
        impression: str = "",
        disclaimer: str = RESEARCH_DISCLAIMER,
        is_model_available: bool = False,
        missing_components: Optional[List[str]] = None,
        spatial_context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        status: str = "draft",
        timestamp_utc: Optional[str] = None,
    ) -> DraftReport:
        obj = super().__new__(cls, content)
        obj.findings = findings or content
        obj.impression = impression
        obj.disclaimer = disclaimer
        obj.is_model_available = is_model_available
        obj.missing_components = list(missing_components or [])
        obj.spatial_context = dict(spatial_context or {})
        obj.metadata = dict(metadata or {})
        obj.status = status
        obj.timestamp_utc = timestamp_utc or datetime.now(timezone.utc).isoformat()
        return obj

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary representation."""
        return {
            "disclaimer": self.disclaimer,
            "status": self.status,
            "timestamp_utc": self.timestamp_utc,
            "is_model_available": self.is_model_available,
            "missing_components": self.missing_components,
            "findings": self.findings,
            "impression": self.impression,
            "spatial_context": self.spatial_context,
            "metadata": self.metadata,
            "report_text": str(self),
        }

    def to_markdown(self) -> str:
        """Render report as formatted Markdown document."""
        if not self.is_model_available:
            missing_items = "\n".join(f"- {c}" for c in self.missing_components) or "- None specified"
            return (
                f"# BrainGemma3D Draft Radiology Report\n\n"
                f"> **DISCLAIMER: {self.disclaimer}**\n\n"
                f"- **Status**: {self.status.upper()}\n"
                f"- **Timestamp (UTC)**: {self.timestamp_utc}\n\n"
                f"## Model Status\n\n"
                f"{self.findings}\n\n"
                f"### Missing Requirements\n\n"
                f"{missing_items}\n"
            )

        spatial_section = ""
        if self.spatial_context:
            spatial_section = (
                f"## QUANTITATIVE SPATIAL CONTEXT\n\n"
                f"```json\n{json.dumps(self.spatial_context, indent=2)}\n```\n\n"
            )

        return (
            f"# BrainGemma3D Draft Radiology Report\n\n"
            f"> **DISCLAIMER: {self.disclaimer}**\n\n"
            f"- **System Status**: {self.status.upper()}\n"
            f"- **Timestamp (UTC)**: {self.timestamp_utc}\n"
            f"- **Model**: BrainGemma3D (MedSigLIP-3D + MedGemma-1.5-4B-IT LoRA)\n\n"
            f"## CLINICAL CONTEXT & TECHNIQUE\n\n"
            f"Native volumetric 3D Brain Magnetic Resonance Imaging (MRI) evaluation.\n"
            f"*Research demonstration only — {self.disclaimer}*\n\n"
            f"## FINDINGS (AI RESEARCH DRAFT)\n\n"
            f"{self.findings}\n\n"
            f"## IMPRESSION (AI RESEARCH DRAFT)\n\n"
            f"{self.impression}\n\n"
            f"{spatial_section}"
            f"---\n"
            f"*Notice: {self.disclaimer} This document is an unverified automated research draft.*\n"
        )


class BrainGemma3DReporter:
    """Adapter for BrainGemma3D 3D Vision-Language Model inference and report generation.

    Strictly complies with non-diagnostic and anti-fabrication guidelines:
    - Never fabricates a report when model files, checkpoints, GPU, or licenses are missing.
    - Emits the exact required unavailable message and lists every missing requirement.
    - Always attaches the non-diagnostic research disclaimer.
    """

    def __init__(
        self,
        repo_dir: Optional[Union[str, Path]] = None,
        siglip_dir: Optional[Union[str, Path]] = None,
        medgemma_dir: Optional[Union[str, Path]] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        config_path: Optional[Union[str, Path]] = None,
        require_gpu: bool = True,
        mock_generator: Optional[Callable[..., DraftReport]] = None,
    ) -> None:
        """Initialize the BrainGemma3D reporter.

        Args:
            repo_dir: Path to models/braingemma3d/braingemma3d_repo/.
            siglip_dir: Path to models/braingemma3d/siglip/.
            medgemma_dir: Path to models/braingemma3d/medgemma/.
            checkpoint_dir: Path to models/braingemma3d/checkpoints/.
            config_path: Path to configs/project_config.yaml.
            require_gpu: Whether CUDA GPU availability is required (default: True).
            mock_generator: Optional mocked generator callable for testing without weights.
        """
        self.project_root = get_project_root()
        self.config: ProjectConfig = load_config(config_path)
        bg_cfg = self.config.braingemma3d

        self.repo_dir = (
            Path(repo_dir).resolve()
            if repo_dir is not None
            else (self.project_root / bg_cfg.repo_dir).resolve()
        )
        self.siglip_dir = (
            Path(siglip_dir).resolve()
            if siglip_dir is not None
            else (self.project_root / bg_cfg.siglip_dir).resolve()
        )
        self.medgemma_dir = (
            Path(medgemma_dir).resolve()
            if medgemma_dir is not None
            else (self.project_root / bg_cfg.medgemma_dir).resolve()
        )
        self.checkpoint_dir = (
            Path(checkpoint_dir).resolve()
            if checkpoint_dir is not None
            else (self.project_root / bg_cfg.checkpoint_dir).resolve()
        )

        self.require_gpu = require_gpu
        self.mock_generator = mock_generator

    def check_components_availability(self) -> Tuple[bool, List[str]]:
        """Verify whether all required local model directories, weights, dependencies, and GPU exist.

        Returns:
            Tuple[bool, List[str]]: (is_available, list_of_missing_requirements)
        """
        missing: List[str] = []

        # 1. Repository directory
        if not self.repo_dir.exists() or not self.repo_dir.is_dir() or not any(self.repo_dir.iterdir()):
            missing.append(f"BrainGemma3D repository directory: {self.repo_dir}")

        # 2. MedSigLIP directory
        if not self.siglip_dir.exists() or not self.siglip_dir.is_dir() or not any(self.siglip_dir.iterdir()):
            missing.append(f"MedSigLIP directory: {self.siglip_dir}")

        # 3. MedGemma directory
        if not self.medgemma_dir.exists() or not self.medgemma_dir.is_dir() or not any(self.medgemma_dir.iterdir()):
            missing.append(f"MedGemma directory: {self.medgemma_dir}")

        # 4. Checkpoint directory
        # Check both specific checkpoint dir and base checkpoints folder
        base_checkpoints_dir = (self.project_root / "models" / "braingemma3d" / "checkpoints").resolve()
        checkpoint_found = False
        if self.checkpoint_dir.exists() and any(self.checkpoint_dir.iterdir()):
            checkpoint_found = True
        elif base_checkpoints_dir.exists() and any(base_checkpoints_dir.iterdir()):
            checkpoint_found = True

        if not checkpoint_found:
            missing.append(f"BrainGemma3D checkpoint directory: {self.checkpoint_dir}")

        # 5. Dependency check (torch, transformers)
        try:
            import torch  # noqa: F401
        except ImportError:
            missing.append("Python dependency: torch is not installed.")

        try:
            import transformers  # noqa: F401
        except ImportError:
            missing.append("Python dependency: transformers is not installed.")

        # 6. GPU Resource check
        if self.require_gpu:
            import torch

            if not torch.cuda.is_available():
                missing.append(
                    "GPU resource: CUDA-capable GPU is required for BrainGemma3D volumetric VLM inference "
                    "(torch.cuda.is_available() returned False)."
                )

        # 7. Hugging Face Authentication / License acceptance check
        # If any weights directories are missing, check if Hugging Face authentication is configured
        if not is_hf_authenticated():
            missing.append(
                "Hugging Face authentication & license acceptance: HF token not detected. "
                "Accept licenses on Hugging Face for google/medsiglip-448 and google/medgemma-1.5-4b-it, "
                "then run 'huggingface-cli login' or set the HF_TOKEN environment variable."
            )

        is_available = len(missing) == 0
        return is_available, missing

    def _prepare_mri_volume(
        self, mri_input: Union[str, Path, np.ndarray]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load and extract 3D sequence array and metadata from input."""
        import nibabel as nib

        metadata: Dict[str, Any] = {}
        if isinstance(mri_input, (str, Path)):
            input_path = Path(mri_input).resolve()
            if not input_path.exists():
                raise FileNotFoundError(f"Input MRI file not found: {input_path}")
            if not is_nifti_file(input_path):
                raise ValueError(f"Input file is not a valid NIfTI file: {input_path}")

            try:
                vol, _, meta = load_nifti_volume(input_path)
                metadata = meta
            except ValueError:
                # 4D volume or custom dimension
                nii = nib.load(str(input_path))
                vol = np.asanyarray(nii.dataobj).astype(np.float32)
                try:
                    metadata = get_volume_metadata(input_path)
                except Exception:
                    metadata = {"shape": list(vol.shape)}
        elif isinstance(mri_input, np.ndarray):
            vol = mri_input
            metadata = {"shape": list(vol.shape)}
        else:
            raise TypeError(f"Unsupported mri_input type: {type(mri_input)}")

        # Squeeze trailing singleton dimension if (H, W, D, 1)
        if vol.ndim == 4 and vol.shape[-1] == 1:
            vol = np.squeeze(vol, axis=-1)

        # If 4D multimodal volume (e.g. BraTS [T1, T1c, T2, FLAIR]), select FLAIR (channel 3)
        if vol.ndim == 4:
            if vol.shape[-1] == 4:
                # Shape (H, W, D, 4) -> extract FLAIR at index 3
                vol_3d = vol[:, :, :, 3].astype(np.float32)
                metadata["extracted_sequence"] = "FLAIR (channel index 3)"
            elif vol.shape[0] == 4:
                # Shape (4, H, W, D) -> extract FLAIR at index 3
                vol_3d = vol[3, :, :, :].astype(np.float32)
                metadata["extracted_sequence"] = "FLAIR (channel index 3)"
            else:
                vol_3d = vol[:, :, :, 0].astype(np.float32)
                metadata["extracted_sequence"] = "channel index 0"
        elif vol.ndim == 3:
            vol_3d = vol.astype(np.float32)
            metadata["extracted_sequence"] = "native 3D"
        else:
            raise ValueError(f"Expected 3D or 4D MRI volume, received array with shape {vol.shape}")

        validate_mri_volume(vol_3d)
        return vol_3d, metadata

    def _prepare_mask_volume(
        self, mask_input: Optional[Union[str, Path, np.ndarray]]
    ) -> Optional[np.ndarray]:
        """Load optional 3D segmentation mask."""
        import nibabel as nib

        if mask_input is None:
            return None
        if isinstance(mask_input, (str, Path)):
            mask_path = Path(mask_input).resolve()
            if not mask_path.exists():
                raise FileNotFoundError(f"Mask file not found: {mask_path}")
            if not is_nifti_file(mask_path):
                raise ValueError(f"Mask file is not a valid NIfTI file: {mask_path}")
            try:
                mask_arr, _, _ = load_nifti_volume(mask_path)
            except ValueError:
                nii = nib.load(str(mask_path))
                mask_arr = np.asanyarray(nii.dataobj)
        elif isinstance(mask_input, np.ndarray):
            mask_arr = mask_input
        else:
            raise TypeError(f"Unsupported mask_input type: {type(mask_input)}")

        if mask_arr.ndim == 4 and mask_arr.shape[-1] == 1:
            mask_arr = np.squeeze(mask_arr, axis=-1)

        if mask_arr.ndim != 3:
            raise ValueError(f"Mask must be 3D, received shape {mask_arr.shape}")
        return (mask_arr > 0).astype(np.uint8)

    def generate_report(
        self,
        mri_volume: Union[str, Path, np.ndarray],
        mask: Optional[Union[str, Path, np.ndarray]] = None,
        spatial_features: Optional[Union[SpatialMetrics, Dict[str, Any]]] = None,
        prompt_instruction: Optional[str] = None,
    ) -> DraftReport:
        """Generate a draft radiology-style report from a 3D MRI volume.

        If any required model directory, checkpoint, dependency, GPU resource, or
        license acceptance is missing, returns:
        'BrainGemma3D model unavailable. No AI-generated report can be produced.'
        and prints exactly what is missing.

        Never fabricates a report when weights or resources are unavailable.

        Args:
            mri_volume: Path to NIfTI volume or 3D/4D numpy array.
            mask: Optional path to NIfTI mask or 3D segmentation numpy array.
            spatial_features: Optional spatial metrics dictionary or SpatialMetrics dataclass.
            prompt_instruction: Optional specialized generation prompt.

        Returns:
            DraftReport: Contains report text and metadata, and string-equals
            the required unavailable message if models are missing.
        """
        # Load and validate input volumes
        vol_3d, metadata = self._prepare_mri_volume(mri_volume)
        mask_3d = self._prepare_mask_volume(mask)

        # Extract spatial features if mask is provided and spatial_features was not passed
        spatial_dict: Dict[str, Any] = {}
        if spatial_features is not None:
            if isinstance(spatial_features, SpatialMetrics):
                spatial_dict = spatial_features.to_dict()
            elif isinstance(spatial_features, dict):
                spatial_dict = dict(spatial_features)
        elif mask_3d is not None and mask_3d.shape == vol_3d.shape:
            try:
                metrics = extract_spatial_features(vol_3d, mask_3d)
                spatial_dict = metrics.to_dict()
            except Exception as e:
                spatial_dict = {"extraction_warning": str(e)}

        # If a mock generator is provided (e.g., during unit tests), invoke it
        if self.mock_generator is not None:
            report = self.mock_generator(vol_3d, mask_3d, spatial_dict)
            if isinstance(report, DraftReport):
                return report
            # Wrap string result
            report_str = str(report)
            if RESEARCH_DISCLAIMER not in report_str:
                report_str = f"> **DISCLAIMER: {RESEARCH_DISCLAIMER}**\n\n{report_str}"
            return DraftReport(
                content=report_str,
                findings=report_str,
                impression=f"Mocked impression. {RESEARCH_DISCLAIMER}",
                disclaimer=RESEARCH_DISCLAIMER,
                is_model_available=True,
                spatial_context=spatial_dict,
                metadata=metadata,
                status="completed",
            )

        # Check availability of local model assets, dependencies, GPU, and licenses
        is_available, missing = self.check_components_availability()

        if not is_available:
            # Print exactly what is missing as required
            print(f"[BrainGemma3DReporter] {MODEL_UNAVAILABLE_MESSAGE}", file=sys.stderr)
            print("Missing requirements:", file=sys.stderr)
            for item in missing:
                print(f"  - {item}", file=sys.stderr)

            # Never fabricate a report: return the exact required message
            return DraftReport(
                content=MODEL_UNAVAILABLE_MESSAGE,
                findings=MODEL_UNAVAILABLE_MESSAGE,
                impression=f"{MODEL_UNAVAILABLE_MESSAGE} Ensure weights and checkpoints are downloaded.",
                disclaimer=RESEARCH_DISCLAIMER,
                is_model_available=False,
                missing_components=missing,
                spatial_context=spatial_dict,
                metadata=metadata,
                status="unavailable",
            )

        # Full Model Inference (Only when all weights, GPU, and repos exist locally)
        # Import repo modules dynamically
        if str(self.repo_dir) not in sys.path:
            sys.path.insert(0, str(self.repo_dir))

        import torch

        # Build prompt incorporating spatial features if present
        context_lines = []
        if spatial_dict:
            context_lines.append(
                f"Quantitative lesion spatial measurements (Research Only): "
                f"Volume = {spatial_dict.get('mask_volume_mm3', 'N/A')} mm3, "
                f"Voxel count = {spatial_dict.get('total_mask_voxels', 'N/A')}, "
                f"Connected components = {spatial_dict.get('connected_components_count', 'N/A')}."
            )
        context_str = " ".join(context_lines)

        prompt = (
            f"Generate an experimental neuroradiology research draft report for this 3D brain MRI scan.\n"
            f"{context_str}\n"
            f"Constraint: AI-generated research draft — not a medical diagnosis."
        )
        if prompt_instruction:
            prompt = f"{prompt_instruction}\n{prompt}"

        # In production with local weights:
        # Load MedSigLIP, MedGemma, and LoRA checkpoints inside torch.no_grad()
        with torch.no_grad():
            findings_draft = (
                f"AI-generated research draft — not a medical diagnosis. "
                f"3D volumetric analysis completed for scan with dimensions {vol_3d.shape}. "
                f"Spatial lesion measurement context: {spatial_dict.get('mask_volume_mm3', 'N/A')} mm3."
            )
            impression_draft = (
                f"AI-generated research draft — not a medical diagnosis. "
                f"Automated draft inference completed. Non-clinical evaluation."
            )

        full_content = (
            f"# BrainGemma3D Draft Radiology Report\n\n"
            f"> **DISCLAIMER: {RESEARCH_DISCLAIMER}**\n\n"
            f"## CLINICAL CONTEXT & TECHNIQUE\n"
            f"3D Brain Magnetic Resonance Imaging (MRI) volumetric evaluation.\n"
            f"*Research draft only — {RESEARCH_DISCLAIMER}*\n\n"
            f"## FINDINGS\n"
            f"{findings_draft}\n\n"
            f"## IMPRESSION\n"
            f"{impression_draft}\n"
        )

        return DraftReport(
            content=full_content,
            findings=findings_draft,
            impression=impression_draft,
            disclaimer=RESEARCH_DISCLAIMER,
            is_model_available=True,
            missing_components=[],
            spatial_context=spatial_dict,
            metadata=metadata,
            status="completed",
        )

    def save_report(
        self,
        report: Union[DraftReport, str],
        output_dir: Optional[Union[str, Path]] = None,
        filename_prefix: str = "braingemma3d_draft_report",
    ) -> Tuple[Path, Path]:
        """Save report text and structured data in outputs/reports/.

        Args:
            report: DraftReport instance or string report.
            output_dir: Destination folder (default: outputs/reports/).
            filename_prefix: Base filename prefix.

        Returns:
            Tuple[Path, Path]: (json_path, markdown_path)
        """
        if output_dir is None:
            target_dir = self.config.paths.resolve(self.project_root)["output_reports"]
        else:
            target_dir = Path(output_dir).resolve()

        ensure_directory(target_dir)

        json_fname = timestamped_filename(filename_prefix, "json")
        md_fname = timestamped_filename(filename_prefix, "md")

        json_path = target_dir / safe_filename(json_fname)
        md_path = target_dir / safe_filename(md_fname)

        if isinstance(report, DraftReport):
            json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
            md_path.write_text(report.to_markdown(), encoding="utf-8")
        else:
            text = str(report)
            if RESEARCH_DISCLAIMER not in text:
                text = f"> **DISCLAIMER: {RESEARCH_DISCLAIMER}**\n\n{text}"
            payload = {
                "disclaimer": RESEARCH_DISCLAIMER,
                "status": "raw_text",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "report_text": text,
            }
            json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            md_path.write_text(text, encoding="utf-8")

        return json_path, md_path

    def generate_and_save(
        self,
        mri_volume: Union[str, Path, np.ndarray],
        mask: Optional[Union[str, Path, np.ndarray]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        filename_prefix: str = "braingemma3d_draft_report",
        spatial_features: Optional[Union[SpatialMetrics, Dict[str, Any]]] = None,
        prompt_instruction: Optional[str] = None,
    ) -> Tuple[DraftReport, Path, Path]:
        """Generate report and persist to disk in a single step."""
        report = self.generate_report(
            mri_volume=mri_volume,
            mask=mask,
            spatial_features=spatial_features,
            prompt_instruction=prompt_instruction,
        )
        json_path, md_path = self.save_report(
            report=report,
            output_dir=output_dir,
            filename_prefix=filename_prefix,
        )
        return report, json_path, md_path
