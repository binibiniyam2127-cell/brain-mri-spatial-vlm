"""Tests for MONAI BraTS 3D segmentation adapter using mocked inference."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

import nibabel as nib
import numpy as np

from src.config import load_config
from src.mri_io import load_nifti_volume
from src.segmentation import (
    BraTSSegmenter,
    MissingChannelError,
    REQUIRED_CHANNELS,
)
from src.utils import get_project_root


class TestBraTSSegmenter(unittest.TestCase):
    """Unit tests for BraTSSegmenter using mocked inference."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        self.shape_3d = (8, 10, 6)
        self.affine = np.eye(4, dtype=np.float64)

        # Create 4 synthetic modality arrays
        self.t1 = np.ones(self.shape_3d, dtype=np.float32) * 10
        self.t1c = np.ones(self.shape_3d, dtype=np.float32) * 20
        self.t2 = np.ones(self.shape_3d, dtype=np.float32) * 30
        self.flair = np.ones(self.shape_3d, dtype=np.float32) * 40

        # Save single modality files
        self.modality_files = {}
        for mod, arr in [("t1", self.t1), ("t1ce", self.t1c), ("t2", self.t2), ("flair", self.flair)]:
            p = self.temp_path / f"synthetic_{mod}.nii.gz"
            img = nib.Nifti1Image(arr, self.affine)
            nib.save(img, str(p))
            self.modality_files[mod] = p

        # 4D combined volume (4, H, W, D)
        self.vol_4d = np.stack([self.t1, self.t1c, self.t2, self.flair], axis=0)

        # Build mock model function that returns a dummy 3D label mask
        def mock_model_func(tensor_in):
            # Input tensor is shape (1, 4, H, W, D)
            # Create dummy classification logits for 4 classes: 0 (background), 1 (NCR), 2 (ED), 3 (ET)
            b, c, h, w, d = tensor_in.shape
            mock_logits = np.zeros((b, 4, h, w, d), dtype=np.float32)
            mock_logits[:, 0, :, :, :] = 1.0  # Background default
            # Add a small mock tumor region
            mock_logits[:, 1, 2:4, 2:4, 2:4] = 5.0
            mock_logits[:, 2, 4:6, 4:6, 2:4] = 5.0
            mock_logits[:, 3, 3:5, 3:5, 2:4] = 6.0
            return mock_logits

        self.mock_model = mock_model_func

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_missing_channels_in_dictionary_raises_error(self) -> None:
        """Verify MissingChannelError when any required modality is missing from dict."""
        segmenter = BraTSSegmenter(device="cpu")
        # Incomplete dictionary missing 'flair' and 't1c'
        incomplete_dict = {
            "t1": self.t1,
            "t2": self.t2,
        }

        with self.assertRaises(MissingChannelError) as ctx:
            segmenter.prepare_multichannel_input(incomplete_dict)

        err_msg = str(ctx.exception)
        self.assertIn("t1c", err_msg)
        self.assertIn("flair", err_msg)
        self.assertIn("MONAI BraTS segmentation requires 4 aligned input channels", err_msg)

    def test_single_channel_3d_file_raises_error(self) -> None:
        """Verify MissingChannelError when passing a single 3D volume."""
        segmenter = BraTSSegmenter(device="cpu")
        single_3d_file = self.modality_files["t1"]

        with self.assertRaises(MissingChannelError) as ctx:
            segmenter.prepare_multichannel_input(single_3d_file)

        err_msg = str(ctx.exception)
        self.assertIn("Missing channels: ['t1c', 't2', 'flair']", err_msg)

    def test_stacking_all_four_modalities_success(self) -> None:
        """Verify prepare_multichannel_input succeeds with 4 complete channels."""
        segmenter = BraTSSegmenter(device="cpu")
        stacked, aff = segmenter.prepare_multichannel_input(self.modality_files)
        self.assertEqual(stacked.shape, (4, *self.shape_3d))
        self.assertEqual(stacked.dtype, np.float32)
        np.testing.assert_allclose(aff, self.affine)

    def test_mocked_prediction(self) -> None:
        """Verify predict produces a 3D mask using mocked inference without model download."""
        segmenter = BraTSSegmenter(device="cpu", model=self.mock_model)
        mask = segmenter.predict(self.vol_4d)

        self.assertEqual(mask.shape, self.shape_3d)
        self.assertTrue(np.issubdtype(mask.dtype, np.integer))
        unique_labels = set(np.unique(mask))
        self.assertTrue(unique_labels.issubset({0, 1, 2, 3}))

    def test_segment_and_save_dataflow_rules(self) -> None:
        """Verify segment_and_save writes strictly to outputs/masks/ and not annotations/raw."""
        out_masks_dir = self.temp_path / "outputs" / "masks"
        segmenter = BraTSSegmenter(device="cpu", model=self.mock_model)

        saved_mask, metadata = segmenter.segment_and_save(
            inputs=self.vol_4d,
            output_dir=out_masks_dir,
            output_filename="test_tumor_mask.nii.gz",
            affine=self.affine,
        )

        # Check saved location
        self.assertTrue(saved_mask.exists())
        self.assertEqual(saved_mask.parent, out_masks_dir)
        self.assertNotIn("data/annotations", str(saved_mask))
        self.assertNotIn("data/raw", str(saved_mask))

        # Check reload
        reloaded_mask, _, _ = load_nifti_volume(saved_mask)
        self.assertEqual(reloaded_mask.shape, self.shape_3d)

        # Check metadata
        self.assertEqual(metadata["mask_shape"], self.shape_3d)
        self.assertIn("voxel_counts", metadata)
        self.assertEqual(metadata["channels_used"], REQUIRED_CHANNELS)

    def test_uninitialized_model_raises_runtime_error(self) -> None:
        """Verify RuntimeError is raised when no model checkpoint is loaded, directing to download script."""
        empty_dir = self.temp_path / "empty_models"
        empty_dir.mkdir()
        segmenter = BraTSSegmenter(model_dir=empty_dir, device="cpu", model=None)

        with self.assertRaises(RuntimeError) as ctx:
            segmenter.predict(self.vol_4d)
        self.assertIn("python scripts/download_monai_brats.py", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

