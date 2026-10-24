"""Tests for src/mri_io.py module using synthetic 3D volumes."""

from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np

from src.config import load_config
from src.mri_io import (
    get_volume_metadata,
    list_raw_nifti_files,
    load_nifti_volume,
    save_nifti_volume,
    save_processed_volume,
    validate_mri_volume,
)
from src.utils import get_project_root


class TestMriIO(unittest.TestCase):
    """Test suite for NIfTI MRI I/O, validation, and metadata extraction."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Generate a small synthetic 3D volume (e.g. 12 x 14 x 10)
        self.shape = (12, 14, 10)
        self.synthetic_volume = np.linspace(0.0, 100.0, num=12 * 14 * 10, dtype=np.float32).reshape(self.shape)
        self.affine = np.array(
            [
                [1.5, 0.0, 0.0, -10.0],
                [0.0, 1.5, 0.0, -20.0],
                [0.0, 0.0, 2.0, -30.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        # Save synthetic NIfTI volume
        self.synthetic_nii_path = self.temp_path / "synthetic_brain.nii.gz"
        img = nib.Nifti1Image(self.synthetic_volume, self.affine)
        nib.save(img, str(self.synthetic_nii_path))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_validate_mri_volume_success(self) -> None:
        """Verify that a valid 3D array passes validation."""
        self.assertTrue(validate_mri_volume(self.synthetic_volume))

    def test_validate_mri_volume_invalid_type(self) -> None:
        """Verify TypeError on non-numpy input."""
        with self.assertRaises(TypeError):
            validate_mri_volume([[1, 2], [3, 4]])  # type: ignore

    def test_validate_mri_volume_invalid_dimensions(self) -> None:
        """Verify ValueError on 2D or 4D arrays."""
        two_d = np.ones((10, 10), dtype=np.float32)
        with self.assertRaises(ValueError) as ctx:
            validate_mri_volume(two_d)
        self.assertIn("must be 3-dimensional", str(ctx.exception))

        four_d = np.ones((10, 10, 10, 2), dtype=np.float32)
        with self.assertRaises(ValueError) as ctx:
            validate_mri_volume(four_d)
        self.assertIn("must be 3-dimensional", str(ctx.exception))

    def test_validate_mri_volume_non_finite_values(self) -> None:
        """Verify ValueError when volume contains NaN or Inf."""
        nan_vol = self.synthetic_volume.copy()
        nan_vol[2, 2, 2] = np.nan
        with self.assertRaises(ValueError) as ctx:
            validate_mri_volume(nan_vol)
        self.assertIn("non-finite", str(ctx.exception))

        inf_vol = self.synthetic_volume.copy()
        inf_vol[3, 3, 3] = np.inf
        with self.assertRaises(ValueError) as ctx:
            validate_mri_volume(inf_vol)
        self.assertIn("non-finite", str(ctx.exception))

    def test_load_nifti_volume(self) -> None:
        """Verify loading an existing NIfTI volume returns expected array and affine."""
        vol, aff, meta = load_nifti_volume(self.synthetic_nii_path)
        self.assertEqual(vol.shape, self.shape)
        self.assertEqual(vol.dtype, np.float32)
        np.testing.assert_allclose(vol, self.synthetic_volume, rtol=1e-5)
        np.testing.assert_allclose(aff, self.affine, rtol=1e-5)
        self.assertIn("spacing_mm", meta)
        self.assertAlmostEqual(meta["spacing_mm"][0], 1.5, places=2)

    def test_get_volume_metadata(self) -> None:
        """Verify metadata fields extracted from NIfTI file."""
        meta = get_volume_metadata(self.synthetic_nii_path)
        self.assertEqual(meta["filename"], "synthetic_brain.nii.gz")
        self.assertEqual(meta["shape"], self.shape)
        self.assertEqual(meta["ndim"], 3)
        self.assertTrue(meta["is_3d"])
        self.assertEqual(len(meta["spacing_mm"]), 3)
        self.assertIsNotNone(meta["orientation"])

    def test_save_and_reload_nifti_volume(self) -> None:
        """Verify saving a 3D volume as .nii and reloading it."""
        out_path = self.temp_path / "saved_brain.nii"
        saved = save_nifti_volume(self.synthetic_volume, self.affine, out_path)
        self.assertTrue(saved.exists())

        loaded_vol, loaded_aff, _ = load_nifti_volume(saved)
        self.assertEqual(loaded_vol.shape, self.shape)
        np.testing.assert_allclose(loaded_vol, self.synthetic_volume, rtol=1e-5)
        np.testing.assert_allclose(loaded_aff, self.affine, rtol=1e-5)

    def test_list_raw_nifti_files(self) -> None:
        """Verify listing raw files filters non-NIfTI files correctly."""
        raw_mock = self.temp_path / "mock_raw"
        raw_mock.mkdir()
        (raw_mock / "patient1.nii").touch()
        (raw_mock / "patient2.nii.gz").touch()
        (raw_mock / "notes.txt").touch()
        (raw_mock / "patient3.dcm").touch()

        listed = list_raw_nifti_files(raw_mock)
        listed_names = [p.name for p in listed]
        self.assertEqual(listed_names, ["patient1.nii", "patient2.nii.gz"])

    def test_save_processed_volume_protects_raw(self) -> None:
        """Verify save_processed_volume refuses to write into data/raw."""
        cfg = load_config()
        raw_dir = cfg.paths.resolve(get_project_root())["data_raw"]
        forbidden_target = raw_dir / "should_fail.nii.gz"

        with self.assertRaises(PermissionError):
            save_processed_volume(self.synthetic_volume, self.affine, output_path=forbidden_target)

    def test_save_processed_volume_default_processed_dir(self) -> None:
        """Verify save_processed_volume writes into data/processed/."""
        saved_file = save_processed_volume(
            self.synthetic_volume,
            self.affine,
            filename="unit_test_processed.nii.gz",
        )
        try:
            self.assertTrue(saved_file.exists())
            self.assertIn("data", str(saved_file))
            self.assertIn("processed", str(saved_file))
            reloaded, _, _ = load_nifti_volume(saved_file)
            self.assertEqual(reloaded.shape, self.shape)
        finally:
            if saved_file.exists():
                saved_file.unlink()


if __name__ == "__main__":
    unittest.main()

