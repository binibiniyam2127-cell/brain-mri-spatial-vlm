"""Unit tests for Streamlit app helper functions.

Verifies upload file persistence, spatial features table formatting,
and 2D slice image rendering without requiring Streamlit runtime or browser interaction.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from src.app_helpers import (
    build_3d_volumetric_figure,
    format_spatial_features_table,
    render_slice_rgba,
    rows_to_markdown_table,
    save_uploaded_nifti,
)


class TestAppHelpers(unittest.TestCase):
    """Test suite for Streamlit demo helper functions."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Synthetic 3D volume (20, 20, 10)
        self.shape_3d = (20, 20, 10)
        self.synth_mri = np.ones(self.shape_3d, dtype=np.float32) * 50.0
        self.synth_mask = np.zeros(self.shape_3d, dtype=np.int32)
        self.synth_mask[8:12, 8:12, 4:7] = 2  # Edema region

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_save_uploaded_nifti_success(self) -> None:
        """Verify saving uploaded NIfTI bytes to destination directory."""
        content = b"SIMULATED_NIFTI_BYTES_12345"
        dest_dir = self.temp_path / "raw"
        saved = save_uploaded_nifti(
            uploaded_bytes=content,
            original_filename="sample_scan.nii.gz",
            target_dir=dest_dir,
        )

        self.assertTrue(saved.is_file())
        self.assertEqual(saved.read_bytes(), content)
        self.assertTrue(str(saved).endswith(".nii.gz"))

    def test_save_uploaded_nifti_invalid_extension_raises_error(self) -> None:
        """Verify non-NIfTI file extensions are rejected."""
        with self.assertRaises(ValueError):
            save_uploaded_nifti(
                uploaded_bytes=b"dummy",
                original_filename="scan.dcm",
                target_dir=self.temp_path,
            )

    def test_save_uploaded_nifti_empty_raises_error(self) -> None:
        """Verify empty files are rejected."""
        with self.assertRaises(ValueError):
            save_uploaded_nifti(
                uploaded_bytes=b"",
                original_filename="scan.nii",
                target_dir=self.temp_path,
            )

    def test_format_spatial_features_table(self) -> None:
        """Verify transformation of spatial features dictionary into flat table rows."""
        sample_features = {
            "total_mask_voxels": 450,
            "mask_volume_mm3": 1250.5,
            "volume_fraction_percent": 0.082,
            "bounding_box": {"min_x": 10, "max_x": 30, "min_y": 15, "max_y": 35, "min_z": 5, "max_z": 15},
            "extent": {
                "extent_x_voxels": 20, "extent_y_voxels": 20, "extent_z_voxels": 10,
                "extent_x_mm": 20.0, "extent_y_mm": 20.0, "extent_z_mm": 10.0,
            },
            "centroid_voxel": (20.0, 25.0, 10.0),
            "centroid_world_mm": (20.0, 25.0, 10.0),
            "connected_components": {"count": 1},
        }

        table = format_spatial_features_table(sample_features)
        self.assertIsInstance(table, list)
        self.assertGreaterEqual(len(table), 5)

        measurements = [row["Measurement"] for row in table]
        self.assertIn("Total Lesion Mask Voxels", measurements)
        self.assertIn("Estimated Physical Volume", measurements)
        self.assertIn("3D Bounding Box", measurements)
        self.assertIn("Connected Components", measurements)

    def test_rows_to_markdown_table(self) -> None:
        """Verify conversion of row dictionaries into a valid Markdown table string."""
        rows = [
            {"Col1": "Val1", "Col2": "Val2"},
            {"Col1": "Val3", "Col2": "Val4"},
        ]
        md = rows_to_markdown_table(rows)
        self.assertIn("| Col1 | Col2 |", md)
        self.assertIn("| :--- | :--- |", md)
        self.assertIn("| Val1 | Val2 |", md)
        self.assertIn("| Val3 | Val4 |", md)

        # Empty rows handling
        self.assertEqual(rows_to_markdown_table([]), "")


    def test_render_slice_rgba_creates_valid_image(self) -> None:
        """Verify 2D slice rendering with mask overlay returns valid PIL Image."""
        # Axial plane
        img_axial = render_slice_rgba(
            mri_volume=self.synth_mri,
            mask=self.synth_mask,
            plane="axial",
            slice_idx=5,
            show_mask=True,
        )
        self.assertIsInstance(img_axial, Image.Image)
        # In axial orientation, 2D slice is rot90(mri[:, :, 5]) -> shape (20, 20)
        self.assertEqual(img_axial.size, (20, 20))

        # Coronal plane
        img_coronal = render_slice_rgba(
            mri_volume=self.synth_mri,
            mask=self.synth_mask,
            plane="coronal",
            slice_idx=10,
            show_mask=True,
        )
        self.assertIsInstance(img_coronal, Image.Image)

        # Sagittal plane
        img_sagittal = render_slice_rgba(
            mri_volume=self.synth_mri,
            mask=self.synth_mask,
            plane="sagittal",
            slice_idx=10,
            show_mask=False,
        )
        self.assertIsInstance(img_sagittal, Image.Image)

    def test_build_3d_volumetric_figure(self) -> None:
        """Verify 3D WebGL mesh figure generation returns valid Plotly Figure."""
        # Create a sphere-like structure in synth_mri and mask
        mri_3d = np.zeros((30, 30, 30), dtype=np.float32)
        mri_3d[5:25, 5:25, 5:25] = 100.0
        mask_3d = np.zeros((30, 30, 30), dtype=np.int32)
        mask_3d[10:15, 10:15, 10:15] = 1  # Necrotic
        mask_3d[15:20, 15:20, 15:20] = 2  # Edema

        fig = build_3d_volumetric_figure(
            mri_volume=mri_3d,
            mask=mask_3d,
            show_brain_envelope=True,
            show_necrotic=True,
            show_edema=True,
            show_enhancing=False,
            step_size=1,
        )

        import plotly.graph_objects as go
        self.assertIsInstance(fig, go.Figure)
        self.assertGreaterEqual(len(fig.data), 1)
        names = [trace.name for trace in fig.data]
        self.assertTrue(any("3D Bounding Box" in n for n in names))
        self.assertTrue(any("3D Centroid" in n for n in names))

    def test_render_slice_rgba_with_bbox(self) -> None:
        """Verify 2D slice rendering with bounding box rectangle."""
        img = render_slice_rgba(
            mri_volume=self.synth_mri,
            mask=self.synth_mask,
            plane="axial",
            slice_idx=5,
            show_mask=True,
            show_bbox=True,
        )
        self.assertIsInstance(img, Image.Image)


