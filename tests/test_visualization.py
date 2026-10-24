"""Unit tests for MRI and segmentation visualization engine.

Uses synthetic 3D MRI and mask arrays to verify slice selection, orthogonal
tri-planar rendering, multi-slice montages, occupancy plots, and non-diagnostic
watermark enforcement without external dependencies or real patient data.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest

import nibabel as nib
import numpy as np
from PIL import Image

from src.visualization import (
    VISUALIZATION_WATERMARK,
    burn_pillow_watermark,
    find_informative_slices,
    generate_all_visualizations,
    load_mask_for_visualization,
    load_volume_for_visualization,
    plot_mask_occupancy,
    plot_orthogonal_slices,
    plot_slice_montage,
)


class TestVisualizationEngine(unittest.TestCase):
    """Test suite for research visualization modules."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.output_dir = self.temp_path / "outputs" / "visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create synthetic 3D MRI volume (30, 30, 20)
        self.shape_3d = (30, 30, 20)
        self.synth_mri = np.ones(self.shape_3d, dtype=np.float32) * 50.0
        # Add gradient to simulate anatomical contrast
        for z in range(self.shape_3d[2]):
            self.synth_mri[:, :, z] += z * 5.0

        self.synth_mri_path = self.temp_path / "synth_mri.nii.gz"
        nib.save(nib.Nifti1Image(self.synth_mri, np.eye(4)), str(self.synth_mri_path))

        # Create synthetic 3D mask with known peak at z=12
        self.synth_mask = np.zeros(self.shape_3d, dtype=np.int32)
        # Small area at z=10
        self.synth_mask[12:16, 12:16, 10] = 1  # 4x4 = 16 voxels
        # Peak area at z=12
        self.synth_mask[10:20, 10:20, 12] = 2  # 10x10 = 100 voxels (peak)
        # Small area at z=14
        self.synth_mask[13:17, 13:17, 14] = 3  # 4x4 = 16 voxels

        self.synth_mask_path = self.temp_path / "synth_mask.nii.gz"
        nib.save(nib.Nifti1Image(self.synth_mask, np.eye(4)), str(self.synth_mask_path))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_find_informative_slices_locates_peak_and_center(self) -> None:
        """Verify automatic selection of slice with largest mask area and center."""
        info = find_informative_slices(self.synth_mri, self.synth_mask)

        # Axial Z-axis checks
        axial = info["axial"]
        self.assertEqual(axial["center_slice"], 10)  # 20 // 2
        self.assertEqual(axial["max_mask_slice"], 12)  # Peak slice is Z=12
        self.assertEqual(axial["active_range"], (10, 14))

        # Coronal & Sagittal checks
        self.assertIn("coronal", info)
        self.assertIn("sagittal", info)
        self.assertGreater(info["coronal"]["occupancy"].max(), 0)
        self.assertGreater(info["sagittal"]["occupancy"].max(), 0)

    def test_plot_orthogonal_slices_generates_valid_image(self) -> None:
        """Verify generation of tri-planar orthogonal view with mask overlay."""
        out_png = self.output_dir / "ortho_test.png"
        fig, path = plot_orthogonal_slices(
            mri_volume=self.synth_mri,
            mask=self.synth_mask,
            output_path=out_png,
        )

        self.assertTrue(path.is_file())
        self.assertGreater(path.stat().st_size, 1000)

        # Verify readable with Pillow
        with Image.open(path) as img:
            self.assertEqual(img.format, "PNG")
            self.assertGreater(img.width, 500)
            self.assertGreater(img.height, 200)

    def test_plot_slice_montage_generates_valid_image(self) -> None:
        """Verify generation of multi-slice montage grid."""
        out_png = self.output_dir / "montage_test.png"
        fig, path = plot_slice_montage(
            mri_volume=self.synth_mri,
            mask=self.synth_mask,
            plane="axial",
            num_slices=8,
            grid_cols=4,
            output_path=out_png,
        )

        self.assertTrue(path.is_file())
        self.assertGreater(path.stat().st_size, 1000)

        with Image.open(path) as img:
            self.assertEqual(img.format, "PNG")
            self.assertGreater(img.width, 400)
            self.assertGreater(img.height, 200)

    def test_plot_mask_occupancy_generates_valid_image(self) -> None:
        """Verify generation of slice-wise mask occupancy profile."""
        out_png = self.output_dir / "occupancy_test.png"
        fig, path = plot_mask_occupancy(
            mask=self.synth_mask,
            output_path=out_png,
        )

        self.assertTrue(path.is_file())
        self.assertGreater(path.stat().st_size, 1000)

        with Image.open(path) as img:
            self.assertEqual(img.format, "PNG")
            self.assertGreater(img.width, 500)

    def test_empty_mask_rendering_graceful(self) -> None:
        """Verify rendering functions operate gracefully without errors when mask is empty or None."""
        empty_mask = np.zeros(self.shape_3d, dtype=np.int32)
        out_ortho = self.output_dir / "empty_ortho.png"
        _, path1 = plot_orthogonal_slices(self.synth_mri, mask=empty_mask, output_path=out_ortho)
        self.assertTrue(path1.is_file())

        out_montage = self.output_dir / "empty_montage.png"
        _, path2 = plot_slice_montage(self.synth_mri, mask=None, output_path=out_montage)
        self.assertTrue(path2.is_file())

    def test_burn_pillow_watermark_applies_banner(self) -> None:
        """Verify burn_pillow_watermark embeds banner onto image."""
        test_img_path = self.output_dir / "test_banner.png"
        base = Image.new("RGB", (300, 200), (128, 128, 128))
        base.save(test_img_path)

        res_path = burn_pillow_watermark(test_img_path, VISUALIZATION_WATERMARK)
        self.assertEqual(res_path, test_img_path)

        with Image.open(res_path) as modified:
            self.assertEqual(modified.size, (300, 200))
            # Verify bottom banner region has altered pixels
            bottom_pixel = modified.getpixel((150, 195))
            self.assertNotEqual(bottom_pixel, (128, 128, 128))

    def test_multichannel_4d_volume_channel_extraction(self) -> None:
        """Verify that 4D volumes have 3D channel extracted for visualization."""
        vol_4d = np.ones((16, 16, 10, 4), dtype=np.float32)
        vol_4d[:, :, :, 3] = 42.0  # FLAIR channel
        vol_4d_path = self.temp_path / "synth_4d.nii.gz"
        nib.save(nib.Nifti1Image(vol_4d, np.eye(4)), str(vol_4d_path))

        vol_3d, meta = load_volume_for_visualization(vol_4d_path, channel_index=3)
        self.assertEqual(vol_3d.ndim, 3)
        self.assertEqual(vol_3d.shape, (16, 16, 10))
        self.assertAlmostEqual(float(np.mean(vol_3d)), 42.0)

    def test_generate_all_visualizations_suite(self) -> None:
        """Verify generate_all_visualizations creates complete PNG suite."""
        results = generate_all_visualizations(
            mri_volume=self.synth_mri_path,
            mask=self.synth_mask_path,
            output_dir=self.output_dir,
            prefix="suite_test",
        )

        self.assertIn("orthogonal", results)
        self.assertIn("montage", results)
        self.assertIn("occupancy", results)

        for key, p in results.items():
            self.assertTrue(p.is_file(), f"Output file missing for {key}: {p}")

    def test_run_visualization_cli_script(self) -> None:
        """Verify scripts/run_visualization.py CLI execution."""
        cmd = [
            "python",
            "scripts/run_visualization.py",
            "--input",
            str(self.synth_mri_path),
            "--mask",
            str(self.synth_mask_path),
            "--output-dir",
            str(self.output_dir),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"run_visualization.py failed: {res.stderr}")
        self.assertIn("Visualizations Generated Successfully", res.stdout)
        self.assertIn(VISUALIZATION_WATERMARK, res.stdout)

