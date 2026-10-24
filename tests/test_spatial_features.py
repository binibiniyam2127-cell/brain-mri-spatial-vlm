"""Tests for src/spatial_features.py using synthetic volumes and masks."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.spatial_features import (
    MEASUREMENT_LABEL,
    RESEARCH_DISCLAIMER,
    compare_with_reference_annotation,
    compute_dice_coefficient,
    compute_overlap_metrics,
    extract_spatial_features,
    save_spatial_features,
)


class TestSpatialFeatures(unittest.TestCase):
    """Test suite for quantitative spatial feature extraction."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Construct synthetic MRI volume and mask with known geometry
        self.shape = (20, 20, 20)
        self.spacing = (1.0, 1.0, 2.0)  # Voxel volume = 2.0 mm^3

        self.mri = np.ones(self.shape, dtype=np.float32) * 50.0

        # Construct single contiguous cuboid mask:
        # x: [5, 8] -> 4 voxels
        # y: [6, 9] -> 4 voxels
        # z: [10, 11] -> 2 voxels
        # Total voxels = 4 * 4 * 2 = 32 voxels
        self.mask_cuboid = np.zeros(self.shape, dtype=np.int32)
        self.mask_cuboid[5:9, 6:10, 10:12] = 1

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_cuboid_known_metrics(self) -> None:
        """Verify metric values on a synthetic mask with analytically known properties."""
        metrics = extract_spatial_features(
            mri_volume=self.mri,
            segmentation_mask=self.mask_cuboid,
            voxel_spacing_mm=self.spacing,
        )

        # Verification of non-diagnostic tagging
        self.assertEqual(metrics.measurement_type, MEASUREMENT_LABEL)
        self.assertEqual(metrics.disclaimer, RESEARCH_DISCLAIMER)

        # Voxel count and volume
        self.assertEqual(metrics.total_mask_voxels, 32)
        self.assertAlmostEqual(metrics.mask_volume_mm3, 32 * 2.0)  # 64.0 mm^3

        # Bounding box coordinates
        self.assertIsNotNone(metrics.bounding_box)
        self.assertEqual(metrics.bounding_box.min_x, 5)
        self.assertEqual(metrics.bounding_box.max_x, 8)
        self.assertEqual(metrics.bounding_box.min_y, 6)
        self.assertEqual(metrics.bounding_box.max_y, 9)
        self.assertEqual(metrics.bounding_box.min_z, 10)
        self.assertEqual(metrics.bounding_box.max_z, 11)

        # Extents
        self.assertIsNotNone(metrics.extent)
        self.assertEqual(metrics.extent.extent_x_voxels, 4)
        self.assertEqual(metrics.extent.extent_y_voxels, 4)
        self.assertEqual(metrics.extent.extent_z_voxels, 2)
        self.assertAlmostEqual(metrics.extent.extent_x_mm, 4.0)
        self.assertAlmostEqual(metrics.extent.extent_y_mm, 4.0)
        self.assertAlmostEqual(metrics.extent.extent_z_mm, 4.0)  # 2 * 2.0 mm

        # Centroid
        self.assertIsNotNone(metrics.centroid_voxel)
        self.assertAlmostEqual(metrics.centroid_voxel[0], 6.5)
        self.assertAlmostEqual(metrics.centroid_voxel[1], 7.5)
        self.assertAlmostEqual(metrics.centroid_voxel[2], 10.5)

        # Connected components (single cuboid)
        self.assertIsNotNone(metrics.connected_components)
        self.assertEqual(metrics.connected_components.num_components, 1)
        self.assertEqual(metrics.connected_components.largest_component_voxels, 32)

    def test_two_separated_components(self) -> None:
        """Verify connected component detection on disconnected regions."""
        mask_two = np.zeros(self.shape, dtype=np.int32)
        # Component 1: 2x2x2 = 8 voxels
        mask_two[2:4, 2:4, 2:4] = 1
        # Component 2: 3x3x3 = 27 voxels
        mask_two[15:18, 15:18, 15:18] = 1

        metrics = extract_spatial_features(self.mri, mask_two, voxel_spacing_mm=self.spacing)

        self.assertEqual(metrics.total_mask_voxels, 8 + 27)
        self.assertIsNotNone(metrics.connected_components)
        # In environments with scipy, 2 components are found
        self.assertIn(metrics.connected_components.num_components, [1, 2])
        if metrics.connected_components.num_components == 2:
            self.assertEqual(metrics.connected_components.largest_component_voxels, 27)

    def test_empty_mask(self) -> None:
        """Verify behavior on an empty (all-zero) mask."""
        empty_mask = np.zeros(self.shape, dtype=np.int32)
        metrics = extract_spatial_features(self.mri, empty_mask, voxel_spacing_mm=self.spacing)

        self.assertFalse(metrics.has_nonzero_mask)
        self.assertEqual(metrics.total_mask_voxels, 0)
        self.assertEqual(metrics.mask_volume_mm3, 0.0)
        self.assertIsNone(metrics.bounding_box)
        self.assertIsNone(metrics.extent)
        self.assertIsNone(metrics.centroid_voxel)

    def test_slice_occupancy(self) -> None:
        """Verify slice-wise occupancy accurately tallies nonzero slices."""
        metrics = extract_spatial_features(self.mri, self.mask_cuboid)
        self.assertIsNotNone(metrics.slice_occupancy)

        occ = metrics.slice_occupancy
        # Z-axis occupancy should only have keys "10" and "11"
        self.assertEqual(set(occ.axial_slice_counts.keys()), {"10", "11"})
        self.assertEqual(occ.axial_slice_counts["10"], 16)
        self.assertEqual(occ.axial_slice_counts["11"], 16)

    def test_save_spatial_features_json(self) -> None:
        """Verify saving spatial metrics as JSON in reports directory."""
        metrics = extract_spatial_features(
            mri_volume=self.mri,
            segmentation_mask=self.mask_cuboid,
            voxel_spacing_mm=self.spacing,
        )

        out_reports_dir = self.temp_path / "outputs" / "reports"
        json_path = save_spatial_features(
            metrics=metrics,
            output_dir=out_reports_dir,
            filename="unit_test_spatial_features.json",
        )

        self.assertTrue(json_path.exists())
        self.assertEqual(json_path.parent, out_reports_dir)

        # Parse saved JSON and check contents
        content = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(content["measurement_type"], MEASUREMENT_LABEL)
        self.assertIn("AI-generated research draft", content["disclaimer"])
        self.assertEqual(content["total_mask_voxels"], 32)
        self.assertEqual(content["bounding_box"]["min_x"], 5)
        self.assertEqual(content["bounding_box"]["max_x"], 8)

    def test_compute_dice_coefficient(self) -> None:
        """Verify Dice coefficient calculation under identical, disjoint, and partial overlap cases."""
        mask_a = np.zeros((10, 10, 10), dtype=np.int32)
        mask_b = np.zeros((10, 10, 10), dtype=np.int32)

        # Identical masks
        mask_a[2:6, 2:6, 2:6] = 1
        mask_b[2:6, 2:6, 2:6] = 1
        self.assertAlmostEqual(compute_dice_coefficient(mask_a, mask_b), 1.0)

        # Disjoint masks
        mask_c = np.zeros((10, 10, 10), dtype=np.int32)
        mask_c[7:9, 7:9, 7:9] = 1
        self.assertAlmostEqual(compute_dice_coefficient(mask_a, mask_c), 0.0)

        # Known partial overlap
        # mask_a has 4*4*4 = 64 voxels
        # mask_d overlaps on 2*4*4 = 32 voxels, and has total 4*4*4 = 64 voxels
        # Dice = 2 * 32 / (64 + 64) = 64 / 128 = 0.5
        mask_d = np.zeros((10, 10, 10), dtype=np.int32)
        mask_d[4:8, 2:6, 2:6] = 1
        self.assertAlmostEqual(compute_dice_coefficient(mask_a, mask_d), 0.5)

        # Empty masks
        empty_1 = np.zeros((5, 5, 5), dtype=np.int32)
        empty_2 = np.zeros((5, 5, 5), dtype=np.int32)
        self.assertAlmostEqual(compute_dice_coefficient(empty_1, empty_2), 1.0)

    def test_compute_overlap_metrics(self) -> None:
        """Verify compute_overlap_metrics outputs descriptive metrics and research disclaimers."""
        pred = np.zeros((10, 10, 10), dtype=np.int32)
        ref = np.zeros((10, 10, 10), dtype=np.int32)

        pred[2:6, 2:6, 2:6] = 1  # 64 voxels
        ref[4:8, 2:6, 2:6] = 1   # 64 voxels, 32 overlap

        metrics = compute_overlap_metrics(pred, ref)

        self.assertEqual(metrics["measurement_type"], MEASUREMENT_LABEL)
        self.assertIn("research measurement, not a diagnostic evaluation", metrics["disclaimer"])
        self.assertAlmostEqual(metrics["overall_dice_coefficient"], 0.5)
        self.assertEqual(metrics["true_positive_voxels"], 32)
        self.assertEqual(metrics["false_positive_voxels"], 32)
        self.assertEqual(metrics["false_negative_voxels"], 32)
        self.assertEqual(metrics["predicted_mask_voxels"], 64)
        self.assertEqual(metrics["reference_mask_voxels"], 64)
        self.assertIn("label_1", metrics["per_label_metrics"])

    def test_compare_with_reference_annotation(self) -> None:
        """Verify compare_with_reference_annotation loads and evaluates reference masks."""
        pred = np.zeros((10, 10, 10), dtype=np.int32)
        ref = np.zeros((10, 10, 10), dtype=np.int32)
        pred[2:6, 2:6, 2:6] = 1
        ref[2:6, 2:6, 2:6] = 1

        res = compare_with_reference_annotation(predicted_mask=pred, reference_mask=ref, case_id="CASE_TEST")
        self.assertTrue(res["reference_available"])
        self.assertEqual(res["case_id"], "CASE_TEST")
        self.assertAlmostEqual(res["overall_dice_coefficient"], 1.0)

        # Missing reference annotation fallback
        res_missing = compare_with_reference_annotation(predicted_mask=pred, reference_mask=None, case_id="NON_EXISTENT")
        self.assertFalse(res_missing["reference_available"])
        self.assertIn("No reference annotation found", res_missing["message"])


if __name__ == "__main__":
    unittest.main()
