"""Integration tests for the end-to-end multimodal inference pipeline.

Uses mocked MONAI segmentation and mocked BrainGemma3D to verify pipeline
orchestration, independent stage fault tolerance, data flow, artifact creation,
and non-diagnostic disclaimer persistence completely offline.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock

import nibabel as nib
import numpy as np

from src.pipeline import RESEARCH_DISCLAIMER, PipelineResult, run_full_pipeline
from src.reporting import DraftReport, MODEL_UNAVAILABLE_MESSAGE


class TestFullPipeline(unittest.TestCase):
    """Integration test suite for run_full_pipeline."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.output_dir = self.temp_path / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create synthetic 4D multimodal MRI volume (24, 24, 16, 4)
        self.shape_4d = (24, 24, 16, 4)
        self.synth_mri = np.ones(self.shape_4d, dtype=np.float32) * 50.0
        # Add values across channels
        for c in range(4):
            self.synth_mri[:, :, :, c] += c * 15.0

        self.synth_mri_path = self.temp_path / "BRATS_TEST_001.nii.gz"
        nib.save(nib.Nifti1Image(self.synth_mri, np.eye(4)), str(self.synth_mri_path))

        # Expected synthetic mask (24, 24, 16)
        self.shape_3d = (24, 24, 16)
        self.synth_mask = np.zeros(self.shape_3d, dtype=np.int32)
        self.synth_mask[8:14, 8:14, 6:10] = 1  # Necrotic
        self.synth_mask[6:16, 6:16, 5:11] = 2  # Edema
        self.synth_mask[9:13, 9:13, 7:9] = 3   # Enhancing

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_full_pipeline_success_with_mocks(self) -> None:
        """Verify end-to-end pipeline execution with mocked segmenter and reporter."""
        # Mock Segmenter
        mock_segmenter = MagicMock()
        mock_mask_path = self.output_dir / "masks" / "BRATS_TEST_001_mock_mask.nii.gz"
        mock_mask_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(self.synth_mask, np.eye(4)), str(mock_mask_path))
        mock_segmenter.segment_and_save.return_value = (mock_mask_path, {"mock": True})

        # Mock Reporter
        mock_reporter = MagicMock()
        mock_draft = DraftReport(
            content=f"# Report\n{RESEARCH_DISCLAIMER}\nMocked findings.",
            findings="Mocked volumetric findings.",
            impression=f"Mocked research impression. {RESEARCH_DISCLAIMER}",
            disclaimer=RESEARCH_DISCLAIMER,
            is_model_available=True,
            status="completed",
        )
        report_json = self.output_dir / "reports" / "mock_report.json"
        report_md = self.output_dir / "reports" / "mock_report.md"
        report_json.parent.mkdir(parents=True, exist_ok=True)
        report_json.write_text(json.dumps(mock_draft.to_dict()), encoding="utf-8")
        report_md.write_text(mock_draft.to_markdown(), encoding="utf-8")
        mock_reporter.generate_and_save.return_value = (mock_draft, report_json, report_md)

        # Run pipeline
        result = run_full_pipeline(
            input_nifti_path=self.synth_mri_path,
            segmenter=mock_segmenter,
            reporter=mock_reporter,
            output_dir=self.output_dir,
        )

        self.assertIsInstance(result, PipelineResult)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.disclaimer, RESEARCH_DISCLAIMER)

        # Verify all 5 required output paths exist
        self.assertIsNotNone(result.mask_path)
        self.assertTrue(result.mask_path.is_file())

        self.assertIsNotNone(result.spatial_features_path)
        self.assertTrue(result.spatial_features_path.is_file())

        self.assertIsNotNone(result.draft_report_path)
        self.assertTrue(result.draft_report_path.is_file())

        self.assertGreaterEqual(len(result.visualization_paths), 2)
        for v_name, v_path in result.visualization_paths.items():
            self.assertTrue(v_path.is_file(), f"Visualization {v_name} missing: {v_path}")

        self.assertIsNotNone(result.combined_json_path)
        self.assertTrue(result.combined_json_path.is_file())

        # Verify subscripting interface
        paths_dict = result.paths
        self.assertIn("mask", paths_dict)
        self.assertIn("spatial_features", paths_dict)
        self.assertIn("draft_report", paths_dict)
        self.assertIn("visualizations", paths_dict)
        self.assertIn("combined_json", paths_dict)

        # Inspect combined JSON content
        combined_data = json.loads(result.combined_json_path.read_text(encoding="utf-8"))
        self.assertEqual(combined_data["disclaimer"], RESEARCH_DISCLAIMER)
        self.assertEqual(combined_data["overall_status"], "completed")
        self.assertIn("spatial_features", combined_data)
        self.assertIn("paths", combined_data)

    def test_pipeline_when_segmentation_fails_gracefully(self) -> None:
        """Verify pipeline handles segmentation failure without crashing and reports error."""
        failing_segmenter = MagicMock()
        failing_segmenter.segment_and_save.side_effect = RuntimeError("MONAI model bundle failed to load")

        mock_reporter = MagicMock()
        mock_draft = DraftReport(
            content=f"# Report\n{RESEARCH_DISCLAIMER}",
            findings="No mask available.",
            disclaimer=RESEARCH_DISCLAIMER,
            is_model_available=True,
        )
        report_json = self.output_dir / "reports" / "mock_report2.json"
        report_md = self.output_dir / "reports" / "mock_report2.md"
        report_json.parent.mkdir(parents=True, exist_ok=True)
        report_json.write_text(json.dumps(mock_draft.to_dict()), encoding="utf-8")
        report_md.write_text(mock_draft.to_markdown(), encoding="utf-8")
        mock_reporter.generate_and_save.return_value = (mock_draft, report_json, report_md)

        result = run_full_pipeline(
            input_nifti_path=self.synth_mri_path,
            segmenter=failing_segmenter,
            reporter=mock_reporter,
            output_dir=self.output_dir,
        )

        # Never fabricate missing mask
        self.assertIsNone(result.mask_path)
        self.assertEqual(result.stage_statuses.get("segmentation"), "failed")
        self.assertIn("segmentation", result.errors)
        self.assertIn("MONAI model bundle failed", result.errors["segmentation"])

        # Visualizations must still succeed for the base volume
        self.assertGreaterEqual(len(result.visualization_paths), 2)
        self.assertIsNotNone(result.combined_json_path)
        self.assertTrue(result.combined_json_path.is_file())

    def test_pipeline_when_braingemma3d_unavailable(self) -> None:
        """Verify that when BrainGemma3D weights are absent, other stages still succeed."""
        # Mock Segmenter
        mock_segmenter = MagicMock()
        mock_mask_path = self.output_dir / "masks" / "BRATS_TEST_001_mock_mask.nii.gz"
        mock_mask_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(self.synth_mask, np.eye(4)), str(mock_mask_path))
        mock_segmenter.segment_and_save.return_value = (mock_mask_path, {"mock": True})

        # Do NOT provide reporter mock -> uses real BrainGemma3DReporter
        result = run_full_pipeline(
            input_nifti_path=self.synth_mri_path,
            segmenter=mock_segmenter,
            reporter=None,  # Real reporter will report unavailable
            output_dir=self.output_dir,
        )

        # Segmentation, spatial features, and visualizations are all saved
        self.assertIsNotNone(result.mask_path)
        self.assertTrue(result.mask_path.is_file())
        self.assertIsNotNone(result.spatial_features_path)
        self.assertTrue(result.spatial_features_path.is_file())
        self.assertGreaterEqual(len(result.visualization_paths), 2)

        # Draft report reflects unavailable status without fabricating
        self.assertIsNotNone(result.draft_report_path)
        report_data = json.loads(result.draft_report_path.read_text(encoding="utf-8"))
        self.assertEqual(report_data.get("findings"), MODEL_UNAVAILABLE_MESSAGE)
        self.assertFalse(report_data.get("is_model_available"))

    def test_run_pipeline_cli_script(self) -> None:
        """Verify scripts/run_pipeline.py CLI runs and exits cleanly."""
        cmd = [
            "python",
            "scripts/run_pipeline.py",
            "--input",
            str(self.synth_mri_path),
            "--output-dir",
            str(self.output_dir),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"run_pipeline.py failed: {res.stderr}")
        self.assertIn("Pipeline Execution Finished", res.stdout)
        self.assertIn(RESEARCH_DISCLAIMER, res.stdout)

