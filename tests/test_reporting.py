"""Mocked unit tests for BrainGemma3D report generation adapter.

Ensures report generation logic, component validation, spatial context integration,
and disclaimer enforcement function completely offline without model downloads.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

from src.reporting import (
    MODEL_UNAVAILABLE_MESSAGE,
    RESEARCH_DISCLAIMER,
    BrainGemma3DReporter,
    DraftReport,
)


class TestBrainGemma3DReporting(unittest.TestCase):
    """Test suite for BrainGemma3DReporter and DraftReport."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Temporary paths
        self.mock_repo = self.temp_path / "braingemma3d_repo"
        self.mock_siglip = self.temp_path / "siglip"
        self.mock_medgemma = self.temp_path / "medgemma"
        self.mock_ckpt = self.temp_path / "checkpoints" / "phase2b"
        self.output_reports = self.temp_path / "outputs" / "reports"

        # Synthetic 3D MRI volume
        self.mri_shape = (32, 32, 16)
        self.synth_mri = np.ones(self.mri_shape, dtype=np.float32) * 100.0
        self.synth_mri_path = self.temp_path / "synth_mri.nii.gz"
        img = nib.Nifti1Image(self.synth_mri, np.eye(4))
        nib.save(img, str(self.synth_mri_path))

        # Synthetic 3D Mask
        self.synth_mask = np.zeros(self.mri_shape, dtype=np.uint8)
        self.synth_mask[10:15, 10:15, 5:10] = 1  # 5x5x5 = 125 voxels
        self.synth_mask_path = self.temp_path / "synth_mask.nii.gz"
        mask_img = nib.Nifti1Image(self.synth_mask, np.eye(4))
        nib.save(mask_img, str(self.synth_mask_path))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_missing_components_returns_exact_unavailable_message(self) -> None:
        """When local model directories are absent, returns exact unavailable message."""
        reporter = BrainGemma3DReporter(
            repo_dir=self.mock_repo,
            siglip_dir=self.mock_siglip,
            medgemma_dir=self.mock_medgemma,
            checkpoint_dir=self.mock_ckpt,
            require_gpu=False,
        )

        is_avail, missing = reporter.check_components_availability()
        self.assertFalse(is_avail)
        self.assertGreaterEqual(len(missing), 4)

        report = reporter.generate_report(self.synth_mri_path)

        # Must strictly match the required message
        self.assertEqual(str(report), MODEL_UNAVAILABLE_MESSAGE)
        self.assertEqual(report, MODEL_UNAVAILABLE_MESSAGE)
        self.assertEqual(report.findings, MODEL_UNAVAILABLE_MESSAGE)
        self.assertFalse(report.is_model_available)
        self.assertEqual(report.disclaimer, RESEARCH_DISCLAIMER)
        self.assertGreaterEqual(len(report.missing_components), 4)

    def test_gpu_resource_flagged_when_unavailable(self) -> None:
        """Verify missing GPU resource is detected when require_gpu is True."""
        reporter = BrainGemma3DReporter(
            repo_dir=self.mock_repo,
            siglip_dir=self.mock_siglip,
            medgemma_dir=self.mock_medgemma,
            checkpoint_dir=self.mock_ckpt,
            require_gpu=True,
        )

        with patch("torch.cuda.is_available", return_value=False):
            is_avail, missing = reporter.check_components_availability()
            self.assertFalse(is_avail)
            gpu_missing = any("GPU resource" in item for item in missing)
            self.assertTrue(gpu_missing, "Expected GPU resource to be reported as missing.")

    def test_mocked_generator_returns_completed_report_with_disclaimer(self) -> None:
        """Verify report generation with a mocked generator offline."""
        def custom_mock_gen(mri_vol, mask_vol, spatial_ctx):
            findings = f"Volumetric evaluation: mask voxels={spatial_ctx.get('total_mask_voxels', 0)}."
            impression = "Draft research evaluation. No clinical pathology assessed."
            return DraftReport(
                content=f"# Draft Report\n{findings}\n{impression}",
                findings=findings,
                impression=impression,
                disclaimer=RESEARCH_DISCLAIMER,
                is_model_available=True,
                spatial_context=spatial_ctx,
                status="completed",
            )

        reporter = BrainGemma3DReporter(
            repo_dir=self.mock_repo,
            siglip_dir=self.mock_siglip,
            medgemma_dir=self.mock_medgemma,
            checkpoint_dir=self.mock_ckpt,
            mock_generator=custom_mock_gen,
        )

        report = reporter.generate_report(
            mri_volume=self.synth_mri_path,
            mask=self.synth_mask_path,
        )

        self.assertTrue(report.is_model_available)
        self.assertEqual(report.status, "completed")
        self.assertEqual(report.disclaimer, RESEARCH_DISCLAIMER)
        self.assertIn(RESEARCH_DISCLAIMER, report.to_markdown())
        self.assertEqual(report.spatial_context.get("total_mask_voxels"), 125)

    def test_save_report_creates_json_and_markdown_with_disclaimer(self) -> None:
        """Verify that save_report writes valid JSON and Markdown files with mandatory disclaimer."""
        reporter = BrainGemma3DReporter(
            repo_dir=self.mock_repo,
            siglip_dir=self.mock_siglip,
            medgemma_dir=self.mock_medgemma,
            checkpoint_dir=self.mock_ckpt,
            require_gpu=False,
        )

        report = reporter.generate_report(self.synth_mri_path)
        json_path, md_path = reporter.save_report(report, output_dir=self.output_reports)

        self.assertTrue(json_path.exists())
        self.assertTrue(md_path.exists())

        # Inspect JSON content
        json_data = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(json_data["disclaimer"], RESEARCH_DISCLAIMER)
        self.assertEqual(json_data["findings"], MODEL_UNAVAILABLE_MESSAGE)
        self.assertFalse(json_data["is_model_available"])

        # Inspect Markdown content
        md_text = md_path.read_text(encoding="utf-8")
        self.assertIn(RESEARCH_DISCLAIMER, md_text)
        self.assertIn(MODEL_UNAVAILABLE_MESSAGE, md_text)

    def test_multichannel_4d_volume_handling(self) -> None:
        """Verify that 4D multimodal MRI volumes (e.g. MSD BraTS) have 3D channel extracted."""
        vol_4d = np.ones((16, 16, 8, 4), dtype=np.float32)
        nii_4d_path = self.temp_path / "synth_4d.nii.gz"
        nib.save(nib.Nifti1Image(vol_4d, np.eye(4)), str(nii_4d_path))

        reporter = BrainGemma3DReporter(
            repo_dir=self.mock_repo,
            siglip_dir=self.mock_siglip,
            medgemma_dir=self.mock_medgemma,
            checkpoint_dir=self.mock_ckpt,
            require_gpu=False,
        )

        vol_3d, meta = reporter._prepare_mri_volume(nii_4d_path)
        self.assertEqual(vol_3d.ndim, 3)
        self.assertEqual(vol_3d.shape, (16, 16, 8))
        self.assertIn("FLAIR", meta.get("extracted_sequence", ""))

    def test_run_report_script_cli(self) -> None:
        """Verify scripts/run_report.py execution using CLI arguments."""
        import subprocess

        cmd = [
            "python",
            "scripts/run_report.py",
            "--input",
            str(self.synth_mri_path),
            "--mask",
            str(self.synth_mask_path),
            "--output-dir",
            str(self.output_reports),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"run_report.py failed with: {res.stderr}")
        self.assertIn("Report Processing Complete", res.stdout)
        self.assertIn(RESEARCH_DISCLAIMER, res.stdout)

