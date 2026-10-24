"""Tests for BrainGemma3D report generation adapter."""

import json
from pathlib import Path
import tempfile
import unittest

from src.report_generator import (
    MODEL_UNAVAILABLE_MESSAGE,
    RESEARCH_DISCLAIMER,
    BrainGemma3DAdapter,
    DraftReport,
)


class TestReportGenerator(unittest.TestCase):
    """Test suite for BrainGemma3D report generation adapter."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create mock directories
        self.mock_repo = self.temp_path / "braingemma3d_repo"
        self.mock_siglip = self.temp_path / "siglip"
        self.mock_medgemma = self.temp_path / "medgemma"
        self.mock_ckpt = self.temp_path / "checkpoints" / "phase2b"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_missing_directories_returns_unavailable_message(self) -> None:
        """Verify that missing local model files returns the required unavailable string."""
        adapter = BrainGemma3DAdapter(
            repo_dir=self.mock_repo,
            siglip_dir=self.mock_siglip,
            medgemma_dir=self.mock_medgemma,
            checkpoint_dir=self.mock_ckpt,
        )

        is_avail, missing = adapter.check_components_availability()
        self.assertFalse(is_avail)
        self.assertEqual(len(missing), 4)

        report = adapter.generate_draft_report(spatial_metrics={"total_mask_voxels": 150})
        self.assertEqual(report.findings_draft, MODEL_UNAVAILABLE_MESSAGE)
        self.assertIn(MODEL_UNAVAILABLE_MESSAGE, report.impression_draft)
        self.assertFalse(report.is_model_available)
        self.assertEqual(report.disclaimer, RESEARCH_DISCLAIMER)

    def test_mock_generation_offline(self) -> None:
        """Verify draft report formatting using a mocked generator without downloading weights."""
        def mock_gen(metrics):
            return DraftReport(
                disclaimer=RESEARCH_DISCLAIMER,
                status="completed",
                is_model_available=True,
                spatial_summary=metrics or {},
                findings_draft="Mocked research findings based on spatial metrics.",
                impression_draft="Mocked research impression. Non-diagnostic.",
            )

        adapter = BrainGemma3DAdapter(
            repo_dir=self.mock_repo,
            siglip_dir=self.mock_siglip,
            medgemma_dir=self.mock_medgemma,
            checkpoint_dir=self.mock_ckpt,
            mock_generator=mock_gen,
        )

        report = adapter.generate_draft_report(spatial_metrics={"mask_volume_mm3": 450.0})
        self.assertTrue(report.is_model_available)
        self.assertIn("Mocked research findings", report.findings_draft)
        self.assertEqual(report.disclaimer, RESEARCH_DISCLAIMER)

    def test_save_report_json_and_markdown(self) -> None:
        """Verify saving draft report to JSON and Markdown in destination directory."""
        adapter = BrainGemma3DAdapter(
            repo_dir=self.mock_repo,
            siglip_dir=self.mock_siglip,
            medgemma_dir=self.mock_medgemma,
            checkpoint_dir=self.mock_ckpt,
        )

        report = adapter.generate_draft_report(spatial_metrics={"total_mask_voxels": 42})
        out_reports = self.temp_path / "outputs" / "reports"

        json_path, md_path = adapter.save_report(report, output_dir=out_reports)
        self.assertTrue(json_path.exists())
        self.assertTrue(md_path.exists())

        # Inspect JSON content
        data = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(data["findings_draft"], MODEL_UNAVAILABLE_MESSAGE)
        self.assertIn("AI-generated research draft", data["disclaimer"])

        # Inspect Markdown content
        md_text = md_path.read_text(encoding="utf-8")
        self.assertIn("AI-generated research draft — not a medical diagnosis.", md_text)
        self.assertIn(MODEL_UNAVAILABLE_MESSAGE, md_text)


if __name__ == "__main__":
    unittest.main()
