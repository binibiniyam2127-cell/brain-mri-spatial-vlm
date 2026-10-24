"""Tests for scripts/download_sample_data.py CLI, arguments, and configuration."""

from pathlib import Path
import subprocess
import sys
import unittest

from src.config import load_config
from src.utils import get_project_root


class TestSampleDataScript(unittest.TestCase):
    """Test suite for sample data downloader script without triggering downloads."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = get_project_root()
        cls.script_path = cls.project_root / "scripts" / "download_sample_data.py"
        cls.config_path = cls.project_root / "configs" / "project_config.yaml"
        cls.expected_url = (
            "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
        )

    def test_script_exists_and_non_empty(self) -> None:
        """Verify download_sample_data.py exists on disk and is non-empty."""
        self.assertTrue(self.script_path.is_file(), f"Script not found: {self.script_path}")
        self.assertGreater(self.script_path.stat().st_size, 0)

    def test_script_contains_expected_dataset_url(self) -> None:
        """Verify the script contains the official Task01_BrainTumour URL."""
        content = self.script_path.read_text(encoding="utf-8")
        self.assertIn(self.expected_url, content)

    def test_script_accepts_expected_cli_arguments(self) -> None:
        """Verify the script CLI help output documents all required flags."""
        cmd = [sys.executable, str(self.script_path), "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        help_text = result.stdout

        expected_flags = [
            "--url",
            "--archive-path",
            "--max-cases",
            "--output-dir",
            "--annotations-dir",
            "--keep-archive",
        ]
        for flag in expected_flags:
            with self.subTest(flag=flag):
                self.assertIn(flag, help_text)

    def test_default_max_cases_is_25(self) -> None:
        """Verify the default maximum case count is 25 in script help and configuration."""
        import re

        cmd = [sys.executable, str(self.script_path), "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        help_text = re.sub(r"\s+", " ", result.stdout)
        self.assertIn("default: 25", help_text)

        cfg = load_config(self.config_path)
        self.assertEqual(cfg.sample_data.max_cases, 25)

    def test_config_contains_sample_data_section(self) -> None:
        """Verify configs/project_config.yaml contains sample_data entries including annotations_dir."""
        cfg = load_config(self.config_path)
        self.assertTrue(hasattr(cfg, "sample_data"))
        self.assertEqual(cfg.sample_data.download_url, self.expected_url)
        self.assertEqual(cfg.sample_data.raw_data_dir, "data/raw")
        self.assertEqual(cfg.sample_data.annotations_dir, "data/annotations")
        self.assertEqual(cfg.sample_data.downloads_dir, "data/downloads")
        self.assertEqual(cfg.sample_data.max_cases, 25)


if __name__ == "__main__":
    unittest.main()
