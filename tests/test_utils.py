"""Tests for src/utils.py module."""

from datetime import datetime
from pathlib import Path
import tempfile
import unittest

from src.utils import (
    ensure_directory,
    get_project_root,
    is_nifti_file,
    safe_filename,
    timestamped_filename,
)


class TestUtils(unittest.TestCase):
    """Test suite for utility helper functions."""

    def test_get_project_root(self) -> None:
        """Verify get_project_root returns existing project root."""
        root = get_project_root()
        self.assertIsInstance(root, Path)
        self.assertTrue(root.exists())
        self.assertTrue((root / "configs" / "project_config.yaml").exists())

    def test_ensure_directory(self) -> None:
        """Verify ensure_directory creates directory hierarchy correctly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "sub1" / "sub2" / "test_dir"
            self.assertFalse(target.exists())
            created = ensure_directory(target)
            self.assertTrue(created.exists())
            self.assertTrue(created.is_dir())
            self.assertEqual(created, target.resolve())

    def test_is_nifti_file(self) -> None:
        """Verify identification of valid NIfTI file extensions."""
        valid_cases = [
            "brain.nii",
            "brain.nii.gz",
            "SCAN_T1CE.NII.GZ",
            "PATIENT_001.NII",
            "/path/to/nested/scan.nii.gz",
            Path("relative/path/volume.nii"),
        ]
        invalid_cases = [
            "brain.dcm",
            "image.png",
            "scan.nii.tar",
            "archive.tar.gz",
            "data.csv",
            "nii",
            ".nii.backup",
        ]

        for case in valid_cases:
            with self.subTest(valid_case=case):
                self.assertTrue(
                    is_nifti_file(case), f"Expected True for {case}"
                )

        for case in invalid_cases:
            with self.subTest(invalid_case=case):
                self.assertFalse(
                    is_nifti_file(case), f"Expected False for {case}"
                )

    def test_safe_filename(self) -> None:
        """Verify safe_filename sanitizes invalid characters and whitespace."""
        test_cases = [
            ("brain scan 01.nii.gz", "brain_scan_01.nii.gz"),
            ("tumor:scan/test?.nii", "test.nii"),
            ("  whitespace_test   ", "whitespace_test"),
            ("../../../etc/passwd", "passwd"),
            ("bad*chars<>|\".txt", "badchars.txt"),
            ("...", "unnamed"),
            ("", "unnamed"),
        ]

        for raw_input, expected in test_cases:
            with self.subTest(raw_input=raw_input):
                sanitized = safe_filename(raw_input)
                self.assertEqual(sanitized, expected)

    def test_timestamped_filename(self) -> None:
        """Verify timestamped_filename formats prefixes, datetimes, and extensions."""
        fixed_dt = datetime(2026, 10, 23, 12, 30, 45)
        # With clean extension
        res1 = timestamped_filename("tumor_mask", "nii.gz", dt=fixed_dt)
        self.assertEqual(res1, "tumor_mask_20261023_123045.nii.gz")

        # With leading dot in extension
        res2 = timestamped_filename("report", ".json", dt=fixed_dt)
        self.assertEqual(res2, "report_20261023_123045.json")

        # Unsanitized prefix input
        res3 = timestamped_filename("draft report 01!", "txt", dt=fixed_dt)
        self.assertEqual(res3, "draft_report_01_20261023_123045.txt")

        # Default datetime (not None) generates valid timestamp format
        res_current = timestamped_filename("scan", "nii")
        self.assertTrue(res_current.startswith("scan_"))
        self.assertTrue(res_current.endswith(".nii"))


if __name__ == "__main__":
    unittest.main()

