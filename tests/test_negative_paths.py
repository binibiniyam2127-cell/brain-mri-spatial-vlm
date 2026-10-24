"""Negative-path and edge-case testing suite for BrainMRI-SpatialVLM.

Tests error boundaries, fail-safe fallbacks, input validation,
clinical safety governance, and cryptographic provenance tracking.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np

from src.mri_io import validate_mri_volume
from src.pipeline import PipelineResult, run_full_pipeline
from src.reporting import (
    ClinicalPolicyValidator,
    DraftReport,
    RESEARCH_DISCLAIMER,
    SafetyPolicyViolationError,
)
from src.spatial_features import extract_spatial_features
from src.utils import compute_file_sha256, get_git_commit_sha, get_pipeline_logger


class TestClinicalSafetyPolicyEngine(unittest.TestCase):
    """Test suite for ClinicalPolicyValidator and safety guardrails."""

    def test_clean_research_text_passes_validation(self) -> None:
        """Text without forbidden clinical terms passes validation without modification."""
        clean_text = (
            f"Hyperintense signal abnormality in the frontoparietal white matter. "
            f"Automated research observation. {RESEARCH_DISCLAIMER}"
        )
        sanitized, audit = ClinicalPolicyValidator.validate_and_sanitize(clean_text)

        self.assertEqual(sanitized, clean_text)
        self.assertTrue(audit["is_clean"])
        self.assertEqual(audit["quarantined_count"], 0)
        self.assertEqual(audit["violations_detected"], [])
        self.assertTrue(audit["disclaimer_verified"])

        # assert_compliant should not raise
        ClinicalPolicyValidator.assert_compliant(clean_text)

    def test_forbidden_diagnostic_claims_quarantined(self) -> None:
        """Forbidden diagnostic statements are replaced with non-diagnostic quarantine tags."""
        violating_text = (
            "The scan provides a definitive diagnosis of cancer. "
            "Biopsy is not needed because it confirms glioblastoma. "
            "Patient must receive radiotherapy immediately. Prescribe temozolomide."
        )

        sanitized, audit = ClinicalPolicyValidator.validate_and_sanitize(violating_text)

        self.assertFalse(audit["is_clean"])
        self.assertGreaterEqual(audit["quarantined_count"], 4)
        self.assertIn("definitive diagnosis", audit["violations_detected"])
        self.assertIn("confirms glioblastoma", audit["violations_detected"])
        self.assertIn("biopsy is not needed", audit["violations_detected"])
        self.assertIn("prescribe", audit["violations_detected"])

        # Confirm replacement occurred in text
        self.assertNotIn("definitive diagnosis", sanitized.lower())
        self.assertNotIn("confirms glioblastoma", sanitized.lower())
        self.assertIn("[Clinical Claim Quarantined: Non-diagnostic research observation only]", sanitized)

    def test_assert_compliant_raises_on_violations(self) -> None:
        """assert_compliant raises SafetyPolicyViolationError when forbidden terms are present."""
        bad_text = "The imaging features show unequivocal cancer."
        with self.assertRaises(SafetyPolicyViolationError) as ctx:
            ClinicalPolicyValidator.assert_compliant(bad_text)

        self.assertIn("unequivocal cancer", str(ctx.exception))

    def test_draft_report_attaches_safety_policy_audit(self) -> None:
        """DraftReport stores and exports safety policy audit metadata."""
        report = DraftReport(
            content="Sample research draft",
            findings="Sample findings",
            impression="Sample impression",
            safety_policy_audit={
                "policy_version": "1.0.0",
                "is_clean": True,
                "quarantined_count": 0,
                "violations_detected": [],
                "disclaimer_verified": True,
            },
        )

        rep_dict = report.to_dict()
        self.assertIn("safety_policy_audit", rep_dict)
        self.assertTrue(rep_dict["safety_policy_audit"]["is_clean"])


class TestDataIntegrityAndDimensionality(unittest.TestCase):
    """Test suite for input validation, malformed volumes, and edge cases."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_validate_volume_nan_inf_rejection(self) -> None:
        """Volumes containing NaN or Inf values are rejected."""
        nan_vol = np.ones((20, 20, 10), dtype=np.float32)
        nan_vol[5, 5, 5] = np.nan
        with self.assertRaises(ValueError):
            validate_mri_volume(nan_vol)

        inf_vol = np.ones((20, 20, 10), dtype=np.float32)
        inf_vol[5, 5, 5] = np.inf
        with self.assertRaises(ValueError):
            validate_mri_volume(inf_vol)

    def test_pipeline_rejects_unsupported_dimensions(self) -> None:
        """run_full_pipeline fails gracefully when input dimensionality is < 3 or > 4."""
        # Create a 2D NIfTI
        arr_2d = np.ones((32, 32), dtype=np.float32)
        p_2d = self.temp_path / "scan_2d.nii.gz"
        nib.save(nib.Nifti1Image(arr_2d, np.eye(4)), str(p_2d))

        res = run_full_pipeline(input_nifti_path=p_2d, output_dir=self.temp_path / "out")
        self.assertEqual(res.status, "failed")
        self.assertIn("preprocessing", res.errors)
        self.assertIn("Unsupported MRI volume dimensionality", res.errors["preprocessing"])

    def test_pipeline_rejects_invalid_4d_channel_counts(self) -> None:
        """4D NIfTI with insufficient channels (e.g. 2 channels instead of 4) is rejected."""
        arr_4d_bad = np.ones((20, 20, 10, 2), dtype=np.float32)
        p_bad_4d = self.temp_path / "scan_bad_4d.nii.gz"
        nib.save(nib.Nifti1Image(arr_4d_bad, np.eye(4)), str(p_bad_4d))

        res = run_full_pipeline(input_nifti_path=p_bad_4d, output_dir=self.temp_path / "out")
        self.assertEqual(res.status, "failed")
        self.assertIn("preprocessing", res.errors)
        self.assertIn("must contain 4 modality channels", res.errors["preprocessing"])

    def test_spatial_features_shape_mismatch_raises_error(self) -> None:
        """Spatial feature extraction raises ValueError when volume and mask shapes differ."""
        vol = np.ones((30, 30, 15), dtype=np.float32)
        mask_mismatch = np.zeros((30, 30, 20), dtype=np.uint8)

        with self.assertRaises(ValueError) as ctx:
            extract_spatial_features(vol, mask_mismatch)
        self.assertIn("does not match", str(ctx.exception).lower())


class TestCryptographicProvenanceAndLogging(unittest.TestCase):
    """Test suite for cryptographic hashing, git tracking, and structured logging."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_compute_file_sha256_accuracy(self) -> None:
        """compute_file_sha256 computes consistent 64-character lowercase hex digest."""
        test_file = self.temp_path / "sample.bin"
        test_content = b"BrainMRI-SpatialVLM-Provenance-Verification-Bytes"
        test_file.write_bytes(test_content)

        digest = compute_file_sha256(test_file)
        self.assertEqual(len(digest), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in digest))

        # Re-computing on same file produces exact match
        digest2 = compute_file_sha256(test_file)
        self.assertEqual(digest, digest2)

    def test_compute_file_sha256_missing_file_raises(self) -> None:
        """compute_file_sha256 raises FileNotFoundError on missing file."""
        with self.assertRaises(FileNotFoundError):
            compute_file_sha256(self.temp_path / "non_existent.nii.gz")

    def test_get_git_commit_sha(self) -> None:
        """get_git_commit_sha returns 40-character hex SHA from local git repo."""
        sha = get_git_commit_sha()
        self.assertIsNotNone(sha)
        self.assertEqual(len(sha), 40)

    def test_pipeline_generates_provenance_manifest(self) -> None:
        """End-to-end pipeline run produces a valid cryptographic provenance manifest."""
        synth_arr = np.ones((24, 24, 12), dtype=np.float32) * 50.0
        synth_arr[8:16, 8:16, 4:8] = 200.0
        synth_path = self.temp_path / "provenance_test_scan.nii.gz"
        nib.save(nib.Nifti1Image(synth_arr, np.eye(4)), str(synth_path))

        out_dir = self.temp_path / "outputs"
        result = run_full_pipeline(
            input_nifti_path=synth_path,
            output_dir=out_dir,
        )

        # Confirm provenance path and file exist
        self.assertIsNotNone(result.provenance_path)
        self.assertTrue(result.provenance_path.is_file())

        # Load and verify manifest structure
        manifest = json.loads(result.provenance_path.read_text(encoding="utf-8"))

        self.assertIn("run_id", manifest)
        self.assertIn("case_id", manifest)
        self.assertIn("git_commit_sha", manifest)
        self.assertIn("system_environment", manifest)
        self.assertIn("inputs", manifest)
        self.assertIn("outputs", manifest)
        self.assertIn("stage_latencies_ms", manifest)
        self.assertIn("safety_governance", manifest)

        # Verify input hash is present and valid
        self.assertIsNotNone(manifest["inputs"]["input_scan_sha256"])
        self.assertEqual(len(manifest["inputs"]["input_scan_sha256"]), 64)

        # Verify latencies are recorded
        latencies = manifest["stage_latencies_ms"]
        self.assertIn("total_pipeline_latency_ms", latencies)
        self.assertGreater(latencies["total_pipeline_latency_ms"], 0)

        # Verify safety governance disclaimer
        self.assertEqual(manifest["safety_governance"]["disclaimer"], RESEARCH_DISCLAIMER)


if __name__ == "__main__":
    unittest.main()

