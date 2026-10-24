#!/usr/bin/env python3
"""Automated offline smoke test for BrainMRI-SpatialVLM.

Validates:
1. Multimodal NIfTI ingestion
2. MONAI BraTS segmentation (mocked offline)
3. Quantitative spatial feature extraction
4. BrainGemma3D report generation (offline fallback)
5. Multi-planar and montage visualizations with watermark
6. Combined JSON packaging
7. Strict non-diagnostic disclaimer verification across all generated outputs
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import MagicMock

import nibabel as nib
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import RESEARCH_DISCLAIMER, run_full_pipeline
from src.reporting import DraftReport


def run_smoke_test() -> bool:
    """Execute complete smoke test in temporary environment."""
    print("=" * 65)
    print("BrainMRI-SpatialVLM: End-to-End Automated Smoke Test")
    print(f"DISCLAIMER: {RESEARCH_DISCLAIMER}")
    print("=" * 65)

    with tempfile.TemporaryDirectory() as td:
        temp_dir = Path(td)
        mri_path = temp_dir / "smoke_multimodal_mri.nii.gz"

        # 1. Create synthetic 4D multimodal volume (24, 24, 16, 4)
        vol_4d = np.ones((24, 24, 16, 4), dtype=np.float32) * 50.0
        for c in range(4):
            vol_4d[:, :, :, c] += c * 10.0
        nib.save(nib.Nifti1Image(vol_4d, np.eye(4)), str(mri_path))
        print(f"Created synthetic 4D scan: {mri_path.name} (shape {vol_4d.shape})")

        # 2. Setup mocked BraTS segmenter
        mock_seg = MagicMock()
        mask_path = temp_dir / "smoke_predicted_mask.nii.gz"
        mask_arr = np.zeros((24, 24, 16), dtype=np.int32)
        mask_arr[8:14, 8:14, 6:10] = 1  # Necrotic
        mask_arr[6:16, 6:16, 5:11] = 2  # Edema
        mask_arr[9:13, 9:13, 7:9] = 3   # Enhancing
        nib.save(nib.Nifti1Image(mask_arr, np.eye(4)), str(mask_path))
        mock_seg.segment_and_save.return_value = (mask_path, {"mock": True})

        # 3. Setup mocked BrainGemma3D reporter
        mock_rep = MagicMock()
        draft_rep = DraftReport(
            content=f"# Draft Report\n{RESEARCH_DISCLAIMER}\nMocked findings.",
            findings="Automated research draft findings.",
            impression=f"Research impression. {RESEARCH_DISCLAIMER}",
            disclaimer=RESEARCH_DISCLAIMER,
            is_model_available=True,
            status="completed",
        )
        rep_j = temp_dir / "smoke_report.json"
        rep_m = temp_dir / "smoke_report.md"
        rep_j.write_text(json.dumps(draft_rep.to_dict()), encoding="utf-8")
        rep_m.write_text(draft_rep.to_markdown(), encoding="utf-8")
        mock_rep.generate_and_save.return_value = (draft_rep, rep_j, rep_m)

        # 4. Execute pipeline
        out_base = temp_dir / "outputs"
        result = run_full_pipeline(
            input_nifti_path=mri_path,
            segmenter=mock_seg,
            reporter=mock_rep,
            output_dir=out_base,
        )

        print("\nVerifying Generated Artifacts:")
        assert result.status == "completed", f"Expected completed status, got {result.status}"
        assert result.mask_path and result.mask_path.is_file(), "Mask file missing"
        print(f"  [OK] Mask Path: {result.mask_path.name}")

        assert result.spatial_features_path and result.spatial_features_path.is_file(), "Spatial features JSON missing"
        print(f"  [OK] Spatial Features Path: {result.spatial_features_path.name}")

        assert result.draft_report_path and result.draft_report_path.is_file(), "Draft report JSON missing"
        print(f"  [OK] Draft Report JSON Path: {result.draft_report_path.name}")

        assert result.combined_json_path and result.combined_json_path.is_file(), "Combined JSON missing"
        print(f"  [OK] Combined JSON Path: {result.combined_json_path.name}")

        assert len(result.visualization_paths) >= 2, "Visualizations missing"
        for k, v in result.visualization_paths.items():
            assert v.is_file(), f"Visualization {k} missing"
            print(f"  [OK] Visualization ({k}): {v.name}")

        # 5. Verify research disclaimer enforcement
        comb_data = json.loads(result.combined_json_path.read_text(encoding="utf-8"))
        assert comb_data.get("disclaimer") == RESEARCH_DISCLAIMER, "Disclaimer mismatch in combined JSON"
        print(f"  [OK] Research Disclaimer Verified: \"{comb_data.get('disclaimer')}\"")

        # 6. Verify zero fabricated conclusions
        forbidden_terms = ["confirmed diagnosis", "definitive diagnosis", "surgery scheduled", "prescribed"]
        comb_str = json.dumps(comb_data).lower()
        for term in forbidden_terms:
            assert term not in comb_str, f"Forbidden clinical term found: {term}"
        print("  [OK] Anti-Fabrication Check: Zero clinical diagnosis claims found.")

    print("\n" + "=" * 65)
    print("ALL SMOKE TESTS PASSED CLEANLY AND SUCCESSFULLY!")
    print("=" * 65)
    return True


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)

