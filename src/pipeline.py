"""End-to-End Multimodal Inference Pipeline for BrainMRI-SpatialVLM.

Orchestrates the entire research inference workflow:
1. Input NIfTI validation and raw data loading (data/raw/ preserved read-only)
2. Multimodal sequence stacking and saving preprocessed inputs to data/processed/
3. MONAI BraTS 3D tumor segmentation with mask output in outputs/masks/
4. Quantitative spatial feature extraction with metrics saved in outputs/reports/
5. BrainGemma3D vision-language draft report generation in outputs/reports/
6. Tri-planar, montage, and occupancy visualizations in outputs/visualizations/
7. Unified combined JSON results packaging in outputs/reports/

Independent Stage Handling & Guardrails:
- If segmentation fails, reports the failure clearly without aborting downstream
  validations.
- If BrainGemma3D is unavailable or encounters an error, still computes and saves
  segmentation, spatial features, and visualizations.
- Never fabricates missing or uncomputed results.
- Mandatory research disclaimer applied across all outputs:
  "AI-generated research draft — not a medical diagnosis."
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np

from src.config import ProjectConfig, load_config
from src.mri_io import (
    get_volume_metadata,
    load_nifti_volume,
    save_nifti_volume,
    save_processed_volume,
    validate_mri_volume,
)
from src.reporting import BrainGemma3DReporter, DraftReport, MODEL_UNAVAILABLE_MESSAGE
from src.segmentation import BraTSSegmenter, MissingChannelError
from src.spatial_features import (
    SpatialMetrics,
    compare_with_reference_annotation,
    extract_spatial_features,
    save_spatial_features,
)
from src.utils import ensure_directory, get_project_root, is_nifti_file, safe_filename, timestamped_filename
from src.visualization import (
    VISUALIZATION_WATERMARK,
    generate_all_visualizations,
    load_volume_for_visualization,
)

RESEARCH_DISCLAIMER = "AI-generated research draft — not a medical diagnosis."


@dataclass
class PipelineResult:
    """Strongly-typed container for end-to-end pipeline execution artifacts."""

    case_id: str
    timestamp_utc: str
    status: str  # "completed", "partial_success", "failed"
    disclaimer: str = RESEARCH_DISCLAIMER

    # Output artifact file paths
    mask_path: Optional[Path] = None
    processed_input_path: Optional[Path] = None
    spatial_features_path: Optional[Path] = None
    draft_report_path: Optional[Path] = None
    draft_report_md_path: Optional[Path] = None
    visualization_paths: Dict[str, Path] = field(default_factory=dict)
    combined_json_path: Optional[Path] = None

    # Structured data payloads
    spatial_features: Dict[str, Any] = field(default_factory=dict)
    report_data: Dict[str, Any] = field(default_factory=dict)
    reference_comparison: Optional[Dict[str, Any]] = None

    # Stage tracking and error reporting
    stage_statuses: Dict[str, str] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)

    @property
    def paths(self) -> Dict[str, Any]:
        """Convenience dictionary mapping required artifact categories to file paths."""
        return {
            "mask": self.mask_path,
            "spatial_features": self.spatial_features_path,
            "draft_report": self.draft_report_path,
            "draft_report_md": self.draft_report_md_path,
            "visualizations": self.visualization_paths,
            "combined_json": self.combined_json_path,
            "processed_input": self.processed_input_path,
        }

    def __getitem__(self, key: str) -> Any:
        """Allow subscripting for standard path and data access."""
        if key in self.paths:
            return self.paths[key]
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Key '{key}' not found in PipelineResult.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert pipeline result into JSON-serializable dictionary."""
        return {
            "case_id": self.case_id,
            "timestamp_utc": self.timestamp_utc,
            "status": self.status,
            "disclaimer": self.disclaimer,
            "stage_statuses": self.stage_statuses,
            "errors": self.errors,
            "paths": {
                "mask": str(self.mask_path) if self.mask_path else None,
                "processed_input": str(self.processed_input_path) if self.processed_input_path else None,
                "spatial_features": str(self.spatial_features_path) if self.spatial_features_path else None,
                "draft_report": str(self.draft_report_path) if self.draft_report_path else None,
                "draft_report_md": str(self.draft_report_md_path) if self.draft_report_md_path else None,
                "visualizations": {k: str(v) for k, v in self.visualization_paths.items()},
                "combined_json": str(self.combined_json_path) if self.combined_json_path else None,
            },
            "spatial_features": self.spatial_features,
            "reference_comparison": self.reference_comparison,
            "report_data": self.report_data,
        }


def run_full_pipeline(
    input_nifti_path: Union[str, Path],
    config: Optional[Union[str, Path, ProjectConfig]] = None,
    segmenter: Optional[Any] = None,
    reporter: Optional[Any] = None,
    output_dir: Optional[Union[str, Path]] = None,
    reference_mask_path: Optional[Union[str, Path]] = None,
) -> PipelineResult:
    """Execute complete end-to-end multimodal inference pipeline.

    Stages:
    1. Validation of input scan from data/raw/
    2. Sequence stacking & saving preprocessed input to data/processed/
    3. MONAI BraTS 3D tumor segmentation saving to outputs/masks/
    4. Quantitative spatial feature extraction saving to outputs/reports/
    5. BrainGemma3D report generation saving to outputs/reports/
    6. Visualization generation saving to outputs/visualizations/
    7. Combined JSON result aggregation saving to outputs/reports/

    Each stage operates independently: failures in segmentation or report generation
    are recorded clearly without blocking subsequent valid operations.

    Args:
        input_nifti_path: Path to input 3D or 4D NIfTI MRI scan.
        config: Optional ProjectConfig object or path to YAML config file.
        segmenter: Optional pre-configured BraTSSegmenter or mock.
        reporter: Optional pre-configured BrainGemma3DReporter or mock.
        output_dir: Optional base output directory (defaults to configured outputs/).
        reference_mask_path: Optional path to ground-truth reference mask.

    Returns:
        PipelineResult: Strongly-typed result container with all artifact paths and metrics.
    """
    project_root = get_project_root()
    if isinstance(config, ProjectConfig):
        cfg = config
    else:
        cfg = load_config(config)

    resolved_paths = cfg.paths.resolve(project_root)

    # 1. Resolve and validate input file
    input_path = Path(input_nifti_path).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input scan not found: {input_path}")
    if not is_nifti_file(input_path):
        raise ValueError(f"Input file '{input_path.name}' is not a valid NIfTI file.")

    case_id = input_path.name.replace(".nii.gz", "").replace(".nii", "")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    timestamp_iso = datetime.now(timezone.utc).isoformat()

    # Determine destination directories
    base_out = Path(output_dir).resolve() if output_dir else project_root / "outputs"
    masks_dir = base_out / "masks" if output_dir else resolved_paths["output_masks"]
    reports_dir = base_out / "reports" if output_dir else resolved_paths["output_reports"]
    viz_dir = base_out / "visualizations" if output_dir else resolved_paths["output_visualizations"]
    processed_dir = resolved_paths["data_processed"]

    ensure_directory(masks_dir)
    ensure_directory(reports_dir)
    ensure_directory(viz_dir)
    ensure_directory(processed_dir)

    stage_statuses: Dict[str, str] = {}
    errors: Dict[str, str] = {}

    processed_input_path: Optional[Path] = None
    mask_path: Optional[Path] = None
    mask_3d: Optional[np.ndarray] = None
    vol_3d: Optional[np.ndarray] = None
    ref_affine: Optional[np.ndarray] = None
    spatial_features_path: Optional[Path] = None
    spatial_dict: Dict[str, Any] = {}
    ref_comparison: Optional[Dict[str, Any]] = None
    draft_report_path: Optional[Path] = None
    draft_report_md_path: Optional[Path] = None
    report_data: Dict[str, Any] = {}
    viz_paths: Dict[str, Path] = {}

    # -------------------------------------------------------------------------
    # Stage 1: Load NIfTI & Save Preprocessed Input (data/processed/)
    # -------------------------------------------------------------------------
    try:
        nii = nib.load(str(input_path))
        raw_arr = np.asanyarray(nii.dataobj)
        ref_affine = np.array(nii.affine, dtype=np.float64)

        # Squeeze singleton dimension if present
        if raw_arr.ndim == 4 and raw_arr.shape[-1] == 1:
            raw_arr = np.squeeze(raw_arr, axis=-1)

        # Extract 3D volume for visualizations and spatial features
        if raw_arr.ndim == 4:
            # Detect 4-channel orientation: BraTS standard supports (H, W, D, 4) or (4, H, W, D)
            # Standard BraTS order: [0: T1, 1: T1c, 2: T2, 3: FLAIR] -> display FLAIR (index 3)
            flair_idx = 3
            if raw_arr.shape[-1] >= 4:
                vol_3d = raw_arr[:, :, :, flair_idx].astype(np.float32)
            elif raw_arr.shape[0] >= 4:
                vol_3d = raw_arr[flair_idx, :, :, :].astype(np.float32)
            else:
                raise ValueError(
                    f"4D MRI volume must contain 4 modality channels (T1, T1c, T2, FLAIR). "
                    f"Found shape {raw_arr.shape} with insufficient channel dimensions."
                )

            # Save preprocessed multi-sequence input in data/processed/ (never touch data/raw/)
            proc_fname = f"{case_id}_preprocessed_{ts}.nii.gz"
            processed_input_path = save_processed_volume(
                volume=raw_arr.astype(np.float32),
                affine=ref_affine,
                output_path=processed_dir / proc_fname,
                validate_3d=False,
            )
        elif raw_arr.ndim == 3:
            vol_3d = raw_arr.astype(np.float32)
            proc_fname = f"{case_id}_preprocessed_{ts}.nii.gz"
            processed_input_path = save_processed_volume(
                volume=vol_3d,
                affine=ref_affine,
                output_path=processed_dir / proc_fname,
                validate_3d=True,
            )
        else:
            raise ValueError(f"Unsupported input dimensions: {raw_arr.shape}")

        validate_mri_volume(vol_3d)
        stage_statuses["preprocessing"] = "completed"

    except Exception as e:
        stage_statuses["preprocessing"] = "failed"
        errors["preprocessing"] = str(e)
        print(f"[Pipeline] Preprocessing failed: {e}", file=sys.stderr)

    # -------------------------------------------------------------------------
    # Stage 2: MONAI BraTS Segmentation (outputs/masks/)
    # -------------------------------------------------------------------------
    try:
        seg_inst = segmenter
        if seg_inst is None:
            seg_inst = BraTSSegmenter(config_path=cfg.raw_config_path if hasattr(cfg, "raw_config_path") else None)

        mask_out_name = f"{case_id}_mask_{ts}.nii.gz"
        saved_mask, seg_meta = seg_inst.segment_and_save(
            inputs=input_path,
            output_dir=masks_dir,
            output_filename=mask_out_name,
            affine=ref_affine,
            save_processed_input=False,  # Already handled in Stage 1
        )
        mask_path = Path(saved_mask).resolve()

        # Load 3D mask array for downstream features
        mask_nii = nib.load(str(mask_path))
        mask_3d = np.asanyarray(mask_nii.dataobj).astype(np.int32)
        if mask_3d.ndim == 4 and mask_3d.shape[-1] == 1:
            mask_3d = np.squeeze(mask_3d, axis=-1)

        stage_statuses["segmentation"] = "completed"

    except Exception as e:
        stage_statuses["segmentation"] = "failed"
        errors["segmentation"] = str(e)
        print(f"[Pipeline] Segmentation stage failed: {e}", file=sys.stderr)
        # Never fabricate missing results
        mask_path = None
        mask_3d = None

    # -------------------------------------------------------------------------
    # Stage 3: Spatial Feature Extraction (outputs/reports/)
    # -------------------------------------------------------------------------
    try:
        if mask_3d is not None and vol_3d is not None:
            # Enforce spatial dimension consistency between volume and segmentation mask
            if mask_3d.shape != vol_3d.shape:
                raise ValueError(
                    f"Spatial dimension mismatch between MRI volume {vol_3d.shape} and segmentation mask {mask_3d.shape}."
                )

            zooms = None
            if ref_affine is not None:
                # Estimate voxel spacing from affine column norms
                zooms = tuple(float(np.linalg.norm(ref_affine[:3, i])) for i in range(3))

            spatial_metrics = extract_spatial_features(
                mri_volume=vol_3d,
                segmentation_mask=mask_3d,
                voxel_spacing_mm=zooms,
                affine=ref_affine,
            )
            spatial_dict = spatial_metrics.to_dict()

            feat_fname = f"{case_id}_spatial_features_{ts}.json"
            spatial_features_path = save_spatial_features(
                metrics=spatial_metrics,
                output_dir=reports_dir,
                filename=feat_fname,
            )

            # Optional research reference annotation comparison
            try:
                ref_comparison = compare_with_reference_annotation(
                    predicted_mask=mask_3d,
                    reference_mask=reference_mask_path,
                    case_id=case_id,
                )
            except Exception as ref_e:
                ref_comparison = {"status": "skipped", "reason": str(ref_e)}

            stage_statuses["spatial_features"] = "completed"
        else:
            stage_statuses["spatial_features"] = "skipped (no mask generated)"

    except Exception as e:
        stage_statuses["spatial_features"] = "failed"
        errors["spatial_features"] = str(e)
        print(f"[Pipeline] Spatial feature extraction failed: {e}", file=sys.stderr)

    # -------------------------------------------------------------------------
    # Stage 4: BrainGemma3D Draft Report Generation (outputs/reports/)
    # -------------------------------------------------------------------------
    try:
        rep_inst = reporter
        if rep_inst is None:
            rep_inst = BrainGemma3DReporter(
                config_path=cfg.raw_config_path if hasattr(cfg, "raw_config_path") else None,
                require_gpu=False,  # Allow running pipeline check without terminating on CPU
            )

        report_prefix = f"{case_id}_braingemma3d_draft_report_{ts}"
        mri_input_for_rep = vol_3d if vol_3d is not None else input_path

        report, j_path, m_path = rep_inst.generate_and_save(
            mri_volume=mri_input_for_rep,
            mask=mask_3d,
            output_dir=reports_dir,
            filename_prefix=report_prefix,
            spatial_features=spatial_dict,
        )

        draft_report_path = Path(j_path).resolve()
        draft_report_md_path = Path(m_path).resolve()
        report_data = report.to_dict() if isinstance(report, DraftReport) else {"text": str(report)}

        if isinstance(report, DraftReport) and not report.is_model_available:
            stage_statuses["reporting"] = "model_unavailable"
        else:
            stage_statuses["reporting"] = "completed"

    except Exception as e:
        stage_statuses["reporting"] = "failed"
        errors["reporting"] = str(e)
        print(f"[Pipeline] BrainGemma3D reporting stage failed: {e}", file=sys.stderr)
        # Never fabricate a report when absent or failed
        report_data = {
            "status": "failed",
            "findings": MODEL_UNAVAILABLE_MESSAGE,
            "disclaimer": RESEARCH_DISCLAIMER,
        }

    # -------------------------------------------------------------------------
    # Stage 5: Visualization Generation (outputs/visualizations/)
    # -------------------------------------------------------------------------
    try:
        if vol_3d is not None:
            viz_prefix = f"{case_id}_{ts}"
            viz_paths = generate_all_visualizations(
                mri_volume=vol_3d,
                mask=mask_3d,
                output_dir=viz_dir,
                prefix=viz_prefix,
            )
            stage_statuses["visualization"] = "completed"
        else:
            stage_statuses["visualization"] = "skipped (no volume loaded)"

    except Exception as e:
        stage_statuses["visualization"] = "failed"
        errors["visualization"] = str(e)
        print(f"[Pipeline] Visualization generation failed: {e}", file=sys.stderr)

    # -------------------------------------------------------------------------
    # Stage 6: Combined JSON Result (outputs/reports/)
    # -------------------------------------------------------------------------
    # Determine overall status
    if len(errors) == 0 and mask_path is not None:
        overall_status = "completed"
    elif mask_path is not None or len(viz_paths) > 0:
        overall_status = "partial_success"
    else:
        overall_status = "failed"

    combined_json_path: Optional[Path] = None
    try:
        combined_payload = {
            "system": "BrainMRI-SpatialVLM",
            "disclaimer": RESEARCH_DISCLAIMER,
            "case_id": case_id,
            "timestamp_utc": timestamp_iso,
            "overall_status": overall_status,
            "stage_statuses": stage_statuses,
            "errors": errors,
            "paths": {
                "input_scan": str(input_path),
                "processed_input": str(processed_input_path) if processed_input_path else None,
                "mask": str(mask_path) if mask_path else None,
                "spatial_features": str(spatial_features_path) if spatial_features_path else None,
                "draft_report": str(draft_report_path) if draft_report_path else None,
                "draft_report_md": str(draft_report_md_path) if draft_report_md_path else None,
                "visualizations": {k: str(v) for k, v in viz_paths.items()},
            },
            "spatial_features": spatial_dict,
            "reference_comparison": ref_comparison,
            "draft_report": report_data,
        }

        combined_fname = f"{case_id}_pipeline_result_{ts}.json"
        comb_path = reports_dir / safe_filename(combined_fname)
        comb_path.write_text(json.dumps(combined_payload, indent=2), encoding="utf-8")
        combined_json_path = comb_path
        stage_statuses["packaging"] = "completed"

    except Exception as e:
        stage_statuses["packaging"] = "failed"
        errors["packaging"] = str(e)

    # -------------------------------------------------------------------------
    # Return PipelineResult
    # -------------------------------------------------------------------------
    return PipelineResult(
        case_id=case_id,
        timestamp_utc=timestamp_iso,
        status=overall_status,
        disclaimer=RESEARCH_DISCLAIMER,
        mask_path=mask_path,
        processed_input_path=processed_input_path,
        spatial_features_path=spatial_features_path,
        draft_report_path=draft_report_path,
        draft_report_md_path=draft_report_md_path,
        visualization_paths=viz_paths,
        combined_json_path=combined_json_path,
        spatial_features=spatial_dict,
        report_data=report_data,
        reference_comparison=ref_comparison,
        stage_statuses=stage_statuses,
        errors=errors,
    )
