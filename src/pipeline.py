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
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

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
from src.utils import (
    compute_file_sha256,
    ensure_directory,
    get_git_commit_sha,
    get_pipeline_logger,
    get_project_root,
    is_nifti_file,
    safe_filename,
    timestamped_filename,
)
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
    provenance_path: Optional[Path] = None

    # Structured data payloads
    spatial_features: Dict[str, Any] = field(default_factory=dict)
    report_data: Dict[str, Any] = field(default_factory=dict)
    reference_comparison: Optional[Dict[str, Any]] = None
    provenance_data: Dict[str, Any] = field(default_factory=dict)

    # Stage tracking, latency, and error reporting
    stage_statuses: Dict[str, str] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    stage_latencies_ms: Dict[str, float] = field(default_factory=dict)
    run_id: str = ""

    # In-memory arrays to prevent repeated file reading and reduce memory pressure
    vol_3d: Optional[np.ndarray] = None
    mask_3d: Optional[np.ndarray] = None
    mask_source: str = "none"

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
            "provenance": self.provenance_path,
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
            "run_id": self.run_id,
            "case_id": self.case_id,
            "timestamp_utc": self.timestamp_utc,
            "status": self.status,
            "disclaimer": self.disclaimer,
            "stage_statuses": self.stage_statuses,
            "stage_latencies_ms": self.stage_latencies_ms,
            "errors": self.errors,
            "paths": {
                "mask": str(self.mask_path) if self.mask_path else None,
                "processed_input": str(self.processed_input_path) if self.processed_input_path else None,
                "spatial_features": str(self.spatial_features_path) if self.spatial_features_path else None,
                "draft_report": str(self.draft_report_path) if self.draft_report_path else None,
                "draft_report_md": str(self.draft_report_md_path) if self.draft_report_md_path else None,
                "visualizations": {k: str(v) for k, v in self.visualization_paths.items()},
                "combined_json": str(self.combined_json_path) if self.combined_json_path else None,
                "provenance": str(self.provenance_path) if self.provenance_path else None,
            },
            "spatial_features": self.spatial_features,
            "reference_comparison": self.reference_comparison,
            "report_data": self.report_data,
            "provenance_data": self.provenance_data,
        }


def run_full_pipeline(
    input_nifti_path: Union[str, Path],
    config: Optional[Union[str, Path, ProjectConfig]] = None,
    segmenter: Optional[Any] = None,
    reporter: Optional[Any] = None,
    output_dir: Optional[Union[str, Path]] = None,
    reference_mask_path: Optional[Union[str, Path]] = None,
    generate_visualizations: bool = True,
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
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    t_total_start = time.perf_counter()

    logger = get_pipeline_logger("pipeline", run_id=run_id, stage="init")
    logger.info(f"Initialized multimodal pipeline for case_id='{case_id}', input='{input_path.name}'")

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
    stage_latencies_ms: Dict[str, float] = {}
    errors: Dict[str, str] = {}

    processed_input_path: Optional[Path] = None
    mask_path: Optional[Path] = None
    mask_3d: Optional[np.ndarray] = None
    mask_source: str = "none"
    vol_3d: Optional[np.ndarray] = None
    ref_affine: Optional[np.ndarray] = None
    spatial_features_path: Optional[Path] = None
    spatial_dict: Dict[str, Any] = {}
    ref_comparison: Optional[Dict[str, Any]] = None
    draft_report_path: Optional[Path] = None
    draft_report_md_path: Optional[Path] = None
    report_data: Dict[str, Any] = {}
    viz_paths: Dict[str, Path] = {}

    # Compute input cryptographic digest
    try:
        input_sha256 = compute_file_sha256(input_path)
    except Exception as e:
        input_sha256 = f"error: {e}"

    # -------------------------------------------------------------------------
    # Stage 1: Load NIfTI & Save Preprocessed Input (data/processed/)
    # -------------------------------------------------------------------------
    t_stage = time.perf_counter()
    logger.extra["stage"] = "preprocessing"
    logger.info("Stage 1/6: Loading NIfTI & validating array dimensionality...")
    try:
        nii = nib.load(str(input_path))
        raw_arr = np.asanyarray(nii.dataobj)
        ref_affine = np.array(nii.affine, dtype=np.float64)

        # Early validation: Reject unsupported dimensions (< 3 or > 4)
        if raw_arr.ndim < 3 or raw_arr.ndim > 4:
            raise ValueError(
                f"Unsupported MRI volume dimensionality: Expected 3D or 4D NIfTI, "
                f"received {raw_arr.ndim}D array with shape {raw_arr.shape}."
            )

        # Squeeze singleton dimension if present
        if raw_arr.ndim == 4 and raw_arr.shape[-1] == 1:
            raw_arr = np.squeeze(raw_arr, axis=-1)
        elif raw_arr.ndim == 4 and raw_arr.shape[0] == 1:
            raw_arr = np.squeeze(raw_arr, axis=0)

        # Extract 3D volume for visualizations and spatial features
        if raw_arr.ndim == 4:
            # Detect 4-channel orientation: BraTS standard requires exactly 4 channels (T1, T1c, T2, FLAIR)
            # Standard BraTS order: [0: T1, 1: T1c, 2: T2, 3: FLAIR] -> display FLAIR (index 3)
            flair_idx = 3
            if raw_arr.shape[-1] == 4:
                vol_3d = raw_arr[:, :, :, flair_idx].astype(np.float32)
            elif raw_arr.shape[0] == 4:
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
        logger.info(f"Preprocessing completed successfully (3D shape: {vol_3d.shape})")

    except Exception as e:
        stage_statuses["preprocessing"] = "failed"
        errors["preprocessing"] = str(e)
        logger.error(f"Preprocessing stage failed: {e}")

    stage_latencies_ms["preprocessing"] = round((time.perf_counter() - t_stage) * 1000, 2)

    # -------------------------------------------------------------------------
    # Stage 2: MONAI BraTS Segmentation (outputs/masks/)
    # -------------------------------------------------------------------------
    t_stage = time.perf_counter()
    logger.extra["stage"] = "segmentation"
    logger.info("Stage 2/6: Executing MONAI BraTS 3D tumor segmentation...")
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
        mask_source = "MONAI 3D SegResNet Model Prediction"
        logger.info(f"Segmentation completed. Output mask saved to {mask_path.name}")

    except Exception as e:
        stage_statuses["segmentation"] = "failed"
        errors["segmentation"] = str(e)
        logger.error(f"Segmentation stage failed: {e}")
        # Never fabricate missing results
        mask_path = None
        mask_3d = None
        mask_source = "none"

    stage_latencies_ms["segmentation"] = round((time.perf_counter() - t_stage) * 1000, 2)

    # -------------------------------------------------------------------------
    # Stage 3: Spatial Feature Extraction (outputs/reports/)
    # -------------------------------------------------------------------------
    t_stage = time.perf_counter()
    logger.extra["stage"] = "spatial_features"
    logger.info("Stage 3/6: Extracting quantitative spatial features & lesion metrics...")
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
            logger.info(f"Spatial analytics extracted: {spatial_dict.get('total_mask_voxels', 0)} lesion voxels")
        else:
            stage_statuses["spatial_features"] = "skipped (no mask generated)"
            logger.warning("Spatial features skipped due to absent segmentation mask.")

    except Exception as e:
        stage_statuses["spatial_features"] = "failed"
        errors["spatial_features"] = str(e)
        logger.error(f"Spatial feature extraction failed: {e}")

    stage_latencies_ms["spatial_features"] = round((time.perf_counter() - t_stage) * 1000, 2)

    # -------------------------------------------------------------------------
    # Stage 4: BrainGemma3D Draft Report Generation (outputs/reports/)
    # -------------------------------------------------------------------------
    t_stage = time.perf_counter()
    logger.extra["stage"] = "reporting"
    logger.info("Stage 4/6: Generating BrainGemma3D vision-language draft report...")
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
            logger.info("Reporting completed: BrainGemma3D model unavailable (transparent non-diagnostic fallback)")
        else:
            stage_statuses["reporting"] = "completed"
            logger.info("BrainGemma3D report generated and verified against clinical safety policy")

    except Exception as e:
        stage_statuses["reporting"] = "failed"
        errors["reporting"] = str(e)
        logger.error(f"BrainGemma3D reporting stage failed: {e}")
        # Never fabricate a report when absent or failed
        report_data = {
            "status": "failed",
            "findings": MODEL_UNAVAILABLE_MESSAGE,
            "disclaimer": RESEARCH_DISCLAIMER,
        }

    stage_latencies_ms["reporting"] = round((time.perf_counter() - t_stage) * 1000, 2)

    # -------------------------------------------------------------------------
    # Stage 5: Visualization Generation (outputs/visualizations/)
    # -------------------------------------------------------------------------
    t_stage = time.perf_counter()
    logger.extra["stage"] = "visualization"
    if generate_visualizations:
        logger.info("Stage 5/6: Generating tri-planar, montage, and slice occupancy visualizations...")
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
                logger.info(f"Visualizations saved ({len(viz_paths)} figures created)")
            else:
                stage_statuses["visualization"] = "skipped (no volume loaded)"
                logger.warning("Visualization stage skipped because 3D volume was not loaded.")

        except Exception as e:
            stage_statuses["visualization"] = "failed"
            errors["visualization"] = str(e)
            logger.error(f"Visualization generation failed: {e}")
    else:
        stage_statuses["visualization"] = "deferred_to_interactive_ui"
        logger.info("Stage 5/6: Static PNG visualizations skipped (deferred to interactive Gradio WebGL/MPR viewer).")

    stage_latencies_ms["visualization"] = round((time.perf_counter() - t_stage) * 1000, 2)

    # -------------------------------------------------------------------------
    # Stage 6: Combined JSON Result (outputs/reports/)
    # -------------------------------------------------------------------------
    t_stage = time.perf_counter()
    logger.extra["stage"] = "packaging"
    logger.info("Stage 6/6: Aggregating combined execution results...")
    # Determine overall status
    if "preprocessing" in errors:
        overall_status = "failed"
    elif len(errors) == 0 and mask_path is not None:
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
            "run_id": run_id,
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
        logger.error(f"Packaging failed: {e}")

    stage_latencies_ms["packaging"] = round((time.perf_counter() - t_stage) * 1000, 2)

    # -------------------------------------------------------------------------
    # Stage 7: Run Provenance Manifest (outputs/reports/)
    # -------------------------------------------------------------------------
    t_stage = time.perf_counter()
    logger.extra["stage"] = "provenance"
    logger.info("Generating cryptographic Run Provenance Manifest...")

    provenance_path: Optional[Path] = None
    provenance_payload: Dict[str, Any] = {}
    try:
        # Determine PyTorch and CUDA runtime status
        torch_mod = sys.modules.get("torch")
        torch_ver = getattr(torch_mod, "__version__", "not_loaded") if torch_mod else "not_loaded"
        cuda_avail = bool(torch_mod.cuda.is_available()) if torch_mod and hasattr(torch_mod, "cuda") else False
        cuda_dev = (
            torch_mod.cuda.get_device_name(0)
            if cuda_avail and hasattr(torch_mod.cuda, "get_device_name")
            else "none"
        )

        # Compute output file cryptographic SHA-256 digests
        def safe_digest(p: Optional[Path]) -> Optional[str]:
            if p is not None and p.is_file():
                try:
                    return compute_file_sha256(p)
                except Exception:
                    return None
            return None

        ref_mask_digest = safe_digest(Path(reference_mask_path).resolve()) if reference_mask_path else None

        provenance_payload = {
            "run_id": run_id,
            "case_id": case_id,
            "timestamp_utc": timestamp_iso,
            "git_commit_sha": get_git_commit_sha(project_root) or "unversioned",
            "system_environment": {
                "python_version": sys.version.split()[0],
                "platform": sys.platform,
                "torch_version": torch_ver,
                "cuda_available": cuda_avail,
                "cuda_device_name": cuda_dev,
            },
            "inputs": {
                "input_scan_path": str(input_path),
                "input_scan_sha256": input_sha256,
                "reference_mask_path": str(reference_mask_path) if reference_mask_path else None,
                "reference_mask_sha256": ref_mask_digest,
            },
            "outputs": {
                "mask_sha256": safe_digest(mask_path),
                "processed_input_sha256": safe_digest(processed_input_path),
                "spatial_features_sha256": safe_digest(spatial_features_path),
                "draft_report_sha256": safe_digest(draft_report_path),
                "draft_report_md_sha256": safe_digest(draft_report_md_path),
                "combined_json_sha256": safe_digest(combined_json_path),
            },
            "stage_statuses": stage_statuses,
            "safety_governance": {
                "disclaimer": RESEARCH_DISCLAIMER,
                "clinical_policy_audit": report_data.get("safety_policy_audit", {}),
            },
            "stage_latencies_ms": stage_latencies_ms,
        }

        stage_latencies_ms["provenance"] = round((time.perf_counter() - t_stage) * 1000, 2)
        stage_latencies_ms["total_pipeline_latency_ms"] = round((time.perf_counter() - t_total_start) * 1000, 2)
        provenance_payload["stage_latencies_ms"] = stage_latencies_ms

        prov_fname = f"{case_id}_provenance_{ts}.json"
        prov_path = reports_dir / safe_filename(prov_fname)
        prov_path.write_text(json.dumps(provenance_payload, indent=2), encoding="utf-8")
        provenance_path = prov_path
        stage_statuses["provenance"] = "completed"
        logger.info(f"Run provenance manifest saved to {prov_path.name}")

    except Exception as e:
        stage_statuses["provenance"] = "failed"
        errors["provenance"] = str(e)
        logger.error(f"Provenance generation failed: {e}")

    logger.extra["stage"] = "done"
    logger.info(
        f"Pipeline run {run_id} finished with status '{overall_status}' "
        f"in {stage_latencies_ms.get('total_pipeline_latency_ms', 0)} ms."
    )

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
        provenance_path=provenance_path,
        spatial_features=spatial_dict,
        report_data=report_data,
        reference_comparison=ref_comparison,
        provenance_data=provenance_payload,
        stage_statuses=stage_statuses,
        errors=errors,
        stage_latencies_ms=stage_latencies_ms,
        run_id=run_id,
        vol_3d=vol_3d,
        mask_3d=mask_3d,
        mask_source=mask_source,
    )
