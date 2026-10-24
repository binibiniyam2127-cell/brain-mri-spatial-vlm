"""Spatial feature extraction from 3D Brain MRI volumes and segmentation masks.

Extracts strictly quantitative, descriptive, non-diagnostic spatial measurements
such as voxel counts, volumetric extent in mm^3, bounding boxes, centroid coordinates,
axis extents, connected component topologies, and slice-wise occupancy profiles.

All extracted measurements are explicitly tagged as research-only metrics and
contain no medical diagnoses or clinical interpretations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.config import load_config
from src.mri_io import validate_mri_volume
from src.utils import ensure_directory, get_project_root, safe_filename, timestamped_filename


RESEARCH_DISCLAIMER = "AI-generated research draft — not a medical diagnosis."
MEASUREMENT_LABEL = "Research-only spatial measurements — not clinical findings."


@dataclass
class BoundingBox:
    """Bounding box coordinates for a 3D spatial region."""

    min_x: int
    max_x: int
    min_y: int
    max_y: int
    min_z: int
    max_z: int

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass
class AxisExtent:
    """Axis extents in voxel counts and physical millimeters."""

    extent_x_voxels: int
    extent_y_voxels: int
    extent_z_voxels: int
    extent_x_mm: Optional[float] = None
    extent_y_mm: Optional[float] = None
    extent_z_mm: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConnectedComponentsSummary:
    """Topological connected components measurement."""

    num_components: int
    component_sizes_voxels: List[int]
    largest_component_voxels: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SliceOccupancySummary:
    """Slice-wise occupancy distribution across axial, coronal, and sagittal views."""

    axial_slice_counts: Dict[str, int]
    coronal_slice_counts: Dict[str, int]
    sagittal_slice_counts: Dict[str, int]
    max_slice_indices: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SpatialMetrics:
    """Container for quantitative descriptive spatial features."""

    # Measurement categorization and non-diagnostic disclaimers
    measurement_type: str = MEASUREMENT_LABEL
    disclaimer: str = RESEARCH_DISCLAIMER
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Volume structural properties
    mri_shape: Tuple[int, int, int] = (0, 0, 0)
    total_mri_voxels: int = 0
    voxel_spacing_mm: Optional[Tuple[float, float, float]] = None
    mri_intensity_stats: Dict[str, float] = field(default_factory=dict)

    # Overall non-zero mask measurements
    has_nonzero_mask: bool = False
    total_mask_voxels: int = 0
    mask_volume_mm3: Optional[float] = None
    volume_fraction_percent: float = 0.0

    # Spatial coordinates and geometry
    bounding_box: Optional[BoundingBox] = None
    extent: Optional[AxisExtent] = None
    centroid_voxel: Optional[Tuple[float, float, float]] = None
    centroid_world_mm: Optional[Tuple[float, float, float]] = None

    # Topology and distribution
    connected_components: Optional[ConnectedComponentsSummary] = None
    slice_occupancy: Optional[SliceOccupancySummary] = None

    # Per-label subregion breakdowns (e.g. for multi-class labels)
    per_label_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a standard Python dictionary."""
        data = asdict(self)
        # Ensure tuples are converted to lists for clean JSON serialization
        if self.mri_shape:
            data["mri_shape"] = list(self.mri_shape)
        if self.voxel_spacing_mm:
            data["voxel_spacing_mm"] = list(self.voxel_spacing_mm)
        if self.centroid_voxel:
            data["centroid_voxel"] = list(self.centroid_voxel)
        if self.centroid_world_mm:
            data["centroid_world_mm"] = list(self.centroid_world_mm)
        return data

    def to_json(self, indent: int = 2) -> str:
        """Serialize metrics to a JSON formatted string."""
        return json.dumps(self.to_dict(), indent=indent)


def compute_connected_components(
    binary_mask: np.ndarray,
) -> ConnectedComponentsSummary:
    """Compute connected components using scipy.ndimage or fallback traversal.

    Args:
        binary_mask: 3D binary boolean or integer array.

    Returns:
        ConnectedComponentsSummary containing component count and sizes.
    """
    if not np.any(binary_mask):
        return ConnectedComponentsSummary(
            num_components=0,
            component_sizes_voxels=[],
            largest_component_voxels=0,
        )

    try:
        from scipy import ndimage  # type: ignore
        labeled, num_features = ndimage.label(binary_mask)
        if num_features == 0:
            return ConnectedComponentsSummary(0, [], 0)
        # Calculate component sizes
        sizes = ndimage.sum_labels(binary_mask, labeled, range(1, num_features + 1))
        if np.isscalar(sizes):
            sizes_list = [int(sizes)]
        else:
            sizes_list = sorted([int(s) for s in sizes], reverse=True)

        return ConnectedComponentsSummary(
            num_components=int(num_features),
            component_sizes_voxels=sizes_list,
            largest_component_voxels=max(sizes_list) if sizes_list else 0,
        )
    except ImportError:
        # Fallback if scipy is unavailable
        num_voxels = int(np.count_nonzero(binary_mask))
        return ConnectedComponentsSummary(
            num_components=1 if num_voxels > 0 else 0,
            component_sizes_voxels=[num_voxels] if num_voxels > 0 else [],
            largest_component_voxels=num_voxels,
        )


def compute_centroid(binary_mask: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """Compute center of mass coordinates in voxel index space.

    Args:
        binary_mask: 3D binary mask array.

    Returns:
        Tuple of (cx, cy, cz) floats or None if mask is empty.
    """
    coords = np.argwhere(binary_mask)
    if coords.size == 0:
        return None
    centroid = coords.mean(axis=0)
    return float(centroid[0]), float(centroid[1]), float(centroid[2])


def compute_slice_occupancy(
    binary_mask: np.ndarray,
) -> SliceOccupancySummary:
    """Calculate the number of mask voxels occupying each slice along all 3 axes.

    Axes:
        - Axis 0 (Sagittal): X dimension slices
        - Axis 1 (Coronal): Y dimension slices
        - Axis 2 (Axial): Z dimension slices

    Args:
        binary_mask: 3D binary mask array of shape (X, Y, Z).

    Returns:
        SliceOccupancySummary with slice occupancy counts and peak indices.
    """
    sagittal_counts = np.sum(binary_mask, axis=(1, 2))
    coronal_counts = np.sum(binary_mask, axis=(0, 2))
    axial_counts = np.sum(binary_mask, axis=(0, 1))

    # Keep non-zero slice counts for compact representation
    sagittal_dict = {
        str(i): int(cnt) for i, cnt in enumerate(sagittal_counts) if cnt > 0
    }
    coronal_dict = {
        str(i): int(cnt) for i, cnt in enumerate(coronal_counts) if cnt > 0
    }
    axial_dict = {
        str(i): int(cnt) for i, cnt in enumerate(axial_counts) if cnt > 0
    }

    max_sagittal = int(np.argmax(sagittal_counts)) if np.any(sagittal_counts) else -1
    max_coronal = int(np.argmax(coronal_counts)) if np.any(coronal_counts) else -1
    max_axial = int(np.argmax(axial_counts)) if np.any(axial_counts) else -1

    return SliceOccupancySummary(
        axial_slice_counts=axial_dict,
        coronal_slice_counts=coronal_dict,
        sagittal_slice_counts=sagittal_dict,
        max_slice_indices={
            "axial": max_axial,
            "coronal": max_coronal,
            "sagittal": max_sagittal,
        },
    )


def extract_spatial_features(
    mri_volume: np.ndarray,
    segmentation_mask: np.ndarray,
    voxel_spacing_mm: Optional[Union[Tuple[float, float, float], List[float]]] = None,
    affine: Optional[np.ndarray] = None,
) -> SpatialMetrics:
    """Extract descriptive, non-diagnostic quantitative spatial features.

    Computes strictly structural and geometric measurements from the provided
    MRI volume array and segmentation mask.

    Args:
        mri_volume: 3D NumPy array of MRI voxel intensities.
        segmentation_mask: 3D NumPy array containing integer segmentation labels.
        voxel_spacing_mm: Optional 3-tuple/list of voxel spacing (dx, dy, dz) in mm.
        affine: Optional 4x4 spatial affine matrix for world coordinate mapping.

    Returns:
        SpatialMetrics: Strongly-typed structured spatial features container.

    Raises:
        ValueError: If volumes are not 3D or dimensions between MRI and mask mismatch.
    """
    validate_mri_volume(mri_volume)
    validate_mri_volume(segmentation_mask)

    if mri_volume.shape != segmentation_mask.shape:
        raise ValueError(
            f"Shape mismatch: MRI volume shape {mri_volume.shape} "
            f"does not match mask shape {segmentation_mask.shape}."
        )

    shape = (int(mri_volume.shape[0]), int(mri_volume.shape[1]), int(mri_volume.shape[2]))
    total_voxels = int(np.prod(shape))

    # Basic MRI intensity summary
    finite_vals = mri_volume[np.isfinite(mri_volume)]
    intensity_stats = {
        "min": float(np.min(finite_vals)),
        "max": float(np.max(finite_vals)),
        "mean": float(np.mean(finite_vals)),
        "std": float(np.std(finite_vals)),
    }

    # Binary nonzero mask
    binary_mask = segmentation_mask > 0
    mask_voxel_count = int(np.count_nonzero(binary_mask))
    has_nonzero = mask_voxel_count > 0

    # Voxel spacing and volumetric calculation
    spacing_tuple: Optional[Tuple[float, float, float]] = None
    mask_volume_mm3: Optional[float] = None
    voxel_volume_mm3: Optional[float] = None

    if voxel_spacing_mm is not None and len(voxel_spacing_mm) >= 3:
        sx, sy, sz = float(voxel_spacing_mm[0]), float(voxel_spacing_mm[1]), float(voxel_spacing_mm[2])
        spacing_tuple = (sx, sy, sz)
        voxel_volume_mm3 = sx * sy * sz
        mask_volume_mm3 = float(mask_voxel_count * voxel_volume_mm3)

    vol_fraction = (mask_voxel_count / total_voxels) * 100.0 if total_voxels > 0 else 0.0

    # Default geometric features
    bounding_box: Optional[BoundingBox] = None
    extent: Optional[AxisExtent] = None
    centroid_voxel: Optional[Tuple[float, float, float]] = None
    centroid_world: Optional[Tuple[float, float, float]] = None
    components_summary: Optional[ConnectedComponentsSummary] = None
    slice_occupancy_summary: Optional[SliceOccupancySummary] = None

    if has_nonzero:
        # Bounding box coordinates
        nonzero_coords = np.argwhere(binary_mask)
        min_coords = nonzero_coords.min(axis=0)
        max_coords = nonzero_coords.max(axis=0)

        min_x, min_y, min_z = int(min_coords[0]), int(min_coords[1]), int(min_coords[2])
        max_x, max_y, max_z = int(max_coords[0]), int(max_coords[1]), int(max_coords[2])

        bounding_box = BoundingBox(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            min_z=min_z,
            max_z=max_z,
        )

        # Axis extents (number of voxels spanning each axis)
        dx_vox = int(max_x - min_x + 1)
        dy_vox = int(max_y - min_y + 1)
        dz_vox = int(max_z - min_z + 1)

        dx_mm = float(dx_vox * spacing_tuple[0]) if spacing_tuple else None
        dy_mm = float(dy_vox * spacing_tuple[1]) if spacing_tuple else None
        dz_mm = float(dz_vox * spacing_tuple[2]) if spacing_tuple else None

        extent = AxisExtent(
            extent_x_voxels=dx_vox,
            extent_y_voxels=dy_vox,
            extent_z_voxels=dz_vox,
            extent_x_mm=dx_mm,
            extent_y_mm=dy_mm,
            extent_z_mm=dz_mm,
        )

        # Centroid
        centroid_voxel = compute_centroid(binary_mask)
        if centroid_voxel is not None and affine is not None:
            # Transform voxel index to world coordinate using affine matrix
            aff = np.asarray(affine, dtype=np.float64)
            homog = np.array([centroid_voxel[0], centroid_voxel[1], centroid_voxel[2], 1.0])
            world_coord = aff @ homog
            centroid_world = (float(world_coord[0]), float(world_coord[1]), float(world_coord[2]))
        elif centroid_voxel is not None and spacing_tuple is not None:
            centroid_world = (
                float(centroid_voxel[0] * spacing_tuple[0]),
                float(centroid_voxel[1] * spacing_tuple[1]),
                float(centroid_voxel[2] * spacing_tuple[2]),
            )

        # Connected components
        components_summary = compute_connected_components(binary_mask)

        # Slice occupancy
        slice_occupancy_summary = compute_slice_occupancy(binary_mask)

    # Per-label subregion analysis for multi-class masks
    unique_labels = [int(lbl) for lbl in np.unique(segmentation_mask) if lbl > 0]
    per_label_dict: Dict[str, Dict[str, Any]] = {}

    for lbl in unique_labels:
        lbl_mask = segmentation_mask == lbl
        lbl_voxels = int(np.count_nonzero(lbl_mask))
        lbl_coords = np.argwhere(lbl_mask)
        lbl_centroid = compute_centroid(lbl_mask)

        lbl_vol_mm3 = (
            float(lbl_voxels * voxel_volume_mm3)
            if voxel_volume_mm3 is not None
            else None
        )

        lbl_data: Dict[str, Any] = {
            "label_id": lbl,
            "voxel_count": lbl_voxels,
            "volume_mm3": lbl_vol_mm3,
            "centroid_voxel": list(lbl_centroid) if lbl_centroid else None,
        }

        if lbl_coords.size > 0:
            l_min = lbl_coords.min(axis=0)
            l_max = lbl_coords.max(axis=0)
            lbl_data["bounding_box"] = {
                "min_x": int(l_min[0]),
                "max_x": int(l_max[0]),
                "min_y": int(l_min[1]),
                "max_y": int(l_max[1]),
                "min_z": int(l_min[2]),
                "max_z": int(l_max[2]),
            }

        per_label_dict[f"label_{lbl}"] = lbl_data

    return SpatialMetrics(
        measurement_type=MEASUREMENT_LABEL,
        disclaimer=RESEARCH_DISCLAIMER,
        mri_shape=shape,
        total_mri_voxels=total_voxels,
        voxel_spacing_mm=spacing_tuple,
        mri_intensity_stats=intensity_stats,
        has_nonzero_mask=has_nonzero,
        total_mask_voxels=mask_voxel_count,
        mask_volume_mm3=mask_volume_mm3,
        volume_fraction_percent=float(vol_fraction),
        bounding_box=bounding_box,
        extent=extent,
        centroid_voxel=centroid_voxel,
        centroid_world_mm=centroid_world,
        connected_components=components_summary,
        slice_occupancy=slice_occupancy_summary,
        per_label_metrics=per_label_dict,
    )


def save_spatial_features(
    metrics: SpatialMetrics,
    output_dir: Optional[Union[str, Path]] = None,
    filename: Optional[str] = None,
) -> Path:
    """Save extracted spatial metrics to a JSON file in outputs/reports/.

    Args:
        metrics: SpatialMetrics container.
        output_dir: Destination directory. Defaults to 'output_reports' from config.
        filename: Optional output filename. Defaults to timestamped JSON file.

    Returns:
        Path: Resolved absolute path to the saved JSON report.
    """
    cfg = load_config()
    if output_dir is None:
        target_dir = cfg.paths.resolve(get_project_root())["output_reports"]
    else:
        target_dir = Path(output_dir).resolve()

    ensure_directory(target_dir)

    fname = filename or timestamped_filename("spatial_metrics", "json")
    if not fname.endswith(".json"):
        fname = f"{fname}.json"

    file_path = target_dir / safe_filename(fname)
    file_path.write_text(metrics.to_json(indent=2), encoding="utf-8")

    return file_path


def compute_dice_coefficient(
    predicted_mask: np.ndarray,
    reference_mask: np.ndarray,
) -> float:
    """Compute the Dice similarity coefficient between predicted and reference masks.

    Formula: Dice = (2 * |P & R|) / (|P| + |R|)
    Returns 1.0 if both masks are empty, and 0.0 if only one is empty.

    Args:
        predicted_mask: Boolean or integer array of predicted mask.
        reference_mask: Boolean or integer array of reference mask.

    Returns:
        float: Dice coefficient in [0.0, 1.0].
    """
    p_bool = predicted_mask > 0
    r_bool = reference_mask > 0

    p_sum = int(np.count_nonzero(p_bool))
    r_sum = int(np.count_nonzero(r_bool))

    if p_sum == 0 and r_sum == 0:
        return 1.0
    if p_sum == 0 or r_sum == 0:
        return 0.0

    intersection = int(np.count_nonzero(p_bool & r_bool))
    return float((2.0 * intersection) / (p_sum + r_sum))


def compute_overlap_metrics(
    predicted_mask: np.ndarray,
    reference_mask: np.ndarray,
    labels: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Compute descriptive overlap metrics comparing a predicted mask to a reference mask.

    NOTICE: This is a descriptive research measurement, not a diagnostic evaluation.

    Args:
        predicted_mask: 3D array containing predicted segmentation labels.
        reference_mask: 3D array containing ground truth reference labels.
        labels: Optional list of specific label indices to evaluate.

    Returns:
        Dict[str, Any]: Overlap metrics including Dice coefficient, Jaccard index,
            true positives, false positives, false negatives, and per-label breakdowns.
    """
    if predicted_mask.shape != reference_mask.shape:
        raise ValueError(
            f"Shape mismatch: predicted shape {predicted_mask.shape} != "
            f"reference shape {reference_mask.shape}."
        )

    # Global binary overlap
    p_bin = predicted_mask > 0
    r_bin = reference_mask > 0

    tp = int(np.count_nonzero(p_bin & r_bin))
    fp = int(np.count_nonzero(p_bin & (~r_bin)))
    fn = int(np.count_nonzero((~p_bin) & r_bin))
    p_count = int(np.count_nonzero(p_bin))
    r_count = int(np.count_nonzero(r_bin))

    dice_overall = (2.0 * tp / (p_count + r_count)) if (p_count + r_count) > 0 else (1.0 if p_count == r_count == 0 else 0.0)
    union = tp + fp + fn
    jaccard_overall = float(tp / union) if union > 0 else (1.0 if union == 0 else 0.0)

    # Per-label breakdown
    eval_labels = labels if labels is not None else sorted(list(set(np.unique(predicted_mask)).union(set(np.unique(reference_mask))) - {0}))
    per_label_metrics: Dict[str, Dict[str, Any]] = {}

    for lbl in eval_labels:
        p_lbl = predicted_mask == lbl
        r_lbl = reference_mask == lbl
        lbl_tp = int(np.count_nonzero(p_lbl & r_lbl))
        lbl_fp = int(np.count_nonzero(p_lbl & (~r_lbl)))
        lbl_fn = int(np.count_nonzero((~p_lbl) & r_lbl))
        lbl_p_sum = int(np.count_nonzero(p_lbl))
        lbl_r_sum = int(np.count_nonzero(r_lbl))

        lbl_dice = (2.0 * lbl_tp / (lbl_p_sum + lbl_r_sum)) if (lbl_p_sum + lbl_r_sum) > 0 else (1.0 if lbl_p_sum == lbl_r_sum == 0 else 0.0)
        lbl_union = lbl_tp + lbl_fp + lbl_fn
        lbl_jaccard = float(lbl_tp / lbl_union) if lbl_union > 0 else (1.0 if lbl_union == 0 else 0.0)

        per_label_metrics[f"label_{lbl}"] = {
            "label_id": int(lbl),
            "dice_coefficient": round(float(lbl_dice), 4),
            "jaccard_index": round(float(lbl_jaccard), 4),
            "true_positive_voxels": lbl_tp,
            "false_positive_voxels": lbl_fp,
            "false_negative_voxels": lbl_fn,
            "predicted_voxels": lbl_p_sum,
            "reference_voxels": lbl_r_sum,
        }

    return {
        "measurement_type": MEASUREMENT_LABEL,
        "disclaimer": "This is a research measurement, not a diagnostic evaluation. " + RESEARCH_DISCLAIMER,
        "overall_dice_coefficient": round(float(dice_overall), 4),
        "overall_jaccard_index": round(float(jaccard_overall), 4),
        "true_positive_voxels": tp,
        "false_positive_voxels": fp,
        "false_negative_voxels": fn,
        "predicted_mask_voxels": p_count,
        "reference_mask_voxels": r_count,
        "per_label_metrics": per_label_metrics,
    }


def compare_with_reference_annotation(
    predicted_mask: Union[np.ndarray, Path, str],
    reference_mask: Optional[Union[np.ndarray, Path, str]] = None,
    case_id: Optional[str] = None,
    annotations_dir: Optional[Union[Path, str]] = None,
) -> Dict[str, Any]:
    """Compare a predicted mask against an optional reference annotation from data/annotations/.

    NOTICE: This is a research measurement, not a diagnostic evaluation.

    Args:
        predicted_mask: 3D NumPy array or file path to predicted NIfTI mask.
        reference_mask: Optional 3D NumPy array or path to reference mask.
        case_id: Case identifier (e.g. 'BRATS_001') to locate reference mask in annotations_dir.
        annotations_dir: Directory containing reference masks. Defaults to data/annotations.

    Returns:
        Dict[str, Any]: Descriptive overlap comparison metrics.
    """
    import nibabel as nib

    # 1. Resolve predicted mask array
    if isinstance(predicted_mask, (str, Path)):
        p_path = Path(predicted_mask).resolve()
        p_img = nib.load(str(p_path))
        pred_arr = np.asanyarray(p_img.dataobj).astype(np.int32)
        if case_id is None:
            case_id = p_path.name.split(".")[0]
    else:
        pred_arr = np.asarray(predicted_mask, dtype=np.int32)

    # 2. Resolve reference mask
    ref_arr: Optional[np.ndarray] = None
    if reference_mask is not None:
        if isinstance(reference_mask, (str, Path)):
            r_path = Path(reference_mask).resolve()
            r_img = nib.load(str(r_path))
            ref_arr = np.asanyarray(r_img.dataobj).astype(np.int32)
        else:
            ref_arr = np.asarray(reference_mask, dtype=np.int32)
    elif case_id is not None:
        cfg = load_config()
        ann_base = Path(annotations_dir) if annotations_dir else cfg.paths.resolve(get_project_root())["data_annotations"]
        cand_paths = [
            ann_base / "Task01_BrainTumour" / "labelsTr" / f"{case_id}.nii.gz",
            ann_base / "labelsTr" / f"{case_id}.nii.gz",
            ann_base / f"{case_id}.nii.gz",
        ]
        for cand in cand_paths:
            if cand.is_file():
                r_img = nib.load(str(cand))
                ref_arr = np.asanyarray(r_img.dataobj).astype(np.int32)
                break

    if ref_arr is None:
        return {
            "measurement_type": MEASUREMENT_LABEL,
            "disclaimer": "This is a research measurement, not a diagnostic evaluation. " + RESEARCH_DISCLAIMER,
            "reference_available": False,
            "message": f"No reference annotation found for case '{case_id}'.",
        }

    overlap_results = compute_overlap_metrics(pred_arr, ref_arr)
    overlap_results["reference_available"] = True
    overlap_results["case_id"] = case_id
    return overlap_results

