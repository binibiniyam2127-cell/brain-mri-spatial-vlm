"""Helper utilities and business logic for the BrainMRI-SpatialVLM research platform.

Contains pure functions for:
- Upload file validation and secure saving
- Flattening spatial features into display tables
- 2D slice extraction and color overlay rendering for interactive sliders
- True 3D interactive mesh extraction and WebGL volumetric rendering via Plotly
- Formatting stage statuses and error summaries

Strict Non-Diagnostic Guardrails:
- All labels and tables include: "Research-only spatial measurements — not clinical findings."
- Never fabricates results or provides medical advice.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import plotly.graph_objects as go
from skimage.measure import marching_cubes

from src.mri_io import validate_mri_volume
from src.utils import ensure_directory, get_project_root, is_nifti_file, safe_filename, timestamped_filename
from src.visualization import _extract_2d_slice, get_mask_colormap

RESEARCH_DISCLAIMER = "AI-generated research draft — not a medical diagnosis."
MEASUREMENT_LABEL = "Research-only spatial measurements — not clinical findings."


def save_uploaded_nifti(
    uploaded_bytes: bytes,
    original_filename: str,
    target_dir: Union[str, Path],
) -> Path:
    """Validate and securely write an uploaded NIfTI file to the target directory.

    Args:
        uploaded_bytes: Byte payload of the uploaded file.
        original_filename: Name of the uploaded file.
        target_dir: Directory where file will be persisted.

    Returns:
        Path: Absolute path to the saved file.

    Raises:
        ValueError: If file is not .nii or .nii.gz or empty.
    """
    clean_name = safe_filename(original_filename)
    if not clean_name.endswith(".nii") and not clean_name.endswith(".nii.gz"):
        raise ValueError(
            f"Invalid file extension: '{original_filename}'. Only .nii and .nii.gz are supported."
        )

    if len(uploaded_bytes) == 0:
        raise ValueError("Uploaded file is empty (0 bytes).")

    dest_dir = Path(target_dir).resolve()
    ensure_directory(dest_dir)

    target_path = dest_dir / clean_name
    # If file exists, create a timestamped copy to avoid silently overwriting raw scans
    if target_path.exists():
        stem = clean_name.replace(".nii.gz", "").replace(".nii", "")
        ext = ".nii.gz" if clean_name.endswith(".nii.gz") else ".nii"
        target_path = dest_dir / safe_filename(timestamped_filename(stem, ext.lstrip(".")))

    target_path.write_bytes(uploaded_bytes)
    return target_path


def format_spatial_features_table(
    spatial_dict: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Transform nested spatial features dictionary into a flattened display table.

    Args:
        spatial_dict: Dictionary returned from extract_spatial_features().to_dict().

    Returns:
        List[Dict[str, Any]]: Tabular rows for table/dataframe display.
    """
    if not spatial_dict:
        return []

    rows: List[Dict[str, Any]] = []

    # 1. Total lesion metrics
    vox_count = spatial_dict.get("total_mask_voxels", 0)
    vol_mm3 = spatial_dict.get("mask_volume_mm3")
    vol_str = f"{vol_mm3:,.1f}" if vol_mm3 is not None else "N/A"
    rows.append({
        "Measurement": "Total Lesion Mask Voxels",
        "Value": f"{vox_count:,}",
        "Unit / Note": "voxels",
    })
    rows.append({
        "Measurement": "Estimated Physical Volume",
        "Value": vol_str,
        "Unit / Note": "mm³ (Research measurement)",
    })

    # 2. Volume fraction
    vol_frac = spatial_dict.get("volume_fraction_percent")
    if vol_frac is not None:
        rows.append({
            "Measurement": "Brain Volume Fraction",
            "Value": f"{vol_frac:.3f}",
            "Unit / Note": "%",
        })

    # 3. Bounding box & extents
    bbox = spatial_dict.get("bounding_box")
    if bbox:
        bbox_str = f"X:[{bbox.get('min_x')}, {bbox.get('max_x')}], Y:[{bbox.get('min_y')}, {bbox.get('max_y')}], Z:[{bbox.get('min_z')}, {bbox.get('max_z')}]"
        rows.append({
            "Measurement": "3D Bounding Box",
            "Value": bbox_str,
            "Unit / Note": "voxel indices [min, max]",
        })

    extent = spatial_dict.get("extent")
    if extent:
        ext_vox = f"{extent.get('extent_x_voxels')} x {extent.get('extent_y_voxels')} x {extent.get('extent_z_voxels')}"
        rows.append({
            "Measurement": "Axis Extent (Voxels)",
            "Value": ext_vox,
            "Unit / Note": "dx x dy x dz (voxels)",
        })
        if extent.get("extent_x_mm") is not None:
            ext_mm = f"{extent.get('extent_x_mm'):.1f} x {extent.get('extent_y_mm'):.1f} x {extent.get('extent_z_mm'):.1f}"
            rows.append({
                "Measurement": "Axis Extent (Physical)",
                "Value": ext_mm,
                "Unit / Note": "mm (dx x dy x dz)",
            })

    # 4. Centroid
    cent_vox = spatial_dict.get("centroid_voxel")
    if cent_vox:
        rows.append({
            "Measurement": "Centroid (Voxel Coordinates)",
            "Value": f"({cent_vox[0]:.1f}, {cent_vox[1]:.1f}, {cent_vox[2]:.1f})",
            "Unit / Note": "index coordinates",
        })

    cent_world = spatial_dict.get("centroid_world_mm")
    if cent_world:
        rows.append({
            "Measurement": "Centroid (World Coordinates)",
            "Value": f"({cent_world[0]:.1f}, {cent_world[1]:.1f}, {cent_world[2]:.1f})",
            "Unit / Note": "RAS / scanner mm",
        })

    # 5. Connected components
    conn = spatial_dict.get("connected_components")
    if conn:
        rows.append({
            "Measurement": "Connected Components",
            "Value": str(conn.get("count", 0)),
            "Unit / Note": "discrete spatial clusters",
        })

    return rows


def rows_to_markdown_table(
    rows: List[Dict[str, Any]],
    headers: Optional[List[str]] = None,
) -> str:
    """Convert list of row dictionaries to a GitHub Flavored Markdown table string.

    Provides zero-dependency table rendering without importing pandas or C-extension DLLs.

    Args:
        rows: List of dictionary records representing table rows.
        headers: Optional list of column header names.

    Returns:
        str: Formatted Markdown table string.
    """
    if not rows:
        return ""

    if headers is None:
        headers = list(rows[0].keys())

    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join([":---"] * len(headers)) + " |"

    data_lines = []
    for r in rows:
        cells = []
        for h in headers:
            val = str(r.get(h, ""))
            # Sanitize markdown table characters
            val = val.replace("|", r"\|").replace("\n", " ").strip()
            cells.append(val)
        data_lines.append("| " + " | ".join(cells) + " |")

    return "\n".join([header_line, separator_line] + data_lines)



def render_slice_rgba(
    mri_volume: np.ndarray,
    mask: Optional[np.ndarray],
    plane: str,
    slice_idx: int,
    show_mask: bool = True,
    alpha: float = 0.55,
    show_bbox: bool = True,
) -> Image.Image:
    """Render a 2D MRI slice with optional multi-label segmentation overlay as a PIL Image.

    Args:
        mri_volume: 3D NumPy array of voxel intensities.
        mask: Optional 3D NumPy array of integer segmentation labels.
        plane: 'axial', 'coronal', or 'sagittal'.
        slice_idx: Index of slice along the plane's axis.
        show_mask: Whether to composite the segmentation mask.
        alpha: Opacity for mask overlay blending (0.0 to 1.0).
        show_bbox: Whether to draw the 2D lesion bounding box rectangle.

    Returns:
        Image.Image: RGB PIL Image ready for UI display.
    """
    mri_slice = _extract_2d_slice(mri_volume, plane, slice_idx).astype(np.float32)

    # Contrast normalization (1st to 99th percentile)
    vmin, vmax = np.percentile(mri_slice, [1, 99])
    if vmax > vmin:
        norm_slice = np.clip((mri_slice - vmin) / (vmax - vmin), 0.0, 1.0)
    else:
        norm_slice = np.zeros_like(mri_slice)

    gray_uint8 = (norm_slice * 255.0).astype(np.uint8)
    base_rgb = np.stack([gray_uint8, gray_uint8, gray_uint8], axis=-1)

    if show_mask and mask is not None:
        mask_slice = _extract_2d_slice(mask, plane, slice_idx)
        if np.any(mask_slice > 0):
            palette = {
                1: np.array([230, 25, 25], dtype=np.float32),
                2: np.array([50, 205, 50], dtype=np.float32),
                3: np.array([255, 165, 0], dtype=np.float32),
                4: np.array([0, 190, 255], dtype=np.float32),
            }

            overlay = base_rgb.astype(np.float32)
            for lbl, color in palette.items():
                lbl_mask = (mask_slice == lbl)
                if np.any(lbl_mask):
                    overlay[lbl_mask] = (
                        (1.0 - alpha) * overlay[lbl_mask] + alpha * color
                    )

            extra_mask = (mask_slice > 4)
            if np.any(extra_mask):
                overlay[extra_mask] = (
                    (1.0 - alpha) * overlay[extra_mask] + alpha * np.array([180, 50, 220])
                )

            base_rgb = np.clip(overlay, 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(base_rgb)

    # Draw 2D Bounding Box rectangle if requested
    if show_bbox and mask is not None and np.any(mask > 0):
        try:
            pos = np.argwhere(mask > 0)
            x0, y0, z0 = pos.min(axis=0)
            x1, y1, z1 = pos.max(axis=0)

            # Check if current slice intersects the 3D bounding box range
            intersects = (
                (plane == "axial" and z0 <= slice_idx <= z1)
                or (plane == "coronal" and y0 <= slice_idx <= y1)
                or (plane == "sagittal" and x0 <= slice_idx <= x1)
            )

            if intersects:
                box_vol = np.zeros_like(mask, dtype=np.uint8)
                box_vol[x0 : x1 + 1, y0 : y1 + 1, z0 : z1 + 1] = 1
                box_sl = _extract_2d_slice(box_vol, plane, slice_idx)
                if np.any(box_sl > 0):
                    b_pos = np.argwhere(box_sl > 0)
                    r_min, c_min = b_pos.min(axis=0)
                    r_max, c_max = b_pos.max(axis=0)
                    from PIL import ImageDraw

                    draw = ImageDraw.Draw(pil_img)
                    draw.rectangle(
                        [int(c_min), int(r_min), int(c_max), int(r_max)],
                        outline="#00e5ff",
                        width=2,
                    )
        except Exception:
            pass

    return pil_img


def build_3d_volumetric_figure(
    mri_volume: np.ndarray,
    mask: Optional[np.ndarray] = None,
    show_brain_envelope: bool = True,
    brain_opacity: float = 0.12,
    show_necrotic: bool = True,
    show_edema: bool = True,
    show_enhancing: bool = True,
    show_bounding_box: bool = True,
    show_centroid: bool = True,
    bbox_target: str = "Active Visible Subregions",
    step_size: int = 2,
) -> go.Figure:
    """Extract 3D polygonal surface meshes and build interactive Plotly WebGL figure.

    Allows mouse-based 3D rotation (orbiting), zooming, panning, and toggling
    of individual 3D anatomical contours, tumor subregions, 3D bounding box, and centroid.

    Args:
        mri_volume: 3D NumPy array of MRI intensities.
        mask: Optional 3D NumPy array of segmentation labels (0=bg, 1=core, 2=edema, 3=enhancing).
        show_brain_envelope: Whether to render the semi-transparent 3D brain outer shell.
        brain_opacity: Opacity of the 3D brain shell (0.0 to 1.0).
        show_necrotic: Whether to display Label 1 (Necrotic core).
        show_edema: Whether to display Label 2 (Edema).
        show_enhancing: Whether to display Label 3 (Enhancing tumor).
        show_bounding_box: Whether to render 3D bounding box wireframe.
        show_centroid: Whether to render 3D centroid marker.
        bbox_target: Target region for bounding box calculation.
        step_size: Marching cubes subsampling step for high responsiveness.

    Returns:
        go.Figure: Interactive 3D Plotly WebGL figure.
    """
    traces = []

    # 1. 3D Brain Contour (Semi-transparent mesh envelope)
    if show_brain_envelope and mri_volume is not None:
        try:
            pos_vals = mri_volume[mri_volume > 0]
            if len(pos_vals) > 0:
                thresh = float(np.percentile(pos_vals, 25))
                brain_bin = mri_volume > thresh
                # Use slightly larger step size for brain shell to maintain snappy 60fps rendering
                b_step = max(3, step_size)
                verts_b, faces_b, _, _ = marching_cubes(brain_bin, level=0.5, step_size=b_step)

                brain_trace = go.Mesh3d(
                    x=verts_b[:, 0],
                    y=verts_b[:, 1],
                    z=verts_b[:, 2],
                    i=faces_b[:, 0],
                    j=faces_b[:, 1],
                    k=faces_b[:, 2],
                    color="#b0bec5",
                    opacity=float(brain_opacity),
                    name="Brain Contour (3D)",
                    hoverinfo="skip",
                    lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.5),
                )
                traces.append(brain_trace)
        except Exception:
            pass

    # 2. 3D Tumor Subregions (Solid & Semi-transparent Meshes)
    if mask is not None and np.any(mask > 0):
        # Configuration for each BraTS subregion
        label_configs = [
            (1, "Necrotic Core (Label 1)", "#e53935", 0.90, show_necrotic),
            (2, "Peritumoral Edema (Label 2)", "#43a047", 0.45, show_edema),
            (3, "Enhancing Tumor (Label 3)", "#fb8c00", 0.85, show_enhancing),
        ]

        has_subregions = False
        for lbl_id, name, color, opac, is_visible in label_configs:
            if not is_visible:
                continue
            lbl_bin = (mask == lbl_id)
            if np.any(lbl_bin):
                has_subregions = True
                try:
                    verts_t, faces_t, _, _ = marching_cubes(lbl_bin, level=0.5, step_size=step_size)
                    t_trace = go.Mesh3d(
                        x=verts_t[:, 0],
                        y=verts_t[:, 1],
                        z=verts_t[:, 2],
                        i=faces_t[:, 0],
                        j=faces_t[:, 1],
                        k=faces_t[:, 2],
                        color=color,
                        opacity=opac,
                        name=name,
                        lighting=dict(ambient=0.4, diffuse=0.9, specular=0.5, roughness=0.3),
                    )
                    traces.append(t_trace)
                except Exception:
                    pass

        # Fallback if binary mask without subregion labels 1, 2, 3
        if not has_subregions and np.any(mask > 0):
            try:
                verts_t, faces_t, _, _ = marching_cubes(mask > 0, level=0.5, step_size=step_size)
                t_trace = go.Mesh3d(
                    x=verts_t[:, 0],
                    y=verts_t[:, 1],
                    z=verts_t[:, 2],
                    i=faces_t[:, 0],
                    j=faces_t[:, 1],
                    k=faces_t[:, 2],
                    color="#e53935",
                    opacity=0.85,
                    name="Segmented Lesion (3D)",
                )
                traces.append(t_trace)
            except Exception:
                pass

        # 3. 3D Bounding Box Wireframe (Cyan)
        if show_bounding_box:
            try:
                if bbox_target == "Enhancing Tumor (ET)":
                    target_mask = (mask == 3)
                    target_name = "Enhancing Tumor"
                elif bbox_target == "Tumor Core (TC: Necrotic + Enhancing)":
                    target_mask = (mask == 1) | (mask == 3)
                    target_name = "Tumor Core"
                elif bbox_target == "Whole Lesion (WT: Core + Edema)":
                    target_mask = (mask > 0)
                    target_name = "Whole Lesion"
                else:  # "Active Visible Subregions"
                    active_mask = np.zeros_like(mask, dtype=bool)
                    active_labels = []
                    if show_necrotic:
                        active_mask |= (mask == 1)
                        active_labels.append("Necrotic")
                    if show_edema:
                        active_mask |= (mask == 2)
                        active_labels.append("Edema")
                    if show_enhancing:
                        active_mask |= (mask == 3)
                        active_labels.append("Enhancing")

                    if len(active_labels) > 0 and np.any(active_mask):
                        target_mask = active_mask
                        target_name = "+".join(active_labels)
                    else:
                        target_mask = (mask > 0)
                        target_name = "Lesion"

                pos = np.argwhere(target_mask)
                if len(pos) > 0:
                    x0, y0, z0 = pos.min(axis=0)
                    x1, y1, z1 = pos.max(axis=0)

                    bx = [
                        x0, x1, x1, x0, x0, None,
                        x0, x1, x1, x0, x0, None,
                        x0, x0, None,
                        x1, x1, None,
                        x1, x1, None,
                        x0, x0,
                    ]
                    by = [
                        y0, y0, y1, y1, y0, None,
                        y0, y0, y1, y1, y0, None,
                        y0, y0, None,
                        y0, y0, None,
                        y1, y1, None,
                        y1, y1,
                    ]
                    bz = [
                        z0, z0, z0, z0, z0, None,
                        z1, z1, z1, z1, z1, None,
                        z0, z1, None,
                        z0, z1, None,
                        z0, z1, None,
                        z0, z1,
                    ]

                    bbox_trace = go.Scatter3d(
                        x=bx,
                        y=by,
                        z=bz,
                        mode="lines",
                        line=dict(color="#00e5ff", width=5),
                        name=f"3D Bounding Box ({target_name})",
                        hoverinfo="text",
                        hovertext=f"3D Bounding Box ({target_name}): X[{x0}:{x1}] Y[{y0}:{y1}] Z[{z0}:{z1}]",
                    )
                    traces.append(bbox_trace)

                    # 4. 3D Centroid (Yellow Diamond Marker)
                    if show_centroid:
                        cx, cy, cz = pos.mean(axis=0)
                        cent_trace = go.Scatter3d(
                            x=[cx],
                            y=[cy],
                            z=[cz],
                            mode="markers",
                            marker=dict(
                                size=7,
                                color="#ffeb3b",
                                symbol="diamond",
                                line=dict(color="#000000", width=1),
                            ),
                            name=f"3D Centroid ({target_name})",
                            hoverinfo="text",
                            hovertext=f"3D Centroid ({target_name}): ({cx:.1f}, {cy:.1f}, {cz:.1f})",
                        )
                        traces.append(cent_trace)
            except Exception:
                pass

    # Build Layout with dark background matching medical theme
    fig = go.Figure(data=traces)

    fig.update_layout(
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        scene=dict(
            aspectmode="data",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="#0e1117",
            camera=dict(
                eye=dict(x=1.6, y=1.6, z=1.2),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        legend=dict(
            font=dict(color="#e0e0e0", size=11),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(20, 20, 20, 0.7)",
        ),
        margin=dict(l=0, r=0, t=10, b=10),
    )

    return fig
