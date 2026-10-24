"""MRI and Segmentation Mask Visualization Module.

Generates research-only, non-diagnostic 2D and 3D visual representations:
- Tri-planar orthogonal views (axial, coronal, sagittal) with segmentation overlays
- Multi-slice axial/coronal/sagittal montages
- Slice-wise mask occupancy profiles
- Informative slice selection (peak lesion area and volume center)

All visualizations strictly embed visible non-diagnostic disclaimers:
"AI-generated research visualization — not a medical diagnosis."
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.config import load_config
from src.mri_io import get_volume_metadata, load_nifti_volume, validate_mri_volume
from src.utils import ensure_directory, get_project_root, is_nifti_file, safe_filename, timestamped_filename

VISUALIZATION_WATERMARK = "AI-generated research visualization — not a medical diagnosis."


def get_mask_colormap(alpha: float = 0.55) -> Tuple[ListedColormap, BoundaryNorm]:
    """Build discrete colormap for multi-region or binary segmentation masks.

    Supports:
    0: Background (completely transparent)
    1: Label 1 / Necrotic tumor core (Red)
    2: Label 2 / Peritumoral edema (Green)
    3: Label 3 / GD-enhancing tumor (Orange/Yellow)
    4: Label 4 / Extra region (Cyan/Blue)
    """
    colors = [
        (0.0, 0.0, 0.0, 0.0),       # 0: Transparent background
        (0.89, 0.10, 0.11, alpha),   # 1: Red
        (0.20, 0.63, 0.17, alpha),   # 2: Green
        (1.00, 0.55, 0.00, alpha),   # 3: Orange/Yellow
        (0.12, 0.47, 0.71, alpha),   # 4: Blue/Cyan
    ]
    cmap = ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def add_figure_watermark(
    fig: plt.Figure,
    text: str = VISUALIZATION_WATERMARK,
    y_pos: float = 0.015,
    font_size: int = 9,
) -> None:
    """Embed a visible non-diagnostic research disclaimer banner onto a matplotlib figure."""
    fig.text(
        0.5,
        y_pos,
        text,
        ha="center",
        va="bottom",
        fontsize=font_size,
        fontweight="bold",
        color="crimson",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#fff0f0",
            edgecolor="#ff8888",
            linewidth=1.2,
            alpha=0.95,
        ),
    )


def burn_pillow_watermark(
    image_path: Union[str, Path],
    text: str = VISUALIZATION_WATERMARK,
) -> Path:
    """Use Pillow to ensure a high-contrast watermark banner is burned into the saved PNG."""
    img_path = Path(image_path).resolve()
    if not img_path.is_file():
        raise FileNotFoundError(f"Image not found: {img_path}")

    with Image.open(img_path) as base_img:
        img = base_img.convert("RGBA")
        width, height = img.size

        # Create banner overlay
        banner_height = max(24, int(height * 0.045))
        banner = Image.new("RGBA", (width, banner_height), (255, 240, 240, 240))
        draw = ImageDraw.Draw(banner)

        # Attempt to load font or fallback to default
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        text_bbox = draw.textbbox((0, 0), text, font=font) if hasattr(draw, "textbbox") else (0, 0, 300, 15)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        text_x = max(10, (width - text_w) // 2)
        text_y = max(2, (banner_height - text_h) // 2)

        # Draw red border top line
        draw.line([(0, 0), (width, 0)], fill=(220, 40, 40, 255), width=2)
        draw.text((text_x, text_y), text, fill=(180, 20, 20, 255), font=font)

        # Composite banner at bottom of image
        img.paste(banner, (0, height - banner_height), banner)
        img.convert("RGB").save(img_path, format="PNG")

    return img_path


def load_volume_for_visualization(
    input_volume: Union[str, Path, np.ndarray],
    channel_index: int = 3,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load or extract a 3D MRI volume for visualization.

    If given a 4D multimodal volume (e.g. MSD BraTS [T1, T1c, T2, FLAIR]), extracts
    the sequence at channel_index (default: 3, corresponding to FLAIR in BraTS).

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]: (volume_3d, metadata)
    """
    metadata: Dict[str, Any] = {}
    if isinstance(input_volume, (str, Path)):
        p = Path(input_volume).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Input file not found: {p}")
        if not is_nifti_file(p):
            raise ValueError(f"Input file is not a valid NIfTI: {p}")
        try:
            vol, _, meta = load_nifti_volume(p)
            metadata = meta
        except ValueError:
            nii = nib.load(str(p))
            vol = np.asanyarray(nii.dataobj).astype(np.float32)
            try:
                metadata = get_volume_metadata(p)
            except Exception:
                metadata = {"shape": list(vol.shape)}
    elif isinstance(input_volume, np.ndarray):
        vol = input_volume
        metadata = {"shape": list(vol.shape)}
    else:
        raise TypeError(f"Unsupported volume type: {type(input_volume)}")

    if vol.ndim == 4 and vol.shape[-1] == 1:
        vol = np.squeeze(vol, axis=-1)

    if vol.ndim == 4:
        if vol.shape[-1] > channel_index:
            vol_3d = vol[:, :, :, channel_index].astype(np.float32)
            metadata["extracted_channel"] = channel_index
        elif vol.shape[0] > channel_index:
            vol_3d = vol[channel_index, :, :, :].astype(np.float32)
            metadata["extracted_channel"] = channel_index
        else:
            vol_3d = vol[:, :, :, 0].astype(np.float32)
            metadata["extracted_channel"] = 0
    elif vol.ndim == 3:
        vol_3d = vol.astype(np.float32)
    else:
        raise ValueError(f"Expected 3D or 4D volume array, got shape {vol.shape}")

    validate_mri_volume(vol_3d)
    return vol_3d, metadata


def load_mask_for_visualization(
    input_mask: Optional[Union[str, Path, np.ndarray]],
) -> Optional[np.ndarray]:
    """Load or prepare an optional 3D segmentation mask array."""
    if input_mask is None:
        return None

    if isinstance(input_mask, (str, Path)):
        p = Path(input_mask).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Mask file not found: {p}")
        if not is_nifti_file(p):
            raise ValueError(f"Mask file is not a valid NIfTI: {p}")
        try:
            mask_arr, _, _ = load_nifti_volume(p)
        except ValueError:
            nii = nib.load(str(p))
            mask_arr = np.asanyarray(nii.dataobj)
    elif isinstance(input_mask, np.ndarray):
        mask_arr = input_mask
    else:
        raise TypeError(f"Unsupported mask type: {type(input_mask)}")

    if mask_arr.ndim == 4 and mask_arr.shape[-1] == 1:
        mask_arr = np.squeeze(mask_arr, axis=-1)

    if mask_arr.ndim != 3:
        raise ValueError(f"Mask must be 3D, got shape {mask_arr.shape}")

    return mask_arr.astype(np.int32)


def find_informative_slices(
    mri_volume: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, Any]]:
    """Determine informative slice indices across axial, coronal, and sagittal orientations.

    Finds:
    - Center anatomical slice
    - Slice with largest mask area (if mask contains non-zero voxels)
    - Active slice range containing mask voxels
    - Slice-wise voxel occupancy profile

    Returns:
        Dict[str, Dict[str, Any]]: Plane mappings ('axial', 'coronal', 'sagittal')
    """
    dims = mri_volume.shape  # (X, Y, Z)
    has_mask = mask is not None and np.any(mask > 0)

    results: Dict[str, Dict[str, Any]] = {}

    # 1. Axial Plane (along Z-axis, axis=2)
    z_len = dims[2]
    z_center = z_len // 2
    if has_mask:
        z_counts = np.sum(mask > 0, axis=(0, 1))
        z_max = int(np.argmax(z_counts)) if np.max(z_counts) > 0 else z_center
        z_active = np.where(z_counts > 0)[0]
        z_range = (int(z_active[0]), int(z_active[-1])) if len(z_active) > 0 else (0, z_len - 1)
    else:
        z_counts = np.zeros(z_len, dtype=int)
        z_max = z_center
        z_range = (0, z_len - 1)

    results["axial"] = {
        "axis": 2,
        "center_slice": z_center,
        "max_mask_slice": z_max,
        "active_range": z_range,
        "occupancy": z_counts,
    }

    # 2. Coronal Plane (along Y-axis, axis=1)
    y_len = dims[1]
    y_center = y_len // 2
    if has_mask:
        y_counts = np.sum(mask > 0, axis=(0, 2))
        y_max = int(np.argmax(y_counts)) if np.max(y_counts) > 0 else y_center
        y_active = np.where(y_counts > 0)[0]
        y_range = (int(y_active[0]), int(y_active[-1])) if len(y_active) > 0 else (0, y_len - 1)
    else:
        y_counts = np.zeros(y_len, dtype=int)
        y_max = y_center
        y_range = (0, y_len - 1)

    results["coronal"] = {
        "axis": 1,
        "center_slice": y_center,
        "max_mask_slice": y_max,
        "active_range": y_range,
        "occupancy": y_counts,
    }

    # 3. Sagittal Plane (along X-axis, axis=0)
    x_len = dims[0]
    x_center = x_len // 2
    if has_mask:
        x_counts = np.sum(mask > 0, axis=(1, 2))
        x_max = int(np.argmax(x_counts)) if np.max(x_counts) > 0 else x_center
        x_active = np.where(x_counts > 0)[0]
        x_range = (int(x_active[0]), int(x_active[-1])) if len(x_active) > 0 else (0, x_len - 1)
    else:
        x_counts = np.zeros(x_len, dtype=int)
        x_max = x_center
        x_range = (0, x_len - 1)

    results["sagittal"] = {
        "axis": 0,
        "center_slice": x_center,
        "max_mask_slice": x_max,
        "active_range": x_range,
        "occupancy": x_counts,
    }

    return results


def _extract_2d_slice(
    volume: np.ndarray,
    plane: str,
    slice_idx: int,
) -> np.ndarray:
    """Extract and rotate a 2D slice to standard anatomical display orientation."""
    if plane == "axial":
        # volume: (X, Y, Z) -> extract (X, Y) at Z=slice_idx
        sl = volume[:, :, slice_idx]
        return np.rot90(sl)
    elif plane == "coronal":
        # volume: (X, Y, Z) -> extract (X, Z) at Y=slice_idx
        sl = volume[:, slice_idx, :]
        return np.rot90(sl)
    elif plane == "sagittal":
        # volume: (X, Y, Z) -> extract (Y, Z) at X=slice_idx
        sl = volume[slice_idx, :, :]
        return np.rot90(sl)
    else:
        raise ValueError(f"Unknown plane: {plane}. Must be 'axial', 'coronal', or 'sagittal'.")


def plot_orthogonal_slices(
    mri_volume: np.ndarray,
    mask: Optional[np.ndarray] = None,
    axial_slice: Optional[int] = None,
    coronal_slice: Optional[int] = None,
    sagittal_slice: Optional[int] = None,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Brain MRI Orthogonal View (Tri-Planar)",
) -> Tuple[plt.Figure, Path]:
    """Plot Axial, Coronal, and Sagittal orthogonal views with mask overlays.

    Automatically selects peak-mask slices if specific indices are omitted.
    """
    info = find_informative_slices(mri_volume, mask)

    z_idx = axial_slice if axial_slice is not None else info["axial"]["max_mask_slice"]
    y_idx = coronal_slice if coronal_slice is not None else info["coronal"]["max_mask_slice"]
    x_idx = sagittal_slice if sagittal_slice is not None else info["sagittal"]["max_mask_slice"]

    cmap, norm = get_mask_colormap()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), facecolor="#1e1e1e")
    fig.suptitle(f"{title}\n", color="white", fontsize=13, fontweight="bold", y=0.98)

    planes = [
        ("axial", z_idx, f"Axial (Z = {z_idx})"),
        ("coronal", y_idx, f"Coronal (Y = {y_idx})"),
        ("sagittal", x_idx, f"Sagittal (X = {x_idx})"),
    ]

    for ax, (plane_name, s_idx, plane_title) in zip(axes, planes):
        ax.set_facecolor("black")
        mri_slice = _extract_2d_slice(mri_volume, plane_name, s_idx)
        # Normalize slice for display contrast
        vmin, vmax = np.percentile(mri_slice, [1, 99])
        ax.imshow(mri_slice, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")

        if mask is not None:
            mask_slice = _extract_2d_slice(mask, plane_name, s_idx)
            if np.any(mask_slice > 0):
                masked = np.ma.masked_where(mask_slice == 0, mask_slice)
                ax.imshow(masked, cmap=cmap, norm=norm, origin="lower", interpolation="nearest")

        ax.set_title(plane_title, color="#e0e0e0", fontsize=11, pad=6)
        ax.axis("off")

    add_figure_watermark(fig, VISUALIZATION_WATERMARK, y_pos=0.03, font_size=10)
    plt.tight_layout(rect=[0, 0.06, 1, 0.94])

    if output_path is None:
        cfg = load_config()
        out_dir = cfg.paths.resolve(get_project_root())["output_visualizations"]
        ensure_directory(out_dir)
        output_path = out_dir / safe_filename(timestamped_filename("orthogonal_view", "png"))
    else:
        output_path = Path(output_path).resolve()
        ensure_directory(output_path.parent)

    fig.savefig(str(output_path), dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    burn_pillow_watermark(output_path, VISUALIZATION_WATERMARK)
    return fig, output_path


def plot_slice_montage(
    mri_volume: np.ndarray,
    mask: Optional[np.ndarray] = None,
    plane: str = "axial",
    num_slices: int = 12,
    grid_cols: int = 4,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Brain MRI Multi-Slice Montage",
) -> Tuple[plt.Figure, Path]:
    """Plot an N-slice montage grid spanning informative slice ranges."""
    info = find_informative_slices(mri_volume, mask)
    plane_info = info[plane]
    active_range = plane_info["active_range"]
    total_len = mri_volume.shape[plane_info["axis"]]

    has_mask_voxels = mask is not None and np.any(mask > 0)
    if has_mask_voxels and (active_range[1] - active_range[0]) >= (num_slices // 2):
        # Sample across active lesion span with a small buffer
        margin = max(2, int((active_range[1] - active_range[0]) * 0.15))
        start = max(0, active_range[0] - margin)
        stop = min(total_len - 1, active_range[1] + margin)
        selected_slices = np.linspace(start, stop, num_slices, dtype=int)
    else:
        # Sample across brain parenchyma
        start = int(total_len * 0.15)
        stop = int(total_len * 0.85)
        selected_slices = np.linspace(start, stop, num_slices, dtype=int)

    # Remove duplicates if volume is small
    selected_slices = np.unique(selected_slices)
    num_plots = len(selected_slices)

    grid_cols = min(grid_cols, num_plots)
    grid_rows = int(np.ceil(num_plots / grid_cols))

    cmap, norm = get_mask_colormap()

    fig, axes = plt.subplots(
        grid_rows,
        grid_cols,
        figsize=(grid_cols * 3.2, grid_rows * 3.2 + 0.8),
        facecolor="#1e1e1e",
    )
    fig.suptitle(
        f"{title} ({plane.capitalize()} Plane)\n",
        color="white",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )

    flat_axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(flat_axes):
        ax.set_facecolor("black")
        if i < num_plots:
            s_idx = int(selected_slices[i])
            mri_slice = _extract_2d_slice(mri_volume, plane, s_idx)
            vmin, vmax = np.percentile(mri_slice, [1, 99])
            ax.imshow(mri_slice, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")

            mask_label = ""
            if mask is not None:
                mask_slice = _extract_2d_slice(mask, plane, s_idx)
                cnt = int(np.sum(mask_slice > 0))
                if cnt > 0:
                    mask_label = f" ({cnt} voxels)"
                    masked = np.ma.masked_where(mask_slice == 0, mask_slice)
                    ax.imshow(masked, cmap=cmap, norm=norm, origin="lower", interpolation="nearest")

            ax.set_title(f"Slice {s_idx}{mask_label}", color="#d0d0d0", fontsize=9, pad=4)
        ax.axis("off")

    add_figure_watermark(fig, VISUALIZATION_WATERMARK, y_pos=0.015, font_size=10)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    if output_path is None:
        cfg = load_config()
        out_dir = cfg.paths.resolve(get_project_root())["output_visualizations"]
        ensure_directory(out_dir)
        output_path = out_dir / safe_filename(timestamped_filename(f"montage_{plane}", "png"))
    else:
        output_path = Path(output_path).resolve()
        ensure_directory(output_path.parent)

    fig.savefig(str(output_path), dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    burn_pillow_watermark(output_path, VISUALIZATION_WATERMARK)
    return fig, output_path


def plot_mask_occupancy(
    mask: np.ndarray,
    voxel_spacing_mm: Optional[Tuple[float, float, float]] = None,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Slice-Wise Mask Occupancy Profile",
) -> Tuple[plt.Figure, Path]:
    """Plot slice-wise mask occupancy profiles across Axial, Coronal, and Sagittal axes."""
    info = find_informative_slices(mask, mask)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor="#fafafa")
    fig.suptitle(f"{title} (Research Only)\n", fontsize=12, fontweight="bold", y=0.98)

    curves = [
        ("axial", info["axial"]["occupancy"], "Axial Slice (Z)", "#d9381e"),
        ("coronal", info["coronal"]["occupancy"], "Coronal Slice (Y)", "#2a9d8f"),
        ("sagittal", info["sagittal"]["occupancy"], "Sagittal Slice (X)", "#264653"),
    ]

    for ax, (plane_name, counts, xlabel, color) in zip(axes, curves):
        indices = np.arange(len(counts))
        ax.plot(indices, counts, color=color, linewidth=2, label="Mask Voxels")
        ax.fill_between(indices, counts, color=color, alpha=0.25)

        peak_idx = int(np.argmax(counts)) if np.max(counts) > 0 else len(counts) // 2
        peak_val = int(counts[peak_idx])

        if peak_val > 0:
            ax.plot(peak_idx, peak_val, marker="o", markersize=6, color="darkred")
            ax.annotate(
                f"Peak: idx {peak_idx}\n({peak_val} voxels)",
                xy=(peak_idx, peak_val),
                xytext=(peak_idx + len(counts) * 0.05, peak_val * 0.9 + 1),
                arrowprops=dict(arrowstyle="->", color="darkred", lw=1),
                fontsize=8,
                fontweight="bold",
                color="darkred",
            )

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Mask Voxel Count", fontsize=10)
        ax.set_title(f"{plane_name.capitalize()} Occupancy Profile", fontsize=11, fontweight="semibold")
        ax.grid(True, linestyle="--", alpha=0.5)

    add_figure_watermark(fig, VISUALIZATION_WATERMARK, y_pos=0.02, font_size=9)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    if output_path is None:
        cfg = load_config()
        out_dir = cfg.paths.resolve(get_project_root())["output_visualizations"]
        ensure_directory(out_dir)
        output_path = out_dir / safe_filename(timestamped_filename("mask_occupancy_profile", "png"))
    else:
        output_path = Path(output_path).resolve()
        ensure_directory(output_path.parent)

    fig.savefig(str(output_path), dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    burn_pillow_watermark(output_path, VISUALIZATION_WATERMARK)
    return fig, output_path


def generate_all_visualizations(
    mri_volume: Union[str, Path, np.ndarray],
    mask: Optional[Union[str, Path, np.ndarray]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    prefix: str = "viz",
) -> Dict[str, Path]:
    """Generate and save complete suite of research visualizations.

    Generates:
    1. Tri-planar orthogonal view PNG (axial, coronal, sagittal)
    2. Multi-slice axial montage PNG
    3. Slice-wise mask occupancy profile PNG (if mask provided)

    Returns:
        Dict[str, Path]: Paths to all generated visualization images.
    """
    cfg = load_config()
    if output_dir is None:
        dest_dir = cfg.paths.resolve(get_project_root())["output_visualizations"]
    else:
        dest_dir = Path(output_dir).resolve()
    ensure_directory(dest_dir)

    vol_3d, _ = load_volume_for_visualization(mri_volume)
    mask_3d = load_mask_for_visualization(mask)

    generated: Dict[str, Path] = {}

    # 1. Orthogonal Tri-Planar View
    ortho_path = dest_dir / safe_filename(timestamped_filename(f"{prefix}_orthogonal", "png"))
    _, out_ortho = plot_orthogonal_slices(
        mri_volume=vol_3d,
        mask=mask_3d,
        output_path=ortho_path,
    )
    generated["orthogonal"] = out_ortho

    # 2. Multi-Slice Montage (Axial)
    montage_path = dest_dir / safe_filename(timestamped_filename(f"{prefix}_montage_axial", "png"))
    _, out_montage = plot_slice_montage(
        mri_volume=vol_3d,
        mask=mask_3d,
        plane="axial",
        num_slices=12,
        grid_cols=4,
        output_path=montage_path,
    )
    generated["montage"] = out_montage

    # 3. Mask Occupancy Profile (if mask is present)
    if mask_3d is not None and np.any(mask_3d > 0):
        occupancy_path = dest_dir / safe_filename(timestamped_filename(f"{prefix}_occupancy", "png"))
        _, out_occupancy = plot_mask_occupancy(
            mask=mask_3d,
            output_path=occupancy_path,
        )
        generated["occupancy"] = out_occupancy

    return generated

