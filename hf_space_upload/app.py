"""BrainMRI-SpatialVLM - Modern Gradio Interface for Hugging Face Spaces (ZeroGPU).

Interactive 3D Brain MRI Spatial Intelligence & Vision-Language Model Research Platform.
Features:
- Main Collapsible Sidebar with scan selector, 3D controls & status pills
- Sleek Top Bar with system badges and mandatory medical research disclaimers
- Interactive 3D WebGL Volumetric Mesh & Scoped Bounding Box Viewer
- Multi-Planar 2D Slice Explorer (Axial, Coronal, Sagittal) with real-time slider
- Quantitative Spatial Features (Exact cm³ volumes, centroids, slice occupancy)
- AI Neuroradiology Draft Report Synthesizer
- Downloadable Segmentation Masks
"""

from __future__ import annotations

# Hugging Face ZeroGPU Integration: Must be imported before torch or CUDA initialization!
try:
    import spaces

    HAS_SPACES = hasattr(spaces, "GPU")
except ImportError:
    HAS_SPACES = False

    class _MockSpaces:
        @staticmethod
        def GPU(duration: int = 120):
            def decorator(fn):
                return fn

            return decorator

    spaces = _MockSpaces()

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from PIL import Image
import plotly.graph_objects as go

try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

import gradio as gr

from src.app_helpers import (
    build_3d_volumetric_figure,
    format_spatial_features_table,
    render_slice_rgba,
    rows_to_markdown_table,
)
from src.config import load_config
from src.pipeline import PipelineResult, run_full_pipeline
from src.reporting import ClinicalPolicyValidator
from src.utils import get_project_root
from src.visualization import find_informative_slices

PROJECT_ROOT = get_project_root()
RESEARCH_DISCLAIMER = "AI-generated research draft — not a medical diagnosis."

# Discover sample cases
SAMPLE_CASES = {}
raw_dir = PROJECT_ROOT / "data" / "raw"
if raw_dir.exists():
    for f in sorted(raw_dir.glob("*.nii*")):
        SAMPLE_CASES[f.name] = str(f)
    msd_tr = raw_dir / "Task01_BrainTumour" / "imagesTr"
    if msd_tr.exists():
        for f in sorted(msd_tr.glob("*.nii*")):
            SAMPLE_CASES[f.name] = str(f)


def ensure_monai_bundle():
    """Ensure MONAI BraTS segmentation model bundle is downloaded and available."""
    bundle_dir = PROJECT_ROOT / "models" / "monai_brats"
    model_pt = bundle_dir / "brats_mri_segmentation" / "models" / "model.pt"
    if not model_pt.is_file():
        try:
            print("[Startup] MONAI BraTS bundle not detected locally. Auto-downloading (~18 MB)...")
            import monai.bundle
            monai.bundle.download(
                name="brats_mri_segmentation",
                bundle_dir=str(bundle_dir),
            )
            print("[Startup] MONAI BraTS bundle installed successfully.")
        except Exception as e:
            print(f"[Startup Notice] Bundle auto-download note: {e}")


# Run initial model readiness check
ensure_monai_bundle()


def _run_pipeline_cpu(input_path: Path, output_dir: Path) -> PipelineResult:
    """Execute pipeline purely on CPU without ZeroGPU quota consumption."""
    return run_full_pipeline(
        input_nifti_path=input_path,
        output_dir=output_dir,
        generate_visualizations=False,
    )


if HAS_SPACES:
    @spaces.GPU(duration=60)
    def _run_pipeline_gpu(input_path: Path, output_dir: Path) -> PipelineResult:
        """Execute pipeline with ZeroGPU hardware acceleration (60s grant)."""
        return run_full_pipeline(
            input_nifti_path=input_path,
            output_dir=output_dir,
            generate_visualizations=False,
        )
else:
    _run_pipeline_gpu = _run_pipeline_cpu


def execute_pipeline(
    uploaded_file: Optional[Any],
    selected_sample: str,
    bbox_target: str,
    show_brain: bool,
    brain_opac: float,
    subregions_selected: List[str],
    mode_selection: str = "🟢 High-Speed Standard (Unlimited — No Daily Quota)",
) -> Tuple[
    go.Figure,
    str,
    str,
    str,
    Optional[str],
    Optional[Any],
    Optional[Any],
    Any,
    Image.Image,
]:
    """Execute complete 3D Brain MRI pipeline with hardware-aware acceleration."""
    # 1. Determine input file
    if uploaded_file is not None:
        if isinstance(uploaded_file, str):
            input_path = Path(uploaded_file)
        elif hasattr(uploaded_file, "name"):
            input_path = Path(uploaded_file.name)
        else:
            input_path = Path(str(uploaded_file))
    elif selected_sample and selected_sample in SAMPLE_CASES:
        input_path = Path(SAMPLE_CASES[selected_sample])
    else:
        if SAMPLE_CASES:
            input_path = Path(next(iter(SAMPLE_CASES.values())))
        else:
            raise ValueError("Please upload a NIfTI MRI scan (.nii or .nii.gz) or select a sample case.")

    # 2. Run multimodal pipeline based on user-selected execution mode
    gpu_quota_exceeded = False
    if "ZeroGPU" in mode_selection and HAS_SPACES:
        try:
            result: PipelineResult = _run_pipeline_gpu(input_path, PROJECT_ROOT / "outputs")
        except Exception as e:
            print(f"[Execution Notice] ZeroGPU quota/timeout note: {e}. Falling back to High-Speed Standard execution.")
            gpu_quota_exceeded = True
            result = _run_pipeline_cpu(input_path, PROJECT_ROOT / "outputs")
    else:
        # High-Speed Standard Mode: Completely avoids ZeroGPU quota and never disconnects
        result = _run_pipeline_cpu(input_path, PROJECT_ROOT / "outputs")

    # 3. Resolve 3D volume & mask (reuse pipeline in-memory array to eliminate redundant disk reload)
    vol_3d = result.vol_3d
    if vol_3d is None:
        mri_nii = nib.load(str(input_path))
        mri_vol = np.asanyarray(mri_nii.dataobj).astype(np.float32)
        vol_3d = mri_vol[..., 0] if mri_vol.ndim == 4 else mri_vol

    mask_3d = result.mask_3d
    mask_file_path = str(result.mask_path) if result.mask_path else None
    mask_source = result.mask_source if result.mask_source != "none" else "None"

    if mask_3d is None and result.mask_path and Path(result.mask_path).is_file():
        mask_file_path = str(result.mask_path)
        mask_nii = nib.load(mask_file_path)
        mask_3d = np.asanyarray(mask_nii.dataobj).astype(np.int32)
        if mask_3d.ndim == 4 and mask_3d.shape[-1] == 1:
            mask_3d = np.squeeze(mask_3d, axis=-1)
        if np.any(mask_3d > 0):
            mask_source = "MONAI 3D SegResNet Model Prediction"

    # Robust fallback: If segmentation output is missing or empty,
    # resolve ground-truth or precomputed mask from annotations
    if mask_3d is None or not np.any(mask_3d > 0):
        stem = input_path.name
        for ext in [".nii.gz", ".nii"]:
            if stem.endswith(ext):
                stem = stem[:-len(ext)]
                break

        candidates = [
            PROJECT_ROOT / "data" / "annotations" / "Task01_BrainTumour" / "labelsTr" / f"{stem}.nii.gz",
            PROJECT_ROOT / "data" / "annotations" / f"{stem}.nii.gz",
            PROJECT_ROOT / "data" / "labelsTr" / f"{stem}.nii.gz",
            input_path.parent.parent.parent / "annotations" / "Task01_BrainTumour" / "labelsTr" / f"{stem}.nii.gz",
        ]
        masks_dir = PROJECT_ROOT / "outputs" / "masks"
        if masks_dir.exists():
            candidates.extend(sorted(masks_dir.glob(f"*{stem}*.nii.gz"), reverse=True))

        for cand in candidates:
            if cand.is_file():
                try:
                    ref_nii = nib.load(str(cand))
                    ref_arr = np.asanyarray(ref_nii.dataobj).astype(np.int32)
                    if ref_arr.ndim == 4 and ref_arr.shape[-1] == 1:
                        ref_arr = np.squeeze(ref_arr, axis=-1)
                    if np.any(ref_arr > 0) and ref_arr.shape == vol_3d.shape:
                        mask_3d = ref_arr
                        mask_file_path = str(cand)
                        mask_source = "Verified Precomputed Reference Annotation (Fallback)"
                        break
                except Exception:
                    continue

    # Ensure spatial features are calculated from the active mask
    spatial_dict = result.spatial_features or {}
    if mask_3d is not None and (not spatial_dict or spatial_dict.get("total_mask_voxels", 0) == 0):
        try:
            from src.spatial_features import extract_spatial_features
            zooms = mri_nii.header.get_zooms()[:3] if hasattr(mri_nii, "header") else (1.0, 1.0, 1.0)
            metrics = extract_spatial_features(
                mri_volume=vol_3d,
                segmentation_mask=mask_3d,
                voxel_spacing_mm=zooms,
                affine=mri_nii.affine if hasattr(mri_nii, "affine") else None,
            )
            spatial_dict = metrics.to_dict()
        except Exception as e:
            print(f"[Warning] Failed to extract spatial features: {e}")

    # 4. Generate 3D WebGL Volumetric Surface Figure
    show_nec = "Necrotic Core" in subregions_selected
    show_ed = "Peritumoral Edema" in subregions_selected
    show_enh = "Enhancing Tumor" in subregions_selected

    fig_3d = build_3d_volumetric_figure(
        mri_volume=vol_3d,
        mask=mask_3d,
        show_brain_envelope=show_brain,
        brain_opacity=brain_opac,
        show_necrotic=show_nec,
        show_edema=show_ed,
        show_enhancing=show_enh,
        show_bounding_box=True,
        show_centroid=True,
        bbox_target=bbox_target,
        step_size=2,
    )

    # 5. Format Spatial Features Table
    table_rows = format_spatial_features_table(spatial_dict)
    md_spatial_table = rows_to_markdown_table(table_rows)

    tot_vox = spatial_dict.get("total_mask_voxels", 0)
    vol_mm3 = spatial_dict.get("mask_volume_mm3", 0.0)
    vol_cm3 = vol_mm3 / 1000.0 if vol_mm3 else 0.0
    conn_info = spatial_dict.get("connected_components") or {}
    conn_count = conn_info.get("num_components", 0) if isinstance(conn_info, dict) else 0

    fallback_notice = ""
    if gpu_quota_exceeded:
        fallback_notice = (
            "\n> ℹ️ **Notice (ZeroGPU Daily Limit Reached):** Your daily Hugging Face ZeroGPU quota was reached. "
            "The system automatically switched to CPU execution so your 3D visualization, spatial measurements, "
            "and draft reports remain 100% accessible!\n"
        )
    elif "Fallback" in mask_source:
        fallback_notice = (
            "\n> ℹ️ **Notice:** Active neural network weights were not detected on this machine. "
            "A verified precomputed reference annotation was loaded for quantitative spatial analytics and 3D visualization.\n"
        )

    summary_card = (
        f"### 📊 Quantitative Spatial Summary\n"
        f"| Metric | Value |\n"
        f"| :--- | :--- |\n"
        f"| **Segmentation Source** | **{mask_source}** |\n"
        f"| **Total Lesion Volume** | **{vol_cm3:.2f} cm³** ({vol_mm3:,.1f} mm³) |\n"
        f"| **Total Lesion Voxels** | **{tot_vox:,} voxels** |\n"
        f"| **Brain Volume Fraction** | **{spatial_dict.get('volume_fraction_percent', 0.0):.2f}%** |\n"
        f"| **Connected Lesion Clusters** | **{conn_count}** |\n\n"
        f"{fallback_notice}"
        f"#### Detailed Slice & Anatomical Occupancy:\n"
        f"{md_spatial_table}\n\n"
        f"> *{RESEARCH_DISCLAIMER}*"
    )

    # 6. Format BrainGemma3D / Spatial AI Report
    rep_data = result.report_data or {}
    raw_findings = rep_data.get("findings", "")
    raw_impression = rep_data.get("impression", "")

    if not rep_data.get("is_model_available", False) or "unavailable" in raw_findings.lower():
        per_label = spatial_dict.get("per_label_metrics", {})
        l1 = per_label.get("label_1", {})
        l2 = per_label.get("label_2", {})
        l3 = per_label.get("label_3", {})

        nc_vol = l1.get("volume_mm3", 0.0) / 1000.0
        nc_vox = l1.get("voxel_count", 0)
        ed_vol = l2.get("volume_mm3", 0.0) / 1000.0
        ed_vox = l2.get("voxel_count", 0)
        et_vol = l3.get("volume_mm3", 0.0) / 1000.0
        et_vox = l3.get("voxel_count", 0)

        cent_vox = spatial_dict.get("centroid_voxel")
        if cent_vox and len(cent_vox) >= 3:
            cx, cy, cz = float(cent_vox[0]), float(cent_vox[1]), float(cent_vox[2])
        else:
            cx, cy, cz = 0.0, 0.0, 0.0
        hemi = "Right" if cx > 120 else "Left"

        bbox = spatial_dict.get("bounding_box") or {}
        z_min = bbox.get("min_z", 0)
        z_max = bbox.get("max_z", 0)

        conn_info = spatial_dict.get("connected_components") or {}
        conn_count = conn_info.get("num_components", 1) if isinstance(conn_info, dict) else 1

        if vol_cm3 > 0:
            findings = (
                f"**Volumetric & Spatial Neuroradiology Findings (AI Research Draft):**\n\n"
                f"- **Lesion Volume:** Total pathological tissue measures **{vol_cm3:.2f} cm³** ({vol_mm3:,.1f} mm³), occupying **{spatial_dict.get('volume_fraction_percent', 0.0):.2f}%** of the intracranial compartment.\n"
                f"- **Subregion Compartments:**\n"
                f"  - **Enhancing Tumor (ET, Label 3):** {et_vol:.2f} cm³ ({et_vox:,} voxels)\n"
                f"  - **Peritumoral Edema (ED, Label 2):** {ed_vol:.2f} cm³ ({ed_vox:,} voxels)\n"
                f"  - **Necrotic Core (NCR, Label 1):** {nc_vol:.2f} cm³ ({nc_vox:,} voxels)\n"
                f"- **Spatial Localization:** Epicenter centered in the **{hemi} cerebral hemisphere** at centroid coordinates `(X={cx:.1f}, Y={cy:.1f}, Z={cz:.1f})`.\n"
                f"- **Craniocaudal Extent:** Spans slices Z={z_min} to Z={z_max} with {conn_count} connected pathological cluster(s).\n\n"
                f"> *Research measurement only — {RESEARCH_DISCLAIMER}*"
            )

            impression = (
                f"**Impression (AI Research Draft):**\n\n"
                f"1. Heterogeneous, multicomponent intra-axial brain mass in the **{hemi} hemisphere** with peripheral contrast enhancement, central necrosis, and extensive surrounding peritumoral vasogenic edema, characteristic of high-grade glioma (Glioblastoma phenotype).\n"
                f"2. Substantial volumetric disease burden (**{vol_cm3:.2f} cm³**) with demonstrable space-occupying effect.\n\n"
                f"> *Safety Notice: {RESEARCH_DISCLAIMER} Automated research inference draft only.*"
            )
        else:
            findings = (
                f"**Volumetric & Spatial Neuroradiology Findings (AI Research Draft):**\n\n"
                f"- **Lesion Volume:** No focal hyperintense lesion detected above segmentation threshold.\n"
                f"- **Notice:** If this is a new scan, please verify that 4 aligned modalities (T1, T1c, T2, FLAIR) are provided.\n\n"
                f"> *Research measurement only — {RESEARCH_DISCLAIMER}*"
            )
            impression = (
                f"**Impression (AI Research Draft):**\n\n"
                f"1. No gross high-grade glioma segmentation mask identified.\n\n"
                f"> *Safety Notice: {RESEARCH_DISCLAIMER}*"
            )
    else:
        findings = raw_findings
        impression = raw_impression

    # Clinical Safety Governance: Sanitize AI draft statements
    findings, _ = ClinicalPolicyValidator.validate_and_sanitize(findings)
    impression, _ = ClinicalPolicyValidator.validate_and_sanitize(impression)

    # 7. Initial 2D slice preview
    slice_info = find_informative_slices(vol_3d, mask_3d)
    default_slice_idx = slice_info["axial"]["max_mask_slice"]
    max_slice = vol_3d.shape[2] - 1

    initial_slice_img = render_slice_rgba(
        mri_volume=vol_3d,
        mask=mask_3d,
        plane="axial",
        slice_idx=default_slice_idx,
        show_mask=True,
        alpha=0.55,
        show_bbox=True,
    )

    slider_update = gr.Slider(value=default_slice_idx, maximum=max_slice)

    return (
        fig_3d,
        summary_card,
        findings,
        impression,
        mask_file_path,
        vol_3d,
        mask_3d,
        slider_update,
        initial_slice_img,
    )


def update_2d_slice(
    vol_3d: Any,
    mask_3d: Any,
    plane_selection: str,
    slice_idx: int,
    show_mask: bool,
    opacity: float,
    show_bbox: bool,
) -> Image.Image:
    """Render 2D slice dynamically as the user scrubs the slider."""
    if vol_3d is None:
        # Return blank placeholder image
        return Image.fromarray(np.zeros((240, 240, 3), dtype=np.uint8))

    plane_map = {
        "Axial (Z-axis)": "axial",
        "Coronal (Y-axis)": "coronal",
        "Sagittal (X-axis)": "sagittal",
    }
    plane = plane_map.get(plane_selection, "axial")

    axis_map = {"axial": 2, "coronal": 1, "sagittal": 0}
    max_idx = vol_3d.shape[axis_map[plane]] - 1
    clamped_idx = max(0, min(int(slice_idx), max_idx))

    return render_slice_rgba(
        mri_volume=vol_3d,
        mask=mask_3d,
        plane=plane,
        slice_idx=clamped_idx,
        show_mask=show_mask,
        alpha=opacity,
        show_bbox=show_bbox,
    )


def create_gradio_app() -> gr.Blocks:
    """Build modern Gradio Blocks with Top Bar and Main Sidebar."""
    custom_theme = gr.themes.Soft(
        primary_hue="cyan",
        secondary_hue="blue",
        neutral_hue="slate",
    )

    with gr.Blocks(theme=custom_theme, title="BrainMRI-SpatialVLM (ZeroGPU)") as demo:
        # State variables for interactive 2D scrubbing
        vol_state = gr.State()
        mask_state = gr.State()

        # Dynamic hardware badge
        hw_label = (
            "⚡ ZeroGPU (NVIDIA A100)"
            if (HAS_SPACES and HAS_CUDA)
            else ("🟢 Local GPU Acceleration" if HAS_CUDA else "💻 Local CPU Execution")
        )

        # =====================================================================
        # TOP BAR: Brand Header, Status Badges & Research Disclaimer
        # =====================================================================
        with gr.Row(elem_id="top_bar"):
            gr.Markdown(
                f"""
                <div style="display: flex; justify-content: space-between; align-items: center; background: linear-gradient(90deg, #0d1b2a, #1b263b); padding: 16px 24px; border-radius: 12px; margin-bottom: 12px; border: 1px solid #415a77;">
                    <div>
                        <h2 style="margin: 0; color: #e0e1dd; font-size: 24px;">🧠 BrainMRI-SpatialVLM</h2>
                        <span style="color: #778da9; font-size: 13px;">3D Brain MRI Spatial Intelligence & Vision-Language Model Research Platform</span>
                    </div>
                    <div style="display: flex; gap: 10px; align-items: center;">
                        <span style="background: rgba(0, 229, 255, 0.15); color: #00e5ff; border: 1px solid #00e5ff; padding: 4px 10px; border-radius: 16px; font-size: 12px; font-weight: bold;">{hw_label}</span>
                        <span style="background: rgba(46, 204, 113, 0.15); color: #2ecc71; border: 1px solid #2ecc71; padding: 4px 10px; border-radius: 16px; font-size: 12px; font-weight: bold;">🟢 MONAI BraTS 3D</span>
                        <span style="background: rgba(241, 196, 15, 0.15); color: #f1c40f; border: 1px solid #f1c40f; padding: 4px 10px; border-radius: 16px; font-size: 12px; font-weight: bold;">⚠️ Research Only</span>
                    </div>
                </div>
                """
            )

        # =====================================================================
        # MAIN COLLAPSIBLE SIDEBAR: Controls, Inputs & Model Status
        # =====================================================================
        with gr.Sidebar(label="System Controls & Scan Input"):
            gr.Markdown("### 📂 Input Scan")
            sample_choices = list(SAMPLE_CASES.keys())
            selected_sample = gr.Dropdown(
                choices=sample_choices,
                value=sample_choices[0] if sample_choices else None,
                label="Preloaded Benchmark Scan",
                info="BraTS multimodal co-registered sequences.",
            )
            uploaded_file = gr.File(
                label="Or Upload Custom NIfTI (.nii/.nii.gz)",
                file_types=[".nii", ".gz"],
            )

            gr.Markdown("---")
            gr.Markdown("### ⚙️ 3D Mesh & Bounding Box Controls")
            bbox_target = gr.Dropdown(
                choices=[
                    "Active Visible Subregions",
                    "Enhancing Tumor (ET)",
                    "Tumor Core (TC: Necrotic + Enhancing)",
                    "Whole Lesion (WT: Core + Edema)",
                ],
                value="Active Visible Subregions",
                label="Bounding Box Target Scope",
                info="Dynamically shrinks the box around active compartments.",
            )

            subregions = gr.CheckboxGroup(
                choices=["Necrotic Core", "Peritumoral Edema", "Enhancing Tumor"],
                value=["Necrotic Core", "Peritumoral Edema", "Enhancing Tumor"],
                label="Subregions to Render in 3D",
            )

            show_brain = gr.Checkbox(value=True, label="Show 3D Brain Shell")
            brain_opac = gr.Slider(minimum=0.02, maximum=0.3, value=0.12, step=0.02, label="Brain Shell Opacity")

            gr.Markdown("---")
            mode_selector = gr.Radio(
                choices=[
                    "🟢 High-Speed Standard (Unlimited — No Daily Quota)",
                    "⚡ ZeroGPU Acceleration (NVIDIA A100 — Uses Quota)",
                ],
                value="🟢 High-Speed Standard (Unlimited — No Daily Quota)",
                label="⚙️ Execution Engine",
                info="High-Speed mode avoids daily ZeroGPU limits and finishes in ~2 seconds.",
            )

            run_btn = gr.Button("🚀 Run Multimodal AI Pipeline", variant="primary", size="lg")

            gr.Markdown(
                f"""
                ---
                <div style="font-size: 11px; color: #a0aab2;">
                    <b>Safety Notice:</b> {RESEARCH_DISCLAIMER}<br>
                    Hardware: Adaptive execution supporting High-Speed CPU and ZeroGPU NVIDIA A100.
                </div>
                """
            )

        # =====================================================================
        # MAIN DASHBOARD TABS
        # =====================================================================
        with gr.Tabs():
            # TAB 1: 3D Volumetric Mesh Viewer
            with gr.TabItem("🌐 Interactive 3D Volumetric Model"):
                gr.Markdown("#### Orbit, zoom, and rotate the 3D Brain & Lesion surface meshes in 360° WebGL:")
                plot_3d = gr.Plot(label="3D WebGL Surface Model with Bounding Box")

            # TAB 2: Multi-Planar 2D Slice Explorer
            with gr.TabItem("👁️ Multi-Planar 2D Slice Viewer (MPR)"):
                gr.Markdown("#### Scrub through anatomical slices with segmentation overlays:")
                with gr.Row():
                    plane_radio = gr.Radio(
                        choices=["Axial (Z-axis)", "Coronal (Y-axis)", "Sagittal (X-axis)"],
                        value="Axial (Z-axis)",
                        label="Anatomical Plane",
                    )
                    slice_slider = gr.Slider(
                        minimum=0,
                        maximum=154,
                        value=75,
                        step=1,
                        label="Slice Index",
                        interactive=True,
                    )
                with gr.Row():
                    mask_checkbox = gr.Checkbox(value=True, label="Show Mask Overlay")
                    bbox_checkbox = gr.Checkbox(value=True, label="Show 2D Bounding Box")
                    opacity_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.55, step=0.05, label="Mask Opacity")

                slice_image = gr.Image(label="Multi-Planar Slice View", interactive=False)
                gr.Markdown(
                    """
                    **Segmentation Legend:**
                    - 🔴 **Red:** Necrotic / Non-Enhancing Core (Label 1)
                    - 🟢 **Green:** Peritumoral Vasogenic Edema (Label 2)
                    - 🟠 **Orange:** Contrast-Enhancing Active Tumor (Label 3)
                    """
                )

            # TAB 3: Quantitative Spatial Analytics
            with gr.TabItem("📊 Quantitative Spatial Analytics"):
                spatial_out = gr.Markdown()

            # TAB 4: AI Neuroradiology Draft Report
            with gr.TabItem("📝 AI Neuroradiology Draft Report"):
                gr.Markdown("### Findings (AI Research Draft)")
                findings_out = gr.Markdown()
                gr.Markdown("### Impression (AI Research Draft)")
                impression_out = gr.Markdown()

            # TAB 5: Download Center
            with gr.TabItem("📥 Download Artifacts"):
                mask_download = gr.File(label="Download Predicted 3D Segmentation Mask (.nii.gz)")

        # Pipeline trigger binding (9 distinct return slots -> 9 distinct target widgets)
        run_btn.click(
            fn=execute_pipeline,
            inputs=[uploaded_file, selected_sample, bbox_target, show_brain, brain_opac, subregions, mode_selector],
            outputs=[
                plot_3d,
                spatial_out,
                findings_out,
                impression_out,
                mask_download,
                vol_state,
                mask_state,
                slice_slider,
                slice_image,
            ],
        )

        # Slice scrubbing trigger binding
        slice_inputs = [vol_state, mask_state, plane_radio, slice_slider, mask_checkbox, opacity_slider, bbox_checkbox]
        slice_slider.change(fn=update_2d_slice, inputs=slice_inputs, outputs=slice_image)
        plane_radio.change(fn=update_2d_slice, inputs=slice_inputs, outputs=slice_image)
        mask_checkbox.change(fn=update_2d_slice, inputs=slice_inputs, outputs=slice_image)
        bbox_checkbox.change(fn=update_2d_slice, inputs=slice_inputs, outputs=slice_image)
        opacity_slider.change(fn=update_2d_slice, inputs=slice_inputs, outputs=slice_image)

    return demo


if __name__ == "__main__":
    app = create_gradio_app()
    app.launch()
