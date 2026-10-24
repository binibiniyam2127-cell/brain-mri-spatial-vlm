# BrainMRI-SpatialVLM: 3D Brain MRI Spatial Intelligence and Vision-Language Model System

> **DISCLAIMER: AI-generated research draft — not a medical diagnosis.**  
> This project is developed strictly for computational neuroscience, machine learning, and medical AI research demonstration purposes. It does not provide medical diagnoses, treatment advice, or clinical evaluations, and must **never** be used in clinical workflows, diagnostic interpretations, or patient care decisions.

---

## 1. System & Architecture Overview

**BrainMRI-SpatialVLM** is a research inference platform designed to integrate 3D Brain Magnetic Resonance Imaging (MRI) segmentation with spatial intelligence analysis and vision-language report generation. 

```
                                  [ Input MRI Scan (.nii / .nii.gz) ]
                                 (4 channels: T1, T1c, T2, FLAIR)
                                                │
                                                ▼
                             [ Stage 1: Validation & Ingestion ]
                                  (data/raw/ strictly read-only)
                                                │
                                                ▼
                         [ Stage 2: Stacking & Preprocessing ]
                                     (data/processed/)
                                                │
                                                ▼
                   [ Stage 3: MONAI BraTS 3D Tumor Segmentation ]
                                      (outputs/masks/)
                                                │
                                                ▼
                 [ Stage 4: Quantitative Spatial Feature Extraction ]
                        (outputs/reports/ — Physical volume,
                         centroid, bounding box, connected components)
                                                │
                                                ▼
                [ Stage 5: BrainGemma3D Vision-Language Draft Report ]
                       (outputs/reports/ — MedSigLIP 3D + MedGemma
                         AI-generated research draft findings)
                                                │
                                                ▼
             [ Stage 6: Multi-Planar & Montage Visualizations ]
                     (outputs/visualizations/ — Tri-planar,
                      montage grid, slice occupancy profile)
                                                │
                                                ▼
                      [ Stage 7: Unified Packaging & Interactive Web UI ]
```

### Core Architecture Components:
1. **Multimodal MRI Ingestion**: Validates and normalizes 3D/4D NIfTI neuroimaging volumes. Original raw data in `data/raw/` is protected and preserved as read-only.
2. **MONAI BraTS 3D Segmentation**: Utilizes the pretrained MONAI Model Zoo `brats_mri_segmentation` bundle (3D UNet) to segment brain tumor subregions:
   - Label 1: Necrotic / Non-enhancing tumor core (NCR/NET)
   - Label 2: Peritumoral edema (ED)
   - Label 3: GD-enhancing tumor (ET)
3. **Spatial Feature Extraction Engine**: Extracts non-diagnostic geometric and structural measurements:
   - Total mask voxel count and physical volume in $\text{mm}^3$ (using NIfTI voxel spacing zooms)
   - Brain volume fraction (%)
   - 3D axis-aligned bounding box coordinates
   - Centroid coordinates (both voxel index and scanner world coordinates)
   - Axis extents along $X$, $Y$, and $Z$ axes (voxels and millimeters)
   - Topological connected component count and multi-cluster analysis (`scipy.ndimage.label`)
   - Slice-wise occupancy distributions across axial, coronal, and sagittal orientations
   - Descriptive ground-truth reference mask comparisons (Dice coefficient and Jaccard index)
4. **BrainGemma3D Vision-Language Reporting**: Couples an inflated 3D vision encoder (`google/medsiglip-448` with Conv2D $\rightarrow$ Conv3D center-frame expansion) and a 2-layer Perceiver token compressor with a 4-bit quantized language model (`google/medgemma-1.5-4b-it`) and fine-tuned LoRA adapters to produce research draft radiology findings and impressions.
5. **Interactive Visualization & Web UI**: Renders multi-slice montages, orthogonal tri-planar views, and occupancy profiles with visible watermark banners, served through an interactive Gradio application (`app.py`).

---

## 2. Directory Structure & Data-Folder Organization

The repository strictly separates raw source data, preprocessed inputs, models, and generated outputs:

```
brain_mri_spatial_vlm/
├── app.py                     # Modern interactive research web application (Gradio / ZeroGPU)
├── configs/
│   └── project_config.yaml    # Centralized configuration (paths, devices, models)
├── data/
│   ├── raw/                   # Original MRI files (strictly read-only)
│   ├── processed/             # Inference-ready, stacked/preprocessed MRI volumes
│   ├── annotations/           # Optional reference ground-truth segmentation masks
│   └── downloads/             # Temporary dataset archive downloads (git-ignored)
├── models/
│   ├── monai_brats/           # MONAI BraTS bundle and model weights
│   └── braingemma3d/          # BrainGemma3D repo, MedSigLIP, MedGemma, checkpoints
├── notebooks/
│   └── demo_walkthrough.ipynb # Step-by-step Jupyter demonstration notebook
├── outputs/
│   ├── masks/                 # Predicted 3D segmentation masks (.nii.gz)
│   ├── reports/               # Spatial features JSON, draft reports (.md/.json), combined JSON
│   └── visualizations/        # Tri-planar, montage, and occupancy PNG figures
├── scripts/
│   ├── download_sample_data.py   # Downloads 25 public MSD BraTS research cases
│   ├── download_monai_brats.py   # Downloads MONAI BraTS segmentation bundle
│   ├── download_braingemma3d.py  # Downloads BrainGemma3D, MedSigLIP, MedGemma
│   ├── run_segmentation.py       # CLI runner for MONAI BraTS segmentation
│   ├── run_report.py             # CLI runner for BrainGemma3D draft reporting
│   ├── run_visualization.py      # CLI runner for multi-planar visualizer
│   └── run_pipeline.py           # CLI runner for end-to-end multimodal pipeline
├── src/
│   ├── app_helpers.py         # UI business logic and 3D WebGL rendering helpers
│   ├── config.py              # Configuration loader and validator
│   ├── mri_io.py              # Robust NIfTI I/O and dimensional validation
│   ├── pipeline.py            # End-to-end multimodal orchestration engine
│   ├── reporting.py           # BrainGemma3D vision-language model adapter
│   ├── segmentation.py        # MONAI BraTS segmentation adapter
│   ├── spatial_features.py    # Quantitative spatial feature extraction engine
│   ├── utils.py               # Path resolution, logging, and filename sanitization
│   └── visualization.py       # MRI & mask visualization and watermark burning
└── tests/                     # Comprehensive offline test suite (70+ unit/integration tests)
```

---

## 3. Required MRI Input Format

The MONAI BraTS segmentation model expects four co-registered multimodal MRI sequences:
- **T1**: T1-weighted native scan
- **T1c / T1gd**: T1-weighted contrast-enhanced scan (gadolinium contrast)
- **T2**: T2-weighted scan
- **FLAIR**: Fluid Attenuated Inversion Recovery scan

Inputs can be supplied as:
- A single 4D NIfTI file (`.nii` or `.nii.gz`) with shape `(H, W, D, 4)` or `(4, H, W, D)` containing all 4 aligned sequences (such as the Medical Segmentation Decathlon Task01 BraTS dataset).
- A directory or dictionary containing the individual aligned 3D sequence volumes.

---

## 4. Environment Setup & Installation

### Prerequisites
- Python 3.10 to 3.14 (64-bit)
- NVIDIA CUDA GPU recommended for fast 3D deep learning and 4B parameter VLM inference (CPU fallback supported)

### Installation (PowerShell on Windows)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Model & Data Download Instructions

### Step A: Download 25 Public Research MRI Cases
Downloads the first 25 training cases from the Medical Segmentation Decathlon (MSD) Task01 BrainTumour dataset into `data/raw/` and reference masks into `data/annotations/`:
```powershell
python scripts/download_sample_data.py
```
CLI options:
- `--max-cases 25`: Number of cases to extract.
- `--output-dir data/raw/Task01_BrainTumour/imagesTr`: Destination for MRI scans.
- `--annotations-dir data/annotations/Task01_BrainTumour/labelsTr`: Destination for reference masks.

### Step B: Download MONAI BraTS Segmentation Bundle
Downloads the official MONAI Model Zoo `brats_mri_segmentation` bundle into `models/monai_brats/`:
```powershell
python scripts/download_monai_brats.py
```

### Step C: Download BrainGemma3D, MedSigLIP, and MedGemma
> [!IMPORTANT]
> `google/medsiglip-448` and `google/medgemma-1.5-4b-it` are gated models on Hugging Face. You must have an active Hugging Face account and accept the applicable model licenses on their model pages prior to downloading.

1. Log in to Hugging Face:
   ```powershell
   huggingface-cli login
   ```
2. Run the downloader utility:
   ```powershell
   python scripts/download_braingemma3d.py
   ```
   Optional checkpoint link argument:
   ```powershell
   python scripts/download_braingemma3d.py --checkpoint-dir /path/to/phase2b_final
   ```

---

## 6. Running the Research Web UI

Launch the interactive web user interface:
```powershell
python app.py
```

### UI Features:
- **Case Selection**: Choose from 25 preloaded MSD BraTS cases or upload custom `.nii` / `.nii.gz` files.
- **Reference Overlap**: Upload optional ground-truth masks to evaluate research overlap metrics (Dice / Jaccard).
- **One-Click Pipeline**: Click **"Run Research Pipeline"** to run preprocessing, segmentation, spatial extraction, draft reporting, and visualization.
- **🧊 Interactive 3D Volumetric Surface Mesh**:
  - True 3D WebGL polygonal surface rendering powered by Marching Cubes and Plotly (`go.Mesh3d`).
  - **Full 360° Mouse Orbiting & Rotation**: Left-click and drag to rotate the brain and tumor in full 3D space.
  - **Interactive Zooming & Panning**: Scroll wheel to zoom in/out, right-click and drag to pan across structures.
  - **Transparent Brain Shell & Tumor Subregion Controls**: Adjust brain contour opacity and toggle individual 3D subregions (Necrotic Core in red, Peritumoral Edema in green, Enhancing Tumor in orange).
  - **Reference Mask Fallback**: Instantly visualizes ground-truth 3D reference masks even before downloading model weights.
- **👁️ Interactive Multi-Planar 2D Slice Viewer (MPR)**:
  - Seamlessly switch between Axial (Z), Coronal (Y), and Sagittal (X) planes.
  - Automatically defaults slider to the anatomical slice with the peak predicted lesion surface area.
  - Interactive multi-label color overlays with opacity controls.
- **Spatial Metrics Table**: Formatted physical measurements labeled as:
  `“Research-only spatial measurements — not clinical findings.”`
- **Vision-Language Report Panel**: Displays draft radiology findings. If model weights are not installed, clearly displays:
  `“BrainGemma3D model unavailable. No AI-generated report can be produced.”`
- **One-Click Downloads**: Download predicted masks (`.nii.gz`), spatial metrics (`.json`), draft reports (`.md`), visualization figures (`.png`), and combined JSON results (`.json`).

---

## 7. Command-Line Runners (CLI)

### 1. End-to-End Multimodal Pipeline
```powershell
python scripts/run_pipeline.py --input data/raw/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz --config configs/project_config.yaml --output-dir outputs
```

### 2. Standalone MONAI BraTS Segmentation
```powershell
python scripts/run_segmentation.py --input data/raw/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz --output-dir outputs/masks
```

### 3. Standalone BrainGemma3D Report Generator
```powershell
python scripts/run_report.py --input data/raw/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz --mask data/annotations/Task01_BrainTumour/labelsTr/BRATS_001.nii.gz --output-dir outputs/reports
```

### 4. Standalone Visualization Generator
```powershell
python scripts/run_visualization.py --input data/raw/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz --mask data/annotations/Task01_BrainTumour/labelsTr/BRATS_001.nii.gz --output-dir outputs/visualizations
```

---

## 8. Running Tests

The test suite runs completely offline with mocked models and synthetic arrays, requiring zero model weight downloads:
```powershell
python -m pytest tests/ -v
```

Run specific test modules:
```powershell
python -m pytest tests/test_pipeline.py -v       # End-to-end pipeline integration
python -m pytest tests/test_segmentation.py -v   # MONAI segmentation adapter
python -m pytest tests/test_spatial_features.py -v # Spatial feature extraction
python -m pytest tests/test_reporting.py -v      # BrainGemma3D reporter adapter
python -m pytest tests/test_visualization.py -v  # Visualization and watermark tests
python -m pytest tests/test_app_helpers.py -v    # UI helpers and 3D WebGL tests
```

---

## 9. Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `MissingChannelError` | Input scan does not contain all 4 required BraTS channels (T1, T1c, T2, FLAIR). | Provide a 4D NIfTI file containing all 4 modalities, or select one of the preloaded MSD BraTS cases from `data/raw/`. |
| `BrainGemma3D model unavailable` | Weights or repository not yet downloaded into `models/braingemma3d/`. | This is expected behavior before weights are fetched. Run `python scripts/download_braingemma3d.py`. |
| `HFValidationError` / 401 Unauthorized | Gated model licenses for `medsiglip-448` or `medgemma-1.5-4b-it` not accepted on Hugging Face. | Accept licenses on Hugging Face model pages, then authenticate with `huggingface-cli login` or set `HF_TOKEN`. |
| `CUDA out of memory` | Insufficient GPU VRAM for 3D volumetric token processing. | Ensure you have $\ge 12\text{--}16\text{ GB}$ VRAM for 4-bit quantized execution, or configure `device: "cpu"` in `configs/project_config.yaml`. |
| Matplotlib font warnings | Standard server headless environment missing specific desktop font files. | Harmless warning; the visualization engine automatically falls back to default system fonts. |

---

## 10. Non-Diagnostic Research Guardrails

> **Strict Non-Clinical Policy**:
> - All spatial measurements are tagged: `"Research-only spatial measurements — not clinical findings."`
> - All reference comparisons are tagged: `"This is a research measurement, not a diagnostic evaluation."`
> - Every report, visualization figure, and JSON payload embeds the disclaimer:  
>   **“AI-generated research draft — not a medical diagnosis.”**
> - The system strictly prohibits the fabrication of medical conclusions, clinical diagnoses, or therapeutic claims.
#   b r a i n - m r i - s p a t i a l - v l m  
 