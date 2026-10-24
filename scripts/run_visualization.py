#!/usr/bin/env python3
"""CLI script for generating research-only Brain MRI and segmentation visualizations.

DISCLAIMER: AI-generated research visualization — not a medical diagnosis.
Research and demonstration use only. Never presents outputs as clinical diagnoses.

Usage:
    python scripts/run_visualization.py --input data/raw/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz --mask data/annotations/Task01_BrainTumour/labelsTr/BRATS_001.nii.gz
    python scripts/run_visualization.py --input <nifti_path> --mask <mask_path> --output-dir outputs/visualizations
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.utils import ensure_directory, is_nifti_file
from src.visualization import (
    VISUALIZATION_WATERMARK,
    generate_all_visualizations,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="BrainMRI-SpatialVLM: Generate Brain MRI and Segmentation Visualizations."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input 3D/4D NIfTI brain MRI volume (.nii or .nii.gz).",
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="Optional path to 3D segmentation mask NIfTI file (.nii or .nii.gz).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/visualizations",
        help="Destination directory for generated PNG visualizations (default: outputs/visualizations).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/project_config.yaml",
        help="Path to project configuration YAML file (default: configs/project_config.yaml).",
    )
    return parser.parse_args()


def main() -> int:
    """Execute visualization generator."""
    args = parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Error: Input file does not exist: {input_path}", file=sys.stderr)
        return 1

    if not is_nifti_file(input_path):
        print(
            f"Error: Input file '{input_path.name}' is not a recognized NIfTI format (.nii or .nii.gz).",
            file=sys.stderr,
        )
        return 1

    mask_path: Optional[Path] = None
    if args.mask is not None:
        mask_path = Path(args.mask).resolve()
        if not mask_path.exists():
            print(f"Error: Mask file does not exist: {mask_path}", file=sys.stderr)
            return 1
        if not is_nifti_file(mask_path):
            print(
                f"Error: Mask file '{mask_path.name}' is not a recognized NIfTI format (.nii or .nii.gz).",
                file=sys.stderr,
            )
            return 1

    output_dir = Path(args.output_dir).resolve()
    ensure_directory(output_dir)

    print("=" * 65)
    print("BrainMRI-SpatialVLM: Visualization Generator")
    print(f"DISCLAIMER: {VISUALIZATION_WATERMARK}")
    print("=" * 65)
    print(f"Input scan  : {input_path}")
    if mask_path:
        print(f"Mask path   : {mask_path}")
    print(f"Output dir  : {output_dir}")
    print("=" * 65)

    case_stem = input_path.name.replace(".nii.gz", "").replace(".nii", "")
    generated = generate_all_visualizations(
        mri_volume=input_path,
        mask=mask_path,
        output_dir=output_dir,
        prefix=case_stem,
    )

    print("\nVisualizations Generated Successfully:")
    for key, path in generated.items():
        print(f"  [{key.upper()}] -> {path}")

    print(f"\nCaption / Watermark: \"{VISUALIZATION_WATERMARK}\"")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())

