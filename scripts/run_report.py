#!/usr/bin/env python3
"""CLI script for running BrainGemma3D report generation adapter.

DISCLAIMER: AI-generated research draft — not a medical diagnosis.
Research and demonstration use only. Never presents outputs as clinical diagnoses.

Usage:
    python scripts/run_report.py --input data/raw/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz
    python scripts/run_report.py --input data/raw/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz --mask outputs/masks/BRATS_001_mask.nii.gz
    python scripts/run_report.py --input <nifti_path> --mask <optional_mask_path> --output-dir outputs/reports --config configs/project_config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.reporting import BrainGemma3DReporter, MODEL_UNAVAILABLE_MESSAGE, RESEARCH_DISCLAIMER
from src.utils import ensure_directory, is_nifti_file


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="BrainMRI-SpatialVLM: Run BrainGemma3D Vision-Language Model Report Generation."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input 3D NIfTI brain MRI volume (.nii or .nii.gz).",
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
        default="outputs/reports",
        help="Directory to save generated draft reports (default: outputs/reports).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/project_config.yaml",
        help="Path to project configuration YAML file (default: configs/project_config.yaml).",
    )
    return parser.parse_args()


def main() -> int:
    """Execute BrainGemma3D report generation pipeline."""
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

    cfg_path = Path(args.config).resolve()
    if not cfg_path.is_file():
        print(f"Error: Config file not found: {cfg_path}", file=sys.stderr)
        return 1

    config = load_config(cfg_path)
    output_dir = Path(args.output_dir).resolve()
    ensure_directory(output_dir)

    print("=" * 65)
    print("BrainMRI-SpatialVLM: BrainGemma3D Report Generation Adapter")
    print(f"DISCLAIMER: {config.system.disclaimer}")
    print("=" * 65)
    print(f"Input scan  : {input_path}")
    if mask_path:
        print(f"Mask input  : {mask_path}")
    print(f"Output dir  : {output_dir}")
    print("=" * 65)

    reporter = BrainGemma3DReporter(
        config_path=cfg_path,
        require_gpu=False,  # Allow running pipeline check without terminating on CPU
    )

    # Check availability
    is_avail, missing = reporter.check_components_availability()
    if not is_avail:
        print(f"\n{MODEL_UNAVAILABLE_MESSAGE}", file=sys.stderr)
        print("Missing requirements:", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)

    report, json_path, md_path = reporter.generate_and_save(
        mri_volume=input_path,
        mask=mask_path,
        output_dir=output_dir,
    )

    print("\nReport Processing Complete.")
    print(f"  Status        : {report.status.upper()}")
    print(f"  JSON output   : {json_path}")
    print(f"  Markdown text : {md_path}")
    print(f"  Disclaimer    : {report.disclaimer}")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())

