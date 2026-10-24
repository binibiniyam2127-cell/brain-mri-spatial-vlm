#!/usr/bin/env python3
"""CLI script for running the complete BrainMRI-SpatialVLM end-to-end inference pipeline.

DISCLAIMER: AI-generated research draft — not a medical diagnosis.
Research and demonstration use only. Never presents outputs as clinical diagnoses.

Usage:
    python scripts/run_pipeline.py --input data/raw/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz
    python scripts/run_pipeline.py --input <nifti_path> --config configs/project_config.yaml --output-dir outputs
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
from src.pipeline import RESEARCH_DISCLAIMER, run_full_pipeline
from src.utils import ensure_directory, is_nifti_file


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="BrainMRI-SpatialVLM: End-to-End Multimodal Inference Pipeline."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input 3D or 4D NIfTI brain MRI volume (.nii or .nii.gz).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/project_config.yaml",
        help="Path to project configuration YAML file (default: configs/project_config.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Base output directory for generated masks, reports, and visualizations (default: outputs).",
    )
    return parser.parse_args()


def main() -> int:
    """Execute end-to-end pipeline CLI."""
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

    cfg_path = Path(args.config).resolve()
    if not cfg_path.is_file():
        print(f"Error: Config file not found: {cfg_path}", file=sys.stderr)
        return 1

    config = load_config(cfg_path)
    output_dir = Path(args.output_dir).resolve()
    ensure_directory(output_dir)

    print("=" * 70)
    print("BrainMRI-SpatialVLM: End-to-End Inference Pipeline")
    print(f"DISCLAIMER: {RESEARCH_DISCLAIMER}")
    print("=" * 70)
    print(f"Input scan  : {input_path}")
    print(f"Config path : {cfg_path}")
    print(f"Output dir  : {output_dir}")
    print("=" * 70)

    try:
        result = run_full_pipeline(
            input_nifti_path=input_path,
            config=config,
            output_dir=output_dir,
        )

        print("\nPipeline Execution Finished.")
        print(f"  Overall Status    : {result.status.upper()}")
        print(f"  Case ID           : {result.case_id}")
        print(f"  Timestamp (UTC)   : {result.timestamp_utc}")
        print("\nGenerated Artifacts:")
        print(f"  - Preprocessed    : {result.processed_input_path or 'None'}")
        print(f"  - Predicted Mask  : {result.mask_path or 'None'}")
        print(f"  - Spatial Features: {result.spatial_features_path or 'None'}")
        print(f"  - Draft Report    : {result.draft_report_path or 'None'}")
        print(f"  - Combined JSON   : {result.combined_json_path or 'None'}")
        if result.visualization_paths:
            print("  - Visualizations  :")
            for v_name, v_path in result.visualization_paths.items():
                print(f"      * {v_name.capitalize()}: {v_path}")

        if result.errors:
            print("\nStage Warnings/Errors:")
            for stage, err in result.errors.items():
                print(f"  - [{stage.upper()}]: {err}")

        print("\n" + "=" * 70)
        print(f"Notice: {result.disclaimer}")
        print("=" * 70)
        return 0

    except Exception as e:
        print(f"\nPipeline Fatal Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

