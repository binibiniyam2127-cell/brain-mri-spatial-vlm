#!/usr/bin/env python3
"""CLI script for running MONAI BraTS 3D brain tumor segmentation inference.

DISCLAIMER: AI-generated research draft -- not a medical diagnosis.
Research and demonstration use only.

Usage:
    python scripts/run_segmentation.py --input path/to/4channel_mri.nii.gz
    python scripts/run_segmentation.py --input path/to/scan.nii.gz --output-dir outputs/masks --config configs/project_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.segmentation import BraTSSegmenter, MissingChannelError
from src.utils import ensure_directory, is_nifti_file


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="BrainMRI-SpatialVLM: Run MONAI BraTS 3D Tumor Segmentation Inference."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to 4-channel NIfTI volume (.nii or .nii.gz) containing aligned [T1, T1c, T2, FLAIR] modalities.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/masks",
        help="Directory to save predicted segmentation mask (default: outputs/masks).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/project_config.yaml",
        help="Path to project configuration YAML file (default: configs/project_config.yaml).",
    )
    return parser.parse_args()


def main() -> int:
    """Execute BraTS segmentation pipeline."""
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

    disclaimer_text = config.system.disclaimer.replace("—", "--")
    print("=" * 65)
    print("BrainMRI-SpatialVLM: MONAI BraTS 3D Tumor Segmentation Inference")
    print(f"DISCLAIMER: {disclaimer_text}")
    print("=" * 65)
    print(f"Input scan  : {input_path}")
    print(f"Output dir  : {output_dir}")

    segmenter = BraTSSegmenter(
        model_dir=(PROJECT_ROOT / config.monai_brats.bundle_dir).resolve(),
        bundle_name=config.monai_brats.bundle_name,
        device=config.inference.device,
        config_path=cfg_path,
    )

    try:
        saved_mask, metadata = segmenter.segment_and_save(
            inputs=input_path,
            output_dir=output_dir,
            save_processed_input=True,
        )
        print("\nSegmentation Completed Successfully.")
        print(f"Mask saved to: {saved_mask}")
        print(f"Mask shape   : {metadata['mask_shape']}")
        print(f"Label counts : {metadata['voxel_counts']}")
        return 0

    except MissingChannelError as err:
        print(f"\nInput Error: {err}", file=sys.stderr)
        return 2

    except RuntimeError as err:
        print(f"\nExecution Error: {err}", file=sys.stderr)
        return 3

    except Exception as err:
        print(f"\nUnexpected error during segmentation: {err}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())

