#!/usr/bin/env python3
"""CLI utility to download the pretrained MONAI BraTS segmentation model bundle.

Usage:
    python scripts/download_monai_brats.py
    python scripts/download_monai_brats.py --bundle_name brats_mri_segmentation --bundle_dir models/monai_brats
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.utils import ensure_directory


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download pretrained MONAI BraTS 3D MRI segmentation bundle."
    )
    parser.add_argument(
        "--bundle_name",
        type=str,
        default="brats_mri_segmentation",
        help="Name of MONAI bundle to download (default: brats_mri_segmentation).",
    )
    parser.add_argument(
        "--bundle_dir",
        type=str,
        default="models/monai_brats",
        help="Target directory to store bundle (default: models/monai_brats).",
    )
    return parser.parse_args()


def verify_bundle_exists(bundle_dir: Path, bundle_name: str) -> bool:
    """Verify that the bundle directory and model weights exist after download.

    Args:
        bundle_dir: Parent bundle directory.
        bundle_name: Name of the bundle (e.g. brats_mri_segmentation).

    Returns:
        bool: True if bundle directory and model weights exist, False otherwise.
    """
    target_bundle_folder = bundle_dir / bundle_name
    possible_weight_paths = [
        target_bundle_folder / "models" / "model.pt",
        target_bundle_folder / "models" / "model.pth",
        bundle_dir / "models" / "model.pt",
        bundle_dir / "model.pt",
        bundle_dir / "best_metric_model.pt",
    ]

    has_folder = target_bundle_folder.exists() or bundle_dir.exists()
    has_weights = any(p.is_file() for p in possible_weight_paths)

    return has_folder and has_weights


def download_bundle(bundle_name: str, bundle_dir: Path) -> int:
    """Execute bundle download using MONAI bundle CLI.

    Runs:
        python -m monai.bundle download brats_mri_segmentation --bundle_dir models/monai_brats

    Args:
        bundle_name: Identifier of the MONAI bundle.
        bundle_dir: Destination directory.

    Returns:
        int: Process exit code.
    """
    ensure_directory(bundle_dir)

    print("=" * 65)
    print("MONAI BraTS Pretrained Bundle Downloader")
    print("=" * 65)
    print(f"Bundle Name : {bundle_name}")
    print(f"Bundle Dir  : {bundle_dir}")
    print("Starting download via MONAI bundle CLI...\n")

    cmd = [
        sys.executable,
        "-m",
        "monai.bundle",
        "download",
        bundle_name,
        "--bundle_dir",
        str(bundle_dir),
    ]

    cmd_str = f"python -m monai.bundle download {bundle_name} --bundle_dir {bundle_dir}"

    try:
        # First attempt programmatic download via MONAI Python API (does not require external CLI 'fire')
        try:
            from monai.bundle import download as monai_download
            monai_download(name=bundle_name, bundle_dir=str(bundle_dir), progress=True)
        except Exception as py_err:
            print(f"[Info] Python API download notice ({py_err}). Trying CLI fallback...")
            subprocess.run(cmd, check=True)

        # Verify that bundle directory and weights exist
        if verify_bundle_exists(bundle_dir, bundle_name):
            print(f"\n[OK] Verification successful: MONAI bundle and weights exist in {bundle_dir}")
            return 0
        else:
            print(
                f"\n[!] Warning: Download completed, but expected model weights were not found in {bundle_dir}.",
                file=sys.stderr,
            )
            print(f"To retry downloading the bundle, execute:\n    {cmd_str}", file=sys.stderr)
            return 1

    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Failed to download MONAI bundle: exit code {e.returncode}", file=sys.stderr)
        print(f"To retry downloading the bundle, execute:\n    {cmd_str}", file=sys.stderr)
        return e.returncode
    except Exception as e:
        print(f"\n[ERROR] Error during bundle download: {e}", file=sys.stderr)
        print(f"To retry downloading the bundle, execute:\n    {cmd_str}", file=sys.stderr)
        return 1


def main() -> int:
    args = parse_args()
    cfg = load_config()

    bundle_name = (
        args.bundle_name
        if args.bundle_name != "brats_mri_segmentation" or not hasattr(cfg, "monai_brats")
        else getattr(cfg.monai_brats, "bundle_name", "brats_mri_segmentation")
    )

    if args.bundle_dir != "models/monai_brats":
        bundle_dir = Path(args.bundle_dir).resolve()
    else:
        rel_dir = getattr(cfg.monai_brats, "bundle_dir", "models/monai_brats")
        bundle_dir = (PROJECT_ROOT / rel_dir).resolve()

    return download_bundle(bundle_name, bundle_dir)


if __name__ == "__main__":
    sys.exit(main())
