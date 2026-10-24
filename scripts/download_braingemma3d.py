#!/usr/bin/env python3
"""CLI utility to download and organize pretrained BrainGemma3D and foundation weights.

Downloads:
  1. praiselab-picuslab/BrainGemma3D -> models/braingemma3d/braingemma3d_repo/
  2. google/medsiglip-448           -> models/braingemma3d/siglip/
  3. google/medgemma-1.5-4b-it      -> models/braingemma3d/medgemma/
  4. Optional checkpoint copying    -> models/braingemma3d/checkpoints/

Usage:
    python scripts/download_braingemma3d.py
    python scripts/download_braingemma3d.py --checkpoint-dir /path/to/checkpoint
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# Disable experimental Hugging Face Xet/CAS client to prevent "CAS Client Error: error decoding response body"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.utils import ensure_directory


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Download BrainGemma3D repository, MedSigLIP, MedGemma, and link checkpoints."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Optional local path to BrainGemma3D checkpoint (e.g., phase2b_final) to copy into models/braingemma3d/checkpoints/.",
    )
    parser.add_argument(
        "--skip-auth-check",
        action="store_true",
        help="Bypass Hugging Face authentication check.",
    )
    parser.add_argument(
        "--repo-only",
        action="store_true",
        help="Download only the main BrainGemma3D repository (praiselab-picuslab/BrainGemma3D), skipping gated base models.",
    )
    return parser.parse_args()


def is_hf_authenticated() -> bool:
    """Check whether Hugging Face token is present in the environment or cache."""
    # Check environment variable
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True

    # Check huggingface_hub token store
    try:
        from huggingface_hub import get_token
        token = get_token()
        return token is not None and len(token) > 0
    except Exception:
        pass

    # Check standard huggingface token path on disk
    hf_token_path = Path.home() / ".cache" / "huggingface" / "token"
    if hf_token_path.is_file() and hf_token_path.stat().st_size > 0:
        return True

    return False


def print_auth_instructions() -> None:
    """Print clear instructions for authenticating with Hugging Face."""
    print("=" * 65, file=sys.stderr)
    print("HUGGING FACE AUTHENTICATION & LICENSE ACCEPTANCE REQUIRED", file=sys.stderr)
    print("=" * 65, file=sys.stderr)
    print(
        "Access to google/medsiglip-448 and google/medgemma-1.5-4b-it requires\n"
        "an active Hugging Face account. The user must accept the applicable model licenses\n"
        "on their respective Hugging Face model pages before downloading:\n"
        "  - https://huggingface.co/google/medsiglip-448\n"
        "  - https://huggingface.co/google/medgemma-1.5-4b-it\n\n"
        "Please authenticate your local environment by running:\n"
        "    huggingface-cli login\n\n"
        "Enter your Hugging Face User Access Token (read permissions)\n"
        "when prompted, or set the HF_TOKEN environment variable.",
        file=sys.stderr,
    )
    print("=" * 65, file=sys.stderr)


def download_hf_repo(repo_id: str, local_dir: Path, max_retries: int = 5) -> bool:
    """Download a Hugging Face repository snapshot with file-level resume and retry logic.

    Args:
        repo_id: Hugging Face repository identifier.
        local_dir: Local destination path.
        max_retries: Maximum number of retry attempts per file on network drop.

    Returns:
        bool: True if successful, False otherwise.
    """
    import time
    ensure_directory(local_dir)
    print(f"\n[+] Downloading '{repo_id}' into: {local_dir}...")
    try:
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi()
        try:
            repo_files = api.list_repo_files(repo_id)
        except Exception:
            # Fall back to snapshot_download if list_repo_files fails
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
            print(f"[OK] Successfully downloaded '{repo_id}'.")
            return True

        total_files = len(repo_files)
        print(f"[*] Repository has {total_files} files. Verifying local files and resuming...")

        for idx, filename in enumerate(repo_files, 1):
            dest_path = local_dir / filename
            if dest_path.is_file() and dest_path.stat().st_size > 0:
                size_mb = dest_path.stat().st_size / (1024 * 1024)
                print(f"[{idx}/{total_files}] [OK: Exists ({size_mb:.1f} MB)] {filename}")
                continue

            print(f"[{idx}/{total_files}] [DOWNLOADING] {filename}...")
            file_success = False
            for attempt in range(1, max_retries + 1):
                try:
                    ensure_directory(dest_path.parent)
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        local_dir=str(local_dir),
                    )
                    file_success = True
                    print(f"[{idx}/{total_files}] [OK] Finished: {filename}")
                    break
                except Exception as file_err:
                    print(
                        f"    [!] Attempt {attempt}/{max_retries} failed for '{filename}': {file_err}",
                        file=sys.stderr,
                    )
                    if attempt < max_retries:
                        sleep_time = min(5 * attempt, 30)
                        print(f"    [*] Waiting {sleep_time}s before retrying...", file=sys.stderr)
                        time.sleep(sleep_time)

            if not file_success:
                print(f"[ERROR] Failed to download '{filename}' after {max_retries} attempts.", file=sys.stderr)
                return False

        print(f"[OK] Successfully verified and downloaded all files for '{repo_id}'.")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download '{repo_id}': {e}", file=sys.stderr)
        return False


def copy_or_link_checkpoint(source_dir: Path, target_dir: Path) -> bool:
    """Copy or link a BrainGemma3D checkpoint into models/braingemma3d/checkpoints/.

    Args:
        source_dir: Source folder containing checkpoint weights.
        target_dir: Target destination inside models/braingemma3d/checkpoints/.

    Returns:
        bool: True if copied/linked successfully.
    """
    if not source_dir.exists():
        print(f"Error: Source checkpoint directory does not exist: {source_dir}", file=sys.stderr)
        return False

    ensure_directory(target_dir)
    print(f"[+] Copying checkpoint files from {source_dir} to {target_dir}...")
    try:
        for item in source_dir.iterdir():
            dest = target_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)
        print(f"[OK] Checkpoint installed into: {target_dir}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to copy checkpoint: {e}", file=sys.stderr)
        return False


def main() -> int:
    args = parse_args()
    cfg = load_config()

    # Step 1: Authentication check
    if not args.skip_auth_check and not args.repo_only and not is_hf_authenticated():
        print_auth_instructions()
        return 1

    repo_dir = (PROJECT_ROOT / cfg.braingemma3d.repo_dir).resolve()
    siglip_dir = (PROJECT_ROOT / cfg.braingemma3d.siglip_dir).resolve()
    medgemma_dir = (PROJECT_ROOT / cfg.braingemma3d.medgemma_dir).resolve()
    checkpoint_dir = (PROJECT_ROOT / cfg.braingemma3d.checkpoint_dir).resolve()

    print("=" * 65)
    print("BrainGemma3D Model Downloader and Environment Setup")
    print("=" * 65)
    print(f"Target BrainGemma3D Repo : {repo_dir}")
    print(f"Target MedSigLIP Dir     : {siglip_dir}")
    print(f"Target MedGemma Dir      : {medgemma_dir}")
    print(f"Target Checkpoint Dir    : {checkpoint_dir}")

    # 1. Download BrainGemma3D code/repo
    ok1 = download_hf_repo("praiselab-picuslab/BrainGemma3D", repo_dir)

    if args.repo_only:
        if ok1:
            print("\n[OK] BrainGemma3D repository downloaded successfully (--repo-only mode).")
            return 0
        else:
            print("\n[!] BrainGemma3D repository download failed.", file=sys.stderr)
            return 2

    # 2. Download MedSigLIP vision encoder
    ok2 = download_hf_repo("google/medsiglip-448", siglip_dir)

    # 3. Download MedGemma language model
    ok3 = download_hf_repo("google/medgemma-1.5-4b-it", medgemma_dir)

    # 4. Optional checkpoint linking
    if args.checkpoint_dir:
        src_chk = Path(args.checkpoint_dir).resolve()
        copy_or_link_checkpoint(src_chk, checkpoint_dir)

    if ok1 and ok2 and ok3:
        print("\n[OK] All BrainGemma3D components downloaded successfully.")
        return 0
    else:
        print("\n[!] One or more downloads failed. Check permissions and authentication.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
