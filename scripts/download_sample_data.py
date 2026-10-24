#!/usr/bin/env python3
"""CLI utility to download and prepare 25 public brain MRI research cases.

Downloads cases BRATS_001 through BRATS_025 from the Medical Segmentation Decathlon
Task01 BrainTumour dataset for research and inference testing.

Places:
  - MRI volumes into: data/raw/Task01_BrainTumour/imagesTr/
  - Reference label masks into: data/annotations/Task01_BrainTumour/labelsTr/

DISCLAIMER: AI-generated research draft -- not a medical diagnosis.
Research and demonstration use only.
"""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import urllib.request

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.mri_io import is_nifti_file
from src.utils import ensure_directory


DEFAULT_DATASET_URL = (
    "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download and prepare 25 public brain MRI research cases (Task01 BrainTumour)."
    )
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_DATASET_URL,
        help="Download URL for Task01_BrainTumour.tar archive.",
    )
    parser.add_argument(
        "--archive-path",
        type=str,
        default=None,
        help="Optional path to existing local Task01_BrainTumour.tar file.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=25,
        help="Maximum number of training cases to extract (default: 25 for BRATS_001 to BRATS_025).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory for MRI volumes (default: data/raw).",
    )
    parser.add_argument(
        "--annotations-dir",
        type=str,
        default="data/annotations",
        help="Output directory for reference label masks (default: data/annotations).",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the downloaded tar archive in data/downloads/ after extraction.",
    )
    return parser.parse_args()


class TeeStream(io.RawIOBase):
    """File-like wrapper that reads from an underlying stream and tees bytes to a sink file."""

    def __init__(self, source_stream: Any, sink_file: Optional[Any] = None) -> None:
        self.source = source_stream
        self.sink = sink_file

    def read(self, size: int = -1) -> bytes:
        data = self.source.read(size)
        if data and self.sink:
            self.sink.write(data)
        return data

    def readable(self) -> bool:
        return True


def check_disk_space(path: Path, min_required_gb: float = 20.0) -> float:
    """Check available disk space and warn if below threshold.

    Args:
        path: Path on the filesystem to check.
        min_required_gb: Minimum free space threshold in GB.

    Returns:
        float: Available free space in GB.
    """
    total, used, free = shutil.disk_usage(str(path))
    free_gb = free / (1024 ** 3)
    if free_gb < min_required_gb:
        print(
            f"[!] WARNING: Low disk space detected ({free_gb:.2f} GB free). "
            f"At least {min_required_gb:.1f} GB free disk space is recommended.",
            file=sys.stderr,
        )
    return free_gb


def verify_nifti_file(file_path: Path) -> bool:
    """Verify that a file exists, is non-empty, and has a readable NIfTI header.

    Args:
        file_path: Path to the .nii or .nii.gz file.

    Returns:
        bool: True if file exists and has valid NIfTI header.
    """
    if not file_path.is_file() or file_path.stat().st_size == 0:
        return False
    if not is_nifti_file(file_path):
        return False

    try:
        import nibabel as nib
        img = nib.load(str(file_path))
        _ = img.shape
        _ = img.affine
        return True
    except Exception:
        return False


def extract_targeted_cases_from_tar(
    tar: tarfile.TarFile,
    raw_images_dir: Path,
    annotations_labels_dir: Path,
    target_case_ids: List[str],
) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    """Extract specified target case images and labels from a tar archive.

    Args:
        tar: Open TarFile instance.
        raw_images_dir: Destination for MRI volumes (data/raw/Task01_BrainTumour/imagesTr/).
        annotations_labels_dir: Destination for labels (data/annotations/Task01_BrainTumour/labelsTr/).
        target_case_ids: List of case IDs (e.g. ['BRATS_001', ..., 'BRATS_025']).

    Returns:
        Tuple[Dict[str, Path], Dict[str, Path]]: (images_dict, labels_dict) mapping case_id to Path.
    """
    ensure_directory(raw_images_dir)
    ensure_directory(annotations_labels_dir)

    target_set = set(target_case_ids)
    extracted_images: Dict[str, Path] = {}
    extracted_labels: Dict[str, Path] = {}

    while True:
        try:
            member = tar.next()
        except Exception:
            break

        if member is None:
            break

        if not member.isreg() or "/._" in member.name or member.name.startswith("._"):
            continue

        clean_name = member.name
        if clean_name.startswith("Task01_BrainTumour/"):
            rel_path = clean_name[len("Task01_BrainTumour/"):]
        else:
            rel_path = clean_name

        # Extract dataset.json if found
        if rel_path == "dataset.json":
            ds_json_path = raw_images_dir.parent / "dataset.json"
            f_in = tar.extractfile(member)
            if f_in:
                ds_json_path.write_bytes(f_in.read())
            continue

        filename = Path(rel_path).name
        case_id = filename.split(".")[0]

        if case_id not in target_set:
            continue

        # Route images to data/raw and labels to data/annotations
        if "imagesTr/" in rel_path and is_nifti_file(rel_path):
            dest_file = raw_images_dir / filename
            print(f"  -> Extracting MRI Image: {case_id} ({member.size / (1024 * 1024):.2f} MB)...", flush=True)
            f_in = tar.extractfile(member)
            if f_in:
                dest_file.write_bytes(f_in.read())
                extracted_images[case_id] = dest_file

        elif "labelsTr/" in rel_path and is_nifti_file(rel_path):
            dest_file = annotations_labels_dir / filename
            print(f"  -> Extracting Label Mask: {case_id} ({member.size / (1024 * 1024):.2f} MB)...", flush=True)
            f_in = tar.extractfile(member)
            if f_in:
                dest_file.write_bytes(f_in.read())
                extracted_labels[case_id] = dest_file

        # Stop early once all targeted images and labels are collected
        if len(extracted_images) >= len(target_set) and len(extracted_labels) >= len(target_set):
            print("\n[OK] All targeted images and labels successfully extracted. Stopping stream early.")
            break

    return extracted_images, extracted_labels


def print_summary_table(
    target_case_ids: List[str],
    images_dict: Dict[str, Path],
    labels_dict: Dict[str, Path],
) -> None:
    """Print formatted summary table showing case ID, MRI path, label path, and file size."""
    print("\n" + "=" * 95)
    print(f"{'Case ID':<12} | {'MRI Image Path':<38} | {'Label Mask Path':<32} | {'MRI Size':<8}")
    print("-" * 95)

    for case_id in sorted(target_case_ids):
        mri_path = images_dict.get(case_id)
        lbl_path = labels_dict.get(case_id)

        if mri_path and mri_path.exists():
            rel_mri = str(mri_path.relative_to(PROJECT_ROOT))
            mri_size_str = f"{mri_path.stat().st_size / (1024 * 1024):.2f} MB"
        else:
            rel_mri = "MISSING"
            mri_size_str = "N/A"

        if lbl_path and lbl_path.exists():
            rel_lbl = str(lbl_path.relative_to(PROJECT_ROOT))
        else:
            rel_lbl = "MISSING"

        print(f"{case_id:<12} | {rel_mri:<38} | {rel_lbl:<32} | {mri_size_str:<8}")

    print("=" * 95)


def download_from_tar_index(
    index_file: Path,
    base_url: str,
    raw_images_dir: Path,
    annotations_labels_dir: Path,
    target_case_ids: List[str],
) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    """Download targeted files using pre-indexed byte ranges for high speed."""
    import concurrent.futures
    import json

    with open(index_file, "r") as f:
        idx_data = json.load(f)

    images_info = idx_data.get("images", {})
    labels_info = idx_data.get("labels", {})

    extracted_images: Dict[str, Path] = {}
    extracted_labels: Dict[str, Path] = {}

    tasks: List[Tuple[Dict[str, Any], Path, str]] = []
    for cid in target_case_ids:
        if cid in images_info:
            info = dict(images_info[cid])
            info["case_id"] = cid
            dest_p = raw_images_dir / info["filename"]
            tasks.append((info, dest_p, "image"))
        if cid in labels_info:
            info = dict(labels_info[cid])
            info["case_id"] = cid
            dest_p = annotations_labels_dir / info["filename"]
            tasks.append((info, dest_p, "label"))

    def _fetch_one(item_info: Dict[str, Any], dest: Path, kind: str) -> Tuple[str, Path, str, bool]:
        cid = item_info["case_id"]
        size = item_info["size"]
        offset = item_info["data_offset"]
        if dest.is_file() and dest.stat().st_size == size and verify_nifti_file(dest):
            return cid, dest, kind, True

        tmp = dest.with_suffix(dest.suffix + ".tmp")
        req = urllib.request.Request(
            base_url,
            headers={
                "User-Agent": "BrainMRI-SpatialVLM/1.0",
                "Range": f"bytes={offset}-{offset + size - 1}",
            },
        )
        for _ in range(3):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    with open(tmp, "wb") as out:
                        shutil.copyfileobj(resp, out)
                if tmp.stat().st_size == size:
                    tmp.replace(dest)
                    return cid, dest, kind, True
            except Exception:
                if tmp.exists():
                    tmp.unlink()
                time.sleep(1)
        return cid, dest, kind, False

    print(f"\nDownloading cases via fast indexed range requests ({len(tasks)} files)...", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(_fetch_one, info, dest, kind) for info, dest, kind in tasks]
        for f in concurrent.futures.as_completed(futures):
            cid, dest, kind, success = f.result()
            if success:
                if kind == "image":
                    extracted_images[cid] = dest
                else:
                    extracted_labels[cid] = dest

    return extracted_images, extracted_labels


def download_and_prepare_dataset(
    url: str,
    output_dir: Path,
    annotations_dir: Path,
    downloads_dir: Path,
    max_cases: int = 25,
    archive_path: Optional[Path] = None,
    keep_archive: bool = False,
) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    """Download and extract 25 multimodal MRI research cases and labels."""
    ensure_directory(output_dir)
    ensure_directory(annotations_dir)
    ensure_directory(downloads_dir)

    raw_images_dir = output_dir / "Task01_BrainTumour" / "imagesTr"
    annotations_labels_dir = annotations_dir / "Task01_BrainTumour" / "labelsTr"
    ensure_directory(raw_images_dir)
    ensure_directory(annotations_labels_dir)

    target_case_ids = [f"BRATS_{i:03d}" for i in range(1, max_cases + 1)]

    print("=" * 70)
    print("BrainMRI-SpatialVLM: 25-Case Public Research MRI Data Downloader")
    print("DISCLAIMER: AI-generated research draft -- not a medical diagnosis.")
    print("Research testing demonstration only.")
    print("=" * 70)
    print(f"Target MRI Directory   : {raw_images_dir}")
    print(f"Target Label Directory : {annotations_labels_dir}")
    print(f"Target Cases           : {target_case_ids[0]} through {target_case_ids[-1]} ({len(target_case_ids)} cases)")

    # Check disk space
    free_gb = check_disk_space(output_dir, min_required_gb=20.0)
    print(f"Available Disk Space   : {free_gb:.2f} GB (Recommended >= 20.0 GB)")

    # Check if all targeted files already exist on disk and are valid
    existing_images = {
        cid: raw_images_dir / f"{cid}.nii.gz"
        for cid in target_case_ids
        if verify_nifti_file(raw_images_dir / f"{cid}.nii.gz")
    }
    existing_labels = {
        cid: annotations_labels_dir / f"{cid}.nii.gz"
        for cid in target_case_ids
        if verify_nifti_file(annotations_labels_dir / f"{cid}.nii.gz")
    }

    if len(existing_images) == len(target_case_ids) and len(existing_labels) == len(target_case_ids) and archive_path is None:
        print(f"\n[OK] All {len(target_case_ids)} targeted MRI cases and matching labels already exist and are valid.")
        print_summary_table(target_case_ids, existing_images, existing_labels)
        return existing_images, existing_labels

    # Case A: Local archive provided
    if archive_path is not None and Path(archive_path).is_file():
        local_archive = Path(archive_path).resolve()
        print(f"\nReading from local archive: {local_archive}")
        with tarfile.open(local_archive, mode="r:*") as tar:
            images_dict, labels_dict = extract_targeted_cases_from_tar(
                tar,
                raw_images_dir,
                annotations_labels_dir,
                target_case_ids,
            )

    # Case B: Download via pre-indexed byte ranges if available
    elif (downloads_dir / "tar_index.json").is_file() and not keep_archive:
        try:
            images_dict, labels_dict = download_from_tar_index(
                index_file=downloads_dir / "tar_index.json",
                base_url=url,
                raw_images_dir=raw_images_dir,
                annotations_labels_dir=annotations_labels_dir,
                target_case_ids=target_case_ids,
            )
        except Exception as e:
            print(f"[!] Indexed range download failed ({e}), falling back to stream download...")
            archive_dest = downloads_dir / "Task01_BrainTumour.tar"
            print(f"\nConnecting to dataset URL:\n  {url}")
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "BrainMRI-SpatialVLM-Research/1.0"},
            )
            sink_handle = open(archive_dest, "wb") if keep_archive else None
            try:
                with urllib.request.urlopen(req) as resp:
                    stream_obj = TeeStream(resp, sink_handle) if sink_handle else resp
                    with tarfile.open(fileobj=stream_obj, mode="r|*") as tar:
                        images_dict, labels_dict = extract_targeted_cases_from_tar(
                            tar,
                            raw_images_dir,
                            annotations_labels_dir,
                            target_case_ids,
                        )
            finally:
                if sink_handle:
                    sink_handle.close()
                    if not keep_archive and archive_dest.exists():
                        archive_dest.unlink()

    # Case C: Stream from URL
    else:
        archive_dest = downloads_dir / "Task01_BrainTumour.tar"
        print(f"\nConnecting to dataset URL:\n  {url}")
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "BrainMRI-SpatialVLM-Research/1.0"},
        )

        sink_handle = open(archive_dest, "wb") if keep_archive else None

        try:
            with urllib.request.urlopen(req) as resp:
                stream_obj = TeeStream(resp, sink_handle) if sink_handle else resp
                with tarfile.open(fileobj=stream_obj, mode="r|*") as tar:
                    images_dict, labels_dict = extract_targeted_cases_from_tar(
                        tar,
                        raw_images_dir,
                        annotations_labels_dir,
                        target_case_ids,
                    )
        finally:
            if sink_handle:
                sink_handle.close()
                if not keep_archive and archive_dest.exists():
                    archive_dest.unlink()

    # Verify extracted files
    verified_images: Dict[str, Path] = {}
    verified_labels: Dict[str, Path] = {}

    for cid in target_case_ids:
        img_p = images_dict.get(cid) or (raw_images_dir / f"{cid}.nii.gz")
        if img_p.is_file() and verify_nifti_file(img_p):
            verified_images[cid] = img_p

        lbl_p = labels_dict.get(cid) or (annotations_labels_dir / f"{cid}.nii.gz")
        if lbl_p.is_file() and verify_nifti_file(lbl_p):
            verified_labels[cid] = lbl_p

    print_summary_table(target_case_ids, verified_images, verified_labels)
    return verified_images, verified_labels


def main() -> int:
    """CLI Entry point."""
    args = parse_args()
    cfg = load_config()

    url = (
        args.url
        if args.url != DEFAULT_DATASET_URL or not hasattr(cfg, "sample_data")
        else getattr(cfg.sample_data, "download_url", DEFAULT_DATASET_URL)
    )

    out_dir = (PROJECT_ROOT / args.output_dir).resolve()
    ann_dir = (
        PROJECT_ROOT
        / (
            args.annotations_dir
            if args.annotations_dir != "data/annotations" or not hasattr(cfg, "sample_data")
            else getattr(cfg.sample_data, "annotations_dir", "data/annotations")
        )
    ).resolve()

    dl_dir = (
        PROJECT_ROOT
        / getattr(cfg.sample_data, "downloads_dir", "data/downloads")
    ).resolve()

    max_cases = (
        args.max_cases
        if args.max_cases != 25 or not hasattr(cfg, "sample_data")
        else getattr(cfg.sample_data, "max_cases", 25)
    )

    archive_path = Path(args.archive_path).resolve() if args.archive_path else None

    try:
        images, labels = download_and_prepare_dataset(
            url=url,
            output_dir=out_dir,
            annotations_dir=ann_dir,
            downloads_dir=dl_dir,
            max_cases=max_cases,
            archive_path=archive_path,
            keep_archive=args.keep_archive,
        )

        expected_count = max_cases
        if len(images) == expected_count and len(labels) == expected_count:
            print(f"\n[OK] Successfully prepared all {expected_count} MRI cases and matching reference labels.")
            return 0
        else:
            print(
                f"\n[!] Warning: Found {len(images)}/{expected_count} images and {len(labels)}/{expected_count} labels.",
                file=sys.stderr,
            )
            return 1
    except Exception as e:
        print(f"\n[ERROR] Download and extraction failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
