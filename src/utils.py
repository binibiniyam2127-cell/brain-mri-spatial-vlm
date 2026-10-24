"""Shared utility functions for BrainMRI-SpatialVLM.

Provides filesystem helpers, path resolution, NIfTI validation,
and filename sanitation utilities.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Union


def get_project_root() -> Path:
    """Return the absolute Path to the project root directory.

    Resolves from the location of this module (`src/utils.py`),
    walking upward to locate known repository anchor files.

    Returns:
        Path: Absolute path to the repository root directory.
    """
    candidate = Path(__file__).resolve().parent.parent
    # Verify expected anchors
    if (candidate / "configs" / "project_config.yaml").exists() or (
        candidate / "requirements.txt"
    ).exists():
        return candidate
    return candidate


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating parents if necessary.

    Args:
        path: Path or string representing the directory to create.

    Returns:
        Path: Resolved absolute Path of the verified directory.
    """
    dir_path = Path(path).resolve()
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def is_nifti_file(path: Union[str, Path]) -> bool:
    """Check whether a path or filename has a valid NIfTI extension.

    Supports both single-extension (.nii) and compressed (.nii.gz) files,
    evaluated in a case-insensitive manner.

    Args:
        path: File path or filename string.

    Returns:
        bool: True if file is .nii or .nii.gz, False otherwise.
    """
    filename = Path(path).name.lower().strip()
    return filename.endswith(".nii") or filename.endswith(".nii.gz")


def safe_filename(name: str) -> str:
    """Sanitize a filename by replacing unsafe characters and whitespace.

    Strips directory traversals, path separators, and filesystem-reserved
    characters (such as < > : " / \\ | ? *), preserving alphanumeric characters,
    underscores, hyphens, and periods.

    Args:
        name: Proposed filename or label.

    Returns:
        str: Sanitized, filesystem-safe filename string.
    """
    # Extract only the basename if a full path was passed
    clean = Path(name).name.strip()

    # Replace whitespace sequences with a single underscore
    clean = re.sub(r"\s+", "_", clean)

    # Remove characters not permitted in common file systems
    clean = re.sub(r'[^A-Za-z0-9_.-]', "", clean)

    # Avoid leading dots or spaces
    clean = clean.lstrip(". ")

    if not clean:
        return "unnamed"
    return clean


def timestamped_filename(
    prefix: str,
    extension: str,
    dt: datetime | None = None,
) -> str:
    """Generate a sanitized filename formatted with a timestamp.

    Format: `{sanitized_prefix}_{YYYYMMDD_HHMMSS}.{extension}`

    Args:
        prefix: Informative prefix for the file (e.g., 'tumor_segmentation').
        extension: File extension (with or without leading dot, e.g. 'nii.gz' or '.json').
        dt: Optional datetime object. If None, current local time is used.

    Returns:
        str: Formatted timestamped filename.
    """
    sanitized_pfx = safe_filename(prefix)
    clean_ext = extension.strip().lstrip(".")

    now = dt if dt is not None else datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")

    if clean_ext:
        return f"{sanitized_pfx}_{timestamp_str}.{clean_ext}"
    return f"{sanitized_pfx}_{timestamp_str}"

