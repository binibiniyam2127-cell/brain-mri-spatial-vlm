"""Shared utility functions for BrainMRI-SpatialVLM.

Provides filesystem helpers, path resolution, NIfTI validation,
and filename sanitation utilities.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import logging
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Optional, Union


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


def compute_file_sha256(path: Union[str, Path], chunk_size: int = 65536) -> str:
    """Compute hex SHA-256 digest of a file in streaming chunks.

    Args:
        path: Path to the target file.
        chunk_size: Size in bytes of stream read buffers.

    Returns:
        str: Lowercase 64-character hexadecimal SHA-256 digest.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"File not found for cryptographic hashing: {p}")

    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_git_commit_sha(repo_dir: Optional[Union[str, Path]] = None) -> Optional[str]:
    """Retrieve the current Git commit SHA-1 if inside a git repository.

    Args:
        repo_dir: Optional repository path (defaults to project root).

    Returns:
        Optional[str]: 40-character commit SHA string, or None if unavailable.
    """
    base = Path(repo_dir).resolve() if repo_dir else get_project_root()
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(base),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return None


class StructuredLogAdapter(logging.LoggerAdapter):
    """Logger adapter that guarantees 'run_id' and 'stage' context fields."""

    def process(self, msg: str, kwargs: Any) -> tuple[str, Any]:
        extra = kwargs.setdefault("extra", {})
        if "run_id" not in extra:
            extra["run_id"] = self.extra.get("run_id", "N/A")
        if "stage" not in extra:
            extra["stage"] = self.extra.get("stage", "general")
        return msg, kwargs


def get_pipeline_logger(
    name: str = "brain_mri_pipeline",
    run_id: str = "N/A",
    stage: str = "init",
) -> StructuredLogAdapter:
    """Return a configured logger adapter with run_id and stage context.

    Format: `%(asctime)s [%(levelname)s] [run_id=%(run_id)s] [%(stage)s]: %(message)s`

    Args:
        name: Logger name.
        run_id: Unique execution identifier.
        stage: Pipeline stage name.

    Returns:
        StructuredLogAdapter: Adapter wrapping logging.Logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] [run_id=%(run_id)s] [%(stage)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.propagate = False
    return StructuredLogAdapter(logger, {"run_id": run_id, "stage": stage})


