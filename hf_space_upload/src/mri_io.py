"""NIfTI MRI volume loading, validation, metadata extraction, and saving.

Provides standardized I/O utilities for 3D Brain MRI neuroimaging volumes
using NiBabel and NumPy, strictly maintaining data integrity without
altering raw data files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np

from src.config import load_config
from src.utils import ensure_directory, get_project_root, is_nifti_file, safe_filename, timestamped_filename


def list_raw_nifti_files(
    data_raw_dir: Optional[Union[str, Path]] = None,
) -> List[Path]:
    """List all valid NIfTI files (.nii, .nii.gz) in the raw data directory.

    Args:
        data_raw_dir: Optional path to raw data directory. Defaults to the
            'data_raw' path configured in project_config.yaml.

    Returns:
        List[Path]: Sorted list of paths to found NIfTI volumes.

    Raises:
        FileNotFoundError: If the raw data directory does not exist.
    """
    if data_raw_dir is None:
        cfg = load_config()
        raw_path = cfg.paths.resolve(get_project_root())["data_raw"]
    else:
        raw_path = Path(data_raw_dir).resolve()

    if not raw_path.exists() or not raw_path.is_dir():
        raise FileNotFoundError(f"Raw data directory not found: {raw_path}")

    files: List[Path] = [
        p for p in raw_path.rglob("*")
        if p.is_file() and is_nifti_file(p)
    ]
    return sorted(files, key=lambda p: p.name)


def validate_mri_volume(volume: np.ndarray) -> bool:
    """Validate that an MRI volume array is 3-dimensional and numerically finite.

    Args:
        volume: NumPy ndarray containing volume voxel intensities.

    Returns:
        bool: True if validation succeeds.

    Raises:
        TypeError: If volume is not a NumPy ndarray.
        ValueError: If volume is not 3D, has zero dimension size, or contains
            non-finite numbers (NaN or Inf).
    """
    if not isinstance(volume, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(volume).__name__}")

    if volume.ndim != 3:
        raise ValueError(
            f"MRI volume must be 3-dimensional (X, Y, Z). Received shape: {volume.shape} (ndim={volume.ndim})"
        )

    if any(s <= 0 for s in volume.shape):
        raise ValueError(f"MRI volume dimensions must be > 0. Received shape: {volume.shape}")

    if not np.isfinite(volume).all():
        raise ValueError("MRI volume contains non-finite values (NaN or Inf).")

    return True


def load_nifti_volume(
    path: Union[str, Path],
    dtype: np.dtype = np.float32,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load a 3D NIfTI volume, its affine matrix, and metadata from disk.

    Args:
        path: Path to the .nii or .nii.gz file.
        dtype: Target NumPy numeric type for voxel intensity conversion.

    Returns:
        Tuple containing:
            - np.ndarray: Validated 3D voxel array cast to specified dtype.
            - np.ndarray: 4x4 spatial affine matrix.
            - Dict[str, Any]: Dictionary containing volume metadata.

    Raises:
        FileNotFoundError: If the target file does not exist.
        ValueError: If file is not a valid NIfTI file or fails 3D validation.
    """
    file_path = Path(path).resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"NIfTI file does not exist: {file_path}")

    if not is_nifti_file(file_path):
        raise ValueError(f"File '{file_path.name}' is not a recognized NIfTI format (.nii or .nii.gz).")

    nii_img = nib.load(str(file_path))
    data = np.asanyarray(nii_img.dataobj)

    # Squeeze trailing singleton dimensions if present (e.g. (X, Y, Z, 1) -> (X, Y, Z))
    if data.ndim == 4 and data.shape[-1] == 1:
        data = np.squeeze(data, axis=-1)

    volume = data.astype(dtype, copy=False)
    validate_mri_volume(volume)

    affine = np.array(nii_img.affine, dtype=np.float64)
    metadata = get_volume_metadata(file_path)

    return volume, affine, metadata


def get_volume_metadata(path: Union[str, Path]) -> Dict[str, Any]:
    """Extract structural metadata from a NIfTI file without modifying it.

    Extracts shape, voxel spacing (zooms in mm), data type, and anatomical
    orientation (e.g. 'RAS', 'LPS') if affine orientation is resolvable.

    Args:
        path: Path to .nii or .nii.gz file.

    Returns:
        Dict[str, Any]: Extracted metadata attributes.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a NIfTI file.
    """
    file_path = Path(path).resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"NIfTI file does not exist: {file_path}")

    if not is_nifti_file(file_path):
        raise ValueError(f"File '{file_path.name}' is not a recognized NIfTI format (.nii or .nii.gz).")

    nii_img = nib.load(str(file_path))
    header = nii_img.header

    # Extract voxel spacing (zooms)
    zooms = header.get_zooms()
    spacing = tuple(float(z) for z in zooms[:3]) if len(zooms) >= 3 else tuple(float(z) for z in zooms)

    # Extract orientation codes (e.g. 'RAS', 'LPS')
    orientation: Optional[str] = None
    try:
        axcodes = nib.aff2axcodes(nii_img.affine)
        orientation = "".join(axcodes)
    except Exception:
        orientation = "UNKNOWN"

    shape = tuple(int(s) for s in nii_img.shape)
    data_dtype = str(header.get_data_dtype())

    return {
        "filename": file_path.name,
        "path": str(file_path),
        "shape": shape,
        "ndim": len(shape),
        "spacing_mm": spacing,
        "dtype": data_dtype,
        "orientation": orientation,
        "affine": nii_img.affine.tolist(),
        "is_3d": len(shape) == 3 or (len(shape) == 4 and shape[-1] == 1),
    }


def save_nifti_volume(
    volume: np.ndarray,
    affine: Optional[np.ndarray],
    output_path: Union[str, Path],
    validate_3d: bool = True,
) -> Path:
    """Save a NumPy array as a NIfTI volume to the specified output destination.

    Args:
        volume: 3D or 4D NumPy array of voxel intensities.
        affine: 4x4 spatial affine transformation matrix. If None, np.eye(4) is used.
        output_path: Destination file path (.nii or .nii.gz).
        validate_3d: If True, enforces strict 3D dimensionality. If False, allows 4D volumes.

    Returns:
        Path: Resolved absolute path to the saved NIfTI file.

    Raises:
        ValueError: If volume fails validation or output path is invalid.
    """
    if validate_3d:
        validate_mri_volume(volume)
    else:
        if not isinstance(volume, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(volume).__name__}")
        if not np.isfinite(volume).all():
            raise ValueError("MRI volume contains non-finite values (NaN or Inf).")

    target_path = Path(output_path).resolve()
    if not is_nifti_file(target_path):
        raise ValueError(f"Destination '{target_path.name}' must have a .nii or .nii.gz extension.")

    ensure_directory(target_path.parent)

    aff = np.eye(4, dtype=np.float64) if affine is None else np.asarray(affine, dtype=np.float64)
    nii_obj = nib.Nifti1Image(volume, aff)
    nib.save(nii_obj, str(target_path))

    return target_path


def save_processed_volume(
    volume: np.ndarray,
    affine: Optional[np.ndarray],
    output_path: Optional[Union[str, Path]] = None,
    filename: Optional[str] = None,
    validate_3d: Optional[bool] = None,
) -> Path:
    """Save a validated or lightly preprocessed MRI volume in data/processed/.

    Ensures data/raw/ is never overwritten or mutated. If output_path is omitted,
    stores under data/processed/ with either the provided filename or a
    timestamped default.

    Args:
        volume: 3D or 4D NumPy array of voxel intensities.
        affine: 4x4 spatial affine transformation matrix.
        output_path: Optional full destination path.
        filename: Optional filename within data/processed/.
        validate_3d: If True, enforces 3D. If False, allows 4D. If None, auto-detected from volume.ndim.

    Returns:
        Path: Resolved absolute path to saved processed volume.
    """
    cfg = load_config()
    proc_dir = cfg.paths.resolve(get_project_root())["data_processed"]
    ensure_directory(proc_dir)

    if output_path is not None:
        destination = Path(output_path).resolve()
    elif filename is not None:
        destination = proc_dir / safe_filename(filename)
        if not is_nifti_file(destination):
            destination = destination.with_suffix(".nii.gz")
    else:
        destination = proc_dir / timestamped_filename("processed_mri", "nii.gz")

    # Safety check: Prevent writing to data/raw
    raw_dir = cfg.paths.resolve(get_project_root())["data_raw"]
    if destination.parent == raw_dir or raw_dir in destination.parents:
        raise PermissionError(
            f"Direct write to raw data directory '{raw_dir}' is strictly forbidden."
        )

    should_validate_3d = (volume.ndim == 3) if validate_3d is None else validate_3d
    return save_nifti_volume(volume, affine, destination, validate_3d=should_validate_3d)

