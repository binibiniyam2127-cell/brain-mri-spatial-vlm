"""Configuration module for BrainMRI-SpatialVLM.

Loads and validates system, path, and inference configurations from
project YAML configuration files with support for typed accessors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union


REQUIRED_PATH_KEYS = [
    "data_raw",
    "data_processed",
    "data_annotations",
    "model_monai_brats",
    "model_braingemma3d",
    "output_masks",
    "output_reports",
    "output_visualizations",
]


@dataclass
class SystemConfig:
    """System-level metadata and configuration flags."""

    name: str = "BrainMRI-SpatialVLM"
    description: str = ""
    disclaimer: str = "AI-generated research draft — not a medical diagnosis."
    mode: str = "inference"


@dataclass
class PathsConfig:
    """Path configuration for data, models, and generated outputs."""

    data_raw: str
    data_processed: str
    data_annotations: str
    model_monai_brats: str
    model_braingemma3d: str
    output_masks: str
    output_reports: str
    output_visualizations: str

    def resolve(self, base_dir: Path) -> Dict[str, Path]:
        """Resolve all relative paths against a provided base directory.

        Args:
            base_dir: Directory against which to resolve relative paths.

        Returns:
            Dictionary mapping path key to resolved absolute Path object.
        """
        return {
            key: (base_dir / getattr(self, key)).resolve()
            for key in REQUIRED_PATH_KEYS
        }


@dataclass
class InferenceConfig:
    """Inference settings including compute device and supported file types."""

    device: str = "cuda"
    allowed_formats: List[str] = field(
        default_factory=lambda: [".nii", ".nii.gz"]
    )


@dataclass
class MonaiBratsConfig:
    """Configuration for MONAI BraTS bundle download and inference."""

    bundle_dir: str = "models/monai_brats"
    bundle_name: str = "brats_mri_segmentation"


@dataclass
class BrainGemma3DConfig:
    """Configuration for BrainGemma3D models, dependencies, and checkpoints."""

    repo_dir: str = "models/braingemma3d/braingemma3d_repo"
    siglip_dir: str = "models/braingemma3d/siglip"
    medgemma_dir: str = "models/braingemma3d/medgemma"
    checkpoint_dir: str = "models/braingemma3d/checkpoints/exp1/phase2b_final"


@dataclass
class SampleDataConfig:
    """Configuration for downloading sample public research MRI data."""

    download_url: str = (
        "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
    )
    raw_data_dir: str = "data/raw"
    annotations_dir: str = "data/annotations"
    downloads_dir: str = "data/downloads"
    max_cases: int = 25


@dataclass
class ProjectConfig:
    """Typed container for all project configuration domains."""

    system: SystemConfig
    paths: PathsConfig
    inference: InferenceConfig
    monai_brats: MonaiBratsConfig = field(default_factory=MonaiBratsConfig)
    braingemma3d: BrainGemma3DConfig = field(default_factory=BrainGemma3DConfig)
    sample_data: SampleDataConfig = field(default_factory=SampleDataConfig)
    raw_dict: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, item: str) -> Any:
        """Allow dict-style subscripting for backward-compatible access."""
        return self.raw_dict[item]

    def get(self, key: str, default: Any = None) -> Any:
        """Allow dict-style get access."""
        return self.raw_dict.get(key, default)

    def validate_runtime_environment(self) -> Dict[str, Any]:
        """Validate device availability and gracefully fall back to CPU if needed.

        Returns:
            Dict[str, Any]: Status summary of runtime device and warnings.
        """
        report = {"status": "ok", "device": self.inference.device, "warnings": []}
        try:
            import torch

            if self.inference.device.lower().startswith("cuda") and not torch.cuda.is_available():
                report["warnings"].append(
                    f"CUDA compute device requested ('{self.inference.device}'), but torch.cuda.is_available() is False. "
                    "Gracefully falling back to CPU."
                )
                self.inference.device = "cpu"
                report["device"] = "cpu"
        except ImportError:
            self.inference.device = "cpu"
            report["device"] = "cpu"

        return report


def _parse_yaml_fallback(content: str) -> Dict[str, Any]:
    """Parse basic YAML structure when PyYAML is not installed.

    Supports nested dictionaries, string/number values, and list items.

    Args:
        content: Raw YAML string content.

    Returns:
        Parsed dictionary structure.
    """
    result: Dict[str, Any] = {}
    current_section: Optional[str] = None
    current_list_key: Optional[str] = None

    for line in content.splitlines():
        # Remove comments and trailing whitespace
        line_clean = re.sub(r"#.*$", "", line).rstrip()
        if not line_clean.strip():
            continue

        indent = len(line_clean) - len(line_clean.lstrip())
        stripped = line_clean.strip()

        # Top-level section
        if indent == 0 and stripped.endswith(":"):
            current_section = stripped[:-1].strip()
            result[current_section] = {}
            current_list_key = None
            continue

        # Sub-level key-value
        if current_section is not None:
            if stripped.startswith("- ") and current_list_key is not None:
                val = stripped[2:].strip().strip("\"'")
                if not isinstance(result[current_section][current_list_key], list):
                    result[current_section][current_list_key] = []
                result[current_section][current_list_key].append(val)
                continue

            if ":" in stripped:
                key, val = stripped.split(":", 1)
                key = key.strip()
                val = val.strip()

                if not val:
                    # Potential list or empty mapping
                    result[current_section][key] = []
                    current_list_key = key
                else:
                    current_list_key = None
                    # Strip quotes
                    if (val.startswith('"') and val.endswith('"')) or (
                        val.startswith("'") and val.endswith("'")
                    ):
                        val = val[1:-1]
                    result[current_section][key] = val

    return result


def _read_yaml(file_path: Path) -> Dict[str, Any]:
    """Read a YAML configuration file with PyYAML or built-in fallback.

    Args:
        file_path: Path to the YAML file.

    Returns:
        Dictionary representation of YAML content.
    """
    text = file_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        parsed = yaml.safe_load(text)
        if isinstance(parsed, dict):
            return parsed
        return {}
    except ImportError:
        return _parse_yaml_fallback(text)


def validate_path_keys(paths_dict: Dict[str, Any]) -> None:
    """Validate that all mandatory path keys exist in configuration.

    Args:
        paths_dict: Dictionary containing path configuration entries.

    Raises:
        KeyError: If any mandatory path key is missing.
    """
    missing = [key for key in REQUIRED_PATH_KEYS if key not in paths_dict]
    if missing:
        raise KeyError(
            f"Configuration missing required path key(s): {', '.join(missing)}"
        )


def load_config(
    config_path: Optional[Union[str, Path]] = None,
) -> ProjectConfig:
    """Load, validate, and return the system configuration.

    Args:
        config_path: Optional explicit path to configuration YAML file.
            If None, defaults to 'configs/project_config.yaml' relative to
            the repository root.

    Returns:
        ProjectConfig dataclass populated with parsed values.

    Raises:
        FileNotFoundError: If configuration file does not exist.
        KeyError: If mandatory configuration sections or path keys are missing.
    """
    if config_path is None:
        from src.utils import get_project_root

        resolved_path = get_project_root() / "configs" / "project_config.yaml"
    else:
        resolved_path = Path(config_path).resolve()

    if not resolved_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {resolved_path}")

    raw_dict = _read_yaml(resolved_path)

    if "paths" not in raw_dict or not isinstance(raw_dict["paths"], dict):
        raise KeyError("Configuration file missing mandatory 'paths' section.")

    paths_dict = raw_dict["paths"]
    validate_path_keys(paths_dict)

    sys_dict = raw_dict.get("system", {})
    inf_dict = raw_dict.get("inference", {})

    system_config = SystemConfig(
        name=str(sys_dict.get("name", "BrainMRI-SpatialVLM")),
        description=str(sys_dict.get("description", "")),
        disclaimer=str(
            sys_dict.get(
                "disclaimer",
                "AI-generated research draft — not a medical diagnosis.",
            )
        ),
        mode=str(sys_dict.get("mode", "inference")),
    )

    paths_config = PathsConfig(
        data_raw=str(paths_dict["data_raw"]),
        data_processed=str(paths_dict["data_processed"]),
        data_annotations=str(paths_dict["data_annotations"]),
        model_monai_brats=str(paths_dict["model_monai_brats"]),
        model_braingemma3d=str(paths_dict["model_braingemma3d"]),
        output_masks=str(paths_dict["output_masks"]),
        output_reports=str(paths_dict["output_reports"]),
        output_visualizations=str(paths_dict["output_visualizations"]),
    )

    allowed_formats = inf_dict.get("allowed_formats", [".nii", ".nii.gz"])
    if isinstance(allowed_formats, str):
        allowed_formats = [allowed_formats]

    inference_config = InferenceConfig(
        device=str(inf_dict.get("device", "cuda")),
        allowed_formats=list(allowed_formats),
    )

    brats_dict = raw_dict.get("monai_brats", {})
    monai_brats_config = MonaiBratsConfig(
        bundle_dir=str(brats_dict.get("bundle_dir", "models/monai_brats")),
        bundle_name=str(brats_dict.get("bundle_name", "brats_mri_segmentation")),
    )

    bg_dict = raw_dict.get("braingemma3d", {})
    braingemma3d_config = BrainGemma3DConfig(
        repo_dir=str(bg_dict.get("repo_dir", "models/braingemma3d/braingemma3d_repo")),
        siglip_dir=str(bg_dict.get("siglip_dir", "models/braingemma3d/siglip")),
        medgemma_dir=str(bg_dict.get("medgemma_dir", "models/braingemma3d/medgemma")),
        checkpoint_dir=str(
            bg_dict.get(
                "checkpoint_dir",
                "models/braingemma3d/checkpoints/exp1/phase2b_final",
            )
        ),
    )

    sd_dict = raw_dict.get("sample_data", {})
    sample_data_config = SampleDataConfig(
        download_url=str(
            sd_dict.get(
                "download_url",
                "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar",
            )
        ),
        raw_data_dir=str(sd_dict.get("raw_data_dir", "data/raw")),
        annotations_dir=str(sd_dict.get("annotations_dir", "data/annotations")),
        downloads_dir=str(sd_dict.get("downloads_dir", "data/downloads")),
        max_cases=int(sd_dict.get("max_cases", 25)),
    )

    cfg = ProjectConfig(
        system=system_config,
        paths=paths_config,
        inference=inference_config,
        monai_brats=monai_brats_config,
        braingemma3d=braingemma3d_config,
        sample_data=sample_data_config,
        raw_dict=raw_dict,
    )
    cfg.validate_runtime_environment()
    return cfg

