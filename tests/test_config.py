"""Tests for src/config.py module."""

from pathlib import Path
import tempfile
import unittest

from src.config import (
    REQUIRED_PATH_KEYS,
    ProjectConfig,
    load_config,
    validate_path_keys,
)
from src.utils import get_project_root


class TestConfig(unittest.TestCase):
    """Test suite for configuration loading and validation."""

    def setUp(self) -> None:
        self.project_root = get_project_root()
        self.default_config_file = (
            self.project_root / "configs" / "project_config.yaml"
        )

    def test_load_default_config(self) -> None:
        """Verify default configuration loads correctly."""
        config = load_config(self.default_config_file)
        self.assertIsInstance(config, ProjectConfig)
        self.assertEqual(config.system.name, "BrainMRI-SpatialVLM")
        self.assertEqual(config.system.mode, "inference")
        self.assertIn(
            "AI-generated research draft — not a medical diagnosis.",
            config.system.disclaimer,
        )
        self.assertEqual(config.monai_brats.bundle_dir, "models/monai_brats")
        self.assertEqual(config.monai_brats.bundle_name, "brats_mri_segmentation")
        self.assertEqual(config.braingemma3d.repo_dir, "models/braingemma3d/braingemma3d_repo")
        self.assertEqual(config.braingemma3d.siglip_dir, "models/braingemma3d/siglip")
        self.assertEqual(config.braingemma3d.medgemma_dir, "models/braingemma3d/medgemma")
        self.assertEqual(config.braingemma3d.checkpoint_dir, "models/braingemma3d/checkpoints/exp1/phase2b_final")

    def test_required_path_keys_exist(self) -> None:
        """Verify that all required path keys are loaded onto paths object."""
        config = load_config(self.default_config_file)
        for key in REQUIRED_PATH_KEYS:
            self.assertTrue(
                hasattr(config.paths, key),
                f"PathsConfig missing attribute: {key}",
            )
            val = getattr(config.paths, key)
            self.assertIsInstance(val, str)
            self.assertGreater(len(val), 0)

    def test_paths_resolve_method(self) -> None:
        """Verify paths.resolve returns a mapping of resolved Path objects."""
        config = load_config(self.default_config_file)
        resolved = config.paths.resolve(self.project_root)
        self.assertEqual(len(resolved), len(REQUIRED_PATH_KEYS))
        for key, path_obj in resolved.items():
            self.assertIsInstance(path_obj, Path)
            self.assertTrue(path_obj.is_absolute())

    def test_dict_subscripting_and_get(self) -> None:
        """Verify dict-style access works on ProjectConfig."""
        config = load_config(self.default_config_file)
        self.assertEqual(config["system"]["name"], "BrainMRI-SpatialVLM")
        self.assertIn("data_raw", config["paths"])
        self.assertEqual(config.get("system")["mode"], "inference")
        self.assertIsNone(config.get("nonexistent_key"))

    def test_missing_path_key_validation(self) -> None:
        """Verify validate_path_keys raises KeyError when keys are missing."""
        incomplete_paths = {"data_raw": "data/raw"}
        with self.assertRaises(KeyError) as ctx:
            validate_path_keys(incomplete_paths)
        self.assertIn("missing required path key(s)", str(ctx.exception))

    def test_missing_config_file_raises_error(self) -> None:
        """Verify FileNotFoundError is raised when file does not exist."""
        non_existent = self.project_root / "configs" / "does_not_exist.yaml"
        with self.assertRaises(FileNotFoundError):
            load_config(non_existent)

    def test_custom_yaml_loading(self) -> None:
        """Verify loading an explicitly passed valid YAML file."""
        custom_yaml_content = """
system:
  name: "BrainMRI-SpatialVLM-Test"
  disclaimer: "AI-generated research draft — not a medical diagnosis."
  mode: "inference"
paths:
  data_raw: "tmp/data/raw"
  data_processed: "tmp/data/processed"
  data_annotations: "tmp/data/annotations"
  model_monai_brats: "tmp/models/monai_brats"
  model_braingemma3d: "tmp/models/braingemma3d"
  output_masks: "tmp/outputs/masks"
  output_reports: "tmp/outputs/reports"
  output_visualizations: "tmp/outputs/visualizations"
inference:
  device: "cpu"
  allowed_formats:
    - ".nii"
    - ".nii.gz"
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(custom_yaml_content)
            tmp_path = Path(tmp.name)

        try:
            cfg = load_config(tmp_path)
            self.assertEqual(cfg.system.name, "BrainMRI-SpatialVLM-Test")
            self.assertEqual(cfg.inference.device, "cpu")
            self.assertEqual(cfg.paths.data_raw, "tmp/data/raw")
            self.assertEqual(cfg.inference.allowed_formats, [".nii", ".nii.gz"])
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


if __name__ == "__main__":
    unittest.main()

