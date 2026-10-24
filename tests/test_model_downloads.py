"""Tests for model download scripts and model configuration integrity.

Verifies:
  1. The download scripts exist.
  2. The configuration contains the required model paths.
  3. The scripts contain the expected repository names and authentication instructions.
  4. Tests run strictly offline without downloading weights.
"""

from pathlib import Path
import unittest

from src.config import load_config
from src.utils import get_project_root


class TestModelDownloads(unittest.TestCase):
    """Test suite for verifying download scripts and model configurations."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = get_project_root()
        cls.scripts_dir = cls.project_root / "scripts"
        cls.config_file = cls.project_root / "configs" / "project_config.yaml"

    def test_download_scripts_exist(self) -> None:
        """Verify that download scripts exist and are non-empty."""
        monai_script = self.scripts_dir / "download_monai_brats.py"
        braingemma_script = self.scripts_dir / "download_braingemma3d.py"

        self.assertTrue(monai_script.is_file(), f"Missing script: {monai_script}")
        self.assertGreater(monai_script.stat().st_size, 0)

        self.assertTrue(braingemma_script.is_file(), f"Missing script: {braingemma_script}")
        self.assertGreater(braingemma_script.stat().st_size, 0)

    def test_config_contains_required_model_paths(self) -> None:
        """Verify project_config.yaml specifies all mandatory model bundle and repo paths."""
        cfg = load_config(self.config_file)

        # monai_brats paths
        self.assertTrue(hasattr(cfg, "monai_brats"))
        self.assertEqual(cfg.monai_brats.bundle_dir, "models/monai_brats")
        self.assertEqual(cfg.monai_brats.bundle_name, "brats_mri_segmentation")

        # braingemma3d paths
        self.assertTrue(hasattr(cfg, "braingemma3d"))
        self.assertEqual(cfg.braingemma3d.repo_dir, "models/braingemma3d/braingemma3d_repo")
        self.assertEqual(cfg.braingemma3d.siglip_dir, "models/braingemma3d/siglip")
        self.assertEqual(cfg.braingemma3d.medgemma_dir, "models/braingemma3d/medgemma")
        self.assertEqual(
            cfg.braingemma3d.checkpoint_dir,
            "models/braingemma3d/checkpoints/exp1/phase2b_final",
        )

    def test_monai_script_expected_repos_and_commands(self) -> None:
        """Verify download_monai_brats.py specifies the required MONAI bundle and commands."""
        monai_script = self.scripts_dir / "download_monai_brats.py"
        content = monai_script.read_text(encoding="utf-8")

        self.assertIn("brats_mri_segmentation", content)
        self.assertIn("models/monai_brats", content)
        self.assertIn("monai.bundle", content)
        self.assertIn("download", content)

    def test_braingemma3d_script_expected_repos_and_auth(self) -> None:
        """Verify download_braingemma3d.py contains all target repository names, checkpoints, and auth guidance."""
        braingemma_script = self.scripts_dir / "download_braingemma3d.py"
        content = braingemma_script.read_text(encoding="utf-8")

        # Required repository targets
        self.assertIn("praiselab-picuslab/BrainGemma3D", content)
        self.assertIn("google/medsiglip-448", content)
        self.assertIn("google/medgemma-1.5-4b-it", content)

        # Checkpoint argument and destination
        self.assertIn("--checkpoint-dir", content)
        self.assertIn("models/braingemma3d", content)

        # Authentication guidance
        self.assertIn("huggingface-cli login", content)
        self.assertIn("license", content.lower())


if __name__ == "__main__":
    unittest.main()

