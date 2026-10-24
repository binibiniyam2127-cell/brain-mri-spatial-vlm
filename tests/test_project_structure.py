"""Tests for verifying the BrainMRI-SpatialVLM project scaffold and structure.

This test module verifies the existence of all required directories,
configuration files, project documentation, and path specifications.
"""

from pathlib import Path
import unittest


class TestProjectStructure(unittest.TestCase):
    """Test suite to validate project directory scaffold and file integrity."""

    @classmethod
    def setUpClass(cls) -> None:
        """Locate project root based on repository layout."""
        cls.project_root = Path(__file__).resolve().parent.parent

    def test_required_directories_exist(self) -> None:
        """Verify that all mandatory directories are present."""
        expected_directories = [
            "data/raw",
            "data/processed",
            "data/annotations",
            "src",
            "models/monai_brats",
            "models/braingemma3d",
            "outputs/masks",
            "outputs/reports",
            "outputs/visualizations",
            "notebooks",
            "configs",
            "tests",
        ]
        for rel_dir in expected_directories:
            dir_path = self.project_root / rel_dir
            with self.subTest(directory=rel_dir):
                self.assertTrue(
                    dir_path.exists(),
                    f"Required directory does not exist: {rel_dir}",
                )
                self.assertTrue(
                    dir_path.is_dir(),
                    f"Path is not a directory: {rel_dir}",
                )

    def test_required_root_files_exist(self) -> None:
        """Verify that critical root repository files exist and are not empty."""
        expected_files = [
            "requirements.txt",
            "README.md",
            ".gitignore",
            "configs/project_config.yaml",
        ]
        for rel_file in expected_files:
            file_path = self.project_root / rel_file
            with self.subTest(file=rel_file):
                self.assertTrue(
                    file_path.exists(),
                    f"Required file does not exist: {rel_file}",
                )
                self.assertTrue(
                    file_path.is_file(),
                    f"Path is not a file: {rel_file}",
                )
                self.assertGreater(
                    file_path.stat().st_size,
                    0,
                    f"File should not be empty: {rel_file}",
                )

    def test_readme_disclaimer_present(self) -> None:
        """Verify README contains the mandatory research-only disclaimer."""
        readme_path = self.project_root / "README.md"
        content = readme_path.read_text(encoding="utf-8")
        disclaimer = "AI-generated research draft — not a medical diagnosis."
        self.assertIn(disclaimer, content)

    def test_config_paths_specified(self) -> None:
        """Verify project_config.yaml contains required path definitions."""
        config_path = self.project_root / "configs" / "project_config.yaml"
        content = config_path.read_text(encoding="utf-8")

        required_keys = [
            "data_raw",
            "data_processed",
            "data_annotations",
            "model_monai_brats",
            "model_braingemma3d",
            "output_masks",
            "output_reports",
            "output_visualizations",
        ]
        for key in required_keys:
            with self.subTest(config_key=key):
                self.assertIn(
                    key,
                    content,
                    f"Expected key '{key}' in configs/project_config.yaml",
                )


if __name__ == "__main__":
    unittest.main()

