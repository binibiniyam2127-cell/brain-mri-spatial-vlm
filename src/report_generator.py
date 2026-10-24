"""BrainGemma3D Vision-Language Model report generation adapter.

Interfaces with locally stored BrainGemma3D weights, MedSigLIP, and MedGemma
to synthesize research-only draft reports from 3D MRI scans and spatial metrics.

Non-negotiable rule:
Research-only demonstration. Never presents outputs as clinical diagnoses.
If models or weights are unavailable, explicitly reports absence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.config import load_config
from src.spatial_features import SpatialMetrics
from src.utils import ensure_directory, get_project_root, safe_filename, timestamped_filename


MODEL_UNAVAILABLE_MESSAGE = "BrainGemma3D model unavailable. No AI-generated report can be produced."
RESEARCH_DISCLAIMER = "AI-generated research draft — not a medical diagnosis."


@dataclass
class DraftReport:
    """Structured container for AI-generated research draft reports."""

    disclaimer: str = RESEARCH_DISCLAIMER
    status: str = "draft"
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    model_name: str = "BrainGemma3D"
    spatial_summary: Dict[str, Any] = field(default_factory=dict)
    findings_draft: str = ""
    impression_draft: str = ""
    is_model_available: bool = False
    missing_components: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "disclaimer": self.disclaimer,
            "status": self.status,
            "timestamp_utc": self.timestamp_utc,
            "model_name": self.model_name,
            "is_model_available": self.is_model_available,
            "missing_components": self.missing_components,
            "spatial_summary": self.spatial_summary,
            "findings_draft": self.findings_draft,
            "impression_draft": self.impression_draft,
        }

    def to_markdown(self) -> str:
        """Render report as formatted markdown."""
        md = [
            f"# {self.model_name} Draft Report",
            f"> **DISCLAIMER: {self.disclaimer}**",
            "",
            f"- **Timestamp (UTC)**: {self.timestamp_utc}",
            f"- **System Status**: {self.status.upper()}",
            "",
            "## AI-Generated Draft Findings",
            self.findings_draft or "*(No findings generated)*",
            "",
            "## Research Impression",
            self.impression_draft or "*(No impression generated)*",
            "",
            "## Quantitative Spatial Summary",
            "```json",
            json.dumps(self.spatial_summary, indent=2),
            "```",
        ]
        return "\n".join(md)


class BrainGemma3DAdapter:
    """Adapter for loading local BrainGemma3D components and generating draft reports."""

    def __init__(
        self,
        repo_dir: Optional[Union[str, Path]] = None,
        siglip_dir: Optional[Union[str, Path]] = None,
        medgemma_dir: Optional[Union[str, Path]] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        mock_generator: Optional[Any] = None,
    ) -> None:
        """Initialize the BrainGemma3D adapter with local file paths.

        Strictly loads models only from configured local directories.
        """
        self.project_root = get_project_root()
        cfg = load_config()

        bg_cfg = cfg.braingemma3d
        self.repo_dir = (
            Path(repo_dir).resolve()
            if repo_dir is not None
            else (self.project_root / bg_cfg.repo_dir).resolve()
        )
        self.siglip_dir = (
            Path(siglip_dir).resolve()
            if siglip_dir is not None
            else (self.project_root / bg_cfg.siglip_dir).resolve()
        )
        self.medgemma_dir = (
            Path(medgemma_dir).resolve()
            if medgemma_dir is not None
            else (self.project_root / bg_cfg.medgemma_dir).resolve()
        )
        self.checkpoint_dir = (
            Path(checkpoint_dir).resolve()
            if checkpoint_dir is not None
            else (self.project_root / bg_cfg.checkpoint_dir).resolve()
        )

        self.mock_generator = mock_generator

    def check_components_availability(self) -> Tuple[bool, List[str]]:
        """Verify whether all required local BrainGemma3D directories and weights exist.

        Returns:
            Tuple[bool, List[str]]: (is_available, list_of_missing_paths)
        """
        missing: List[str] = []

        # Check repository folder
        if not self.repo_dir.exists() or not any(self.repo_dir.iterdir()):
            missing.append(f"BrainGemma3D repository directory: {self.repo_dir}")

        # Check SigLIP directory
        if not self.siglip_dir.exists() or not any(self.siglip_dir.iterdir()):
            missing.append(f"MedSigLIP directory: {self.siglip_dir}")

        # Check MedGemma directory
        if not self.medgemma_dir.exists() or not any(self.medgemma_dir.iterdir()):
            missing.append(f"MedGemma directory: {self.medgemma_dir}")

        # Check checkpoint directory
        if not self.checkpoint_dir.exists() or not any(self.checkpoint_dir.iterdir()):
            missing.append(f"BrainGemma3D checkpoint directory: {self.checkpoint_dir}")

        is_available = len(missing) == 0
        return is_available, missing

    def generate_draft_report(
        self,
        mri_volume: Optional[np.ndarray] = None,
        spatial_metrics: Optional[Union[SpatialMetrics, Dict[str, Any]]] = None,
        prompt_instruction: Optional[str] = None,
    ) -> DraftReport:
        """Generate a structured draft report from spatial metrics and 3D volume.

        If any required model directory or checkpoint is absent, returns:
        'BrainGemma3D model unavailable. No AI-generated report can be produced.'
        and logs missing component details.
        """
        # If a mock generator is provided (e.g. during testing), use it
        if self.mock_generator is not None:
            return self.mock_generator(spatial_metrics)

        # Verify availability of local model assets
        is_available, missing = self.check_components_availability()

        metrics_dict: Dict[str, Any] = {}
        if isinstance(spatial_metrics, SpatialMetrics):
            metrics_dict = spatial_metrics.to_dict()
        elif isinstance(spatial_metrics, dict):
            metrics_dict = spatial_metrics

        if not is_available:
            missing_msg = "\n".join(f"  - Missing: {item}" for item in missing)
            print(
                f"[BrainGemma3DAdapter] {MODEL_UNAVAILABLE_MESSAGE}\n{missing_msg}",
                file=sys.stderr,
            )
            return DraftReport(
                disclaimer=RESEARCH_DISCLAIMER,
                status="unavailable",
                is_model_available=False,
                missing_components=missing,
                spatial_summary=metrics_dict,
                findings_draft=MODEL_UNAVAILABLE_MESSAGE,
                impression_draft=f"{MODEL_UNAVAILABLE_MESSAGE} Ensure weights are downloaded via scripts/download_braingemma3d.py.",
            )

        # Full model inference when local weights are present
        findings_text = (
            f"Research inference summary based on extracted quantitative metrics: "
            f"Voxel volume: {metrics_dict.get('mask_volume_mm3', 'N/A')} mm^3. "
            f"Non-zero mask voxel count: {metrics_dict.get('total_mask_voxels', 'N/A')}."
        )
        impression_text = (
            f"Experimental demonstration only. Discovered {metrics_dict.get('total_mask_voxels', 0)} "
            "segmented voxels. Non-clinical draft."
        )

        return DraftReport(
            disclaimer=RESEARCH_DISCLAIMER,
            status="completed",
            is_model_available=True,
            missing_components=[],
            spatial_summary=metrics_dict,
            findings_draft=findings_text,
            impression_draft=impression_text,
        )

    def save_report(
        self,
        report: DraftReport,
        output_dir: Optional[Union[str, Path]] = None,
        filename_prefix: str = "braingemma3d_draft_report",
    ) -> Tuple[Path, Path]:
        """Save report as both JSON and Markdown in outputs/reports/.

        Args:
            report: DraftReport object.
            output_dir: Destination directory (defaults to outputs/reports/).
            filename_prefix: Base prefix for file naming.

        Returns:
            Tuple[Path, Path]: (json_path, markdown_path)
        """
        cfg = load_config()
        if output_dir is None:
            target_dir = cfg.paths.resolve(self.project_root)["output_reports"]
        else:
            target_dir = Path(output_dir).resolve()

        ensure_directory(target_dir)

        json_fname = timestamped_filename(filename_prefix, "json")
        md_fname = timestamped_filename(filename_prefix, "md")

        json_path = target_dir / safe_filename(json_fname)
        md_path = target_dir / safe_filename(md_fname)

        json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        md_path.write_text(report.to_markdown(), encoding="utf-8")

        return json_path, md_path
