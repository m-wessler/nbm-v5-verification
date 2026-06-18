"""Metadata tracking for verification runs."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class MetadataTracker:
    """
    Tracks metadata for verification runs.

    Records configuration, processing statistics, warnings, and
    other run information.
    """

    def __init__(self, run_name: str):
        """
        Initialize metadata tracker.

        Parameters
        ----------
        run_name : str
            Verification run name
        """
        self.run_name = run_name
        self.metadata: Dict[str, Any] = {
            "run_name": run_name,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "status": "running",
            "configuration": {},
            "processing_stats": {},
            "warnings": [],
            "errors": [],
        }

    def set_configuration(self, config: Dict) -> None:
        """
        Store run configuration.

        Parameters
        ----------
        config : Dict
            Configuration dictionary
        """
        self.metadata["configuration"] = config

    def add_processing_stat(self, key: str, value: Any) -> None:
        """
        Add a processing statistic.

        Parameters
        ----------
        key : str
            Statistic key
        value : Any
            Statistic value
        """
        self.metadata["processing_stats"][key] = value

    def add_warning(self, warning: str) -> None:
        """
        Add a warning message.

        Parameters
        ----------
        warning : str
            Warning message
        """
        self.metadata["warnings"].append(
            {"timestamp": datetime.now().isoformat(), "message": warning}
        )

    def add_error(self, error: str) -> None:
        """
        Add an error message.

        Parameters
        ----------
        error : str
            Error message
        """
        self.metadata["errors"].append(
            {"timestamp": datetime.now().isoformat(), "message": error}
        )

    def mark_complete(self) -> None:
        """Mark run as complete."""
        self.metadata["end_time"] = datetime.now().isoformat()
        self.metadata["status"] = "completed"

    def mark_failed(self, error_message: str) -> None:
        """
        Mark run as failed.

        Parameters
        ----------
        error_message : str
            Error message
        """
        self.metadata["end_time"] = datetime.now().isoformat()
        self.metadata["status"] = "failed"
        self.add_error(error_message)

    def save(self, output_dir: Path) -> Path:
        """
        Save metadata to JSON file.

        Parameters
        ----------
        output_dir : Path
            Output directory

        Returns
        -------
        Path
            Path to metadata file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = output_dir / "metadata.json"

        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

        logger.info(f"Saved metadata to {metadata_path}")
        return metadata_path

    def load(self, metadata_path: Path) -> None:
        """
        Load metadata from JSON file.

        Parameters
        ----------
        metadata_path : Path
            Path to metadata file
        """
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        logger.info(f"Loaded metadata from {metadata_path}")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of metadata.

        Returns
        -------
        Dict[str, Any]
            Summary dictionary
        """
        return {
            "run_name": self.run_name,
            "status": self.metadata["status"],
            "start_time": self.metadata["start_time"],
            "end_time": self.metadata["end_time"],
            "num_warnings": len(self.metadata["warnings"]),
            "num_errors": len(self.metadata["errors"]),
        }
