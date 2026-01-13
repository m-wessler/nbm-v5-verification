"""Warning management for tracking issues during verification."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


class WarningManager:
    """
    Manages warnings generated during verification processing.

    Tracks various types of warnings (missing files, low sample counts,
    data quality issues) and writes them to log files.
    """

    def __init__(self):
        """Initialize warning manager."""
        self.warnings: Dict[str, List[Dict]] = {
            "file_availability": [],
            "sample_count": [],
            "data_quality": [],
            "processing": [],
            "other": [],
        }

    def add_warning(self, category: str, message: str, details: Dict = None) -> None:
        """
        Add a warning.

        Parameters
        ----------
        category : str
            Warning category (file_availability, sample_count, data_quality, processing, other)
        message : str
            Warning message
        details : Dict, optional
            Additional details

        Examples
        --------
        >>> wm = WarningManager()
        >>> wm.add_warning("file_availability", "NBM file missing", {"date": "2024-01-01"})
        """
        if category not in self.warnings:
            category = "other"

        warning_entry = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "details": details or {},
        }

        self.warnings[category].append(warning_entry)

        # Also log the warning
        logger.warning(f"[{category}] {message}")

    def get_warnings(self, category: str = None) -> List[Dict]:
        """
        Get warnings, optionally filtered by category.

        Parameters
        ----------
        category : str, optional
            Category to filter by. If None, returns all warnings.

        Returns
        -------
        List[Dict]
            List of warning entries
        """
        if category:
            return self.warnings.get(category, [])

        # Return all warnings
        all_warnings = []
        for cat_warnings in self.warnings.values():
            all_warnings.extend(cat_warnings)
        return all_warnings

    def get_warning_counts(self) -> Dict[str, int]:
        """
        Get count of warnings by category.

        Returns
        -------
        Dict[str, int]
            Dictionary of warning counts by category
        """
        return {category: len(warnings) for category, warnings in self.warnings.items()}

    def has_warnings(self, category: str = None) -> bool:
        """
        Check if there are any warnings.

        Parameters
        ----------
        category : str, optional
            Category to check. If None, checks all categories.

        Returns
        -------
        bool
            True if warnings exist
        """
        if category:
            return len(self.warnings.get(category, [])) > 0

        return any(len(warnings) > 0 for warnings in self.warnings.values())

    def save(self, output_dir: Path) -> None:
        """
        Save warnings to files.

        Parameters
        ----------
        output_dir : Path
            Output directory for warning files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save each category to a separate file
        for category, warnings in self.warnings.items():
            if len(warnings) > 0:
                warning_file = output_dir / f"{category}.log"

                with open(warning_file, "w") as f:
                    f.write(f"# {category.upper()} WARNINGS\n")
                    f.write(f"# Total: {len(warnings)}\n")
                    f.write("#" + "=" * 78 + "\n\n")

                    for warning in warnings:
                        f.write(f"[{warning['timestamp']}]\n")
                        f.write(f"{warning['message']}\n")

                        if warning['details']:
                            f.write(f"Details: {warning['details']}\n")

                        f.write("-" * 80 + "\n\n")

                logger.info(f"Saved {len(warnings)} {category} warnings to {warning_file}")

        # Also save summary
        summary_file = output_dir / "summary.txt"
        with open(summary_file, "w") as f:
            f.write("VERIFICATION WARNINGS SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            counts = self.get_warning_counts()
            total = sum(counts.values())

            f.write(f"Total warnings: {total}\n\n")

            for category, count in counts.items():
                if count > 0:
                    f.write(f"  {category}: {count}\n")

        logger.info(f"Saved warning summary to {summary_file}")

    def clear_warnings(self, category: str = None) -> None:
        """
        Clear warnings.

        Parameters
        ----------
        category : str, optional
            Category to clear. If None, clears all warnings.
        """
        if category:
            self.warnings[category] = []
        else:
            for category in self.warnings:
                self.warnings[category] = []
