"""File inventory management for tracking available data files."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

from nbm_verification.utils.exceptions import DataAccessError

logger = logging.getLogger(__name__)


class FileInventory:
    """
    Tracks availability of NBM and URMA files.

    This class maintains an inventory of which files exist and which are missing,
    allowing the verification system to handle missing data gracefully.
    """

    def __init__(self):
        """Initialize file inventory."""
        self.nbm_files: Dict[str, Path] = {}
        self.urma_files: Dict[str, Path] = {}
        self.missing_nbm: Set[str] = set()
        self.missing_urma: Set[str] = set()

    def scan_nbm_directory(
        self,
        base_path: Path,
        start_date: datetime,
        end_date: datetime,
        init_hours: List[int],
        forecast_hours: List[int],
    ) -> None:
        """
        Scan directory for NBM files.

        Parameters
        ----------
        base_path : Path
            Base path to NBM data
        start_date : datetime
            Start date to scan
        end_date : datetime
            End date to scan
        init_hours : List[int]
            Initialization hours to look for
        forecast_hours : List[int]
            Forecast hours to look for
        """
        logger.info(f"Scanning NBM directory: {base_path}")

        from datetime import timedelta

        current_date = start_date
        while current_date <= end_date:
            for init_hour in init_hours:
                for forecast_hour in forecast_hours:
                    # Construct expected file path
                    # This is a simplified pattern - actual pattern would come from config
                    file_key = self._make_nbm_key(current_date, init_hour, forecast_hour)

                    # Example path pattern (should be configurable)
                    date_str = current_date.strftime("%Y%m%d")
                    filename = f"blend.t{init_hour:02d}z.core.f{forecast_hour:03d}.co.grib2"
                    file_path = base_path / date_str / f"{init_hour:02d}" / "core" / filename

                    if file_path.exists():
                        self.nbm_files[file_key] = file_path
                    else:
                        self.missing_nbm.add(file_key)

            current_date += timedelta(days=1)

        logger.info(
            f"NBM inventory: {len(self.nbm_files)} found, {len(self.missing_nbm)} missing"
        )

    def scan_urma_s3_bucket(
        self,
        bucket: str,
        prefix: str,
        start_date: datetime,
        end_date: datetime,
        hours: List[int],
    ) -> None:
        """
        Scan S3 bucket for URMA files.

        Parameters
        ----------
        bucket : str
            S3 bucket name
        prefix : str
            S3 key prefix
        start_date : datetime
            Start date to scan
        end_date : datetime
            End date to scan
        hours : List[int]
            Hours to look for
        """
        logger.info(f"Scanning URMA S3 bucket: {bucket}/{prefix}")

        # This would use boto3 to list S3 objects
        # Simplified implementation for now
        logger.warning("S3 scanning not yet implemented - assuming all files present")

    def check_nbm_file(
        self, init_date: datetime, init_hour: int, forecast_hour: int
    ) -> Optional[Path]:
        """
        Check if NBM file exists.

        Parameters
        ----------
        init_date : datetime
            Initialization date
        init_hour : int
            Initialization hour
        forecast_hour : int
            Forecast hour

        Returns
        -------
        Optional[Path]
            Path to file if exists, None otherwise
        """
        key = self._make_nbm_key(init_date, init_hour, forecast_hour)
        return self.nbm_files.get(key)

    def check_urma_file(self, valid_time: datetime) -> Optional[str]:
        """
        Check if URMA file exists.

        Parameters
        ----------
        valid_time : datetime
            Valid time for URMA analysis

        Returns
        -------
        Optional[str]
            S3 key if exists, None otherwise
        """
        key = self._make_urma_key(valid_time)
        return self.urma_files.get(key)

    def get_missing_files_summary(self) -> Dict[str, int]:
        """
        Get summary of missing files.

        Returns
        -------
        Dict[str, int]
            Dictionary with counts of missing files by type
        """
        return {
            "missing_nbm": len(self.missing_nbm),
            "missing_urma": len(self.missing_urma),
            "total_nbm": len(self.nbm_files) + len(self.missing_nbm),
            "total_urma": len(self.urma_files) + len(self.missing_urma),
        }

    def _make_nbm_key(
        self, init_date: datetime, init_hour: int, forecast_hour: int
    ) -> str:
        """Create unique key for NBM file."""
        date_str = init_date.strftime("%Y%m%d")
        return f"NBM_{date_str}_{init_hour:02d}Z_f{forecast_hour:03d}"

    def _make_urma_key(self, valid_time: datetime) -> str:
        """Create unique key for URMA file."""
        return valid_time.strftime("URMA_%Y%m%d_%H00")
