"""URMA GRIB2 file reader with S3 support."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr

from nbm_verification.utils.exceptions import DataAccessError

logger = logging.getLogger(__name__)


class URMAReader:
    """
    Reader for URMA GRIB2 files from S3 or local storage.

    Downloads and processes URMA analysis files on-the-fly with
    support for spatial subsetting.
    """

    def __init__(
        self,
        source_type: str = "s3",
        bucket: Optional[str] = None,
        prefix: Optional[str] = None,
        local_cache: Optional[Path] = None,
    ):
        """
        Initialize URMA reader.

        Parameters
        ----------
        source_type : str, optional
            Data source type ("s3" or "local"), by default "s3"
        bucket : str, optional
            S3 bucket name (for s3 source)
        prefix : str, optional
            S3 key prefix (for s3 source)
        local_cache : Path, optional
            Local directory for caching downloaded files
        """
        self.source_type = source_type
        self.bucket = bucket
        self.prefix = prefix or ""
        self.local_cache = Path(local_cache) if local_cache else None

        if self.local_cache:
            self.local_cache.mkdir(parents=True, exist_ok=True)

    def read_variable(
        self,
        valid_time: datetime,
        variable_name: str,
        spatial_subset: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[xr.DataArray]:
        """
        Read a variable from URMA analysis file.

        Parameters
        ----------
        valid_time : datetime
            Valid time for URMA analysis
        variable_name : str
            Variable name (e.g., "TMP_2m", "DPT_2m")
        spatial_subset : Tuple[int, int, int, int], optional
            Spatial subset (i_start, i_end, j_start, j_end)

        Returns
        -------
        Optional[xr.DataArray]
            Data array with the variable, or None if not found

        Examples
        --------
        >>> reader = URMAReader(source_type="s3", bucket="noaa-nbm-grib2-pds")
        >>> data = reader.read_variable(
        ...     datetime(2024, 1, 1, 12, 0), "TMP_2m",
        ...     spatial_subset=(0, 100, 0, 100)
        ... )
        """
        # Get file path or S3 key
        if self.source_type == "s3":
            file_source = self._get_s3_key(valid_time)
            file_path = self._download_from_s3(file_source, valid_time)
        else:
            file_path = self._get_local_path(valid_time)

        if file_path is None or not file_path.exists():
            logger.warning(f"URMA file not found for {valid_time}")
            return None

        try:
            import pygrib

            grbs = pygrib.open(str(file_path))

            # Find the requested variable
            grb = self._find_variable(grbs, variable_name)

            if grb is None:
                logger.warning(f"Variable {variable_name} not found in URMA file")
                grbs.close()
                return None

            # Read values
            values = grb.values

            # Apply spatial subset if specified
            if spatial_subset is not None:
                i_start, i_end, j_start, j_end = spatial_subset
                values = values[j_start:j_end, i_start:i_end]

            # Get lat/lon coordinates
            lats, lons = grb.latlons()
            if spatial_subset is not None:
                i_start, i_end, j_start, j_end = spatial_subset
                lats = lats[j_start:j_end, i_start:i_end]
                lons = lons[j_start:j_end, i_start:i_end]

            grbs.close()

            # Create xarray DataArray
            data_array = xr.DataArray(
                values,
                dims=["y", "x"],
                coords={
                    "lat": (["y", "x"], lats),
                    "lon": (["y", "x"], lons),
                },
                attrs={
                    "variable_name": variable_name,
                    "valid_time": valid_time.isoformat(),
                    "source": "URMA",
                    "units": getattr(grb, "units", "unknown"),
                },
            )

            return data_array

        except Exception as e:
            logger.error(f"Error reading URMA file: {e}")
            raise DataAccessError(f"Failed to read URMA file: {e}")

    def _get_s3_key(self, valid_time: datetime) -> str:
        """
        Construct S3 key for URMA file.

        Parameters
        ----------
        valid_time : datetime
            Valid time

        Returns
        -------
        str
            S3 key
        """
        date_str = valid_time.strftime("%Y%m%d")
        hour = valid_time.hour
        filename = f"urma2p5.t{hour:02d}z.2dvaranl_ndfd.grb2"
        s3_key = f"{self.prefix}urma2p5.{date_str}/{filename}"
        return s3_key

    def _get_local_path(self, valid_time: datetime) -> Path:
        """Get local file path for URMA file."""
        date_str = valid_time.strftime("%Y%m%d")
        hour = valid_time.hour
        filename = f"urma2p5.t{hour:02d}z.2dvaranl_ndfd.grb2"
        return Path(self.prefix) / f"urma2p5.{date_str}" / filename

    def _download_from_s3(
        self, s3_key: str, valid_time: datetime
    ) -> Optional[Path]:
        """
        Download URMA file from S3.

        Parameters
        ----------
        s3_key : str
            S3 key
        valid_time : datetime
            Valid time (for cache naming)

        Returns
        -------
        Optional[Path]
            Path to downloaded file, or None if download failed
        """
        # Check cache first
        if self.local_cache:
            cache_filename = f"urma_{valid_time.strftime('%Y%m%d_%H00')}.grb2"
            cache_path = self.local_cache / cache_filename

            if cache_path.exists():
                logger.debug(f"Using cached URMA file: {cache_path}")
                return cache_path

        try:
            import boto3

            s3_client = boto3.client("s3")

            # Download to cache or temp location
            if self.local_cache:
                download_path = cache_path
            else:
                import tempfile

                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".grb2")
                download_path = Path(temp_file.name)
                temp_file.close()

            logger.info(f"Downloading URMA file from s3://{self.bucket}/{s3_key}")
            s3_client.download_file(self.bucket, s3_key, str(download_path))

            return download_path

        except Exception as e:
            logger.error(f"Failed to download URMA file from S3: {e}")
            return None

    def _find_variable(self, grbs, variable_name: str):
        """
        Find variable in GRIB messages.

        Parameters
        ----------
        grbs : pygrib.open
            Opened GRIB file
        variable_name : str
            Variable name to find

        Returns
        -------
        GRIB message or None
        """
        # Map variable names to URMA GRIB parameters
        variable_mapping = {
            "TMP_2m": {"name": "2 metre temperature"},
            "DPT_2m": {"name": "2 metre dewpoint temperature"},
            "UGRD_10m": {"name": "10 metre U wind component"},
            "VGRD_10m": {"name": "10 metre V wind component"},
        }

        if variable_name not in variable_mapping:
            logger.warning(f"Unknown URMA variable mapping for {variable_name}")
            return None

        search_params = variable_mapping[variable_name]

        try:
            grb = grbs.select(name=search_params["name"])[0]
            return grb
        except (ValueError, IndexError):
            return None
