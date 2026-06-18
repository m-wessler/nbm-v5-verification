"""NBM GRIB2 file reader using pygrib."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

from nbm_verification.utils.exceptions import DataAccessError

logger = logging.getLogger(__name__)


class NBMReader:
    """
    Reader for NBM GRIB2 files stored locally.

    Uses pygrib to read GRIB2 files and organizes data into xarray DataArrays.
    Supports spatial subsetting for memory-efficient chunked processing.
    """

    def __init__(self, base_path: Path, file_pattern: Optional[str] = None):
        """
        Initialize NBM reader.

        Parameters
        ----------
        base_path : Path
            Base directory containing NBM files
        file_pattern : str, optional
            File naming pattern (with format placeholders)
        """
        self.base_path = Path(base_path)
        self.file_pattern = file_pattern or "blend.t{cyc:02d}z.core.f{fhour:03d}.co.grib2"

    def read_variable(
        self,
        init_date: datetime,
        init_hour: int,
        forecast_hour: int,
        variable_name: str,
        spatial_subset: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[xr.DataArray]:
        """
        Read a variable from NBM GRIB2 file.

        Parameters
        ----------
        init_date : datetime
            Initialization date
        init_hour : int
            Initialization hour (0-23)
        forecast_hour : int
            Forecast hour offset
        variable_name : str
            Variable name (e.g., "TMP_2m", "APCP")
        spatial_subset : Tuple[int, int, int, int], optional
            Spatial subset (i_start, i_end, j_start, j_end)

        Returns
        -------
        Optional[xr.DataArray]
            Data array with the variable, or None if not found

        Examples
        --------
        >>> reader = NBMReader(Path("/data/nbm"))
        >>> data = reader.read_variable(
        ...     datetime(2024, 1, 1), 0, 6, "TMP_2m",
        ...     spatial_subset=(0, 100, 0, 100)
        ... )
        """
        file_path = self._get_file_path(init_date, init_hour, forecast_hour)

        if not file_path.exists():
            logger.warning(f"NBM file not found: {file_path}")
            return None

        try:
            # Import pygrib here to avoid dependency if not used
            import pygrib

            grbs = pygrib.open(str(file_path))

            # Find the requested variable
            # This is simplified - actual GRIB parameter matching is more complex
            grb = self._find_variable(grbs, variable_name)

            if grb is None:
                logger.warning(f"Variable {variable_name} not found in {file_path}")
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
                    "init_date": init_date.isoformat(),
                    "init_hour": init_hour,
                    "forecast_hour": forecast_hour,
                    "units": getattr(grb, "units", "unknown"),
                },
            )

            return data_array

        except Exception as e:
            logger.error(f"Error reading NBM file {file_path}: {e}")
            raise DataAccessError(f"Failed to read NBM file: {e}")

    def _get_file_path(
        self, init_date: datetime, init_hour: int, forecast_hour: int
    ) -> Path:
        """
        Construct file path from date and time information.

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
        Path
            Full path to GRIB2 file
        """
        date_str = init_date.strftime("%Y%m%d")
        filename = self.file_pattern.format(cyc=init_hour, fhour=forecast_hour)

        # Typical NBM directory structure
        file_path = self.base_path / date_str / f"{init_hour:02d}" / "core" / filename

        return file_path

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
        # Map common variable names to GRIB parameter names
        # This mapping should be configurable
        variable_mapping = {
            "TMP_2m": {"name": "2 metre temperature"},
            "DPT_2m": {"name": "2 metre dewpoint temperature"},
            "APCP": {"name": "Total Precipitation"},
            "GUST_10m": {"name": "Wind speed (gust)"},
        }

        if variable_name not in variable_mapping:
            logger.warning(f"Unknown variable mapping for {variable_name}")
            return None

        search_params = variable_mapping[variable_name]

        try:
            # Try to select by name
            grb = grbs.select(name=search_params["name"])[0]
            return grb
        except (ValueError, IndexError):
            return None

    def get_grid_info(self) -> Dict:
        """
        Get grid information from a sample NBM file.

        Returns
        -------
        Dict
            Dictionary with grid dimensions and coordinates
        """
        # This would read a sample file to get grid info
        # Placeholder implementation
        return {
            "ni": 2345,  # Example CONUS grid size
            "nj": 1597,
            "projection": "Lambert Conformal",
        }
