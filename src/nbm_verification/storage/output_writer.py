"""Output writer for NetCDF and CSV formats."""

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


class OutputWriter:
    """
    Writes verification results to NetCDF and CSV files.

    Handles gridpoint metrics (NetCDF), regional metrics (CSV),
    and station metrics (CSV).
    """

    def __init__(self, output_base_path: Path, run_name: str):
        """
        Initialize output writer.

        Parameters
        ----------
        output_base_path : Path
            Base output directory
        run_name : str
            Verification run name
        """
        self.output_base_path = Path(output_base_path)
        self.run_name = run_name
        self.output_dir = self.output_base_path / run_name

        # Create output directories
        self.gridpoint_dir = self.output_dir / "gridpoint"
        self.regional_dir = self.output_dir / "regional"
        self.station_dir = self.output_dir / "station"

        for directory in [self.gridpoint_dir, self.regional_dir, self.station_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def write_gridpoint_metrics(
        self,
        metrics: Dict,
        variable_name: str,
        init_group_id: str,
        fhour_group_id: str,
        ni: int,
        nj: int,
    ) -> Path:
        """
        Write gridpoint metrics to NetCDF file.

        Parameters
        ----------
        metrics : Dict
            Dictionary of gridpoint metrics keyed by (i, j)
        variable_name : str
            Variable name
        init_group_id : str
            Initialization hour group ID
        fhour_group_id : str
            Forecast hour group ID
        ni : int
            Grid size in i-direction
        nj : int
            Grid size in j-direction

        Returns
        -------
        Path
            Path to output file

        Examples
        --------
        >>> writer = OutputWriter(Path("/output"), "run1")
        >>> path = writer.write_gridpoint_metrics(metrics, "TMP_2m", "00Z", "f003", 2345, 1597)
        """
        filename = f"{variable_name}_{fhour_group_id}_{init_group_id}.nc"
        output_path = self.gridpoint_dir / filename

        logger.info(f"Writing gridpoint metrics to {output_path}")

        # Initialize 2D arrays for each metric
        metric_arrays = {}
        metric_keys = ["mae", "bias", "rmse", "bias_ratio", "n_samples"]

        for key in metric_keys:
            metric_arrays[key] = np.full((nj, ni), np.nan)

        # Also initialize lat/lon arrays
        lats = np.full((nj, ni), np.nan)
        lons = np.full((nj, ni), np.nan)

        # Fill arrays from metrics dictionary
        for (i, j), gridpoint_metrics in metrics.items():
            lats[j, i] = gridpoint_metrics.get("lat", np.nan)
            lons[j, i] = gridpoint_metrics.get("lon", np.nan)

            for key in metric_keys:
                metric_arrays[key][j, i] = gridpoint_metrics.get(key, np.nan)

        # Create xarray Dataset
        ds = xr.Dataset(
            {
                key: (["y", "x"], metric_arrays[key]) for key in metric_keys
            },
            coords={
                "lat": (["y", "x"], lats),
                "lon": (["y", "x"], lons),
                "x": np.arange(ni),
                "y": np.arange(nj),
            },
            attrs={
                "variable": variable_name,
                "init_group": init_group_id,
                "fhour_group": fhour_group_id,
                "description": "NBM verification gridpoint metrics",
            },
        )

        # Write to NetCDF
        ds.to_netcdf(output_path, encoding={key: {"zlib": True, "complevel": 4} for key in metric_keys})

        logger.info(f"Wrote gridpoint metrics: {output_path}")
        return output_path

    def write_regional_metrics(
        self,
        metrics: List[Dict],
        variable_name: str,
        init_group_id: str,
        fhour_group_id: str,
        region_type: str,
    ) -> Path:
        """
        Write regional metrics to CSV file.

        Parameters
        ----------
        metrics : List[Dict]
            List of regional metric dictionaries
        variable_name : str
            Variable name
        init_group_id : str
            Initialization hour group ID
        fhour_group_id : str
            Forecast hour group ID
        region_type : str
            Region type (CWA, RFC, Zone)

        Returns
        -------
        Path
            Path to output file
        """
        filename = f"{region_type}_{variable_name}_{fhour_group_id}_{init_group_id}.csv"
        output_path = self.regional_dir / filename

        logger.info(f"Writing regional metrics to {output_path}")

        # Convert to DataFrame
        df = pd.DataFrame(metrics)

        # Write to CSV
        df.to_csv(output_path, index=False)

        logger.info(f"Wrote regional metrics: {output_path}")
        return output_path

    def write_station_metrics(
        self,
        metrics: List[Dict],
        variable_name: str,
        init_group_id: str,
        fhour_group_id: str,
    ) -> Path:
        """
        Write station metrics to CSV file.

        Parameters
        ----------
        metrics : List[Dict]
            List of station metric dictionaries
        variable_name : str
            Variable name
        init_group_id : str
            Initialization hour group ID
        fhour_group_id : str
            Forecast hour group ID

        Returns
        -------
        Path
            Path to output file
        """
        filename = f"{variable_name}_{fhour_group_id}_{init_group_id}.csv"
        output_path = self.station_dir / filename

        logger.info(f"Writing station metrics to {output_path}")

        # Convert to DataFrame
        df = pd.DataFrame(metrics)

        # Write to CSV
        df.to_csv(output_path, index=False)

        logger.info(f"Wrote station metrics: {output_path}")
        return output_path
