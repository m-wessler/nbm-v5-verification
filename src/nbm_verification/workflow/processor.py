"""Core chunk processor for verification computations."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from nbm_verification.accumulation.spatial_accumulator import (
    GridpointAccumulator,
    RegionalAccumulator,
    StationAccumulator,
)
from nbm_verification.data_access.nbm_reader import NBMReader
from nbm_verification.data_access.urma_reader import URMAReader
from nbm_verification.spatial.spatial_index import SpatialIndex
from nbm_verification.spatial.stations import StationManager

logger = logging.getLogger(__name__)


class ChunkProcessor:
    """
    Processes individual spatial chunks for verification.

    Loads data, computes statistics, and updates accumulators for
    a single spatial chunk.
    """

    def __init__(
        self,
        nbm_reader: NBMReader,
        urma_reader: URMAReader,
        spatial_index: Optional[SpatialIndex] = None,
        station_manager: Optional[StationManager] = None,
    ):
        """
        Initialize chunk processor.

        Parameters
        ----------
        nbm_reader : NBMReader
            NBM data reader
        urma_reader : URMAReader
            URMA data reader
        spatial_index : SpatialIndex, optional
            Spatial index for regional mapping
        station_manager : StationManager, optional
            Station manager for station-based verification
        """
        self.nbm_reader = nbm_reader
        self.urma_reader = urma_reader
        self.spatial_index = spatial_index
        self.station_manager = station_manager

    def process_chunk(
        self,
        chunk: Dict,
        time_entry: Dict,
        variable_name: str,
        thresholds: List[float],
        gridpoint_accumulators: Dict,
        regional_accumulators: Optional[Dict] = None,
    ) -> None:
        """
        Process a single spatial chunk for one time.

        Parameters
        ----------
        chunk : Dict
            Spatial chunk definition
        time_entry : Dict
            Time information (date, init_hour, forecast_hour, valid_time)
        variable_name : str
            Variable to process
        thresholds : List[float]
            Thresholds for categorical metrics
        gridpoint_accumulators : Dict
            Dictionary of gridpoint accumulators (keyed by (i, j))
        regional_accumulators : Dict, optional
            Dictionary of regional accumulators

        Examples
        --------
        >>> processor = ChunkProcessor(nbm_reader, urma_reader)
        >>> processor.process_chunk(chunk, time_entry, "TMP_2m", [273.15], accumulators)
        """
        # Extract chunk bounds
        spatial_subset = (
            chunk["i_start"],
            chunk["i_end"],
            chunk["j_start"],
            chunk["j_end"],
        )

        # Load NBM forecast data
        nbm_data = self.nbm_reader.read_variable(
            time_entry["date"],
            time_entry["init_hour"],
            time_entry["forecast_hour"],
            variable_name,
            spatial_subset=spatial_subset,
        )

        if nbm_data is None:
            logger.warning(
                f"NBM data not found for {time_entry['date']} "
                f"{time_entry['init_hour']:02d}Z f{time_entry['forecast_hour']:03d}"
            )
            return

        # Load URMA analysis data
        urma_data = self.urma_reader.read_variable(
            time_entry["valid_time"], variable_name, spatial_subset=spatial_subset
        )

        if urma_data is None:
            logger.warning(f"URMA data not found for {time_entry['valid_time']}")
            return

        # Ensure data shapes match
        if nbm_data.shape != urma_data.shape:
            logger.error(
                f"Data shape mismatch: NBM {nbm_data.shape} vs URMA {urma_data.shape}"
            )
            return

        # Process each gridpoint in chunk
        nj, ni = nbm_data.shape

        for j in range(nj):
            for i in range(ni):
                # Global grid indices
                global_i = chunk["i_start"] + i
                global_j = chunk["j_start"] + j

                forecast_value = nbm_data.values[j, i]
                obs_value = urma_data.values[j, i]

                # Skip if either is NaN
                if np.isnan(forecast_value) or np.isnan(obs_value):
                    continue

                # Update gridpoint accumulator
                gridpoint_key = (global_i, global_j)

                if gridpoint_key not in gridpoint_accumulators:
                    # Create new accumulator
                    lat = float(nbm_data.coords["lat"].values[j, i])
                    lon = float(nbm_data.coords["lon"].values[j, i])
                    gridpoint_accumulators[gridpoint_key] = GridpointAccumulator(
                        global_i, global_j, lat, lon, thresholds
                    )

                gridpoint_accumulators[gridpoint_key].update(
                    np.array([forecast_value]), np.array([obs_value])
                )

                # Update regional accumulators if spatial index available
                if regional_accumulators and self.spatial_index:
                    self._update_regional_accumulators(
                        global_i,
                        global_j,
                        forecast_value,
                        obs_value,
                        regional_accumulators,
                    )

    def _update_regional_accumulators(
        self,
        i: int,
        j: int,
        forecast_value: float,
        obs_value: float,
        regional_accumulators: Dict,
    ) -> None:
        """
        Update regional accumulators for a gridpoint.

        Parameters
        ----------
        i : int
            Grid i-index
        j : int
            Grid j-index
        forecast_value : float
            Forecast value
        obs_value : float
            Observation value
        regional_accumulators : Dict
            Dictionary of regional accumulators
        """
        # Get all region types
        region_types = self.spatial_index.region_to_gridpoints.keys()

        for region_type in region_types:
            region_id = self.spatial_index.get_region_for_gridpoint(i, j, region_type)

            if region_id:
                key = (region_type, region_id)

                if key in regional_accumulators:
                    regional_accumulators[key].update(
                        np.array([forecast_value]), np.array([obs_value])
                    )
