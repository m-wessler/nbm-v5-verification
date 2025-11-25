"""Station management and station-to-gridpoint mapping."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StationManager:
    """
    Manages weather station metadata and mapping to nearest gridpoints.

    Handles station lists from various sources and computes the nearest
    gridpoint for each station for forecast-observation pairing.
    """

    def __init__(self):
        """Initialize station manager."""
        self.stations: pd.DataFrame = pd.DataFrame()
        self.station_to_gridpoint: Dict[str, Tuple[int, int]] = {}

    def load_stations_from_dataframe(self, stations_df: pd.DataFrame) -> None:
        """
        Load stations from a pandas DataFrame.

        Parameters
        ----------
        stations_df : pd.DataFrame
            DataFrame with station metadata including station_id, lat, lon

        Examples
        --------
        >>> sm = StationManager()
        >>> stations = pd.DataFrame({
        ...     'station_id': ['KORD', 'KMDW'],
        ...     'lat': [41.98, 41.79],
        ...     'lon': [-87.90, -87.75]
        ... })
        >>> sm.load_stations_from_dataframe(stations)
        """
        required_cols = ["station_id", "lat", "lon"]
        if not all(col in stations_df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")

        self.stations = stations_df.copy()
        logger.info(f"Loaded {len(self.stations)} stations")

    def map_stations_to_grid(
        self, lats: np.ndarray, lons: np.ndarray
    ) -> None:
        """
        Map each station to its nearest gridpoint.

        Parameters
        ----------
        lats : np.ndarray
            2D array of gridpoint latitudes
        lons : np.ndarray
            2D array of gridpoint longitudes

        Examples
        --------
        >>> sm = StationManager()
        >>> sm.load_stations_from_dataframe(stations_df)
        >>> sm.map_stations_to_grid(grid_lats, grid_lons)
        """
        logger.info("Mapping stations to nearest gridpoints")

        for idx, station in self.stations.iterrows():
            station_lat = station["lat"]
            station_lon = station["lon"]
            station_id = station["station_id"]

            # Find nearest gridpoint
            i, j = self._find_nearest_gridpoint(station_lat, station_lon, lats, lons)

            self.station_to_gridpoint[station_id] = (i, j)

        logger.info(f"Mapped {len(self.station_to_gridpoint)} stations to gridpoints")

    def _find_nearest_gridpoint(
        self, lat: float, lon: float, lats: np.ndarray, lons: np.ndarray
    ) -> Tuple[int, int]:
        """
        Find nearest gridpoint to a given lat/lon.

        Parameters
        ----------
        lat : float
            Station latitude
        lon : float
            Station longitude
        lats : np.ndarray
            2D array of gridpoint latitudes
        lons : np.ndarray
            2D array of gridpoint longitudes

        Returns
        -------
        Tuple[int, int]
            (i, j) indices of nearest gridpoint
        """
        # Compute distance using simple Euclidean approximation
        # For better accuracy, could use haversine distance
        dist = np.sqrt((lats - lat) ** 2 + (lons - lon) ** 2)

        # Find minimum distance
        j, i = np.unravel_index(np.argmin(dist), dist.shape)

        return int(i), int(j)

    def get_nearest_gridpoint(self, station_id: str) -> Optional[Tuple[int, int]]:
        """
        Get nearest gridpoint for a station.

        Parameters
        ----------
        station_id : str
            Station identifier

        Returns
        -------
        Optional[Tuple[int, int]]
            (i, j) gridpoint indices, or None if station not found
        """
        return self.station_to_gridpoint.get(station_id)

    def get_station_info(self, station_id: str) -> Optional[Dict]:
        """
        Get station metadata.

        Parameters
        ----------
        station_id : str
            Station identifier

        Returns
        -------
        Optional[Dict]
            Station metadata dictionary, or None if not found
        """
        station = self.stations[self.stations["station_id"] == station_id]

        if len(station) == 0:
            return None

        return station.iloc[0].to_dict()

    def get_stations_in_bbox(
        self, min_lat: float, max_lat: float, min_lon: float, max_lon: float
    ) -> pd.DataFrame:
        """
        Get stations within a bounding box.

        Parameters
        ----------
        min_lat : float
            Minimum latitude
        max_lat : float
            Maximum latitude
        min_lon : float
            Minimum longitude
        max_lon : float
            Maximum longitude

        Returns
        -------
        pd.DataFrame
            DataFrame of stations within bbox
        """
        mask = (
            (self.stations["lat"] >= min_lat)
            & (self.stations["lat"] <= max_lat)
            & (self.stations["lon"] >= min_lon)
            & (self.stations["lon"] <= max_lon)
        )

        return self.stations[mask].copy()

    def get_all_stations(self) -> pd.DataFrame:
        """
        Get all loaded stations.

        Returns
        -------
        pd.DataFrame
            DataFrame of all stations
        """
        return self.stations.copy()

    def filter_stations(
        self,
        networks: Optional[List[str]] = None,
        states: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Filter stations by network or state.

        Parameters
        ----------
        networks : List[str], optional
            List of network IDs to include
        states : List[str], optional
            List of state abbreviations to include

        Returns
        -------
        pd.DataFrame
            Filtered stations DataFrame
        """
        filtered = self.stations.copy()

        if networks and "network" in filtered.columns:
            # Handle comma-separated network strings
            mask = filtered["network"].apply(
                lambda x: any(net in str(x) for net in networks)
            )
            filtered = filtered[mask]

        if states and "state" in filtered.columns:
            filtered = filtered[filtered["state"].isin(states)]

        return filtered
