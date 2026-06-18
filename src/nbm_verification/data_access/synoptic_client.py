"""Synoptic Data API client for point observations."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from nbm_verification.utils.exceptions import DataAccessError

logger = logging.getLogger(__name__)


class SynopticClient:
    """
    Client for retrieving point observations from Synoptic Data API.

    Handles API authentication, request formatting, and response parsing.
    """

    BASE_URL = "https://api.synopticdata.com/v2/"

    def __init__(self, api_token: Optional[str] = None):
        """
        Initialize Synoptic Data client.

        Parameters
        ----------
        api_token : str, optional
            Synoptic API token. If None, reads from environment variable
            SYNOPTIC_API_TOKEN
        """
        if api_token is None:
            import os

            api_token = os.environ.get("SYNOPTIC_API_TOKEN")

        if not api_token:
            raise DataAccessError(
                "Synoptic API token required. Set SYNOPTIC_API_TOKEN environment variable."
            )

        self.api_token = api_token

    def get_stations(
        self,
        networks: Optional[List[str]] = None,
        states: Optional[List[str]] = None,
        bbox: Optional[tuple] = None,
    ) -> pd.DataFrame:
        """
        Get list of stations matching criteria.

        Parameters
        ----------
        networks : List[str], optional
            List of network IDs (e.g., ["ASOS", "AWOS"])
        states : List[str], optional
            List of state abbreviations (e.g., ["IL", "IN"])
        bbox : tuple, optional
            Bounding box (min_lon, min_lat, max_lon, max_lat)

        Returns
        -------
        pd.DataFrame
            DataFrame with station metadata

        Examples
        --------
        >>> client = SynopticClient()
        >>> stations = client.get_stations(networks=["ASOS"], states=["IL"])
        """
        endpoint = "stations/metadata"
        url = self.BASE_URL + endpoint

        params = {"token": self.api_token}

        if networks:
            params["network"] = ",".join(networks)
        if states:
            params["state"] = ",".join(states)
        if bbox:
            params["bbox"] = ",".join(map(str, bbox))

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if "STATION" not in data:
                logger.warning("No stations found matching criteria")
                return pd.DataFrame()

            # Parse station metadata
            stations = []
            for station in data["STATION"]:
                stations.append(
                    {
                        "station_id": station.get("STID"),
                        "station_name": station.get("NAME"),
                        "lat": station.get("LATITUDE"),
                        "lon": station.get("LONGITUDE"),
                        "elevation": station.get("ELEVATION"),
                        "state": station.get("STATE"),
                        "network": ",".join(station.get("MNET_ID", [])),
                    }
                )

            return pd.DataFrame(stations)

        except requests.RequestException as e:
            logger.error(f"Error fetching stations from Synoptic API: {e}")
            raise DataAccessError(f"Failed to fetch stations: {e}")

    def get_observations(
        self,
        station_ids: List[str],
        start_time: datetime,
        end_time: datetime,
        variables: List[str],
    ) -> pd.DataFrame:
        """
        Get observations for specified stations and time range.

        Parameters
        ----------
        station_ids : List[str]
            List of station IDs
        start_time : datetime
            Start of time range
        end_time : datetime
            End of time range
        variables : List[str]
            List of variable names to retrieve

        Returns
        -------
        pd.DataFrame
            DataFrame with observations

        Examples
        --------
        >>> client = SynopticClient()
        >>> obs = client.get_observations(
        ...     ["KORD"], datetime(2024, 1, 1), datetime(2024, 1, 2), ["air_temp"]
        ... )
        """
        endpoint = "stations/timeseries"
        url = self.BASE_URL + endpoint

        params = {
            "token": self.api_token,
            "stid": ",".join(station_ids),
            "start": start_time.strftime("%Y%m%d%H%M"),
            "end": end_time.strftime("%Y%m%d%H%M"),
            "vars": ",".join(variables),
            "units": "metric",
        }

        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            if "STATION" not in data:
                logger.warning("No observations found")
                return pd.DataFrame()

            # Parse observations
            obs_list = []
            for station in data["STATION"]:
                station_id = station.get("STID")
                observations = station.get("OBSERVATIONS", {})

                # Get time series
                date_times = observations.get("date_time", [])

                for var in variables:
                    if var in observations:
                        values = observations[var]
                        for dt, value in zip(date_times, values):
                            obs_list.append(
                                {
                                    "station_id": station_id,
                                    "datetime": pd.to_datetime(dt),
                                    "variable": var,
                                    "value": value,
                                }
                            )

            return pd.DataFrame(obs_list)

        except requests.RequestException as e:
            logger.error(f"Error fetching observations from Synoptic API: {e}")
            raise DataAccessError(f"Failed to fetch observations: {e}")

    def get_latest_observations(
        self, station_ids: List[str], variables: List[str], within_minutes: int = 60
    ) -> pd.DataFrame:
        """
        Get latest observations for stations.

        Parameters
        ----------
        station_ids : List[str]
            List of station IDs
        variables : List[str]
            List of variable names
        within_minutes : int, optional
            Get observations within this many minutes of current time

        Returns
        -------
        pd.DataFrame
            DataFrame with latest observations
        """
        endpoint = "stations/latest"
        url = self.BASE_URL + endpoint

        params = {
            "token": self.api_token,
            "stid": ",".join(station_ids),
            "vars": ",".join(variables),
            "within": str(within_minutes),
            "units": "metric",
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if "STATION" not in data:
                return pd.DataFrame()

            # Parse latest observations
            obs_list = []
            for station in data["STATION"]:
                station_id = station.get("STID")
                observations = station.get("OBSERVATIONS", {})

                for var in variables:
                    var_data = observations.get(var, {})
                    if isinstance(var_data, dict):
                        value = var_data.get("value")
                        date_time = var_data.get("date_time")

                        if value is not None and date_time is not None:
                            obs_list.append(
                                {
                                    "station_id": station_id,
                                    "datetime": pd.to_datetime(date_time),
                                    "variable": var,
                                    "value": value,
                                }
                            )

            return pd.DataFrame(obs_list)

        except requests.RequestException as e:
            logger.error(f"Error fetching latest observations: {e}")
            raise DataAccessError(f"Failed to fetch latest observations: {e}")

    @staticmethod
    def map_variable_name(nbm_variable: str) -> Optional[str]:
        """
        Map NBM variable name to Synoptic API variable name.

        Parameters
        ----------
        nbm_variable : str
            NBM variable name (e.g., "TMP_2m")

        Returns
        -------
        Optional[str]
            Synoptic variable name, or None if no mapping exists
        """
        mapping = {
            "TMP_2m": "air_temp",
            "DPT_2m": "dew_point_temperature",
            "APCP": "precip_accum",
            "GUST_10m": "wind_gust",
            "UGRD_10m": "wind_speed",  # Actually u-component
            "VGRD_10m": "wind_speed",  # Actually v-component
        }
        return mapping.get(nbm_variable)
