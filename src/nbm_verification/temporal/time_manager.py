"""Time management for verification runs."""

from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np


class TimeManager:
    """
    Manages temporal configuration and expansion for verification runs.

    Handles date ranges, initialization hours, forecast hours, and the
    computation of valid times.
    """

    def __init__(self, temporal_config: Dict):
        """
        Initialize time manager from configuration.

        Parameters
        ----------
        temporal_config : Dict
            Temporal section of verification configuration

        Examples
        --------
        >>> config = {
        ...     "start_date": "2024-01-01",
        ...     "end_date": "2024-01-03",
        ...     "init_hours": [{"group_id": "00Z", "hours": [0]}],
        ...     "forecast_hours": [{"group_id": "f003", "hours": [3]}]
        ... }
        >>> tm = TimeManager(config)
        """
        self.start_date = datetime.strptime(temporal_config["start_date"], "%Y-%m-%d")
        self.end_date = datetime.strptime(temporal_config["end_date"], "%Y-%m-%d")
        self.init_hour_groups = temporal_config["init_hours"]
        self.forecast_hour_groups = temporal_config["forecast_hours"]

    def get_date_range(self) -> List[datetime]:
        """
        Get list of all dates in the verification period.

        Returns
        -------
        List[datetime]
            List of dates from start_date to end_date (inclusive)

        Examples
        --------
        >>> tm = TimeManager({"start_date": "2024-01-01", "end_date": "2024-01-03", ...})
        >>> dates = tm.get_date_range()
        >>> len(dates)
        3
        """
        dates = []
        current = self.start_date
        while current <= self.end_date:
            dates.append(current)
            current += timedelta(days=1)
        return dates

    def get_init_hour_groups(self) -> List[Dict]:
        """
        Get initialization hour groups.

        Returns
        -------
        List[Dict]
            List of init hour group configurations
        """
        return self.init_hour_groups

    def get_forecast_hour_groups(self) -> List[Dict]:
        """
        Get forecast hour groups.

        Returns
        -------
        List[Dict]
            List of forecast hour group configurations
        """
        return self.forecast_hour_groups

    def compute_valid_time(
        self, init_date: datetime, init_hour: int, forecast_hour: int
    ) -> datetime:
        """
        Compute valid time from initialization time and forecast hour.

        Parameters
        ----------
        init_date : datetime
            Initialization date
        init_hour : int
            Initialization hour (0-23)
        forecast_hour : int
            Forecast hour offset

        Returns
        -------
        datetime
            Valid time

        Examples
        --------
        >>> tm = TimeManager({...})
        >>> init_date = datetime(2024, 1, 1)
        >>> valid_time = tm.compute_valid_time(init_date, 0, 6)
        >>> valid_time
        datetime.datetime(2024, 1, 1, 6, 0)
        """
        init_time = init_date.replace(hour=init_hour, minute=0, second=0, microsecond=0)
        valid_time = init_time + timedelta(hours=forecast_hour)
        return valid_time

    def expand_verification_times(self) -> List[Dict]:
        """
        Expand all verification time combinations.

        This generates a list of all (date, init_hour, forecast_hour) combinations
        for the verification run.

        Returns
        -------
        List[Dict]
            List of dictionaries with date, init_hour, forecast_hour, and valid_time

        Examples
        --------
        >>> tm = TimeManager({...})
        >>> times = tm.expand_verification_times()
        >>> times[0].keys()
        dict_keys(['date', 'init_hour', 'forecast_hour', 'valid_time', 'init_group_id', 'fhour_group_id'])
        """
        times = []
        dates = self.get_date_range()

        for date in dates:
            for init_group in self.init_hour_groups:
                for init_hour in init_group["hours"]:
                    for fhour_group in self.forecast_hour_groups:
                        for forecast_hour in fhour_group["hours"]:
                            valid_time = self.compute_valid_time(date, init_hour, forecast_hour)

                            times.append(
                                {
                                    "date": date,
                                    "init_hour": init_hour,
                                    "forecast_hour": forecast_hour,
                                    "valid_time": valid_time,
                                    "init_group_id": init_group["group_id"],
                                    "fhour_group_id": fhour_group["group_id"],
                                }
                            )

        return times

    def group_times_for_processing(self) -> List[Tuple[str, str, List[Dict]]]:
        """
        Group verification times by init_group and fhour_group for processing.

        This organizes times into groups that should be processed together
        (i.e., pooled as one statistical population).

        Returns
        -------
        List[Tuple[str, str, List[Dict]]]
            List of (init_group_id, fhour_group_id, times) tuples

        Examples
        --------
        >>> tm = TimeManager({...})
        >>> groups = tm.group_times_for_processing()
        >>> groups[0][0]  # init_group_id
        '00Z'
        """
        times = self.expand_verification_times()

        # Group by (init_group_id, fhour_group_id)
        groups_dict: Dict[Tuple[str, str], List[Dict]] = {}

        for time_entry in times:
            key = (time_entry["init_group_id"], time_entry["fhour_group_id"])
            if key not in groups_dict:
                groups_dict[key] = []
            groups_dict[key].append(time_entry)

        # Convert to list of tuples
        groups = [(init_id, fhour_id, times) for (init_id, fhour_id), times in groups_dict.items()]

        return groups
