"""Temporal alignment utilities for matching forecast and observation times."""

from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np


def align_valid_times(
    forecast_time: datetime,
    observation_times: List[datetime],
    tolerance_minutes: int = 30,
) -> Optional[datetime]:
    """
    Find the observation time that best matches the forecast valid time.

    Parameters
    ----------
    forecast_time : datetime
        Forecast valid time
    observation_times : List[datetime]
        Available observation times
    tolerance_minutes : int, optional
        Maximum time difference allowed for a match (minutes), by default 30

    Returns
    -------
    Optional[datetime]
        Matched observation time, or None if no match within tolerance

    Examples
    --------
    >>> forecast_time = datetime(2024, 1, 1, 12, 0)
    >>> obs_times = [datetime(2024, 1, 1, 11, 55), datetime(2024, 1, 1, 12, 5)]
    >>> matched = align_valid_times(forecast_time, obs_times)
    >>> matched
    datetime.datetime(2024, 1, 1, 11, 55)
    """
    if not observation_times:
        return None

    # Convert to numpy array for vectorized operations
    obs_times_array = np.array([t.timestamp() for t in observation_times])
    forecast_timestamp = forecast_time.timestamp()

    # Compute time differences in seconds
    time_diffs = np.abs(obs_times_array - forecast_timestamp)

    # Find minimum difference
    min_diff_idx = np.argmin(time_diffs)
    min_diff_seconds = time_diffs[min_diff_idx]

    # Check if within tolerance
    tolerance_seconds = tolerance_minutes * 60
    if min_diff_seconds <= tolerance_seconds:
        return observation_times[min_diff_idx]

    return None


def compute_valid_time(init_time: datetime, forecast_hour: int) -> datetime:
    """
    Compute valid time from initialization time and forecast hour.

    Parameters
    ----------
    init_time : datetime
        Model initialization time
    forecast_hour : int
        Forecast hour offset

    Returns
    -------
    datetime
        Valid time

    Examples
    --------
    >>> init = datetime(2024, 1, 1, 0, 0)
    >>> valid = compute_valid_time(init, 6)
    >>> valid
    datetime.datetime(2024, 1, 1, 6, 0)
    """
    return init_time + timedelta(hours=forecast_hour)


def round_to_nearest_hour(dt: datetime) -> datetime:
    """
    Round datetime to nearest hour.

    Parameters
    ----------
    dt : datetime
        Input datetime

    Returns
    -------
    datetime
        Datetime rounded to nearest hour

    Examples
    --------
    >>> dt = datetime(2024, 1, 1, 12, 35)
    >>> rounded = round_to_nearest_hour(dt)
    >>> rounded
    datetime.datetime(2024, 1, 1, 13, 0)
    """
    # If minutes >= 30, round up, otherwise round down
    if dt.minute >= 30:
        rounded = dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        rounded = dt.replace(minute=0, second=0, microsecond=0)
    return rounded


def get_hourly_times_in_range(
    start_time: datetime, end_time: datetime
) -> List[datetime]:
    """
    Get list of hourly times within a range.

    Parameters
    ----------
    start_time : datetime
        Start of time range
    end_time : datetime
        End of time range

    Returns
    -------
    List[datetime]
        List of hourly datetimes from start to end (inclusive)

    Examples
    --------
    >>> start = datetime(2024, 1, 1, 0, 0)
    >>> end = datetime(2024, 1, 1, 3, 0)
    >>> times = get_hourly_times_in_range(start, end)
    >>> len(times)
    4
    """
    times = []
    current = start_time
    while current <= end_time:
        times.append(current)
        current += timedelta(hours=1)
    return times


def parse_datetime_from_filename(
    filename: str, pattern: str = "%Y%m%d%H"
) -> Optional[datetime]:
    """
    Extract datetime from filename using a pattern.

    Parameters
    ----------
    filename : str
        Filename containing datetime information
    pattern : str, optional
        strptime pattern to extract datetime, by default "%Y%m%d%H"

    Returns
    -------
    Optional[datetime]
        Extracted datetime, or None if parsing fails

    Examples
    --------
    >>> filename = "data_2024010112.grib2"
    >>> dt = parse_datetime_from_filename(filename)
    """
    import re

    # Try to find a string matching the pattern length
    pattern_length = len(datetime.now().strftime(pattern))

    # Look for continuous digits of the expected length
    matches = re.findall(r"\d{" + str(pattern_length) + "}", filename)

    if not matches:
        return None

    try:
        return datetime.strptime(matches[0], pattern)
    except ValueError:
        return None
