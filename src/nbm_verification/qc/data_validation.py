"""Data validation and quality control checks."""

import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def validate_data(
    data: np.ndarray,
    variable_name: str,
    expected_range: Optional[Tuple[float, float]] = None,
    check_all_nan: bool = True,
) -> Dict[str, bool]:
    """
    Validate data array for quality issues.

    Parameters
    ----------
    data : np.ndarray
        Data array to validate
    variable_name : str
        Variable name for logging
    expected_range : Tuple[float, float], optional
        Expected (min, max) range for values
    check_all_nan : bool, optional
        Check if all values are NaN, by default True

    Returns
    -------
    Dict[str, bool]
        Dictionary with validation results

    Examples
    --------
    >>> data = np.array([273.15, 283.15, 293.15])
    >>> result = validate_data(data, "TMP_2m", expected_range=(200.0, 330.0))
    >>> result['valid']
    True
    """
    results = {
        "valid": True,
        "all_nan": False,
        "out_of_range": False,
        "has_inf": False,
    }

    # Check for all NaN
    if check_all_nan:
        if np.all(np.isnan(data)):
            results["all_nan"] = True
            results["valid"] = False
            logger.warning(f"{variable_name}: All values are NaN")

    # Check for inf values
    if np.any(np.isinf(data)):
        results["has_inf"] = True
        results["valid"] = False
        logger.warning(f"{variable_name}: Contains inf values")

    # Check expected range
    if expected_range is not None:
        min_val, max_val = expected_range
        valid_data = data[~np.isnan(data)]

        if len(valid_data) > 0:
            if np.any(valid_data < min_val) or np.any(valid_data > max_val):
                results["out_of_range"] = True
                results["valid"] = False

                data_min = np.min(valid_data)
                data_max = np.max(valid_data)
                logger.warning(
                    f"{variable_name}: Values outside expected range "
                    f"[{min_val}, {max_val}]. Actual range: [{data_min}, {data_max}]"
                )

    return results


def check_units_consistency(
    forecast_data: np.ndarray,
    obs_data: np.ndarray,
    variable_name: str,
    tolerance: float = 1e-3,
) -> bool:
    """
    Check if forecast and observation data appear to have consistent units.

    Parameters
    ----------
    forecast_data : np.ndarray
        Forecast data
    obs_data : np.ndarray
        Observation data
    variable_name : str
        Variable name for logging
    tolerance : float, optional
        Tolerance for unit check

    Returns
    -------
    bool
        True if units appear consistent
    """
    # Get valid pairs
    valid_mask = ~(np.isnan(forecast_data) | np.isnan(obs_data))
    valid_fcst = forecast_data[valid_mask]
    valid_obs = obs_data[valid_mask]

    if len(valid_fcst) == 0:
        return True  # Can't check with no data

    # Check if scales are similar (within an order of magnitude)
    fcst_scale = np.median(np.abs(valid_fcst))
    obs_scale = np.median(np.abs(valid_obs))

    if fcst_scale == 0 or obs_scale == 0:
        return True  # Can't check with zero scales

    scale_ratio = fcst_scale / obs_scale

    # If scales differ by more than factor of 10, likely unit mismatch
    if scale_ratio > 10 or scale_ratio < 0.1:
        logger.warning(
            f"{variable_name}: Possible unit mismatch. "
            f"Forecast scale: {fcst_scale:.2f}, Obs scale: {obs_scale:.2f}"
        )
        return False

    return True


def detect_outliers(
    data: np.ndarray, std_threshold: float = 5.0
) -> Tuple[np.ndarray, int]:
    """
    Detect outliers using standard deviation threshold.

    Parameters
    ----------
    data : np.ndarray
        Data array
    std_threshold : float, optional
        Number of standard deviations for outlier threshold, by default 5.0

    Returns
    -------
    Tuple[np.ndarray, int]
        Boolean mask of outliers and count of outliers

    Examples
    --------
    >>> data = np.array([1, 2, 3, 100])  # 100 is an outlier
    >>> outlier_mask, count = detect_outliers(data, std_threshold=3.0)
    >>> count
    1
    """
    valid_data = data[~np.isnan(data)]

    if len(valid_data) < 3:
        return np.zeros_like(data, dtype=bool), 0

    mean = np.mean(valid_data)
    std = np.std(valid_data)

    # Create outlier mask for all data
    outlier_mask = np.abs(data - mean) > (std_threshold * std)

    # Don't count NaN as outliers
    outlier_mask = outlier_mask & ~np.isnan(data)

    outlier_count = np.sum(outlier_mask)

    return outlier_mask, int(outlier_count)


def get_data_statistics(data: np.ndarray) -> Dict:
    """
    Get summary statistics for data array.

    Parameters
    ----------
    data : np.ndarray
        Data array

    Returns
    -------
    Dict
        Dictionary with statistics
    """
    valid_data = data[~np.isnan(data)]

    if len(valid_data) == 0:
        return {
            "count": 0,
            "valid_count": 0,
            "nan_count": len(data),
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    return {
        "count": len(data),
        "valid_count": len(valid_data),
        "nan_count": len(data) - len(valid_data),
        "mean": float(np.mean(valid_data)),
        "std": float(np.std(valid_data)),
        "min": float(np.min(valid_data)),
        "max": float(np.max(valid_data)),
        "median": float(np.median(valid_data)),
    }
