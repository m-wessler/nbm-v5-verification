"""Categorical verification metrics (Hit Rate, FAR, CSI)."""

from typing import Dict, Tuple

import numpy as np


def compute_contingency_table(
    forecasts: np.ndarray, observations: np.ndarray, threshold: float
) -> Dict[str, int]:
    """
    Compute contingency table for a given threshold.

    Parameters
    ----------
    forecasts : np.ndarray
        Array of forecast values
    observations : np.ndarray
        Array of observation values
    threshold : float
        Threshold value for binary classification

    Returns
    -------
    Dict[str, int]
        Dictionary with hits, misses, false_alarms, correct_negatives

    Examples
    --------
    >>> forecasts = np.array([1.0, 2.0, 3.0, 4.0])
    >>> observations = np.array([1.5, 2.5, 2.5, 4.5])
    >>> ct = compute_contingency_table(forecasts, observations, 2.0)
    """
    # Filter out NaN values
    valid_mask = ~(np.isnan(forecasts) | np.isnan(observations))
    valid_forecasts = forecasts[valid_mask]
    valid_obs = observations[valid_mask]

    if len(valid_forecasts) == 0:
        return {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 0}

    # Binary classification
    forecast_yes = valid_forecasts >= threshold
    observed_yes = valid_obs >= threshold

    hits = np.sum(forecast_yes & observed_yes)
    misses = np.sum(~forecast_yes & observed_yes)
    false_alarms = np.sum(forecast_yes & ~observed_yes)
    correct_negatives = np.sum(~forecast_yes & ~observed_yes)

    return {
        "hits": int(hits),
        "misses": int(misses),
        "false_alarms": int(false_alarms),
        "correct_negatives": int(correct_negatives),
    }


def compute_hit_rate(contingency_table: Dict[str, int]) -> float:
    """
    Compute Hit Rate (Probability of Detection).

    HR = hits / (hits + misses)

    Parameters
    ----------
    contingency_table : Dict[str, int]
        Contingency table with hits, misses, false_alarms, correct_negatives

    Returns
    -------
    float
        Hit Rate (0-1)

    Examples
    --------
    >>> ct = {"hits": 10, "misses": 5, "false_alarms": 3, "correct_negatives": 20}
    >>> hr = compute_hit_rate(ct)
    """
    hits = contingency_table["hits"]
    misses = contingency_table["misses"]

    denominator = hits + misses
    if denominator == 0:
        return np.nan

    return hits / denominator


def compute_false_alarm_ratio(contingency_table: Dict[str, int]) -> float:
    """
    Compute False Alarm Ratio.

    FAR = false_alarms / (hits + false_alarms)

    Parameters
    ----------
    contingency_table : Dict[str, int]
        Contingency table with hits, misses, false_alarms, correct_negatives

    Returns
    -------
    float
        False Alarm Ratio (0-1)

    Examples
    --------
    >>> ct = {"hits": 10, "misses": 5, "false_alarms": 3, "correct_negatives": 20}
    >>> far = compute_false_alarm_ratio(ct)
    """
    hits = contingency_table["hits"]
    false_alarms = contingency_table["false_alarms"]

    denominator = hits + false_alarms
    if denominator == 0:
        return np.nan

    return false_alarms / denominator


def compute_critical_success_index(contingency_table: Dict[str, int]) -> float:
    """
    Compute Critical Success Index (Threat Score).

    CSI = hits / (hits + misses + false_alarms)

    Parameters
    ----------
    contingency_table : Dict[str, int]
        Contingency table with hits, misses, false_alarms, correct_negatives

    Returns
    -------
    float
        Critical Success Index (0-1)

    Examples
    --------
    >>> ct = {"hits": 10, "misses": 5, "false_alarms": 3, "correct_negatives": 20}
    >>> csi = compute_critical_success_index(ct)
    """
    hits = contingency_table["hits"]
    misses = contingency_table["misses"]
    false_alarms = contingency_table["false_alarms"]

    denominator = hits + misses + false_alarms
    if denominator == 0:
        return np.nan

    return hits / denominator


def compute_categorical_metrics(contingency_table: Dict[str, int]) -> Dict[str, float]:
    """
    Compute all categorical metrics from contingency table.

    Parameters
    ----------
    contingency_table : Dict[str, int]
        Contingency table with hits, misses, false_alarms, correct_negatives

    Returns
    -------
    Dict[str, float]
        Dictionary with HR, FAR, CSI, and contingency table counts

    Examples
    --------
    >>> ct = {"hits": 10, "misses": 5, "false_alarms": 3, "correct_negatives": 20}
    >>> metrics = compute_categorical_metrics(ct)
    >>> print(metrics['hit_rate'], metrics['false_alarm_ratio'], metrics['csi'])
    """
    return {
        "hit_rate": compute_hit_rate(contingency_table),
        "false_alarm_ratio": compute_false_alarm_ratio(contingency_table),
        "critical_success_index": compute_critical_success_index(contingency_table),
        **contingency_table,
    }
