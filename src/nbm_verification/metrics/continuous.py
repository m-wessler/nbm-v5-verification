"""Continuous verification metrics (MAE, Bias, RMSE, Bias Ratio)."""

from typing import Tuple

import numpy as np


def compute_mae(forecasts: np.ndarray, observations: np.ndarray) -> float:
    """
    Compute Mean Absolute Error.

    MAE = mean(|forecast - observation|)

    Parameters
    ----------
    forecasts : np.ndarray
        Array of forecast values
    observations : np.ndarray
        Array of observation values

    Returns
    -------
    float
        Mean Absolute Error

    Examples
    --------
    >>> forecasts = np.array([1.0, 2.0, 3.0])
    >>> observations = np.array([1.1, 2.2, 2.8])
    >>> mae = compute_mae(forecasts, observations)
    """
    # Filter out NaN values
    valid_mask = ~(np.isnan(forecasts) | np.isnan(observations))
    valid_forecasts = forecasts[valid_mask]
    valid_obs = observations[valid_mask]

    if len(valid_forecasts) == 0:
        return np.nan

    return np.mean(np.abs(valid_forecasts - valid_obs))


def compute_bias(forecasts: np.ndarray, observations: np.ndarray) -> float:
    """
    Compute Mean Error (Bias).

    Bias = mean(forecast - observation)

    Parameters
    ----------
    forecasts : np.ndarray
        Array of forecast values
    observations : np.ndarray
        Array of observation values

    Returns
    -------
    float
        Mean Error (Bias)

    Examples
    --------
    >>> forecasts = np.array([1.0, 2.0, 3.0])
    >>> observations = np.array([1.1, 2.2, 2.8])
    >>> bias = compute_bias(forecasts, observations)
    """
    valid_mask = ~(np.isnan(forecasts) | np.isnan(observations))
    valid_forecasts = forecasts[valid_mask]
    valid_obs = observations[valid_mask]

    if len(valid_forecasts) == 0:
        return np.nan

    return np.mean(valid_forecasts - valid_obs)


def compute_rmse(forecasts: np.ndarray, observations: np.ndarray) -> float:
    """
    Compute Root Mean Square Error.

    RMSE = sqrt(mean((forecast - observation)^2))

    Parameters
    ----------
    forecasts : np.ndarray
        Array of forecast values
    observations : np.ndarray
        Array of observation values

    Returns
    -------
    float
        Root Mean Square Error

    Examples
    --------
    >>> forecasts = np.array([1.0, 2.0, 3.0])
    >>> observations = np.array([1.1, 2.2, 2.8])
    >>> rmse = compute_rmse(forecasts, observations)
    """
    valid_mask = ~(np.isnan(forecasts) | np.isnan(observations))
    valid_forecasts = forecasts[valid_mask]
    valid_obs = observations[valid_mask]

    if len(valid_forecasts) == 0:
        return np.nan

    return np.sqrt(np.mean((valid_forecasts - valid_obs) ** 2))


def compute_bias_ratio(forecasts: np.ndarray, observations: np.ndarray) -> float:
    """
    Compute Bias Ratio.

    Bias Ratio = mean(forecast) / mean(observation)

    Parameters
    ----------
    forecasts : np.ndarray
        Array of forecast values
    observations : np.ndarray
        Array of observation values

    Returns
    -------
    float
        Bias Ratio

    Examples
    --------
    >>> forecasts = np.array([1.0, 2.0, 3.0])
    >>> observations = np.array([1.1, 2.2, 2.8])
    >>> bias_ratio = compute_bias_ratio(forecasts, observations)
    """
    valid_mask = ~(np.isnan(forecasts) | np.isnan(observations))
    valid_forecasts = forecasts[valid_mask]
    valid_obs = observations[valid_mask]

    if len(valid_forecasts) == 0:
        return np.nan

    mean_obs = np.mean(valid_obs)
    if abs(mean_obs) < 1e-10:
        return np.nan

    return np.mean(valid_forecasts) / mean_obs


def compute_all_continuous_metrics(
    forecasts: np.ndarray, observations: np.ndarray
) -> dict:
    """
    Compute all continuous verification metrics.

    Parameters
    ----------
    forecasts : np.ndarray
        Array of forecast values
    observations : np.ndarray
        Array of observation values

    Returns
    -------
    dict
        Dictionary containing MAE, Bias, RMSE, and Bias Ratio

    Examples
    --------
    >>> forecasts = np.array([1.0, 2.0, 3.0])
    >>> observations = np.array([1.1, 2.2, 2.8])
    >>> metrics = compute_all_continuous_metrics(forecasts, observations)
    >>> print(metrics['mae'])
    """
    return {
        "mae": compute_mae(forecasts, observations),
        "bias": compute_bias(forecasts, observations),
        "rmse": compute_rmse(forecasts, observations),
        "bias_ratio": compute_bias_ratio(forecasts, observations),
    }
