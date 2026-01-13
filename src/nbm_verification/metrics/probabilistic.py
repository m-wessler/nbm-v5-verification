"""Probabilistic verification metrics (Brier Score, Reliability, CRPSS)."""

from typing import Dict, List, Optional, Tuple

import numpy as np


def compute_brier_score(
    probabilistic_forecasts: np.ndarray, binary_observations: np.ndarray
) -> float:
    """
    Compute Brier Score for probabilistic forecasts.

    BS = mean((probability - observation)^2)

    where observation is 0 or 1.

    Parameters
    ----------
    probabilistic_forecasts : np.ndarray
        Array of probabilistic forecast values (0-1)
    binary_observations : np.ndarray
        Array of binary observation values (0 or 1)

    Returns
    -------
    float
        Brier Score (lower is better, 0 is perfect)

    Examples
    --------
    >>> prob_fcst = np.array([0.1, 0.5, 0.9])
    >>> obs = np.array([0, 1, 1])
    >>> bs = compute_brier_score(prob_fcst, obs)
    """
    valid_mask = ~(np.isnan(probabilistic_forecasts) | np.isnan(binary_observations))
    valid_probs = probabilistic_forecasts[valid_mask]
    valid_obs = binary_observations[valid_mask]

    if len(valid_probs) == 0:
        return np.nan

    return np.mean((valid_probs - valid_obs) ** 2)


def compute_brier_skill_score(
    probabilistic_forecasts: np.ndarray,
    binary_observations: np.ndarray,
    climatology: Optional[float] = None,
) -> float:
    """
    Compute Brier Skill Score.

    BSS = 1 - (BS_forecast / BS_climatology)

    Parameters
    ----------
    probabilistic_forecasts : np.ndarray
        Array of probabilistic forecast values (0-1)
    binary_observations : np.ndarray
        Array of binary observation values (0 or 1)
    climatology : float, optional
        Climatological probability. If None, computed from observations.

    Returns
    -------
    float
        Brier Skill Score (1 is perfect, 0 is same as climatology, negative is worse)

    Examples
    --------
    >>> prob_fcst = np.array([0.1, 0.5, 0.9])
    >>> obs = np.array([0, 1, 1])
    >>> bss = compute_brier_skill_score(prob_fcst, obs)
    """
    bs_forecast = compute_brier_score(probabilistic_forecasts, binary_observations)

    if np.isnan(bs_forecast):
        return np.nan

    # Compute climatological Brier score if not provided
    if climatology is None:
        valid_mask = ~np.isnan(binary_observations)
        climatology = np.mean(binary_observations[valid_mask])

    # Climatological forecast is constant probability
    bs_climatology = np.mean((climatology - binary_observations) ** 2)

    if bs_climatology == 0:
        return np.nan

    return 1 - (bs_forecast / bs_climatology)


def compute_reliability(
    probabilistic_forecasts: np.ndarray,
    binary_observations: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Compute reliability diagram components.

    Parameters
    ----------
    probabilistic_forecasts : np.ndarray
        Array of probabilistic forecast values (0-1)
    binary_observations : np.ndarray
        Array of binary observation values (0 or 1)
    n_bins : int, optional
        Number of probability bins, by default 10

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with:
        - bin_centers: Center of each probability bin
        - observed_frequency: Observed frequency in each bin
        - forecast_probability: Mean forecast probability in each bin
        - counts: Number of forecasts in each bin

    Examples
    --------
    >>> prob_fcst = np.random.rand(100)
    >>> obs = np.random.randint(0, 2, 100)
    >>> reliability = compute_reliability(prob_fcst, obs)
    """
    valid_mask = ~(np.isnan(probabilistic_forecasts) | np.isnan(binary_observations))
    valid_probs = probabilistic_forecasts[valid_mask]
    valid_obs = binary_observations[valid_mask]

    if len(valid_probs) == 0:
        return {
            "bin_centers": np.full(n_bins, np.nan),
            "observed_frequency": np.full(n_bins, np.nan),
            "forecast_probability": np.full(n_bins, np.nan),
            "counts": np.zeros(n_bins, dtype=int),
        }

    # Create bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Digitize forecasts into bins
    bin_indices = np.digitize(valid_probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    observed_frequency = np.full(n_bins, np.nan)
    forecast_probability = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        mask = bin_indices == i
        counts[i] = np.sum(mask)

        if counts[i] > 0:
            observed_frequency[i] = np.mean(valid_obs[mask])
            forecast_probability[i] = np.mean(valid_probs[mask])

    return {
        "bin_centers": bin_centers,
        "observed_frequency": observed_frequency,
        "forecast_probability": forecast_probability,
        "counts": counts,
    }


def compute_crpss(
    ensemble_forecasts: np.ndarray,
    observations: np.ndarray,
    climatology_ensemble: Optional[np.ndarray] = None,
) -> float:
    """
    Compute Continuous Ranked Probability Skill Score.

    This is a placeholder implementation. Full CRPS calculation requires
    the properscoring library or similar.

    Parameters
    ----------
    ensemble_forecasts : np.ndarray
        Array of ensemble forecast values (shape: n_samples x n_members)
    observations : np.ndarray
        Array of observation values
    climatology_ensemble : np.ndarray, optional
        Climatological ensemble for reference

    Returns
    -------
    float
        CRPSS (1 is perfect, 0 is same as climatology)

    Notes
    -----
    This is a simplified placeholder. Full implementation should use
    the properscoring library: properscoring.crps_ensemble()
    """
    # This is a placeholder - full implementation would use properscoring
    # For now, return NaN to indicate not yet implemented
    return np.nan


def compute_roc_curve(
    probabilistic_forecasts: np.ndarray, binary_observations: np.ndarray, n_thresholds: int = 100
) -> Dict[str, np.ndarray]:
    """
    Compute ROC curve components.

    Parameters
    ----------
    probabilistic_forecasts : np.ndarray
        Array of probabilistic forecast values (0-1)
    binary_observations : np.ndarray
        Array of binary observation values (0 or 1)
    n_thresholds : int, optional
        Number of probability thresholds to evaluate

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with:
        - thresholds: Probability thresholds
        - hit_rates: Hit rate at each threshold
        - false_alarm_rates: False alarm rate at each threshold

    Examples
    --------
    >>> prob_fcst = np.random.rand(100)
    >>> obs = np.random.randint(0, 2, 100)
    >>> roc = compute_roc_curve(prob_fcst, obs)
    """
    valid_mask = ~(np.isnan(probabilistic_forecasts) | np.isnan(binary_observations))
    valid_probs = probabilistic_forecasts[valid_mask]
    valid_obs = binary_observations[valid_mask]

    if len(valid_probs) == 0:
        return {
            "thresholds": np.array([]),
            "hit_rates": np.array([]),
            "false_alarm_rates": np.array([]),
        }

    thresholds = np.linspace(0, 1, n_thresholds)
    hit_rates = np.zeros(n_thresholds)
    false_alarm_rates = np.zeros(n_thresholds)

    for i, threshold in enumerate(thresholds):
        forecast_yes = valid_probs >= threshold
        observed_yes = valid_obs >= 0.5  # Binary

        # True positives and false positives
        tp = np.sum(forecast_yes & observed_yes)
        fp = np.sum(forecast_yes & ~observed_yes)
        fn = np.sum(~forecast_yes & observed_yes)
        tn = np.sum(~forecast_yes & ~observed_yes)

        # Hit rate = TP / (TP + FN)
        if tp + fn > 0:
            hit_rates[i] = tp / (tp + fn)

        # False alarm rate = FP / (FP + TN)
        if fp + tn > 0:
            false_alarm_rates[i] = fp / (fp + tn)

    return {
        "thresholds": thresholds,
        "hit_rates": hit_rates,
        "false_alarm_rates": false_alarm_rates,
    }
