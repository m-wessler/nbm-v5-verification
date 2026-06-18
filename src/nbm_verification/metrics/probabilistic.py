"""Probabilistic verification metrics (Brier Score, Reliability, CRPSS).

CRPS and CRPSS are computed from *percentile* (quantile) forecasts using the
``scoringrules`` library, which is the correct approach for NBM output that
stores forecast CDFs as a set of named percentile levels (e.g. 10th, 25th,
50th, 75th, 90th).

Expected array shape for quantile-based CRPS helpers::

    quantile_forecasts : (n_samples, n_quantiles)   – columns ordered by level
    quantile_levels    : (n_quantiles,)              – values in [0, 1]
    observations       : (n_samples,)
"""

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


def _try_import_scoringrules():
    """Return the scoringrules module, or None with a warning if unavailable."""
    try:
        import scoringrules as sr  # type: ignore[import]

        return sr
    except ImportError:
        return None


def compute_crps_from_quantiles(
    quantile_forecasts: np.ndarray,
    quantile_levels: np.ndarray,
    observations: np.ndarray,
) -> np.ndarray:
    """
    Compute per-sample CRPS from quantile (percentile) forecast CDFs.

    Uses ``scoringrules.crps_quantile`` when available; falls back to a
    trapezoidal-rule approximation otherwise.

    Parameters
    ----------
    quantile_forecasts : np.ndarray, shape (n_samples, n_quantiles)
        Forecast quantile values.  Columns must be ordered by ascending
        quantile level (i.e. column 0 corresponds to the smallest quantile).
    quantile_levels : np.ndarray, shape (n_quantiles,)
        Quantile levels in **[0, 1]** (e.g. ``[0.10, 0.25, 0.50, 0.75, 0.90]``).
        If values are in [1, 100] they are automatically rescaled to [0, 1].
    observations : np.ndarray, shape (n_samples,)
        Scalar observations.

    Returns
    -------
    np.ndarray, shape (n_samples,)
        Per-sample CRPS values (non-negative; 0 is a perfect forecast).

    Examples
    --------
    >>> levels = np.array([0.10, 0.25, 0.50, 0.75, 0.90])
    >>> fcsts  = np.array([[270, 272, 275, 278, 280]])   # one sample
    >>> obs    = np.array([276.0])
    >>> crps   = compute_crps_from_quantiles(fcsts, levels, obs)
    >>> float(crps[0]) > 0
    True
    """
    quantile_forecasts = np.asarray(quantile_forecasts, dtype=float)
    quantile_levels = np.asarray(quantile_levels, dtype=float)
    observations = np.asarray(observations, dtype=float)

    # Rescale percentages to [0, 1] if necessary
    if quantile_levels.max() > 1.0:
        quantile_levels = quantile_levels / 100.0

    # --- Try scoringrules first ---
    sr = _try_import_scoringrules()
    if sr is not None and hasattr(sr, "crps_quantile"):
        try:
            # scoringrules expects (n_samples, n_quantiles) and a 1-D levels vector
            return sr.crps_quantile(observations, quantile_forecasts, quantile_levels)
        except Exception:
            pass  # Fall through to manual implementation

    # --- Trapezoidal-rule fallback ---
    # CRPS = ∫_{-∞}^{∞} (F_hat(y) - 1{y >= obs})² dy
    # Approximated as weighted sum over quantile intervals.
    n_samples, n_q = quantile_forecasts.shape
    crps_vals = np.zeros(n_samples)

    for s in range(n_samples):
        q_vals = quantile_forecasts[s]
        o = observations[s]
        if np.isnan(o) or np.any(np.isnan(q_vals)):
            crps_vals[s] = np.nan
            continue

        # Quantile (pinball) loss decomposition of CRPS:
        #   CRPS = 2 * sum_k [ τ_k * max(o - q_k, 0)
        #                     + (1 - τ_k) * max(q_k - o, 0) ]
        # When o < q_k  → max(o - q_k, 0) = 0,  max(q_k - o, 0) = q_k - o
        # When o >= q_k → max(o - q_k, 0) = o - q_k, max(q_k - o, 0) = 0
        score = 0.0
        for k in range(n_q):
            tau = quantile_levels[k]
            delta = o - q_vals[k]
            if delta < 0:
                score += 2.0 * tau * (-delta)
            else:
                score += 2.0 * (1.0 - tau) * delta
        crps_vals[s] = score

    return crps_vals


def compute_mean_crps_from_quantiles(
    quantile_forecasts: np.ndarray,
    quantile_levels: np.ndarray,
    observations: np.ndarray,
) -> float:
    """
    Compute mean CRPS over all valid forecast-observation pairs.

    Parameters
    ----------
    quantile_forecasts : np.ndarray, shape (n_samples, n_quantiles)
        Quantile forecast values.
    quantile_levels : np.ndarray, shape (n_quantiles,)
        Quantile levels in [0, 1] or [1, 100].
    observations : np.ndarray, shape (n_samples,)
        Scalar observations.

    Returns
    -------
    float
        Mean CRPS (NaN if no valid pairs).

    Examples
    --------
    >>> levels = np.array([0.10, 0.25, 0.50, 0.75, 0.90])
    >>> fcsts  = np.tile([270, 272, 275, 278, 280], (5, 1)).astype(float)
    >>> obs    = np.array([274., 276., 275., 278., 281.])
    >>> mean_crps = compute_mean_crps_from_quantiles(fcsts, levels, obs)
    >>> mean_crps >= 0
    True
    """
    crps_vals = compute_crps_from_quantiles(quantile_forecasts, quantile_levels, observations)
    valid = ~np.isnan(crps_vals)
    if not np.any(valid):
        return np.nan
    return float(np.mean(crps_vals[valid]))


def compute_crpss(
    quantile_forecasts: np.ndarray,
    quantile_levels: np.ndarray,
    observations: np.ndarray,
    climatology_quantile_forecasts: Optional[np.ndarray] = None,
    climatology_quantile_levels: Optional[np.ndarray] = None,
) -> float:
    """
    Compute Continuous Ranked Probability Skill Score (CRPSS) from
    quantile-based forecasts.

    CRPSS = 1 - CRPS_forecast / CRPS_reference

    Parameters
    ----------
    quantile_forecasts : np.ndarray, shape (n_samples, n_quantiles)
        Forecast quantile values.
    quantile_levels : np.ndarray, shape (n_quantiles,)
        Quantile levels in [0, 1] or [1, 100].
    observations : np.ndarray, shape (n_samples,)
        Scalar observations.
    climatology_quantile_forecasts : np.ndarray, optional
        Reference (climatological) quantile forecasts.  If omitted, a simple
        sample-climatology is estimated from the observations.
    climatology_quantile_levels : np.ndarray, optional
        Quantile levels for the climatological forecasts.  Required when
        ``climatology_quantile_forecasts`` is provided.

    Returns
    -------
    float
        CRPSS (1 is perfect, 0 means equal to climatology, negative is worse).

    Notes
    -----
    When no explicit climatology is provided the reference CDF is constructed
    from the sample quantiles of ``observations`` at ``quantile_levels``.

    Examples
    --------
    >>> levels = np.array([0.10, 0.25, 0.50, 0.75, 0.90])
    >>> fcsts  = np.tile([270, 272, 275, 278, 280], (20, 1)).astype(float)
    >>> obs    = np.random.normal(275, 2, 20)
    >>> crpss  = compute_crpss(fcsts, levels, obs)
    >>> isinstance(crpss, float)
    True
    """
    quantile_levels = np.asarray(quantile_levels, dtype=float)
    if quantile_levels.max() > 1.0:
        quantile_levels = quantile_levels / 100.0

    observations = np.asarray(observations, dtype=float)
    valid_mask = ~np.isnan(observations)
    if not np.any(valid_mask):
        return np.nan

    crps_fcst = compute_mean_crps_from_quantiles(
        quantile_forecasts[valid_mask] if quantile_forecasts.ndim == 2 else quantile_forecasts,
        quantile_levels,
        observations[valid_mask],
    )

    if np.isnan(crps_fcst):
        return np.nan

    # Build reference climatological CDF from sample quantiles of observations
    if climatology_quantile_forecasts is None:
        climo_q = np.nanquantile(observations[valid_mask], quantile_levels)
        # Broadcast to same number of valid samples
        n_valid = int(np.sum(valid_mask))
        climo_fcsts = np.tile(climo_q, (n_valid, 1))
        climo_levels = quantile_levels
    else:
        climo_fcsts = np.asarray(climatology_quantile_forecasts, dtype=float)
        climo_levels = (
            np.asarray(climatology_quantile_levels, dtype=float)
            if climatology_quantile_levels is not None
            else quantile_levels
        )
        if climo_levels.max() > 1.0:
            climo_levels = climo_levels / 100.0
        if climo_fcsts.ndim == 1:
            climo_fcsts = np.tile(climo_fcsts, (int(np.sum(valid_mask)), 1))

    crps_ref = compute_mean_crps_from_quantiles(
        climo_fcsts, climo_levels, observations[valid_mask]
    )

    if np.isnan(crps_ref) or crps_ref == 0:
        return np.nan

    return float(1.0 - crps_fcst / crps_ref)


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
