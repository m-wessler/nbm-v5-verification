"""Base statistics accumulator class for verification metrics."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class StatisticsAccumulator(ABC):
    """
    Base class for accumulating raw statistical components across spatial/temporal chunks.

    This class implements the accumulator pattern for efficient memory usage when
    processing large datasets. Instead of storing all forecast-observation pairs,
    we accumulate statistical components (sums, counts, contingency tables) that
    can be merged across chunks and used to compute final metrics.

    Attributes
    ----------
    n_samples : int
        Number of valid forecast-observation pairs
    sum_forecast : float
        Sum of all forecast values
    sum_obs : float
        Sum of all observation values
    sum_abs_error : float
        Sum of absolute errors |forecast - observation|
    sum_squared_error : float
        Sum of squared errors (forecast - observation)²
    sum_forecast_squared : float
        Sum of squared forecast values (for variance calculations)
    sum_obs_squared : float
        Sum of squared observation values (for variance calculations)
    contingency_tables : Dict[float, Dict[str, int]]
        Contingency table elements for each threshold (for categorical metrics)
    """

    def __init__(self, thresholds: Optional[List[float]] = None):
        """
        Initialize the accumulator.

        Parameters
        ----------
        thresholds : List[float], optional
            List of thresholds for categorical metrics, by default None
        """
        self.n_samples = 0
        self.sum_forecast = 0.0
        self.sum_obs = 0.0
        self.sum_abs_error = 0.0
        self.sum_squared_error = 0.0
        self.sum_forecast_squared = 0.0
        self.sum_obs_squared = 0.0

        # Initialize contingency tables for each threshold
        self.thresholds = thresholds or []
        self.contingency_tables: Dict[float, Dict[str, int]] = {}
        for threshold in self.thresholds:
            self.contingency_tables[threshold] = {
                "hits": 0,  # forecast yes, observed yes
                "misses": 0,  # forecast no, observed yes
                "false_alarms": 0,  # forecast yes, observed no
                "correct_negatives": 0,  # forecast no, observed no
            }

        # For probabilistic metrics
        self.probabilistic_components: Dict[str, Any] = {}

    def update(
        self,
        forecasts: np.ndarray,
        observations: np.ndarray,
        probabilistic_forecasts: Optional[np.ndarray] = None,
        quantile_levels: Optional[np.ndarray] = None,
    ) -> None:
        """
        Update accumulator with new forecast-observation pairs.

        Parameters
        ----------
        forecasts : np.ndarray
            Array of deterministic forecast values (shape: ``(n,)``).
        observations : np.ndarray
            Array of observation values (same shape as ``forecasts``).
        probabilistic_forecasts : np.ndarray, optional
            Array of probabilistic forecast values.  Two shapes are supported:

            * ``(n,)``  – pre-computed exceedance probabilities (0–1) for Brier
              Score.
            * ``(n, n_quantiles)`` – quantile forecast CDF for CRPS.  In this
              case ``quantile_levels`` must also be provided.

        quantile_levels : np.ndarray, optional
            Quantile levels corresponding to the last axis of
            ``probabilistic_forecasts`` when it is 2-D.  Values may be in
            [0, 1] or [1, 100]; values > 1 are automatically rescaled.

        Notes
        -----
        Only valid (non-NaN) pairs are accumulated.  Missing data is
        automatically filtered out before any statistic is updated.
        """
        # Ensure arrays are numpy arrays
        forecasts = np.asarray(forecasts, dtype=float)
        observations = np.asarray(observations, dtype=float)

        # Create mask for valid pairs (both forecast and observation are non-NaN)
        valid_mask = ~(np.isnan(forecasts) | np.isnan(observations))

        # Extract valid pairs
        valid_forecasts = forecasts[valid_mask]
        valid_observations = observations[valid_mask]

        # Update sample count
        n_valid = int(np.sum(valid_mask))
        self.n_samples += n_valid

        if n_valid == 0:
            return  # No valid pairs to process

        # Update continuous statistics
        self.sum_forecast += float(np.sum(valid_forecasts))
        self.sum_obs += float(np.sum(valid_observations))
        self.sum_forecast_squared += float(np.sum(valid_forecasts**2))
        self.sum_obs_squared += float(np.sum(valid_observations**2))

        # Calculate errors
        errors = valid_forecasts - valid_observations
        self.sum_abs_error += float(np.sum(np.abs(errors)))
        self.sum_squared_error += float(np.sum(errors**2))

        # Update contingency tables for categorical metrics
        for threshold in self.thresholds:
            forecast_yes = valid_forecasts >= threshold
            observed_yes = valid_observations >= threshold

            self.contingency_tables[threshold]["hits"] += int(np.sum(forecast_yes & observed_yes))
            self.contingency_tables[threshold]["misses"] += int(np.sum(~forecast_yes & observed_yes))
            self.contingency_tables[threshold]["false_alarms"] += int(np.sum(
                forecast_yes & ~observed_yes
            ))
            self.contingency_tables[threshold]["correct_negatives"] += int(np.sum(
                ~forecast_yes & ~observed_yes
            ))

        # Update probabilistic components if provided
        if probabilistic_forecasts is not None:
            prob_arr = np.asarray(probabilistic_forecasts, dtype=float)
            if prob_arr.ndim == 2:
                # Quantile CDF forecasts → CRPS accumulation
                if quantile_levels is not None:
                    self._update_crps(
                        prob_arr[valid_mask], quantile_levels, valid_observations
                    )
            else:
                # 1-D exceedance probability → Brier Score accumulation
                self._update_probabilistic(prob_arr[valid_mask], valid_observations)

    def _update_crps(
        self,
        quantile_forecasts: np.ndarray,
        quantile_levels: np.ndarray,
        observations: np.ndarray,
    ) -> None:
        """
        Accumulate CRPS components from quantile (percentile) forecast CDFs.

        Parameters
        ----------
        quantile_forecasts : np.ndarray, shape ``(n, n_quantiles)``
            Quantile forecast values for ``n`` valid samples.
        quantile_levels : np.ndarray, shape ``(n_quantiles,)``
            Quantile levels in [0, 1] or [1, 100].
        observations : np.ndarray, shape ``(n,)``
            Observation values for the same ``n`` valid samples.
        """
        from nbm_verification.metrics.probabilistic import compute_crps_from_quantiles

        crps_vals = compute_crps_from_quantiles(quantile_forecasts, quantile_levels, observations)
        valid = ~np.isnan(crps_vals)

        if "crps_sum" not in self.probabilistic_components:
            self.probabilistic_components["crps_sum"] = 0.0
            self.probabilistic_components["crps_n"] = 0
            # Store quantile levels so merge can validate compatibility
            self.probabilistic_components["quantile_levels"] = np.asarray(
                quantile_levels, dtype=float
            )

        self.probabilistic_components["crps_sum"] += float(np.sum(crps_vals[valid]))
        self.probabilistic_components["crps_n"] += int(np.sum(valid))

    def _update_probabilistic(
        self, prob_forecasts: np.ndarray, observations: np.ndarray
    ) -> None:
        """
        Update probabilistic forecast components.

        Parameters
        ----------
        prob_forecasts : np.ndarray
            Probabilistic forecast values (0-1)
        observations : np.ndarray
            Binary observation values (0 or 1)
        """
        # For Brier Score: sum of (probability - observation)^2
        if "brier_sum" not in self.probabilistic_components:
            self.probabilistic_components["brier_sum"] = 0.0

        brier_errors = (prob_forecasts - observations) ** 2
        self.probabilistic_components["brier_sum"] += np.sum(brier_errors)

        # For reliability diagrams, we'd need to bin probabilities
        # This is a simplified version - full implementation would track bins
        if "prob_bins" not in self.probabilistic_components:
            self.probabilistic_components["prob_bins"] = {}

    def merge(self, other: "StatisticsAccumulator") -> None:
        """
        Merge another accumulator into this one.

        This allows combining statistics from different spatial chunks or time periods.

        Parameters
        ----------
        other : StatisticsAccumulator
            Another accumulator to merge into this one

        Examples
        --------
        >>> acc1 = StatisticsAccumulator([273.15])
        >>> acc2 = StatisticsAccumulator([273.15])
        >>> acc1.update(forecast1, obs1)
        >>> acc2.update(forecast2, obs2)
        >>> acc1.merge(acc2)  # Now acc1 contains statistics from both
        """
        self.n_samples += other.n_samples
        self.sum_forecast += other.sum_forecast
        self.sum_obs += other.sum_obs
        self.sum_abs_error += other.sum_abs_error
        self.sum_squared_error += other.sum_squared_error
        self.sum_forecast_squared += other.sum_forecast_squared
        self.sum_obs_squared += other.sum_obs_squared

        # Merge contingency tables
        for threshold in self.thresholds:
            if threshold in other.contingency_tables:
                for key in ["hits", "misses", "false_alarms", "correct_negatives"]:
                    self.contingency_tables[threshold][key] += other.contingency_tables[
                        threshold
                    ][key]

        # Merge probabilistic components
        if "brier_sum" in other.probabilistic_components:
            if "brier_sum" not in self.probabilistic_components:
                self.probabilistic_components["brier_sum"] = 0.0
            self.probabilistic_components["brier_sum"] += other.probabilistic_components[
                "brier_sum"
            ]

        if "crps_sum" in other.probabilistic_components:
            if "crps_sum" not in self.probabilistic_components:
                self.probabilistic_components["crps_sum"] = 0.0
                self.probabilistic_components["crps_n"] = 0

            # Warn if the quantile levels used to compute CRPS differ between
            # the two accumulators.  Merging CRPS values built from different
            # quantile sets is not statistically meaningful.
            self_levels = self.probabilistic_components.get("quantile_levels")
            other_levels = other.probabilistic_components.get("quantile_levels")
            if self_levels is not None and other_levels is not None:
                if not np.array_equal(
                    np.sort(self_levels), np.sort(np.asarray(other_levels, dtype=float))
                ):
                    logger.warning(
                        "Merging CRPS accumulators with different quantile levels "
                        "(%s vs %s). The merged CRPS will not be meaningful.",
                        self_levels.tolist(),
                        np.asarray(other_levels).tolist(),
                    )

            self.probabilistic_components["crps_sum"] += other.probabilistic_components["crps_sum"]
            self.probabilistic_components["crps_n"] += other.probabilistic_components["crps_n"]

    @abstractmethod
    def compute_metrics(self) -> Dict[str, Any]:
        """
        Compute final metrics from accumulated statistics.

        This method must be implemented by subclasses to compute
        specific metrics based on the accumulated components.

        Returns
        -------
        Dict[str, Any]
            Dictionary of computed metrics
        """
        pass

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of accumulated statistics.

        Returns
        -------
        Dict[str, Any]
            Summary statistics including sample count and basic components
        """
        return {
            "n_samples": self.n_samples,
            "mean_forecast": self.sum_forecast / self.n_samples if self.n_samples > 0 else np.nan,
            "mean_obs": self.sum_obs / self.n_samples if self.n_samples > 0 else np.nan,
            "thresholds": self.thresholds,
        }
