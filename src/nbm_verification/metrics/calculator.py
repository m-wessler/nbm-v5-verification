"""Metric computation orchestration."""

from typing import Any, Dict, List, Optional

import numpy as np

from nbm_verification.accumulation.accumulator_base import StatisticsAccumulator
from nbm_verification.metrics.categorical import compute_categorical_metrics
from nbm_verification.metrics.continuous import compute_all_continuous_metrics
from nbm_verification.metrics.probabilistic import (
    compute_brier_score,
    compute_reliability,
)


class MetricsCalculator:
    """
    Orchestrates the calculation of verification metrics from accumulators.

    This class takes accumulated statistics and computes final metrics
    based on the requested metric types.
    """

    CONTINUOUS_METRICS = ["mae", "bias", "rmse", "bias_ratio"]
    CATEGORICAL_METRICS = ["hr", "far", "csi"]
    PROBABILISTIC_METRICS = ["brier_score", "reliability", "crpss"]

    def __init__(self, requested_metrics: Optional[List[str]] = None):
        """
        Initialize metrics calculator.

        Parameters
        ----------
        requested_metrics : List[str], optional
            List of metrics to compute. If None, computes all available metrics.
        """
        self.requested_metrics = requested_metrics

    def compute_from_accumulator(
        self, accumulator: StatisticsAccumulator
    ) -> Dict[str, Any]:
        """
        Compute metrics from an accumulator.

        This is the main entry point for computing metrics. It delegates
        to the accumulator's compute_metrics method.

        Parameters
        ----------
        accumulator : StatisticsAccumulator
            Accumulator containing statistics to compute metrics from

        Returns
        -------
        Dict[str, Any]
            Dictionary of computed metrics

        Examples
        --------
        >>> calculator = MetricsCalculator()
        >>> metrics = calculator.compute_from_accumulator(accumulator)
        """
        # Use the accumulator's compute_metrics method
        metrics = accumulator.compute_metrics()

        # Filter to requested metrics if specified
        if self.requested_metrics is not None:
            metrics = self._filter_metrics(metrics)

        return metrics

    def compute_from_arrays(
        self,
        forecasts: np.ndarray,
        observations: np.ndarray,
        thresholds: Optional[List[float]] = None,
        probabilistic_forecasts: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Compute metrics directly from forecast and observation arrays.

        This is a convenience method for one-time metric calculation
        without using the accumulator pattern.

        Parameters
        ----------
        forecasts : np.ndarray
            Array of forecast values
        observations : np.ndarray
            Array of observation values
        thresholds : List[float], optional
            Thresholds for categorical metrics
        probabilistic_forecasts : np.ndarray, optional
            Probabilistic forecast values for probabilistic metrics

        Returns
        -------
        Dict[str, Any]
            Dictionary of computed metrics
        """
        metrics = {}

        # Compute continuous metrics
        if self._should_compute_continuous():
            continuous = compute_all_continuous_metrics(forecasts, observations)
            metrics.update(continuous)

        # Compute categorical metrics
        if self._should_compute_categorical() and thresholds:
            from nbm_verification.metrics.categorical import compute_contingency_table

            metrics["categorical"] = {}
            for threshold in thresholds:
                ct = compute_contingency_table(forecasts, observations, threshold)
                cat_metrics = compute_categorical_metrics(ct)
                metrics["categorical"][threshold] = cat_metrics

        # Compute probabilistic metrics
        if self._should_compute_probabilistic() and probabilistic_forecasts is not None:
            # Convert to binary for Brier score
            if thresholds and len(thresholds) > 0:
                binary_obs = (observations >= thresholds[0]).astype(float)
                metrics["brier_score"] = compute_brier_score(
                    probabilistic_forecasts, binary_obs
                )
                metrics["reliability"] = compute_reliability(
                    probabilistic_forecasts, binary_obs
                )

        return metrics

    def _should_compute_continuous(self) -> bool:
        """Check if continuous metrics should be computed."""
        if self.requested_metrics is None:
            return True
        return any(m in self.requested_metrics for m in self.CONTINUOUS_METRICS)

    def _should_compute_categorical(self) -> bool:
        """Check if categorical metrics should be computed."""
        if self.requested_metrics is None:
            return True
        return any(m in self.requested_metrics for m in self.CATEGORICAL_METRICS)

    def _should_compute_probabilistic(self) -> bool:
        """Check if probabilistic metrics should be computed."""
        if self.requested_metrics is None:
            return True
        return any(m in self.requested_metrics for m in self.PROBABILISTIC_METRICS)

    def _filter_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter metrics to only include requested ones.

        Parameters
        ----------
        metrics : Dict[str, Any]
            Full metrics dictionary

        Returns
        -------
        Dict[str, Any]
            Filtered metrics dictionary
        """
        filtered = {}

        for key, value in metrics.items():
            # Always include metadata fields
            if key in [
                "grid_i",
                "grid_j",
                "lat",
                "lon",
                "n_samples",
                "region_id",
                "region_type",
                "region_name",
                "station_id",
                "station_name",
                "elevation",
                "network",
            ]:
                filtered[key] = value
            # Include requested metrics
            elif key in self.requested_metrics:
                filtered[key] = value
            # Handle categorical metrics specially
            elif key == "categorical" and self._should_compute_categorical():
                filtered[key] = value

        return filtered
