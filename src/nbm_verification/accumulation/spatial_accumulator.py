"""Spatial accumulators for gridpoint, regional, and station verification."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from nbm_verification.accumulation.accumulator_base import StatisticsAccumulator


class GridpointAccumulator(StatisticsAccumulator):
    """
    Accumulator for individual gridpoint statistics.

    Stores statistics for a single grid cell, accumulating data across
    multiple time steps.
    """

    def __init__(
        self,
        grid_i: int,
        grid_j: int,
        lat: float,
        lon: float,
        thresholds: Optional[List[float]] = None,
    ):
        """
        Initialize gridpoint accumulator.

        Parameters
        ----------
        grid_i : int
            Grid i-index
        grid_j : int
            Grid j-index
        lat : float
            Latitude of gridpoint
        lon : float
            Longitude of gridpoint
        thresholds : List[float], optional
            Thresholds for categorical metrics
        """
        super().__init__(thresholds)
        self.grid_i = grid_i
        self.grid_j = grid_j
        self.lat = lat
        self.lon = lon

    def compute_metrics(self) -> Dict[str, Any]:
        """
        Compute verification metrics for this gridpoint.

        Returns
        -------
        Dict[str, Any]
            Dictionary of computed metrics including MAE, bias, RMSE, etc.
        """
        if self.n_samples == 0:
            return self._empty_metrics()

        metrics = {
            "grid_i": self.grid_i,
            "grid_j": self.grid_j,
            "lat": self.lat,
            "lon": self.lon,
            "n_samples": self.n_samples,
        }

        # Continuous metrics
        metrics["mae"] = self.sum_abs_error / self.n_samples
        metrics["bias"] = (self.sum_forecast - self.sum_obs) / self.n_samples
        metrics["rmse"] = np.sqrt(self.sum_squared_error / self.n_samples)

        mean_forecast = self.sum_forecast / self.n_samples
        mean_obs = self.sum_obs / self.n_samples

        # Bias ratio (avoid division by zero)
        if abs(mean_obs) > 1e-10:
            metrics["bias_ratio"] = mean_forecast / mean_obs
        else:
            metrics["bias_ratio"] = np.nan

        # Categorical metrics for each threshold
        metrics["categorical"] = {}
        for threshold, ct in self.contingency_tables.items():
            threshold_metrics = self._compute_categorical_metrics(ct)
            metrics["categorical"][threshold] = threshold_metrics

        # Probabilistic metrics
        if "brier_sum" in self.probabilistic_components:
            metrics["brier_score"] = (
                self.probabilistic_components["brier_sum"] / self.n_samples
            )

        return metrics

    def _compute_categorical_metrics(self, contingency_table: Dict[str, int]) -> Dict[str, float]:
        """
        Compute categorical metrics from contingency table.

        Parameters
        ----------
        contingency_table : Dict[str, int]
            Contingency table with hits, misses, false_alarms, correct_negatives

        Returns
        -------
        Dict[str, float]
            Dictionary with HR, FAR, CSI
        """
        hits = contingency_table["hits"]
        misses = contingency_table["misses"]
        false_alarms = contingency_table["false_alarms"]

        # Hit Rate (HR) = hits / (hits + misses)
        denominator = hits + misses
        hr = hits / denominator if denominator > 0 else np.nan

        # False Alarm Ratio (FAR) = false_alarms / (hits + false_alarms)
        denominator = hits + false_alarms
        far = false_alarms / denominator if denominator > 0 else np.nan

        # Critical Success Index (CSI) = hits / (hits + misses + false_alarms)
        denominator = hits + misses + false_alarms
        csi = hits / denominator if denominator > 0 else np.nan

        return {
            "hit_rate": hr,
            "false_alarm_ratio": far,
            "critical_success_index": csi,
            **contingency_table,
        }

    def _empty_metrics(self) -> Dict[str, Any]:
        """Return NaN metrics for empty accumulator."""
        return {
            "grid_i": self.grid_i,
            "grid_j": self.grid_j,
            "lat": self.lat,
            "lon": self.lon,
            "n_samples": 0,
            "mae": np.nan,
            "bias": np.nan,
            "rmse": np.nan,
            "bias_ratio": np.nan,
            "categorical": {},
        }


class RegionalAccumulator(StatisticsAccumulator):
    """
    Accumulator for regional statistics (CWA, RFC, Zone).

    Accumulates statistics across all gridpoints within a region,
    treating them as one statistical population.
    """

    def __init__(
        self,
        region_id: str,
        region_type: str,
        region_name: str,
        thresholds: Optional[List[float]] = None,
    ):
        """
        Initialize regional accumulator.

        Parameters
        ----------
        region_id : str
            Unique identifier for the region
        region_type : str
            Type of region (CWA, RFC, Zone)
        region_name : str
            Human-readable name of the region
        thresholds : List[float], optional
            Thresholds for categorical metrics
        """
        super().__init__(thresholds)
        self.region_id = region_id
        self.region_type = region_type
        self.region_name = region_name

    def compute_metrics(self) -> Dict[str, Any]:
        """
        Compute verification metrics for this region.

        Returns
        -------
        Dict[str, Any]
            Dictionary of computed metrics
        """
        if self.n_samples == 0:
            return self._empty_metrics()

        metrics = {
            "region_id": self.region_id,
            "region_type": self.region_type,
            "region_name": self.region_name,
            "n_samples": self.n_samples,
        }

        # Same continuous metrics as gridpoint
        metrics["mae"] = self.sum_abs_error / self.n_samples
        metrics["bias"] = (self.sum_forecast - self.sum_obs) / self.n_samples
        metrics["rmse"] = np.sqrt(self.sum_squared_error / self.n_samples)

        mean_forecast = self.sum_forecast / self.n_samples
        mean_obs = self.sum_obs / self.n_samples

        if abs(mean_obs) > 1e-10:
            metrics["bias_ratio"] = mean_forecast / mean_obs
        else:
            metrics["bias_ratio"] = np.nan

        # Categorical metrics
        metrics["categorical"] = {}
        for threshold, ct in self.contingency_tables.items():
            threshold_metrics = self._compute_categorical_metrics(ct)
            metrics["categorical"][threshold] = threshold_metrics

        return metrics

    def _compute_categorical_metrics(self, contingency_table: Dict[str, int]) -> Dict[str, float]:
        """Compute categorical metrics from contingency table."""
        hits = contingency_table["hits"]
        misses = contingency_table["misses"]
        false_alarms = contingency_table["false_alarms"]

        denominator = hits + misses
        hr = hits / denominator if denominator > 0 else np.nan

        denominator = hits + false_alarms
        far = false_alarms / denominator if denominator > 0 else np.nan

        denominator = hits + misses + false_alarms
        csi = hits / denominator if denominator > 0 else np.nan

        return {
            "hit_rate": hr,
            "false_alarm_ratio": far,
            "critical_success_index": csi,
            **contingency_table,
        }

    def _empty_metrics(self) -> Dict[str, Any]:
        """Return NaN metrics for empty accumulator."""
        return {
            "region_id": self.region_id,
            "region_type": self.region_type,
            "region_name": self.region_name,
            "n_samples": 0,
            "mae": np.nan,
            "bias": np.nan,
            "rmse": np.nan,
            "bias_ratio": np.nan,
            "categorical": {},
        }


class StationAccumulator(StatisticsAccumulator):
    """
    Accumulator for station-based verification.

    Accumulates statistics for point observations at a specific station,
    comparing against the nearest gridpoint forecast.
    """

    def __init__(
        self,
        station_id: str,
        station_name: str,
        lat: float,
        lon: float,
        elevation: Optional[float] = None,
        network: Optional[str] = None,
        thresholds: Optional[List[float]] = None,
    ):
        """
        Initialize station accumulator.

        Parameters
        ----------
        station_id : str
            Unique station identifier
        station_name : str
            Station name
        lat : float
            Station latitude
        lon : float
            Station longitude
        elevation : float, optional
            Station elevation in meters
        network : str, optional
            Observing network (e.g., ASOS, AWOS)
        thresholds : List[float], optional
            Thresholds for categorical metrics
        """
        super().__init__(thresholds)
        self.station_id = station_id
        self.station_name = station_name
        self.lat = lat
        self.lon = lon
        self.elevation = elevation
        self.network = network

    def compute_metrics(self) -> Dict[str, Any]:
        """
        Compute verification metrics for this station.

        Returns
        -------
        Dict[str, Any]
            Dictionary of computed metrics
        """
        if self.n_samples == 0:
            return self._empty_metrics()

        metrics = {
            "station_id": self.station_id,
            "station_name": self.station_name,
            "lat": self.lat,
            "lon": self.lon,
            "elevation": self.elevation,
            "network": self.network,
            "n_samples": self.n_samples,
        }

        # Same continuous metrics
        metrics["mae"] = self.sum_abs_error / self.n_samples
        metrics["bias"] = (self.sum_forecast - self.sum_obs) / self.n_samples
        metrics["rmse"] = np.sqrt(self.sum_squared_error / self.n_samples)

        mean_forecast = self.sum_forecast / self.n_samples
        mean_obs = self.sum_obs / self.n_samples

        if abs(mean_obs) > 1e-10:
            metrics["bias_ratio"] = mean_forecast / mean_obs
        else:
            metrics["bias_ratio"] = np.nan

        # Categorical metrics
        metrics["categorical"] = {}
        for threshold, ct in self.contingency_tables.items():
            threshold_metrics = self._compute_categorical_metrics(ct)
            metrics["categorical"][threshold] = threshold_metrics

        return metrics

    def _compute_categorical_metrics(self, contingency_table: Dict[str, int]) -> Dict[str, float]:
        """Compute categorical metrics from contingency table."""
        hits = contingency_table["hits"]
        misses = contingency_table["misses"]
        false_alarms = contingency_table["false_alarms"]

        denominator = hits + misses
        hr = hits / denominator if denominator > 0 else np.nan

        denominator = hits + false_alarms
        far = false_alarms / denominator if denominator > 0 else np.nan

        denominator = hits + misses + false_alarms
        csi = hits / denominator if denominator > 0 else np.nan

        return {
            "hit_rate": hr,
            "false_alarm_ratio": far,
            "critical_success_index": csi,
            **contingency_table,
        }

    def _empty_metrics(self) -> Dict[str, Any]:
        """Return NaN metrics for empty accumulator."""
        return {
            "station_id": self.station_id,
            "station_name": self.station_name,
            "lat": self.lat,
            "lon": self.lon,
            "elevation": self.elevation,
            "network": self.network,
            "n_samples": 0,
            "mae": np.nan,
            "bias": np.nan,
            "rmse": np.nan,
            "bias_ratio": np.nan,
            "categorical": {},
        }
