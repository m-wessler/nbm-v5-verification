"""
Unit tests for continuous verification metrics.
"""

import numpy as np
import pytest

from nbm_verification.metrics.continuous import (
    compute_mae,
    compute_bias,
    compute_rmse,
    compute_bias_ratio,
    compute_all_continuous_metrics,
)


class TestContinuousMetrics:
    """Test continuous verification metrics."""

    def test_mae_perfect_forecast(self):
        """Test MAE with perfect forecast."""
        forecasts = np.array([1.0, 2.0, 3.0])
        observations = np.array([1.0, 2.0, 3.0])
        mae = compute_mae(forecasts, observations)
        assert mae == 0.0

    def test_mae_simple_case(self):
        """Test MAE with known values."""
        forecasts = np.array([1.0, 2.0, 3.0])
        observations = np.array([1.5, 2.5, 2.5])
        mae = compute_mae(forecasts, observations)
        expected = (0.5 + 0.5 + 0.5) / 3
        assert abs(mae - expected) < 1e-10

    def test_bias_zero(self):
        """Test bias with zero bias."""
        forecasts = np.array([1.0, 2.0, 3.0])
        observations = np.array([1.0, 2.0, 3.0])
        bias = compute_bias(forecasts, observations)
        assert bias == 0.0

    def test_bias_positive(self):
        """Test bias with positive (warm) bias."""
        forecasts = np.array([2.0, 3.0, 4.0])
        observations = np.array([1.0, 2.0, 3.0])
        bias = compute_bias(forecasts, observations)
        assert bias == 1.0

    def test_rmse_perfect_forecast(self):
        """Test RMSE with perfect forecast."""
        forecasts = np.array([1.0, 2.0, 3.0])
        observations = np.array([1.0, 2.0, 3.0])
        rmse = compute_rmse(forecasts, observations)
        assert rmse == 0.0

    def test_compute_all_continuous_metrics(self):
        """Test computing all metrics at once."""
        forecasts = np.array([1.0, 2.0, 3.0])
        observations = np.array([1.5, 2.5, 2.5])

        metrics = compute_all_continuous_metrics(forecasts, observations)

        assert "mae" in metrics
        assert "bias" in metrics
        assert "rmse" in metrics
        assert "bias_ratio" in metrics
