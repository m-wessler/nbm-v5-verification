"""
Tests for NBMXarrayReader and updated probabilistic metrics (CRPS/CRPSS).

These tests use synthetic in-memory xarray Datasets that mirror the actual
NBM dataset coordinate structure described in the PR comment::

    ds.init_time   – timestamps
    ds.f_hour      – integer forecast-hour offsets
    ds.percentile  – percentile levels [10, 25, 50, 75, 90]
    ds.y / ds.x    – spatial row / column indices
    ds["maxt"]     – maximum temperature (init_time, f_hour, percentile, y, x)
    ds["mint"]     – minimum temperature (init_time, f_hour, percentile, y, x)
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from nbm_verification.data_access.nbm_xarray_reader import NBMXarrayReader
from nbm_verification.metrics.probabilistic import (
    compute_crps_from_quantiles,
    compute_crpss,
    compute_mean_crps_from_quantiles,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PERCENTILES = [10.0, 25.0, 50.0, 75.0, 90.0]
F_HOURS = [6, 12, 24, 48]
NY, NX = 20, 30


def _make_nbm_dataset(n_inits: int = 3) -> xr.Dataset:
    """
    Build a small synthetic NBM Dataset with the expected coordinate structure.

    The ``maxt`` and ``mint`` values are deterministic functions of the
    coordinates so tests can assert exact values.
    """
    rng = np.random.default_rng(42)
    init_times = pd.date_range("2024-01-01", periods=n_inits, freq="12h")
    pcts = np.array(PERCENTILES, dtype=float)
    f_hours = np.array(F_HOURS, dtype=int)

    # Shape: (n_inits, n_f_hours, n_pcts, NY, NX)
    shape = (n_inits, len(f_hours), len(pcts), NY, NX)
    base_temp = 275.0  # K

    # Generate raw noise and sort along the percentile axis to guarantee a
    # monotone CDF at every gridpoint (required for realistic NBM data).
    raw = base_temp + rng.normal(0, 3.0, shape)
    maxt_data = np.sort(raw, axis=2)
    # mint is cooler overall
    raw_mint = base_temp - 5.0 + rng.normal(0, 3.0, shape)
    mint_data = np.sort(raw_mint, axis=2)

    # 2-D lat/lon arrays
    lat = np.linspace(35.0, 45.0, NY)[:, np.newaxis] * np.ones((NY, NX))
    lon = np.linspace(-100.0, -85.0, NX)[np.newaxis, :] * np.ones((NY, NX))

    ds = xr.Dataset(
        {
            "maxt": xr.DataArray(
                maxt_data,
                dims=["init_time", "f_hour", "percentile", "y", "x"],
            ),
            "mint": xr.DataArray(
                mint_data,
                dims=["init_time", "f_hour", "percentile", "y", "x"],
            ),
        },
        coords={
            "init_time": init_times,
            "f_hour": f_hours,
            "percentile": pcts,
            "lat": xr.DataArray(lat, dims=["y", "x"]),
            "lon": xr.DataArray(lon, dims=["y", "x"]),
        },
    )
    return ds


@pytest.fixture
def nbm_ds() -> xr.Dataset:
    return _make_nbm_dataset()


@pytest.fixture
def reader(nbm_ds) -> NBMXarrayReader:
    return NBMXarrayReader(nbm_ds)


# ---------------------------------------------------------------------------
# NBMXarrayReader – coordinate inspection
# ---------------------------------------------------------------------------


class TestNBMXarrayReaderCoordinates:
    """Verify that the reader correctly exposes dataset coordinate axes."""

    def test_get_init_times_range(self, reader, nbm_ds):
        """init_time range must match the dataset."""
        it = reader.get_init_times()
        assert it[0] == nbm_ds.init_time.values[0]
        assert it[-1] == nbm_ds.init_time.values[-1]

    def test_get_forecast_hours(self, reader):
        """f_hour values must match F_HOURS (sorted)."""
        assert reader.get_forecast_hours() == sorted(F_HOURS)

    def test_get_percentiles(self, reader):
        """Percentile levels must be returned sorted and match fixture."""
        assert reader.get_percentiles() == sorted(PERCENTILES)

    def test_get_grid_shape(self, reader):
        """Grid shape must reflect fixture dimensions."""
        ny, nx = reader.get_grid_shape()
        assert ny == NY
        assert nx == NX

    def test_has_variable_present(self, reader):
        assert reader.has_variable("maxt") is True

    def test_has_variable_absent(self, reader):
        assert reader.has_variable("wind_speed") is False


# ---------------------------------------------------------------------------
# NBMXarrayReader – deterministic slice
# ---------------------------------------------------------------------------


class TestReadDeterministic:
    """Verify spatial subsetting and deterministic (median) extraction."""

    def test_returns_2d_dataarray(self, reader, nbm_ds):
        init_t = pd.Timestamp(nbm_ds.init_time.values[0])
        da = reader.read_deterministic("maxt", init_t, f_hour=24)
        assert da is not None
        assert da.dims == ("y", "x")
        assert da.shape == (NY, NX)

    def test_spatial_chunk_reduces_shape(self, reader, nbm_ds):
        init_t = pd.Timestamp(nbm_ds.init_time.values[0])
        da = reader.read_deterministic(
            "maxt", init_t, f_hour=24, chunk_bounds=(0, 10, 0, 15)
        )
        assert da is not None
        assert da.shape == (10, 15)

    def test_unknown_variable_returns_none(self, reader, nbm_ds):
        init_t = pd.Timestamp(nbm_ds.init_time.values[0])
        result = reader.read_deterministic("nonexistent", init_t, f_hour=24)
        assert result is None

    def test_custom_percentile_used(self, reader, nbm_ds):
        """90th pct slice must differ from 10th pct slice for maxt."""
        init_t = pd.Timestamp(nbm_ds.init_time.values[0])
        da_10 = reader.read_deterministic("maxt", init_t, f_hour=24, deterministic_percentile=10.0)
        da_90 = reader.read_deterministic("maxt", init_t, f_hour=24, deterministic_percentile=90.0)
        assert da_10 is not None and da_90 is not None
        # 90th pct should be systematically warmer
        assert float(da_90.mean()) > float(da_10.mean())


# ---------------------------------------------------------------------------
# NBMXarrayReader – CDF slice
# ---------------------------------------------------------------------------


class TestReadPercentileCDF:
    """Verify full CDF extraction for probabilistic verification."""

    def test_returns_3d_dataarray(self, reader, nbm_ds):
        init_t = pd.Timestamp(nbm_ds.init_time.values[0])
        da = reader.read_percentile_cdf("maxt", init_t, f_hour=24)
        assert da is not None
        assert da.dims == ("percentile", "y", "x")
        assert da.shape == (len(PERCENTILES), NY, NX)

    def test_percentile_coord_present(self, reader, nbm_ds):
        init_t = pd.Timestamp(nbm_ds.init_time.values[0])
        da = reader.read_percentile_cdf("maxt", init_t, f_hour=24)
        assert "percentile" in da.coords
        np.testing.assert_array_equal(
            sorted(da.coords["percentile"].values), sorted(PERCENTILES)
        )

    def test_spatial_chunk_reduces_shape(self, reader, nbm_ds):
        init_t = pd.Timestamp(nbm_ds.init_time.values[0])
        da = reader.read_percentile_cdf(
            "maxt", init_t, f_hour=24, chunk_bounds=(0, 5, 0, 8)
        )
        assert da is not None
        assert da.shape == (len(PERCENTILES), 5, 8)

    def test_cdf_monotone_in_percentile(self, reader, nbm_ds):
        """maxt CDF must be non-decreasing along the percentile axis."""
        init_t = pd.Timestamp(nbm_ds.init_time.values[0])
        da = reader.read_percentile_cdf("maxt", init_t, f_hour=24)
        assert da is not None
        vals = da.values  # (n_pct, NY, NX)
        diffs = np.diff(vals, axis=0)
        # Allow tiny numerical noise but no strict reversals
        assert np.all(diffs >= -0.01), "CDF must be non-decreasing in percentile"


# ---------------------------------------------------------------------------
# NBMXarrayReader – lat/lon coordinates
# ---------------------------------------------------------------------------


class TestGetLatLon:
    def test_full_grid(self, reader):
        lat, lon = reader.get_lat_lon()
        assert lat.shape == (NY, NX)
        assert lon.shape == (NY, NX)
        assert float(lat.min()) == pytest.approx(35.0, abs=0.1)
        assert float(lon.max()) == pytest.approx(-85.0, abs=0.1)

    def test_spatial_chunk(self, reader):
        lat, lon = reader.get_lat_lon(chunk_bounds=(0, 5, 0, 6))
        assert lat.shape == (5, 6)
        assert lon.shape == (5, 6)


# ---------------------------------------------------------------------------
# CRPS from quantile forecasts
# ---------------------------------------------------------------------------


class TestCRPSFromQuantiles:
    """Unit tests for quantile-based CRPS."""

    LEVELS = np.array([0.10, 0.25, 0.50, 0.75, 0.90])

    def test_perfect_deterministic_forecast_zero_crps(self):
        """A point-mass forecast exactly at the observation has CRPS = 0."""
        # All quantiles equal to observation → perfect deterministic
        obs = np.array([275.0])
        fcst = np.full((1, len(self.LEVELS)), 275.0)
        crps = compute_crps_from_quantiles(fcst, self.LEVELS, obs)
        assert crps[0] == pytest.approx(0.0, abs=1e-6)

    def test_wider_spread_increases_crps(self):
        """Wider CDF spread around the observation should increase CRPS."""
        obs = np.array([275.0])
        narrow_fcst = np.array([[274.5, 274.8, 275.0, 275.2, 275.5]])
        wide_fcst = np.array([[270.0, 272.0, 275.0, 278.0, 280.0]])
        crps_narrow = compute_crps_from_quantiles(narrow_fcst, self.LEVELS, obs)
        crps_wide = compute_crps_from_quantiles(wide_fcst, self.LEVELS, obs)
        assert float(crps_wide[0]) > float(crps_narrow[0])

    def test_nan_observation_produces_nan_crps(self):
        obs = np.array([np.nan])
        fcst = np.full((1, len(self.LEVELS)), 275.0)
        crps = compute_crps_from_quantiles(fcst, self.LEVELS, obs)
        assert np.isnan(crps[0])

    def test_percentile_rescaling(self):
        """Passing levels in [1, 100] should give the same result as [0, 1]."""
        obs = np.array([275.0, 278.0])
        fcst = np.tile([270, 272, 275, 278, 280], (2, 1)).astype(float)
        crps_01 = compute_crps_from_quantiles(fcst, self.LEVELS, obs)
        crps_pct = compute_crps_from_quantiles(
            fcst, np.array([10.0, 25.0, 50.0, 75.0, 90.0]), obs
        )
        np.testing.assert_allclose(crps_01, crps_pct, rtol=1e-6)

    def test_mean_crps_non_negative(self):
        obs = np.random.default_rng(0).normal(275, 2, 50)
        fcst = np.tile([270, 272, 275, 278, 280], (50, 1)).astype(float)
        mean_crps = compute_mean_crps_from_quantiles(fcst, self.LEVELS, obs)
        assert mean_crps >= 0.0

    def test_multiple_samples(self):
        """All-NaN observations → NaN mean CRPS."""
        obs = np.full(5, np.nan)
        fcst = np.tile([270, 272, 275, 278, 280], (5, 1)).astype(float)
        result = compute_mean_crps_from_quantiles(fcst, self.LEVELS, obs)
        assert np.isnan(result)


# ---------------------------------------------------------------------------
# CRPSS
# ---------------------------------------------------------------------------


class TestCRPSS:
    LEVELS = np.array([0.10, 0.25, 0.50, 0.75, 0.90])

    def test_perfect_forecast_crpss_one(self):
        """A forecast with zero CRPS (perfect CDF) should have CRPSS → 1."""
        obs = np.linspace(273.0, 278.0, 20)
        # Set all quantiles equal to observation → CRPS = 0
        fcst = np.column_stack([obs] * len(self.LEVELS))
        crpss = compute_crpss(fcst, self.LEVELS, obs)
        # CRPS_forecast = 0, reference > 0  → CRPSS = 1
        assert crpss == pytest.approx(1.0, abs=1e-6)

    def test_crpss_range(self):
        """CRPSS must be ≤ 1 and not far below 0 for a reasonable forecast."""
        rng = np.random.default_rng(7)
        obs = rng.normal(275, 2, 100)
        # Reasonable forecast: spread centred near obs
        fcst = np.column_stack(
            [obs + delta for delta in [-3, -1.5, 0, 1.5, 3]]
        ).astype(float)
        crpss = compute_crpss(fcst, self.LEVELS, obs)
        assert crpss <= 1.0

    def test_all_nan_observations_returns_nan(self):
        obs = np.full(10, np.nan)
        fcst = np.tile([270, 272, 275, 278, 280], (10, 1)).astype(float)
        result = compute_crpss(fcst, self.LEVELS, obs)
        assert np.isnan(result)
