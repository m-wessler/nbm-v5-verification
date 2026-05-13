"""
NBM xarray-based reader for multi-dimensional lazy-loaded datasets.

Handles datasets with the coordinate structure produced by the NBM lazy-loading
pipeline::

    ds.init_time   – 1-D array of pandas Timestamps (model run times)
    ds.f_hour      – 1-D array of integer forecast-hour offsets
    ds.percentile  – 1-D array of percentile levels (e.g. [10, 25, 50, 75, 90])
    ds.y / ds.x    – spatial dimensions (row / column indices)

Variables such as ``maxt`` and ``mint`` live on
``(init_time, f_hour, percentile, y, x)`` and are *not* loaded into memory
until values are explicitly requested, keeping the working-set small even
when the full dataset spans many GiB.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from nbm_verification.utils.exceptions import DataAccessError

logger = logging.getLogger(__name__)


class NBMXarrayReader:
    """
    Reader for NBM forecasts stored as a lazy-loaded multi-dimensional xarray
    Dataset.

    The Dataset is expected to follow the coordinate convention::

        Dimensions  : (init_time, f_hour, percentile, y, x)
        Coordinates :
            init_time   (init_time)  datetime64[ns]
            f_hour      (f_hour)     int32/int64        forecast-hour offset
            percentile  (percentile) float32/float64    e.g. 10, 25, 50, 75, 90
            lat         (y, x)       float32/float64    2-D latitude grid
            lon         (y, x)       float32/float64    2-D longitude grid
        Data vars   :
            maxt        (init_time, f_hour, percentile, y, x)
            mint        (init_time, f_hour, percentile, y, x)
            ...

    The dataset may be backed by Zarr, NetCDF, or cfgrib-opened GRIB2 files;
    no data is loaded until values are explicitly requested.

    Parameters
    ----------
    dataset_or_path : xr.Dataset or path-like
        Either a pre-opened xarray Dataset or a path/URL to open.
        Paths ending in ``.zarr`` are opened with the Zarr engine; all
        others fall back to the default xarray engine.
    open_kwargs : dict, optional
        Extra keyword arguments forwarded to ``xr.open_dataset`` or
        ``xr.open_zarr`` when a path is supplied.

    Examples
    --------
    >>> reader = NBMXarrayReader("/data/nbm/blend_v5.zarr")
    >>> print("init_time range:", reader.ds.init_time.values[0],
    ...       "→", reader.ds.init_time.values[-1])
    >>> print("f_hour values  :", reader.ds.f_hour.values.tolist())
    >>> print("percentiles    :", reader.ds.percentile.values.tolist())
    >>> print("grid shape (y,x):", reader.ds.dims["y"], "×", reader.ds.dims["x"])
    """

    # Coordinate dimension names expected on the Dataset
    DIM_INIT_TIME = "init_time"
    DIM_F_HOUR = "f_hour"
    DIM_PERCENTILE = "percentile"
    DIM_Y = "y"
    DIM_X = "x"

    # Deterministic percentile used when a single "best estimate" is needed
    DETERMINISTIC_PERCENTILE = 50.0

    def __init__(
        self,
        dataset_or_path: Union[xr.Dataset, str, Path],
        open_kwargs: Optional[Dict] = None,
    ):
        self._open_kwargs = open_kwargs or {}
        if isinstance(dataset_or_path, xr.Dataset):
            self.ds: xr.Dataset = dataset_or_path
        else:
            self.ds = self._open_dataset(Path(dataset_or_path))
        self._validate_dimensions()

    # ------------------------------------------------------------------
    # Opening helpers
    # ------------------------------------------------------------------

    def _open_dataset(self, path: Path) -> xr.Dataset:
        """Open dataset from path, choosing engine by extension."""
        path_str = str(path)
        try:
            if path_str.endswith(".zarr") or path_str.endswith(".zarr/"):
                logger.debug("Opening NBM dataset as Zarr: %s", path_str)
                return xr.open_zarr(path_str, **self._open_kwargs)
            else:
                logger.debug("Opening NBM dataset with default engine: %s", path_str)
                return xr.open_dataset(path_str, **self._open_kwargs)
        except Exception as exc:
            raise DataAccessError(f"Cannot open NBM dataset at {path_str}: {exc}") from exc

    def _validate_dimensions(self) -> None:
        """Warn if expected coordinate dimensions are missing."""
        required = {
            self.DIM_INIT_TIME,
            self.DIM_F_HOUR,
            self.DIM_PERCENTILE,
            self.DIM_Y,
            self.DIM_X,
        }
        missing = required - set(self.ds.dims)
        if missing:
            logger.warning(
                "NBM dataset is missing expected dimension(s): %s. "
                "Slicing operations may fail.",
                sorted(missing),
            )

    # ------------------------------------------------------------------
    # Coordinate inspection helpers
    # ------------------------------------------------------------------

    def get_init_times(self) -> np.ndarray:
        """Return array of available initialization timestamps."""
        return self.ds[self.DIM_INIT_TIME].values

    def get_forecast_hours(self) -> List[int]:
        """Return sorted list of available forecast-hour offsets."""
        return sorted(int(h) for h in self.ds[self.DIM_F_HOUR].values)

    def get_percentiles(self) -> List[float]:
        """Return sorted list of available percentile levels."""
        return sorted(float(p) for p in self.ds[self.DIM_PERCENTILE].values)

    def get_grid_shape(self) -> Tuple[int, int]:
        """Return ``(n_y, n_x)`` grid dimensions."""
        sizes = self.ds.sizes
        return (sizes[self.DIM_Y], sizes[self.DIM_X])

    def summarize(self) -> None:
        """Log a compact summary of the dataset's forecast coordinate axes."""
        init_vals = self.get_init_times()
        logger.info(
            "NBM dataset summary:\n"
            "  init_time range   : %s → %s\n"
            "  f_hour values     : %s\n"
            "  percentiles       : %s\n"
            "  grid shape (y, x) : %d × %d",
            init_vals[0] if len(init_vals) else "N/A",
            init_vals[-1] if len(init_vals) else "N/A",
            self.get_forecast_hours(),
            self.get_percentiles(),
            *self.get_grid_shape(),
        )
        total_gb = sum(
            var.nbytes / 2**30
            for var in self.ds.data_vars.values()
            if hasattr(var, "nbytes")
        )
        logger.info(
            "  Total uncompressed: %.0f GiB  — lazy, nothing read yet", total_gb
        )

    # ------------------------------------------------------------------
    # Spatial subset helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _spatial_slice(
        chunk_bounds: Optional[Tuple[int, int, int, int]],
    ) -> Dict[str, slice]:
        """Convert ``(y_start, y_end, x_start, x_end)`` to isel-compatible slices."""
        if chunk_bounds is None:
            return {}
        y_start, y_end, x_start, x_end = chunk_bounds
        return {
            "y": slice(y_start, y_end),
            "x": slice(x_start, x_end),
        }

    # ------------------------------------------------------------------
    # Core extraction methods
    # ------------------------------------------------------------------

    def select_time(
        self,
        init_time: Union[datetime, pd.Timestamp, np.datetime64],
        f_hour: int,
    ) -> xr.Dataset:
        """
        Return a Dataset slice for a single ``(init_time, f_hour)`` combination.

        The returned object still has ``(percentile, y, x)`` dimensions and is
        backed lazily — no data is loaded until ``.values`` is accessed.

        Parameters
        ----------
        init_time : datetime-like
            Initialization timestamp to select.
        f_hour : int
            Forecast-hour offset to select.

        Returns
        -------
        xr.Dataset
            Dataset slice with dimensions ``(percentile, y, x)``.

        Raises
        ------
        DataAccessError
            If the requested ``init_time`` or ``f_hour`` is not present.
        """
        ts = pd.Timestamp(init_time)
        try:
            ds_t = self.ds.sel(
                {self.DIM_INIT_TIME: ts, self.DIM_F_HOUR: f_hour},
                method="nearest",
            )
        except KeyError as exc:
            raise DataAccessError(
                f"init_time={ts} f_hour={f_hour} not found in NBM dataset: {exc}"
            ) from exc
        return ds_t

    def read_deterministic(
        self,
        variable: str,
        init_time: Union[datetime, pd.Timestamp, np.datetime64],
        f_hour: int,
        chunk_bounds: Optional[Tuple[int, int, int, int]] = None,
        deterministic_percentile: Optional[float] = None,
    ) -> Optional[xr.DataArray]:
        """
        Load a single deterministic (median or specified percentile) 2-D spatial
        slice for ``(variable, init_time, f_hour)``.

        Parameters
        ----------
        variable : str
            Variable name as it appears in the dataset (e.g. ``"maxt"``).
        init_time : datetime-like
            Initialization timestamp.
        f_hour : int
            Forecast-hour offset.
        chunk_bounds : (y_start, y_end, x_start, x_end), optional
            If provided, only the specified spatial tile is loaded, keeping
            memory usage proportional to the tile size rather than the full grid.
        deterministic_percentile : float, optional
            Percentile level to use as the "best estimate". Defaults to
            ``DETERMINISTIC_PERCENTILE`` (50.0 = median).

        Returns
        -------
        xr.DataArray or None
            2-D ``(y, x)`` DataArray with the requested data, or ``None`` if
            the variable or time combination is not available.

        Examples
        --------
        >>> reader = NBMXarrayReader(ds)
        >>> from datetime import datetime
        >>> da = reader.read_deterministic("maxt", datetime(2024, 1, 1, 0), f_hour=24)
        >>> da.shape  # (n_y, n_x)
        (1597, 2345)
        """
        if variable not in self.ds.data_vars:
            logger.warning("Variable '%s' not found in NBM dataset.", variable)
            return None

        pct = deterministic_percentile if deterministic_percentile is not None else self.DETERMINISTIC_PERCENTILE

        try:
            ds_t = self.select_time(init_time, f_hour)
            da = ds_t[variable].sel(
                {self.DIM_PERCENTILE: pct}, method="nearest"
            )
        except DataAccessError:
            return None
        except Exception as exc:
            logger.error("Error selecting deterministic slice: %s", exc)
            return None

        # Apply spatial subsetting (lazy isel — no data loaded yet)
        spatial_slices = self._spatial_slice(chunk_bounds)
        if spatial_slices:
            da = da.isel(**spatial_slices)

        # Trigger load for just this tile
        try:
            return da.load()
        except Exception as exc:
            logger.error(
                "Error loading NBM data for %s init=%s f_hour=%d: %s",
                variable,
                init_time,
                f_hour,
                exc,
            )
            return None

    def read_percentile_cdf(
        self,
        variable: str,
        init_time: Union[datetime, pd.Timestamp, np.datetime64],
        f_hour: int,
        chunk_bounds: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[xr.DataArray]:
        """
        Load all percentile levels for a spatial tile, returning a 3-D array
        ``(percentile, y, x)`` suitable for CRPS/probabilistic verification.

        Parameters
        ----------
        variable : str
            Variable name (e.g. ``"maxt"``).
        init_time : datetime-like
            Initialization timestamp.
        f_hour : int
            Forecast-hour offset.
        chunk_bounds : (y_start, y_end, x_start, x_end), optional
            Spatial tile bounds.

        Returns
        -------
        xr.DataArray or None
            3-D DataArray ``(percentile, y, x)`` or ``None`` on error.

        Examples
        --------
        >>> cdf = reader.read_percentile_cdf("maxt", datetime(2024, 1, 1), 24)
        >>> cdf.dims
        ('percentile', 'y', 'x')
        >>> cdf.coords['percentile'].values
        array([10., 25., 50., 75., 90.])
        """
        if variable not in self.ds.data_vars:
            logger.warning("Variable '%s' not found in NBM dataset.", variable)
            return None

        try:
            ds_t = self.select_time(init_time, f_hour)
            da = ds_t[variable]  # (percentile, y, x)
        except DataAccessError:
            return None
        except Exception as exc:
            logger.error("Error selecting CDF slice: %s", exc)
            return None

        spatial_slices = self._spatial_slice(chunk_bounds)
        if spatial_slices:
            da = da.isel(**spatial_slices)

        try:
            return da.load()
        except Exception as exc:
            logger.error(
                "Error loading NBM CDF for %s init=%s f_hour=%d: %s",
                variable,
                init_time,
                f_hour,
                exc,
            )
            return None

    # ------------------------------------------------------------------
    # Convenience / introspection
    # ------------------------------------------------------------------

    def has_variable(self, variable: str) -> bool:
        """Return True if *variable* is present in the dataset."""
        return variable in self.ds.data_vars

    def get_lat_lon(
        self,
        chunk_bounds: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return ``(lat, lon)`` 2-D arrays for the full grid or a spatial tile.

        These coordinate arrays are cheap to load (they don't depend on the
        time/forecast/percentile dimensions) and are used to build the
        spatial index or station-to-gridpoint mapping.

        Parameters
        ----------
        chunk_bounds : (y_start, y_end, x_start, x_end), optional
            Tile bounds.

        Returns
        -------
        lat, lon : np.ndarray
            2-D arrays of shape ``(n_y, n_x)``.
        """
        lat_key = "lat" if "lat" in self.ds.coords else (
            "latitude" if "latitude" in self.ds.coords else None
        )
        lon_key = "lon" if "lon" in self.ds.coords else (
            "longitude" if "longitude" in self.ds.coords else None
        )
        if lat_key is None or lon_key is None:
            raise DataAccessError(
                "Dataset does not have 'lat'/'lon' (or 'latitude'/'longitude') coordinates."
            )
        lat_da = self.ds.coords[lat_key]
        lon_da = self.ds.coords[lon_key]

        spatial_slices = self._spatial_slice(chunk_bounds)
        if spatial_slices:
            lat_da = lat_da.isel(**spatial_slices)
            lon_da = lon_da.isel(**spatial_slices)

        return lat_da.values, lon_da.values
