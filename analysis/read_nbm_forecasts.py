"""
NBM v5 MaxT / MinT Forecast Reader
====================================
Lazily opens all maxt_qmd / mint_qmd GRIB2 files from the NBM para archive
into a single xarray Dataset backed by dask arrays.  Nothing is read from
disk until you explicitly compute a subset.

Grid
----
Lambert conformal, 2345 × 1597 (Ni × Nj), ~2.5 km, CONUS.
~390 GB per variable if fully loaded — dask is mandatory.

Message structure (per file, discovered from sample)
-----------------------------------------------------
  maxt_qmd : 5 probability msgs + 8 percentile msgs (5,10,25,50,75,90,95,100)
  mint_qmd : 1 probability msg  + 8 percentile msgs (5,10,25,50,75,90,95,100)
Only the percentile messages are loaded by default.

Coordinates
-----------
  init_time  : datetime64[ns]  — model initialization time (00Z or 12Z cycles)
  f_hour     : int             — forecast lead in hours
  valid_time : datetime64[ns]  — derived coordinate (init_time + f_hour)
  percentile : int             — 5, 10, 25, 50, 75, 90, 95, 100
  lat        : float32 (y, x)  — 2-D Lambert lat (non-dimensional coordinate)
  lon        : float32 (y, x)  — 2-D Lambert lon (non-dimensional coordinate)

Units
-----
Kelvin (native GRIB units, unconverted).

Usage
-----
    from analysis.read_nbm_forecasts import open_nbm_maxt_mint

    ds = open_nbm_maxt_mint()          # all dates, both cycles
    ds = open_nbm_maxt_mint(
            start_date=date(2025, 10, 1),
            end_date=date(2025, 10, 31),
            cycles=(0, 12),
         )

    # Lazy selection — no data read yet:
    median = ds["maxt"].sel(percentile=50)          # (init_time, f_hour, y, x)
    day1   = median.isel(init_time=0, f_hour=0)     # (y, x)

    # Compute a small subset:
    arr = day1.compute()                             # triggers actual GRIB reads
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Sequence

import dask
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from eccodes import (
    CodesInternalError,
    codes_get,
    codes_get_array,
    codes_get_values,
    codes_grib_new_from_file,
    codes_release,
)

# ---------------------------------------------------------------------------
# Paths and defaults
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
NBM_PARA_ROOT = Path(r"N:\data\nbm_para")

_PRODUCTS = {
    "maxt": "maxt_qmd",
    "mint": "mint_qmd",
}

# ---------------------------------------------------------------------------
# Grid metadata  (read once, reused for all files)
# ---------------------------------------------------------------------------

_GRID_CACHE: dict[str, dict] = {}


def _read_grid_meta(filepath: Path) -> dict:
    """Read grid shape and lat/lon arrays from the first message of a GRIB2 file."""
    key = str(filepath)
    if key in _GRID_CACHE:
        return _GRID_CACHE[key]

    with open(filepath, "rb") as fh:
        gid = codes_grib_new_from_file(fh)
        try:
            lats = codes_get_array(gid, "latitudes").astype(np.float32)
            lons = codes_get_array(gid, "longitudes").astype(np.float32)
            # Lambert grids use Ni/Nj; fall back to Nx/Ny or shape inference
            for key_ni, key_nj in [("Ni", "Nj"),
                                    ("numberOfPointsAlongXAxis", "numberOfPointsAlongYAxis")]:
                try:
                    ni = codes_get(gid, key_ni)
                    nj = codes_get(gid, key_nj)
                    break
                except CodesInternalError:
                    continue
            else:
                # last resort: assume roughly square
                n = len(lats)
                nj = int(n ** 0.5)
                ni = n // nj
        finally:
            codes_release(gid)

    meta = {
        "shape": (nj, ni),
        "lats": lats.reshape(nj, ni),
        "lons": lons.reshape(nj, ni),
    }
    _GRID_CACHE[str(filepath)] = meta
    return meta


# ---------------------------------------------------------------------------
# Message catalog  (index one sample file per product)
# ---------------------------------------------------------------------------

def _discover_message_map(sample_file: Path) -> dict[int, int]:
    """Return {percentile_value: 1-based_msg_id} by indexing one sample file."""
    sys.path.insert(0, str(_REPO_ROOT))
    import nbm_grib_tools as nt  # local module

    df = nt.index_nbm5_grib(str(sample_file), convert_imperial=False)
    pct_rows = df[df["param_type"] == "Percentile"].dropna(subset=["percentile"])
    return {int(row.percentile): int(row.msg_id) for row in pct_rows.itertuples()}


# ---------------------------------------------------------------------------
# File catalog builder
# ---------------------------------------------------------------------------

def _scan_files(
    root: Path,
    product_prefix: str,
    start_date: Optional[date],
    end_date: Optional[date],
    cycles: Sequence[int],
) -> pd.DataFrame:
    """Walk the date/cycle tree and catalogue every matching GRIB2 file."""
    records = []
    cycle_strs = {str(c).zfill(2) for c in cycles}

    for date_dir in sorted(root.iterdir()):
        if not date_dir.name.isdigit() or len(date_dir.name) != 8 or not date_dir.is_dir():
            continue
        try:
            d = datetime.strptime(date_dir.name, "%Y%m%d").date()
        except ValueError:
            continue
        if start_date and d < start_date:
            continue
        if end_date and d > end_date:
            continue

        for cycle_dir in sorted(date_dir.iterdir()):
            if not cycle_dir.is_dir() or cycle_dir.name not in cycle_strs:
                continue
            cycle_h = int(cycle_dir.name)

            for grib_file in sorted(cycle_dir.glob(f"{product_prefix}_f*.grib2")):
                try:
                    f_hour = int(grib_file.stem.rsplit("_f", 1)[1])
                except (ValueError, IndexError):
                    continue

                init_dt = datetime(d.year, d.month, d.day, cycle_h)
                valid_dt = init_dt + timedelta(hours=f_hour)
                records.append(
                    {
                        "filepath": grib_file,
                        "date": d,
                        "cycle": cycle_h,
                        "init_time": pd.Timestamp(init_dt),
                        "f_hour": f_hour,
                        "valid_time": pd.Timestamp(valid_dt),
                    }
                )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Dask-delayed GRIB reader  (one file → all percentile planes)
# ---------------------------------------------------------------------------

@dask.delayed
def _read_percentile_planes(
    filepath: str,
    pct_to_msg: dict,          # {percentile_int: msg_id_1based}
    sorted_pcts: tuple,        # ordered tuple of percentile ints
    grid_shape: tuple,         # (nj, ni)
) -> np.ndarray:
    """Open one GRIB2 file and read all requested percentile messages in a single
    sequential pass.  Returns float32 array of shape (n_pct, nj, ni).
    Missing / fill values are set to NaN.
    """
    n_pct = len(sorted_pcts)
    nj, ni = grid_shape
    result = np.full((n_pct, nj * ni), np.nan, dtype=np.float32)

    # build reverse map: msg_id → index in sorted_pcts
    msg_to_idx: dict[int, int] = {
        pct_to_msg[pct]: i for i, pct in enumerate(sorted_pcts)
    }
    max_msg = max(pct_to_msg[p] for p in sorted_pcts)

    with open(filepath, "rb") as fh:
        msg_count = 0
        while msg_count < max_msg:
            gid = codes_grib_new_from_file(fh)
            if gid is None:
                break
            msg_count += 1
            if msg_count in msg_to_idx:
                idx = msg_to_idx[msg_count]
                try:
                    vals = codes_get_values(gid).astype(np.float32)
                    vals[vals > 1.0e10] = np.nan  # eccodes fill value
                except CodesInternalError:
                    pass
                else:
                    result[idx] = vals
            codes_release(gid)

    return result.reshape(n_pct, nj, ni)


def _missing_plane(n_pct: int, grid_shape: tuple) -> da.Array:
    """All-NaN placeholder for (init_time, f_hour) combinations with no file."""
    nj, ni = grid_shape
    return da.full((n_pct, nj, ni), np.nan, dtype=np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def open_nbm_maxt_mint(
    root: str | Path = NBM_PARA_ROOT,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    cycles: Sequence[int] = (0, 12),
    variables: Sequence[str] = ("maxt", "mint"),
) -> xr.Dataset:
    """
    Open all NBM v5 MaxT / MinT GRIB2 files as a lazy xarray Dataset.

    Parameters
    ----------
    root        : Path to the NBM para archive root (default: N:\\data\\nbm_para\\)
    start_date  : First date to include (inclusive).  None → all available.
    end_date    : Last  date to include (inclusive).  None → all available.
    cycles      : Init cycles to include.  Default (0, 12).
    variables   : Which products to open.  Subset of ("maxt", "mint").

    Returns
    -------
    xr.Dataset
        Variables : ``maxt``, ``mint``  (whichever were requested)
        Dims      : (init_time, f_hour, percentile, y, x)
        Coords    : init_time, f_hour, valid_time, percentile, lat, lon

    All arrays are **lazy dask arrays**.  No GRIB file is opened until you
    call ``.compute()``, ``.load()``, or access ``.values``.

    Examples
    --------
    >>> ds = open_nbm_maxt_mint(start_date=date(2025, 10, 1),
    ...                          end_date=date(2025, 10, 7))
    >>> # Extract the 50th-percentile MaxT at every point for the first forecast:
    >>> da = ds["maxt"].sel(percentile=50).isel(init_time=0, f_hour=0).compute()
    """
    root = Path(root)
    data_vars: dict[str, xr.DataArray] = {}
    grid_meta: dict | None = None

    for var_name in variables:
        product_prefix = _PRODUCTS[var_name]

        # ---- 1. Scan files -----------------------------------------------
        print(f"[{var_name}] Scanning {root} ...")
        catalog = _scan_files(root, product_prefix, start_date, end_date, cycles)
        if catalog.empty:
            print(f"  [{var_name}] No files found — skipping.")
            continue
        print(f"  [{var_name}] {len(catalog)} files  "
              f"| init range: {catalog['init_time'].min()} → {catalog['init_time'].max()}")

        # ---- 2. Message structure (sample one file) -----------------------
        sample = catalog.iloc[0]["filepath"]
        print(f"  [{var_name}] Indexing sample file to discover message layout ...")
        pct_to_msg = _discover_message_map(sample)
        sorted_pcts = tuple(sorted(pct_to_msg))
        n_pct = len(sorted_pcts)
        print(f"  [{var_name}] Percentiles: {list(sorted_pcts)}")

        # ---- 3. Grid coordinates (read once, shared across variables) -----
        if grid_meta is None:
            grid_meta = _read_grid_meta(sample)
        grid_shape = grid_meta["shape"]   # (nj, ni)
        nj, ni = grid_shape

        # ---- 4. Build coordinate axes ------------------------------------
        init_times = sorted(catalog["init_time"].unique())
        f_hours    = sorted(catalog["f_hour"].unique())
        n_init     = len(init_times)
        n_fhour    = len(f_hours)

        # Fast lookup: (init_time, f_hour) → filepath string
        cat_idx = (
            catalog
            .drop_duplicates(subset=["init_time", "f_hour"])
            .set_index(["init_time", "f_hour"])["filepath"]
        )

        # ---- 5. Build dask array  ----------------------------------------
        # Shape: (n_init, n_fhour, n_pct, nj, ni)
        # Each chunk = one file = (n_pct, nj, ni)

        print(f"  [{var_name}] Building dask graph "
              f"({n_init} init × {n_fhour} f_hours × {n_pct} pct "
              f"× {nj}×{ni} grid) ...")

        fhour_slabs: list[da.Array] = []  # each: (n_init, 1, n_pct, nj, ni)

        for fh in f_hours:
            init_slabs: list[da.Array] = []  # each: (1, 1, n_pct, nj, ni)
            for it in init_times:
                key = (pd.Timestamp(it), fh)
                if key in cat_idx.index:
                    fp = str(cat_idx.loc[key])
                    delayed = _read_percentile_planes(fp, pct_to_msg, sorted_pcts, grid_shape)
                    arr = da.from_delayed(
                        delayed,
                        shape=(n_pct, nj, ni),
                        dtype=np.float32,
                    )
                else:
                    arr = _missing_plane(n_pct, grid_shape)

                # add init_time and f_hour dims: → (1, 1, n_pct, nj, ni)
                init_slabs.append(arr[np.newaxis, np.newaxis])

            fhour_slabs.append(da.concatenate(init_slabs, axis=0))  # (n_init, 1, n_pct, nj, ni)

        full = da.concatenate(fhour_slabs, axis=1)  # (n_init, n_fhour, n_pct, nj, ni)

        # ---- 6. Wrap as DataArray -----------------------------------------
        init_arr  = np.array(init_times, dtype="datetime64[ns]")
        fhour_arr = np.array(f_hours, dtype=np.int32)
        # valid_time as a 2-D coordinate: (init_time, f_hour)
        valid_times = (
            init_arr[:, np.newaxis].astype("datetime64[h]")
            + fhour_arr[np.newaxis, :].astype("timedelta64[h]")
        ).astype("datetime64[ns]")

        data_vars[var_name] = xr.DataArray(
            full,
            dims=["init_time", "f_hour", "percentile", "y", "x"],
            coords={
                "init_time":  ("init_time",  init_arr),
                "f_hour":     ("f_hour",     fhour_arr),
                "valid_time": (["init_time", "f_hour"], valid_times),
                "percentile": ("percentile", np.array(sorted_pcts, dtype=np.int16)),
                "lat":        (["y", "x"],   grid_meta["lats"]),
                "lon":        (["y", "x"],   grid_meta["lons"]),
            },
            name=var_name,
            attrs={
                "units": "K",
                "long_name": (
                    "Maximum 2-m Temperature"
                    if var_name == "maxt"
                    else "Minimum 2-m Temperature"
                ),
                "source": f"NBM v5 para — {product_prefix}",
            },
        )
        print(f"  [{var_name}] Done. Shape: {dict(zip(full.shape, full.shape))} "
              f"(lazy, ~{full.nbytes / 2**30:.1f} GiB uncompressed)")

    ds = xr.Dataset(data_vars)
    ds.attrs["source"]      = str(root)
    ds.attrs["created"]     = datetime.utcnow().isoformat()
    ds.attrs["units_note"]  = "All temperatures in Kelvin (native GRIB units)."
    ds.attrs["dask_note"]   = (
        "All arrays are lazy. Call .compute() / .load() only on subsets."
    )
    return ds


# ---------------------------------------------------------------------------
# Convenience: K → °F / °C conversion helpers
# ---------------------------------------------------------------------------

def k_to_f(da_k: xr.DataArray) -> xr.DataArray:
    """Convert a DataArray from Kelvin to Fahrenheit."""
    out = (da_k - 273.15) * 1.8 + 32.0
    out.attrs = {**da_k.attrs, "units": "°F"}
    return out


def k_to_c(da_k: xr.DataArray) -> xr.DataArray:
    """Convert a DataArray from Kelvin to Celsius."""
    out = da_k - 273.15
    out.attrs = {**da_k.attrs, "units": "°C"}
    return out


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from datetime import date as _date

    print("Building lazy dataset for Oct 1–3, 2025 ...")
    ds = open_nbm_maxt_mint(
        start_date=_date(2025, 10, 1),
        end_date=_date(2025, 10, 3),
    )
    print(ds)
    print()
    print("Extracting MaxT 50th-pct, init_time=0, f_hour=30 (single GRIB read) ...")
    subset = ds["maxt"].sel(percentile=50, f_hour=30).isel(init_time=0)
    arr = subset.compute()
    valid = arr.values[~np.isnan(arr.values)]
    print(f"  Non-NaN points: {len(valid):,}")
    print(f"  Min/Max (K): {valid.min():.1f} / {valid.max():.1f}")
    print(f"  Min/Max (°F): {(valid.min()-273.15)*1.8+32:.1f} / {(valid.max()-273.15)*1.8+32:.1f}")
    print("Smoke test passed.")
