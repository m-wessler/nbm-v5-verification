"""
Creates dev/nbm_verification.ipynb — clean, reorganised verification notebook.
Reads cell source code directly from the old notebook to avoid all string-embedding
and escaping issues.

Run once:
    cd C:\\Users\\Michael.wessler\\Code\\nbm-v5-verification
    python dev/make_notebook.py
"""
import nbformat as nbf
from pathlib import Path
import re

# ── Load old notebook ────────────────────────────────────────────────────────
OLD_NB = Path(r'C:\Users\Michael.wessler\Code\nbm-v5-verification\dev\nbm_forecasts_explore.ipynb')
with open(OLD_NB, encoding='utf-8') as f:
    old = nbf.read(f, as_version=4)

def src(i):
    """Return the source of old cell i as a string."""
    return old.cells[i]['source']

def strip_lines(text, *patterns):
    """Remove lines matching any of the given regex patterns; trim leading blanks."""
    lines = text.split('\n')
    kept = [l for l in lines if not any(re.search(p, l) for p in patterns)]
    while kept and kept[0].strip() == '':
        kept.pop(0)
    return '\n'.join(kept)

def keep_from(text, pattern):
    """Keep only lines from the first line matching `pattern` onwards."""
    lines = text.split('\n')
    for i, l in enumerate(lines):
        if re.search(pattern, l):
            while i > 0 and lines[i-1].strip() == '':
                i -= 1
            return '\n'.join(lines[i:])
    return text

def keep_until(text, pattern):
    """Keep only lines up to (but not including) the first line matching `pattern`."""
    lines = text.split('\n')
    result = []
    for l in lines:
        if re.search(pattern, l):
            break
        result.append(l)
    while result and result[-1].strip() == '':
        result.pop()
    return '\n'.join(result)

# ─────────────────────────────────────────────────────────────────────────────
# New cell content helpers
# ─────────────────────────────────────────────────────────────────────────────

TITLE_SRC = (
    "# NBM v5 MaxT/MinT Verification\n"
    "Point verification of NBM QMD probabilistic temperature forecasts for a NWS CWA.\n"
    "Covers deterministic skill, percentile reliability, and threshold-exceedance reliability."
)

IMPORTS_SRC = "\n".join([
    "import sys",
    "import math",
    "import numpy as np",
    "import xarray as xr",
    "import pandas as pd",
    "import matplotlib.pyplot as plt",
    "import matplotlib.dates as mdates",
    "import matplotlib.ticker as mticker",
    "import dask",
    "import dask.array as da",
    "import requests",
    "from pathlib import Path",
    "from datetime import date",
    "from scipy.spatial import cKDTree",
    "from eccodes import (",
    "    CodesInternalError, codes_grib_new_from_file, codes_get_values, codes_release,",
    ")",
    "",
    r"sys.path.insert(0, r'C:\Users\Michael.wessler\Code\nbm-v5-verification')",
    "from analysis.read_nbm_forecasts import open_nbm_maxt_mint, NBM_PARA_ROOT",
    "import nbm_grib_tools as nt",
])

CONFIG_SRC = "\n".join([
    "# \u2500\u2500 Study period \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
    "START_DATE = date(2026, 3, 1)",
    "END_DATE   = date(2026, 5, 1)",
    "",
    "# \u2500\u2500 CWA & API credentials \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
    'CWA            = "PHI"',
    'SYNOPTIC_TOKEN = "a2386b75ecbc4c2784db1270695dde73"',
    "",
    "# \u2500\u2500 Paths \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
    r'OBS_DIR = Path(r"N:\data\nbm_para\observations")',
    "",
    "# \u2500\u2500 Forecast mean method \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
    "# 'qmd'    \u2014 trapezoidal integration over the quantile density function",
    "# 'simple' \u2014 unweighted mean of the 8 available percentile values",
    "MEAN_METHOD = 'simple'",
    "",
    "# \u2500\u2500 Valid-time offsets (hours past obs file base date \u2192 NBM valid_time) \u2500\u2500\u2500",
    "# MaxT: obs file day D \u2192 valid_time D+1 06Z  (+30 h)",
    "# MinT: obs file day D \u2192 valid_time D 18Z    (+18 h)",
    "VAR_HOURS = {'maxt': 30, 'mint': 18}",
    "",
    "# \u2500\u2500 Station matching \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
    "TOL_DEG    = 0.05               # max station\u2194obs tolerance (~5 km)",
    'KEEP_MNETS = {"1", "2", "153"} # ASOS/AWOS (1), RAWS (2), GHCN-Daily (153)',
    "",
    "# \u2500\u2500 Per-station distribution histogram: lead day to display \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
    "LEAD_DAY = 1",
    "",
    "# \u2500\u2500 Probabilistic verification \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
    "SKIP_PERCS  = {100}                    # p=100 excluded from reliability diagram",
    "PROB_BINS   = np.linspace(0, 1, 11)   # reliability diagram bin edges",
    "BIN_CENTERS = 0.5 * (PROB_BINS[:-1] + PROB_BINS[1:])",
    "",
    "# \u2500\u2500 Time-matching diagnostic (Section 5) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
    "DIAG_STATION   = 'KPHL'",
    "DIAG_START     = '2026-03-15'  # event-day window start (inclusive)",
    "DIAG_END       = '2026-03-28'  # event-day window end   (inclusive)",
    "DIAG_LEAD      = 1             # lead_day to show",
    "DIAG_INIT_HOUR = 12            # restrict to one init cycle (0, 12, or None = all)",
])

# ── Section 1: Dataset ───────────────────────────────────────────────────────
# Combine open_nbm_maxt_mint (cell 3) with QMD means for both vars (cells 4+5)
# Use START_DATE/END_DATE from config instead of hardcoded dates

_ds_open = (
    "ds = open_nbm_maxt_mint(start_date=START_DATE, end_date=END_DATE)\n"
    "\n"
    "_probs = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.00])\n"
    "\n"
    "for var in ['maxt', 'mint']:\n"
    "    # QMD mean: trapezoidal rule  E[X] = trapz over [0.05,1.00] + rect for [0,0.05]\n"
    "    ds[f'{var}_qmd_mean'] = (\n"
    "        xr.apply_ufunc(\n"
    "            np.trapz, ds[var],\n"
    "            input_core_dims=[['percentile']],\n"
    "            kwargs={'x': _probs},\n"
    "            dask='parallelized', output_dtypes=[np.float32],\n"
    "        )\n"
    "        + 0.05 * ds[var].sel(percentile=5)\n"
    "    )\n"
    "    ds[f'{var}_simple_mean'] = ds[var].mean(dim='percentile')\n"
    "    ds[f'{var}_mean'] = ds[f'{var}_{MEAN_METHOD}_mean']\n"
    "    ds[f'{var}_mean'].attrs = {\n"
    "        **ds[var].attrs,\n"
    "        'long_name': f'{var.upper()} 2-m temperature ({MEAN_METHOD} mean)',\n"
    "    }\n"
    "\n"
    "print(f\"Dataset: {dict(ds.dims)}  |  mean method: '{MEAN_METHOD}'\")\n"
    "ds"
)

# ── Section 2: Stations ───────────────────────────────────────────────────────
# Cell 7 + Cell 8 merged, stripping local imports and config variables

_stations_7 = strip_lines(
    src(7),
    r'^import requests',
    r'^import pandas',
    r'^CWA\s*=',
    r'^data\s*=',  # the intermediate 'data = resp.json()' isn't in cell 7 in the clean version
)
# Fix: cell 7 uses 'data = resp.json()' then builds df; strip that line and inline
_stations_7 = _stations_7.replace(
    'data = resp.json()\n\nstations = pd.DataFrame([\n    {\n',
    'stations = pd.DataFrame([\n    {\n'
)
# If data still referenced via data["STATION"], replace with resp.json()["STATION"]
_stations_7 = _stations_7.replace('data["STATION"]', 'resp.json()["STATION"]')

_stations_8 = strip_lines(
    src(8),
    r'^from pathlib',
    r'^from scipy',
    r'^OBS_DIR\s*=',
    r'^TOL_DEG\s*=',
    r'^KEEP_MNETS\s*=',
)
# Remove the verbose breakdown print at the end
_stations_8 = strip_lines(
    _stations_8,
    r'print\(f"Obs stations in',
    r'print\(f"Synoptic',
    r'print\(f"After network',
    r'print\(stations_filtered\[',
)
# Add a single summary print + display
_stations_8 = (
    _stations_8.rstrip()
    + "\n\n"
    + "print(f\"{CWA}: {len(stations)} active | {len(stations_obs)} matched obs | \"\n"
    + "      f\"{len(stations_filtered)} after network filter\")\n"
    + "stations_filtered"
)

STATIONS_SRC = _stations_7.rstrip() + "\n\n" + _stations_8

# ── Section 2: Grid matching (first part of cell 9) ──────────────────────────
_grid_src = strip_lines(src(9), r'^import dask')
# Keep only up to the print at the end of grid matching
# Cell 9 has grid match then extraction — split at "# ── 2. Lazy" comment
_grid_src = keep_until(_grid_src, r'# \u2500+ 2\. Lazy|maxt_pts\s*=\s*ds')
# Add a print
_grid_src = (
    _grid_src.rstrip()
    + "\n\n"
    + "print(f\"Grid: {nj} \u00d7 {ni} pixels  |  {len(stations_filtered)} stations mapped\")"
)

# ── Section 4: Extraction + pairing ──────────────────────────────────────────
# Take extraction portion of cell 9 (from "# ── 2. Lazy" onwards)
_extpart = keep_from(src(9), r'# \u2500+ 2\. Lazy|# shape:.*still lazy')
# Strip the 'import dask' that appeared at the top of cell 9 if still present
_extpart = strip_lines(_extpart, r'^import dask')

# Replace the verbose final print with a cleaner summary
_extpart = strip_lines(_extpart, r"print\(f\"\{len\(fcst_df\)")
_extpart = strip_lines(_extpart, r"fcst_df\.head\(\)")

# Combine with _make_paired cell (cell 14), but replace the verbose for-loop
_paired_src = src(14)
# Replace the verbose print loop with a compact summary
old_loop = (
    "for name, df in [('maxt_paired', maxt_paired), ('mint_paired', mint_paired)]:\n"
    "    print(f\"{name}: {len(df):,} rows | {df['station'].nunique()} stations | \"\n"
    "          f\"valid_time {df['valid_time'].min().date()} \u2192 {df['valid_time'].max().date()}\")\n"
    "    print(f\"  lead_days : {sorted(df['lead_day'].unique())}\")\n"
    "    print(f\"  pairs/stn @ lead 1: \"\n"
    "          f\"{len(df[df['lead_day']==1]) // df['station'].nunique()}\")\n"
    "    print()"
)
new_loop = (
    "for nm, df in [('maxt', maxt_paired), ('mint', mint_paired)]:\n"
    "    err = df['fcst_f'] - df['obs_f']\n"
    "    print(f\"{nm}_paired: {len(df):,} rows | {df['station'].nunique()} stns | \"\n"
    "          f\"bias={err.mean():+.2f}\u00b0F  MAE={err.abs().mean():.2f}\u00b0F\")"
)
if old_loop in _paired_src:
    _paired_src = _paired_src.replace(old_loop, new_loop)
else:
    # Fallback: strip individual verbose lines and append the compact summary
    _paired_src = strip_lines(
        _paired_src,
        r"print\(f\"\{name\}: \{len",
        r"print\(f\"  lead_days",
        r"print\(f\"  pairs/stn",
        r"^\s*print\(\)$",
    )
    _paired_src = (
        _paired_src.rstrip()
        + "\n\nfor nm, df in [('maxt', maxt_paired), ('mint', mint_paired)]:\n"
        + "    err = df['fcst_f'] - df['obs_f']\n"
        + "    print(f\"{nm}_paired: {len(df):,} rows | {df['station'].nunique()} stns | \"\n"
        + "          f\"bias={err.mean():+.2f}\u00b0F  MAE={err.abs().mean():.2f}\u00b0F\")"
    )

EXTRACT_PAIR_SRC = _extpart.rstrip() + "\n\n# \u2500\u2500 Pair with observations \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n" + _paired_src

# ── Section 3: Obs loading (cell 11) ─────────────────────────────────────────
# Strip the VAR_HOURS comment block + definition (it's now in config)
_obs_src = src(11)
# Remove lines from the big VAR_HOURS comment block through its definition
_obs_src = re.sub(
    r'# \u2500+ Valid-time offsets.*?VAR_HOURS\s*=\s*\{[^\}]+\}\n\n',
    '',
    _obs_src,
    flags=re.DOTALL,
)
# Remove the verbose final prints (keep only the summary)
_obs_src = strip_lines(
    _obs_src,
    r'print\(f"\{len\(obs_df\).*rows.*unique_times"',
    r'print\(f"valid_time range',
    r'obs_df\.head\(',
)
# Add clean summary print
_obs_src = (
    _obs_src.rstrip()
    + "\nprint(f\"Observations: {len(obs_df):,} rows | {obs_df['stid'].nunique()} stations | \"\n"
    + "      f\"{obs_df['valid_time'].min().date()} \u2192 {obs_df['valid_time'].max().date()}\")"
)

# ── Section 5: Diagnostic C (cell 18) ────────────────────────────────────────
# Strip local imports and local DIAG_* variable definitions (now in config)
_diagc_src = strip_lines(
    src(18),
    r'^import matplotlib',
    r'^DIAG_STATION\s*=',
    r'^DIAG_START\s*=',
    r'^DIAG_END\s*=',
    r'^DIAG_LEAD\s*=',
    r'^DIAG_INIT_HOUR\s*=',
)

# ── Section 6: Histogram (cell 16) ───────────────────────────────────────────
_hist_src = strip_lines(
    src(16),
    r'^import matplotlib',
    r'^import math',
    r'^LEAD_DAY\s*=',
)

# ── Section 6: Skill by f_hour (cell 19) ─────────────────────────────────────
_skill_fh_src = strip_lines(
    src(19),
    r'^import matplotlib',
)

# ── Section 6: Skill by date (cell 20) ───────────────────────────────────────
_skill_date_src = strip_lines(
    src(20),
    r'^import matplotlib',
    r'^SHORT_FHOURS\s*=',
    r'^MEDIUM_FHOURS\s*=',
)

# ── Section 7: Percentile extraction (cell 21) ── use as-is
# ── Section 7: Percentile reliability (cell 22) ─────────────────────────────
_perc_rel_src = strip_lines(
    src(22),
    r'^import matplotlib',
    r'^SKIP_PERCS\s*=',
)

# ── Section 8: Prob extraction (cell 23) ─────────────────────────────────────
_prob_ext_src = strip_lines(
    src(23),
    r'^import dask',
    r'^import dask\.array',
    r'^import numpy',
    r'^import pandas',
    r'^import xarray',
    r'^from pathlib',
    r'^from eccodes',
    r'^from analysis\.read_nbm_forecasts',
    r'^import nbm_grib_tools',
)

# ── Section 8: Threshold reliability (cell 24) ───────────────────────────────
_thresh_rel_src = strip_lines(
    src(24),
    r'^import matplotlib',
    r'^import numpy',
    r'^import pandas',
    r'^PROB_BINS\s*=',
    r'^BIN_CENTERS\s*=',
)

# ─────────────────────────────────────────────────────────────────────────────
# Assemble notebook
# ─────────────────────────────────────────────────────────────────────────────
nb = nbf.v4.new_notebook()
nb.metadata.update({
    "kernelspec": {
        "display_name": "nbmv5",
        "language": "python",
        "name": "nbmv5",
    },
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "pygments_lexer": "ipython3",
        "version": "3.12.7",
    },
})

def md(text):  return nbf.v4.new_markdown_cell(text)
def code(text): return nbf.v4.new_code_cell(text)

cells = [
    # ── Preamble ──────────────────────────────────────────────────────────────
    md(TITLE_SRC),
    code(IMPORTS_SRC),
    code(CONFIG_SRC),

    # ── 1. Load Forecast Dataset ──────────────────────────────────────────────
    md("## 1. Load Forecast Dataset\n"
       "Open the NBM QMD GRIB2 archive and compute estimated deterministic means."),
    code(_ds_open),

    # ── 2. Station Metadata & Grid Matching ───────────────────────────────────
    md("## 2. Station Metadata & Grid Matching\n"
       "Fetch active stations from the Synoptic API, match against obs-file stations, "
       "apply network filter, then map to the nearest NBM grid pixel."),
    code(STATIONS_SRC),
    code(_grid_src),

    # ── 3. Load Observations ──────────────────────────────────────────────────
    md("## 3. Load Observations\n"
       "Read obs CSV files for the study period, melt to long form, "
       "and assign valid_times matching the NBM forecast grid."),
    code(_obs_src),

    # ── 4. Extract Forecast Points & Pair ─────────────────────────────────────
    md("## 4. Extract Forecast Points & Pair with Observations\n"
       "Materialise mean point values with dask, convert K \u2192 \u00b0F, "
       "then inner-join with obs on station + valid_time."),
    code(EXTRACT_PAIR_SRC),

    # ── 5. Time-Matching Verification Diagnostic ──────────────────────────────
    md("## 5. Time-Matching Verification Diagnostic\n"
       "Verify obs and forecast are correctly aligned in time for a single station.\n"
       "Event day = calendar day the extreme occurred (= obs CSV file date = valid_time \u2212 VAR_HOURS offset).\n\n"
       "**Pairing convention** (`VAR_HOURS = {'maxt': 30, 'mint': 18}`):\n"
       "- MaxT for event day D: obs file \u2192 `obs_maxtmint_D.csv`, valid_time \u2192 D+1 06Z (+30 h)\n"
       "- MinT for event day D: obs file \u2192 `obs_maxtmint_D.csv`, valid_time \u2192 D 18Z (+18 h)\n\n"
       "Set `DIAG_INIT_HOUR=12` for a clean single-cycle series."),
    code(_diagc_src),

    # ── 6. Deterministic Skill ────────────────────────────────────────────────
    md("## 6. Deterministic Skill\n"
       "MAE, RMSE, and bias aggregated across all paired station-days."),
    code(_hist_src),
    code(_skill_fh_src),
    code(_skill_date_src),

    # ── 7. Probabilistic Verification \u2014 Percentile Reliability ──────────────────
    md("## 7. Probabilistic Verification \u2014 Percentile Reliability\n"
       "Extract QMD percentile values at station grid pixels, pair with obs, "
       "and plot reliability diagrams.\n"
       "A perfectly calibrated system lies on the 1:1 diagonal."),
    code(src(21)),        # percentile extraction — use as-is
    code(_perc_rel_src),  # reliability diagram

    # ── 8. Probabilistic Verification \u2014 Threshold Exceedance ──────────────────
    md("## 8. Probabilistic Verification \u2014 Threshold Exceedance\n"
       "Extract P(T > threshold) probability messages, pair with obs, "
       "and plot reliability diagrams.\n"
       "X-axis: NBM issued probability. Y-axis: observed exceedance rate. "
       "Perfect calibration = 1:1 diagonal."),
    code(_prob_ext_src),  # probability extraction
    code(_thresh_rel_src),  # threshold reliability diagram
]

nb.cells = cells

out = Path(r'C:\Users\Michael.wessler\Code\nbm-v5-verification\dev\nbm_verification.ipynb')
with open(out, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Created: {out}")
print(f"  {len(nb.cells)} cells")
for i, c in enumerate(nb.cells):
    t = c['cell_type']
    preview = (c['source'] or '')[:60].replace('\n', ' ')
    print(f"  [{i:2d}] {t:<8}  {preview!r}")
