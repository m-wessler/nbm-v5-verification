#!/usr/bin/env python3
"""Build a SQLite database of IEM METAR daily MaxT/MinT observations.

Fetches official ASOS METAR 6-hourly max/min temperatures (Remarks groups
1sTTT / 2sTTT) from the Iowa Environmental Mesonet (IEM) for all CONUS
(Lower 48 + DC) stations and stores the NWS climate-day high and low for
each station and calendar day.  Each station is spatially assigned to its
NWS CWA (County Warning Area).

Date range:  2025-10-01 through 2026-05-12
Data source: https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py
CWA shapes:  shapefiles/cwa.geojson

Time window definitions (NWS climate-day convention).
IEM 'valid' timestamp is the END of the 6-hr reporting period, so windows
use exclusive-start / inclusive-end bounds:
  MaxT for date D : max of max_tmpf_6hr reports with valid in (18:00Z D,  06:00Z D+1]
  MinT for date D : min of min_tmpf_6hr reports with valid in (00:00Z D,  18:00Z D  ]

API strategy: one request per CWA (using station= params) covering the full period.
Raw CSVs are cached; re-processing does not re-download unless --redownload is set.
Resume-safe: CWAs already written to the DB are skipped on subsequent runs.

Output:
  data/obs/cache/<STATE>_asos.csv   raw hourly CSV per state (cached)
  data/obs/observations.db          SQLite database (stations + daily_obs)

Usage:
  python analysis/download_obs_database.py
  python analysis/download_obs_database.py --redownload
  python analysis/download_obs_database.py --workers 12
  python analysis/download_obs_database.py --dry-run
  python analysis/download_obs_database.py --db path/to/custom.db
  python analysis/download_obs_database.py --cwa BOU GJT PUB   # CONUS CWAs only
  python analysis/download_obs_database.py --states CO WY       # explicit state filter
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

try:
    from shapely.geometry import Point, shape as shapely_shape
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
OBS_DIR    = ROOT / "data" / "obs"
CACHE_DIR  = OBS_DIR / "cache"
DEFAULT_DB = OBS_DIR / "observations.db"
CWA_GEOJSON = ROOT / "shapefiles" / "cwa.geojson"

IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
IEM_NETWORK_GEOJSON = "https://mesonet.agron.iastate.edu/geojson/network/{state}_ASOS.geojson"

# Fetch window: slightly padded beyond the date range to cover all edge windows.
#   MinT window for 2025-10-01 starts at 2025-10-01 00:00Z  -> fetch from 2025-09-30 18:00Z
#   MaxT METAR at ~06:53Z 2026-05-13 is the last relevant obs  -> fetch through 07:00Z
FETCH_START = datetime(2025, 9, 30, 18, 0, tzinfo=timezone.utc)
FETCH_END   = datetime(2026, 5, 13,  7, 0, tzinfo=timezone.utc)

# Calendar days for which we compute observations (inclusive on both ends).
OBS_DATE_START = date(2025, 10, 1)
OBS_DATE_END   = date(2026, 5, 12)

# NWS climate-day synoptic-hour assignment:
#   MaxT for date D : 1sTTT groups from METARs snapped to 00Z or 06Z on D+1
#                     covers periods (18Z D, 00Z D+1] and (00Z D+1, 06Z D+1]
#   MinT for date D : 2sTTT groups from METARs snapped to 06Z, 12Z, or 18Z on D
#                     covers periods (00Z D, 06Z D], (06Z D, 12Z D], (12Z D, 18Z D]
# ASOS METARs at :53 past the hour are snapped to the nearest synoptic hour.
# Max possible 6-hr group reports: MaxT=2, MinT=3.
MIN_OBS_MAXT = 1
MIN_OBS_MINT = 1

DEFAULT_WORKERS   = 1
MAX_RETRIES       = 6
STAGGER_DELAY_S   = 2    # seconds between submissions when using multiple workers
PRE_REQUEST_DELAY = 5    # seconds to wait before each IEM request (single-threaded politeness)

CONUS_STATES = (
    "AL", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI",
    "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY",
    "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN",
    "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS stations (
    stid         TEXT PRIMARY KEY,
    name         TEXT,
    state        TEXT,
    cwa          TEXT,
    lat          REAL,
    lon          REAL,
    elevation_m  REAL
);

CREATE TABLE IF NOT EXISTS daily_obs (
    stid              TEXT    NOT NULL,
    date              TEXT    NOT NULL,
    maxt_f            REAL,
    maxt_c            REAL,
    maxt_obs_time_utc TEXT,
    maxt_obs_count    INTEGER,
    mint_f            REAL,
    mint_c            REAL,
    mint_obs_time_utc TEXT,
    mint_obs_count    INTEGER,
    PRIMARY KEY (stid, date),
    FOREIGN KEY (stid) REFERENCES stations(stid)
);

CREATE INDEX IF NOT EXISTS idx_daily_obs_date ON daily_obs(date);
CREATE INDEX IF NOT EXISTS idx_daily_obs_stid ON daily_obs(stid);
CREATE INDEX IF NOT EXISTS idx_stations_cwa   ON stations(cwa);
CREATE INDEX IF NOT EXISTS idx_stations_state ON stations(state);
"""


def open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def upsert_stations(conn: sqlite3.Connection, stations: list[dict]) -> None:
    conn.executemany(
        """
        INSERT INTO stations (stid, name, state, cwa, lat, lon, elevation_m)
        VALUES (:stid, :name, :state, :cwa, :lat, :lon, :elevation_m)
        ON CONFLICT(stid) DO UPDATE SET
            name        = excluded.name,
            state       = excluded.state,
            cwa         = excluded.cwa,
            lat         = excluded.lat,
            lon         = excluded.lon,
            elevation_m = excluded.elevation_m
        """,
        stations,
    )
    conn.commit()


def get_completed_cwas(conn: sqlite3.Connection) -> set[str]:
    """Return set of CWA codes that already have observations written to the DB."""
    rows = conn.execute(
        """
        SELECT DISTINCT s.cwa
        FROM daily_obs d
        JOIN stations s ON d.stid = s.stid
        WHERE s.cwa IS NOT NULL
        """
    ).fetchall()
    return {r[0] for r in rows}


def upsert_daily_obs(conn: sqlite3.Connection, rows: list[dict]) -> None:
    if not rows:
        return
    conn.executemany(
        """
        INSERT INTO daily_obs
            (stid, date,
             maxt_f, maxt_c, maxt_obs_time_utc, maxt_obs_count,
             mint_f, mint_c, mint_obs_time_utc, mint_obs_count)
        VALUES
            (:stid, :date,
             :maxt_f, :maxt_c, :maxt_obs_time_utc, :maxt_obs_count,
             :mint_f, :mint_c, :mint_obs_time_utc, :mint_obs_count)
        ON CONFLICT(stid, date) DO UPDATE SET
            maxt_f            = excluded.maxt_f,
            maxt_c            = excluded.maxt_c,
            maxt_obs_time_utc = excluded.maxt_obs_time_utc,
            maxt_obs_count    = excluded.maxt_obs_count,
            mint_f            = excluded.mint_f,
            mint_c            = excluded.mint_c,
            mint_obs_time_utc = excluded.mint_obs_time_utc,
            mint_obs_count    = excluded.mint_obs_count
        """,
        rows,
    )
    conn.commit()


# ---------------------------------------------------------------------------
# CWA spatial assignment
# ---------------------------------------------------------------------------

def load_cwa_polygons() -> list[tuple[str, object]]:
    """Load CWA polygons from the project GeoJSON.  Returns list of (cwa_code, shapely_geom)."""
    if not HAS_SHAPELY:
        raise RuntimeError(
            "shapely is required for CWA assignment.  "
            "Install with: conda install shapely  or  pip install shapely"
        )
    with CWA_GEOJSON.open() as f:
        fc = json.load(f)
    polygons: list[tuple[str, object]] = []
    for feat in fc.get("features", []):
        cwa = (feat.get("properties") or {}).get("CWA", "").strip()
        if not cwa:
            continue
        geom = feat.get("geometry")
        if geom is None:
            continue
        polygons.append((cwa, shapely_shape(geom)))
    return polygons


def assign_cwas_to_stations(
    stations: dict[str, dict],
    cwa_polygons: list[tuple[str, object]],
) -> None:
    """Mutate each station dict in-place, adding a 'cwa' key.

    Uses point-in-polygon for exact assignment; falls back to the nearest
    CWA centroid for stations that don't fall cleanly inside any polygon
    (e.g., coastal or border stations).
    """
    for rec in stations.values():
        lat = rec.get("lat")
        lon = rec.get("lon")
        if lat is None or lon is None:
            rec["cwa"] = None
            continue
        pt = Point(lon, lat)
        matched = None
        for cwa_code, geom in cwa_polygons:
            if geom.contains(pt):
                matched = cwa_code
                break
        if matched is None:
            # Fallback: nearest CWA centroid
            min_dist = float("inf")
            for cwa_code, geom in cwa_polygons:
                d = pt.distance(geom.centroid)
                if d < min_dist:
                    min_dist = d
                    matched = cwa_code
        rec["cwa"] = matched


def cwas_to_states(
    target_cwas: set[str],
    station_meta: dict[str, dict],
) -> set[str]:
    """Return the set of state codes that contain at least one station in any of the target CWAs."""
    states: set[str] = set()
    for rec in station_meta.values():
        if rec.get("cwa") in target_cwas and rec.get("state"):
            states.add(rec["state"])
    return states


# ---------------------------------------------------------------------------
# IEM Station Metadata
# ---------------------------------------------------------------------------

def fetch_state_metadata(state: str) -> list[dict]:
    """Fetch IEM ASOS station metadata for one state. Returns list of station dicts."""
    url = IEM_NETWORK_GEOJSON.format(state=state)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        print(f"  WARNING: metadata fetch failed for {state}: {exc}")
        return []

    result: list[dict] = []
    for feature in data.get("features", []):
        props = feature.get("properties") or {}
        geom  = feature.get("geometry")  or {}
        stid  = (props.get("sid") or "").strip()
        if not stid:
            continue
        coords = geom.get("coordinates") or []
        lon = float(coords[0]) if len(coords) > 0 else None
        lat = float(coords[1]) if len(coords) > 1 else None
        result.append({
            "stid":        stid,
            "name":        (props.get("sname") or "").strip(),
            "state":       (props.get("state") or state).strip().upper(),
            "lat":         lat,
            "lon":         lon,
            "elevation_m": props.get("elevation"),
        })
    return result


def fetch_all_station_metadata(workers: int) -> dict[str, dict]:
    """Fetch ASOS station metadata for all CONUS states in parallel.
    Returns dict keyed by stid."""
    stations: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fetch_state_metadata, s): s for s in CONUS_STATES}
        for future in as_completed(futures):
            for rec in future.result():
                stations[rec["stid"]] = rec
    return stations


# ---------------------------------------------------------------------------
# IEM ASOS CSV Download  (CWA-native)
# ---------------------------------------------------------------------------

def build_cwa_asos_url(stids: list[str]) -> str:
    """Build an IEM asos.py URL that requests specific station IDs.

    Uses repeated ``station=`` parameters so IEM returns only the stations
    belonging to this CWA — no state/network routing needed.
    """
    base: list[tuple[str, str]] = [
        ("data",        "metar"),
        ("tz",          "UTC"),
        ("format",      "onlycomma"),
        ("latlon",      "yes"),
        ("report_type", "3"),
        ("year1",   str(FETCH_START.year)),
        ("month1",  str(FETCH_START.month)),
        ("day1",    str(FETCH_START.day)),
        ("hour1",   str(FETCH_START.hour)),
        ("minute1", "0"),
        ("year2",   str(FETCH_END.year)),
        ("month2",  str(FETCH_END.month)),
        ("day2",    str(FETCH_END.day)),
        ("hour2",   str(FETCH_END.hour)),
        ("minute2", "0"),
    ]
    station_params = [("station", s) for s in sorted(stids)]
    return f"{IEM_ASOS_URL}?{urllib.parse.urlencode(base + station_params)}"


def download_cwa_csv(cwa: str, stids: list[str], cache_path: Path, redownload: bool) -> Path:
    """Download the IEM ASOS CSV for a CWA's station list and cache it."""
    if cache_path.exists() and not redownload:
        return cache_path
    if redownload and cache_path.exists():
        cache_path.unlink()

    url = build_cwa_asos_url(stids)

    for attempt in range(1, MAX_RETRIES + 1):
        time.sleep(PRE_REQUEST_DELAY)  # be polite to IEM before every attempt
        try:
            with urllib.request.urlopen(url, timeout=300) as resp:
                data = resp.read()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(data)
            return cache_path
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 503):
                wait = 60 * attempt
                print(f"  [{cwa}] HTTP {exc.code}, retrying in {wait}s (attempt {attempt}/{MAX_RETRIES})")
            elif exc.code in (500, 502, 504) and attempt < MAX_RETRIES:
                wait = 30 * attempt
                print(f"  [{cwa}] HTTP {exc.code}, retrying in {wait}s (attempt {attempt}/{MAX_RETRIES})")
            else:
                raise
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt < MAX_RETRIES:
                wait = 30 * attempt
                print(f"  [{cwa}] Network error: {exc} — retrying in {wait}s (attempt {attempt}/{MAX_RETRIES})")
                time.sleep(wait)
            else:
                raise

    return cache_path  # unreachable


# ---------------------------------------------------------------------------
# Daily MaxT / MinT computation
# ---------------------------------------------------------------------------

# Regex for METAR 6-hr max (1sTTT) and min (2sTTT) groups in the RMK section.
# s = '0' (positive/zero) or '1' (negative); TTT = tenths of °C (e.g. 283 → 28.3°C).
_RE_6HR_MAX = re.compile(r'(?<!\d)1([01])(\d{3})(?!\d)')
_RE_6HR_MIN = re.compile(r'(?<!\d)2([01])(\d{3})(?!\d)')


def _decode_6hr_temp(sign: str, digits: str) -> float:
    """Convert METAR 6-hr group sign + 3-digit tenths-of-°C to °F."""
    temp_c = int(digits) / 10.0
    if sign == '1':
        temp_c = -temp_c
    return temp_c * 9.0 / 5.0 + 32.0


def _parse_metar_6hr(metar_text: str) -> tuple[float | None, float | None]:
    """Extract (max_tempF, min_tempF) from METAR remarks 1sTTT / 2sTTT groups.

    Searches only in the RMK section so as not to confuse body groups.
    Returns (None, None) when no groups are found.
    """
    rmk_idx = metar_text.find(' RMK ')
    if rmk_idx < 0:
        rmk_idx = metar_text.rfind(' RMK')
        if rmk_idx < 0:
            return None, None
    rmk = metar_text[rmk_idx:]

    mx_f = mn_f = None
    m = _RE_6HR_MAX.search(rmk)
    if m:
        mx_f = _decode_6hr_temp(m.group(1), m.group(2))
    m = _RE_6HR_MIN.search(rmk)
    if m:
        mn_f = _decode_6hr_temp(m.group(1), m.group(2))
    return mx_f, mn_f


def _snap_to_synoptic(t: datetime) -> datetime:
    """Round a datetime to the nearest 6-hr synoptic time (00Z, 06Z, 12Z, 18Z).

    ASOS stations typically report at :53–:55 past each hour.  This function
    maps those times to the synoptic slot they represent so that 06:53Z becomes
    06:00Z and 18:53Z becomes 18:00Z, etc.
    """
    frac_h = t.hour + t.minute / 60.0 + t.second / 3600.0
    snap_h = round(frac_h / 6) * 6   # 0, 6, 12, 18, or 24
    base = t.replace(hour=0, minute=0, second=0, microsecond=0)
    return base + timedelta(hours=snap_h)


def _parse_valid_time(s: str) -> datetime | None:
    """Parse IEM's 'YYYY-MM-DD HH:MM' UTC timestamp to an aware datetime."""
    s = s.strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def compute_daily_obs_from_csv(
    csv_path: Path,
    station_meta: dict[str, dict],
) -> list[dict]:
    """Parse a CWA's cached METAR CSV and extract daily MaxT/MinT via 6-hr groups.

    Reads the raw METAR text column, extracts 1sTTT (6-hr max) and 2sTTT
    (6-hr min) remarks groups, snaps each observation to its synoptic hour
    (00Z/06Z/12Z/18Z), and assigns values to the correct climate day.

    Returns a list of daily_obs row dicts ready to upsert into SQLite.
    """
    # accum[stid][date] = {"mx_val": float|None, "mx_t": str|None, "mx_n": int,
    #                       "mn_val": float|None, "mn_t": str|None, "mn_n": int}
    accum: dict[str, dict[date, dict]] = {}

    try:
        raw = csv_path.read_bytes().decode("utf-8", errors="replace")
    except Exception as exc:
        print(f"  WARNING: Cannot read {csv_path.name}: {exc}")
        return []

    data_lines = [ln for ln in raw.splitlines() if ln.strip() and not ln.startswith("#")]
    if not data_lines:
        print(f"  WARNING: Empty CSV {csv_path.name}")
        return []

    reader = csv.DictReader(io.StringIO("\n".join(data_lines)))
    for row in reader:
        stid = (row.get("station") or "").strip()
        if not stid or stid not in station_meta:
            continue

        obs_time = _parse_valid_time(row.get("valid") or "")
        if obs_time is None:
            continue

        # Parse the 6-hr max (1sTTT) and min (2sTTT) groups from the raw METAR text.
        metar_text = (row.get("metar") or "").strip()
        if not metar_text:
            continue
        mx_f, mn_f = _parse_metar_6hr(metar_text)
        if mx_f is None and mn_f is None:
            continue

        obs_time_str = f"{obs_time:%Y-%m-%d %H:%MZ}"

        # Snap the METAR time to its synoptic slot (00Z, 06Z, 12Z, or 18Z).
        # ASOS stations report at :53–:55; snapping maps them to the correct slot.
        snap_t = _snap_to_synoptic(obs_time)
        snap_h = snap_t.hour   # 0, 6, 12, or 18
        snap_d = snap_t.date()

        # MaxT: 1sTTT groups appear in METARs at 00Z and 06Z on D+1.
        #   00Z D+1 covers (18Z D, 00Z D+1]; 06Z D+1 covers (00Z D+1, 06Z D+1].
        if mx_f is not None and snap_h in (0, 6):
            mx_cd = snap_d - timedelta(days=1)
            if OBS_DATE_START <= mx_cd <= OBS_DATE_END:
                accum.setdefault(stid, {}).setdefault(mx_cd, {
                    "mx_val": None, "mx_t": None, "mx_n": 0,
                    "mn_val": None, "mn_t": None, "mn_n": 0,
                })
                slot = accum[stid][mx_cd]
                slot["mx_n"] += 1
                if slot["mx_val"] is None or mx_f > slot["mx_val"]:
                    slot["mx_val"] = mx_f
                    slot["mx_t"]   = obs_time_str

        # MinT: 2sTTT groups appear in METARs at 06Z, 12Z, 18Z on D.
        #   06Z D covers (00Z D, 06Z D]; 12Z covers (06Z, 12Z]; 18Z covers (12Z, 18Z].
        if mn_f is not None and snap_h in (6, 12, 18):
            mn_cd = snap_d
            if OBS_DATE_START <= mn_cd <= OBS_DATE_END:
                accum.setdefault(stid, {}).setdefault(mn_cd, {
                    "mx_val": None, "mx_t": None, "mx_n": 0,
                    "mn_val": None, "mn_t": None, "mn_n": 0,
                })
                slot = accum[stid][mn_cd]
                slot["mn_n"] += 1
                if slot["mn_val"] is None or mn_f < slot["mn_val"]:
                    slot["mn_val"] = mn_f
                    slot["mn_t"]   = obs_time_str

    # Build output rows
    rows: list[dict] = []
    for stid, date_dict in accum.items():
        for cd, slot in date_dict.items():
            mx_val = slot["mx_val"]
            mn_val = slot["mn_val"]
            # Skip entirely if neither value meets minimum obs threshold.
            if (mx_val is None or slot["mx_n"] < MIN_OBS_MAXT) and \
               (mn_val is None or slot["mn_n"] < MIN_OBS_MINT):
                continue

            maxt_f = round(mx_val, 1)          if (mx_val is not None and slot["mx_n"] >= MIN_OBS_MAXT) else None
            maxt_c = round((maxt_f - 32) * 5/9, 1) if maxt_f is not None else None
            mint_f = round(mn_val, 1)          if (mn_val is not None and slot["mn_n"] >= MIN_OBS_MINT) else None
            mint_c = round((mint_f - 32) * 5/9, 1) if mint_f is not None else None

            rows.append({
                "stid":              stid,
                "date":              cd.isoformat(),
                "maxt_f":            maxt_f,
                "maxt_c":            maxt_c,
                "maxt_obs_time_utc": slot["mx_t"] if maxt_f is not None else None,
                "maxt_obs_count":    slot["mx_n"] if maxt_f is not None else None,
                "mint_f":            mint_f,
                "mint_c":            mint_c,
                "mint_obs_time_utc": slot["mn_t"] if mint_f is not None else None,
                "mint_obs_count":    slot["mn_n"] if mint_f is not None else None,
            })

    return rows


# ---------------------------------------------------------------------------
# Per-CWA download + processing
# ---------------------------------------------------------------------------

def download_cwa_task(
    cwa: str,
    cwa_station_meta: dict[str, dict],
    redownload: bool,
    dry_run: bool,
) -> tuple[str, str | None]:
    """Download the CSV for one CWA directly from IEM.  Returns (cwa, error_or_None)."""
    cache_path = CACHE_DIR / f"{cwa}_asos.csv"
    stids = list(cwa_station_meta.keys())

    if dry_run:
        url = build_cwa_asos_url(stids)
        cached = "EXISTS" if cache_path.exists() else "PENDING"
        print(f"    [{cwa}] {cached}  ({len(stids)} stations)  {url[:120]}...")
        return cwa, None

    try:
        existed = cache_path.exists() and not redownload
        download_cwa_csv(cwa, stids, cache_path, redownload)
        status = "EXISTS" if existed else "DOWNLOADED"
        print(f"    [{cwa}] {status} ({cache_path.stat().st_size // 1024} KB, {len(stids)} stations)")
        return cwa, None
    except Exception as exc:
        print(f"    [{cwa}] ERROR: {exc}")
        return cwa, str(exc)


def process_cwa(
    cwa: str,
    cwa_station_meta: dict[str, dict],
) -> list[dict]:
    """Compute daily MaxT/MinT rows for a CWA from its cached CSV."""
    csv_path = CACHE_DIR / f"{cwa}_asos.csv"
    if not csv_path.exists():
        print(f"  [{cwa}] WARNING: no cached CSV — skipping")
        return []
    return compute_daily_obs_from_csv(csv_path, cwa_station_meta)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download IEM METAR daily MaxT/MinT for CONUS and store in SQLite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help=f"Path to output SQLite database (default: {DEFAULT_DB})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel download threads (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--redownload",
        action="store_true",
        help="Force re-download of cached CSVs and recompute all observations.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned downloads without writing any files.",
    )
    parser.add_argument(
        "--states",
        nargs="+",
        metavar="ST",
        help="Process only these state codes (e.g. --states CO WY MT). Cannot combine with --cwa.",
    )
    parser.add_argument(
        "--cwa",
        nargs="+",
        metavar="CWA",
        help=(
            "Process only these NWS CWA codes (e.g. --cwa BOU GJT PUB). "
            "The required states are derived automatically. Cannot combine with --states."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    db_path: Path = args.db
    workers: int  = args.workers
    dry_run: bool = args.dry_run
    redownload: bool = args.redownload

    if args.states and args.cwa:
        print("ERROR: --states and --cwa are mutually exclusive.")
        raise SystemExit(1)

    # CWA assignment requires shapely; fail fast with a clear message.
    if not HAS_SHAPELY:
        print("ERROR: shapely is required. Install with: conda install shapely")
        raise SystemExit(1)

    # Load CWA polygons up front — needed for metadata spatial join.
    print("Loading CWA polygons...")
    cwa_polygons = load_cwa_polygons()
    all_cwa_codes = {c for c, _ in cwa_polygons}
    print(f"  Loaded {len(cwa_polygons)} CWA polygons from {CWA_GEOJSON.name}")
    print()

    # Validate --cwa input if provided.
    target_cwas: set[str] | None = None
    if args.cwa:
        target_cwas = {c.upper() for c in args.cwa}
        unknown = target_cwas - all_cwa_codes
        if unknown:
            print(f"ERROR: Unknown CWA code(s): {', '.join(sorted(unknown))}")
            print(f"  Known CWAs: {', '.join(sorted(all_cwa_codes))}")
            raise SystemExit(1)

    # Validate --states input if provided.
    if args.states:
        explicit_states = tuple(s.upper() for s in args.states)
        invalid = [s for s in explicit_states if s not in CONUS_STATES]
        if invalid:
            print(f"ERROR: Unknown state code(s): {', '.join(invalid)}")
            raise SystemExit(1)
    else:
        explicit_states = None

    print("=" * 60)
    print("IEM METAR Observation Database Builder")
    print("=" * 60)
    print(f"Date range   : {OBS_DATE_START}  →  {OBS_DATE_END}")
    print(f"Fetch window : {FETCH_START:%Y-%m-%d %H:%MZ}  →  {FETCH_END:%Y-%m-%d %H:%MZ}")
    if args.cwa:
        print(f"CWA filter   : {', '.join(sorted(target_cwas))}")
    elif explicit_states:
        print(f"State filter : {', '.join(explicit_states)}")
    else:
        print(f"Coverage     : all CONUS ({len(CONUS_STATES)} states)")
    print(f"Workers      : {workers}")
    print(f"Database     : {db_path}")
    print(f"Cache dir    : {CACHE_DIR}")
    if dry_run:
        print("*** DRY RUN — no files will be written ***")
    print()

    # ------------------------------------------------------------------
    # Step 1: Station metadata + CWA spatial join
    # ------------------------------------------------------------------
    print("Step 1: Fetching ASOS station metadata for all CONUS states...")
    station_meta = fetch_all_station_metadata(workers)
    print(f"  Loaded {len(station_meta)} raw ASOS station records.")

    print("  Assigning CWAs via spatial join...")
    assign_cwas_to_stations(station_meta, cwa_polygons)
    cwa_counts = {}
    for rec in station_meta.values():
        c = rec.get("cwa") or "UNKNOWN"
        cwa_counts[c] = cwa_counts.get(c, 0) + 1
    print(f"  Stations assigned to {len(cwa_counts)} CWAs "
          f"({cwa_counts.get('UNKNOWN', 0)} unassigned).")

    # Resolve which CWAs and states to work with.
    if target_cwas is not None:
        # --cwa: narrow station_meta to only stations in the target CWAs.
        station_meta = {k: v for k, v in station_meta.items() if v.get("cwa") in target_cwas}
        if not station_meta:
            print(f"ERROR: No CONUS stations found for CWA(s): {', '.join(sorted(target_cwas))}")
            raise SystemExit(1)
    elif explicit_states is not None:
        station_meta = {k: v for k, v in station_meta.items() if v.get("state") in set(explicit_states)}

    # Collect the ordered set of CWAs present in the (filtered) station_meta.
    all_cwas: list[str] = sorted({v["cwa"] for v in station_meta.values() if v.get("cwa")})
    # States required to download data for those CWAs.
    required_states: list[str] = sorted(
        {v["state"] for v in station_meta.values() if v.get("state")} & set(CONUS_STATES)
    )

    print(f"  Final: {len(station_meta)} stations across {len(all_cwas)} CWA(s) "
          f"in {len(required_states)} state(s).")
    print()

    # ------------------------------------------------------------------
    # Step 2: Initialize database and upsert station records
    # ------------------------------------------------------------------
    if not dry_run:
        print("Step 2: Initializing database...")
        conn = open_db(db_path)
        upsert_stations(conn, list(station_meta.values()))
        print(f"  Upserted {len(station_meta)} station records.")
        completed_cwas = get_completed_cwas(conn)
        if completed_cwas:
            print(f"  Resuming: {len(completed_cwas)} CWA(s) already in DB — will skip unless --redownload.")
            print(f"    Skipping: {', '.join(sorted(completed_cwas))}")
        print()
    else:
        conn = None  # type: ignore[assignment]
        completed_cwas: set[str] = set()  # type: ignore[no-redef]
        print("Step 2: (dry run) Database initialization skipped.")
        print()

    pending_cwas = [c for c in all_cwas if redownload or c not in completed_cwas]
    skipped_cwas = len(all_cwas) - len(pending_cwas)

    # ------------------------------------------------------------------
    # Step 3a: Download CWA CSVs directly from IEM (parallel)
    # ------------------------------------------------------------------
    print(f"Step 3: Processing {len(pending_cwas)} CWA(s) "
          f"({skipped_cwas} already in DB, skipped)")
    print(f"  Downloading {len(pending_cwas)} CWA CSV(s) from IEM "
          f"[{workers} workers, {STAGGER_DELAY_S}s stagger]...")

    dl_errors: list[str] = []

    if pending_cwas:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {}
            for i, cwa in enumerate(pending_cwas):
                cwa_meta = {k: v for k, v in station_meta.items() if v.get("cwa") == cwa}
                futures[ex.submit(download_cwa_task, cwa, cwa_meta, redownload, dry_run)] = cwa
                if not dry_run and i < len(pending_cwas) - 1:
                    time.sleep(STAGGER_DELAY_S)
            for future in as_completed(futures):
                cwa_code, err = future.result()
                if err:
                    dl_errors.append(f"{cwa_code}: {err}")

    # ------------------------------------------------------------------
    # Step 3b: Process per-CWA and write to DB immediately
    # ------------------------------------------------------------------
    errors: list[str] = list(dl_errors)
    total_rows_written = 0

    if not dry_run:
        print(f"  Processing {len(pending_cwas)} CWA(s)...")
        for cwa in pending_cwas:
            cwa_meta = {k: v for k, v in station_meta.items() if v.get("cwa") == cwa}
            rows = process_cwa(cwa, cwa_meta)
            if conn is not None:
                upsert_daily_obs(conn, rows)
                total_rows_written += len(rows)
                print(f"  [{cwa}] {len(cwa_meta)} stations, {len(rows)} obs rows — written to DB")
    print()

    # ------------------------------------------------------------------
    # Step 4: Final summary
    # ------------------------------------------------------------------
    if not dry_run and conn is not None:
        conn.close()

        print(f"Step 4: {total_rows_written} new observation rows written this run.")
        print()

        # Summary stats
        conn2 = sqlite3.connect(str(db_path))
        n_stations_db = conn2.execute("SELECT COUNT(*) FROM stations").fetchone()[0]
        n_cwas_db     = conn2.execute("SELECT COUNT(DISTINCT cwa) FROM stations WHERE cwa IS NOT NULL").fetchone()[0]
        n_obs_db      = conn2.execute("SELECT COUNT(*) FROM daily_obs").fetchone()[0]
        n_dates_db    = conn2.execute("SELECT COUNT(DISTINCT date) FROM daily_obs").fetchone()[0]
        n_maxt_db     = conn2.execute("SELECT COUNT(*) FROM daily_obs WHERE maxt_f IS NOT NULL").fetchone()[0]
        n_mint_db     = conn2.execute("SELECT COUNT(*) FROM daily_obs WHERE mint_f IS NOT NULL").fetchone()[0]
        conn2.close()

        print()
        print("=" * 60)
        print("Database Summary")
        print("=" * 60)
        print(f"  Stations          : {n_stations_db}")
        print(f"  CWAs              : {n_cwas_db}")
        print(f"  Total obs rows    : {n_obs_db}")
        print(f"  Distinct dates    : {n_dates_db}")
        print(f"  MaxT values       : {n_maxt_db}")
        print(f"  MinT values       : {n_mint_db}")
        print(f"  DB path           : {db_path}")

    if errors:
        print()
        print(f"WARNINGS: {len(errors)} download error(s):")
        for e in errors:
            print(f"  {e}")

    if dry_run:
        print()
        print("Dry run complete. No files written.")


if __name__ == "__main__":
    main()

