"""Group W — Overseas weather (Open-Meteo archive API).

Purpose
-------
Daily weather data for US state capitals, Japanese prefecture capitals, and
EU-10 country capitals.  Used as exogenous climate features for overseas ILI
rate modelling (temperature, humidity, precipitation correlate with flu spread).

Source
------
Open-Meteo Historical Weather API — free, no API key required.
  URL: https://archive-api.open-meteo.com/v1/archive
  Docs: https://open-meteo.com/en/docs/historical-weather-api
  Rate limit: 10,000 calls/day; 5 req/s.  No auth needed.

Variables (daily):
  temperature_2m_max, temperature_2m_min,
  precipitation_sum, windspeed_10m_max,
  relative_humidity_2m_max, relative_humidity_2m_min

Coverage
--------
  US:  51 state + DC capitals (lat/lon of capital city)
  JP:  47 prefecture capitals
  EU:  10 country capitals — FR, GB, IT, ES, NL, BE, AT, PL, RO, DE

Output table: overseas_weather
  (source TEXT, country TEXT, location TEXT, date TEXT,
   temp_max REAL, temp_min REAL, precipitation REAL,
   windspeed REAL, humidity_max REAL, humidity_min REAL,
   collected_at TEXT,
   PRIMARY KEY (source, country, location, date))

CLI:
  .venv/bin/python -m simulation.collectors.group_w_overseas_weather
  .venv/bin/python -m simulation.collectors.group_w_overseas_weather --years-back 3 --skip-us

Design (D-4 Deep Module)
------------------------
Public surface: run(backfill_days=None, db_path=None, years_back=3,
                    skip_us=False, skip_jp=False, skip_eu=False) → dict
All location lists, API calls, and DB writes are encapsulated.

Gray-box contract (D-5):
  - Fetches in parallel (ThreadPoolExecutor, max_workers=6).
  - Per-location error → logged + skipped, does not abort run.
  - Open-Meteo: 1 API call per location, 1 retry on 429/5xx.
  - DB upsert via INSERT OR REPLACE (full row update on conflict).
  - Returns {"inserted": N, "skipped": N, "errors": [...], "locations": N}.
  - Side effects: writes overseas_weather in epi_real_seoul.db only.
"""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path
from sqlite3 import Connection as _Conn
from typing import Optional

# G-116/G-117: safe_connect via lazy import
def _safe_connect_import():
    from simulation.database import safe_connect
    return safe_connect


log = logging.getLogger(__name__)

_DEFAULT_DB = Path("simulation/data/db/epi_real_seoul.db")
_DEFAULT_YEARS_BACK = 3
# Sprint α Item 9 (2026-05-26): Open-Meteo URL centralized in _endpoints
from simulation.collectors._endpoints import _OPEN_METEO_URL
_TIMEOUT_S = 45
_MAX_WORKERS = 2   # Open-Meteo free tier: hourly limit → reduce to 2 to avoid 429

# Daily variables to request
_DAILY_VARS = (
    "temperature_2m_max,temperature_2m_min,"
    "precipitation_sum,windspeed_10m_max,"
    "relative_humidity_2m_max,relative_humidity_2m_min"
)

# ─────────────────────────────────────────────────────────────────────────────
# Location tables
# ─────────────────────────────────────────────────────────────────────────────

# (location_name, state_code_or_pref_code, lat, lon)
_US_CAPITALS: list[tuple[str, str, float, float]] = [
    ("Montgomery",    "AL",  32.3617, -86.2791),
    ("Juneau",        "AK",  58.3005, -134.4197),
    ("Phoenix",       "AZ",  33.4484, -112.0740),
    ("Little Rock",   "AR",  34.7361, -92.3311),
    ("Sacramento",    "CA",  38.5556, -121.4689),
    ("Denver",        "CO",  39.7392, -104.9903),
    ("Hartford",      "CT",  41.7658, -72.6734),
    ("Dover",         "DE",  39.1582, -75.5244),
    ("Washington DC", "DC",  38.8951, -77.0364),
    ("Tallahassee",   "FL",  30.4383, -84.2807),
    ("Atlanta",       "GA",  33.7490, -84.3880),
    ("Honolulu",      "HI",  21.3069, -157.8583),
    ("Boise",         "ID",  43.6150, -116.2023),
    ("Springfield",   "IL",  39.7989, -89.6544),
    ("Indianapolis",  "IN",  39.7684, -86.1581),
    ("Des Moines",    "IA",  41.5868, -93.6250),
    ("Topeka",        "KS",  39.0558, -95.6890),
    ("Frankfort",     "KY",  38.2009, -84.8733),
    ("Baton Rouge",   "LA",  30.4515, -91.1871),
    ("Augusta",       "ME",  44.3106, -69.7795),
    ("Annapolis",     "MD",  38.9784, -76.4922),
    ("Boston",        "MA",  42.3601, -71.0589),
    ("Lansing",       "MI",  42.7325, -84.5555),
    ("Saint Paul",    "MN",  44.9537, -93.0900),
    ("Jackson",       "MS",  32.2988, -90.1848),
    ("Jefferson City","MO",  38.5767, -92.1735),
    ("Helena",        "MT",  46.5958, -112.0270),
    ("Lincoln",       "NE",  40.8136, -96.7026),
    ("Carson City",   "NV",  39.1638, -119.7674),
    ("Concord",       "NH",  43.2081, -71.5376),
    ("Trenton",       "NJ",  40.2171, -74.7429),
    ("Santa Fe",      "NM",  35.6870, -105.9378),
    ("Albany",        "NY",  42.6526, -73.7562),
    ("Raleigh",       "NC",  35.7796, -78.6382),
    ("Bismarck",      "ND",  46.8083, -100.7837),
    ("Columbus",      "OH",  39.9612, -82.9988),
    ("Oklahoma City", "OK",  35.4676, -97.5164),
    ("Salem",         "OR",  44.9429, -123.0351),
    ("Harrisburg",    "PA",  40.2732, -76.8867),
    ("Providence",    "RI",  41.8240, -71.4128),
    ("Columbia",      "SC",  34.0007, -81.0348),
    ("Pierre",        "SD",  44.3683, -100.3510),
    ("Nashville",     "TN",  36.1627, -86.7816),
    ("Austin",        "TX",  30.2672, -97.7431),
    ("Salt Lake City","UT",  40.7608, -111.8910),
    ("Montpelier",    "VT",  44.2601, -72.5754),
    ("Richmond",      "VA",  37.5407, -77.4360),
    ("Olympia",       "WA",  47.0379, -122.9007),
    ("Charleston",    "WV",  38.3498, -81.6326),
    ("Madison",       "WI",  43.0731, -89.4012),
    ("Cheyenne",      "WY",  41.1400, -104.8202),
]

# (location_name, pref_code_2digit, lat, lon)
_JP_CAPITALS: list[tuple[str, str, float, float]] = [
    ("Sapporo",    "01",  43.0642, 141.3469),
    ("Aomori",     "02",  40.8222, 140.7444),
    ("Morioka",    "03",  39.7036, 141.1527),
    ("Sendai",     "04",  38.2682, 140.8694),
    ("Akita",      "05",  39.7186, 140.1024),
    ("Yamagata",   "06",  38.2553, 140.3396),
    ("Fukushima",  "07",  37.7608, 140.4748),
    ("Mito",       "08",  36.3416, 140.4469),
    ("Utsunomiya", "09",  36.5548, 139.8827),
    ("Maebashi",   "10",  36.3911, 139.0608),
    ("Saitama",    "11",  35.8617, 139.6455),
    ("Chiba",      "12",  35.6074, 140.1065),
    ("Tokyo",      "13",  35.6762, 139.6503),
    ("Yokohama",   "14",  35.4437, 139.6380),
    ("Niigata",    "15",  37.9162, 139.0364),
    ("Toyama",     "16",  36.6953, 137.2113),
    ("Kanazawa",   "17",  36.5948, 136.6256),
    ("Fukui",      "18",  36.0652, 136.2217),
    ("Kofu",       "19",  35.6639, 138.5681),
    ("Nagano",     "20",  36.6513, 138.1810),
    ("Gifu",       "21",  35.3912, 136.7223),
    ("Shizuoka",   "22",  34.9756, 138.3828),
    ("Nagoya",     "23",  35.1815, 136.9066),
    ("Tsu",        "24",  34.7303, 136.5086),
    ("Otsu",       "25",  35.0045, 135.8686),
    ("Kyoto",      "26",  35.0116, 135.7681),
    ("Osaka",      "27",  34.6937, 135.5023),
    ("Kobe",       "28",  34.6901, 135.1956),
    ("Nara",       "29",  34.6851, 135.8050),
    ("Wakayama",   "30",  34.2260, 135.1675),
    ("Tottori",    "31",  35.5011, 134.2351),
    ("Matsue",     "32",  35.4723, 133.0505),
    ("Okayama",    "33",  34.6554, 133.9197),
    ("Hiroshima",  "34",  34.3853, 132.4553),
    ("Yamaguchi",  "35",  34.1858, 131.4706),
    ("Tokushima",  "36",  34.0658, 134.5593),
    ("Takamatsu",  "37",  34.3428, 134.0440),
    ("Matsuyama",  "38",  33.8416, 132.7658),
    ("Kochi",      "39",  33.5597, 133.5311),
    ("Fukuoka",    "40",  33.5904, 130.4017),
    ("Saga",       "41",  33.2494, 130.2990),
    ("Nagasaki",   "42",  32.7503, 129.8779),
    ("Kumamoto",   "43",  32.8031, 130.7079),
    ("Oita",       "44",  33.2382, 131.6126),
    ("Miyazaki",   "45",  31.9077, 131.4202),
    ("Kagoshima",  "46",  31.5602, 130.5581),
    ("Naha",       "47",  26.2124, 127.6809),
]

# (capital_name, country_iso2, lat, lon)
_EU_CAPITALS: list[tuple[str, str, float, float]] = [
    ("Paris",     "FR",  48.8566,   2.3522),
    ("London",    "GB",  51.5074,  -0.1278),
    ("Rome",      "IT",  41.9028,  12.4964),
    ("Madrid",    "ES",  40.4168,  -3.7038),
    ("Amsterdam", "NL",  52.3676,   4.9041),
    ("Brussels",  "BE",  50.8503,   4.3517),
    ("Vienna",    "AT",  48.2082,  16.3738),
    ("Warsaw",    "PL",  52.2297,  21.0122),
    ("Bucharest", "RO",  44.4268,  26.1025),
    ("Berlin",    "DE",  52.5200,  13.4050),
    ("Stockholm", "SE",  59.3293,  18.0686),  # Sweden — previously missing
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_db(db_path: Optional[str | Path]) -> Path:
    if db_path is not None:
        return Path(db_path)
    cwd = Path.cwd()
    for p in [_DEFAULT_DB, cwd / _DEFAULT_DB, cwd.parent / _DEFAULT_DB]:
        if p.exists():
            return p
    return _DEFAULT_DB


def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        return f if f == f else None  # NaN guard
    except (TypeError, ValueError):
        return None


def _connect(db_path: Path) -> _Conn:
    safe_connect = _safe_connect_import()
    return safe_connect(str(db_path), timeout=60.0)


def _ensure_table(con: _Conn) -> None:
    """Create overseas_weather if absent; add missing columns for back-compat."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS overseas_weather (
            source        TEXT NOT NULL DEFAULT 'open_meteo',
            country       TEXT NOT NULL,
            location      TEXT NOT NULL,
            date          TEXT NOT NULL,
            temp_max      REAL,
            temp_min      REAL,
            precipitation REAL,
            windspeed     REAL,
            humidity_max  REAL,
            humidity_min  REAL,
            collected_at  TEXT,
            PRIMARY KEY (source, country, location, date)
        )
    """)
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_ow_country_date "
        "ON overseas_weather(country, date)"
    )
    con.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Open-Meteo fetch (one location)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_location(
    location: str,
    country: str,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    now_iso: str,
) -> list[dict]:
    """Fetch daily weather for one location from Open-Meteo archive.

    Args:
        location:   Human-readable location name (city/capital).
        country:    ISO2 country code (stored in DB).
        lat, lon:   Coordinates for Open-Meteo query.
        start_date: YYYY-MM-DD start (inclusive).
        end_date:   YYYY-MM-DD end (inclusive).
        now_iso:    Collection timestamp string (UTC ISO-8601).

    Returns:
        List of row dicts (one per day) with overseas_weather schema keys.
        Empty list on API error (logged; does not raise).

    Raises:
        Nothing — errors are caught and logged.

    Performance: 1-2 HTTP requests, ~50 KB JSON per location.
    Side effects: None (pure fetch).
    """
    try:
        import requests
    except ImportError:
        log.error("[group_w] requests not installed — pip install requests")
        return []

    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start_date,
        "end_date":   end_date,
        "daily":      _DAILY_VARS,
        "timezone":   "auto",
    }

    data: dict = {}
    for attempt in range(2):
        try:
            resp = requests.get(_OPEN_METEO_URL, params=params, timeout=_TIMEOUT_S)
            if resp.status_code == 429:
                time.sleep(20)
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as e:
            if attempt == 0:
                log.warning("[group_w] %s/%s attempt 1 failed: %s — retrying", country, location, e)
                time.sleep(5)
                continue
            log.error("[group_w] %s/%s fetch failed: %s", country, location, e)
            return []
    else:
        # for loop exhausted without break → both attempts hit 429 or non-exception failure
        log.error("[group_w] %s/%s all attempts exhausted (rate-limited or empty response)", country, location)
        return []

    daily = data.get("daily", {})
    dates         = daily.get("time", [])
    temp_max_list = daily.get("temperature_2m_max", [])
    temp_min_list = daily.get("temperature_2m_min", [])
    precip_list   = daily.get("precipitation_sum", [])
    wind_list     = daily.get("windspeed_10m_max", [])
    hum_max_list  = daily.get("relative_humidity_2m_max", [])
    hum_min_list  = daily.get("relative_humidity_2m_min", [])

    rows: list[dict] = []
    for i, date in enumerate(dates):
        def _get(lst):
            return _safe_float(lst[i]) if i < len(lst) else None

        rows.append({
            "source":        "open_meteo",
            "country":       country,
            "location":      location,
            "date":          date,
            "temp_max":      _get(temp_max_list),
            "temp_min":      _get(temp_min_list),
            "precipitation": _get(precip_list),
            "windspeed":     _get(wind_list),
            "humidity_max":  _get(hum_max_list),
            "humidity_min":  _get(hum_min_list),
            "collected_at":  now_iso,
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Batch fetch (parallel)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_locations_parallel(
    location_specs: list[tuple[str, str, float, float]],  # (name, country_iso2, lat, lon)
    start_date: str,
    end_date: str,
    now_iso: str,
) -> tuple[list[dict], list[str]]:
    """Fetch weather for a list of locations in parallel.

    Args:
        location_specs: List of (location_name, country_iso2, lat, lon).
        start_date:     YYYY-MM-DD.
        end_date:       YYYY-MM-DD.
        now_iso:        Collection timestamp.

    Returns:
        (all_rows, errors) — all_rows is the flat list of daily dicts;
        errors is a list of "country/location: msg" strings for failed fetches.

    Performance: ThreadPoolExecutor(max_workers=_MAX_WORKERS), ≤5 req/s.
    Side effects: None.
    Caller responsibility: caller writes rows to DB.
    """
    all_rows: list[dict] = []
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        future_to_spec = {
            pool.submit(
                _fetch_location,
                name, country, lat, lon, start_date, end_date, now_iso,
            ): (name, country)
            for name, country, lat, lon in location_specs
        }
        for future in as_completed(future_to_spec):
            name, country = future_to_spec[future]
            try:
                rows = future.result()
                all_rows.extend(rows)
            except Exception as e:
                msg = f"{country}/{name}: {e}"
                log.error("[group_w] fetch error: %s", msg)
                errors.append(msg)

    return all_rows, errors


# ─────────────────────────────────────────────────────────────────────────────
# DB upsert
# ─────────────────────────────────────────────────────────────────────────────

def _upsert_rows(con: _Conn, rows: list[dict]) -> tuple[int, int]:
    """UPSERT rows into overseas_weather.

    Args:
        con:  Open SQLite connection (WAL mode).
        rows: List of dicts matching overseas_weather schema.

    Returns:
        (inserted, skipped) where inserted = rows with changed data.

    Performance: single transaction per chunk of 2,000 rows.
    Side effects: writes overseas_weather.
    """
    if not rows:
        return 0, 0

    sql = """
        INSERT INTO overseas_weather
          (source, country, location, date,
           temp_max, temp_min, precipitation, windspeed,
           humidity_max, humidity_min, collected_at)
        VALUES
          (:source, :country, :location, :date,
           :temp_max, :temp_min, :precipitation, :windspeed,
           :humidity_max, :humidity_min, :collected_at)
        ON CONFLICT(source, country, location, date) DO UPDATE SET
          temp_max      = COALESCE(excluded.temp_max,      temp_max),
          temp_min      = COALESCE(excluded.temp_min,      temp_min),
          precipitation = COALESCE(excluded.precipitation, precipitation),
          windspeed     = COALESCE(excluded.windspeed,     windspeed),
          humidity_max  = COALESCE(excluded.humidity_max,  humidity_max),
          humidity_min  = COALESCE(excluded.humidity_min,  humidity_min),
          collected_at  = excluded.collected_at
    """
    inserted = 0
    chunk_size = 2000
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i:i + chunk_size]
        cur = con.executemany(sql, chunk)
        inserted += cur.rowcount
        con.commit()

    skipped = max(0, len(rows) - inserted)
    return inserted, skipped


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(
    backfill_days: Optional[int] = None,
    db_path: Optional[str | Path] = None,
    years_back: int = _DEFAULT_YEARS_BACK,
    skip_us: bool = False,
    skip_jp: bool = False,
    skip_eu: bool = False,
) -> dict:
    """Run overseas weather collection (Open-Meteo archive API).

    Fetches daily temperature, precipitation, wind, and humidity for
    51 US state capitals, 47 Japanese prefecture capitals, and 10 EU
    country capitals using the Open-Meteo historical weather API (free,
    no key required).

    Args:
        backfill_days: Orchestrator arg; overrides years_back if set
                       (backfill_days / 365 = years_back, min 1).
        db_path:       Path to epi_real_seoul.db.  Uses default if None.
        years_back:    Calendar years of daily data to fetch (default 3).
        skip_us:       Skip US state capitals (for testing).
        skip_jp:       Skip JP prefecture capitals (for testing).
        skip_eu:       Skip EU country capitals (for testing).

    Returns:
        dict: inserted, skipped, errors, locations_attempted.

    Raises:
        Nothing — per-location errors logged and collected in errors list.

    Performance: ~110 parallel HTTP calls, ~2-3 min total (6 workers).
    Side effects: writes overseas_weather in epi_real_seoul.db.
    Caller responsibility: DB must be accessible with WAL mode enabled.
    """
    # backfill_days overrides years_back if explicitly set
    if backfill_days is not None:
        years_back = max(1, backfill_days // 365)

    resolved = _resolve_db(db_path)
    result: dict = {
        "inserted": 0,
        "skipped":  0,
        "errors":   [],
        "locations_attempted": 0,
    }

    if not resolved.exists():
        msg = f"DB not found: {resolved}"
        log.error("[group_w] %s", msg)
        result["errors"].append(msg)
        return result

    now = datetime.now(timezone.utc)
    now_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_date   = (now - timedelta(days=2)).strftime("%Y-%m-%d")   # archive lag ~1-2 days
    start_date = (now - timedelta(days=years_back * 365)).strftime("%Y-%m-%d")

    log.info("[group_w] Date range: %s → %s (%d years)", start_date, end_date, years_back)

    # Build location list — skip locations already fully collected (up to end_date)
    # to avoid wasting hourly API quota on re-runs
    con_check = _connect(resolved)
    try:
        _ensure_table(con_check)
        already_done: set[tuple[str, str]] = {
            (r[0], r[1])
            for r in con_check.execute(
                "SELECT location, country FROM overseas_weather "
                "WHERE date = ? GROUP BY location, country",
                (end_date,),
            ).fetchall()
        }
    finally:
        con_check.close()

    all_candidates: list[tuple[str, str, float, float]] = []
    if not skip_us:
        all_candidates += [(name, "US", lat, lon) for name, _, lat, lon in _US_CAPITALS]
    if not skip_jp:
        all_candidates += [(name, "JP", lat, lon) for name, _, lat, lon in _JP_CAPITALS]
    if not skip_eu:
        all_candidates += [(name, country, lat, lon) for name, country, lat, lon in _EU_CAPITALS]

    specs = [
        (name, country, lat, lon)
        for name, country, lat, lon in all_candidates
        if (name, country) not in already_done
    ]
    skipped_locs = len(all_candidates) - len(specs)
    if skipped_locs:
        log.info("[group_w] %d locations already up-to-date (skipped)", skipped_locs)

    result["locations_attempted"] = len(specs)
    log.info("[group_w] Fetching %d locations in parallel (workers=%d)", len(specs), _MAX_WORKERS)

    all_rows, fetch_errors = _fetch_locations_parallel(specs, start_date, end_date, now_iso)
    result["errors"].extend(fetch_errors)
    log.info("[group_w] Fetched %d daily rows across %d locations", len(all_rows), len(specs))

    if not all_rows:
        log.warning("[group_w] No rows fetched — check network / Open-Meteo availability")
        return result

    con = _connect(resolved)
    try:
        _ensure_table(con)
        ins, skp = _upsert_rows(con, all_rows)
        result["inserted"] = ins
        result["skipped"]  = skp
    finally:
        con.close()

    log.info(
        "[group_w] Done: %d inserted / %d skipped / %d location errors",
        result["inserted"], result["skipped"], len(result["errors"]),
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = argparse.ArgumentParser(description="Overseas weather (Open-Meteo archive)")
    parser.add_argument("--years-back", type=int, default=_DEFAULT_YEARS_BACK,
                        help="Calendar years of daily data to fetch")
    parser.add_argument("--skip-us",    action="store_true", help="Skip US state capitals")
    parser.add_argument("--skip-jp",    action="store_true", help="Skip JP prefecture capitals")
    parser.add_argument("--skip-eu",    action="store_true", help="Skip EU country capitals")
    parser.add_argument("--db",         default=None,         help="DB path override")
    args = parser.parse_args()

    result = run(
        db_path=args.db,
        years_back=args.years_back,
        skip_us=args.skip_us,
        skip_jp=args.skip_jp,
        skip_eu=args.skip_eu,
    )
    print(json.dumps(result, indent=2))
    sys.exit(1 if result["errors"] else 0)
