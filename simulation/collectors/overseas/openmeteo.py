"""Open-Meteo historical weather (ERA5 reanalysis) — 33 cities × 6 countries.

Sprint β Item 4 full migration (Codex analysis 2026-05-26): real bodies moved
here from group_i_overseas.py. The legacy module re-exports for back-compat.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from simulation.collectors._endpoints import _OPENMETEO_ARCHIVE_URL
from simulation.collectors.overseas._common import (
    _connect,
    _ensure_overseas_weather_regional_table,
    _resolve_db,
    _retry_get,
    _safe_float,
    _upsert_weather_rows,
)

log = logging.getLogger(__name__)


# (country_iso2, region_label, lat, lon)
_OPENMETEO_LOCATIONS: list[tuple[str, str, float, float]] = [
    # KOR
    ("KR", "Seoul",   37.57, 126.98),
    ("KR", "Busan",   35.10, 129.04),
    ("KR", "Daegu",   35.87, 128.60),
    ("KR", "Incheon", 37.46, 126.65),
    ("KR", "Gwangju", 35.16, 126.85),
    ("KR", "Daejeon", 36.35, 127.38),
    ("KR", "Ulsan",   35.54, 129.31),
    ("KR", "Jeju",    33.51, 126.53),
    # JPN
    ("JP", "Tokyo",    35.69, 139.69),
    ("JP", "Osaka",    34.69, 135.50),
    ("JP", "Nagoya",   35.18, 136.91),
    ("JP", "Hokkaido", 43.06, 141.35),
    ("JP", "Fukuoka",  33.59, 130.42),
    # DEU — handled by Bright Sky too, but Open-Meteo provides ERA5 continuity
    ("DE", "Berlin",    52.52,  13.40),
    ("DE", "Hamburg",   53.55,  10.00),
    ("DE", "Munich",    48.14,  11.58),
    ("DE", "Frankfurt", 50.11,   8.68),
    ("DE", "Stuttgart", 48.78,   9.18),
    # FRA
    ("FR", "Paris",     48.85,   2.35),
    ("FR", "Lyon",      45.75,   4.85),
    ("FR", "Marseille", 43.30,   5.37),
    ("FR", "Bordeaux",  44.84,  -0.58),
    ("FR", "Lille",     50.63,   3.07),
    # AUS
    ("AU", "Sydney",    -33.87, 151.21),
    ("AU", "Melbourne", -37.81, 144.96),
    ("AU", "Brisbane",  -27.47, 153.03),
    ("AU", "Perth",     -31.95, 115.86),
    ("AU", "Adelaide",  -34.93, 138.60),
    # CHN
    ("CN", "Beijing",   39.91, 116.39),
    ("CN", "Shanghai",  31.23, 121.47),
    ("CN", "Guangzhou", 23.13, 113.26),
    ("CN", "Chengdu",   30.57, 104.07),
    ("CN", "Wuhan",     30.59, 114.30),
]

_OPENMETEO_DAILY_VARS = (
    "temperature_2m_max,temperature_2m_min,precipitation_sum,"
    "wind_speed_10m_max,relative_humidity_2m_mean"
)


def _fetch_openmeteo_one_year(
    country: str, region: str, lat: float, lon: float,
    year: int, now_iso: str,
) -> list[dict]:
    """Fetch one calendar-year of daily weather for a single location via Open-Meteo.

    Args:
        country:  ISO2 country code (e.g. 'KR').
        region:   Human-readable city/region name.
        lat:      Latitude (decimal degrees, WGS84).
        lon:      Longitude (decimal degrees, WGS84).
        year:     Calendar year to fetch.
        now_iso:  ISO timestamp for collected_at.

    Returns:
        List of daily row dicts for overseas_weather_regional.
        Empty list if the API returns no data or an error occurs.

    Raises:
        Nothing — errors are logged and an empty list is returned.

    Performance: 1 HTTP request per call; Open-Meteo ERA5 ~200 ms.
    Side effects: None.
    Caller responsibility: rate-limit sleep handled by caller.
    """
    import requests  # noqa: F401

    today = datetime.now().date()
    start = f"{year}-01-01"
    end   = min(today, datetime(year, 12, 31).date()).strftime("%Y-%m-%d")
    if start > str(today):
        return []

    params = {
        "latitude":  lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "daily":     _OPENMETEO_DAILY_VARS,
        "timezone":  "UTC",
    }
    try:
        resp = _retry_get(_OPENMETEO_ARCHIVE_URL, params=params, sleep_s=3.0, timeout=60)
        data = resp.json()
    except Exception as e:
        log.warning("[overseas.openmeteo] %s/%s %d: %s", country, region, year, e)
        return []

    daily = data.get("daily") or {}
    dates = daily.get("time") or []
    t_max  = daily.get("temperature_2m_max") or []
    t_min  = daily.get("temperature_2m_min") or []
    precip = daily.get("precipitation_sum") or []
    wind   = daily.get("wind_speed_10m_max") or []
    humid  = daily.get("relative_humidity_2m_mean") or []

    def _pad(lst: list, n: int) -> list:
        return lst + [None] * max(0, n - len(lst))

    n = len(dates)
    t_max  = _pad(t_max,  n); t_min  = _pad(t_min,  n)
    precip = _pad(precip, n); wind   = _pad(wind,   n); humid = _pad(humid, n)

    rows: list[dict] = []
    for i, date in enumerate(dates):
        rows.append({
            "country": country, "region": region, "date": date,
            "lat": lat, "lon": lon,
            "temp_max":      _safe_float(t_max[i]),
            "temp_min":      _safe_float(t_min[i]),
            "precip":        _safe_float(precip[i]),
            "wind_max":      _safe_float(wind[i]),
            "humidity_mean": _safe_float(humid[i]),
            "collected_at":  now_iso,
        })
    return rows


def collect_openmeteo_regional(
    db_path: str = "simulation/data/db/epi_real_seoul.db",
    start_date: str = "2010-01-01",
    end_date: str | None = None,
) -> dict:
    """Collect Open-Meteo ERA5 daily weather for 33 cities across 6 countries.

    Writes one row per (country, region, date) to overseas_weather_regional.
    Batches by calendar year; sleeps 0.3 s between API calls to respect rate limits.

    Args:
        db_path:    Path to epi_real_seoul.db.
        start_date: Earliest date (YYYY-MM-DD).  Year-component is used.
        end_date:   Latest date (YYYY-MM-DD). Defaults to today.

    Returns:
        dict: rows_inserted (int), errors (list[str]), source (str).

    Performance: ~33 locations × ~16 years × 1 request = ~528 HTTP calls (~3 min).
    Side effects: writes overseas_weather_regional.
    Caller responsibility: DB file must exist; caller may pass a subset via subclassing.
    """
    result: dict = {"rows_inserted": 0, "errors": [],
                    "source": _OPENMETEO_ARCHIVE_URL}
    try:
        start_year = int(start_date[:4])
    except (ValueError, TypeError):
        start_year = 2010
    end_year = int((end_date or datetime.now().strftime("%Y-%m-%d"))[:4])
    now_iso  = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    resolved = _resolve_db(db_path)
    con = _connect(resolved)
    try:
        _ensure_overseas_weather_regional_table(con)
    except Exception as e:
        result["errors"].append(f"table setup: {e}")
        con.close()
        return result

    total_ins = 0
    for country, region, lat, lon in _OPENMETEO_LOCATIONS:
        for year in range(start_year, end_year + 1):
            rows = _fetch_openmeteo_one_year(country, region, lat, lon, year, now_iso)
            if rows:
                try:
                    ins, _ = _upsert_weather_rows(con, rows)
                    total_ins += ins
                except Exception as e:
                    result["errors"].append(f"upsert {country}/{region}/{year}: {e}")
            time.sleep(0.3)  # respect Open-Meteo rate limit

    result["rows_inserted"] = total_ins
    log.info("[overseas.openmeteo] collect_openmeteo_regional: %d inserted", total_ins)
    con.close()
    return result


__all__ = [
    "_OPENMETEO_LOCATIONS", "_OPENMETEO_DAILY_VARS",
    "_fetch_openmeteo_one_year", "collect_openmeteo_regional",
]
