"""Germany Bright Sky (DWD wrapper) weather — 16 Bundesland cities.

Sprint β Item 4 full migration (Codex analysis 2026-05-26): real bodies
moved here from group_i_overseas.py. The legacy module re-exports for
back-compat.

Shared weather DDL/upsert with overseas/openmeteo.py — both write to
overseas_weather_regional via `_common._upsert_weather_rows`.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone, date as _date
from typing import Optional

from simulation.collectors._endpoints import _BRIGHTSKY_URL
from simulation.collectors.overseas._common import (
    _connect,
    _ensure_overseas_weather_regional_table,
    _resolve_db,
    _retry_get,
    _safe_float,
    _upsert_weather_rows,
)

log = logging.getLogger(__name__)


# (region_label, lat, lon) — 16 Bundesländer representatives
_DE_BUNDESLAND_CITIES: list[tuple[str, float, float]] = [
    ("Munich",       48.14,  11.58),   # Bayern
    ("Stuttgart",    48.78,   9.18),   # Baden-Württemberg
    ("Cologne",      50.94,   6.96),   # NRW
    ("Frankfurt",    50.11,   8.68),   # Hessen
    ("Dresden",      51.05,  13.74),   # Sachsen
    ("Potsdam",      52.40,  13.06),   # Brandenburg
    ("Erfurt",       50.98,  11.03),   # Thüringen
    ("Magdeburg",    52.13,  11.62),   # Sachsen-Anhalt
    ("Hannover",     52.37,   9.73),   # Niedersachsen
    ("Hamburg",      53.55,  10.00),   # Hamburg
    ("Berlin",       52.52,  13.40),   # Berlin
    ("Bremen",       53.08,   8.81),   # Bremen
    ("Rostock",      54.09,  12.14),   # Mecklenburg-Vorpommern
    ("Mainz",        49.99,   8.27),   # Rheinland-Pfalz
    ("Saarbrücken",  49.23,   6.99),   # Saarland
    ("Kiel",         54.32,  10.13),   # Schleswig-Holstein
]

_BRIGHTSKY_WINDOW_DAYS = 28  # ≤31 per Bright Sky docs


def _fetch_brightsky_window(
    region: str, lat: float, lon: float,
    window_start: str, window_end: str, now_iso: str,
) -> list[dict]:
    """Fetch one 28-day weather window from Bright Sky for a single DE city.

    Args:
        region:       City/region name (e.g. 'Berlin').
        lat:          Latitude (decimal degrees, WGS84).
        lon:          Longitude (decimal degrees, WGS84).
        window_start: Start date string 'YYYY-MM-DD'.
        window_end:   End date string 'YYYY-MM-DD' (inclusive, ≤28 days from start).
        now_iso:      ISO timestamp for collected_at.

    Returns:
        List of daily aggregated row dicts for overseas_weather_regional.
        Daily aggregation: mean temperature, sum precipitation, max wind, mean humidity.

    Performance: 1 HTTP request, ~50 KB JSON per window.
    Side effects: None.
    Caller responsibility: rate-limit sleep handled by caller.
    """
    params = {
        "lat": lat, "lon": lon,
        "date":      window_start,
        "last_date": window_end,
    }
    try:
        resp = _retry_get(_BRIGHTSKY_URL, params=params, sleep_s=2.0, timeout=45)
        data = resp.json()
    except Exception as e:
        log.debug("[overseas.brightsky] %s %s: %s", region, window_start, e)
        return []

    hourly = data.get("weather") or []
    if not hourly:
        return []

    # Aggregate hourly → daily (group by date prefix)
    from collections import defaultdict
    day_buckets: dict[str, list[dict]] = defaultdict(list)
    for obs in hourly:
        ts = (obs.get("timestamp") or "")[:10]
        if ts:
            day_buckets[ts].append(obs)

    rows: list[dict] = []
    for date_str, obs_list in sorted(day_buckets.items()):
        temps    = [_safe_float(o.get("temperature")) for o in obs_list]
        precips  = [_safe_float(o.get("precipitation")) for o in obs_list]
        winds    = [_safe_float(o.get("wind_speed")) for o in obs_list]
        humids   = [_safe_float(o.get("relative_humidity")) for o in obs_list]

        def _fmean(vals: list) -> Optional[float]:
            v = [x for x in vals if x is not None]
            return round(sum(v) / len(v), 2) if v else None

        def _fmax(vals: list) -> Optional[float]:
            v = [x for x in vals if x is not None]
            return max(v) if v else None

        def _fsum(vals: list) -> Optional[float]:
            v = [x for x in vals if x is not None]
            return round(sum(v), 2) if v else None

        rows.append({
            "country":       "DE",
            "region":        region,
            "date":          date_str,
            "lat":           lat,
            "lon":           lon,
            "temp_max":      _fmax(temps),
            "temp_min":      min((x for x in temps if x is not None), default=None),
            "precip":        _fsum(precips),
            "wind_max":      _fmax(winds),
            "humidity_mean": _fmean(humids),
            "collected_at":  now_iso,
        })
    return rows


def collect_brightsky_germany(
    db_path: str = "simulation/data/db/epi_real_seoul.db",
    start_date: str = "2010-01-01",
    end_date: str | None = None,
) -> dict:
    """Collect DWD/Bright Sky daily weather for 16 German Bundesland cities.

    Writes to overseas_weather_regional (same table as Open-Meteo collector).
    Uses 28-day windows to respect Bright Sky API limits.  Sleeps 0.5 s between calls.

    Args:
        db_path:    Path to epi_real_seoul.db.
        start_date: Earliest date (YYYY-MM-DD).
        end_date:   Latest date (YYYY-MM-DD).  Defaults to today.

    Returns:
        dict: rows_inserted (int), errors (list[str]), source (str).

    Performance: ~16 cities × ~16 years × ~13 windows/year ≈ ~3,300 HTTP calls (~28 min).
                 Reduce start_date for backfill vs. incremental runs.
    Side effects: writes overseas_weather_regional.
    Caller responsibility: DB file must exist.
    """
    result: dict = {"rows_inserted": 0, "errors": [], "source": _BRIGHTSKY_URL}
    try:
        s_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        s_date = _date(2010, 1, 1)

    today = datetime.now().date()
    if end_date:
        try:
            e_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            e_date = today
    else:
        e_date = today

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
    for region, lat, lon in _DE_BUNDESLAND_CITIES:
        cursor = s_date
        while cursor <= e_date:
            window_end = min(cursor + timedelta(days=_BRIGHTSKY_WINDOW_DAYS - 1), e_date)
            rows = _fetch_brightsky_window(
                region, lat, lon,
                cursor.strftime("%Y-%m-%d"),
                window_end.strftime("%Y-%m-%d"),
                now_iso,
            )
            if rows:
                try:
                    ins, _ = _upsert_weather_rows(con, rows)
                    total_ins += ins
                except Exception as e:
                    result["errors"].append(f"upsert DE/{region}/{cursor}: {e}")
            cursor = window_end + timedelta(days=1)
            time.sleep(0.5)  # respect Bright Sky rate limit

    result["rows_inserted"] = total_ins
    log.info("[overseas.brightsky] collect_brightsky_germany: %d inserted", total_ins)
    con.close()
    return result


__all__ = [
    "_DE_BUNDESLAND_CITIES", "_BRIGHTSKY_WINDOW_DAYS",
    "_fetch_brightsky_window", "collect_brightsky_germany",
]
