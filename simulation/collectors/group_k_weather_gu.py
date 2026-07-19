"""Group K — Multi-station KMA ASOS weather for Seoul GU-level coverage.

Problem: Group B collects only stn_id=108 (서울 ASOS, Jung-gu area).  Seoul's
25 GUs span ~600 km²; temperature, humidity, and wind can vary ±2°C north-to-
south.  ILI transmission models benefit from GU-weighted weather features.

Solution: Fetch 6 additional KMA ASOS stations that bracket Seoul's geographic
extent, then store all observations in weather_historical (same schema, more
stn_id values).  A static GU↔station proximity map allows downstream feature
engineering to weight observations by nearest station.

Target table: weather_historical (existing schema)
  (obs_date TEXT, stn_id INT, stn_nm TEXT, ta_avg REAL, ta_max REAL,
   ta_min REAL, hm_avg REAL, ... 30+ columns)
  UNIQUE constraint: (obs_date, stn_id)

Stations covered (KMA ASOS, public API)
----------------------------------------
  108  서울    (Jung-gu center — Group B base)
  401  양평    (SE, proxy for Gangdong/Songpa/Gangnam)
  119  수원    (SW, proxy for Geumcheon/Gwanak)
  400  강화    (NW, proxy for Eunpyeong/Gangseo)
  98   동두천  (N, proxy for Dobong/Nowon/Gangbuk)
  203  인제    (NE, proxy for Jungnang/Dongdaemun)

GU proximity map (static, used by feature engineering only)
-----------------------------------------------------------
Groups are approximate — GUs share the weather from their closest station.
See GU_STATION_MAP for the explicit assignment.

CLI:
  .venv/bin/python -m simulation.collectors.group_k_weather_gu
  .venv/bin/python -m simulation.collectors.group_k_weather_gu --days-back 30
"""
from __future__ import annotations

import json
import logging
import os
from sqlite3 import Connection as _Conn
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# Use safe_connect (WAL + quick_check + corruption guard) — G-116/G-117.
def _safe_connect_import():
    from simulation.database import safe_connect
    return safe_connect

log = logging.getLogger(__name__)

_DEFAULT_DB = Path("simulation/data/db/epi_real_seoul.db")

# KMA ASOS station IDs (stn_id → station name) for Seoul spatial coverage.
# Station 108 is already collected by Group B; K skips it to avoid duplicate
# writes unless Group B hasn't run yet.
STATIONS: dict[int, str] = {
    108:  "서울",      # Group B base — included for completeness/backfill only
    401:  "양평",      # SE Seoul proxy
    119:  "수원",      # SW Seoul proxy
    400:  "강화",      # NW Seoul proxy
    98:   "동두천",    # N Seoul proxy
    203:  "인제",      # NE reference
}

# Static GU → closest ASOS station mapping (approx., for downstream use)
GU_STATION_MAP: dict[str, int] = {
    # Center / North-center
    "종로구": 108, "중구": 108, "성북구": 108, "서대문구": 108,
    "용산구": 108, "마포구": 108,
    # North / NW
    "강북구": 98,  "도봉구": 98,  "은평구": 400, "노원구": 98,
    # NE
    "중랑구": 203, "동대문구": 108,
    # East
    "성동구": 108, "광진구": 401,
    # SE
    "강동구": 401, "송파구": 401, "강남구": 401, "서초구": 401,
    # SW
    "관악구": 119, "동작구": 119, "금천구": 119,
    # West
    "구로구": 400, "양천구": 400, "영등포구": 108, "강서구": 400,
}

_API_KEY_ENV = "KMA_API_KEY"
_ASOS_URL = "https://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
_ASOS_DAILY_URL = "https://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"
_TIMEOUT_S = 60


def _resolve_db(db_path: Optional[str | Path]) -> Path:
    if db_path is not None:
        return Path(db_path)
    cwd = Path.cwd()
    for p in [_DEFAULT_DB, cwd / _DEFAULT_DB, cwd.parent / _DEFAULT_DB]:
        if p.exists():
            return p
    return _DEFAULT_DB


def _connect(db_path: Path) -> _Conn:
    safe_connect = _safe_connect_import()
    return safe_connect(str(db_path), timeout=60.0)


def _existing_dates(con: _Conn, stn_id: int, start_dt: str) -> set[str]:
    """Return set of obs_date already stored for this station."""
    cur = con.execute(
        'SELECT obs_date FROM weather_historical WHERE stn_id=? AND obs_date>=?',
        (stn_id, start_dt),
    )
    return {r[0] for r in cur.fetchall()}


def _fetch_asos_daily(
    stn_id: int,
    start_dt: str,
    end_dt: str,
    api_key: str,
) -> list[dict]:
    """Fetch KMA ASOS daily observations for one station via Open API.

    Args:
        stn_id:   KMA station ID (e.g. 108).
        start_dt: Start date string 'YYYYMMDD'.
        end_dt:   End date string 'YYYYMMDD'.
        api_key:  KMA Open API service key.

    Returns:
        List of row dicts matching weather_historical columns.

    Raises:
        RuntimeError: on HTTP error or missing API key.

    Performance: 1 HTTP request per station/date-range (~0.5 s).
    Side effects: None.
    """
    try:
        from simulation.utils.http import http_get  # SSOT retry-session
    except ImportError as e:
        raise RuntimeError("requests not installed") from e

    params = {
        "serviceKey": api_key,
        "pageNo":     "1",
        "numOfRows":  "999",
        "dataType":   "JSON",
        "dataCd":     "ASOS",
        "dateCd":     "DAY",
        "startDt":    start_dt,
        "endDt":      end_dt,
        "stnIds":     str(stn_id),
    }
    resp = http_get(_ASOS_DAILY_URL, params=params, timeout=_TIMEOUT_S)
    resp.raise_for_status()

    data = resp.json()
    items = (data.get("response", {})
                 .get("body", {})
                 .get("items", {})
                 .get("item", []))
    if isinstance(items, dict):
        items = [items]

    rows = []
    for it in items:
        obs_date = str(it.get("tm", ""))[:8]
        if not obs_date or len(obs_date) != 8:
            continue

        def _f(k):
            v = it.get(k)
            try:
                return float(v) if v not in (None, "", "-", " ") else None
            except (TypeError, ValueError):
                return None

        rows.append({
            "obs_date":  obs_date,
            "stn_id":    int(it.get("stn", stn_id)),
            "stn_nm":    STATIONS.get(stn_id, str(stn_id)),
            "ta_avg":    _f("ta"),
            "ta_max":    _f("taMax"),
            "ta_min":    _f("taMin"),
            "hm_avg":    _f("hm"),
            "hm_max":    None,
            "hm_min":    None,
            "ws_avg":    _f("ws"),
            "ws_max":    _f("wsMax"),
            "wd_avg":    _f("wd"),
            "ps_avg":    _f("ps"),
            "ps_sea":    None,
            "ss_sum":    _f("ss"),
            "ca_tot":    _f("caTot"),
            "rn_day":    _f("rn"),
            "sd_max":    _f("sdMax"),
            "sd_new":    None,
            "dz_tot":    None,
            "fg":        None,
            "ts":        None,
        })
    return rows


def _upsert_weather(con: _Conn, rows: list[dict]) -> tuple[int, int]:
    """UPSERT rows into weather_historical (obs_date, stn_id) PK.

    Returns:
        (inserted, skipped) counts.

    Performance: single transaction, chunk_size=200.
    Side effects: writes to weather_historical.
    """
    if not rows:
        return 0, 0

    # Build INSERT OR IGNORE to avoid overwriting Group B's data for stn 108
    sql = """
        INSERT OR IGNORE INTO weather_historical
          (obs_date, stn_id, stn_nm, ta_avg, ta_max, ta_min,
           hm_avg, ws_avg, ws_max, wd_avg, ps_avg, ss_sum,
           ca_tot, rn_day, sd_max)
        VALUES
          (:obs_date, :stn_id, :stn_nm, :ta_avg, :ta_max, :ta_min,
           :hm_avg, :ws_avg, :ws_max, :wd_avg, :ps_avg, :ss_sum,
           :ca_tot, :rn_day, :sd_max)
    """
    inserted = 0
    for i in range(0, len(rows), 200):
        chunk = rows[i:i + 200]
        cur = con.executemany(sql, chunk)
        inserted += cur.rowcount
        con.commit()
    return inserted, len(rows) - inserted


def _save_gu_map(con: _Conn) -> None:
    """Persist GU→station map to a lookup table for downstream use."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS weather_gu_station_map (
            gu_nm   TEXT PRIMARY KEY,
            stn_id  INTEGER NOT NULL,
            stn_nm  TEXT
        )
    """)
    con.executemany(
        "INSERT OR REPLACE INTO weather_gu_station_map (gu_nm, stn_id, stn_nm) "
        "VALUES (?, ?, ?)",
        [(gu, stn, STATIONS.get(stn, str(stn))) for gu, stn in GU_STATION_MAP.items()],
    )
    con.commit()


def run(
    backfill_days: Optional[int] = None,
    db_path: Optional[str | Path] = None,
    days_back: int = 14,
    skip_108: bool = True,
) -> dict:
    """Fetch multi-station ASOS weather and upsert into weather_historical.

    Args:
        backfill_days: Orchestrator override (maps to days_back if provided).
        db_path:       Path to epi_real_seoul.db.
        days_back:     Number of past calendar days to fetch (default 14).
                       Group B already covers stn 108; K focuses on other stations.
        skip_108:      If True (default), skip station 108 (already in Group B).

    Returns:
        dict: {inserted, skipped, errors, stations_fetched}

    Raises:
        Nothing — errors are caught and returned.

    Performance: 5 HTTP requests (1 per additional station), ~5 s total.
    Side effects: writes weather_historical; creates weather_gu_station_map.
    Caller responsibility: KMA_API_KEY env var must be set.
    """
    if backfill_days is not None and backfill_days > 0:
        days_back = backfill_days

    api_key = os.environ.get(_API_KEY_ENV, "")
    if not api_key:
        msg = (f"KMA_API_KEY not set — Group K skipped. "
               f"Set env var {_API_KEY_ENV} to enable multi-station weather.")
        log.warning("[group_k] %s", msg)
        return {"inserted": 0, "skipped": 0, "errors": [msg], "stations_fetched": []}

    resolved = _resolve_db(db_path)
    result: dict = {"inserted": 0, "skipped": 0, "errors": [], "stations_fetched": []}

    if not resolved.exists():
        msg = f"DB not found: {resolved}"
        log.error("[group_k] %s", msg)
        result["errors"].append(msg)
        return result

    today = datetime.now()
    start = today - timedelta(days=days_back)
    start_dt = start.strftime("%Y%m%d")
    end_dt = today.strftime("%Y%m%d")

    con = _connect(resolved)
    try:
        _save_gu_map(con)

        for stn_id, stn_nm in STATIONS.items():
            if skip_108 and stn_id == 108:
                continue
            try:
                existing = _existing_dates(con, stn_id, start_dt)
                rows = _fetch_asos_daily(stn_id, start_dt, end_dt, api_key)
                # Filter rows already stored
                new_rows = [r for r in rows if r["obs_date"] not in existing]
                ins, skp = _upsert_weather(con, new_rows)
                result["inserted"] += ins
                result["skipped"]  += skp + (len(rows) - len(new_rows))
                result["stations_fetched"].append(stn_id)
                log.info("[group_k] stn=%d %s: %d rows → %d inserted",
                         stn_id, stn_nm, len(rows), ins)
                time.sleep(0.3)  # KMA rate limit (3 req/s)
            except Exception as e:
                log.error("[group_k] stn=%d error: %s", stn_id, e)
                result["errors"].append(f"stn_{stn_id}: {e}")
    finally:
        con.close()

    log.info("[group_k] Done: inserted=%d skipped=%d errors=%d",
             result["inserted"], result["skipped"], len(result["errors"]))
    return result


if __name__ == "__main__":
    import argparse, sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Multi-station KMA weather (GU-level)")
    parser.add_argument("--days-back", type=int, default=14)
    parser.add_argument("--include-108", action="store_true", help="Also fetch stn 108")
    parser.add_argument("--db", default=None)
    args = parser.parse_args()
    r = run(db_path=args.db, days_back=args.days_back, skip_108=not args.include_108)
    print(json.dumps(r, indent=2))
    sys.exit(1 if r["errors"] else 0)
