"""Group G — Google Trends flu search interest (KR + overseas).

Purpose
-------
Weekly Google Trends search interest for flu-related keywords, used as
a leading indicator for ILI rate prediction.  Interest is relative (0-100
peak-scaled within each query group) and comparable week-over-week but NOT
cross-country or cross-group.

Sources
-------
  pytrends (unofficial Google Trends scraper, no API key required).
  Geo codes: KR, US, JP, FR, GB, IT, ES, NL, BE, AT, PL, RO, DE.

Keyword strategy
----------------
  KR:  3 groups × 5 Korean flu keywords (독감, 인플루엔자, …)
  US:  1 group  × 5 English keywords (flu, influenza, …)
  JP:  1 group  × 5 Japanese keywords (インフルエンザ, …)
  EU:  1 group  × 5 local-language keywords per country

Rate limiting
-------------
  30 s delay between geo transitions.
  5 s delay between keyword groups within a geo.
  60 s backoff + 1 retry on HTTP 429.
  All errors logged; failed geos do NOT abort the run.

Output table: google_search_trends
  (collected_at TEXT, period TEXT, geo TEXT, keyword TEXT,
   interest INTEGER, group_idx INTEGER)
  Unique index: (period, geo, keyword)

CLI:
  .venv/bin/python -m simulation.collectors.group_g_gtrends
  .venv/bin/python -m simulation.collectors.group_g_gtrends --years-back 3 --skip-overseas

Design (D-4 Deep Module)
------------------------
Public surface: run(backfill_days=None, db_path=None, years_back=3,
                    skip_kr=False, skip_overseas=False) → dict
All geo configs, pytrends calls, rate limiting, and DB writes encapsulated.

Gray-box contract (D-5):
  - pytrends timeframe > 9 months → weekly data (Google's auto-granularity).
  - Interest = 0-100 peak-scaled within each (geo, group_idx) query.
  - DB upsert via INSERT OR IGNORE (safe if unique index absent on old tables).
  - Returns {"inserted": N, "skipped": N, "errors": [...], "geos_attempted": [...]}.
  - Side effects: writes google_search_trends in epi_real_seoul.db only.
"""
from __future__ import annotations

import json
import logging
import time
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

# Delay between geo requests (seconds) — Google Trends rate limit avoidance
_INTER_GEO_SLEEP_S   = 30
_INTER_GROUP_SLEEP_S = 5
_RETRY_SLEEP_S       = 60  # on HTTP 429

# ─────────────────────────────────────────────────────────────────────────────
# Geo + keyword configuration
# ─────────────────────────────────────────────────────────────────────────────
# Each entry: geo_code → {hl (locale), tz (UTC offset minutes), groups (list of keyword lists)}
# Max 5 keywords per group (Google Trends hard limit per query).

_GEO_CONFIGS: dict[str, dict] = {
    # ── Korea (3 groups for richer coverage) ────────────────────────────────
    "KR": {
        "hl": "ko",
        "tz": 540,
        "groups": [
            ["독감", "인플루엔자", "감기", "발열", "기침"],
            ["타미플루", "소아과", "이비인후과", "해열제", "응급실"],
            ["콧물", "인후통", "몸살", "오한", "두통"],
        ],
    },
    # ── United States ────────────────────────────────────────────────────────
    "US": {
        "hl": "en",
        "tz": 300,
        "groups": [
            ["flu", "influenza", "flu symptoms", "fever", "tamiflu"],
        ],
    },
    # ── Japan ────────────────────────────────────────────────────────────────
    "JP": {
        "hl": "ja",
        "tz": 540,
        "groups": [
            ["インフルエンザ", "発熱", "タミフル", "風邪", "感染症"],
        ],
    },
    # ── EU-10 ────────────────────────────────────────────────────────────────
    "FR": {
        "hl": "fr",
        "tz": 60,
        "groups": [
            ["grippe", "symptômes grippe", "fièvre", "tamiflu", "rhume"],
        ],
    },
    "GB": {
        "hl": "en-GB",
        "tz": 0,
        "groups": [
            ["flu", "influenza", "flu symptoms", "fever", "flu vaccine"],
        ],
    },
    "IT": {
        "hl": "it",
        "tz": 60,
        "groups": [
            ["influenza", "febbre", "tamiflu", "sintomi influenza", "raffreddore"],
        ],
    },
    "ES": {
        "hl": "es",
        "tz": 60,
        "groups": [
            ["gripe", "influenza", "fiebre", "tamiflu", "síntomas gripe"],
        ],
    },
    "NL": {
        "hl": "nl",
        "tz": 60,
        "groups": [
            ["griep", "influenza", "koorts", "tamiflu", "griepvaccin"],
        ],
    },
    "BE": {
        "hl": "fr",
        "tz": 60,
        "groups": [
            ["grippe", "influenza", "fièvre", "tamiflu", "vaccination grippe"],
        ],
    },
    "AT": {
        "hl": "de",
        "tz": 60,
        "groups": [
            ["Grippe", "Influenza", "Fieber", "Tamiflu", "Grippesymptome"],
        ],
    },
    "PL": {
        "hl": "pl",
        "tz": 60,
        "groups": [
            ["grypa", "influenza", "gorączka", "tamiflu", "objawy grypy"],
        ],
    },
    "RO": {
        "hl": "ro",
        "tz": 120,
        "groups": [
            ["gripa", "influenza", "febra", "tamiflu", "simptome gripa"],
        ],
    },
    "DE": {
        "hl": "de",
        "tz": 60,
        "groups": [
            ["Grippe", "Influenza", "Fieber", "Tamiflu", "Erkältung"],
        ],
    },
}

_KR_GEOS       = frozenset({"KR"})
_OVERSEAS_GEOS = frozenset({
    "US", "JP", "FR", "GB", "IT", "ES", "NL", "BE", "AT", "PL", "RO", "DE"
})


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


def _connect(db_path: Path) -> _Conn:
    safe_connect = _safe_connect_import()
    return safe_connect(str(db_path), timeout=60.0)


def _ensure_gtrends_table(con: _Conn) -> None:
    """Create google_search_trends if absent; add unique index for upsert safety."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS google_search_trends (
            collected_at  TEXT,
            period        TEXT NOT NULL,
            geo           TEXT NOT NULL,
            keyword       TEXT NOT NULL,
            interest      INTEGER,
            group_idx     INTEGER
        )
    """)
    # Unique index — silently skip if legacy data has duplicates
    try:
        con.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_gst_period_geo_kw
            ON google_search_trends(period, geo, keyword)
        """)
    except Exception as e:
        log.warning("[group_g] unique index skipped (legacy duplicates?): %s", e)
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_gst_geo_period "
        "ON google_search_trends(geo, period)"
    )
    con.commit()


# ─────────────────────────────────────────────────────────────────────────────
# pytrends fetcher
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_geo(
    geo: str,
    config: dict,
    timeframe: str,
    now_iso: str,
) -> tuple[list[dict], list[str]]:
    """Fetch all keyword groups for one geo from Google Trends.

    Args:
        geo:       Google Trends geo code (e.g. 'KR', 'US', 'JP').
        config:    Dict with keys hl (locale str), tz (int UTC offset min),
                   groups (list of list[str], max 5 keywords each).
        timeframe: pytrends timeframe string ("YYYY-MM-DD YYYY-MM-DD").
        now_iso:   Collection timestamp (UTC ISO-8601 string).

    Returns:
        (rows, errors) where rows is a list of dicts with google_search_trends
        schema and errors is a list of error strings for failed groups.

    Raises:
        Nothing — per-group errors are caught and returned in errors list.

    Performance: N_groups × 1 HTTP request.  Sleeps _INTER_GROUP_SLEEP_S
                 between groups to avoid 429.
    Side effects: None (pure fetch + return).
    Caller responsibility: caller upserts rows into DB.
    """
    try:
        from pytrends.request import TrendReq
    except ImportError as exc:
        return [], [f"{geo}: pytrends not installed — pip install pytrends ({exc})"]

    rows: list[dict] = []
    errors: list[str] = []

    hl = config.get("hl", "en")
    tz = config.get("tz", 0)
    groups: list[list[str]] = config.get("groups", [])

    try:
        pytrends = TrendReq(hl=hl, tz=tz, timeout=(15, 45))
    except Exception as e:
        return [], [f"{geo}: TrendReq init failed: {e}"]

    for gi, keywords in enumerate(groups):
        log.info("[group_g] %s group %d/%d: %s", geo, gi + 1, len(groups), keywords)

        for attempt in range(2):
            try:
                pytrends.build_payload(
                    keywords,
                    cat=0,
                    timeframe=timeframe,
                    geo=geo,
                )
                df = pytrends.interest_over_time()

                if df is None or df.empty:
                    log.warning("[group_g] %s group %d: empty response", geo, gi + 1)
                    break

                if "isPartial" in df.columns:
                    df = df.drop(columns=["isPartial"])

                for date_idx, row in df.iterrows():
                    period = date_idx.strftime("%Y-%m-%d")
                    for kw in keywords:
                        if kw in row:
                            rows.append({
                                "collected_at": now_iso,
                                "period":       period,
                                "geo":          geo,
                                "keyword":      kw,
                                "interest":     int(row[kw]),
                                "group_idx":    gi,
                            })
                break  # success — exit retry loop

            except Exception as e:
                err_str = str(e)
                if "429" in err_str and attempt == 0:
                    log.warning(
                        "[group_g] %s group %d: HTTP 429 — sleeping %ds then retrying",
                        geo, gi + 1, _RETRY_SLEEP_S,
                    )
                    time.sleep(_RETRY_SLEEP_S)
                    continue
                msg = f"{geo} group {gi}: {err_str[:120]}"
                log.error("[group_g] %s", msg)
                errors.append(msg)
                break

        if gi < len(groups) - 1:
            time.sleep(_INTER_GROUP_SLEEP_S)

    log.info(
        "[group_g] %s: %d rows from %d groups (%d keywords)",
        geo, len(rows), len(groups), sum(len(g) for g in groups),
    )
    return rows, errors


# ─────────────────────────────────────────────────────────────────────────────
# DB upsert
# ─────────────────────────────────────────────────────────────────────────────

def _upsert_rows(con: _Conn, rows: list[dict]) -> tuple[int, int]:
    """UPSERT rows into google_search_trends.

    Uses INSERT OR IGNORE for backward compatibility with legacy tables that may
    lack the unique index.  If the unique index IS present, this correctly skips
    duplicates.  interest is NOT updated on conflict (existing data preserved).

    Args:
        con:  Open SQLite connection (WAL mode).
        rows: List of dicts with google_search_trends schema keys.

    Returns:
        (inserted, skipped) counts.

    Performance: single transaction per chunk of 1,000 rows.
    Side effects: writes google_search_trends.
    """
    if not rows:
        return 0, 0

    sql = """
        INSERT OR IGNORE INTO google_search_trends
          (collected_at, period, geo, keyword, interest, group_idx)
        VALUES
          (:collected_at, :period, :geo, :keyword, :interest, :group_idx)
    """
    inserted = 0
    chunk_size = 1000
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
    skip_kr: bool = False,
    skip_overseas: bool = False,
) -> dict:
    """Run Google Trends flu search interest collection for KR + overseas geos.

    Fetches weekly search interest for flu-related keywords across Korea (3
    groups) and 12 overseas geos (US, JP, FR, GB, IT, ES, NL, BE, AT, PL,
    RO, DE — 1 group each).  Writes to google_search_trends in the project DB.

    Args:
        backfill_days: Orchestrator arg; if set, overrides years_back
                       (backfill_days / 365, min 1).
        db_path:       Path to epi_real_seoul.db.  Uses default if None.
        years_back:    Calendar years of weekly data to fetch (default 3).
        skip_kr:       Skip Korea geo (for testing).
        skip_overseas: Skip all 12 overseas geos (for testing).

    Returns:
        dict with keys: inserted, skipped, errors, geos_attempted.

    Raises:
        Nothing — per-geo errors are caught and collected in errors list.

    Performance: up to 13 geos × (1-3 groups each) × 1 HTTP req + inter-geo
                 sleeps (~10-15 min for a full run). pytrends is sequential
                 (Google rate-limits concurrent scraping).
    Side effects: writes google_search_trends in epi_real_seoul.db.
    Caller responsibility: pytrends must be installed (pip install pytrends).
                           DB must be accessible with WAL mode enabled.
    """
    if backfill_days is not None:
        years_back = max(1, backfill_days // 365)

    resolved = _resolve_db(db_path)
    result: dict = {
        "inserted":       0,
        "skipped":        0,
        "errors":         [],
        "geos_attempted": [],
    }

    if not resolved.exists():
        msg = f"DB not found: {resolved}"
        log.error("[group_g] %s", msg)
        result["errors"].append(msg)
        return result

    now = datetime.now(timezone.utc)
    now_iso  = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_dt   = now.strftime("%Y-%m-%d")
    start_dt = (now - timedelta(days=years_back * 365)).strftime("%Y-%m-%d")
    timeframe = f"{start_dt} {end_dt}"
    log.info("[group_g] timeframe: %s", timeframe)

    # Determine which geos to run
    geos_to_run: list[str] = []
    if not skip_kr:
        geos_to_run.extend(g for g in _GEO_CONFIGS if g in _KR_GEOS)
    if not skip_overseas:
        geos_to_run.extend(g for g in _GEO_CONFIGS if g in _OVERSEAS_GEOS)

    if not geos_to_run:
        log.warning("[group_g] All geos skipped — nothing to do")
        return result

    log.info("[group_g] Running %d geos: %s", len(geos_to_run), geos_to_run)

    con = _connect(resolved)
    try:
        _ensure_gtrends_table(con)

        for idx, geo in enumerate(geos_to_run):
            config = _GEO_CONFIGS[geo]
            result["geos_attempted"].append(geo)

            rows, fetch_errors = _fetch_geo(geo, config, timeframe, now_iso)
            result["errors"].extend(fetch_errors)

            if rows:
                ins, skp = _upsert_rows(con, rows)
                result["inserted"] += ins
                result["skipped"]  += skp
                log.info("[group_g] %s: inserted=%d skipped=%d", geo, ins, skp)
            else:
                log.warning("[group_g] %s: 0 rows returned", geo)

            # Sleep between geos to avoid rate limiting (skip after last geo)
            if idx < len(geos_to_run) - 1:
                log.info("[group_g] Sleeping %ds before next geo…", _INTER_GEO_SLEEP_S)
                time.sleep(_INTER_GEO_SLEEP_S)

    finally:
        con.close()

    log.info(
        "[group_g] Done: %d geos | %d inserted | %d skipped | %d errors",
        len(result["geos_attempted"]),
        result["inserted"],
        result["skipped"],
        len(result["errors"]),
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
    parser = argparse.ArgumentParser(
        description="Google Trends flu search interest (KR + overseas)"
    )
    parser.add_argument("--years-back",     type=int, default=_DEFAULT_YEARS_BACK,
                        help="Calendar years of weekly data to fetch (default 3)")
    parser.add_argument("--skip-kr",        action="store_true",
                        help="Skip Korea geo")
    parser.add_argument("--skip-overseas",  action="store_true",
                        help="Skip all 12 overseas geos")
    parser.add_argument("--db",             default=None,
                        help="DB path override")
    args = parser.parse_args()

    result = run(
        db_path=args.db,
        years_back=args.years_back,
        skip_kr=args.skip_kr,
        skip_overseas=args.skip_overseas,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(1 if result["errors"] else 0)
