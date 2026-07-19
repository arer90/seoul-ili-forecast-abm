"""Shared infrastructure for overseas collectors.

Sprint β Item 4 full migration (Codex analysis 2026-05-26):
12 helpers (DB connect, sqlite upsert, retry, table DDL) used by ≥2 source
modules. Per ENGINEERING_PRINCIPLES.md D-4: small interface (each fn is single-purpose) +
rich implementation (WAL, retry, schema migration logic).

Used by:
- who, cdc, jihs, ecdc, influnet (national ILI → overseas_ili via _upsert_rows)
- sentiweb (regional ILI → overseas_ili_regional via _upsert_regional_ili_rows)
- openmeteo, brightsky (weather → overseas_weather_regional via _upsert_weather_rows)
- nndss (AU flu state → overseas_flu_state via _upsert_flu_state_rows)
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from sqlite3 import Connection as _Conn
from typing import Optional

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_DB = Path("simulation/data/db/epi_real_seoul.db")
_REQUEST_TIMEOUT_S = 60


# ─────────────────────────────────────────────────────────────────────────────
# Connection helpers (G-116/G-117 safe_connect)
# ─────────────────────────────────────────────────────────────────────────────

# Use safe_connect (WAL + quick_check + corruption guard).
# Import lazily so the module loads even without the DB present at import time.
def _safe_connect_import():
    from simulation.database import safe_connect
    return safe_connect


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


# ─────────────────────────────────────────────────────────────────────────────
# Value coercion (used by all CSV/JSON parsers)
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(str(v).strip())
        return f if f == f else None  # NaN guard
    except (ValueError, TypeError):
        return None


def _safe_int(v) -> Optional[int]:
    try:
        return int(float(str(v).strip()))
    except (ValueError, TypeError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# HTTP retry
# ─────────────────────────────────────────────────────────────────────────────

def _retry_get(url: str, params: dict | None = None, max_retries: int = 3,
               sleep_s: float = 5.0, timeout: int = _REQUEST_TIMEOUT_S) -> "requests.Response":  # noqa: F821
    """GET with retry on 429/5xx.

    Args:
        url:         Request URL.
        params:      Query parameters dict (passed to requests.get).
        max_retries: Maximum number of attempts (default 3).
        sleep_s:     Sleep between retries in seconds (doubled on 429).
        timeout:     Per-request timeout in seconds.

    Returns:
        requests.Response with status 2xx.

    Raises:
        RuntimeError: if all retries exhausted.

    Performance: up to max_retries × (timeout + sleep_s) seconds.
    Side effects: None.
    Caller responsibility: `requests` must be importable.
    """
    import requests
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                wait = sleep_s * (attempt + 2)
                log.warning("[overseas] 429 rate-limited on %s — sleeping %.0fs", url[:60], wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"GET {url[:80]} failed after {max_retries} attempts: {e}") from e
            log.warning("[overseas] attempt %d/%d failed (%s) — retrying", attempt + 1, max_retries, e)
            time.sleep(sleep_s)
    raise RuntimeError(f"GET {url[:80]} exhausted {max_retries} retries")


# ─────────────────────────────────────────────────────────────────────────────
# Table DDL (4 schemas)
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_overseas_ili_table(con: _Conn) -> None:
    """Create overseas_ili if it doesn't exist; add any missing columns."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS overseas_ili (
            source             TEXT NOT NULL,
            country            TEXT NOT NULL,
            year               INTEGER NOT NULL,
            week_no            INTEGER NOT NULL,
            ili_rate           REAL,
            specimen_positive  INTEGER,
            specimen_total     INTEGER,
            influenza_a        INTEGER,
            influenza_b        INTEGER,
            positivity_pct     REAL,
            collected_at       TEXT,
            PRIMARY KEY (source, country, year, week_no)
        )
    """)
    # Add new columns if the table existed without them (back-compat)
    existing = {r[1] for r in con.execute("PRAGMA table_info(overseas_ili)")}
    for col, defn in [("influenza_b", "INTEGER"), ("positivity_pct", "REAL"),
                      ("collected_at", "TEXT")]:
        if col not in existing:
            con.execute(f"ALTER TABLE overseas_ili ADD COLUMN {col} {defn}")
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_overseas_ili_country_year "
        "ON overseas_ili(country, year, week_no)"
    )
    con.commit()


def _ensure_overseas_ili_regional_table(con: _Conn) -> None:
    """Create overseas_ili_regional if missing; add columns back-compat.

    Args:
        con: Open SQLite connection.

    Side effects: may ALTER TABLE to add missing columns; commits.
    """
    con.execute("""
        CREATE TABLE IF NOT EXISTS overseas_ili_regional (
            source       TEXT NOT NULL,
            country      TEXT NOT NULL,
            region       TEXT NOT NULL,
            year         INTEGER NOT NULL,
            week_no      INTEGER NOT NULL,
            ili_rate     REAL,
            collected_at TEXT,
            PRIMARY KEY (source, country, region, year, week_no)
        )
    """)
    existing = {r[1] for r in con.execute("PRAGMA table_info(overseas_ili_regional)")}
    for col, defn in [("collected_at", "TEXT")]:
        if col not in existing:
            con.execute(f"ALTER TABLE overseas_ili_regional ADD COLUMN {col} {defn}")
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_overseas_ili_reg_ccy "
        "ON overseas_ili_regional(country, year, week_no)"
    )
    con.commit()


def _ensure_overseas_weather_regional_table(con: _Conn) -> None:
    """Create overseas_weather_regional if missing.

    Args:
        con: Open SQLite connection.

    Side effects: creates table + index; commits.
    """
    con.execute("""
        CREATE TABLE IF NOT EXISTS overseas_weather_regional (
            country       TEXT NOT NULL,
            region        TEXT NOT NULL,
            date          TEXT NOT NULL,
            lat           REAL,
            lon           REAL,
            temp_max      REAL,
            temp_min      REAL,
            precip        REAL,
            wind_max      REAL,
            humidity_mean REAL,
            collected_at  TEXT,
            PRIMARY KEY (country, region, date)
        )
    """)
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_weather_reg_ccy "
        "ON overseas_weather_regional(country, region, date)"
    )
    con.commit()


def _ensure_overseas_flu_state_table(con: _Conn) -> None:
    """Create overseas_flu_state if missing.

    Args:
        con: Open SQLite connection.

    Side effects: creates table + index; commits.
    """
    con.execute("""
        CREATE TABLE IF NOT EXISTS overseas_flu_state (
            country        TEXT NOT NULL DEFAULT 'AU',
            state          TEXT NOT NULL,
            epiweek        INTEGER NOT NULL,
            confirmed_flu_a INTEGER,
            confirmed_flu_b INTEGER,
            total_flu       INTEGER,
            population      INTEGER,
            rate_per_100k   REAL,
            collected_at    TEXT,
            PRIMARY KEY (country, state, epiweek)
        )
    """)
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_flu_state_cc "
        "ON overseas_flu_state(country, state, epiweek)"
    )
    con.commit()


# ─────────────────────────────────────────────────────────────────────────────
# DB upserts (4 tables)
# ─────────────────────────────────────────────────────────────────────────────

def _upsert_rows(con: _Conn, rows: list[dict]) -> tuple[int, int]:
    """UPSERT rows into overseas_ili.

    Args:
        con:  Open SQLite connection.
        rows: List of dicts with schema keys.

    Returns:
        (inserted, skipped) counts.  ``inserted`` counts rows where
        at least one column value changed; ``skipped`` counts identical
        existing rows.

    Performance: single transaction per call, chunk_size=500.
    Side effects: writes to overseas_ili.
    """
    if not rows:
        return 0, 0

    sql = """
        INSERT INTO overseas_ili
          (source, country, year, week_no, ili_rate,
           specimen_positive, specimen_total, influenza_a,
           influenza_b, positivity_pct, collected_at)
        VALUES
          (:source, :country, :year, :week_no, :ili_rate,
           :specimen_positive, :specimen_total, :influenza_a,
           :influenza_b, :positivity_pct, :collected_at)
        ON CONFLICT(source, country, year, week_no) DO UPDATE SET
          ili_rate          = COALESCE(excluded.ili_rate, ili_rate),
          specimen_positive = COALESCE(excluded.specimen_positive, specimen_positive),
          specimen_total    = COALESCE(excluded.specimen_total, specimen_total),
          influenza_a       = COALESCE(excluded.influenza_a, influenza_a),
          influenza_b       = COALESCE(excluded.influenza_b, influenza_b),
          positivity_pct    = COALESCE(excluded.positivity_pct, positivity_pct),
          collected_at      = excluded.collected_at
    """
    inserted = 0
    chunk_size = 500
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i:i + chunk_size]
        cur = con.executemany(sql, chunk)
        inserted += cur.rowcount
        con.commit()

    skipped = len(rows) - inserted
    return inserted, max(0, skipped)


def _upsert_regional_ili_rows(con: _Conn, rows: list[dict]) -> tuple[int, int]:
    """UPSERT rows into overseas_ili_regional.

    Args:
        con:  Open SQLite connection.
        rows: List of dicts with keys: source, country, region, year, week_no,
              ili_rate, collected_at.

    Returns:
        (inserted, skipped) counts.

    Performance: single transaction per chunk of 500.
    Side effects: writes to overseas_ili_regional.
    """
    if not rows:
        return 0, 0
    sql = """
        INSERT INTO overseas_ili_regional
          (source, country, region, year, week_no, ili_rate, collected_at)
        VALUES
          (:source, :country, :region, :year, :week_no, :ili_rate, :collected_at)
        ON CONFLICT(source, country, region, year, week_no) DO UPDATE SET
          ili_rate     = COALESCE(excluded.ili_rate, ili_rate),
          collected_at = excluded.collected_at
    """
    inserted = 0
    for i in range(0, len(rows), 500):
        cur = con.executemany(sql, rows[i:i + 500])
        inserted += cur.rowcount
        con.commit()
    return inserted, max(0, len(rows) - inserted)


def _upsert_weather_rows(con: _Conn, rows: list[dict]) -> tuple[int, int]:
    """UPSERT rows into overseas_weather_regional.

    Args:
        con:  Open SQLite connection.
        rows: List of dicts with keys matching overseas_weather_regional schema.

    Returns:
        (inserted, skipped) counts.

    Performance: single transaction per chunk of 500.
    Side effects: writes to overseas_weather_regional.
    """
    if not rows:
        return 0, 0
    sql = """
        INSERT INTO overseas_weather_regional
          (country, region, date, lat, lon,
           temp_max, temp_min, precip, wind_max, humidity_mean, collected_at)
        VALUES
          (:country, :region, :date, :lat, :lon,
           :temp_max, :temp_min, :precip, :wind_max, :humidity_mean, :collected_at)
        ON CONFLICT(country, region, date) DO UPDATE SET
          temp_max      = COALESCE(excluded.temp_max, temp_max),
          temp_min      = COALESCE(excluded.temp_min, temp_min),
          precip        = COALESCE(excluded.precip, precip),
          wind_max      = COALESCE(excluded.wind_max, wind_max),
          humidity_mean = COALESCE(excluded.humidity_mean, humidity_mean),
          collected_at  = excluded.collected_at
    """
    inserted = 0
    for i in range(0, len(rows), 500):
        cur = con.executemany(sql, rows[i:i + 500])
        inserted += cur.rowcount
        con.commit()
    return inserted, max(0, len(rows) - inserted)


def _upsert_flu_state_rows(con: _Conn, rows: list[dict]) -> tuple[int, int]:
    """UPSERT rows into overseas_flu_state.

    Args:
        con:  Open SQLite connection.
        rows: List of dicts with flu_state schema keys.

    Returns:
        (inserted, skipped) counts.

    Performance: single transaction per chunk of 500.
    Side effects: writes to overseas_flu_state.
    """
    if not rows:
        return 0, 0
    sql = """
        INSERT INTO overseas_flu_state
          (country, state, epiweek, confirmed_flu_a, confirmed_flu_b,
           total_flu, population, rate_per_100k, collected_at)
        VALUES
          (:country, :state, :epiweek, :confirmed_flu_a, :confirmed_flu_b,
           :total_flu, :population, :rate_per_100k, :collected_at)
        ON CONFLICT(country, state, epiweek) DO UPDATE SET
          confirmed_flu_a = COALESCE(excluded.confirmed_flu_a, confirmed_flu_a),
          confirmed_flu_b = COALESCE(excluded.confirmed_flu_b, confirmed_flu_b),
          total_flu       = COALESCE(excluded.total_flu, total_flu),
          population      = COALESCE(excluded.population, population),
          rate_per_100k   = COALESCE(excluded.rate_per_100k, rate_per_100k),
          collected_at    = excluded.collected_at
    """
    inserted = 0
    for i in range(0, len(rows), 500):
        cur = con.executemany(sql, rows[i:i + 500])
        inserted += cur.rowcount
        con.commit()
    return inserted, max(0, len(rows) - inserted)


__all__ = [
    # config
    "_DEFAULT_DB", "_REQUEST_TIMEOUT_S",
    # connect
    "_safe_connect_import", "_resolve_db", "_connect",
    # coercion
    "_safe_float", "_safe_int",
    # http
    "_retry_get",
    # ddl
    "_ensure_overseas_ili_table",
    "_ensure_overseas_ili_regional_table",
    "_ensure_overseas_weather_regional_table",
    "_ensure_overseas_flu_state_table",
    # upsert
    "_upsert_rows",
    "_upsert_regional_ili_rows",
    "_upsert_weather_rows",
    "_upsert_flu_state_rows",
]
