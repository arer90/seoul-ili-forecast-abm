"""
simulation.database.storage — Hardened SQLite storage layer
============================================================

Single source of truth for:
* Opening WAL-mode connections with ``PRAGMA quick_check`` gating
* Creating/validating the project schema (idempotent)
* Bulk inserts with explicit transactions
* Lightweight query/aggregate helpers
* A schema audit that covers *both* tables defined here and tables created
  by collectors (``kosis_disease_gender``, ``commuter_matrix``,
  ``who_flunet``, ``seoul_annual_report_*``, ...).

The module is intentionally dependency-free (only stdlib) so every other
subsystem — collectors, feature_engine, runner — can rely on a stable
connection primitive without pulling in heavier layers.
"""
from __future__ import annotations

import datetime as dt
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from .config import COLLECT_DIR, DB_PATH  # noqa: F401  (re-export)

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Connection primitives
# ══════════════════════════════════════════════════════════════════════════
class DatabaseCorruptError(RuntimeError):
    """Raised when ``PRAGMA quick_check`` detects a malformed database."""


#: Baseline PRAGMAs for normal (mixed read/write) workloads.
#: Tuned for a 680MB SQLite with single-writer ingestion + heavy analytical reads.
_PRAGMAS_NORMAL = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA synchronous=NORMAL",
    "PRAGMA foreign_keys=ON",
    "PRAGMA temp_store=MEMORY",
    "PRAGMA busy_timeout=30000",
    "PRAGMA cache_size=-200000",      # 200 MB page cache (negative = KiB)
    "PRAGMA mmap_size=1073741824",    # 1 GiB memory-mapped I/O for reads
)

#: Aggressive PRAGMAs for bulk-load phases. Trades durability for throughput.
#: Always follow with :func:`tune_for_normal` after the load finishes.
_PRAGMAS_BULK = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA synchronous=OFF",         # skip fsync — DB may lose last txn on power loss
    "PRAGMA foreign_keys=OFF",        # skip FK checks during load
    "PRAGMA temp_store=MEMORY",
    "PRAGMA cache_size=-500000",      # 500 MB
    "PRAGMA mmap_size=2147483648",    # 2 GiB
    "PRAGMA locking_mode=EXCLUSIVE",  # skip filesystem lock contention
)


def _apply_pragmas(
    conn: sqlite3.Connection,
    pragmas: tuple[str, ...] = _PRAGMAS_NORMAL,
) -> None:
    for stmt in pragmas:
        conn.execute(stmt)


def tune_for_bulk_load(conn: sqlite3.Connection) -> None:
    """Switch connection to aggressive bulk-load PRAGMAs.

    Use inside a dedicated ingestion session::

        conn = safe_connect()
        tune_for_bulk_load(conn)
        try:
            with transaction(conn):
                conn.executemany(sql, big_batch)
        finally:
            tune_for_normal(conn)
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    """
    _apply_pragmas(conn, _PRAGMAS_BULK)


def tune_for_normal(conn: sqlite3.Connection) -> None:
    """Restore production-safe PRAGMAs after a bulk load."""
    _apply_pragmas(conn, _PRAGMAS_NORMAL)


def get_conn(
    db_path: Optional[str] = None,
    *,
    timeout: float = 30,
    isolation_level: Any = "__default__",
) -> sqlite3.Connection:
    """Return a WAL-mode SQLite connection with sensible defaults.

 : ``timeout`` and ``isolation_level`` are forwardable from
 :func:`safe_connect` so external-import / PDF-extract collectors can
 opt into autocommit or longer busy-wait windows.
 """
    p = db_path or DB_PATH
    kwargs: dict[str, Any] = {"timeout": timeout, "check_same_thread": False}
    if isolation_level != "__default__":
        kwargs["isolation_level"] = isolation_level
    conn = sqlite3.connect(p, **kwargs)
    conn.row_factory = sqlite3.Row
    _apply_pragmas(conn)
    return conn


def quick_check(db_path: Optional[str] = None) -> str:
    """Run ``PRAGMA quick_check`` and return the raw result string.

    Returns ``'ok'`` on healthy DBs.  On malformed DBs, returns the first
    error message from SQLite (which may span multiple lines).
    """
    conn = sqlite3.connect(db_path or DB_PATH, timeout=30)
    try:
        row = conn.execute("PRAGMA quick_check").fetchone()
        return (row[0] if row else "unknown") or "unknown"
    finally:
        conn.close()


def safe_connect(
    db_path: Optional[str] = None,
    *,
    verify: bool = True,
    timeout: Optional[float] = None,
    isolation_level: Any = "__default__",
) -> sqlite3.Connection:
    """Open a connection only if ``PRAGMA quick_check`` says ``'ok'``.

 Raises :class:`DatabaseCorruptError` otherwise so callers cannot
 silently write into a malformed database. Pass ``verify=False`` to
 skip the pre-flight check (used by ``init_db`` on fresh files).

 kwargs (2026-04-19):
 * ``timeout`` — forward to ``sqlite3.connect`` busy-wait timeout.
 Collectors (D-CATALOG, external imports, PDF extract) need
 longer-than-default windows under WAL contention. Falls back to
 sqlite3's 5 s default when None.
 * ``isolation_level`` — passed straight through. ``"__default__"``
 sentinel keeps the module-wide tuned default (BEGIN IMMEDIATE
 behavior); ``None`` enables autocommit which the external
 importer needs for PRAGMA batches.
 """
    p = db_path or DB_PATH
    if verify and Path(p).exists() and Path(p).stat().st_size > 0:
        status = quick_check(p)
        if status.strip().lower() != "ok":
            raise DatabaseCorruptError(
                f"SQLite quick_check failed for {p}: {status!r}"
            )
    kwargs: dict[str, Any] = {}
    if timeout is not None:
        kwargs["timeout"] = timeout
    if isolation_level != "__default__":
        kwargs["isolation_level"] = isolation_level
    return get_conn(p, **kwargs) if kwargs else get_conn(p)


def read_only_connect(
    db_path: Optional[str] = None, *, timeout: float = 2.0
) -> sqlite3.Connection:
    """Open a LOCK-FREE read-only SQLite connection (``mode=ro`` + busy_timeout).

    The single entry point for analytics / MCP / proof reads that may run WHILE a
    training process holds the write lock. Unlike :func:`safe_connect`, this does
    NOT run ``PRAGMA quick_check`` (which opens read-write and can block on the
    write lock) and never attempts a write — so it cannot deadlock a writer.

    Args:
        db_path: path to the SQLite file; ``None`` → the project ``DB_PATH``.
        timeout: lock-wait, seconds. Sets BOTH the connect busy-wait and
            ``PRAGMA busy_timeout`` (= ``timeout`` ms·1000). Default 2 s — long
            enough to ride out a checkpoint, short enough to fail fast.

    Returns:
        An open, read-only :class:`sqlite3.Connection`. Writes raise
        ``sqlite3.OperationalError`` ("attempt to write a readonly database").

    Raises:
        sqlite3.OperationalError: if the file does not exist (``mode=ro`` will not
            create it) — intentional: a read of a missing DB must fail loudly.

    Performance: O(1) open; no quick_check scan. Side effects: opens a fd; the
    CALLER MUST ``close()`` (or use ``contextlib.closing``). Concurrency: safe to
    open against a DB under active WAL writes; never blocks the writer.
    """
    p = str(db_path or DB_PATH)
    con = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=timeout)
    con.execute(f"PRAGMA busy_timeout={int(timeout * 1000)}")
    return con


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Explicit ``BEGIN IMMEDIATE`` / ``COMMIT`` / ``ROLLBACK`` context."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()


# ══════════════════════════════════════════════════════════════════════════
# Schema
# ══════════════════════════════════════════════════════════════════════════
SCHEMA_SQL = """
-- ── Core epidemiological tables ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS weekly_disease (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 collected_at TEXT NOT NULL,
 week_start TEXT NOT NULL,
 week_end TEXT,
 disease_cd TEXT NOT NULL,
 disease_nm TEXT,
 cnt_confirmed INTEGER DEFAULT 0,
 cnt_suspected INTEGER DEFAULT 0,
 cnt_death INTEGER DEFAULT 0,
 source_type TEXT DEFAULT 'KDCA',
 disease_group TEXT,
 UNIQUE(week_start, disease_cd, source_type)
);

CREATE TABLE IF NOT EXISTS disease_master (
 disease_cd TEXT PRIMARY KEY,
 disease_nm TEXT,
 legal_grade TEXT,
 icd10 TEXT,
 transmission_route TEXT,
 vaccine_available INTEGER DEFAULT 0,
 year INTEGER,
 flags TEXT
);

CREATE TABLE IF NOT EXISTS disease_death (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 collected_at TEXT, year INTEGER, disease_cd TEXT, disease_nm TEXT,
 cnt_death INTEGER DEFAULT 0, disease_group TEXT,
 UNIQUE(year, disease_cd)
);

CREATE TABLE IF NOT EXISTS disease_age (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 collected_at TEXT, year INTEGER, disease_cd TEXT, disease_nm TEXT,
 age_group TEXT, cnt INTEGER DEFAULT 0, disease_group TEXT,
 UNIQUE(year, disease_cd, age_group)
);

CREATE TABLE IF NOT EXISTS disease_gender (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 collected_at TEXT, year INTEGER, disease_cd TEXT, disease_nm TEXT,
 gender TEXT, cnt INTEGER DEFAULT 0, disease_group TEXT,
 UNIQUE(year, disease_cd, gender)
);

-- NOTE: The following tables are created/owned by collectors
-- (simulation.collectors.import_external, .extract_pdf) and intentionally
-- omitted from SCHEMA_SQL so storage.init_db doesn't fight with collector
-- schemas:
-- kosis_disease_gender, kosis_source_registry,
-- seoul_annual_report_district, seoul_annual_report_monthly,
-- commuter_matrix, who_flunet, who_flunet_metadata
-- They still appear in EXPECTED_TABLES so verify_schema reports them.

-- ── Weather ─────────────────────────────────────────────────────────
-- Schema matches what simulation/collectors/legacy/group_b_weather.py
-- inserts (ta_avg / ta_max / ta_min / hm_avg / ws_avg / rn_day / ps_avg /
-- ss_day) and what simulation/models/feature_engine/loaders._load_weather
-- expects. Prior definition (avg_temp / min_temp / max_temp / date) was
-- a schema drift that broke the loader on fresh bootstrap (2026-04-24).
CREATE TABLE IF NOT EXISTS weather_historical (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 obs_date TEXT NOT NULL,
 stn_id INTEGER,
 stn_nm TEXT,
 ta_avg REAL,
 ta_max REAL,
 ta_min REAL,
 hm_avg REAL,
 ws_avg REAL,
 rn_day REAL,
 ps_avg REAL,
 ss_day REAL,
 UNIQUE(obs_date, stn_id)
);

CREATE TABLE IF NOT EXISTS weather_forecast (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 collected_at TEXT, forecast_date TEXT,
 avg_temp REAL, min_temp REAL, max_temp REAL,
 precip_prob REAL, humidity REAL,
 UNIQUE(collected_at, forecast_date)
);

-- legacy cleanup: population_kosis 는 kosis_age_district 로 이관되어 제거.
-- ── Sentinel surveillance ───────────────────────────────────────────
-- Schema matches group_s_sentinel.py and the feature_engine loader's SELECT.
-- Prior definition (week_start / ili_cases / source) was a schema drift
-- that broke _load_sentinel_ili on fresh bootstrap (2026-04-24).
CREATE TABLE IF NOT EXISTS sentinel_influenza (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 collected_at TEXT,
 season_start INTEGER,
 week_seq INTEGER,
 week_label TEXT,
 age_group TEXT,
 ili_rate REAL,
 UNIQUE(season_start, week_seq, age_group)
);

CREATE TABLE IF NOT EXISTS sentinel_ari (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 collected_at TEXT,
 year INTEGER,
 week_no INTEGER,
 pathogen_group TEXT,
 pathogen_nm TEXT,
 count INTEGER,
 UNIQUE(year, week_no, pathogen_nm)
);

-- ── Vaccination ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS vaccination_coverage (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 collected_at TEXT, vaccine_type TEXT, age_group TEXT,
 coverage_pct REAL, year INTEGER,
 UNIQUE(vaccine_type, age_group, year)
);

-- ── Employment ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS employment_workplace (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 collected_at TEXT, year_month TEXT, gu_nm TEXT,
 industry_cd TEXT, emp_count INTEGER,
 UNIQUE(year_month, gu_nm, industry_cd)
);

CREATE TABLE IF NOT EXISTS employment_residence (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 collected_at TEXT, year_month TEXT, gu_nm TEXT,
 emp_count INTEGER,
 UNIQUE(year_month, gu_nm)
);

-- ── School ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS school_info (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 school_code TEXT UNIQUE, school_nm TEXT, school_type TEXT,
 gu_nm TEXT, addr TEXT, student_cnt INTEGER, class_cnt INTEGER,
 lat REAL, lng REAL
);

-- : Group R (NEIS) target table — column layout mirrors the
-- SchoolInfoRow pydantic schema in collectors/schemas.py. Kept distinct
-- from `school_info` above (which is an older legacy stub) so Group R's
-- Seoul-specific insert path is schema-stable.
CREATE TABLE IF NOT EXISTS school_info_seoul (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 collected_at TEXT,
 school_code TEXT UNIQUE,
 school_name TEXT,
 school_type TEXT,
 gu_name TEXT,
 address TEXT,
 found_date TEXT
);

CREATE TABLE IF NOT EXISTS school_closure_seoul (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 collected_at TEXT,
 date TEXT NOT NULL,
 school_name TEXT,
 school_type TEXT,
 event_name TEXT,
 is_closure INTEGER DEFAULT 0,
 event_content TEXT,
 UNIQUE(date, school_name, event_name)
);

-- : Group C7 (중앙의료원 응급의료 실시간 가용병상) target table.
-- Column set reflects the dict keys emitted by
-- collectors/legacy/group_c_daily.py:~980.
CREATE TABLE IF NOT EXISTS emergency_room_availability (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 collected_at TEXT NOT NULL,
 hp_id TEXT NOT NULL,
 hp_nm TEXT,
 sido_nm TEXT,
 gu_nm TEXT,
 hp_tel TEXT,
 latitude REAL,
 longitude REAL,
 hvec INTEGER, hvoc INTEGER, hvcc INTEGER, hvncc INTEGER,
 hvicc INTEGER, hvgc INTEGER,
 hv2 INTEGER, hv3 INTEGER, hv6 INTEGER, hv8 INTEGER,
 hv9 INTEGER, hv10 INTEGER, hv11 INTEGER,
 hvamyn TEXT,
 UNIQUE(collected_at, hp_id)
);

-- legacy cleanup: school_closure / hospital_info / hira_claims /
-- google_trends 는 리팩터링 이후 로더/수집기가 사라져 전부 빈 테이블로
-- 남아 있었음. SCHEMA_SQL 에서 제거. 기존 DB 의 빈 테이블은
-- maintain.drop_legacy_empty_tables 가 DROP 한다.

-- ── Real-time population (Seoul citydata) ───────────────────────────
CREATE TABLE IF NOT EXISTS rt_population_detail (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 collected_at TEXT, area_cd TEXT, area_nm TEXT,
 congestion TEXT, ppltn_min INTEGER, ppltn_max INTEGER,
 male_rate REAL, female_rate REAL,
 resnt_rate REAL, nonresnt_rate REAL,
 UNIQUE(collected_at, area_cd)
);

CREATE TABLE IF NOT EXISTS rt_road_traffic (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 collected_at TEXT, area_cd TEXT, area_nm TEXT,
 road_msg TEXT, road_spd REAL, road_idx REAL,
 UNIQUE(collected_at, area_cd)
);

CREATE TABLE IF NOT EXISTS rt_subway_crowd (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 collected_at TEXT, area_cd TEXT, area_nm TEXT,
 sub_line TEXT, sub_stn TEXT, sub_dir TEXT,
 sub_rcp TEXT, sub_gton INTEGER, sub_gtoff INTEGER,
 UNIQUE(collected_at, area_cd, sub_line, sub_dir)
);

CREATE TABLE IF NOT EXISTS poi_metadata (
 poi_code TEXT PRIMARY KEY,
 area_nm TEXT NOT NULL,
 category TEXT,
 lat REAL, lng REAL,
 first_seen TEXT, last_seen TEXT
);

-- ── Overseas ILI ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS overseas_ili (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 collected_at TEXT, source TEXT, country TEXT,
 year INTEGER, week_no INTEGER, ili_rate REAL,
 UNIQUE(source, country, year, week_no)
);

-- legacy cleanup: subway_hourly / bus_hourly 는 에서
-- monthly_subway_hourly / monthly_bus_hourly 로 이관되었으나
-- 구 스키마가 빈 테이블로 남아 있었음. SCHEMA_SQL 에서 제거.
-- (월별 집계 테이블은 collectors/group_*.py 가 자체 CREATE 한다.)

-- ── Collection log ──────────────────────────────────────────────────
-- NOTE: 실제 DB 스키마와 수집기 caller 패턴에 맞춰 컬럼명을 통일.
-- legacy: source / rows_inserted / error
-- : api_name / rows_saved / error_msg (+ status enum)
-- collection_log 컬럼은 반드시 log_collection 함수 시그니쳐와 일치해야 함.
CREATE TABLE IF NOT EXISTS collection_log (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 collected_at TEXT NOT NULL,
 group_name TEXT NOT NULL,
 api_name TEXT NOT NULL,
 status TEXT NOT NULL,
 rows_saved INTEGER DEFAULT 0,
 error_msg TEXT,
 elapsed_sec REAL
);
CREATE INDEX IF NOT EXISTS idx_collection_log_ts
 ON collection_log(collected_at);
CREATE INDEX IF NOT EXISTS idx_collection_log_group
 ON collection_log(group_name, api_name);
"""

#: Tables that ``verify_schema`` expects to exist after ``init_db``
#: plus the tables created by collectors in ``simulation.collectors``.
#:
#: NOTE: 2026-04-17 DB audit 기준 64개 액티브 테이블을 전부 등록.
#:   이전에는 SCHEMA_SQL 정의 테이블만 26개 올라와 있어서 verify_schema() 가
#:   37개의 콜렉터-소유 테이블을 매번 "extra" 로 분류했음. 이제 실제 DB 에
#:   존재하는 모든 테이블을 명시적 화이트리스트로 관리한다.
EXPECTED_TABLES: frozenset[str] = frozenset({
    # ── Core disease tables (storage.py SCHEMA_SQL) ─────────────────
    "weekly_disease", "disease_master", "disease_death",
    "disease_age", "disease_gender",
    "disease_catalog", "disease_name_mapping",
    # ── KOSIS / Commuter / WHO (import_external.py) ─────────────────
    "kosis_disease_gender", "kosis_source_registry",
    "kosis_age_district",
    "commuter_matrix",
    "who_flunet", "who_flunet_metadata",
    # ── Seoul annual report (extract_pdf.py) ────────────────────────
    "seoul_annual_report_district", "seoul_annual_report_monthly",
    "seoul_annual_report_age", "seoul_annual_report_gender",
    "seoul_annual_report_infection_region",
    "seoul_annual_report_patient_class",
    "seoul_disease_district",
    # ── Weather ─────────────────────────────────────────────────────
    "weather_historical", "weather_forecast",
    # ── Sentinel surveillance (group_s_sentinel.py) ─────────────────
    "sentinel_influenza", "sentinel_ari", "sentinel_sari",
    "sentinel_enterovirus", "sentinel_hfmd", "sentinel_hfmdc",
    "sentinel_intestinal", "sentinel_ophlgc",
    # ── Vaccination / Employment (group_f, group_c) ─────────────────
    "vaccination_coverage", "childhood_vaccination_rates",
    "employment_workplace", "employment_residence", "employment_monthly",
    # ── School / Hospital / ED (group_r, group_q) ───────────────────
    "school_info", "school_closure_seoul",
    "hospitals", "ed_visits_symptom",
    # ── Real-time density & environment (group_a, group_c) ──────────
    "rt_population", "rt_population_detail", "rt_population_forecast",
    "rt_road_traffic", "rt_subway_crowd",
    "rt_air_quality", "rt_sdot_env", "rt_bike_status",
    "poi_metadata",
    # ── Daily & monthly mobility (group_c) ──────────────────────────
    "daily_population_district", "daily_population_dong",
    "daily_population_gu_hourly", "daily_population_hotspot",
    "daily_bus", "daily_subway",
    "monthly_bus_hourly", "monthly_subway_hourly",
    # ── HIRA health claims (group_h) ────────────────────────────────
    "hira_facility", "hira_gender_age", "hira_inpat_opat", "hira_region",
    # ── External signals (group_g, group_p) ─────────────────────────
    "google_search_trends", "pubmed_abstracts",
    # ── Overseas ILI & bookkeeping ──────────────────────────────────
    "overseas_ili",
    "collection_log",
})


def init_db(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Create schema (idempotent) and return a safe connection."""
    p = db_path or DB_PATH
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    # Fresh DBs can't pass quick_check, so skip verify on first init.
    conn = safe_connect(p, verify=Path(p).exists() and Path(p).stat().st_size > 0)
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    log.info("DB schema initialized: %s", p)
    return conn


def verify_schema(
    conn: Optional[sqlite3.Connection] = None,
    *,
    db_path: Optional[str] = None,
) -> dict:
    """Return ``{'ok': bool, 'missing': [...], 'extra': [...]}``.

    ``extra`` lists tables present in the DB but not in
    :data:`EXPECTED_TABLES` — informational only.
    """
    own = conn is None
    if own:
        conn = safe_connect(db_path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        present = {r[0] for r in rows if not r[0].startswith("sqlite_")}
        missing = sorted(EXPECTED_TABLES - present)
        extra = sorted(present - EXPECTED_TABLES)
        return {"ok": not missing, "missing": missing, "extra": extra}
    finally:
        if own:
            conn.close()


# ══════════════════════════════════════════════════════════════════════════
# CRUD helpers
# ══════════════════════════════════════════════════════════════════════════
def insert_rows(
    table: str,
    rows: list[dict],
    conn: Optional[sqlite3.Connection] = None,
    on_conflict: str = "IGNORE",
    *,
    replace: bool = False,
) -> int:
    """INSERT OR <on_conflict> rows into ``table``.

 Uses an explicit ``BEGIN IMMEDIATE`` when opening its own connection
 so concurrent writers don't corrupt partial batches.

 kwarg: ``replace=True`` is a convenience alias for
 ``on_conflict="REPLACE"`` — the NEIS/HIRA reference tables
 (school_info, hospitals, disease_catalog) use upsert-on-unique
 semantics and were calling ``insert_rows(..., replace=True)``
 already. This restores that implicit contract.
 """
    if not rows:
        return 0
    if replace:
        on_conflict = "REPLACE"
    own_conn = conn is None
    if own_conn:
        conn = safe_connect()
    cols = list(rows[0].keys())
    placeholders = ",".join(["?"] * len(cols))
    col_str = ",".join(cols)
    sql = (
        f"INSERT OR {on_conflict} INTO {table} ({col_str}) "
        f"VALUES ({placeholders})"
    )
    data = [tuple(r.get(c) for c in cols) for r in rows]
    try:
        if own_conn:
            with transaction(conn):
                conn.executemany(sql, data)
        else:
            conn.executemany(sql, data)
            conn.commit()
        return len(data)
    except Exception as e:
        log.error("insert_rows(%s): %s", table, e)
        try:
            conn.rollback()
        except Exception:
            pass
        return 0
    finally:
        if own_conn:
            conn.close()


def query(
    sql: str,
    params: Optional[Iterable] = None,
    db_path: Optional[str] = None,
) -> list[dict]:
    """Execute SELECT and return list of dicts."""
    conn = safe_connect(db_path)
    try:
        cur = conn.execute(sql, list(params or []))
        cols = [d[0] for d in cur.description] if cur.description else []
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()


def get_latest(
    table: str,
    date_col: str = "collected_at",
    db_path: Optional[str] = None,
) -> Optional[str]:
    """Get ``MAX(date_col)`` from ``table`` or ``None`` if empty."""
    rows = query(
        f"SELECT MAX({date_col}) AS latest FROM {table}", db_path=db_path
    )
    return rows[0]["latest"] if rows and rows[0]["latest"] else None


def get_table_shapes(db_path: Optional[str] = None) -> dict[str, int]:
    """Return ``{table_name: row_count}`` for all user tables."""
    conn = safe_connect(db_path)
    try:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        ]
        result: dict[str, int] = {}
        for t in tables:
            try:
                cnt = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                result[t] = int(cnt)
            except Exception:
                result[t] = -1
        return result
    finally:
        conn.close()


def bulk_insert(
    table: str,
    rows: list[dict],
    *,
    conn: Optional[sqlite3.Connection] = None,
    on_conflict: str = "IGNORE",
    chunk_size: int = 10_000,
) -> int:
    """High-throughput insert for large batches (>10k rows).

    Differences from :func:`insert_rows`:
    * Applies aggressive bulk PRAGMAs for the duration of the call
    * Splits rows into ``chunk_size`` transactions so WAL doesn't explode
    * Performs ``PRAGMA wal_checkpoint(TRUNCATE)`` at the end
    * Restores normal PRAGMAs before returning
    """
    if not rows:
        return 0
    own_conn = conn is None
    if own_conn:
        conn = safe_connect()
    cols = list(rows[0].keys())
    placeholders = ",".join(["?"] * len(cols))
    col_str = ",".join(cols)
    sql = (
        f"INSERT OR {on_conflict} INTO {table} ({col_str}) "
        f"VALUES ({placeholders})"
    )
    data = [tuple(r.get(c) for c in cols) for r in rows]

    prior = None
    try:
        if own_conn:
            tune_for_bulk_load(conn)
        inserted = 0
        for i in range(0, len(data), chunk_size):
            chunk = data[i : i + chunk_size]
            with transaction(conn):
                conn.executemany(sql, chunk)
            inserted += len(chunk)
        if own_conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return inserted
    except Exception as e:
        log.error("bulk_insert(%s): %s", table, e)
        try:
            conn.rollback()
        except Exception:
            pass
        return 0
    finally:
        if own_conn:
            tune_for_normal(conn)
            conn.close()
        _ = prior  # silence unused


def checkpoint_wal(db_path: Optional[str] = None, mode: str = "TRUNCATE") -> None:
    """Force a WAL checkpoint (TRUNCATE / FULL / PASSIVE / RESTART)."""
    conn = safe_connect(db_path)
    try:
        conn.execute(f"PRAGMA wal_checkpoint({mode})")
    finally:
        conn.close()


def vacuum_analyze(db_path: Optional[str] = None) -> None:
    """Run ``VACUUM`` + ``ANALYZE`` to reclaim space and refresh query planner stats.

    Note: ``VACUUM`` rewrites the entire DB file (expensive on 680MB). Run
    after a large ingest or once per month in cron.
    """
    conn = safe_connect(db_path)
    try:
        conn.isolation_level = None  # VACUUM cannot run inside a transaction
        conn.execute("VACUUM")
        conn.execute("ANALYZE")
    finally:
        conn.close()


def log_collection(
    group: str,
    api_name: str,
    status: str,
    rows_saved: int = 0,
    *,
    elapsed: float = 0.0,
    error: Optional[str] = None,
    error_msg: Optional[str] = None,
    note: Optional[str] = None,
    db_path: Optional[str] = None,
) -> None:
    """Record a single collection event in ``collection_log``.

    Signature matches the caller patterns used throughout
    ``simulation/collectors/legacy/group_*.py``::

        log_collection("A", "citydata_ppltn", "OK", rows_saved, elapsed=dt)
        log_collection("B", "getVilageFcst", "FAIL", error="응답 없음", elapsed=dt)
        log_collection("H", "HIRA_all", "SKIP", note="API key not configured")

    ``error``/``error_msg`` are aliases (caller convenience). ``note`` is
    promoted to ``error_msg`` when no explicit error is set so the single
    DB column still carries operator-visible context.
    """
    now = dt.datetime.now().isoformat()
    # Collapse the three context fields into one DB column.
    msg = error_msg if error_msg is not None else error
    if msg is None and note is not None:
        msg = note
    elif msg is not None and note is not None:
        msg = f"{msg} | note: {note}"

    try:
        rows_saved_int = int(rows_saved) if rows_saved is not None else 0
    except (TypeError, ValueError):
        rows_saved_int = 0
    try:
        elapsed_f = round(float(elapsed), 2) if elapsed is not None else 0.0
    except (TypeError, ValueError):
        elapsed_f = 0.0

    conn = safe_connect(db_path)
    try:
        insert_rows(
            "collection_log",
            [{
                "collected_at": now,
                "group_name": group,
                "api_name": api_name,
                "status": status,
                "rows_saved": rows_saved_int,
                "error_msg": msg,
                "elapsed_sec": elapsed_f,
            }],
            conn=conn,
        )
    finally:
        conn.close()
