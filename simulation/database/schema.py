"""
simulation.database.schema
==============================
DDL migration — adds tables required by RECOMMENDED_PIPELINE.md .

What this adds (all idempotent CREATE TABLE IF NOT EXISTS):
 * model_registry — PAPER_PRIMARY_11 freeze (SHA-256 snapshot)
 * run_ledger — end-to-end run provenance (seed / git / config hash)
 * scenario — named counterfactual / intervention scenarios
 * verifier_audit — @verify_before/@verify_after hook logs
 * rt_estimates — instantaneous Rt estimates (EpiEstim / RtEstimator)
 * nowcast_results — nowcast outputs (per-step)

Vintage/AS-OF hardening (ALTER TABLE additions):
 * weekly_disease.vintage_ts — 원본 게시 시점 (ASOF join anchor)
 * weekly_disease.revision_index — 같은 week_start 의 revision 번호
 * sentinel_influenza.vintage_ts — 동일
 * sentinel_influenza.revision_index

모든 ADD COLUMN 은 SQLite 에서 try/except 로 감싸 idempotent 하게 돌린다.

Usage:
 from simulation.database.schema import apply_schema_migration
 apply_schema_migration # idempotent

또는 init_db 이후 자동 호출:
 init_db → apply_schema_migration
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

from .config import DB_PATH
from .storage import safe_connect, transaction

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# DDL — new tables
# ══════════════════════════════════════════════════════════════════════════
SCHEMA_V22_SQL = """
-- ── Model registry (PAPER_PRIMARY_11 freeze + live coverage meta) ─────
CREATE TABLE IF NOT EXISTS model_registry (
    model_name         TEXT PRIMARY KEY,
    category           TEXT NOT NULL,          -- ts / linear / tree / dl / physics / meta / ensemble
    level              INTEGER NOT NULL,       -- ordering within category
    min_data           INTEGER NOT NULL,       -- min train weeks required
    is_paper_primary   INTEGER DEFAULT 0,      -- 1 = in PAPER_PRIMARY_11
    is_registered      INTEGER DEFAULT 1,      -- 1 = in live REGISTRY (verify_registry_coverage)
    requires_gpu       INTEGER DEFAULT 0,
    source_file        TEXT,                   -- simulation/models/... path
    source_sha256      TEXT,                   -- snapshot hash (SHA-256)
    registered_at      TEXT NOT NULL,
    frozen_at          TEXT,                   -- NULL until PAPER_PRIMARY freeze
    description        TEXT,
    UNIQUE(model_name)
);

CREATE INDEX IF NOT EXISTS idx_model_registry_category
    ON model_registry(category);
CREATE INDEX IF NOT EXISTS idx_model_registry_paper
    ON model_registry(is_paper_primary);

-- ── Run ledger (end-to-end run provenance) ──────────────────────────
CREATE TABLE IF NOT EXISTS run_ledger (
    run_id             TEXT PRIMARY KEY,       -- UUID or timestamp
    started_at         TEXT NOT NULL,
    finished_at        TEXT,
    git_commit         TEXT,
    git_dirty          INTEGER DEFAULT 0,
    seed               INTEGER,
    config_sha256      TEXT,                   -- pipeline config snapshot
    cli_args           TEXT,                   -- json
    scenario           TEXT,                   -- FK-ish to scenario.name
    status             TEXT DEFAULT 'running', -- running / ok / failed
    n_models           INTEGER,
    best_model         TEXT,
    best_metric_name   TEXT,
    best_metric_value  REAL,
    notes              TEXT
);

CREATE INDEX IF NOT EXISTS idx_run_ledger_started
    ON run_ledger(started_at);
CREATE INDEX IF NOT EXISTS idx_run_ledger_status
    ON run_ledger(status);

-- ── Scenario registry (counterfactual / intervention) ───────────────
CREATE TABLE IF NOT EXISTS scenario (
    scenario_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    name               TEXT NOT NULL UNIQUE,   -- 'baseline', 'sch_close_2w', ...
    description        TEXT,
    params_json        TEXT NOT NULL,          -- arbitrary scenario params
    created_at         TEXT NOT NULL,
    is_active          INTEGER DEFAULT 1
);

-- ── Verifier audit log (@verify_before/@verify_after) ───────────────
CREATE TABLE IF NOT EXISTS verifier_audit (
    audit_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                 TEXT NOT NULL,
    run_id             TEXT,                   -- FK-ish to run_ledger
    phase              TEXT NOT NULL,          -- phase1_data / phase4_ar / ...
    hook               TEXT NOT NULL,          -- before / after
    checker            TEXT NOT NULL,          -- leakage / ast / epi_validity / runtime
    status             TEXT NOT NULL,          -- ok / warn / fail
    details_json       TEXT,
    elapsed_ms         INTEGER
);

CREATE INDEX IF NOT EXISTS idx_verifier_audit_ts
    ON verifier_audit(ts);
CREATE INDEX IF NOT EXISTS idx_verifier_audit_run
    ON verifier_audit(run_id, phase);

-- ── Rt estimates (EpiEstim / RtEstimator) ───────────────────────────
CREATE TABLE IF NOT EXISTS rt_estimates (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    computed_at        TEXT NOT NULL,
    run_id             TEXT,
    week_start         TEXT NOT NULL,
    gu_nm              TEXT,                   -- NULL = 전체 서울
    disease_cd         TEXT NOT NULL,          -- 'ILI' for sentinel flu
    method             TEXT NOT NULL,          -- 'epiestim' / 'wallinga-teunis' / 'epidemia'
    rt_mean            REAL,
    rt_ci_low          REAL,
    rt_ci_high         REAL,
    window_days        INTEGER,
    si_mean            REAL,                   -- serial interval mean
    si_std             REAL,
    UNIQUE(week_start, gu_nm, disease_cd, method, window_days)
);

CREATE INDEX IF NOT EXISTS idx_rt_estimates_week
    ON rt_estimates(week_start, disease_cd);

-- ── Nowcast results (per-step output) ───────────────────────────────
CREATE TABLE IF NOT EXISTS nowcast_results (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    computed_at        TEXT NOT NULL,
    run_id             TEXT,
    week_start         TEXT NOT NULL,          -- target week
    vintage_ts         TEXT NOT NULL,          -- AS-OF anchor
    gu_nm              TEXT,
    disease_cd         TEXT NOT NULL,
    model_name         TEXT NOT NULL,
    y_hat              REAL,
    pi_low             REAL,
    pi_high            REAL,
    pi_alpha           REAL DEFAULT 0.1,
    UNIQUE(week_start, vintage_ts, gu_nm, disease_cd, model_name, pi_alpha)
);

CREATE INDEX IF NOT EXISTS idx_nowcast_week
    ON nowcast_results(week_start, disease_cd);
CREATE INDEX IF NOT EXISTS idx_nowcast_vintage
    ON nowcast_results(vintage_ts, model_name);
"""


# ══════════════════════════════════════════════════════════════════════════
# Collector-owned tables — historically created ad-hoc by individual
# collectors, but if you delete the DB and run `python -m simulation bootstrap`
# on an empty machine the collectors CRASH because `insert_rows()` targets
# a table that doesn't exist yet. This block restores the full set so
# `bootstrap` is truly reproducible from zero.
#
# Schema source: dumped 2026-04-24 from the live DB via
#     SELECT sql FROM sqlite_master WHERE name=?
# then normalized to `CREATE TABLE IF NOT EXISTS` (idempotent).
# ══════════════════════════════════════════════════════════════════════════
COLLECTOR_TABLES_SQL = """
-- ── Mobility & daily population (group_c, group_a) ────────────────────
CREATE TABLE IF NOT EXISTS daily_bus (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    use_dt       TEXT NOT NULL,
    route_id     TEXT,
    route_no     TEXT,
    station_id   TEXT,
    station_nm   TEXT NOT NULL,
    ride_cnt     INTEGER,
    alight_cnt   INTEGER,
    UNIQUE(use_dt, station_id, route_id)
);

CREATE TABLE IF NOT EXISTS daily_subway (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    use_dt       TEXT NOT NULL,
    line_num     TEXT,
    station_nm   TEXT NOT NULL,
    ride_pasgr   INTEGER,
    alight_pasgr INTEGER,
    UNIQUE(use_dt, station_nm, line_num)
);

CREATE TABLE IF NOT EXISTS monthly_bus_hourly (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    use_ym      TEXT NOT NULL,
    route_no    TEXT NOT NULL,
    route_nm    TEXT,
    station_id  TEXT,
    station_nm  TEXT NOT NULL,
    hour        INTEGER NOT NULL,
    ride_cnt    INTEGER,
    alight_cnt  INTEGER,
    UNIQUE(use_ym, route_no, station_id, hour)
);

CREATE TABLE IF NOT EXISTS monthly_subway_hourly (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    use_ym      TEXT NOT NULL,
    line_nm     TEXT NOT NULL,
    station_nm  TEXT NOT NULL,
    hour        INTEGER NOT NULL,
    ride_cnt    INTEGER,
    alight_cnt  INTEGER,
    UNIQUE(use_ym, line_nm, station_nm, hour)
);

CREATE TABLE IF NOT EXISTS daily_population_district (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    stdr_de         TEXT NOT NULL,
    signgu_code     TEXT NOT NULL,
    signgu_nm       TEXT,
    tot_livpop      REAL,
    day_livpop      REAL,
    night_livpop    REAL,
    inflow_livpop   REAL,
    move_livpop     REAL,
    UNIQUE(stdr_de, signgu_code)
);

CREATE TABLE IF NOT EXISTS daily_population_dong (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    stdr_de     TEXT NOT NULL,
    dong_code   TEXT NOT NULL,
    gu_code     TEXT,
    gu_nm       TEXT,
    tot_pop     REAL,
    male_pop    REAL,
    female_pop  REAL,
    pop_0_9     REAL,
    pop_10_19   REAL,
    pop_20_29   REAL,
    pop_30_39   REAL,
    pop_40_49   REAL,
    pop_50_59   REAL,
    pop_60_69   REAL,
    pop_70plus  REAL,
    UNIQUE(stdr_de, dong_code)
);

CREATE TABLE IF NOT EXISTS daily_population_gu_hourly (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stdr_de TEXT NOT NULL,
    gu_code TEXT NOT NULL,
    gu_nm TEXT NOT NULL,
    hour INTEGER NOT NULL,
    tot_pop REAL,
    male_pop REAL,
    female_pop REAL,
    pop_0_9 REAL,
    pop_10_19 REAL,
    pop_20_29 REAL,
    pop_30_39 REAL,
    pop_40_49 REAL,
    pop_50_59 REAL,
    pop_60_69 REAL,
    pop_70plus REAL
);

CREATE TABLE IF NOT EXISTS daily_population_hotspot (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    stdr_de              TEXT NOT NULL,
    area_cd              TEXT NOT NULL,
    area_nm              TEXT NOT NULL,
    gu_code              TEXT,
    gu_nm                TEXT,
    congestion           TEXT,
    ppltn_min            INTEGER,
    ppltn_max            INTEGER,
    ppltn_rate_0         REAL,
    ppltn_rate_10        REAL,
    ppltn_rate_20        REAL,
    ppltn_rate_30        REAL,
    ppltn_rate_40        REAL,
    ppltn_rate_50        REAL,
    ppltn_rate_60        REAL,
    ppltn_rate_70        REAL,
    male_ppltn_rate      REAL,
    female_ppltn_rate    REAL,
    resnt_ppltn_rate     REAL,
    non_resnt_ppltn_rate REAL,
    raw_json             TEXT,
    UNIQUE(stdr_de, area_cd)
);

-- ── Real-time environment (group_a) ────────────────────────────────────
CREATE TABLE IF NOT EXISTS rt_population (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    collected_at TEXT NOT NULL,
    area_cd      TEXT NOT NULL,
    area_nm      TEXT NOT NULL,
    congestion   TEXT,
    ppltn_min    INTEGER,
    ppltn_max    INTEGER,
    raw_json     TEXT
);

CREATE TABLE IF NOT EXISTS rt_population_forecast (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collected_at TEXT,
    date TEXT,
    area_nm TEXT,
    fcst_time TEXT,
    fcst_congest TEXT,
    fcst_ppltn_min REAL,
    fcst_ppltn_max REAL,
    area_cd TEXT
);

CREATE TABLE IF NOT EXISTS rt_air_quality (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    collected_at TEXT NOT NULL,
    source       TEXT NOT NULL,
    location_nm  TEXT NOT NULL,
    pm10         REAL,
    pm25         REAL,
    o3           REAL,
    no2          REAL,
    so2          REAL,
    co           REAL,
    khai_grade   INTEGER,
    raw_json     TEXT
);

CREATE TABLE IF NOT EXISTS rt_bike_status (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    collected_at    TEXT NOT NULL,
    total_stations  INTEGER,
    total_racks     INTEGER,
    total_bikes     INTEGER,
    avg_shared_pct  REAL
);

CREATE TABLE IF NOT EXISTS rt_sdot_env (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    collected_at TEXT NOT NULL,
    sensor_id    TEXT,
    cgg          TEXT,
    gu_code      TEXT,
    dong         TEXT,
    temperature  REAL,
    humidity     REAL,
    pm10         REAL,
    pm25         REAL,
    uv_index     REAL,
    noise        REAL,
    wind_speed   REAL,
    wind_dir     TEXT,
    raw_json     TEXT
);

-- ── ED / Hospital (group_q) ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ed_visits_symptom (
    collected_at TEXT,
    date TEXT,
    hospital_name TEXT,
    hospital_addr TEXT,
    bed_total INTEGER,
    bed_icu INTEGER,
    bed_operate INTEGER,
    bed_neuro INTEGER,
    bed_neonatal INTEGER,
    bed_general INTEGER,
    bed_internal INTEGER,
    ct_avail TEXT,
    mri_avail TEXT
);

CREATE TABLE IF NOT EXISTS hospitals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    collected_at    TEXT NOT NULL,
    ykiho           TEXT NOT NULL,
    inst_nm         TEXT,
    addr            TEXT,
    gu_nm           TEXT,
    clcd_nm         TEXT,
    bed_cnt         INTEGER,
    dr_cnt          INTEGER,
    tel             TEXT,
    lat             REAL,
    lng             REAL,
    UNIQUE(ykiho)
);

-- ── External signals (group_g, group_p) ────────────────────────────────
CREATE TABLE IF NOT EXISTS google_search_trends (
    collected_at TEXT,
    period TEXT,
    geo TEXT,
    keyword TEXT,
    interest REAL,
    group_idx INTEGER
);

CREATE TABLE IF NOT EXISTS pubmed_abstracts (
    collected_at TEXT,
    pmid INTEGER,
    title TEXT,
    abstract TEXT,
    journal TEXT,
    year INTEGER,
    mesh_terms TEXT,
    keywords TEXT
);

-- ── Vaccination / Employment (group_f, group_c) ────────────────────────
CREATE TABLE IF NOT EXISTS childhood_vaccination_rates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collected_at TEXT,
    ref_year INTEGER,
    region_code TEXT,
    region_nm TEXT,
    vaccine_code TEXT,
    vaccine_nm TEXT,
    sex TEXT,
    coverage_pct REAL,
    source_table TEXT,
    UNIQUE(ref_year, region_code, vaccine_code, sex)
);

CREATE TABLE IF NOT EXISTS employment_monthly (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    prd_de   TEXT NOT NULL,
    c1       TEXT NOT NULL,
    c1_nm    TEXT,
    itm_id   TEXT,
    itm_nm   TEXT,
    dt       REAL,
    UNIQUE(prd_de, c1, itm_id)
);

-- ── KOSIS (group_e_periodic / import_external) ─────────────────────────
CREATE TABLE IF NOT EXISTS kosis_age_district (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    collected_at TEXT NOT NULL,
    prd_de       TEXT NOT NULL,
    gu_code      TEXT NOT NULL,
    gu_nm        TEXT,
    age_group    TEXT NOT NULL,
    population   INTEGER,
    UNIQUE(prd_de, gu_code, age_group)
);

-- ── Disease catalog & mapping (maintain.py; retained from legacy) ──────
CREATE TABLE IF NOT EXISTS disease_catalog (
    disease_cd       TEXT PRIMARY KEY,
    disease_group    TEXT,
    disease_nm       TEXT NOT NULL,
    has_weekly       INTEGER DEFAULT 0,
    has_monthly      INTEGER DEFAULT 0,
    has_seoul_yearly INTEGER DEFAULT 0,
    has_age          INTEGER DEFAULT 0,
    has_gender       INTEGER DEFAULT 0,
    has_death        INTEGER DEFAULT 0,
    first_year       INTEGER,
    last_year        INTEGER,
    updated_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS disease_name_mapping (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    short_name TEXT UNIQUE,
    kdca_name TEXT,
    seoul_district_name TEXT,
    pdf_name TEXT,
    annual_report_name TEXT,
    kcd_code TEXT,
    grade TEXT,
    transmission_route TEXT,
    tier INTEGER,
    surveillance_type TEXT,
    has_data INTEGER DEFAULT 0
);

-- ── Seoul annual report — extra tables beyond the 3 in extract_pdf.py ──
CREATE TABLE IF NOT EXISTS seoul_annual_report_age (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collected_at TEXT NOT NULL,
    source TEXT DEFAULT '2024_감염병감시연보',
    disease_nm TEXT NOT NULL,
    year INTEGER NOT NULL,
    age_group TEXT NOT NULL,
    cases INTEGER,
    UNIQUE(disease_nm, year, age_group)
);

CREATE TABLE IF NOT EXISTS seoul_annual_report_gender (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collected_at TEXT NOT NULL,
    source TEXT DEFAULT '2024_감염병감시연보',
    disease_nm TEXT NOT NULL,
    year INTEGER NOT NULL,
    gender TEXT NOT NULL,
    cases INTEGER,
    UNIQUE(disease_nm, year, gender)
);

CREATE TABLE IF NOT EXISTS seoul_annual_report_infection_region (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collected_at TEXT NOT NULL,
    source TEXT DEFAULT '2024_감염병감시연보',
    disease_nm TEXT NOT NULL,
    year INTEGER NOT NULL,
    region TEXT NOT NULL,
    cases INTEGER,
    UNIQUE(disease_nm, year, region)
);

CREATE TABLE IF NOT EXISTS seoul_annual_report_patient_class (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collected_at TEXT NOT NULL,
    source TEXT DEFAULT '2024_감염병감시연보',
    disease_nm TEXT NOT NULL,
    year INTEGER NOT NULL,
    patient_class TEXT NOT NULL,
    cases INTEGER,
    UNIQUE(disease_nm, year, patient_class)
);

CREATE TABLE IF NOT EXISTS seoul_disease_district (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    collected_at TEXT NOT NULL,
    year         INTEGER NOT NULL,
    gu_code      TEXT NOT NULL,
    gu_nm        TEXT NOT NULL,
    disease_code TEXT NOT NULL,
    disease_nm   TEXT NOT NULL,
    cat_code     TEXT NOT NULL,
    category     TEXT NOT NULL,
    cases        INTEGER,
    UNIQUE(year, gu_code, disease_code, cat_code)
);

-- ── HIRA health claims (group_h_hira.py) ──────────────────────────────
CREATE TABLE IF NOT EXISTS hira_facility (
    kcd_code TEXT NOT NULL,
    kcd_name TEXT,
    ref_year INTEGER NOT NULL,
    facility_type TEXT,
    patient_count INTEGER,
    spec_count INTEGER,
    visit_days INTEGER,
    insup_brdn_amt INTEGER,
    rpe_tamt_amt INTEGER,
    collected_at TEXT,
    UNIQUE(kcd_code, ref_year, facility_type)
);

CREATE TABLE IF NOT EXISTS hira_gender_age (
    kcd_code TEXT NOT NULL,
    kcd_name TEXT,
    ref_year INTEGER NOT NULL,
    sex TEXT,
    age_group TEXT,
    patient_count INTEGER,
    spec_count INTEGER,
    visit_days INTEGER,
    insup_brdn_amt INTEGER,
    rpe_tamt_amt INTEGER,
    collected_at TEXT,
    UNIQUE(kcd_code, ref_year, sex, age_group)
);

CREATE TABLE IF NOT EXISTS hira_inpat_opat (
    kcd_code TEXT NOT NULL,
    kcd_name TEXT,
    ref_year INTEGER NOT NULL,
    sex TEXT,
    inpat_opat TEXT,
    patient_count INTEGER,
    spec_count INTEGER,
    visit_days INTEGER,
    insup_brdn_amt INTEGER,
    rpe_tamt_amt INTEGER,
    collected_at TEXT,
    UNIQUE(kcd_code, ref_year, sex, inpat_opat)
);

CREATE TABLE IF NOT EXISTS hira_region (
    kcd_code TEXT NOT NULL,
    kcd_name TEXT,
    ref_year INTEGER NOT NULL,
    region TEXT,
    patient_count INTEGER,
    spec_count INTEGER,
    visit_days INTEGER,
    insup_brdn_amt INTEGER,
    rpe_tamt_amt INTEGER,
    collected_at TEXT,
    UNIQUE(kcd_code, ref_year, region)
);

-- ── Sentinel surveillance extras (group_s_sentinel.py) ────────────────
CREATE TABLE IF NOT EXISTS sentinel_enterovirus (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collected_at TEXT,
    year INTEGER,
    week_no INTEGER,
    count INTEGER,
    UNIQUE(year, week_no)
);

CREATE TABLE IF NOT EXISTS sentinel_hfmd (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collected_at TEXT,
    year INTEGER,
    week_no INTEGER,
    week_label TEXT,
    rate REAL,
    UNIQUE(year, week_no)
);

CREATE TABLE IF NOT EXISTS sentinel_hfmdc (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collected_at TEXT,
    year INTEGER,
    count INTEGER,
    UNIQUE(year)
);

CREATE TABLE IF NOT EXISTS sentinel_intestinal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collected_at TEXT,
    year INTEGER,
    week_no INTEGER,
    pathogen_group TEXT,
    pathogen_nm TEXT,
    count INTEGER,
    UNIQUE(year, week_no, pathogen_nm)
);

CREATE TABLE IF NOT EXISTS sentinel_ophlgc (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collected_at TEXT,
    year INTEGER,
    week_no INTEGER,
    disease_nm TEXT,
    rate REAL,
    UNIQUE(year, week_no, disease_nm)
);

CREATE TABLE IF NOT EXISTS sentinel_sari (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collected_at TEXT,
    year INTEGER,
    week_no INTEGER,
    week_label TEXT,
    count INTEGER,
    UNIQUE(year, week_no)
);
"""


# ══════════════════════════════════════════════════════════════════════════
# ALTER TABLE additions (vintage 4-tuple hardening)
# ══════════════════════════════════════════════════════════════════════════
#: ADD COLUMN statements that are safe to re-run.  SQLite will raise
#: "duplicate column name" if already applied — we catch + ignore.
ALTER_V22: tuple[tuple[str, str], ...] = (
    ("weekly_disease", "ALTER TABLE weekly_disease ADD COLUMN vintage_ts TEXT"),
    ("weekly_disease", "ALTER TABLE weekly_disease ADD COLUMN revision_index INTEGER DEFAULT 0"),
    ("sentinel_influenza", "ALTER TABLE sentinel_influenza ADD COLUMN vintage_ts TEXT"),
    ("sentinel_influenza", "ALTER TABLE sentinel_influenza ADD COLUMN revision_index INTEGER DEFAULT 0"),
)


#: Tables created by this migration module.
#: Split into two logical groups so audit scripts can tell the origin.
MIGRATION_TABLES: frozenset[str] = frozenset({
    "model_registry", "run_ledger", "scenario",
    "verifier_audit", "rt_estimates", "nowcast_results",
})

#: Tables that were historically created ad-hoc by individual collectors.
#: We now ensure they exist up-front during migration so `bootstrap` from
#: an empty DB produces the full schema and downstream collectors never
#: hit a "no such table" at first insert.
COLLECTOR_OWNED_TABLES: frozenset[str] = frozenset({
    # Mobility & daily population (group_c, group_a)
    "daily_bus", "daily_subway",
    "monthly_bus_hourly", "monthly_subway_hourly",
    "daily_population_district", "daily_population_dong",
    "daily_population_gu_hourly", "daily_population_hotspot",
    # Real-time environment (group_a)
    "rt_population", "rt_population_forecast",
    "rt_air_quality", "rt_bike_status", "rt_sdot_env",
    # ED / Hospital (group_q)
    "ed_visits_symptom", "hospitals",
    # External signals (group_g, group_p)
    "google_search_trends", "pubmed_abstracts",
    # Vaccination / Employment (group_f, group_c)
    "childhood_vaccination_rates", "employment_monthly",
    # KOSIS age/district
    "kosis_age_district",
    # Disease catalog & mapping (maintain.py)
    "disease_catalog", "disease_name_mapping",
    # Seoul annual report extras
    "seoul_annual_report_age", "seoul_annual_report_gender",
    "seoul_annual_report_infection_region",
    "seoul_annual_report_patient_class",
    "seoul_disease_district",
    # HIRA claims (group_h_hira)
    "hira_facility", "hira_gender_age",
    "hira_inpat_opat", "hira_region",
    # Sentinel surveillance extras (group_s_sentinel)
    "sentinel_enterovirus", "sentinel_hfmd", "sentinel_hfmdc",
    "sentinel_intestinal", "sentinel_ophlgc", "sentinel_sari",
})

#: Full set contributed by this module (used by verify_schema via the
#: `simulation.database.__init__` union with storage.EXPECTED_TABLES).
EXPECTED_TABLES: frozenset[str] = MIGRATION_TABLES | COLLECTOR_OWNED_TABLES


# ══════════════════════════════════════════════════════════════════════════
# Migration driver
# ══════════════════════════════════════════════════════════════════════════
def _safe_add_column(conn: sqlite3.Connection, sql: str) -> bool:
    """Run ALTER TABLE ADD COLUMN; return True if applied, False if already present."""
    try:
        conn.execute(sql)
        return True
    except sqlite3.OperationalError as e:
        msg = str(e).lower()
        if "duplicate column name" in msg or "already exists" in msg:
            return False
        raise


def apply_schema_migration(
    db_path: Optional[str] = None,
    *,
    verbose: bool = True,
) -> dict:
    """Apply schema additions.

 Idempotent: re-running is safe. Returns a summary dict::

 {
 "tables_created": [...], # names from SCHEMA_V22_SQL
 "columns_added": [...], # (table, column) tuples
 "columns_skipped": [...], # already-present columns
 }
 """
    p = db_path or DB_PATH
    if not Path(p).exists():
        raise FileNotFoundError(
            f"DB not found at {p}. Run `python -m simulation bootstrap` first."
        )

    conn = safe_connect(p)
    tables_before = _list_tables(conn)
    added_cols: list[tuple[str, str]] = []
    skipped_cols: list[tuple[str, str]] = []

    try:
        with transaction(conn):
            conn.executescript(SCHEMA_V22_SQL)
            # Collector-owned tables — ensure they exist BEFORE any
            # collector starts inserting. Without this block, a
            # `python -m simulation bootstrap` on an empty machine would
            # leave 27 tables missing and the first collect pass would
            # crash with `no such table: daily_population_district`.
            conn.executescript(COLLECTOR_TABLES_SQL)
        # ALTER TABLEs — one statement per transaction because SQLite
        # can't rollback schema changes inside a multi-statement txn
        # in older versions.
        for table, stmt in ALTER_V22:
            col = stmt.rsplit("ADD COLUMN", 1)[-1].strip().split()[0]
            try:
                applied = _safe_add_column(conn, stmt)
                if applied:
                    added_cols.append((table, col))
                    conn.commit()
                else:
                    skipped_cols.append((table, col))
            except sqlite3.OperationalError as e:
                # Table doesn't exist yet (fresh DB before init_db) — warn & skip
                if "no such table" in str(e).lower():
                    log.warning("ALTER skipped (missing table): %s", table)
                    skipped_cols.append((table, col))
                else:
                    raise
        tables_after = _list_tables(conn)
        created = sorted(set(tables_after) - set(tables_before))
    finally:
        # Force-merge WAL sidecars into the main DB file so Windows doesn't
        # keep `-wal`/`-shm` handles open past close(). Silent on failure —
        # WAL may already be off, or DB may have been opened read-only.
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error:
            pass
        conn.close()

    summary = {
        "tables_created": created,
        "columns_added": added_cols,
        "columns_skipped": skipped_cols,
    }
    if verbose:
        log.info(
            "migration: %d tables created, %d columns added, %d skipped",
            len(created), len(added_cols), len(skipped_cols),
        )
        for t in created:
            log.info("  + table: %s", t)
        for t, c in added_cols:
            log.info("  + column: %s.%s", t, c)
    return summary


def _list_tables(conn: sqlite3.Connection) -> list[str]:
    return [
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    ]


__all__ = [
    "SCHEMA_V22_SQL",
    "ALTER_V22",
    "EXPECTED_TABLES",
    "apply_schema_migration",
]
