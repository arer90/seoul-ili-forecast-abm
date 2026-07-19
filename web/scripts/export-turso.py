#!/usr/bin/env python
"""
export-turso.py — dump the operational tables needed by the Vercel
edge runtime into a libSQL-compatible SQL file.

Sprint 2026-05-06 V2 (Turso Free 5GB target): 25+ tables + raw aggregate.
- Full 12GB epi_real_seoul.db → ~500MB-1GB subset SQL
- Aggregate raw tables (monthly_bus_hourly 77M / monthly_subway_hourly 1.25M /
  daily_population_gu_hourly 1.79M) into hour×month×gu summaries
- Direct copy 18 paper / dashboard / ARIA evidence tables
- Skip raw + WAL pages (12GB → ~500MB-1GB)

Edit ``EXPORT_TABLES`` + ``AGGREGATE_QUERIES`` below to add/remove. The script
writes ``turso_seed.sql`` suitable for ``turso db shell <name> < seed.sql``.

Run from the project root:
    .venv/bin/python web/scripts/export-turso.py
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

# ────────────────────────────────────────────────────────────────────────
# Direct copy tables (~50MB total, sprint 2026-05-06 V2 expansion)
# ────────────────────────────────────────────────────────────────────────
EXPORT_TABLES: list[str] = [
    # paper §결과 evidence (existing)
    "weekly_disease",
    "forecast_runs",
    "forecast_points",
    "rt_history",
    "shap_delta_snapshots",
    "scenario_runs",
    "commuter_matrix",
    "model_registry",
    # paper §4.4 metapop coupling
    "poi_metadata",
    # paper §2.1 ILI multi-pathogen confounding (sprint 2026-05-06)
    "sentinel_influenza",
    "sentinel_ari",
    "sentinel_sari",
    "sentinel_hfmd",
    "sentinel_enterovirus",
    "sentinel_intestinal",
    "sentinel_ophlgc",
    # paper §1 HIRA evidence
    "hira_facility",
    "hira_region",
    "hira_gender_age",
    "hira_inpat_opat",
    # paper §4.5 vaccine 외생변수
    "vaccination_coverage",
    "childhood_vaccination_rates",
    # paper §결과 international evidence
    "overseas_ili",
    "who_flunet",
    "who_flunet_metadata",
    # paper §4.7 demographic strata
    "disease_age",
    "disease_gender",
    "disease_death",
    "kosis_age_district",
    "seoul_disease_district",
    "seoul_annual_report_age",
    "seoul_annual_report_district",
    "seoul_annual_report_gender",
    "seoul_annual_report_infection_region",
    "seoul_annual_report_monthly",
    "seoul_annual_report_patient_class",
    # 정직성 metadata (76 법정감염병 catalog)
    "disease_master",
    "disease_catalog",
    "disease_name_mapping",
    # 사용자 critique: 학교 / 병원
    "school_info_seoul",
    "school_closure_seoul",
    "hospitals",
    "emergency_room_availability",
    # 실시간 surveillance
    "rt_population",
    "rt_population_detail",
    "rt_population_forecast",
    "rt_subway_crowd",
    "rt_air_quality",
    "rt_road_traffic",
    "rt_sdot_env",
    "rt_bike_status",
    "rt_estimates",
    # 실시간 의료
    "ed_visits_symptom",
    # 인구밀집 hotspot (KCDC 명소)
    "daily_population_hotspot",
    # Digital epidemiology (paper §future work)
    "google_search_trends",
    # 통근 calibration
    "employment_residence",
    "employment_workplace",
    "employment_monthly",
    # KMA 날씨 forecast
    "weather_forecast",
    "weather_historical",
    # PubMed RAG (paper §5.7 ARIA Stage 6, 78MB but 가치 大)
    "pubmed_abstracts",
    # 시뮬레이션 catalog
    "kosis_disease_gender",
    "kosis_source_registry",
    # KCDC sentinel additional
    "sentinel_hfmdc",
]

# ────────────────────────────────────────────────────────────────────────
# Aggregate queries — raw 77M+ rows → ~10K aggregated subset
# (Sprint 2026-05-06 V2: monthly_bus_hourly 4.4GB → ~1MB)
# ────────────────────────────────────────────────────────────────────────
AGGREGATE_QUERIES: dict[str, str] = {
    # monthly_bus_hourly 77M rows → gu × hour × use_ym summary
    # NOTE: monthly_bus_hourly schema = (id, use_ym, route_no, route_nm, station_id, station_nm, hour, ride_cnt)
    # 우리는 station 의 location 정보 가 없어서 station_nm 으로만 group. 단 paper / dashboard 사용
    # 측면에서 hourly average per month 가 핵심. Aggregate to hour × use_ym.
    "monthly_bus_hourly_agg": """
        CREATE TABLE IF NOT EXISTS monthly_bus_hourly_agg (
            use_ym TEXT,
            hour INTEGER,
            total_ride_cnt INTEGER,
            station_count INTEGER,
            avg_ride_per_station REAL,
            PRIMARY KEY (use_ym, hour)
        );
        INSERT OR REPLACE INTO monthly_bus_hourly_agg
        SELECT
            use_ym,
            hour,
            SUM(ride_cnt) AS total_ride_cnt,
            COUNT(DISTINCT station_id) AS station_count,
            AVG(ride_cnt) AS avg_ride_per_station
        FROM monthly_bus_hourly
        GROUP BY use_ym, hour;
    """,
    # monthly_subway_hourly 1.25M rows → line × hour × use_ym summary
    "monthly_subway_hourly_agg": """
        CREATE TABLE IF NOT EXISTS monthly_subway_hourly_agg (
            use_ym TEXT,
            line_nm TEXT,
            hour INTEGER,
            total_ride INTEGER,
            total_alight INTEGER,
            station_count INTEGER,
            PRIMARY KEY (use_ym, line_nm, hour)
        );
        INSERT OR REPLACE INTO monthly_subway_hourly_agg
        SELECT
            use_ym,
            line_nm,
            hour,
            SUM(ride_cnt) AS total_ride,
            SUM(alight_cnt) AS total_alight,
            COUNT(DISTINCT station_nm) AS station_count
        FROM monthly_subway_hourly
        GROUP BY use_ym, line_nm, hour;
    """,
    # daily_population_gu_hourly 1.79M rows → gu × hour summary (long-term avg)
    "daily_population_gu_hourly_agg": """
        CREATE TABLE IF NOT EXISTS daily_population_gu_hourly_agg (
            gu_nm TEXT,
            hour INTEGER,
            avg_tot_pop REAL,
            avg_male_pop REAL,
            avg_female_pop REAL,
            sample_days INTEGER,
            PRIMARY KEY (gu_nm, hour)
        );
        INSERT OR REPLACE INTO daily_population_gu_hourly_agg
        SELECT
            gu_nm,
            hour,
            AVG(tot_pop) AS avg_tot_pop,
            AVG(male_pop) AS avg_male_pop,
            AVG(female_pop) AS avg_female_pop,
            COUNT(DISTINCT stdr_de) AS sample_days
        FROM daily_population_gu_hourly
        GROUP BY gu_nm, hour;
    """,
}


def dump_table(con: sqlite3.Connection, table: str) -> list[str]:
    """Return a list of CREATE + INSERT statements for ``table``.

    We use ``iterdump`` semantics but scoped to a single table so the
    output SQL is deterministic and doesn't include the full DB.
    """
    stmts: list[str] = []
    row = con.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    if row is None or row[0] is None:
        print(f"! skip {table}: not in source db", file=sys.stderr)
        return stmts
    stmts.append(f"DROP TABLE IF EXISTS {table};")
    stmts.append(row[0].rstrip(";") + ";")

    cols = [c[1] for c in con.execute(f"PRAGMA table_info({table})").fetchall()]
    col_list = ", ".join(cols)
    n_rows = 0
    for r in con.execute(f"SELECT {col_list} FROM {table}"):
        vals = ", ".join(_fmt(v) for v in r)
        stmts.append(f"INSERT INTO {table} ({col_list}) VALUES ({vals});")
        n_rows += 1
    print(f"- {table}: {n_rows} rows", file=sys.stderr)
    return stmts


def dump_aggregate(con: sqlite3.Connection, name: str, sql: str) -> list[str]:
    """Execute aggregate SQL on source DB, then dump the new aggregate
    table for export. Aggregate is materialized in source DB temporarily
    (drop after export to avoid bloat)."""
    stmts: list[str] = []
    # Execute aggregate (creates temp table in source)
    con.executescript(sql)
    # Dump the aggregate table
    stmts.extend(dump_table(con, name))
    # Drop temp aggregate from source (keep source clean)
    con.execute(f"DROP TABLE IF EXISTS {name}")
    return stmts


def _fmt(v: object) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, bytes):
        return "X'" + v.hex() + "'"
    s = str(v).replace("'", "''")
    return f"'{s}'"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        default="simulation/data/db/epi_real_seoul.db",
        help="source SQLite file",
    )
    ap.add_argument(
        "--out",
        default="web/scripts/turso_seed.sql",
        help="output SQL file",
    )
    ap.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="skip raw table aggregation (for fast testing)",
    )
    ap.add_argument(
        "--skip-pubmed",
        action="store_true",
        help="skip pubmed_abstracts (78MB full text — useful for size-tight tier)",
    )
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_file():
        print(f"! source db missing: {src}", file=sys.stderr)
        return 1

    # Open source as read-only to avoid lock contention with collector
    con = sqlite3.connect(f"file:{src}?mode=rwc", uri=True)
    # NOTE: aggregate creates temp tables, so rwc (read-write) needed.
    # Drop-after-export pattern in dump_aggregate.

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_tables = 0
    n_total_stmts = 0
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("-- Sprint 2026-05-06 V2 — Turso Free 5GB seed (subset).\n")
        fh.write("-- Generated by web/scripts/export-turso.py\n")
        fh.write("PRAGMA foreign_keys=OFF;\n")
        fh.write("BEGIN TRANSACTION;\n")

        # D3 (M7): vintage row → honest "data as of" badge instead of a silently
        # stale snapshot. Web reads web_data_vintage.generated_at / db_max_vintage.
        for s in _vintage_sql(con):
            fh.write(s + "\n")

        # Direct copy tables (sorted for determinism)
        export_list = list(EXPORT_TABLES)
        if args.skip_pubmed and "pubmed_abstracts" in export_list:
            export_list.remove("pubmed_abstracts")
            print("! skipping pubmed_abstracts (78MB)", file=sys.stderr)

        for table in export_list:
            stmts = dump_table(con, table)
            for s in stmts:
                fh.write(s + "\n")
            if stmts:
                n_tables += 1
                n_total_stmts += len(stmts)

        # Aggregate raw tables (raw 77M+ rows → ~10K aggregated)
        if not args.skip_aggregate:
            for name, sql in AGGREGATE_QUERIES.items():
                print(f"\n=== aggregating {name} ===", file=sys.stderr)
                stmts = dump_aggregate(con, name, sql)
                for s in stmts:
                    fh.write(s + "\n")
                if stmts:
                    n_tables += 1
                    n_total_stmts += len(stmts)

        fh.write("COMMIT;\n")

    out_size_mb = out_path.stat().st_size / 1024 / 1024
    print(
        f"\n=== summary ===\n"
        f"  tables: {n_tables}\n"
        f"  total stmts: {n_total_stmts:,}\n"
        f"  output: {out_path} ({out_size_mb:.1f} MB)\n"
        f"  → import: turso db shell <name> < {out_path}",
        file=sys.stderr,
    )
    return 0


def _vintage_sql(con: sqlite3.Connection) -> list[str]:
    """SQL for a single-row ``web_data_vintage`` table (D3/M7).

    Lets the web render an honest "data as of <date>" badge instead of silently
    serving a stale snapshot. ``generated_at`` = export time (ISO seconds);
    ``db_max_vintage`` = the latest surveillance week in the source DB
    (best-effort: weekly_disease.vintage_ts, else MAX(week), else "").

    Args:
        con: open sqlite3 connection to the source DB.

    Returns:
        SQL statement strings (DROP/CREATE/INSERT) to write into the dump.
    """
    from datetime import datetime

    generated_at = datetime.now().isoformat(timespec="seconds")
    db_max_vintage = ""
    for q in ("SELECT MAX(vintage_ts) FROM weekly_disease",
              "SELECT MAX(week) FROM weekly_disease"):
        try:
            row = con.execute(q).fetchone()
            if row and row[0] is not None:
                db_max_vintage = str(row[0])
                break
        except sqlite3.Error:
            continue
    esc = lambda s: s.replace("'", "''")  # noqa: E731
    return [
        "DROP TABLE IF EXISTS web_data_vintage;",
        "CREATE TABLE web_data_vintage (generated_at TEXT, db_max_vintage TEXT, source TEXT);",
        f"INSERT INTO web_data_vintage VALUES "
        f"('{esc(generated_at)}', '{esc(db_max_vintage)}', 'export-turso.py');",
    ]


if __name__ == "__main__":
    sys.exit(main())
