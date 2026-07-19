"""
DB Maintenance & Data Quality Module
=====================================
Fixes, populates disease_master, and validates ILI-relevant data integrity.

Usage:
    uv run -m simulation.database.maintain [--fix] [--report]
    uv run -m simulation maintain              # (CLI entry preferred)
"""
import sqlite3
import datetime
import logging
import argparse
from pathlib import Path

from .config import DB_PATH

log = logging.getLogger(__name__)

# ── Diseases known to have vaccines (for disease_master.vaccine_available) ──
VACCINE_DISEASES = {
    "인플루엔자", "홍역", "풍진", "수두", "백일해", "A형간염", "B형간염",
    "일본뇌염", "장티푸스", "폴리오", "디프테리아", "파상풍",
    "b형헤모필루스인플루엔자", "폐렴구균 감염증",
}

# ── Known newline-contaminated disease names from PDF extraction ──
NEWLINE_FIXES = [
    "A형간염\n백일해",
    "야콥병(vCJD)\n황열",
]

RENAME_FIXES = {
    "크로이츠펠트-야콥병(CJD) 및 변종크로이츠펠트-":
        "크로이츠펠트-야콥병(CJD) 및 변종크로이츠펠트-야콥병(vCJD)",
    "동물인플루엔자인체감염증": "동물인플루엔자 인체감염증",
}

ANNUAL_TABLES = [
    "seoul_annual_report_district",
    "seoul_annual_report_age",
    "seoul_annual_report_gender",
    "seoul_annual_report_monthly",
    "seoul_annual_report_infection_region",
    "seoul_annual_report_patient_class",
]

ILI_DATA_CHECKS = [
    ("sentinel_influenza", "ILI rate by age group (표본감시)"),
    ("sentinel_ari",       "ARI pathogen surveillance"),
    ("sentinel_sari",      "Severe ARI surveillance"),
    ("overseas_ili",       "Global ILI comparison data"),
    ("weekly_disease",     "Weekly notifiable diseases"),
    ("seoul_disease_district",        "Seoul district-level disease data"),
    ("seoul_annual_report_district",  "Annual report district data (from PDF)"),
    ("seoul_annual_report_monthly",   "Annual report monthly trends"),
    ("seoul_annual_report_age",       "Annual report age distribution"),
    ("hira_facility",      "HIRA healthcare facility data"),
]


def fix_disease_names(cur) -> int:
    """Fix newline-contaminated and mismatched disease names."""
    total = 0

    # Rename fixes
    for old_nm, new_nm in RENAME_FIXES.items():
        for tbl in ANNUAL_TABLES:
            try:
                cur.execute(
                    f"UPDATE [{tbl}] SET disease_nm = ? WHERE disease_nm = ?",
                    (new_nm, old_nm),
                )
                if cur.rowcount > 0:
                    log.info("  [%s] '%s' → '%s': %d rows", tbl, old_nm, new_nm, cur.rowcount)
                    total += cur.rowcount
            except Exception as e:
                log.warning("  [%s] rename error: %s", tbl, e)

    # Delete newline-contaminated rows (PDF extraction artifacts)
    for bad_nm in NEWLINE_FIXES:
        for tbl in ANNUAL_TABLES:
            try:
                cur.execute(f"SELECT COUNT(*) FROM [{tbl}] WHERE disease_nm = ?", (bad_nm,))
                cnt = cur.fetchone()[0]
                if cnt > 0:
                    cur.execute(f"DELETE FROM [{tbl}] WHERE disease_nm = ?", (bad_nm,))
                    log.info("  [%s] deleted %d invalid rows: %r", tbl, cnt, bad_nm)
                    total += cnt
            except Exception as e:
                log.warning("  [%s] delete error: %s", tbl, e)

    return total


def populate_disease_master(cur) -> int:
    """Populate disease_master from disease_name_mapping + disease_catalog."""
    # Build catalog lookup
    cur.execute(
        "SELECT disease_cd, disease_nm, disease_group, "
        "has_weekly, has_monthly, has_seoul_yearly FROM disease_catalog"
    )
    catalog_by_name = {}
    for r in cur.fetchall():
        catalog_by_name[r[1]] = r

    # Read mappings
    cur.execute(
        "SELECT short_name, kdca_name, grade, kcd_code, transmission_route, "
        "tier, surveillance_type, has_data FROM disease_name_mapping"
    )
    mapping_rows = cur.fetchall()

    cur.execute("DELETE FROM disease_master")
    inserted = 0

    for row in mapping_rows:
        short_name, kdca_name, grade, icd10, transmission, tier, surv_type, has_data = row
        cat = catalog_by_name.get(kdca_name) or catalog_by_name.get(short_name)

        disease_nm = kdca_name or short_name
        vaccine = 1 if short_name in VACCINE_DISEASES else 0

        flags_parts = []
        if surv_type:
            flags_parts.append(f"surveillance:{surv_type}")
        if tier:
            flags_parts.append(f"tier:{tier}")
        if has_data:
            flags_parts.append("has_data")
        if cat:
            if cat[3]:
                flags_parts.append("has_weekly")
            if cat[4]:
                flags_parts.append("has_monthly")
            if cat[5]:
                flags_parts.append("has_seoul_yearly")

        cur.execute(
            "INSERT OR REPLACE INTO disease_master "
            "(disease_cd, disease_nm, legal_grade, icd10, transmission_route, "
            "vaccine_available, year, flags) VALUES (?,?,?,?,?,?,2024,?)",
            (
                short_name,
                disease_nm,
                grade or "",
                icd10 or "",
                transmission or "",
                vaccine,
                "|".join(flags_parts),
            ),
        )
        inserted += 1

    return inserted


def data_coverage_report(cur) -> dict:
    """Check ILI-relevant data coverage and return summary."""
    report = {}
    for table, desc in ILI_DATA_CHECKS:
        try:
            cur.execute(f"SELECT COUNT(*) FROM [{table}]")
            cnt = cur.fetchone()[0]
            status = "OK" if cnt > 0 else "EMPTY"
            report[table] = {"count": cnt, "status": status, "desc": desc}
        except Exception:
            report[table] = {"count": 0, "status": "MISSING", "desc": desc}
    return report


# legacy cleanup: 구 스키마에서 남은 빈 테이블을 정리.
# CREATE IF NOT EXISTS 로는 이미 만들어진 테이블을 지우지 못하므로
# 명시적으로 DROP 해준다. 행이 있는 경우에만 skip 하고 경고를 찍는다.
LEGACY_EMPTY_TABLES = (
    "population_kosis",   # → kosis_age_district
    "school_closure",     # 이후 로더 제거
    "hospital_info",      # 이후 로더 제거
    "hira_claims",        # 이후 로더 제거
    "google_trends",      # 이후 로더 제거
    "subway_hourly",      # → monthly_subway_hourly
    "bus_hourly",         # → monthly_bus_hourly
)


def drop_legacy_empty_tables(cur) -> dict:
    """Drop v13 legacy tables that are empty. Keep any that still have rows."""
    dropped: list[str] = []
    kept: dict[str, int] = {}
    missing: list[str] = []

    for tbl in LEGACY_EMPTY_TABLES:
        try:
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (tbl,),
            )
            if cur.fetchone() is None:
                missing.append(tbl)
                continue

            cur.execute(f"SELECT COUNT(*) FROM [{tbl}]")
            cnt = cur.fetchone()[0]
            if cnt > 0:
                kept[tbl] = cnt
                log.warning(
                    "  [legacy] %s has %d rows — keeping (manual review required)",
                    tbl, cnt,
                )
                continue

            cur.execute(f"DROP TABLE [{tbl}]")
            dropped.append(tbl)
            log.info("  [legacy] dropped empty table: %s", tbl)
        except Exception as e:  # noqa: BLE001
            log.warning("  [legacy] %s drop error: %s", tbl, e)

    return {"dropped": dropped, "kept_nonempty": kept, "missing": missing}


def run_maintenance(fix: bool = True, report: bool = True) -> dict:
    """Run full DB maintenance cycle."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    result = {}

    if fix:
        log.info("── Fixing disease names ──")
        n_fixed = fix_disease_names(cur)
        result["names_fixed"] = n_fixed

        log.info("── Populating disease_master ──")
        n_master = populate_disease_master(cur)
        result["master_populated"] = n_master

        log.info("── Dropping legacy empty tables ──")
        legacy = drop_legacy_empty_tables(cur)
        result["legacy_cleanup"] = legacy

        conn.commit()
        log.info(
            "Fixes committed: %d name fixes, %d master entries, %d legacy tables dropped",
            n_fixed, n_master, len(legacy["dropped"]),
        )

    if report:
        log.info("── ILI Data Coverage Report ──")
        coverage = data_coverage_report(cur)
        for table, info in coverage.items():
            icon = "✓" if info["status"] == "OK" else "✗"
            log.info(
                "  %s %s: %s rows — %s", icon, table, f"{info['count']:,}", info["desc"]
            )
        result["coverage"] = coverage

    conn.close()
    return result


def main():
    parser = argparse.ArgumentParser(description="DB maintenance & data quality")
    parser.add_argument("--fix", action="store_true", default=True,
                        help="Apply data quality fixes (default: True)")
    parser.add_argument("--no-fix", action="store_true",
                        help="Skip data quality fixes")
    parser.add_argument("--report", action="store_true", default=True,
                        help="Print ILI data coverage report")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    fix = not args.no_fix
    run_maintenance(fix=fix, report=args.report)


if __name__ == "__main__":
    main()
