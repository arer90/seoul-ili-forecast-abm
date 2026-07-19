# simulation/collectors/storage.py
# Shim so legacy collectors can do: from ..storage import get_conn, insert_rows, ...
# Re-exports from simulation.database.storage + adds save_csv

import csv
from datetime import datetime
from pathlib import Path

from simulation.database.storage import (
    get_conn, init_db, insert_rows, query, get_latest,
    get_table_shapes, log_collection,
)
from simulation.database.config import COLLECT_DIR, DB_PATH


def refresh_disease_catalog() -> int:
    """Rebuild disease_catalog from collected disease tables."""
    import sqlite3  # row_factory 참조용
    # : safe_connect 로 일원화
    from simulation.database import safe_connect
    conn = safe_connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    now = datetime.now().isoformat(timespec="seconds")
    catalog = {}

    def merge_row(disease_cd, disease_group, disease_nm, flags):
        if not disease_cd or not disease_nm:
            return
        entry = catalog.setdefault(disease_cd, {
            "disease_cd": disease_cd,
            "disease_group": disease_group or "",
            "disease_nm": disease_nm,
            "has_weekly": 0, "has_monthly": 0, "has_seoul_yearly": 0,
            "first_year": 9999, "last_year": 0,
            "updated_at": now,
        })
        for k, v in flags.items():
            if k in ("first_year",):
                entry[k] = min(entry[k], v)
            elif k in ("last_year",):
                entry[k] = max(entry[k], v)
            else:
                entry[k] = max(entry.get(k, 0), v)

    # Scan weekly_disease
    try:
        for r in conn.execute(
            "SELECT DISTINCT disease_cd, disease_group, disease_nm, year FROM weekly_disease"
        ):
            yr = r["year"] or 0
            merge_row(r["disease_cd"], r["disease_group"], r["disease_nm"],
                      {"has_weekly": 1, "first_year": yr, "last_year": yr})
    except Exception:
        pass

    rows = list(catalog.values())
    if rows:
        insert_rows("disease_catalog", rows, replace=True)
    conn.close()
    return len(rows)


def save_csv(table: str, rows: list[dict], date_str: str = None,
             overwrite: bool = False):
    """Save rows to CSV: data/collected/{table}/{YYYYMMDD}.csv"""
    if not rows:
        return
    ds = date_str or datetime.now().strftime("%Y%m%d")
    out_dir = COLLECT_DIR / table
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ds}.csv"
    fieldnames = list(rows[0].keys())
    mode = "w" if overwrite or not out_path.exists() else "a"
    with open(out_path, mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)