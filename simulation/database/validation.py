"""Schema ↔ Python dict key validation.

Catches the class of bugs where a collector builds `rows = [{"stdr_de": ..., ...}]`
and the target table has `stdr_date` instead → silent KeyError at first insert,
or worse, `insert_rows()` bypasses the mismatch because the first dict happens
to have all valid keys and a later one doesn't.

Usage:
    from simulation.database.validation import validate_rows
    validate_rows("daily_population_district", rows)   # raises ValueError on mismatch

Or wired into `insert_rows(..., validate=True)` when caller wants the check.

The validator is intentionally O(n × c) — cheap for batches up to 100k rows —
and returns detailed diagnostics so log messages pinpoint the offending column.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

from .config import DB_PATH

log = logging.getLogger(__name__)


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    """Return the set of column names for `table` from SQLite's schema."""
    cur = conn.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}


def validate_rows(
    table: str,
    rows: Iterable[dict],
    *,
    conn: Optional[sqlite3.Connection] = None,
    strict: bool = True,
    ignore_pk: bool = True,
) -> dict:
    """Check that every row dict's keys exist as columns in `table`.

    Args:
        table: SQLite table name.
        rows: iterable of dicts about to be inserted.
        conn: optional open SQLite connection. If None, opens DB_PATH briefly.
        strict: True → raise ValueError on any mismatch. False → only log.
        ignore_pk: True → allow rows to omit the auto-increment `id` PK.

    Returns:
        dict with `{"ok": bool, "unknown_keys": [...], "rows_checked": N,
                    "schema_cols": [...], "missing_required": [...]}`.

    Raises:
        ValueError if `strict` and any row has keys not in the schema.
    """
    close_after = conn is None
    conn = conn or sqlite3.connect(str(DB_PATH))
    try:
        schema_cols = _table_columns(conn, table)
        if not schema_cols:
            raise ValueError(f"table '{table}' not found in {DB_PATH}")

        pk_cols = {"id"} if ignore_pk else set()
        required_non_pk = schema_cols - pk_cols

        unknown: set[str] = set()
        seen: set[str] = set()
        n = 0
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"row #{n} not a dict: {type(row)}")
            n += 1
            for k in row.keys():
                seen.add(k)
                if k not in schema_cols:
                    unknown.add(k)

        report = {
            "ok": not unknown,
            "table": table,
            "rows_checked": n,
            "schema_cols": sorted(schema_cols),
            "unknown_keys": sorted(unknown),
            "missing_required_in_first_row": sorted(required_non_pk - seen) if n > 0 else [],
        }

        if unknown:
            msg = (
                f"schema mismatch for table '{table}': "
                f"rows contain keys {sorted(unknown)} that are NOT columns. "
                f"valid columns: {sorted(schema_cols)}"
            )
            if strict:
                raise ValueError(msg)
            log.warning(msg)
        return report
    finally:
        if close_after:
            conn.close()


def validate_collector_batch(
    table: str, rows: Iterable[dict], source: str = ""
) -> None:
    """Helper for collector code that wants "fail loud" behavior.

    Logs a friendly context line then delegates to `validate_rows(strict=True)`.
    """
    prefix = f"[{source}] " if source else ""
    rows = list(rows)
    if not rows:
        return
    report = validate_rows(table, rows, strict=False)
    if not report["ok"]:
        raise ValueError(
            f"{prefix}schema mismatch inserting into {table}: "
            f"unknown keys {report['unknown_keys']} "
            f"(valid schema has {len(report['schema_cols'])} cols)"
        )


def audit_all_tables(db_path: Optional[Path] = None) -> dict:
    """Sanity scan: list every table + column count. For reporting."""
    path = db_path or DB_PATH
    conn = sqlite3.connect(str(path))
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
    tables = [r[0] for r in cur.fetchall()]
    info = {}
    for t in tables:
        info[t] = sorted(_table_columns(conn, t))
    conn.close()
    return info
