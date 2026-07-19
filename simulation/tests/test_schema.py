"""Smoke test: schema migration is idempotent + tables exist.

Windows-safe cleanup:
 - Use pytest's ``tmp_path`` fixture so the OS-level temp dir is deleted by
 pytest after the test (handles lingering SQLite handles).
 - Explicit ``gc.collect`` before unlink to drop any dangling connection
 objects that ``apply_schema_migration`` might have returned.
 - Best-effort removal of ``-wal`` / ``-shm`` sidecars (WAL mode).
"""
from __future__ import annotations

import gc
import sqlite3
from pathlib import Path


def _build_minimal_core_db(db_path: Path) -> None:
    """Create minimal core tables so ALTER TABLEs have targets."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("""
            CREATE TABLE weekly_disease (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                collected_at TEXT NOT NULL,
                week_start TEXT NOT NULL,
                disease_cd TEXT NOT NULL
            )""")
        conn.execute("""
            CREATE TABLE sentinel_influenza (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                collected_at TEXT,
                week_start TEXT
            )""")
        conn.commit()
    finally:
        conn.close()


def _cleanup_sqlite_sidecars(db_path: Path) -> None:
    """Force-drop any lingering SQLite handles and remove WAL sidecars.

    On Windows, SQLite's ``-wal`` / ``-shm`` files can hold the main DB file
    locked even after the primary connection is closed. Explicit gc + unlink
    avoids ``WinError 32`` in test teardown.
    """
    gc.collect()
    for suffix in ("-wal", "-shm", ""):
        p = db_path.with_name(db_path.name + suffix) if suffix else db_path
        try:
            p.unlink(missing_ok=True)
        except PermissionError:
            # Still locked — let pytest's tmp_path cleanup handle it later.
            pass


def test_schema_applies_once(tmp_path):
    from simulation.database.schema import apply_schema_migration

    db = tmp_path / "v22_apply_once.db"
    _build_minimal_core_db(db)
    try:
        summary = apply_schema_migration(str(db), verbose=False)
        created = set(summary["tables_created"])
        assert "model_registry" in created
        assert "run_ledger" in created
        assert "scenario" in created
        assert "verifier_audit" in created
        assert "rt_estimates" in created
        assert "nowcast_results" in created
        assert len(summary["columns_added"]) == 4
    finally:
        _cleanup_sqlite_sidecars(db)


def test_schema_is_idempotent(tmp_path):
    from simulation.database.schema import apply_schema_migration

    db = tmp_path / "v22_idempotent.db"
    _build_minimal_core_db(db)
    try:
        apply_schema_migration(str(db), verbose=False)
        summary2 = apply_schema_migration(str(db), verbose=False)
        # Second run: no tables newly created, all columns skipped
        assert summary2["tables_created"] == []
        assert len(summary2["columns_skipped"]) == 4
    finally:
        _cleanup_sqlite_sidecars(db)


def test_expected_tables_v22_included():
    from simulation.database import EXPECTED_TABLES
    from simulation.database.schema import EXPECTED_TABLES
    assert EXPECTED_TABLES.issubset(EXPECTED_TABLES)
