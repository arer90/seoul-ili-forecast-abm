# simulation.database -- Hardened SQLite storage + analytics overlay
from .config import COLLECT_DIR, DATA_DIR, DB_PATH
from .storage import (
    DatabaseCorruptError,
    EXPECTED_TABLES as _EXPECTED_TABLES_CORE,
    SCHEMA_SQL,
    bulk_insert,
    checkpoint_wal,
    get_conn,
    get_latest,
    get_table_shapes,
    init_db as _init_db_core,
    insert_rows,
    log_collection,
    query,
    quick_check,
    read_only_connect,
    safe_connect,
    transaction,
    tune_for_bulk_load,
    tune_for_normal,
    vacuum_analyze,
    verify_schema as _verify_schema_core,
)
from .schema import (
    EXPECTED_TABLES,
    apply_schema_migration,
)

# : union of core EXPECTED_TABLES + new tables so verify_schema
# doesn't classify model_registry/run_ledger/... as "extra".
EXPECTED_TABLES = frozenset(_EXPECTED_TABLES_CORE | EXPECTED_TABLES)


def init_db(db_path=None):
    """Initialize DB schema (core + ). Idempotent."""
    conn = _init_db_core(db_path)
    # apply_schema_migration opens its own connection (closes it too), so we
    # don't pass `conn` through — this keeps each migration step atomic.
    conn.close()
    apply_schema_migration(db_path, verbose=False)
    # Return a fresh connection to match prior init_db() contract.
    return safe_connect(db_path)


def verify_schema(conn=None, *, db_path=None):
    """verify_schema with tables included in EXPECTED_TABLES."""
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
# DuckDB analytical overlay -- lazy so `simulation.database` still imports
# cleanly on machines that haven't installed duckdb yet. `duckdb_conn` /
# `to_polars` raise a clear ImportError only when actually called.
from .analytics import (
    duckdb_conn,
    to_polars,
    to_pandas,
    to_arrow,
    explain as duckdb_explain,
    benchmark as duckdb_benchmark,
)

__all__ = [
    "DB_PATH", "DATA_DIR", "COLLECT_DIR",
    "DatabaseCorruptError", "EXPECTED_TABLES", "SCHEMA_SQL",
    "get_conn", "safe_connect", "read_only_connect", "quick_check", "transaction",
    "tune_for_bulk_load", "tune_for_normal",
    "init_db", "verify_schema",
    "insert_rows", "bulk_insert", "query", "get_latest", "get_table_shapes",
    "checkpoint_wal", "vacuum_analyze", "log_collection",
    # DuckDB analytical overlay (same epi_real_seoul.db, read-only ATTACH)
    "duckdb_conn", "to_polars", "to_pandas", "to_arrow",
    "duckdb_explain", "duckdb_benchmark",
]
