"""DB / schema CLI commands — extracted from __main__.py.

Phase C2 partial (2026-05-12): 4 db-related subcommand handlers (db-init,
db-status, db-optimize, db-migrate-) moved here. Original __main__.py
imports these names and the dispatch table maps `"db-init": cmd_db_init`
unchanged.

Each handler:
    - Takes argparse Namespace `args`
    - No return value (CLI prints to stdout)
    - No exception handling — propagate to main() which has unified handler
"""
from __future__ import annotations


def cmd_db_init(args) -> None:
    """`python -m simulation db-init` — initialize DB schema (idempotent)."""
    from simulation.database import init_db
    conn = init_db(args.db_path)
    shapes = {}
    for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'"):
        shapes[row[0]] = True
    conn.close()
    print(f"DB initialized with {len(shapes)} tables at: {args.db_path or 'default'}")


def cmd_db_status(args) -> None:
    """`python -m simulation db-status` — show table row counts."""
    from simulation.collectors import print_status
    print_status(args.db_path)


def cmd_db_optimize(args) -> None:
    """`python -m simulation db-optimize` — checkpoint WAL + optional VACUUM."""
    from simulation.database import checkpoint_wal, vacuum_analyze
    print("[db-optimize] wal_checkpoint(TRUNCATE)")
    checkpoint_wal()
    if getattr(args, "vacuum", False):
        print("[db-optimize] VACUUM + ANALYZE")
        vacuum_analyze()
    print("✓ db-optimize complete")


def cmd_db_migrate_v22(args) -> None:
    """`python -m simulation db-migrate-` — apply schema additions (idempotent)."""
    from simulation.database import apply_schema_migration
    verbose = getattr(args, "verbose", True)
    print("[db-migrate-] applying schema additions...")
    summary = apply_schema_migration(verbose=verbose)
    print(f"  tables_created  : {len(summary['tables_created'])}")
    for t in summary["tables_created"]:
        print(f"    + {t}")
    print(f"  columns_added   : {len(summary['columns_added'])}")
    for tab, col in summary["columns_added"]:
        print(f"    + {tab}.{col}")
    print(f"  columns_skipped : {len(summary['columns_skipped'])}  "
          "(already present)")
    print("✓ db-migrate-complete")


__all__ = [
    "cmd_db_init",
    "cmd_db_status",
    "cmd_db_optimize",
    "cmd_db_migrate_v22",
]
