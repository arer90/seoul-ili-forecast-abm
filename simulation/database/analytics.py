"""
simulation.database.analytics -- DuckDB analytical overlay on SQLite
=====================================================================

**Why this exists**
-------------------
SQLite is excellent for the ingestion side of this project:

* single-file, zero-config, WAL durability
* ``safe_connect`` + ``quick_check`` + ``verify_schema`` already hardened
* ``bulk_insert`` pushes >100k rows/s with the aggressive PRAGMAs

But SQLite is a row-store and its optimizer is weak for analytical queries
(large GROUP BYs, multi-table joins, window functions). The feature engine
pulls 35 loaders, each hitting 5–20 million row aggregations.

**DuckDB** is a columnar OLAP engine in a single process with:

* 10–100x faster analytical queries than SQLite on the same data
* Native Polars / Pandas / Arrow zero-copy round-trips
* Can ``ATTACH`` a SQLite file read-only and query it directly — no copy,
  no schema duplication, no sync. Writes still go through the SQLite path.

The rule is simple:

    **Writes → SQLite (safe_connect).**
    **Analytics / feature engineering → analytics.duckdb_conn().**

Both point at the same ``epi_real_seoul.db`` file.

Example
-------
>>> from simulation.database.analytics import duckdb_conn, to_polars
>>> with duckdb_conn() as con:
...     df = con.execute('''
...         SELECT year, disease_nm, SUM(cases) AS total
...         FROM epi.seoul_annual_report_district
...         GROUP BY year, disease_nm
...         ORDER BY year, total DESC
...     ''').pl()
>>> df.shape
(576, 3)

The ``epi.`` prefix comes from the ``ATTACH ... AS epi`` below.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from .config import DB_PATH

log = logging.getLogger(__name__)

#: Schema alias used inside DuckDB. Always prefix SQLite tables with ``epi.``.
ATTACH_ALIAS = "epi"


def _import_duckdb():
    try:
        import duckdb  # noqa
        return duckdb
    except ImportError as e:
        raise ImportError(
            "duckdb is not installed. Run: uv pip install duckdb\n"
            "DuckDB is the recommended analytical engine for this project — "
            "see simulation/database/analytics.py for rationale."
        ) from e


@contextmanager
def duckdb_conn(
    sqlite_path: Optional[str] = None,
    *,
    read_only: bool = True,
    memory_limit: str = "4GB",
    threads: int = 4,
) -> Iterator["duckdb.DuckDBPyConnection"]:  # type: ignore[name-defined]
    """Open an in-memory DuckDB connected to the project SQLite file.

    Parameters
    ----------
    sqlite_path
        Path to ``epi_real_seoul.db`` (default: :data:`DB_PATH`).
    read_only
        When True (default), SQLite is attached read-only so analytical
        queries can never corrupt the ingestion DB. Set False only when
        writing back CTAS results.
    memory_limit, threads
        DuckDB resource caps. Defaults keep the process under ~4 GB RAM
        and 4 cores so the feature engine can run alongside training.
    """
    duckdb = _import_duckdb()
    path = sqlite_path or DB_PATH
    if not Path(path).exists():
        raise FileNotFoundError(f"SQLite DB not found: {path}")

    con = duckdb.connect(":memory:")
    try:
        con.execute(f"SET memory_limit='{memory_limit}'")
        con.execute(f"SET threads={threads}")
        con.execute("INSTALL sqlite; LOAD sqlite;")
        mode = "READ_ONLY" if read_only else "READ_WRITE"
        con.execute(
            f"ATTACH '{path}' AS {ATTACH_ALIAS} (TYPE SQLITE, {mode})"
        )
        yield con
    finally:
        con.close()


def to_polars(sql: str, sqlite_path: Optional[str] = None):
    """Execute an analytical query and return a Polars DataFrame."""
    with duckdb_conn(sqlite_path) as con:
        return con.execute(sql).pl()


def to_pandas(sql: str, sqlite_path: Optional[str] = None):
    """Execute an analytical query and return a Pandas DataFrame."""
    with duckdb_conn(sqlite_path) as con:
        return con.execute(sql).df()


def to_arrow(sql: str, sqlite_path: Optional[str] = None):
    """Execute an analytical query and return an Apache Arrow Table."""
    with duckdb_conn(sqlite_path) as con:
        return con.execute(sql).arrow()


def explain(sql: str, sqlite_path: Optional[str] = None) -> str:
    """Return DuckDB's query plan for ``sql`` (debugging)."""
    with duckdb_conn(sqlite_path) as con:
        rows = con.execute(f"EXPLAIN {sql}").fetchall()
        return "\n".join(r[1] if len(r) > 1 else str(r[0]) for r in rows)


def benchmark(sql: str, n: int = 3, sqlite_path: Optional[str] = None) -> dict:
    """Time a query through DuckDB (columnar) vs raw SQLite (row-store)."""
    import sqlite3
    import time

    path = sqlite_path or DB_PATH
    duckdb_times, sqlite_times = [], []
    for _ in range(n):
        t0 = time.perf_counter()
        with duckdb_conn(path) as con:
            con.execute(sql).fetchall()
        duckdb_times.append(time.perf_counter() - t0)

        # raw sqlite — need to rewrite "epi." prefix
        sql_sqlite = sql.replace(f"{ATTACH_ALIAS}.", "")
        t0 = time.perf_counter()
        conn = sqlite3.connect(path)
        try:
            conn.execute(sql_sqlite).fetchall()
        finally:
            conn.close()
        sqlite_times.append(time.perf_counter() - t0)

    d = sum(duckdb_times) / n
    s = sum(sqlite_times) / n
    return {
        "duckdb_sec": round(d, 4),
        "sqlite_sec": round(s, 4),
        "speedup": round(s / d, 2) if d > 0 else float("inf"),
    }
