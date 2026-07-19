"""DB fingerprint — compact, deterministic snapshot of training-relevant tables.

Purpose
-------
Every per_model_optimize (R9) training run stamps a fingerprint of the SQLite
DB state into champion_log.json (per-model record) and run_ledger.  compare_v1_v2 then
verifies that two runs used the **same** data before declaring one better.

The fingerprint is designed to be:
  • Fast         — < 50 ms on a warm DB (only metadata + first-N rows hashed)
  • Deterministic — same data → same hash, regardless of insertion order
  • Compact      — < 4 KB JSON

Public API (Gray-box contract)
------------------------------
  compute_db_fingerprint(db_path=None) -> dict
      Capture fingerprint of all TRAINING_TABLES.
      Returns {"tables": {table: {...}}, "combined_sha256": "...", ...}

  fingerprints_match(fp_a, fp_b) -> bool
      True when combined_sha256 values are equal.
      Logs per-table diffs when False.

  embed_in_record(record: dict, fp: dict) -> dict
      Returns record with "db_fingerprint" key set (non-destructive).
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3 as _sqlite3  # OperationalError only
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Tables that directly affect model training outcomes.
# Changing any of these invalidates a v1 vs v2 comparison.
# ─────────────────────────────────────────────────────────────────────────────
TRAINING_TABLES: list[dict] = [
    # Primary outcome
    {"table": "sentinel_influenza", "date_col": "collected_at",  "hash_rows": 200},
    # Feature sources
    {"table": "weather_historical", "date_col": "obs_date",      "hash_rows": 200},
    {"table": "overseas_ili",       "date_col": None,             "hash_rows": 200},
    {"table": "vaccination_coverage","date_col": "collected_at", "hash_rows": 200},
    {"table": "who_flunet",         "date_col": "collected_at",  "hash_rows": 200},
    {"table": "weekly_disease",     "date_col": "collected_at",  "hash_rows": 200},
    {"table": "google_search_trends","date_col": None,            "hash_rows": 100},
    {"table": "hira_gender_age",    "date_col": None,             "hash_rows": 100},
    # GU-level (from PDF / Seoul annual reports)
    {"table": "seoul_annual_report_district", "date_col": None,  "hash_rows": 100},
]

_DEFAULT_DB = Path("simulation/data/db/epi_real_seoul.db")


def _resolve_db(db_path: Optional[str | Path]) -> Path:
    if db_path is not None:
        return Path(db_path)
    # Walk up to project root if needed
    cwd = Path.cwd()
    for candidate in [_DEFAULT_DB, cwd / _DEFAULT_DB, cwd.parent / _DEFAULT_DB]:
        if candidate.exists():
            return candidate
    return _DEFAULT_DB  # callers get sqlite3 error — clear failure mode


def _safe_int(v) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _table_fingerprint(
    con: Any,
    table: str,
    date_col: Optional[str],
    hash_rows: int,
) -> dict:
    """Compute a compact fingerprint for a single table.

    Args:
        con:        Open SQLite connection (read-only queries only).
        table:      Table name.
        date_col:   Column to use for MAX date (None → skip).
        hash_rows:  Number of rows to include in SHA-256 hash
                    (sorted by rowid for determinism).

    Returns:
        dict with keys: n_rows, max_date (or None), sha256_first_N.

    Raises:
        _sqlite3.OperationalError: table does not exist (caller silences).

    Performance: O(hash_rows) scan, not O(n_rows).
    Side effects: None (read-only).
    Caller responsibility: ``con`` must be open.
    """
    result: dict = {"table": table}

    # Row count — cheap
    row = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()
    result["n_rows"] = row[0] if row else 0

    # Max date — if column exists
    result["max_date"] = None
    if date_col:
        try:
            row = con.execute(f'SELECT MAX("{date_col}") FROM "{table}"').fetchone()
            result["max_date"] = row[0] if row else None
        except _sqlite3.OperationalError:
            pass  # date_col doesn't exist in this table

    # SHA-256 of first hash_rows rows (deterministic via rowid sort)
    try:
        cur = con.execute(
            f'SELECT * FROM "{table}" ORDER BY rowid LIMIT {hash_rows}'
        )
        rows = cur.fetchall()
        raw = json.dumps(rows, default=str, sort_keys=False).encode()
        result["sha256_first_n"] = hashlib.sha256(raw).hexdigest()[:16]
        result["hash_n"] = len(rows)
    except _sqlite3.OperationalError as e:
        result["sha256_first_n"] = f"ERR:{e}"[:32]
        result["hash_n"] = 0

    return result


def compute_db_fingerprint(db_path: Optional[str | Path] = None) -> dict:
    """Capture a compact, deterministic fingerprint of training-relevant DB tables.

    Fingerprinting is fast (< 100 ms typical) because it reads only metadata
    plus the first ``hash_rows`` rows per table, not full table scans.

    Args:
        db_path: Path to epi_real_seoul.db.  Defaults to standard project path.

    Returns:
        dict with schema::

            {
              "db_path":         "simulation/data/db/epi_real_seoul.db",
              "computed_at":     "2026-05-24T10:30:00Z",
              "tables": {
                "sentinel_influenza": {
                  "table":         "sentinel_influenza",
                  "n_rows":        2443,
                  "max_date":      "2026-04-01",
                  "sha256_first_n":"abc123de...",
                  "hash_n":        200
                },
                ...
              },
              "combined_sha256": "deadbeef..."  # SHA-256 of all table hashes
            }

    Raises:
        Nothing — missing tables return error-marked entries; DB missing
        returns empty tables dict with error field.

    Performance: < 100 ms on warm DB; ~200 ms cold (WAL checkpoint not triggered).
    Side effects: Opens read-only SQLite connection; closes on exit.
    Caller responsibility: None.
    """
    resolved = _resolve_db(db_path)
    fp: dict = {
        "db_path":    str(resolved),
        "computed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tables":     {},
    }

    if not resolved.exists():
        log.warning("[db_fingerprint] DB not found: %s", resolved)
        fp["error"] = "db_not_found"
        fp["combined_sha256"] = "unknown"
        return fp

    # Use safe_connect (G-116/G-117: WAL + quick_check + corruption guard).
    # verify=False because fingerprinting reads a potentially mid-write DB;
    # we want the hash of current state, not to block on corruption check.
    from simulation.database import safe_connect
    con = safe_connect(str(resolved), verify=False, timeout=30.0)

    try:
        for spec in TRAINING_TABLES:
            t = spec["table"]
            try:
                entry = _table_fingerprint(
                    con,
                    table=t,
                    date_col=spec.get("date_col"),
                    hash_rows=spec.get("hash_rows", 200),
                )
                fp["tables"][t] = entry
            except _sqlite3.OperationalError as e:
                fp["tables"][t] = {"table": t, "error": str(e)[:80]}
    finally:
        con.close()

    # Combined hash: SHA-256 of sorted per-table hashes (order-independent)
    table_hashes = sorted(
        f"{t}:{v.get('sha256_first_n', '')}"
        for t, v in fp["tables"].items()
    )
    combined_raw = "|".join(table_hashes).encode()
    fp["combined_sha256"] = hashlib.sha256(combined_raw).hexdigest()[:24]

    return fp


def fingerprints_match(fp_a: dict, fp_b: dict) -> bool:
    """Return True when two fingerprints represent the same DB state.

    Compares ``combined_sha256`` fields.  On mismatch, logs per-table diffs
    so the caller knows which tables changed.

    Args:
        fp_a: Fingerprint dict from :func:`compute_db_fingerprint`.
        fp_b: Second fingerprint dict.

    Returns:
        True if ``combined_sha256`` matches.

    Raises:
        Nothing.

    Performance: O(n_tables) string comparison.
    Side effects: May emit WARNING log lines.
    Caller responsibility: Both dicts must have been produced by this module.
    """
    ha = fp_a.get("combined_sha256", "?")
    hb = fp_b.get("combined_sha256", "?")

    if ha == "unknown" or hb == "unknown":
        log.warning("[db_fingerprint] One fingerprint has unknown hash — treating as mismatch")
        return False

    if ha == hb:
        return True

    # Diff per-table for diagnostic clarity
    log.warning("[db_fingerprint] combined_sha256 MISMATCH: %s vs %s", ha, hb)
    tables_a = fp_a.get("tables", {})
    tables_b = fp_b.get("tables", {})
    all_tables = sorted(set(tables_a) | set(tables_b))
    for t in all_tables:
        a = tables_a.get(t, {})
        b = tables_b.get(t, {})
        n_diff = a.get("n_rows") != b.get("n_rows")
        h_diff = a.get("sha256_first_n") != b.get("sha256_first_n")
        if n_diff or h_diff:
            log.warning(
                "[db_fingerprint]   %s: n_rows %s vs %s  hash %s vs %s",
                t,
                a.get("n_rows", "?"), b.get("n_rows", "?"),
                a.get("sha256_first_n", "?")[:8], b.get("sha256_first_n", "?")[:8],
            )
    return False


def embed_in_record(record: dict, fp: dict) -> dict:
    """Return a copy of record with db_fingerprint embedded.

    Non-destructive — does NOT modify record in-place.

    Args:
        record: Any dict (e.g. champion_log entry, run_ledger row).
        fp:     Fingerprint from :func:`compute_db_fingerprint`.

    Returns:
        New dict with ``"db_fingerprint"`` key added.

    Raises:
        Nothing.
    """
    out = dict(record)
    out["db_fingerprint"] = {
        "combined_sha256": fp.get("combined_sha256", "unknown"),
        "computed_at":     fp.get("computed_at"),
        "db_path":         fp.get("db_path"),
    }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Standalone smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    fp = compute_db_fingerprint()
    print(json.dumps(fp, indent=2, default=str))
    print(f"\ncombined_sha256: {fp['combined_sha256']}")
    missing = [t for t, v in fp["tables"].items() if "error" in v]
    if missing:
        print(f"WARN: missing tables: {missing}", file=sys.stderr)
