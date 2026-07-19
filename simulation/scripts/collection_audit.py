"""Correct classification of collection_log status codes.

The previous DB audit miscounted `SKIP` as `FAIL`, making the SPOP endpoints
look broken when they were actually intentionally skipping already-fresh dates.

Status code semantics (from BaseCollector + group_*.py):
    OK        — rows_saved > 0 (new data persisted)
    SKIP      — collector deliberately did not fetch (data already fresh, or
                target date is in "미게재" state — Seoul API returns INFO-200)
    EMPTY     — API returned 200/OK but zero rows (nothing available yet)
    FAIL/ERROR— HTTP error, exception, or 4xx/5xx after retry exhaustion

Usage:
    uv run python -m simulation.scripts.collection_audit

Output:
    Per-API summary + overall health. Exits 1 if any API has >50% FAIL ratio.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path


def main():
    from simulation.database import safe_connect  # ENGINEERING_PRINCIPLES.md §원칙 #3 — single writer
    conn = safe_connect()
    cur = conn.cursor()

    cur.execute("""
        SELECT api_name,
               MAX(collected_at) AS last_at,
               COUNT(*) AS n_calls,
               SUM(CASE WHEN status='OK'    THEN 1 ELSE 0 END) AS n_ok,
               SUM(CASE WHEN status='SKIP'  THEN 1 ELSE 0 END) AS n_skip,
               SUM(CASE WHEN status='EMPTY' THEN 1 ELSE 0 END) AS n_empty,
               SUM(CASE WHEN status IN ('FAIL','ERROR') THEN 1 ELSE 0 END) AS n_fail,
               SUM(COALESCE(rows_saved, 0)) AS total_rows
        FROM collection_log
        GROUP BY api_name
        ORDER BY last_at DESC
    """)
    rows = cur.fetchall()

    print()
    print("━" * 95)
    print(f"Collection audit — {len(rows)} unique APIs")
    print("━" * 95)
    print(f"{'api_name':<30s} {'last_ok':<22s} {'OK':>4s} {'SKP':>4s} {'EMP':>4s} {'FAIL':>4s}   {'rows':>12s}  health")
    print("─" * 95)

    hard_fail = []
    for r in rows:
        api, last, n_calls, n_ok, n_skip, n_empty, n_fail, total_rows = r
        total_rows = total_rows or 0
        fail_pct = (n_fail / n_calls * 100) if n_calls else 0
        if fail_pct > 50:
            mark = "✗"
            hard_fail.append((api, n_ok, n_fail, fail_pct))
        elif n_ok == 0 and n_fail == 0 and n_skip > 0:
            mark = "·"   # all SKIPs — probably idle (data fresh)
        elif n_fail > 0:
            mark = "⚠"
        else:
            mark = "✓"
        print(f"{api:<30s} {last[:19]:<22s} {n_ok:>4d} {n_skip:>4d} {n_empty:>4d} {n_fail:>4d}   "
              f"{total_rows:>12,d}  {mark}")

    print("─" * 95)
    print(f"Legend:  ✓ healthy    ⚠ some fails    · idle (all skips)    ✗ >50% fail (needs attention)")
    print()

    if hard_fail:
        print(f"Hard-fail APIs ({len(hard_fail)}):")
        for api, ok, fail, pct in hard_fail:
            print(f"  ✗ {api:<30s} {ok} OK / {fail} fail ({pct:.0f}%)")
        print()
    else:
        print("✓ No hard-fail APIs (all endpoints either healthy or idle-skip).")

    conn.close()
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
