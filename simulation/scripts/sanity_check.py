"""Periodic sanity check — catches the class of bugs where variables /
connections / schema drift silently and surface only at runtime.

Run this before any full training or collection pass:
    uv run python -m simulation.scripts.sanity_check

Checks performed:
  1. DB health            : quick_check, verify_schema
  2. API key loaded       : required keys present in KEYS; warn on missing
  3. Status vocabulary    : every status in collection_log is in known set
  4. Storage paths        : MPH_OUTPUT_ROOT resolves, dirs writable
  5. ctypes / native libs : seir_core.dylib loadable, seir_core wheel imported
  6. External tooling     : redis pingable, mlflow tracking URI creatable

Exits non-zero if any check has a CRITICAL failure. WARNs for soft issues.
"""
from __future__ import annotations

import sys
from pathlib import Path


def main():
    fails = 0
    warns = 0

    def ok(msg):   print(f"  ✓  {msg}")
    def warn(msg):
        nonlocal warns
        warns += 1
        print(f"  ⚠  {msg}")
    def fail(msg):
        nonlocal fails
        fails += 1
        print(f"  ✗  {msg}")

    print()
    print("━" * 72)
    print("  Sanity check — catches variable / connection / schema drift")
    print("━" * 72)

    # ── 1. DB health ───────────────────────────────────────────────────
    print("\n[1/6] DB health")
    try:
        from simulation.database import safe_connect, quick_check, verify_schema
        from simulation.database.config import DB_PATH
        db_path = Path(str(DB_PATH))  # DB_PATH may be str or Path
        if not db_path.exists():
            fail(f"DB missing: {db_path}")
        else:
            qc = quick_check(str(db_path))
            (ok if qc == "ok" else fail)(f"quick_check: {qc}")
            vs = verify_schema()
            if vs.get("ok"):
                ok(f"schema ok, missing=[]")
            else:
                fail(f"schema.missing: {vs.get('missing')}")
    except Exception as e:
        fail(f"DB check crashed: {type(e).__name__}: {e}")

    # ── 2. API keys ────────────────────────────────────────────────────
    print("\n[2/6] API keys")
    try:
        from simulation.database.config import KEYS
        required = ["seoul_general", "seoul_general2", "kma_hub",
                    "kosis", "neis", "data_go_kr"]
        missing = [k for k in required if k not in KEYS or not KEYS[k]]
        if missing:
            warn(f"KEYS missing: {missing} — collectors for these sources will FAIL")
        else:
            ok(f"all {len(required)} required keys present")
        # Extra keys (unknown label fallbacks) worth surfacing
        extra = [k for k in KEYS if k.startswith("일반") or k.startswith("번호")]
        if extra:
            warn(f"un-canonicalized labels in KEYS: {extra} — add to _KEY_LABEL_MAP")
    except Exception as e:
        fail(f"KEYS check crashed: {type(e).__name__}: {e}")

    # ── 2b. Schema completeness (bootstrap reproducibility) ─────────────
    print("\n[2b/6] Schema completeness — CREATE TABLE for every expected table")
    try:
        import os, re
        from simulation.database.storage import EXPECTED_TABLES as CORE
        from simulation.database.schema import (
            MIGRATION_TABLES, COLLECTOR_OWNED_TABLES, EXPECTED_TABLES as MIG_ALL,
        )
        all_expected = frozenset(CORE | MIG_ALL)
        created_in_code = set()
        for root, dirs, files in os.walk("simulation"):
            dirs[:] = [d for d in dirs if d not in (
                "__pycache__", "_past", "_root_legacy", "_thesis_archive",
                "_archive", "_legacy", "_sandbox",
            )]
            for fn in files:
                if fn.endswith((".py", ".sql")):
                    try:
                        c = open(os.path.join(root, fn), encoding="utf-8").read()
                        for t in re.findall(
                            r"CREATE TABLE(?:\s+IF NOT EXISTS)?\s+([a-z_0-9]+)",
                            c, re.IGNORECASE,
                        ):
                            created_in_code.add(t.lower())
                    except Exception:
                        pass
        orphaned = sorted(all_expected - created_in_code)
        if orphaned:
            fail(f"{len(orphaned)} expected tables have NO CREATE TABLE statement "
                 f"in active code — bootstrap from empty DB will crash on first insert: "
                 f"{orphaned[:6]}{'...' if len(orphaned) > 6 else ''}")
        else:
            ok(f"all {len(all_expected)} expected tables have CREATE TABLE "
               f"statements — bootstrap is reproducible")
    except Exception as e:
        fail(f"Schema completeness check crashed: {type(e).__name__}: {e}")

    # ── 3. Status vocabulary ────────────────────────────────────────────
    print("\n[3/6] Status vocabulary in collection_log")
    try:
        from simulation.collectors.status import KNOWN_STATUSES, Status
        from simulation.database import safe_connect  # ENGINEERING_PRINCIPLES.md §원칙 #3 — single writer
        conn = safe_connect()
        cur = conn.execute("SELECT DISTINCT status FROM collection_log")
        in_db = {r[0] for r in cur.fetchall()}
        conn.close()
        unknown = in_db - KNOWN_STATUSES
        if unknown:
            warn(f"DB has un-documented status values: {unknown}. "
                 f"Add them to simulation/collectors/status.py:Status or "
                 f"migrate collectors to use the enum.")
        else:
            ok(f"all {len(in_db)} distinct statuses recognized: {sorted(in_db)}")
    except Exception as e:
        fail(f"Status check crashed: {type(e).__name__}: {e}")

    # ── 4. Storage paths ───────────────────────────────────────────────
    print("\n[4/6] Storage paths")
    try:
        from simulation.utils.paths import get_output_root, get_results_dir, get_models_pt_dir
        import os
        root = get_output_root()
        results = get_results_dir()
        models = get_models_pt_dir()
        for label, p in [("output_root", root), ("results", results), ("models_pt", models)]:
            if not p.exists():
                fail(f"{label} does not exist: {p}")
            elif not os.access(p, os.W_OK):
                fail(f"{label} not writable: {p}")
            else:
                ok(f"{label}: {p}")
        from simulation.config_global import GLOBAL as _GCFG  # SSOT (2026-05-28)
        env_override = _GCFG.paths.output_root
        if env_override:
            ok(f"MPH_OUTPUT_ROOT active: {env_override}")
        else:
            ok(f"MPH_OUTPUT_ROOT not set — using project-local defaults")
    except Exception as e:
        fail(f"Paths check crashed: {type(e).__name__}: {e}")

    # ── 5. Native libs ─────────────────────────────────────────────────
    print("\n[5/6] Native libs + backends")
    try:
        from simulation.sim.stepper import HAS_NUMBA, HAS_C_BACKEND, HAS_RUST_BACKEND
        (ok if HAS_NUMBA else warn)(f"Numba: {HAS_NUMBA}")
        (ok if HAS_C_BACKEND else warn)(
            f"C backend: {HAS_C_BACKEND} "
            f"(build via `bash simulation/c/build.sh` if False)"
        )
        (ok if HAS_RUST_BACKEND else warn)(
            f"Rust backend: {HAS_RUST_BACKEND} "
            f"(build via `cd simulation/rust && maturin develop --release` if False)"
        )
    except Exception as e:
        fail(f"Backends check crashed: {type(e).__name__}: {e}")

    # ── 6. External optional services ──────────────────────────────────
    print("\n[6/6] Optional services (Redis, MLflow, RAG)")
    try:
        from simulation.cache import get_cache
        c = get_cache()
        backend = c.backend()
        if backend == "redis":
            ok(f"Redis reachable")
        else:
            ok(f"Redis fallback: {backend} (set REDIS_URL to use a real server)")
    except Exception as e:
        warn(f"Cache check: {type(e).__name__}: {e}")

    try:
        from simulation.tracking import tracking_info
        info = tracking_info()
        if info["mlflow_available"]:
            ok(f"MLflow {info['mlflow_version']}, backend {info['backend_path']}")
        else:
            warn(f"MLflow unavailable")
    except Exception as e:
        warn(f"MLflow check: {type(e).__name__}: {e}")

    try:
        from simulation.server.rag import rag_info
        info = rag_info()
        if info["lancedb_available"] and info["table_exists"]:
            ok(f"RAG index ready at {info['index_dir']}")
        elif info["lancedb_available"]:
            warn(f"LanceDB installed but index not built — run `from simulation.server.rag import build_index; build_index()`")
        else:
            warn(f"LanceDB unavailable — MCP literature_rag will use static fallback")
    except Exception as e:
        warn(f"RAG check: {type(e).__name__}: {e}")

    print()
    print("━" * 72)
    print(f"  Summary: {fails} failures, {warns} warnings")
    if fails:
        print(f"  ✗ FAIL — fix the {fails} issues above before running collection or training.")
        return 1
    elif warns:
        print(f"  ⚠ PASS with warnings — system works, but consider addressing the {warns} soft issues.")
    else:
        print(f"  ✓ ALL GREEN — every layer verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
