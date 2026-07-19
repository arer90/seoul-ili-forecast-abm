"""D7 (M7 SCI-grade): per-run reproducibility manifest.

Seed-fixing alone is not reproducibility — a replication needs the
**environment + data vintage + config** triple. This emits ``run_manifest.json``
into the run's output dir at run start: git SHA, frozen-deps hash, resolved seed,
DB data vintage, and the key ``MPH_*`` env vars. It populates ``config_sha256``
(the schema column that existed but was never written) and is surfaced by the MCP
provenance envelope (D1), so the web + paper cite the same content-addressed run.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

#: env vars that materially change what a run produces.
_KEY_ENV = (
    "MPH_BEST_BY", "MPH_EVAL_FEATURES", "MPH_FEAT_PATH", "MPH_USE_3STAGE",
    "MPH_STABLE_TRANSFORMS", "MPH_STABLE_INTEGRATOR", "MPH_MC_PER_MODEL",
)


def _git_sha() -> Optional[str]:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True, timeout=5)
        return r.stdout.strip() or None
    except Exception:
        return None


def _package_hash() -> Optional[str]:
    try:
        r = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                           capture_output=True, text=True, encoding="utf-8", timeout=60)
        return hashlib.sha256(r.stdout.encode()).hexdigest()[:16] if r.stdout else None
    except Exception:
        return None


def _db_vintage() -> Optional[str]:
    try:
        import sqlite3
        from simulation.database.config import DB_PATH
        con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=2.0)
        try:
            con.execute("PRAGMA busy_timeout=2000")
            row = con.execute("SELECT MAX(collected_at) FROM weekly_disease").fetchone()
            return str(row[0]) if row and row[0] is not None else None
        finally:
            con.close()
    except Exception:
        return None


def build_run_manifest(seed: int = 42) -> dict:
    """Collect the reproducibility manifest dict (no disk write). Never raises.

    Returns ``{git_sha, package_hash, seed, db_vintage_ts, env, config_sha256}``;
    ``config_sha256`` is the SHA-256 of the deterministic manifest content.
    """
    m: dict = {
        "git_sha": _git_sha(),
        "package_hash": _package_hash(),
        "seed": int(seed),
        "db_vintage_ts": _db_vintage(),
        "env": {k: os.environ.get(k) for k in _KEY_ENV if os.environ.get(k) is not None},
    }
    payload = json.dumps(m, sort_keys=True, ensure_ascii=False)
    m["config_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return m


def write_run_manifest(save_dir, seed: int = 42) -> dict:
    """Build + write ``run_manifest.json`` into ``save_dir``. Returns the manifest.

    Args:
        save_dir: run output directory (created if absent). None → no write.
        seed: resolved run seed.

    Side effects: writes ``<save_dir>/run_manifest.json``. Never raises.
    """
    m = build_run_manifest(seed=seed)
    try:
        if save_dir is not None:
            p = Path(save_dir)
            p.mkdir(parents=True, exist_ok=True)
            (p / "run_manifest.json").write_text(
                json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return m
