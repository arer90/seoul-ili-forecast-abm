"""
simulation.verifier.runtime_monitor
====================================
Runtime environment monitor (§9.4, RECOMMENDED_PIPELINE.md ).

Captures the *state* of the environment at pipeline start:
 * git commit + dirty flag
 * seed fixation (PYTHONHASHSEED, random, numpy, torch)
 * DB quick_check status + schema verify
 * Package versions for critical deps (numpy, pandas, torch, polars)
 * CUDA availability

One run → one snapshot written into run_ledger.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)


@dataclass
class RuntimeSnapshot:
    run_id: str
    started_at: str
    git_commit: Optional[str] = None
    git_dirty: bool = False
    seed: Optional[int] = None
    config_sha256: Optional[str] = None
    cli_args: Optional[str] = None
    scenario: Optional[str] = None
    db_quick_check: Optional[str] = None
    schema_ok: Optional[bool] = None
    schema_missing: list[str] = field(default_factory=list)
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    package_versions: dict[str, str] = field(default_factory=dict)
    cuda_available: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class RuntimeMonitor:
    """Collect + persist a RuntimeSnapshot.

    Usage::

        mon = RuntimeMonitor(run_id="2026-04-17T10:00:00", seed=42)
        snap = mon.snapshot()
        mon.persist(snap, scenario="baseline", cli_args=sys.argv[1:])
    """

    def __init__(
        self,
        *,
        run_id: Optional[str] = None,
        seed: Optional[int] = None,
        repo_root: Optional[Path] = None,
    ):
        self.run_id = run_id or datetime.now().strftime("%Y%m%dT%H%M%S")
        self.seed = seed
        self.repo_root = repo_root or _find_repo_root()

    # ──────────────────────────────────────────────────────────────────
    def snapshot(
        self,
        *,
        config: Optional[dict] = None,
        cli_args: Optional[list[str]] = None,
        scenario: Optional[str] = None,
    ) -> RuntimeSnapshot:
        snap = RuntimeSnapshot(
            run_id=self.run_id,
            started_at=datetime.now().isoformat(),
            seed=self.seed,
            cli_args=" ".join(cli_args) if cli_args else None,
            scenario=scenario,
        )
        snap.git_commit, snap.git_dirty = self._git_status()
        snap.db_quick_check, snap.schema_ok, snap.schema_missing = self._db_status()
        snap.package_versions = self._package_versions()
        snap.cuda_available = self._cuda_available()
        if config is not None:
            snap.config_sha256 = self._hash_config(config)
        return snap

    # ──────────────────────────────────────────────────────────────────
    def persist(
        self,
        snap: RuntimeSnapshot,
        *,
        scenario: Optional[str] = None,
        cli_args: Optional[list[str]] = None,
    ) -> None:
        """Write snapshot into `run_ledger` table."""
        try:
            from simulation.database import insert_rows
        except Exception as e:
            log.warning("persist skipped (DB import failed): %s", e)
            return
        row = {
            "run_id": snap.run_id,
            "started_at": snap.started_at,
            "finished_at": None,
            "git_commit": snap.git_commit,
            "git_dirty": 1 if snap.git_dirty else 0,
            "seed": snap.seed,
            "config_sha256": snap.config_sha256,
            "cli_args": json.dumps({
                "args": cli_args or (snap.cli_args.split() if snap.cli_args else []),
                "python": snap.python_version,
                "cuda": snap.cuda_available,
                "db_quick_check": snap.db_quick_check,
                "schema_ok": snap.schema_ok,
                "schema_missing": snap.schema_missing,
                "package_versions": snap.package_versions,
            }),
            "scenario": scenario or snap.scenario,
            "status": "running",
            "n_models": None,
            "best_model": None,
            "best_metric_name": None,
            "best_metric_value": None,
            "notes": None,
        }
        try:
            insert_rows("run_ledger", [row], on_conflict="REPLACE")
        except Exception as e:
            log.warning("run_ledger write failed: %s", e)

    def mark_finished(
        self,
        run_id: str,
        *,
        status: str = "ok",
        best_model: Optional[str] = None,
        best_metric_name: Optional[str] = None,
        best_metric_value: Optional[float] = None,
        n_models: Optional[int] = None,
        notes: Optional[str] = None,
    ) -> None:
        """Update run_ledger row on completion."""
        try:
            from simulation.database import safe_connect, transaction
        except Exception:
            return
        conn = safe_connect()
        try:
            with transaction(conn):
                conn.execute(
                    """UPDATE run_ledger SET
                        finished_at=?, status=?, best_model=?,
                        best_metric_name=?, best_metric_value=?,
                        n_models=?, notes=?
                       WHERE run_id=?""",
                    (datetime.now().isoformat(), status, best_model,
                     best_metric_name, best_metric_value, n_models, notes,
                     run_id),
                )
        finally:
            conn.close()

    # ──────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────
    def _git_status(self) -> tuple[Optional[str], bool]:
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo_root, capture_output=True,
                text=True, timeout=3, check=False,
            )
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.repo_root, capture_output=True,
                text=True, timeout=3, check=False,
            )
            sha = commit.stdout.strip() if commit.returncode == 0 else None
            dirty = bool(status.stdout.strip()) if status.returncode == 0 else False
            return sha, dirty
        except Exception:
            return None, False

    def _db_status(self) -> tuple[Optional[str], Optional[bool], list[str]]:
        try:
            from simulation.database import quick_check, verify_schema
            qc = quick_check()
            vs = verify_schema()
            return qc, bool(vs.get("ok", False)), list(vs.get("missing", []))
        except Exception as e:
            log.warning("db_status check failed: %s", e)
            return None, None, []

    def _package_versions(self) -> dict[str, str]:
        pkgs = {}
        for name in ("numpy", "pandas", "polars", "sklearn", "torch",
                     "xgboost", "lightgbm", "optuna", "duckdb", "statsmodels"):
            try:
                mod = __import__(name)
                pkgs[name] = getattr(mod, "__version__", "unknown")
            except Exception:
                pass
        return pkgs

    def _cuda_available(self) -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _hash_config(self, cfg: dict) -> str:
        try:
            blob = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
            return hashlib.sha256(blob).hexdigest()
        except Exception:
            return ""


def _find_repo_root() -> Path:
    """Walk up from this file looking for `.git`."""
    here = Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        if (p / ".git").exists() or (p / "simulation").exists():
            return p
    return here
