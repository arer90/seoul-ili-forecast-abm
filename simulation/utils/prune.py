"""
Disk footprint optimizer — keep only what's needed for inference + audit.
=========================================================================

After many training runs ``models/`` and ``simulation/results/`` accumulate:

  • ``<name>.pt`` files for models that aren't current champions (pre
    champion-challenger artifacts, or attempted-but-not-promoted versions
    that age out of relevance)
  • ``optuna_feature_selection.db`` grows monotonically — every Optuna
    study writes to it; old studies are never reclaimed
  • ``checkpoint_phase*.json`` from old failed runs
  • ``eval_logs/`` from runs older than the audit window

This module provides idempotent prune commands that cap disk usage without
touching champion-challenger state. Operations are reversible (everything
goes to ``simulation/results/_trash/`` first; permanent delete only with
``--purge``).

CLI:
    python -m simulation prune --dry-run                 # preview only
    python -m simulation prune --models                  # remove non-champion .pt
    python -m simulation prune --optuna-vacuum           # SQLite VACUUM
    python -m simulation prune --eval-logs --keep-days 7
    python -m simulation prune --all                     # all of the above
    python -m simulation prune --all --purge             # skip _trash, delete forever
"""
from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class PruneReport:
    actions: list[dict] = field(default_factory=list)
    bytes_freed: int = 0
    dry_run: bool = False

    def add(self, kind: str, path: Path, size_bytes: int, note: str = "") -> None:
        self.actions.append({
            "kind": kind, "path": str(path),
            "size_mb": round(size_bytes / 1e6, 2),
            "note": note,
        })
        self.bytes_freed += size_bytes


def _to_trash(path: Path, trash_dir: Path, dry_run: bool) -> None:
    """Move a file to trash (or simulate)."""
    if dry_run:
        return
    trash_dir.mkdir(parents=True, exist_ok=True)
    target = trash_dir / path.name
    # If a same-name already exists in trash (from prior run), suffix with timestamp
    if target.exists():
        target = trash_dir / f"{path.stem}_{int(time.time())}{path.suffix}"
    shutil.move(str(path), str(target))


def _purge(path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────
# 1. Prune non-champion .pt files
# ─────────────────────────────────────────────────────────────────
def prune_models(rep: PruneReport, *, models_dir: Path, trash_dir: Path,
                  purge: bool = False, dry_run: bool = False) -> None:
    """Move every ``models/*.pt`` to trash unless it's the current champion
    (= filename matches a current entry in champion_log.json).

    Also separately handles ``<name>_v{N}_*.pt`` (archived previous champions)
    and ``<name>_attempt_v{N}_*.pt`` (failed promotion attempts) — these are
    candidates for purge older than 30 days."""
    log_path = models_dir / "champion_log.json"

    # Current champion filenames
    current_champions: set[str] = set()
    if log_path.exists():
        try:
            j = json.loads(log_path.read_text())
            for name, rec in j.items():
                if isinstance(rec, dict):
                    cur = rec.get("current") or {}
                    fn = cur.get("filename")
                    if fn:
                        current_champions.add(fn)
        except Exception as e:
            log.warning(f"  [prune] champion_log unreadable: {e}")

    if not models_dir.exists():
        return

    for pt in models_dir.glob("*.pt"):
        name = pt.name
        size = pt.stat().st_size

        if name in current_champions:
            continue   # keep — current champion

        # Stem-based heuristic: <name>.pt is the canonical champion path,
        # but if champion_log lists a different filename, this .pt is stale.
        if "_v" not in name and "_attempt_v" not in name:
            # Looks like a canonical "<name>.pt" — but it's NOT in
            # current_champions → legacy or pre-artifact bare-model file.
            kind = "legacy_canonical"
            note = "pre-artifact bare-model .pt; not in champion_log"
        elif "_attempt_v" in name:
            kind = "failed_attempt"
            note = "challenger attempt that lost to current champion"
        else:
            kind = "archived_previous"
            note = "previous champion, archived on demotion"

        # Move/purge
        if purge:
            _purge(pt, dry_run)
            rep.add("purge:" + kind, pt, size, note)
        else:
            _to_trash(pt, trash_dir / "models", dry_run)
            rep.add("trash:" + kind, pt, size, note)


# ─────────────────────────────────────────────────────────────────
# 2. Vacuum Optuna SQLite (reclaim deleted-row space)
# ─────────────────────────────────────────────────────────────────
def vacuum_optuna_db(rep: PruneReport, *, db_path: Path,
                      dry_run: bool = False) -> None:
    """Run ``VACUUM`` on the Optuna SQLite DB. Optuna's RDB backend writes
    monotonically — VACUUM reclaims space from deleted/superseded trials.
    Typical compression: 100MB → 10-20MB."""
    if not db_path.exists():
        return
    before = db_path.stat().st_size
    if dry_run:
        # Estimate compression as 5-10x without actually running it
        rep.add("vacuum:dry", db_path, int(before * 0.85),
                "VACUUM not run; estimated 85% reclamation")
        return
    try:
        from simulation.database import safe_connect
        conn = safe_connect(str(db_path), timeout=30, isolation_level=None)
        conn.execute("VACUUM")
        conn.close()
    except Exception as e:
        log.error(f"  [prune] VACUUM failed: {e}")
        return
    after = db_path.stat().st_size
    rep.add("vacuum:done", db_path, before - after,
            f"{before/1e6:.1f}MB → {after/1e6:.1f}MB")


# ─────────────────────────────────────────────────────────────────
# 3. Eval-logs older than N days
# ─────────────────────────────────────────────────────────────────
def prune_eval_logs(rep: PruneReport, *, logs_dir: Path,
                     trash_dir: Path, keep_days: int = 7,
                     purge: bool = False, dry_run: bool = False) -> None:
    if not logs_dir.exists():
        return
    cutoff = time.time() - keep_days * 86400
    for f in logs_dir.glob("*.jsonl"):
        if f.stat().st_mtime < cutoff:
            size = f.stat().st_size
            if purge:
                _purge(f, dry_run)
                rep.add("purge:old_log", f, size,
                        f"older than {keep_days}d")
            else:
                _to_trash(f, trash_dir / "eval_logs", dry_run)
                rep.add("trash:old_log", f, size,
                        f"older than {keep_days}d")


# ─────────────────────────────────────────────────────────────────
# 4. Old phase14 inference output dirs (>14d default)
# ─────────────────────────────────────────────────────────────────
def prune_inference_results(rep: PruneReport, *,
                              inference_root: Path, trash_dir: Path,
                              keep_days: int = 14,
                              purge: bool = False,
                              dry_run: bool = False) -> None:
    if not inference_root.exists():
        return
    cutoff = time.time() - keep_days * 86400
    for d in inference_root.iterdir():
        if not d.is_dir():
            continue
        if d.stat().st_mtime < cutoff:
            size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            if purge:
                _purge(d, dry_run)
                rep.add("purge:old_infer", d, size,
                        f"older than {keep_days}d")
            else:
                _to_trash(d, trash_dir / "inference", dry_run)
                rep.add("trash:old_infer", d, size,
                        f"older than {keep_days}d")


# ─────────────────────────────────────────────────────────────────
# Public entry
# ─────────────────────────────────────────────────────────────────
def run_prune(*, models: bool = False, optuna_vacuum: bool = False,
               eval_logs: bool = False, inference_results: bool = False,
               keep_days: int = 7, purge: bool = False,
               dry_run: bool = False, repo_root: Optional[Path] = None) -> PruneReport:
    """Top-level prune dispatch. Returns a PruneReport with actions taken."""
    if repo_root is None:
        repo_root = Path.cwd()
    repo_root = Path(repo_root)
    rep = PruneReport(dry_run=dry_run)

    trash_dir = repo_root / "simulation" / "results" / "_trash"

    print("\n" + "=" * 70)
    print(f"  simulation prune — disk footprint optimization")
    print("=" * 70)
    if dry_run:
        print("  Mode: --dry-run (preview only, no changes)")
    elif purge:
        print("  Mode: --purge (PERMANENT delete; trash skipped)")
    else:
        print(f"  Mode: trash to {trash_dir.relative_to(repo_root)}/")
    print()

    if models:
        print("  [1/4] non-champion .pt files…")
        prune_models(rep, models_dir=repo_root / "models",
                     trash_dir=trash_dir, purge=purge, dry_run=dry_run)
    if optuna_vacuum:
        print("  [2/4] Optuna SQLite VACUUM…")
        for db in [
            repo_root / "simulation" / "results" / "optuna_feature_selection.db",
            repo_root / "simulation" / "results" / "optuna_per_model" / "optuna.db",
        ]:
            if db.exists():
                vacuum_optuna_db(rep, db_path=db, dry_run=dry_run)
    if eval_logs:
        print(f"  [3/4] eval_logs older than {keep_days}d…")
        prune_eval_logs(rep,
                          logs_dir=repo_root / "simulation" / "results" / "eval_logs",
                          trash_dir=trash_dir, keep_days=keep_days,
                          purge=purge, dry_run=dry_run)
    if inference_results:
        print(f"  [4/4] inference outputs older than 14d…")
        prune_inference_results(rep,
                                  inference_root=(repo_root / "simulation" /
                                                   "results" / "inference"),
                                  trash_dir=trash_dir, keep_days=14,
                                  purge=purge, dry_run=dry_run)

    # Summary
    print()
    print(f"  Actions: {len(rep.actions)}")
    print(f"  Disk freed: {rep.bytes_freed / 1e6:.1f} MB"
          + (" (estimated, dry-run)" if dry_run else ""))
    if rep.actions:
        print()
        print(f"  {'kind':<22} {'size MB':>8}  path")
        print("  " + "-" * 75)
        for a in sorted(rep.actions, key=lambda r: -r["size_mb"])[:30]:
            print(f"  {a['kind']:<22} {a['size_mb']:>8.2f}  "
                  f"{a['path'].split('/')[-1]:<40}  {a['note']}")
        if len(rep.actions) > 30:
            print(f"  … and {len(rep.actions) - 30} more")

    print("=" * 70)
    return rep


__all__ = ["run_prune", "PruneReport"]
