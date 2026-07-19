"""Maintenance / housekeeping CLI commands — extracted from __main__.py.

Phase C2 partial (2026-05-12): 4 maintenance subcommand handlers (maintain,
prune, doctor, auto-update) moved here. Dispatch table in __main__.py
maps these names unchanged via re-import.

Each handler:
    - Takes argparse Namespace `args`
    - Calls into simulation.utils / simulation.database helpers
    - Exits via sys.exit() on non-zero rc (preserves __main__.py behaviour)
"""
from __future__ import annotations

import logging
import sys


log = logging.getLogger(__name__)


def cmd_maintain(args) -> None:
    """`python -m simulation maintain` — DB maintenance (names fix, master populate)."""
    from simulation.database.maintain import run_maintenance

    no_fix = getattr(args, "no_fix", False)
    result = run_maintenance(fix=not no_fix, report=True)
    if "names_fixed" in result:
        print(f"Names fixed: {result['names_fixed']}, "
              f"Master: {result['master_populated']}")


def cmd_prune(args) -> None:
    """`python -m simulation prune` — trash non-champion .pt + Optuna VACUUM + stale outputs.

    Reversible by default (everything goes to ``simulation/results/_trash/``).
    Use ``--purge`` to skip trash and delete permanently.
    """
    from simulation.utils.prune import run_prune

    do_models = bool(getattr(args, "prune_models", False))
    do_vacuum = bool(getattr(args, "optuna_vacuum", False))
    do_logs = bool(getattr(args, "prune_eval_logs", False))
    do_infer = bool(getattr(args, "prune_inference", False))
    if getattr(args, "prune_all", False):
        do_models = do_vacuum = do_logs = do_infer = True

    if not (do_models or do_vacuum or do_logs or do_infer):
        log.warning("[prune] no targets selected — use --models / --optuna-vacuum / "
                    "--eval-logs / --inference-results / --all")
        sys.exit(2)

    rep = run_prune(
        models=do_models,
        optuna_vacuum=do_vacuum,
        eval_logs=do_logs,
        inference_results=do_infer,
        keep_days=int(getattr(args, "keep_days", 7)),
        purge=bool(getattr(args, "purge", False)),
        dry_run=bool(getattr(args, "dry_run", False)),
    )
    if not rep.actions:
        log.info("[prune] nothing to clean — already optimal")


def cmd_doctor(args) -> None:
    """`python -m simulation doctor` — end-to-end environment + project diagnostic.

    Emits:
      • System / accelerator / package / OMP env / project-layout checks
      • DB ↔ model connectivity (table row counts, age)
      • Code self-test (every phase module + ChampionArtifact roundtrip)
      • Pipeline-readiness (champions, optuna caches, FE cache freshness)
      • Hardware-aware recommendations (n_jobs, scenario, weather mode, …)

    With ``--auto`` the safe subset is applied automatically:
      • mkdir missing directories under simulation/
      • set Apple OMP guards in-process (KMP_DUPLICATE_LIB_OK + thread caps)

    Exit code: 0 if no FAIL (1 if any FAIL; 1 also on WARN with --strict).
    """
    from pathlib import Path as _PPath

    from simulation.utils.doctor import run_doctor

    save_path = _PPath(args.save_report) if getattr(args, "save_report", None) else None
    rc, _rep = run_doctor(
        auto=bool(getattr(args, "auto", False)),
        verbose=bool(getattr(args, "verbose", False)),
        save_report=save_path,
        strict=bool(getattr(args, "strict", False)),
    )
    if rc != 0:
        sys.exit(rc)


def cmd_auto_update(args) -> None:
    """`python -m simulation auto-update` — weekly maintenance loop.

    Idempotent: collect → refit → horizon-stratified forecast.
    Safe for unattended cron / launchd / Task Scheduler runs.
    """
    from simulation.utils.auto_update import run_auto_update

    rep = run_auto_update(
        forecast_only=bool(getattr(args, "forecast_only", False)),
        weeks_ahead=int(getattr(args, "weeks_ahead", 4)),
        min_db_age_hours=float(getattr(args, "min_db_age_hours", 24.0)),
        min_refit_days=float(getattr(args, "min_refit_days", 7.0)),
        force_refit=bool(getattr(args, "force_refit", False)),
        force_collect=bool(getattr(args, "force_collect", False)),
        with_actuals=not bool(getattr(args, "no_actuals", False)),
        dry_run=bool(getattr(args, "dry_run", False)),
    )
    if rep.errors:
        sys.exit(1)


__all__ = [
    "cmd_maintain",
    "cmd_prune",
    "cmd_doctor",
    "cmd_auto_update",
]
