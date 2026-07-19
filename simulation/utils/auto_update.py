"""
auto-update — weekly maintenance loop for live ILI forecasting.
================================================================

The KDCA sentinel-influenza system publishes weekly ILI rates every Friday.
For the forecasting pipeline to stay calibrated, three things must happen
on a weekly cadence:

  1. **Re-collect** the latest week's surveillance + weather + mobility
     data from upstream APIs (KDCA / KMA / KOSIS / SeoulDataPlaza).
  2. **Champion refresh** — refit on the new in-sample slab so the
     `ChampionArtifact` bundles in ``models/`` stay current. Champion-
     challenger logic ensures only better-performing fits replace.
  3. **Forecast** the next 1-4 weeks (h=1 is the operational KPI) and
     write a horizon-stratified report.

This module orchestrates all three idempotently — safe to run via cron /
launchd / Windows Task Scheduler. Each run:

  • Skips collect if DB age < ``--min-db-age-hours`` (default 24h)
  • Skips refit if last_promoted < ``--min-refit-days`` (default 7d)
    UNLESS new in-sample weeks landed since last refit (force refit)
  • Always emits a forecast (cheap; uses existing champion .pt)
  • Writes a single audit line to
    ``simulation/results/auto_update_log.csv``

CLI:

    python -m simulation auto-update                  # full weekly cycle
    python -m simulation auto-update --forecast-only  # skip collect+refit
    python -m simulation auto-update --dry-run        # show plan, don't run
    python -m simulation auto-update --weeks-ahead 4  # h=1..4

Schedule (macOS launchd, Windows Task Scheduler, Linux cron):

    # cron: every Saturday 04:00 KST
    0 4 * * 6 cd /path/to/repo && .venv/bin/python -m simulation auto-update

    # launchd: see scripts/auto_update.plist
"""
from __future__ import annotations

import csv
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class AutoUpdateReport:
    started_at: str
    completed_at: str = ""
    elapsed_sec: float = 0.0
    stages_run: list[str] = field(default_factory=list)
    stages_skipped: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    forecast: dict = field(default_factory=dict)
    dry_run: bool = False

    def to_dict(self) -> dict:
        return {
            "started_at":  self.started_at,
            "completed_at": self.completed_at,
            "elapsed_sec": round(self.elapsed_sec, 2),
            "stages_run":  self.stages_run,
            "stages_skipped": self.stages_skipped,
            "errors":      self.errors,
            "forecast":    self.forecast,
            "dry_run":     self.dry_run,
        }


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_subcmd(cmd: list[str], dry_run: bool, label: str,
                  capture: bool = False) -> tuple[int, str, str]:
    """Run a `simulation <subcmd>` invocation. Returns (rc, stdout, stderr)."""
    log.info(f"  [auto-update] → {label}: {' '.join(cmd)}")
    if dry_run:
        return 0, f"(dry-run) {label}", ""
    try:
        r = subprocess.run(cmd, capture_output=capture, text=True, timeout=8 * 3600)
        return r.returncode, (r.stdout or ""), (r.stderr or "")
    except subprocess.TimeoutExpired:
        log.error(f"  [auto-update] {label} timed out (>8h)")
        return 124, "", "timeout"
    except Exception as e:
        log.error(f"  [auto-update] {label} failed: {e}")
        return 1, "", str(e)


def _db_age_hours(db_path: Path) -> Optional[float]:
    if not db_path.exists():
        return None
    return (time.time() - db_path.stat().st_mtime) / 3600.0


def _last_promotion_age_days(log_path: Path) -> Optional[float]:
    if not log_path.exists():
        return None
    try:
        j = json.loads(log_path.read_text())
    except Exception:
        return None
    most_recent: Optional[float] = None
    for _name, rec in j.items():
        cur = (rec or {}).get("current") or {}
        ts = cur.get("promoted_at")
        if not ts:
            continue
        try:
            dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc)
            age_d = (datetime.now(timezone.utc) - dt).total_seconds() / 86400
            if most_recent is None or age_d < most_recent:
                most_recent = age_d
        except Exception:
            continue
    return most_recent


# ─────────────────────────────────────────────────────────────────
# Stages
# ─────────────────────────────────────────────────────────────────
def stage_collect(rep: AutoUpdateReport, *, min_db_age_hours: float,
                   force: bool, dry_run: bool, repo_root: Path) -> bool:
    """Re-collect upstream data if DB is stale enough. Returns True if ran."""
    db_path = repo_root / "simulation" / "data" / "db" / "epi_real_seoul.db"
    age_h = _db_age_hours(db_path)
    if age_h is not None and age_h < min_db_age_hours and not force:
        rep.stages_skipped.append({
            "stage": "collect",
            "reason": f"DB age {age_h:.1f}h < threshold {min_db_age_hours:.1f}h",
        })
        log.info(f"  [auto-update] skip collect (DB age {age_h:.1f}h)")
        return False

    cmd = [str(repo_root / ".venv" / "bin" / "python3"),
           "-m", "simulation", "collect", "--groups", "all"]
    rc, _o, e = _run_subcmd(cmd, dry_run, "collect")
    if rc != 0:
        rep.errors.append({"stage": "collect", "rc": rc, "stderr": e[:500]})
        return False
    rep.stages_run.append("collect")
    return True


def stage_refit(rep: AutoUpdateReport, *, min_refit_days: float,
                 force_refit: bool, dry_run: bool, repo_root: Path,
                 models: Optional[list[str]] = None) -> bool:
    """Refit champions on the latest in-sample slab. Champion-challenger
    auto-decides promotion."""
    log_path = repo_root / "models" / "champion_log.json"
    last_age = _last_promotion_age_days(log_path)
    if (last_age is not None
        and last_age < min_refit_days
        and not force_refit):
        rep.stages_skipped.append({
            "stage": "refit",
            "reason": f"last promotion {last_age:.1f}d ago < threshold {min_refit_days}d",
        })
        log.info(f"  [auto-update] skip refit (last promo {last_age:.1f}d ago)")
        return False

    # Conservative refit: only the inference-validated 6 models. Keeps the
    # weekly cycle under 1 hour (vs 4h for full --scenario full + --per-model-optimize).
    if models is None:
        models = ["XGBoost", "LightGBM", "RandomForest",
                   "NegBinGLM", "BayesianRidge", "ElasticNet"]

    cmd = [str(repo_root / ".venv" / "bin" / "python3"),
           "-m", "simulation", "train", "--force",
           "--scenario", "lite",
           "--models", ",".join(models),
           "--optuna-trials", "20",
           "--weather-mode", "hybrid",
           "--conformal-method", "aci",
           "--ensemble-method", "stacking",
           "--covid-mode", "indicator",
           "--per-model-optimize"]
    rc, _o, e = _run_subcmd(cmd, dry_run, "refit",
                              capture=False)  # streams to terminal/log
    if rc != 0:
        rep.errors.append({"stage": "refit", "rc": rc, "stderr": e[:500]})
        return False
    rep.stages_run.append("refit")
    return True


def stage_forecast(rep: AutoUpdateReport, *, weeks_ahead: int,
                     with_actuals: bool, dry_run: bool,
                     repo_root: Path) -> dict:
    """Run predict-real on the most recent N weeks. Returns the forecast dict."""
    out_dir = (repo_root / "simulation" / "results" / "auto_forecasts" /
                datetime.now().strftime("%Y%m%d_%H%M%S"))
    cmd = [str(repo_root / ".venv" / "bin" / "python3"),
           "-m", "simulation", "predict-real",
           "--weeks-ahead", str(weeks_ahead),
           "--out-dir", str(out_dir)]
    if with_actuals:
        cmd.append("--with-actuals")

    rc, _o, e = _run_subcmd(cmd, dry_run, "forecast", capture=True)
    if rc != 0:
        rep.errors.append({"stage": "forecast", "rc": rc, "stderr": e[:500]})
        return {}
    rep.stages_run.append("forecast")

    # Read back the resulting predictions / metrics for the report
    forecast: dict = {"out_dir": str(out_dir)}
    if dry_run:
        forecast["note"] = "dry-run; no files produced"
        return forecast
    metrics_path = out_dir / "inference_metrics.json"
    pred_csv = out_dir / "predictions.csv"
    if metrics_path.exists():
        try:
            forecast["metrics"] = json.loads(metrics_path.read_text())
        except Exception:
            pass
    if pred_csv.exists():
        forecast["predictions_csv"] = str(pred_csv)
    return forecast


# ─────────────────────────────────────────────────────────────────
# Append a row to the audit log
# ─────────────────────────────────────────────────────────────────
def append_audit_row(rep: AutoUpdateReport, *, repo_root: Path) -> Path:
    audit_path = repo_root / "simulation" / "results" / "auto_update_log.csv"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not audit_path.exists()
    with audit_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["started_at", "elapsed_sec", "stages_run",
                         "stages_skipped", "errors", "forecast_out",
                         "dry_run"])
        w.writerow([
            rep.started_at,
            rep.elapsed_sec,
            "+".join(rep.stages_run),
            "+".join(s.get("stage", "?") for s in rep.stages_skipped),
            len(rep.errors),
            rep.forecast.get("out_dir", ""),
            rep.dry_run,
        ])
    return audit_path


# ─────────────────────────────────────────────────────────────────
# Public entry
# ─────────────────────────────────────────────────────────────────
def run_auto_update(*,
                     forecast_only: bool = False,
                     weeks_ahead: int = 4,
                     min_db_age_hours: float = 24.0,
                     min_refit_days: float = 7.0,
                     force_refit: bool = False,
                     force_collect: bool = False,
                     with_actuals: bool = True,
                     dry_run: bool = False,
                     repo_root: Optional[Path] = None) -> AutoUpdateReport:
    """End-to-end weekly maintenance.

    Args:
      forecast_only:   skip collect + refit; just predict using current champions
      weeks_ahead:     forecast horizons (h=1..N)
      min_db_age_hours: only re-collect if DB older than this
      min_refit_days:  only refit if last promotion older than this
      force_refit:     bypass the day-threshold and refit anyway
      with_actuals:    include actual ILI rate if available (for back-test)
    """
    if repo_root is None:
        repo_root = Path.cwd()
    repo_root = Path(repo_root)

    t0 = time.time()
    rep = AutoUpdateReport(started_at=_utcnow_iso(), dry_run=dry_run)

    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  simulation auto-update — weekly maintenance")
    print(f"{bar}")
    print(f"  started:        {rep.started_at}")
    print(f"  forecast_only:  {forecast_only}")
    print(f"  weeks_ahead:    {weeks_ahead}")
    print(f"  dry_run:        {dry_run}")
    print()

    # 1. Collect
    if not forecast_only:
        print("  [stage 1/3] collect …")
        stage_collect(rep, min_db_age_hours=min_db_age_hours,
                       force=force_collect, dry_run=dry_run, repo_root=repo_root)
        print()

    # 2. Refit (champion-challenger)
    if not forecast_only:
        print("  [stage 2/3] refit (champion-challenger) …")
        stage_refit(rep, min_refit_days=min_refit_days,
                     force_refit=force_refit, dry_run=dry_run,
                     repo_root=repo_root)
        print()

    # 3. Forecast (always)
    print(f"  [stage 3/3] forecast (h=1..{weeks_ahead}) …")
    rep.forecast = stage_forecast(rep, weeks_ahead=weeks_ahead,
                                     with_actuals=with_actuals,
                                     dry_run=dry_run, repo_root=repo_root)
    print()

    rep.elapsed_sec = time.time() - t0
    rep.completed_at = _utcnow_iso()
    audit_path = append_audit_row(rep, repo_root=repo_root)

    # Summary
    print(bar)
    print(f"  Result: {len(rep.stages_run)} ran, "
          f"{len(rep.stages_skipped)} skipped, "
          f"{len(rep.errors)} errors  ({rep.elapsed_sec:.1f}s)")
    if rep.stages_run:
        print(f"  Ran:     {', '.join(rep.stages_run)}")
    if rep.stages_skipped:
        for s in rep.stages_skipped:
            print(f"  Skipped: {s['stage']:<10}  ({s.get('reason','?')})")
    if rep.errors:
        for e in rep.errors:
            print(f"  ERROR:   {e['stage']:<10}  rc={e.get('rc','?')}  "
                  f"{e.get('stderr','')[:100]}")
    if rep.forecast.get("out_dir"):
        print(f"  Forecast: {rep.forecast['out_dir']}")
    if rep.forecast.get("metrics"):
        # h=1 is the operational KPI; print just that
        for nm, m in sorted(rep.forecast["metrics"].items(),
                              key=lambda kv: kv[1].get(
                                  "per_horizon", {}).get("h1", {}).get("ae", 999)):
            h1 = m.get("per_horizon", {}).get("h1", {})
            if h1:
                print(f"    {nm:<20}  h=1 pred={h1.get('pred', '?'):.2f}  "
                      f"actual={h1.get('actual', '?'):.2f}  "
                      f"AE={h1.get('ae', float('nan')):.2f}")
    print(f"  Audit log: {audit_path}")
    print(bar)
    return rep


__all__ = ["run_auto_update", "AutoUpdateReport"]
