"""
simulation.benchmarks.bench_dl_baselines
========================================

S2-3 (backlog) — TinyMLP vs DNN-variant comparison harness.

Purpose
-------
ENGINEERING_PRINCIPLES.md 알려진 이슈 §S2:
 "DNN over-parameterization: n=343 에 attention+FM 은 과함.
 작은 MLP 베이스라인 비교 필수."

This script runs a fixed set of DL baselines on the SAME R1 (data) feature
matrix and the SAME walk-forward split, then writes a CSV comparing their
out-of-fold (OOF) metrics. If the heavier DNN / TabularDNN variants can't
meaningfully beat TinyMLP's fixed (32 → 16) architecture, then those
extra parameters are noise on n≈343 weekly observations.

Usage
-----
 python -m simulation.benchmarks.bench_dl_baselines \\
 --models TinyMLP TabularDNN DNN \\
 --out simulation/results/bench_dl_baselines.csv

Default models: TinyMLP, DNN, TabularDNN (skip Optuna/TCN — too slow for a
quick sanity sweep). Pass ``--models all`` to include them.

Output CSV columns
------------------
 model, category, n_features, n_folds_completed,
 r2, rmse, mae, mape,
 early_r2, late_r2, stable,
 elapsed_s, timestamp
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


log = logging.getLogger(__name__)


# Default set excludes Optuna / TCN variants — they are much slower and
# orthogonal to the TinyMLP-vs-heavier-MLP question this harness asks.
DEFAULT_MODELS: tuple[str, ...] = ("TinyMLP", "DNN", "TabularDNN")
ALL_DL_MODELS: tuple[str, ...] = (
    "TinyMLP", "DNN", "TabularDNN", "TCN",
    "OptunaDNN", "OptunaTCN",
)


def _ensure_registry_loaded() -> None:
    """simulation.models.__init__ does NOT eagerly load submodules, so
    REGISTRY is empty until dl_models is imported at least once."""
    import simulation.models.dl_models  # noqa: F401 — side-effect: registers models


def _build_config(db_path: Optional[str] = None, use_cache: bool = True,
                  min_train_weeks: int = 120, step_size: int = 1,
                  conformal_holdout_weeks: int = 26):
    """Construct the minimal PipelineConfig needed to drive
    run_wfcv_single_model. Heavy Optuna / phase-specific settings
    irrelevant to the benchmark are left at dataclass defaults.
    """
    from simulation.pipeline.config import PipelineConfig
    cfg = PipelineConfig()
    if db_path:
        cfg.data.db_path = db_path
    cfg.data.use_fe_cache = use_cache
    cfg.wfcv.min_train_weeks = min_train_weeks
    cfg.wfcv.step_size = step_size
    cfg.split.conformal_holdout_weeks = conformal_holdout_weeks
    cfg.memory.use_float32 = True
    return cfg


def _load_features(config) -> dict:
    """Run R1 (data) (or reuse the Parquet cache) and return its dict."""
    from simulation.pipeline.data import run_data
    return run_data(config)


def _run_one_model(model_name: str, X_all, y_all, feature_cols, config,
                   holdout_start: Optional[int]) -> dict:
    """Fit + evaluate a single model via R4 (WF-CV)'s helper."""
    from simulation.models.base import REGISTRY
    from simulation.pipeline.wfcv import run_wfcv_single_model

    cls = REGISTRY.get(model_name)
    if cls is None:
        log.error("  [%s] 등록되지 않은 모델", model_name)
        return {
            "model": model_name, "status": "UNREGISTERED",
            "category": None, "error": "not in REGISTRY",
        }

    def factory():
        return cls()

    t0 = time.time()
    try:
        result = run_wfcv_single_model(
            X_all=X_all, y_all=y_all, feature_cols=feature_cols,
            model_name=model_name, model_factory=factory,
            config=config, holdout_start=holdout_start,
        )
        status = "OK"
        error = None
    except Exception as e:
        log.exception("  [%s] WF-CV 실패", model_name)
        result = {}
        status = "ERROR"
        error = str(e)
    elapsed = time.time() - t0

    overall = result.get("overall_metrics", {}) if isinstance(result, dict) else {}
    tstab = overall.get("temporal_stability", {}) or {}
    return {
        "model": model_name,
        "category": getattr(cls.meta, "category", None),
        "n_features": X_all.shape[1],
        "n_folds_completed": result.get("n_folds_completed"),
        "r2": overall.get("r2"),
        "rmse": overall.get("rmse"),
        "mae": overall.get("mae"),
        "mape": overall.get("mape"),
        "early_r2": tstab.get("early_r2"),
        "late_r2": tstab.get("late_r2"),
        "stable": tstab.get("stable"),
        "elapsed_s": round(elapsed, 1),
        "status": status,
        "error": error,
    }


def _write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().isoformat(timespec="seconds")
    for r in rows:
        r["timestamp"] = ts
    fieldnames = [
        "model", "category", "n_features", "n_folds_completed",
        "r2", "rmse", "mae", "mape",
        "early_r2", "late_r2", "stable",
        "elapsed_s", "status", "error", "timestamp",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _print_table(rows: list[dict]) -> None:
    """Pretty-print to stdout — matches the CSV columns but shorter."""
    print()
    header = f"{'model':<14} {'n_feat':>6} {'folds':>5} " \
             f"{'R²':>7} {'RMSE':>7} {'MAPE%':>7} {'elapsed':>8} {'status':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        r2 = r.get("r2")
        rmse = r.get("rmse")
        mape = r.get("mape")
        print(
            f"{r['model']:<14} "
            f"{r.get('n_features') or 0:>6} "
            f"{r.get('n_folds_completed') or 0:>5} "
            f"{(f'{r2:.4f}' if r2 is not None else '-'):>7} "
            f"{(f'{rmse:.3f}' if rmse is not None else '-'):>7} "
            f"{(f'{mape:.2f}' if mape is not None else '-'):>7} "
            f"{r.get('elapsed_s', 0):>7.1f}s "
            f"{r.get('status', '-'):>8}"
        )
    print()


def run_benchmark(models: list[str], out_path: Path,
                  db_path: Optional[str] = None,
                  use_cache: bool = True,
                  min_train_weeks: int = 120,
                  step_size: int = 1,
                  conformal_holdout_weeks: int = 26) -> list[dict]:
    """Programmatic entry point. Runs each listed model through
    R4 (WF-CV)'s helper with the exact same (X, y, folds) and
    returns a list of metric dicts (one per model)."""
    _ensure_registry_loaded()

    config = _build_config(
        db_path=db_path, use_cache=use_cache,
        min_train_weeks=min_train_weeks, step_size=step_size,
        conformal_holdout_weeks=conformal_holdout_weeks,
    )

    log.info("[bench_dl_baselines] R1 (data, features) 로드...")
    phase1 = _load_features(config)
    X_all = phase1["X_all"]
    y_all = phase1["y_all"]
    feature_cols = phase1["feature_cols"]
    holdout_start = phase1.get("holdout_start")
    log.info(
        "  데이터: n=%d, 피처=%d, holdout_start=%s",
        len(y_all), X_all.shape[1], holdout_start,
    )

    rows: list[dict] = []
    for name in models:
        log.info("[bench_dl_baselines] %s 학습 시작", name)
        row = _run_one_model(
            name, X_all, y_all, feature_cols, config,
            holdout_start=holdout_start,
        )
        rows.append(row)
        if row["status"] == "OK":
            log.info("  [%s] OK r2=%.4f rmse=%.3f (%.1fs)",
                     name, row.get("r2") or float("nan"),
                     row.get("rmse") or float("nan"),
                     row.get("elapsed_s") or 0.0)
        else:
            log.warning("  [%s] %s: %s", name, row["status"], row.get("error"))

    _write_csv(rows, out_path)
    log.info("[bench_dl_baselines] CSV 저장: %s", out_path)
    return rows


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m simulation.benchmarks.bench_dl_baselines",
        description=(
            "S2-3 sanity-floor harness: run TinyMLP and heavier DL baselines "
            "on the same WF-CV split; compare OOF R²/RMSE/MAPE per model."
        ),
    )
    p.add_argument(
        "--models", nargs="*", default=list(DEFAULT_MODELS),
        help=(
            f"Models to benchmark (default: {' '.join(DEFAULT_MODELS)}). "
            "Use 'all' for "
            + " ".join(ALL_DL_MODELS)
        ),
    )
    from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
    p.add_argument(
        "--out", type=Path,
        default=get_results_dir() / "bench_dl_baselines.csv",
        help="Output CSV path.",
    )
    p.add_argument("--db-path", default=None,
                   help="Override DB path (default: simulation.database.DB_PATH).")
    p.add_argument("--no-cache", action="store_true",
                   help="Force FE recompute (ignore feature_cache.parquet).")
    p.add_argument("--min-train-weeks", type=int, default=120)
    p.add_argument("--step-size", type=int, default=1)
    p.add_argument(
        "--conformal-holdout-weeks", type=int, default=26,
        help="S0-1 holdout slab size; 0 disables holdout (not recommended).",
    )
    p.add_argument("--log-level", default="INFO")
    p.add_argument(
        "--log-file", type=Path, default=None,
        help=(
            "Path to write a full run log. Default: alongside --out with "
            "suffix .log (e.g. bench_smoke.csv → bench_smoke.log). "
            "Pass an empty string to disable file logging."
        ),
    )
    return p.parse_args(argv)


def _resolve_log_path(args: argparse.Namespace) -> Optional[Path]:
    """--log-file explicit → use it. None → sibling of --out with .log
    suffix. Empty string → disable file logging."""
    if args.log_file is None:
        return args.out.with_suffix(".log")
    # argparse gives us Path(""), which stringifies to "." — treat as disable.
    if str(args.log_file).strip() in {"", "."}:
        return None
    return args.log_file


def _configure_logging(level_name: str, log_path: Optional[Path]) -> None:
    """Root-level handlers so every module (pipeline.*, models.*) lands
    in both console + file. basicConfig is idempotent-ish, so we set up
    handlers manually to add a FileHandler cleanly."""
    level = getattr(logging, level_name.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)
    # Clear any pre-existing handlers to avoid duplicate lines when
    # run_benchmark is re-invoked in the same process (e.g. tests).
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s")

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    stream.setLevel(level)
    root.addHandler(stream)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(level)
        root.addHandler(fh)
        log.info("[bench_dl_baselines] 로그 파일: %s", log_path)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    log_path = _resolve_log_path(args)
    _configure_logging(args.log_level, log_path)

    try:
        models = list(args.models)
        if len(models) == 1 and models[0].lower() == "all":
            models = list(ALL_DL_MODELS)

        rows = run_benchmark(
            models=models, out_path=args.out,
            db_path=args.db_path, use_cache=not args.no_cache,
            min_train_weeks=args.min_train_weeks,
            step_size=args.step_size,
            conformal_holdout_weeks=args.conformal_holdout_weeks,
        )

        _print_table(rows)
        n_ok = sum(1 for r in rows if r["status"] == "OK")
        if n_ok == 0:
            log.error("[bench_dl_baselines] all models failed")
            return 2
        if n_ok < len(rows):
            log.warning("[bench_dl_baselines] %d/%d models failed",
                        len(rows) - n_ok, len(rows))
            return 1
        return 0
    finally:
        # S2-3 follow-up: flush FileHandler before process exit. Without
        # this, the last few minutes of log can be truncated if Python
        # tears down stdio handlers before the file buffer drains
        # (observed on bench_full.csv run — 12 min of TabularDNN output
        # was missing from the .log file even though the CSV was complete).
        logging.shutdown()


if __name__ == "__main__":
    sys.exit(main())
