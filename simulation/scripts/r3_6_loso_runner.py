"""
r3_6_loso_runner.py
====================
LOSO (Leave-One-Season-Out) — 2022-23 season held out as test fold instead of
the final 15%. Re-runs all 63 models on the altered split so that the test
set (2022-09 ~ 2023-08) is a post-NPI but pre-rebound mixed regime.

Motivation (R3-6 feasibility test, 2026-04-20):
  ElasticNetCV  dRMSE = -43.7%
  BayesianRidge dRMSE = -45.5%
  RandomForest  dRMSE = -53.8%
  -> last-fold WF-CV and LOSO 2022-23 give genuinely different error
     distributions; full retrain is justified (not a trivial re-eval).

Runtime: ~4-5 hours (paper-primary subset ~1-2h if `--primary-only`).

Outputs:
  simulation/results/loso_2022_23/
    ├── phase4_baseline_sidecar.pkl     # per-model val/test preds
    ├── checkpoints/*.json
    └── r3_6_comparison.json            # last-fold WF-CV vs LOSO 2022-23

The altered split uses the same `phase1_data` FE cache (deterministic given
DB snapshot) but passes a custom `(train_idx, test_idx)` pair to the runner.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "simulation" / "results" / "loso_2022_23"

LOSO_SEASON_START = "2022-09-01"
LOSO_SEASON_END = "2023-08-31"


def _identify_loso_indices(dates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dates_pd = pd.to_datetime(dates)
    mask_test = (dates_pd >= LOSO_SEASON_START) & (dates_pd <= LOSO_SEASON_END)
    idx_all = np.arange(len(dates))
    test_idx = idx_all[mask_test.to_numpy()]
    train_idx = idx_all[~mask_test.to_numpy()]
    if len(test_idx) < 10:
        raise RuntimeError(
            f"LOSO test fold too small: n_test={len(test_idx)} for "
            f"{LOSO_SEASON_START}..{LOSO_SEASON_END}"
        )
    return train_idx, test_idx


def _load_features() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """phase1_data 와 동일한 feature_builder 를 호출해 X, y, dates, feature_cols 반환."""
    from simulation.pipeline.config import PipelineConfig

    config = PipelineConfig()
    os.environ.setdefault("MPH_OUTPUT_ROOT", str(DEFAULT_OUT.parent.parent))
    # phase1_data.run_data 은 cache 를 쓰기도 함. 직접 호출하지 않고 builder 재실행.
    from simulation.models.feature_engine.builder import build_features
    df, meta = build_features(
        feature_set="full", disease="influenza", target_column="ili_rate",
        cache_dir=str(config.data.cache_dir),
        use_cache=False,
    )
    y_col = "ili_rate"
    feature_cols = [c for c in df.columns if c not in (y_col, "week_start")]
    X_all = df[feature_cols].to_numpy(dtype=float)
    y_all = df[y_col].to_numpy(dtype=float)
    dates = pd.to_datetime(df["week_start"]).to_numpy()
    return X_all, y_all, dates, feature_cols


def _run_loso(primary_only: bool, save_dir: Path) -> dict:
    save_dir.mkdir(parents=True, exist_ok=True)

    X_all, y_all, dates, feature_cols = _load_features()
    train_idx, test_idx = _identify_loso_indices(dates)
    log.info(f"  [LOSO] n_train={len(train_idx)}  n_test={len(test_idx)}  "
             f"test_range={dates[test_idx[0]]} .. {dates[test_idx[-1]]}")

    X_tr, y_tr = X_all[train_idx], y_all[train_idx]
    X_te, y_te = X_all[test_idx], y_all[test_idx]
    # val: last 15% of train
    n_val = max(int(len(train_idx) * 0.15), 20)
    X_val, y_val = X_tr[-n_val:], y_tr[-n_val:]
    X_tr_core, y_tr_core = X_tr[:-n_val], y_tr[:-n_val]

    from simulation.models.runner import MultiModelRunner
    from simulation.models.target_transform import get_preset, get_per_model_strategy

    tt, _ = get_preset("papermatched")
    per_model = get_per_model_strategy("optimal")

    include = None
    if primary_only:
        from simulation.models.registry import PAPER_PRIMARY_11
        include = [n for (n, _) in PAPER_PRIMARY_11]

    runner = MultiModelRunner(
        target_transformer=tt,
        per_model_transform=per_model,
        per_model_features=None,
        feature_names=feature_cols,
        include_only=include or [],
    )
    results = runner.run(
        X_tr_core, y_tr_core, X_val, y_val, X_te, y_te,
        run_ensembles=True,
        save_models=False,
        save_dir=str(save_dir / "models"),
    )

    # Persist sidecar
    sidecar = save_dir / "phase4_baseline_sidecar.pkl"
    with sidecar.open("wb") as f:
        pickle.dump({
            "individual_results": results.get("individual_results", {}),
            "ensemble_results": results.get("ensemble_results", {}),
        }, f)

    # Summarize test R2 per model
    per_model_r2: dict[str, float] = {}
    ind = results.get("individual_results", {}) or {}
    for name, entry in ind.items():
        tm = entry.get("test_metrics_ar") or entry.get("test_metrics") or {}
        if "r2" in tm:
            per_model_r2[name] = float(tm["r2"])

    return {
        "loso_season_start": LOSO_SEASON_START,
        "loso_season_end": LOSO_SEASON_END,
        "n_train": int(len(train_idx)) - n_val,
        "n_val": int(n_val),
        "n_test": int(len(test_idx)),
        "per_model_test_r2": per_model_r2,
        "sidecar": str(sidecar),
    }


def _load_wfcv_last_fold_r2() -> dict[str, float]:
    """기존 E 의 Phase 7 checkpoint 에서 last-fold R2 를 뽑아 비교 대상 준비."""
    ckpt = ROOT / "simulation" / "results" / "checkpoints" / "checkpoint_phase7.json"
    if not ckpt.exists():
        return {}
    try:
        data = json.loads(ckpt.read_text(encoding="utf-8"))
    except Exception:
        return {}

    # Expected schema: {"per_model_wfcv": {name: {"fold_r2": [...], ...}}}
    per_model = data.get("per_model_wfcv") or data.get("per_model") or {}
    out: dict[str, float] = {}
    for name, stats in per_model.items():
        if isinstance(stats, dict):
            # Try last fold R2 from list, or fall back to mean
            if "fold_r2" in stats and stats["fold_r2"]:
                out[name] = float(stats["fold_r2"][-1])
            elif "r2_mean" in stats:
                out[name] = float(stats["r2_mean"])
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--primary-only", action="store_true",
                   help="Run only PAPER_PRIMARY_11 models (~1-2h vs full 4-5h)")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT),
                   help=f"Output directory (default: {DEFAULT_OUT})")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    save_dir = Path(args.out_dir)

    log.info(f"[r3_6] LOSO 2022-23 start  (primary_only={args.primary_only})")
    loso_res = _run_loso(primary_only=args.primary_only, save_dir=save_dir)
    log.info(f"[r3_6] LOSO run complete -> {loso_res['sidecar']}")

    # Compare with last-fold WF-CV
    wf_r2 = _load_wfcv_last_fold_r2()
    comparison = {
        "loso": loso_res,
        "wfcv_last_fold_r2": wf_r2,
        "delta_r2": {
            name: loso_res["per_model_test_r2"][name] - wf_r2.get(name, float("nan"))
            for name in loso_res["per_model_test_r2"]
            if name in wf_r2
        },
    }
    cmp_path = save_dir / "r3_6_comparison.json"
    cmp_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    log.info(f"[r3_6] comparison -> {cmp_path}")

    # Print headline
    if comparison["delta_r2"]:
        dr2 = list(comparison["delta_r2"].values())
        log.info(
            f"\n[r3_6] n_models_compared={len(dr2)}  "
            f"median(dR2 LOSO - WFCV)={np.median(dr2):+.4f}  "
            f"mean={np.mean(dr2):+.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
