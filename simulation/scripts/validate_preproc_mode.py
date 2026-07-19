"""G-273c-B 재검증: preproc 축소-탐색(k=5)이 transform 선택에서 full 을 추종하는가?

직전 검증(validate_fastpath_selection): fast 단일점이 XGBoost transform 을 full 과 다르게 골랐음
(fast=sqrt vs full=identity). 사용자 B 선택 = preproc 만 작은 HP 탐색(k=5). 이게 실제로 full 을
추종하는지 3-모드(fast 단일점 / preproc k=5 / full k=8) transform 선택으로 확인.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import polars as pl

from simulation.pipeline.per_model_optimize import _evaluate_config
from simulation.models.tree_models import (
    XGBoostForecaster, LightGBMForecaster, RandomForestForecaster,
)

TRANSFORMS = ["identity", "log1p", "sqrt"]
MODELS = [("XGBoost", XGBoostForecaster), ("LightGBM", LightGBMForecaster),
          ("RandomForest", RandomForestForecaster)]


def _oof(factory, X, y, t, mode, n_folds=4):
    """mode: 'fast'(단일점) / 'preproc'(k=5 study) / 'full'(k=8 study)."""
    n = len(X); fs = n // (n_folds + 1); scores = []
    for k in range(1, n_folds + 1):
        e_tr = k * fs; e_va = (k + 1) * fs if k < n_folds else n
        Xtr, ytr, Xva, yva = X[:e_tr], y[:e_tr], X[e_tr:e_va], y[e_tr:e_va]
        if len(Xva) < 4:
            continue
        os.environ.pop("MPH_INNER_HP_PREPROC_TRIALS", None)
        os.environ.pop("MPH_HP_OPTUNA_TRIALS", None)
        fast = False
        if mode == "preproc":
            os.environ["MPH_INNER_HP_PREPROC_TRIALS"] = "5"
        elif mode == "full":
            os.environ["MPH_HP_OPTUNA_TRIALS"] = "8"
        elif mode == "fast":
            fast = True
        cell = _evaluate_config(factory, Xtr, ytr, Xva, yva,
                                transform_name=t, scaler_name="robust",
                                sigma_for_wis=max(float(np.std(ytr)), 1e-3),
                                _fast_inner=fast)
        os.environ.pop("MPH_INNER_HP_PREPROC_TRIALS", None)
        os.environ.pop("MPH_HP_OPTUNA_TRIALS", None)
        if "error" not in cell and np.isfinite(cell.get("wis", float("inf"))):
            scores.append(float(cell["wis"]))
    return float(np.mean(scores)) if scores else float("inf")


def main():
    df = pl.read_parquet("simulation/cache/feature_cache.parquet")
    feat = [c for c in df.columns if c not in ("ili_rate", "week_start")]
    y = df["ili_rate"].to_numpy().astype(float)
    X = np.nan_to_num(df.select(feat).to_numpy().astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    n_tr = len(y) - 68
    X, y = X[:n_tr], y[:n_tr]
    print(f"train pool {X.shape}\n")

    for mname, cls in MODELS:
        fac = lambda c=cls: c()
        picks = {}
        for mode in ("fast", "preproc", "full"):
            wis = [_oof(fac, X, y, t, mode) for t in TRANSFORMS]
            picks[mode] = TRANSFORMS[int(np.argmin(wis))]
            print(f"  {mname:13s} [{mode:7s}] WIS={[round(v,3) for v in wis]} → best={picks[mode]}")
        match = "✅ preproc=full" if picks["preproc"] == picks["full"] else "❌ preproc≠full"
        was = "(fast 일치)" if picks["fast"] == picks["full"] else "(fast 불일치였음)"
        print(f"    ⇒ {match}  {was}\n")


if __name__ == "__main__":
    main()
