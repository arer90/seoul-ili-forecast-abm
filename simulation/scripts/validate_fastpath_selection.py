"""G-273c 실증 검증: fast(고정 HP) vs full(튜닝 HP) 비교가 *같은 선택*을 하는가?

사용자 의문(2026-06-15): "fast-path가 비교를 못 하면 의미 없지 않나? 통일성 위반 아닌가?"
→ 핵심 질문: 비교 단계(preproc transform / feature subset)에서 고정 HP로 매긴 순위가
   후보별 튜닝 HP로 매긴 순위와 일치하는가? 일치하면 fast-path 선택은 full 과 동등(타당).

방법: 현 run 이 쓰는 실제 feature_cache(351×399, target ili_rate)의 train 풀에서,
  Exp1 (transform 선택): 여러 y-transform 후보를 고정 feature 로 OOF-WIS 채점 — fast vs full.
  Exp2 (feature subset 선택): 여러 subset 크기를 고정 transform 으로 OOF-WIS 채점 — fast vs full.
각 (model, exp) 에서 Spearman 순위상관 + argmin(=선택) 일치 여부 보고.
현 run 무중단(읽기 전용 캐시 + 별도 프로세스). 격리: OMP_NUM_THREADS=1.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ.pop("MPH_INNER_HP_FAST", None)
os.environ["MPH_HP_OPTUNA_TRIALS"] = "8"   # full 모드 = 진짜 튜닝(8 trial) — tractable 하면서 HP 적응 포착

import numpy as np
import polars as pl

from simulation.pipeline.per_model_optimize import _evaluate_config
from simulation.models.tree_models import (
    XGBoostForecaster, LightGBMForecaster, RandomForestForecaster,
)


def _load_train_pool(n_test: int = 68):
    df = pl.read_parquet("simulation/cache/feature_cache.parquet")
    drop = [c for c in df.columns if c in ("week_start",)]
    feat_cols = [c for c in df.columns if c not in ("ili_rate",) + tuple(drop)]
    y = df["ili_rate"].to_numpy().astype(float)
    X = df.select(feat_cols).to_numpy().astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    n = len(y)
    n_tr = n - n_test
    return X[:n_tr], y[:n_tr], feat_cols


def _oof_wis(factory, X, y, transform_name, scaler_name, feature_indices, fast, n_folds=4):
    """_oof_cv_wis 와 동일 구조(folds×_evaluate_config), _fast_inner 만 토글."""
    n = len(X)
    if n < (n_folds + 1) * 10:
        return float("inf")
    fs = n // (n_folds + 1)
    scores = []
    for k in range(1, n_folds + 1):
        e_tr = k * fs
        e_va = (k + 1) * fs if k < n_folds else n
        Xtr, ytr, Xva, yva = X[:e_tr], y[:e_tr], X[e_tr:e_va], y[e_tr:e_va]
        if len(Xva) < 4:
            continue
        cell = _evaluate_config(
            factory, Xtr, ytr, Xva, yva,
            transform_name=transform_name, scaler_name=scaler_name,
            feature_indices=feature_indices,
            sigma_for_wis=max(float(np.std(ytr)), 1e-3),
            _fast_inner=fast,
        )
        if "error" not in cell and np.isfinite(cell.get("wis", float("inf"))):
            scores.append(float(cell["wis"]))
    return float(np.mean(scores)) if scores else float("inf")


def _spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a[m])); rb = np.argsort(np.argsort(b[m]))
    if np.std(ra) < 1e-9 or np.std(rb) < 1e-9:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


MODELS = [("XGBoost", XGBoostForecaster), ("LightGBM", LightGBMForecaster),
          ("RandomForest", RandomForestForecaster)]
TRANSFORMS = ["identity", "log1p", "sqrt"]   # 대표 y-transform 후보


def main():
    X, y, feat_cols = _load_train_pool()
    p = X.shape[1]
    print(f"train pool: {X.shape}, target ili_rate, p={p}\n")

    # Exp2 후보: nested subset 크기 (top-variance feature 로 대용 — 실제는 stability freq)
    var = X.var(axis=0)
    order = np.argsort(var)[::-1]
    subset_sizes = [9, 50, p]   # 작은→full
    subsets = {s: sorted(order[:s].tolist()) for s in subset_sizes}

    for mname, cls in MODELS:
        print(f"================ {mname} ================")
        fac = lambda c=cls: c()

        # ── Exp1: transform 선택 (feature 고정 = full pool) ──
        fast_t, full_t = [], []
        for t in TRANSFORMS:
            fast_t.append(_oof_wis(fac, X, y, t, "robust", None, fast=True))
            full_t.append(_oof_wis(fac, X, y, t, "robust", None, fast=False))
        bf = TRANSFORMS[int(np.argmin(fast_t))]
        bF = TRANSFORMS[int(np.argmin(full_t))]
        rho_t = _spearman(fast_t, full_t)
        print(f"  [Exp1 transform] fast WIS={[round(v,3) for v in fast_t]} → best={bf}")
        print(f"                   full WIS={[round(v,3) for v in full_t]} → best={bF}")
        print(f"     ⇒ 선택 일치: {bf==bF}  | Spearman ρ={rho_t:.3f}")

        # ── Exp2: feature subset 선택 (transform 고정 = identity) ──
        fast_s, full_s = [], []
        for s in subset_sizes:
            fi = None if s >= p else subsets[s]
            fast_s.append(_oof_wis(fac, X, y, "identity", "robust", fi, fast=True))
            full_s.append(_oof_wis(fac, X, y, "identity", "robust", fi, fast=False))
        bf2 = subset_sizes[int(np.argmin(fast_s))]
        bF2 = subset_sizes[int(np.argmin(full_s))]
        rho_s = _spearman(fast_s, full_s)
        print(f"  [Exp2 subset]    fast WIS={[round(v,3) for v in fast_s]} → best k={bf2}")
        print(f"                   full WIS={[round(v,3) for v in full_s]} → best k={bF2}")
        print(f"     ⇒ 선택 일치: {bf2==bF2}  | Spearman ρ={rho_s:.3f}\n")


if __name__ == "__main__":
    main()
