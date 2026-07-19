"""G-268 smoke — XGBoost early_stopping: 정상 fit+predict 검증 + per-fit wall-time 실측.

목적: (1) early_stopping 적용 후에도 fit/predict 정상(shape·finite·best_iteration<200) 확인,
      (2) inner Optuna(20-trial×3-fold) 1회 fit wall-time 측정 → preproc=100 곱해 phase-13
          XGBoost 총시간 추정 → "preproc trial 수 적정성" 권고의 실측 근거.

run: KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=2 .venv/bin/python -m simulation.scripts.smoke_xgb_earlystop
"""
from __future__ import annotations

import os
import time

import numpy as np

# inner-study 예산을 운영값과 동일하게 (lean=15, default=20). 미설정시 20.
os.environ.setdefault("MPH_HP_OPTUNA_TRIALS", "20")


def _synth(n=260, p=13, seed=42):
    """phase 4-12 BASIC feature 형태(n≈260, lag+계절 13개) 모사. 비음수 ILI-like target."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float64)
    # 계절성 + lag 의존 + 노이즈 → 외삽 아닌 in-range target
    t = np.arange(n)
    y = (2.0 + 1.5 * np.sin(2 * np.pi * t / 52)
         + 0.6 * X[:, 0] + 0.3 * X[:, 1] + rng.normal(scale=0.4, size=n))
    return X, np.clip(y, 0, None)


def main():
    from simulation.models.tree_models import XGBoostForecaster

    X, y = _synth()
    Xtr, ytr, Xte = X[:-40], y[:-40], X[-40:]
    print(f"[smoke] data: X={X.shape}, y∈[{y.min():.2f},{y.max():.2f}], "
          f"MPH_HP_OPTUNA_TRIALS={os.environ['MPH_HP_OPTUNA_TRIALS']}")

    m = XGBoostForecaster()
    t0 = time.perf_counter()
    m.fit(Xtr, ytr)
    dt = time.perf_counter() - t0
    pred = m.predict(Xte)

    # 정상성 검증
    n_trees = getattr(m._model, "best_iteration", None)
    booster_trees = m._model.get_booster().num_boosted_rounds()
    assert pred.shape == (40,), f"predict shape {pred.shape} != (40,)"
    assert np.all(np.isfinite(pred)), "predict 에 NaN/inf"
    print(f"[smoke] ✓ fit+predict 정상 — pred∈[{pred.min():.2f},{pred.max():.2f}], "
          f"best_iteration={n_trees}, booster_rounds={booster_trees} (200 미만이면 early_stop 작동)")

    # 시간 추정
    print(f"\n[실측] inner-study 1회 fit wall-time = {dt:.1f}s")
    print(f"[추정] phase-13 XGBoost 총 = preproc(100) + feature(20) + final(1) ≈ 121회 fit")
    print(f"         → 100×{dt:.1f}s(preproc) ≈ {100*dt/60:.1f}분, 전체 121회 ≈ {121*dt/60:.1f}분")
    print(f"[추정] lean(preproc=10) 면 10×{dt:.1f}s ≈ {10*dt/60:.1f}분 + feature/final")
    if booster_trees < 200:
        print(f"[판정] ✓ early_stopping 작동 ({booster_trees}<200 그루) — n_estimators=200 낭비 제거")
    else:
        print(f"[판정] ⚠ 200그루 전부 학습 — early_stop 미발동(데이터가 계속 개선됐거나 미적용)")


if __name__ == "__main__":
    main()
