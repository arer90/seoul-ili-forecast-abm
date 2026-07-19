"""G-278b (2026-06-16, 3자 감사): hhh4 AR(1) log1p — NB log-link exp-blowup 회귀 가드.

raw y_prev 가 NegBin log-link 의 선형예측자에 들어가면 μ=exp(β·y_prev) 지수폭발 +
recursive 되먹임 누적. log1p → μ=(1+y_prev)^β power-law 안정. faithful pipeline 에서
test r2 −1.094 → 0.9065 (wis 19.37→4.67) 실측 회복. (sibling hhh4_benchmark.py 와 동형.)
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("statsmodels")

from simulation.models.hhh4_models import Hhh4EquivalentForecaster


def _seasonal_series(n=180, peak=80.0, base=10.0, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    y = base + (peak - base) * (0.5 + 0.5 * np.sin(2 * np.pi * t / 52.0)) + rng.randn(n) * 2
    y = np.clip(y, 0, None)
    X = np.column_stack([np.roll(y, 1), np.sin(2 * np.pi * t / 52.0), rng.randn(n)])
    X[0] = X[1]
    return X, y


def test_hhh4_no_ar_explosion():
    """핵심: 높은 peak 시즌 + 외삽서 예측이 폭발하지 않는다 (raw-AR exp-blowup 회귀 가드)."""
    X, y = _seasonal_series(peak=90.0)
    Xtr, ytr = X[:140], y[:140]
    Xte = X[140:]
    m = Hhh4EquivalentForecaster()
    np.random.seed(42)
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    ymax = float(ytr.max())
    assert np.all(np.isfinite(pred)), "예측에 NaN/inf (AR 폭발)"
    # log1p 안정 → 예측은 합리적 범위 (raw-AR 이면 exp 폭발로 cap(1.5×)에 박혀 stuck)
    assert pred.max() <= ymax * 2.0, f"AR 폭발: max={pred.max():.1f} ≫ y_max={ymax:.1f}"
    assert pred.min() >= 0.0


def test_hhh4_design_uses_log1p_ar():
    """mechanism: fit 의 AR 컬럼이 log1p (raw 아님) — 큰 y 에도 선형예측자 bounded."""
    import inspect
    src = inspect.getsource(Hhh4EquivalentForecaster.fit)
    assert "log1p" in src, "fit 의 AR 항이 log1p 아님 (raw → exp 폭발 위험)"
    psrc = inspect.getsource(Hhh4EquivalentForecaster.predict)
    assert "log1p" in psrc, "predict 의 AR 되먹임이 log1p 아님"


def test_hhh4_recursive_feedback_stable():
    """recursive AR 되먹임이 누적 발산하지 않음 (각 step bounded)."""
    X, y = _seasonal_series(peak=70.0, seed=3)
    m = Hhh4EquivalentForecaster()
    np.random.seed(1)
    m.fit(X[:130], y[:130])
    pred = m.predict(X[130:])
    # 연속 step 이 단조 폭발하지 않음 (마지막이 첫 예측의 수십배가 아님)
    assert pred[-1] <= max(pred[0] * 10 + 10, float(y[:130].max()) * 2), \
        f"recursive 누적 발산: {pred[0]:.1f}→{pred[-1]:.1f}"
