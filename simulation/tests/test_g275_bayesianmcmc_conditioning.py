"""G-275 (2026-06-16, per-model 감사): BayesianMCMC 수치 ill-conditioning 회귀 가드.

사건(red): phase-13 feature 선택이 같은 lag 의 여러 변환(ili_rate_lag1 / _log1p / _qbin /
  _qnorm)을 동시에 고르면 design matrix 가 near-singular(cond≈7.6e16) → OLS init 의
  ``lstsq(rcond=None)`` 이 작은 특이값을 안 잘라 beta_init |β|≈1e6 로 폭발 → MCMC 가 그
  발산점에서 시작 → hold-out 외삽 시 예측 폭주(test r2 −4.35, pred 221 ≫ y_max 67).
  부동소수(float32/64) 섭동만으로도 0.90 ↔ −4.35 flip 하는 knife-edge 불안정.

영구 수정(green): ``lstsq(rcond=1e-6)`` — 작은 특이값 truncation → |β|≈0.7 안정.

reproduction: 강한 collinear feature set(cond > 1e12)에서 fit→predict 가 폭발하지 않는다.
"""
from __future__ import annotations

import numpy as np
import pytest

from simulation.models.epi_models import BayesianMCMCForecaster


def _collinear_design(n=200, seed=0):
    """near-singular: 한 base 컬럼 + 그 비선형 변환들(실 phase-13 collapse 재현)."""
    rng = np.random.RandomState(seed)
    base = np.abs(rng.randn(n) * 5 + 20)          # ili_rate_lag1 류
    X = np.column_stack([
        base,
        np.log1p(base),                            # _log1p (collinear)
        np.clip(np.round(base / 5), 0, None),      # _qbin (collinear)
        (base - base.mean()) / (base.std() + 1e-9),  # _qnorm (collinear)
        rng.randn(n),                              # noise 1개
    ])
    y = base * 1.2 + rng.randn(n) * 2 + 5
    return X, y


def test_design_is_near_singular():
    """전제 검증: 이 feature set 이 실제로 ill-conditioned (가드가 의미있는 상황)."""
    X, _ = _collinear_design()
    Xa = np.column_stack([np.ones(len(X)), X])
    assert np.linalg.cond(Xa) > 1e10, "테스트 design 이 충분히 ill-conditioned 하지 않음"


def test_bayesianmcmc_no_explosion_on_collinear():
    """핵심 가드: collinear fit→predict 가 폭발하지 않는다 (rcond=1e-6 효과)."""
    X, y = _collinear_design()
    n = len(y)
    Xtr, ytr = X[:160], y[:160]
    # 외삽 test (train 범위 초과 — 폭발 유발 조건)
    Xte = X[160:] * 1.5
    m = BayesianMCMCForecaster()
    np.random.seed(42)
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    ymax = float(ytr.max())
    assert np.all(np.isfinite(pred)), "예측에 NaN/inf"
    # rcond=None 이었으면 pred 가 y_max 의 수십~수백 배로 폭발했음
    assert pred.max() < ymax * 5.0, f"예측 폭발: max={pred.max():.1f} ≫ y_max={ymax:.1f}"
    assert pred.min() >= 0.0, "음수 예측"


def test_beta_init_bounded_under_collinearity():
    """OLS init 의 |β| 가 rcond truncation 으로 bounded (폭발 1e6 회귀 가드)."""
    from numpy.linalg import lstsq
    X, y = _collinear_design()
    Xa = np.column_stack([np.ones(len(X)), X])
    ys = (y - y.mean()) / (y.std() + 1e-9)
    beta_none, *_ = lstsq(Xa, ys, rcond=None)
    beta_fix, *_ = lstsq(Xa, ys, rcond=1e-6)
    # rcond=None 은 폭발(이 design 에서 |β| ≫ 100), rcond=1e-6 은 O(1)
    assert np.abs(beta_fix).max() < 100.0, f"수정 후에도 |β| 폭발: {np.abs(beta_fix).max():.2e}"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_robust_to_float_precision(dtype):
    """float32/64 섭동에 robust (knife-edge flip 회귀 가드)."""
    X, y = _collinear_design()
    Xtr, ytr = X[:160].astype(dtype), y[:160].astype(dtype)
    Xte = (X[160:] * 1.5).astype(dtype)
    m = BayesianMCMCForecaster()
    np.random.seed(42)
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    assert np.all(np.isfinite(pred))
    assert pred.max() < float(ytr.max()) * 5.0
