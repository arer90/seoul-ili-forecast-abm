"""G-275 base layer: linear/kernel 외삽 폭발 cap 회귀 가드.

SVR-Linear / ElasticNet / KRR 는 train 범위 밖 test feature 에서 선형 외삽 →
ill-conditioned 시 예측 폭주 가능. 2×y_train_max cap = 정상 예측엔 no-op, 폭발만 bound.
count 가족(NegBinGLM 2×y_max)과 동형. (SVR-RBF=local kernel=bounded, 미적용.)
"""
from __future__ import annotations

import numpy as np

from simulation.models.linear_models import (
    _cap_linear_extrapolation,
    SVRLinearForecaster,
    ElasticNetForecaster,
    KRRForecaster,
)


def test_cap_noop_on_normal():
    """정상 예측(≤2×y_max)엔 cap 이 no-op."""
    pred = np.array([0.0, 5.0, 30.0, 66.0, 90.0])
    out = _cap_linear_extrapolation(pred, y_train_max=66.9)   # 2× = 133.8
    assert np.allclose(out, pred), "정상 예측을 잘못 clip"


def test_cap_bounds_explosion():
    """폭발 예측(221)을 2×y_max 로 bound."""
    pred = np.array([4.0, 50.0, 221.5, 159.0])
    out = _cap_linear_extrapolation(pred, y_train_max=66.9)
    assert out.max() <= 2.0 * 66.9 + 1e-9, f"cap 미작동: {out.max()}"
    assert out[0] == 4.0 and out[1] == 50.0, "정상값 보존 실패"


def test_cap_none_passthrough():
    """y_train_max=None(구버전 artifact) → 통과(back-compat)."""
    pred = np.array([4.0, 500.0])
    assert np.allclose(_cap_linear_extrapolation(pred, None), pred)


def _toy(n=160, p=8, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = np.abs(X[:, 0] * 8 + 20 + rng.randn(n) * 2)
    return X, y


def test_models_store_ymax_and_predict_bounded():
    """3 모델 fit→_y_train_max 저장 + 외삽 test 예측 bounded."""
    X, y = _toy()
    Xtr, ytr = X[:120], y[:120]
    Xte = X[120:] * 3.0   # 외삽
    for Cls in (SVRLinearForecaster, ElasticNetForecaster, KRRForecaster):
        m = Cls()
        m.fit(Xtr, ytr)
        assert hasattr(m, "_y_train_max") and m._y_train_max > 0, f"{Cls.__name__}: _y_train_max 미저장"
        pred = m.predict(Xte)
        assert np.all(np.isfinite(pred)), f"{Cls.__name__}: non-finite"
        assert pred.max() <= 2.0 * m._y_train_max + 1e-6, f"{Cls.__name__}: cap 초과 {pred.max():.1f}"
        assert pred.min() >= 0.0, f"{Cls.__name__}: 음수"


def test_normal_prediction_unchanged_by_cap():
    """정상 in-range test 예측은 cap 영향 0 (회귀 가드: 챔피언급 성능 보존)."""
    X, y = _toy(seed=1)
    Xtr, ytr = X[:120], y[:120]
    Xte = X[120:]   # in-range
    m = SVRLinearForecaster()
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    # in-range 예측은 2×y_max 한참 아래 → cap 무관
    assert pred.max() < 2.0 * m._y_train_max, "in-range 가 cap 에 닿음(비정상)"
