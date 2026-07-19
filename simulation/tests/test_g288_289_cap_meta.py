"""G-288/289 (2026-06-17, 53×3 AI 감사): META 누락 + 외삽 상한 cap 일괄.

G-288: Theta(univariate)·FluSight-Baseline(persistence) → META_MODELS(preproc Optuna skip, 100-trial 낭비 제거).
G-289: apply_extrapolation_cap(safety) — DL/modern-ts/graph/cqr 다수가 0-floor 만 있고 상한 없어
  outbreak 외삽 폭주(DNN/TCN/Mamba 만 cap 보유였음) → modern-ts 6·GCN·DLinear·CQR-QuantReg 에 적용.
"""
from __future__ import annotations

import numpy as np
import pytest


def test_g289_cap_helper():
    from simulation.models.safety import apply_extrapolation_cap
    assert apply_extrapolation_cap(np.array([5., 500.]), 100.0).tolist() == [5., 150.]   # 1.5×
    assert apply_extrapolation_cap(np.array([5., 50.]), 100.0).tolist() == [5., 50.]      # 미초과 보존
    assert apply_extrapolation_cap(np.array([5., 500.]), None).tolist() == [5., 500.]     # None 통과
    assert apply_extrapolation_cap(np.array([5., 500.]), 0.0).tolist() == [5., 500.]      # ≤0 통과
    # floor: 작은 y_max 도 최소 100 cap
    assert apply_extrapolation_cap(np.array([90.]), 10.0).tolist() == [90.]               # floor 100 미만 보존


def test_g288_theta_flusight_in_meta():
    import re
    src = open('simulation/pipeline/per_model_optimize.py', encoding='utf-8').read()
    m = re.search(r'META_MODELS\s*=\s*\{(.+?)\}', src, re.DOTALL)
    assert m, "META_MODELS 못 찾음"
    body = m.group(1)
    assert '"Theta"' in body, "Theta META 미등록 (preproc 낭비)"
    assert '"FluSight-Baseline"' in body, "FluSight-Baseline META 미등록"


def test_g289_dlinear_caps_extrapolation():
    """DLinear forecast 가 외삽서 상한 cap (옛 clip(0,None)은 하한만)."""
    from simulation.models.dlinear import DLinearForecaster
    rng = np.random.RandomState(0)
    series = np.abs(np.sin(np.arange(120) / 8.0) * 20 + 30 + rng.randn(120) * 2)
    m = DLinearForecaster()
    m.fit_series(series)
    assert hasattr(m, "_y_train_max") and m._y_train_max > 0
    pred = m.forecast(20)
    assert np.all(np.isfinite(pred))
    assert pred.max() <= max(m._y_train_max * 1.5, 100.0) + 1e-3, f"DLinear 외삽 폭주: {pred.max()}"


def test_g289_cqr_quantreg_caps():
    pytest.importorskip("statsmodels")
    from simulation.models.cqr_models import CQRQuantRegForecaster
    rng = np.random.RandomState(1)
    X = rng.randn(120, 8); y = np.abs(X[:, 0] * 5 + 20 + rng.randn(120) * 2)
    m = CQRQuantRegForecaster()
    m.fit(X, y)
    assert hasattr(m, "_y_train_max") and m._y_train_max > 0
    pred = m.predict(X[:30] * 6)   # 외삽
    assert np.all(np.isfinite(pred))
    assert pred.max() <= max(m._y_train_max * 1.5, 100.0) + 1e-3
