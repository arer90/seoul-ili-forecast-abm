"""DLinear (Zeng AAAI 2023, G-265) — 분해 + 단일 선형층, 소표본 강건 baseline.

웹 SOTA 커버리지 감사 후 add. "Are Transformers Effective?"의 그 baseline = 우리 결론
("소표본 341주서 단순 모델이 deep 능가")을 직접 입증. 실측 rolling 1-step r2=0.935 > ARIMA 0.915
> TimeMixer 0.366. 이 테스트는 등록·active·계약(USES_FEATURES=False)·분해선형·소표본 fallback 보장.

Run: KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 .venv/bin/python -m pytest simulation/tests/test_dlinear.py -x -q
"""
from __future__ import annotations

import numpy as np


def test_dlinear_registered_active():
    from simulation.models.registry import verify_registry_coverage, CATEGORY_MODELS
    verify_registry_coverage(force_import=True)
    from simulation.models.base import REGISTRY
    assert REGISTRY.get("DLinear") is not None
    assert "DLinear" in CATEGORY_MODELS["modern-ts"]


def test_dlinear_contract():
    from simulation.models.dlinear import DLinearForecaster
    from simulation.models.base import TimeSeriesForecaster
    f = DLinearForecaster()
    assert isinstance(f, TimeSeriesForecaster)
    assert f.USES_FEATURES is False
    assert f.meta.name == "DLinear" and f.meta.dependencies == []  # 순수 numpy — 의존성 0


def test_dlinear_fit_forecast_finite_nonneg():
    from simulation.models.dlinear import DLinearForecaster
    rng = np.random.default_rng(0); t = np.arange(160)
    y = np.clip(10 + 8 * np.sin(2 * np.pi * t / 52) + rng.normal(scale=0.5, size=t.size), 0, None)
    f = DLinearForecaster().fit_series(y[:120])
    p = f.forecast(40)
    assert p.shape == (40,) and np.all(np.isfinite(p)) and np.all(p >= 0)


def test_dlinear_small_sample_fallback():
    """표본이 lookback 보다 작으면 평균 fallback(크래시 없음)."""
    from simulation.models.dlinear import DLinearForecaster
    f = DLinearForecaster(lookback=26).fit_series(np.array([5.0, 6, 7, 8, 9, 10]))  # n=6 < L
    p = f.forecast(3)
    assert p.shape == (3,) and np.all(np.isfinite(p))


def test_dlinear_decomp_is_linear():
    """선형 추세 시계열 → DLinear가 합리적으로 외삽(단일 선형층 특성)."""
    from simulation.models.dlinear import DLinearForecaster
    y = np.linspace(5, 50, 120) + np.random.default_rng(1).normal(scale=0.3, size=120)
    f = DLinearForecaster(lookback=20).fit_series(y)
    p = f.forecast(5)
    assert p[0] > y[-1] * 0.7, "선형 증가 추세인데 예측이 과도히 낮음"


if __name__ == "__main__":
    test_dlinear_registered_active(); print("PASS registered+active")
    test_dlinear_contract(); print("PASS contract")
    test_dlinear_fit_forecast_finite_nonneg(); print("PASS fit/forecast")
    test_dlinear_small_sample_fallback(); print("PASS small-sample fallback")
    test_dlinear_decomp_is_linear(); print("PASS decomp linear")
    print("=== ALL PASS ===")
