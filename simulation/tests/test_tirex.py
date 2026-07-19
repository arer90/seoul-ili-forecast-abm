"""TiRex (NX-AI xLSTM foundation, G-265) — zero-shot, transformers-free.

웹 SOTA 감사 후 add. ILI rolling 1-step r2=0.944(전 모델 최고). transformers 의존 없음(xLSTM) →
메인 env(mlx-lm/ARIA) 충돌 0 (Chronos 가 퇴출된 충돌 회피). 이 테스트는 등록·foundation active 54·
계약(USES_FEATURES=False, TimeSeriesForecaster)·transformers-free·실 forecast 를 보장.

Run: KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 .venv/bin/python -m pytest simulation/tests/test_tirex.py -x -q
"""
from __future__ import annotations

import os

import numpy as np
import pytest


def test_tirex_registered_foundation_active():
    from simulation.models.registry import verify_registry_coverage, verify_registry_coverage, CATEGORY_MODELS
    verify_registry_coverage(force_import=True)
    from simulation.models.base import REGISTRY
    assert REGISTRY.get("TiRex") is not None, "TiRex 미등록"
    assert CATEGORY_MODELS["foundation"] == ["TimesFM-2.5", "OverseasTransfer", "TiRex", "FusedEpi"]
    active = [m for v in CATEGORY_MODELS.values() for m in v]
    assert len(active) == verify_registry_coverage()["total_expected"], (
        f"active {len(active)} disagrees with the registry coverage SSOT — "
        "the active lineup is deliberately reduced over time (53 -> 49 -> 48 -> ...), each step recorded in DEFER_MODELS with its reason, so a hardcoded count turns every documented retirement into a failure. verify_registry_coverage() is the live SSOT.")


def test_tirex_contract():
    from simulation.models.tirex_wrapper import TiRexForecaster
    from simulation.models.base import TimeSeriesForecaster
    f = TiRexForecaster()
    assert isinstance(f, TimeSeriesForecaster)
    assert f.USES_FEATURES is False, "foundation 은 X 미사용"
    assert f.meta.name == "TiRex" and "tirex-ts" in f.meta.dependencies
    assert hasattr(f, "fit_series") and hasattr(f, "forecast")


def test_tirex_transformers_free():
    """tirex-ts 가 transformers 를 *요구*하지 않음 (Chronos 와 같은 mlx-lm 충돌 위험 0)."""
    import importlib.metadata as md
    try:
        reqs = md.requires("tirex-ts") or []
    except md.PackageNotFoundError:
        pytest.skip("tirex-ts 미설치")
    tf = [r for r in reqs if r.lower().startswith("transformers")]
    assert tf == [], f"tirex-ts 가 transformers 의존: {tf}"


@pytest.mark.skipif(os.environ.get("MPH_SKIP_HEAVY_TESTS") == "1",
                    reason="MPH_SKIP_HEAVY_TESTS=1 (35M 모델 로드 생략)")
def test_tirex_fit_forecast():
    from simulation.models.tirex_wrapper import TiRexForecaster, _w_tirex_available
    if not _w_tirex_available():
        pytest.skip("tirex-ts 미설치")
    rng = np.random.default_rng(0); t = np.arange(160)
    y = np.clip(10 + 8 * np.sin(2 * np.pi * t / 52) + rng.normal(scale=0.5, size=t.size), 0, None)
    f = TiRexForecaster().fit_series(y[:120])
    p = f.forecast(8)
    assert p.shape == (8,) and np.all(np.isfinite(p)) and np.all(p >= 0)


if __name__ == "__main__":
    test_tirex_registered_foundation_active(); print("PASS registered+foundation 54")
    test_tirex_contract(); print("PASS contract")
    test_tirex_transformers_free(); print("PASS transformers-free")
    test_tirex_fit_forecast(); print("PASS fit/forecast")
    print("=== ALL PASS ===")
