"""TabPFN v2 wrapper (G-264, 2026-06-13) — tabular foundation model (소표본 SOTA).

SOTA 서베이서 glum 과 함께 incumbent 를 능가한 2번째 add (ILI hold-out r2=0.917 최우수).
가중치 = 공개 HF repo + 공식 model_path (priorlabs-1-1 학술 무료, 사용자 확정). 이 테스트는
등록·active 52·계약(USES_FEATURES=True)·실 fit/predict·라이선스-경로(공개가중치)를 보장.

Run (macOS, 단일 파일):
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
      .venv/bin/python -m pytest simulation/tests/test_tabpfn.py -x -q
"""
from __future__ import annotations

import os

import numpy as np
import pytest


def test_tabpfn_registered_and_active():
    from simulation.models.registry import verify_registry_coverage, verify_registry_coverage, CATEGORY_MODELS
    verify_registry_coverage(force_import=True)
    from simulation.models.base import REGISTRY
    assert REGISTRY.get("TabPFN") is not None, "TabPFN 미등록"
    assert "TabPFN" in CATEGORY_MODELS["dl-tabular"]
    active = [m for v in CATEGORY_MODELS.values() for m in v]
    assert len(active) == verify_registry_coverage()["total_expected"], (
        f"active {len(active)} disagrees with the registry coverage SSOT — "
        "the active lineup is deliberately reduced over time (53 -> 49 -> 48 -> ...), each step recorded in DEFER_MODELS with its reason, so a hardcoded count turns every documented retirement into a failure. verify_registry_coverage() is the live SSOT.")


def test_tabpfn_contract():
    from simulation.models.tabpfn_wrapper import TabPFNForecaster
    from simulation.models.base import BaseForecaster
    f = TabPFNForecaster()
    assert isinstance(f, BaseForecaster)
    assert f.USES_FEATURES is True, "TabPFN 은 tabular feature 사용"
    assert f.meta.name == "TabPFN" and "tabpfn" in f.meta.dependencies
    assert hasattr(f, "fit") and hasattr(f, "predict")


@pytest.mark.skipif(os.environ.get("MPH_SKIP_HEAVY_TESTS") == "1",
                    reason="MPH_SKIP_HEAVY_TESTS=1 (42MB 가중치 로드 생략)")
def test_tabpfn_fit_predict():
    from simulation.models.tabpfn_wrapper import TabPFNForecaster, _w_tabpfn_available
    if not _w_tabpfn_available():
        pytest.skip("tabpfn 미설치")
    rng = np.random.default_rng(0)
    t = np.arange(160)
    y = np.clip(10 + 8 * np.sin(2 * np.pi * t / 52) + rng.normal(scale=0.5, size=t.size), 0, None)
    X = np.column_stack([np.roll(y, k) for k in range(1, 9)]).astype(float); X[:8] = 0
    tr, te = np.arange(120), np.arange(120, 160)
    f = TabPFNForecaster().fit(X[tr], y[tr])
    p = f.predict(X[te])
    assert p.shape == (40,) and np.all(np.isfinite(p)), "예측 shape/유한 실패"


def test_tabpfn_uses_public_weights_offline():
    """가중치 확보 경로가 공개 repo + model_path(offline) 인지 — 토큰 플로우 비의존 증명."""
    import simulation.models.tabpfn_wrapper as tw
    assert tw._HF_REPO == "Prior-Labs/TabPFN-v2-reg"   # 공개(non-gated) repo
    assert tw._CKPT_NAME.endswith(".ckpt")
    # _ensure_weights 는 tabpfn 미설치 시 None (정식 토큰 폴백) — 크래시 없음
    if not tw._w_tabpfn_available():
        assert tw._ensure_weights() is None


if __name__ == "__main__":
    test_tabpfn_registered_and_active(); print("PASS  registered + active 52")
    test_tabpfn_contract(); print("PASS  contract")
    test_tabpfn_uses_public_weights_offline(); print("PASS  public-weights offline")
    test_tabpfn_fit_predict(); print("PASS  fit/predict")
    print("=== ALL PASS ===")
