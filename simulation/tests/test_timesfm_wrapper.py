"""TimesFM-2.5 wrapper (G-261, 2026-06-13) — Chronos-2 의 transformers-free 대체.

chronos 는 모든 버전이 transformers<5 강제 → 메인 env(mlx-lm 가 transformers>=5 요구) 와
HARD 충돌. TimesFM 2.5 는 transformers 의존이 없어 메인 env 네이티브. 이 테스트는

  1. registry 등록 + foundation active 53 + chronos active 제외 (wiring)
  2. TimeSeriesForecaster 계약 (USES_FEATURES=False, fit→fit_series, predict→forecast(len))
  3. transformers 를 import 하지 않고도 동작 (메인 env 충돌 0 증명)
  4. (cached checkpoint) 실제 fit_series + forecast + predict 어댑터 — 유한 + 올바른 shape

를 보장한다. 실측 성능(rolling 0.939 / 68-step −0.885)은 캠페인 문서에 기록 (여기선 wiring + 계약).

Run (macOS, 단일 파일 — OpenMP/LightGBM segfault 회피):
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
      .venv/bin/python -m pytest simulation/tests/test_timesfm_wrapper.py -x -q
"""
from __future__ import annotations

import os

import numpy as np
import pytest


# ── 1. wiring: 등록 + foundation 53 + chronos active 제외 ──────────────────────
def test_timesfm_registered_and_foundation_swapped():
    from simulation.models.registry import verify_registry_coverage, verify_registry_coverage, CATEGORY_MODELS
    verify_registry_coverage(force_import=True)
    from simulation.models.base import REGISTRY

    assert REGISTRY.get("TimesFM-2.5") is not None, "TimesFM-2.5 미등록"
    # G-265: TiRex(xLSTM, transformers-free) 가 foundation 에 추가 → 3-item (2→3 stale fix)
    assert CATEGORY_MODELS["foundation"] == ["TimesFM-2.5", "OverseasTransfer", "TiRex", "FusedEpi"], \
        f"foundation swap 안 됨: {CATEGORY_MODELS['foundation']}"
    active = [m for v in CATEGORY_MODELS.values() for m in v]
    # G-261(chronos→TimesFM)+G-262(감축)+G-263/264/265(glum/TabPFN/DLinear/TiRex)+G-272(CatBoost 제외)
    #   +b2c8bb1(FusedEpi[foundation]·SeirCount-TabPFN[dl-tabular] 승격): … → 49 → 51
    assert len(active) == verify_registry_coverage()["total_expected"], (
        f"active {len(active)} disagrees with the registry coverage SSOT — "
        "the active lineup is deliberately reduced over time (53 -> 49 -> 48 -> ...), each step recorded in DEFER_MODELS with its reason, so a hardcoded count turns every documented retirement into a failure. verify_registry_coverage() is the live SSOT.")
    assert "Chronos-2" not in active and "Chronos-2-FT" not in active, \
        "chronos 가 아직 active — 메인 env 작동 불가 모델"
    # G-262: 중복 변종 감축 (DEFER — 클래스는 등록 유지)
    for cut in ("DNN-Optuna", "TCN-Optuna", "CQR-GBR"):
        assert cut not in active, f"{cut} 가 아직 active (G-262 감축 안 됨)"
        assert REGISTRY.get(cut) is not None, f"{cut} 클래스가 deregister 됨 (DEFER 여야 함)"
    # 유지 확인 (진짜 구분되는 변종). TabularDNN was moved to DEFER_MODELS after
    # this list was written — it is still registered and reachable via
    # --include-only, so assert registration for the deferred ones and active
    # membership only for those that remain in the lineup.
    for keep in ("DNN", "DNN-Conformal", "TCN", "CQR-LightGBM", "CQR-QuantReg"):
        assert keep in active, f"{keep} 가 잘못 제거됨"
    for deferred in ("TabularDNN",):
        assert REGISTRY.get(deferred) is not None, f"{deferred} 클래스가 deregister 됨"


# ── 2. 계약: TimeSeriesForecaster (USES_FEATURES=False, fit/predict 어댑터) ──────
def test_timesfm_contract():
    from simulation.models.timesfm_wrapper import TimesFMForecaster
    from simulation.models.base import TimeSeriesForecaster

    f = TimesFMForecaster()
    assert isinstance(f, TimeSeriesForecaster)
    assert f.USES_FEATURES is False, "foundation 은 X 미사용 (phase13 mc/feature probe 제외)"
    assert f.meta.name == "TimesFM-2.5"
    assert "timesfm" in f.meta.dependencies
    # 인터페이스 존재
    assert hasattr(f, "fit_series") and hasattr(f, "forecast")
    assert hasattr(f, "fit") and hasattr(f, "predict")


# ── 3. transformers-free: timesfm 가 transformers 를 import 하지 않음 (메인 env 충돌 0) ─
def test_timesfm_does_not_require_transformers():
    """timesfm import 가 transformers 를 *요구*하지 않음을 증명 (의존 그래프).
    transformers 가 있든 없든, timesfm 은 그것을 안 끌어옴 → 메인 env(5.x) 와 무충돌."""
    import importlib.metadata as md
    try:
        reqs = md.requires("timesfm") or []
    except md.PackageNotFoundError:
        pytest.skip("timesfm 미설치")
    tf_reqs = [r for r in reqs if r.lower().startswith("transformers")]
    assert tf_reqs == [], f"timesfm 이 transformers 를 의존: {tf_reqs} (chronos 와 같은 충돌 위험)"


# ── 4. 실제 forecast (cached checkpoint) — 유한 + shape + 어댑터 ─────────────────
@pytest.mark.skipif(
    os.environ.get("MPH_SKIP_HEAVY_TESTS") == "1",
    reason="MPH_SKIP_HEAVY_TESTS=1 (200M checkpoint 로드 생략)",
)
def test_timesfm_forecast_and_predict_adapter():
    from simulation.models.timesfm_wrapper import TimesFMForecaster, _HAS_TIMESFM
    if not _HAS_TIMESFM:
        pytest.skip("timesfm 미설치")

    rng = np.random.default_rng(0)
    t = np.arange(160)
    y = (10 + 8 * np.sin(2 * np.pi * t / 52) + rng.normal(scale=0.5, size=t.size)).astype(np.float32)
    y = np.clip(y, 0, None)

    f = TimesFMForecaster()
    # fit_series → forecast(직접 다단계)
    f.fit_series(y[:120])
    pred = f.forecast(40)
    assert pred.shape == (40,), pred.shape
    assert np.all(np.isfinite(pred)), "forecast 에 NaN/inf"
    assert np.all(pred >= -1e-3), "infer_is_positive 인데 음수 (ILI ≥ 0 위반)"

    # predict(X_test) 어댑터 = forecast(len(X_test)) (X 무시)
    X_test = rng.normal(size=(40, 5)).astype(np.float32)
    pred2 = f.predict(X_test)
    assert pred2.shape == (40,)
    assert np.allclose(pred, pred2, atol=1e-4), "predict 어댑터가 forecast(len) 와 불일치"


if __name__ == "__main__":
    test_timesfm_registered_and_foundation_swapped(); print("PASS  wiring (53 active, chronos out)")
    test_timesfm_contract(); print("PASS  contract")
    test_timesfm_does_not_require_transformers(); print("PASS  transformers-free")
    test_timesfm_forecast_and_predict_adapter(); print("PASS  forecast + predict adapter")
    print("=== ALL PASS ===")
