"""NegBinGLM-Glum (G-263, 2026-06-13) — glum elastic-net 진짜 NB-GLM.

소표본(341주) SOTA 서베이에서 **유일하게 incumbent 를 능가한** 후보 (TimeMixer 0.366·CQR-CatBoost
0.574·ExtraTrees 0.780 전부 패배; glum NB-GLM = 0.878 + peak 도달). V7(hard top-K)·NegBinGLM(V6
RidgeCV salvage)와 다른 접근 = elastic-net 연속 shrinkage(full pool) + log link(자연 비음수+외삽).
사용자 add 확정. 이 테스트는 등록·계약·비음수·peak 외삽·fallback·로그청결을 보장한다.

Run (macOS, 단일 파일):
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
      .venv/bin/python -m pytest simulation/tests/test_glum_nb.py -x -q
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest


def _synth(n=200, p=12, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = np.clip(10 + 25 * (np.sin(2 * np.pi * t / 52) ** 8) + rng.normal(scale=1.0, size=n), 0, None)
    X = np.column_stack([np.roll(y, k) for k in range(1, p + 1)]).astype(float)
    X[:p] = 0.0
    return X.astype(float), y.astype(float)


def test_glum_nb_registered_and_active():
    from simulation.models.registry import CATEGORY_MODELS, DEFER_MODELS, verify_registry_coverage
    verify_registry_coverage(force_import=True)
    from simulation.models.base import REGISTRY
    # Still REGISTERED (so --include-only can reach it), but no longer ACTIVE:
    # G-347 deferred it as an inferior duplicate of NegBinGLM V6 (R2 0.800 < 0.904).
    # Asserting active membership made that documented retirement look like a bug.
    assert REGISTRY.get("NegBinGLM-Glum") is not None, "NegBinGLM-Glum 미등록"
    assert "NegBinGLM-Glum" in DEFER_MODELS, (
        "NegBinGLM-Glum should be in DEFER_MODELS (G-347); if it was re-promoted, "
        "move this assertion back to CATEGORY_MODELS['linear']")
    assert "NegBinGLM-Glum" not in CATEGORY_MODELS.get("linear", [])
    active = [m for v in CATEGORY_MODELS.values() for m in v]
    assert len(active) == verify_registry_coverage()["total_expected"], (
        f"active {len(active)} disagrees with the registry coverage SSOT — "
        "the active lineup is deliberately reduced over time (53 -> 49 -> 48 -> ...), each step recorded in DEFER_MODELS with its reason, so a hardcoded count turns every documented retirement into a failure. verify_registry_coverage() is the live SSOT.")


def test_glum_nb_contract_nonneg():
    from simulation.models.negbin_glm import GlumNBForecaster, _w_glum_available
    if not _w_glum_available():
        pytest.skip("glum 미설치")
    X, y = _synth()
    tr, te = np.arange(160), np.arange(160, 200)
    m = GlumNBForecaster().fit(X[tr], y[tr])
    p = m.predict(X[te])
    assert p.shape == (40,)
    assert np.all(np.isfinite(p)), "예측에 NaN/inf"
    assert np.all(p >= -1e-9), "log link NB-GLM 인데 음수"
    assert m._fallback is False, "정상 데이터인데 V6 fallback (glum fit 실패)"


def test_glum_nb_can_exceed_train_max():
    """log link → 트리 cap 과 달리 train max 위로 외삽 가능(단 2×cap 안). peak 예측의 핵심."""
    from simulation.models.negbin_glm import GlumNBForecaster, _w_glum_available
    if not _w_glum_available():
        pytest.skip("glum 미설치")
    rng = np.random.default_rng(1)
    n = 180
    x = np.linspace(0, 6, n)
    y = np.exp(0.6 * x) + rng.normal(scale=0.5, size=n)   # 단조 증가 → test 가 train max 초과
    X = x.reshape(-1, 1)
    tr, te = np.arange(140), np.arange(140, 180)
    m = GlumNBForecaster(alpha=0.01).fit(X[tr], y[tr])
    p = m.predict(X[te])
    if not m._fallback:   # glum 정상 경로일 때만 외삽 특성 검증
        assert p.max() > y[tr].max() * 0.95, "log link 인데 train max 외삽 못 함 (tree-cap 거동)"
        assert p.max() <= 2.0 * y[tr].max() + 1e-6, "2×cap 초과 (발산 가드 실패)"


def test_glum_nb_no_warning_spam(recwarn):
    """fit 중 glum 의 benign matmul 경고가 caller 로 새지 않아야(np.errstate+catch_warnings)."""
    from simulation.models.negbin_glm import GlumNBForecaster, _w_glum_available
    if not _w_glum_available():
        pytest.skip("glum 미설치")
    X, y = _synth()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        GlumNBForecaster().fit(X[:160], y[:160])
        spam = [w for w in rec if "matmul" in str(w.message) or "overflow" in str(w.message)]
        assert not spam, f"glum matmul 경고가 새어나옴: {len(spam)}"


if __name__ == "__main__":
    test_glum_nb_registered_and_active(); print("PASS  registered + active 51")
    test_glum_nb_contract_nonneg(); print("PASS  contract nonneg")
    test_glum_nb_can_exceed_train_max(); print("PASS  peak 외삽")
    test_glum_nb_no_warning_spam(); print("PASS  no warning spam")
    print("=== ALL PASS ===")
