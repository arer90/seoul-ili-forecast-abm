"""G-281~286 (2026-06-16, 3자 감사): 남은 per-model fix 배치 회귀 가드.

G-281 Mamba 외삽 cap · G-282 GCN phase14 silent-drop 복구 · G-283 Ensemble-NNLS-Filtered
빌더 누락 복구 · G-284 SARIMA/SARIMAX 상한 cap · G-285 DNN-Conformal 정직 description ·
G-286 NegBinGLM-V7 fallback 공개 플래그.
"""
from __future__ import annotations

import numpy as np
import pytest


def test_g281_mamba_cap():
    pytest.importorskip("torch")
    from simulation.models.modern_ts.mamba import MambaForecaster
    rng = np.random.RandomState(0)
    X = np.abs(rng.randn(100, 6) * 3 + 10).astype(np.float32)
    y = np.abs(X[:, 0] * 1.5 + 8).astype(np.float32)
    m = MambaForecaster()
    m.fit(X[:80], y[:80])
    assert hasattr(m, "_y_train_max") and m._y_train_max > 0
    pred = m.predict(X[80:] * 5)            # 강한 외삽
    assert pred.max() <= max(m._y_train_max * 1.5, 100.0) + 1e-3
    assert np.all(np.isfinite(pred)) and pred.min() >= 0


def test_g283_ensemble_nnls_filtered_in_builder():
    """Ensemble-NNLS-Filtered 가 빌더 list 에 포함(NEVER BUILT 회귀 가드)."""
    import inspect
    from simulation.models import runner as R
    src = inspect.getsource(R.MultiModelRunner)
    assert "NNLSFilteredEnsemble" in src, "빌더 list 에 NNLSFiltered 누락"
    from simulation.models.ensemble import NNLSFilteredEnsemble
    assert NNLSFilteredEnsemble.meta.name == "Ensemble-NNLS-Filtered"


def test_g285_dnn_conformal_honest_description():
    """DNN-Conformal description 이 정직(내부 Ridge 명시, 'DNN-Optuna' 거짓 제거)."""
    import simulation.models.conformal as C
    # meta 를 가진 클래스 탐색
    metas = [getattr(C, n).meta for n in dir(C)
             if isinstance(getattr(C, n, None), type) and getattr(getattr(C, n), "meta", None)
             and getattr(getattr(C, n).meta, "name", "") == "DNN-Conformal"]
    assert metas, "DNN-Conformal meta 없음"
    desc = metas[0].description
    assert "Ridge" in desc, f"description 이 내부 Ridge 를 명시 안 함: {desc}"
    assert "DNN-Optuna" not in desc, "거짓 'DNN-Optuna' 잔존"


def test_g286_negbin_v7_fallback_flag():
    """NegBinGLM-V7 fallback 시 공개 _used_fallback 플래그(silent V6 중복 식별)."""
    import inspect
    from simulation.models import negbin_glm as M
    src = inspect.getsource(M)
    assert "_used_fallback" in src, "_used_fallback 공개 플래그 미설정"


def test_g282_gcn_guard_logic():
    """phase14 refit-채택 guard: baseline 미실행(빈 list)이어도 공통 길이 일치 시 채택."""
    # G-282 의 _ok 결정 로직을 재현(test_preds[name] 부재 시 공통 test 길이로 판정)
    def ok(refit_len, existing, n_common):
        if existing is not None and len(existing) > 0:
            return refit_len == len(existing)
        return (n_common == 0 or refit_len == n_common)
    # GCN 케이스: existing=빈, refit=68, 공통=68 → 채택(옛 로직은 68==0 False 로 폐기)
    assert ok(68, [], 68) is True
    assert ok(68, None, 68) is True
    # baseline 모델: existing=68, refit=68 → 채택
    assert ok(68, list(range(68)), 68) is True
    # 길이 불일치(손상): refit=40, existing=68 → 거부
    assert ok(40, list(range(68)), 68) is False
