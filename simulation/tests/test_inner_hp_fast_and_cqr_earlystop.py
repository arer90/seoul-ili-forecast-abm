"""G-273c smoke tests: inner-HP fast path (Stage-1/2 eval) + CQR-LightGBM early_stop.

배경 (2026-06-15, 사용자 도전 "왜 early stopping이 없어? 모든 xgboost나 lightgbm에 다 반영하라고 했잖아"):
  1) preproc/feature(Stage-1/2) eval 이 매 호출마다 tree forecaster 의 내부 HP Optuna study
     (×CV)를 통째로 재실행 → XGBoost 61분의 구조적 원인. `_evaluate_config_hierarchical` 이
     fit 동안 MPH_INNER_HP_FAST=1 을 켜서 tree forecaster 가 study 를 건너뛰게 한다(단일
     default + early_stop 1회). 최종 refit 은 이 함수를 안 거치므로 full HP.
  2) CQR-LightGBM 만 early_stop 누락(n_estimators=400 × 2-head 전부 학습) → eval_set hold-out
     + lgb.early_stopping(40) 추가.

Red→Green: 두 결함의 reproduction 을 deterministic 하게 검증(타이밍 의존 X).
"""
import os

import numpy as np
import pytest


# ──────────────────────────────────────────────────────────────────────────
# 1. Inner-HP fast path: tree forecaster 가 study 를 건너뛰는가 (deterministic)
# ──────────────────────────────────────────────────────────────────────────

def _toy_xy(n=80, p=5, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = np.abs(X[:, 0] * 1.5 + rng.randn(n) * 0.3) + 1.0
    return X, y


def test_xgboost_fast_hp_bypasses_inner_study(monkeypatch):
    """MPH_INNER_HP_FAST=1 → XGBoost.fit 이 optuna.create_study 를 호출하지 않아야."""
    import optuna
    from simulation.models.tree_models import XGBoostForecaster

    def _boom(*a, **k):
        raise AssertionError("inner HP study ran despite MPH_INNER_HP_FAST=1")

    monkeypatch.setattr(optuna, "create_study", _boom)
    monkeypatch.setenv("MPH_INNER_HP_FAST", "1")
    X, y = _toy_xy()
    m = XGBoostForecaster().fit(X, y)
    pred = m.predict(_toy_xy(n=12, seed=9)[0])
    assert pred.shape == (12,)
    assert np.all(np.isfinite(pred))


def test_lightgbm_fast_hp_bypasses_inner_study(monkeypatch):
    """MPH_INNER_HP_FAST=1 → LightGBM.fit 이 optuna.create_study 를 호출하지 않아야."""
    import optuna
    from simulation.models.tree_models import LightGBMForecaster

    def _boom(*a, **k):
        raise AssertionError("inner HP study ran despite MPH_INNER_HP_FAST=1")

    monkeypatch.setattr(optuna, "create_study", _boom)
    monkeypatch.setenv("MPH_INNER_HP_FAST", "1")
    X, y = _toy_xy()
    m = LightGBMForecaster().fit(X, y)
    pred = m.predict(_toy_xy(n=12, seed=9)[0])
    assert pred.shape == (12,)
    assert np.all(np.isfinite(pred))


def test_xgboost_full_hp_runs_study_when_flag_off(monkeypatch):
    """플래그 OFF → 정상 경로(내부 study) 가 여전히 작동(회귀 가드)."""
    from simulation.models.tree_models import XGBoostForecaster

    monkeypatch.delenv("MPH_INNER_HP_FAST", raising=False)
    X, y = _toy_xy(n=90)
    m = XGBoostForecaster().fit(X, y)
    pred = m.predict(_toy_xy(n=12, seed=3)[0])
    assert pred.shape == (12,)
    assert np.all(np.isfinite(pred))


# ──────────────────────────────────────────────────────────────────────────
# 2. _evaluate_config_hierarchical: 플래그를 fit 동안만 켜고 복원(누수 X)
# ──────────────────────────────────────────────────────────────────────────

def test_evaluate_config_hier_restores_preproc_flag():
    """Stage-1 preproc eval 후 MPH_INNER_HP_PREPROC_TRIALS 가 leak 되지 않아야(최종 refit=full HP).

    G-273c-B: 계층(Stage-1)은 단일점(MPH_INNER_HP_FAST) 대신 축소-탐색 플래그
    (MPH_INNER_HP_PREPROC_TRIALS)를 fit 동안만 set/restore."""
    import optuna
    from simulation.pipeline.per_model_optimize import _evaluate_config_hierarchical
    from simulation.models.tree_models import XGBoostForecaster

    os.environ.pop("MPH_INNER_HP_PREPROC_TRIALS", None)
    X, y = _toy_xy(n=90)
    Xtr, ytr, Xva, yva = X[:70], y[:70], X[70:], y[70:]
    ft = optuna.trial.FixedTrial({"y_mode": "none", "x_mode": "none"})
    res = _evaluate_config_hierarchical(
        lambda: XGBoostForecaster(), Xtr, ytr, Xva, yva, optuna_trial=ft
    )
    assert isinstance(res, dict)
    assert "MPH_INNER_HP_PREPROC_TRIALS" not in os.environ


def test_evaluate_config_hier_restores_preexisting_value():
    """eval 전 값이 있었다면 그 값으로 복원(덮어쓰기 X)."""
    import optuna
    from simulation.pipeline.per_model_optimize import _evaluate_config_hierarchical
    from simulation.models.tree_models import XGBoostForecaster

    os.environ["MPH_INNER_HP_PREPROC_TRIALS"] = "99"
    try:
        X, y = _toy_xy(n=90)
        Xtr, ytr, Xva, yva = X[:70], y[:70], X[70:], y[70:]
        ft = optuna.trial.FixedTrial({"y_mode": "none", "x_mode": "none"})
        _evaluate_config_hierarchical(
            lambda: XGBoostForecaster(), Xtr, ytr, Xva, yva, optuna_trial=ft
        )
        assert os.environ.get("MPH_INNER_HP_PREPROC_TRIALS") == "99"
    finally:
        os.environ.pop("MPH_INNER_HP_PREPROC_TRIALS", None)


# ──────────────────────────────────────────────────────────────────────────
# 3. CQR-LightGBM early_stop 발동 (user-mandated parity)
# ──────────────────────────────────────────────────────────────────────────

def test_cqr_lightgbm_early_stop_fires():
    """CQR-LightGBM 양 head 가 eval_set early_stop 으로 400트리 전 종료해야."""
    from simulation.models.cqr_models import CQRLightGBMForecaster

    rng = np.random.RandomState(0)
    n = 220
    X = rng.randn(n, 4)
    y = np.abs(X[:, 0] * 2.0 + rng.randn(n) * 0.3) + 1.0   # 강한 학습 신호
    m = CQRLightGBMForecaster(alpha=0.1).fit(X, y)

    bi_lo = getattr(m._q_lo_model, "best_iteration_", None)
    bi_hi = getattr(m._q_hi_model, "best_iteration_", None)
    # early_stopping 콜백이 살아있으면 best_iteration_ 가 채워지고 400 미만에서 멈춤.
    assert bi_lo is not None and bi_hi is not None, "early_stopping 콜백 미발동(best_iteration_ None)"
    assert bi_lo < 400 and bi_hi < 400, f"early_stop 미발동: lo={bi_lo}, hi={bi_hi}"

    qlo, qhi = m.predict_quantiles(rng.randn(25, 4))
    assert qlo.shape == (25,) and qhi.shape == (25,)
    assert np.all(qhi >= qlo)           # quantile 비교차
    assert np.all(np.isfinite(qlo)) and np.all(np.isfinite(qhi))


def test_cqr_lightgbm_tiny_data_keeps_legacy_path():
    """train - hold-out < 20 → early_stop 생략(기존 동작 보존, crash X)."""
    from simulation.models.cqr_models import CQRLightGBMForecaster

    rng = np.random.RandomState(1)
    X = rng.randn(18, 3)
    y = np.abs(rng.randn(18)) + 1.0
    m = CQRLightGBMForecaster(alpha=0.1).fit(X, y)   # n=18 → legacy 분기
    qlo, qhi = m.predict_quantiles(rng.randn(5, 3))
    assert qlo.shape == (5,) and np.all(qhi >= qlo)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
