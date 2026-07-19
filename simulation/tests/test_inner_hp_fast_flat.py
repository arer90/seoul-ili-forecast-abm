"""G-273c (2): inner-HP fast-path 확장 — feature-stability + mc-probe (평면 _evaluate_config).

배경 (2026-06-15, 사용자 "optuna와 feature stability에서도 해야되는데"): 1차 fast-path 는 계층
_evaluate_config_hierarchical(Stage-1 preproc)에만 적용 → feature-stability(_oof_cv_wis)·mc-probe
(_probe_one_model)는 평면 _evaluate_config 를 타서 fast 를 못 받고 내부 HP study 를 풀로 재실행.
(early_stop 은 forecaster 내부라 적용됐으나 속도만 손해.) 평면 _evaluate_config 에 `_fast_inner`
키워드 추가 + comparison 호출처만 True → feature-stability·mc-probe 도 fast. _oof_cv_metrics·final
refit 은 default False → full HP 유지.

검증 전략 (deterministic, segfault 회피): `optuna.create_study` 를 **sabotage(raise)** 로 패치.
fast 경로면 study 를 시도하지 않으므로 fit 이 성공(에러 없음); full 경로면 study 를 시도→raise→
_evaluate_config 가 catch 해 error dict 반환. macOS 의 full XGBoost study in-process 실행이
OpenMP 충돌로 segfault 하므로, study 를 절대 실제 실행하지 않고 "시도 여부"만 본다.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")   # macOS libomp 충돌 완화 (벨트앤서스펜더)

import numpy as np
import pytest


def _toy_xy(n=80, p=5, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = np.abs(X[:, 0] * 1.5 + rng.randn(n) * 0.3) + 1.0
    return X, y


def _sabotage_study(monkeypatch):
    """optuna.create_study → 즉시 raise. fast 경로면 미호출(성공), full 경로면 호출됨(에러로 표면화)."""
    import optuna

    def _boom(*a, **k):
        raise RuntimeError("STUDY_ATTEMPTED")

    monkeypatch.setattr(optuna, "create_study", _boom)


# ──────────────────────────────────────────────────────────────────────────
# 1. _evaluate_config: default=full(study 시도), _fast_inner=True=fast(미시도) + 누수 0
# ──────────────────────────────────────────────────────────────────────────

def test_evaluate_config_fast_vs_default(monkeypatch):
    from simulation.pipeline.per_model_optimize import _evaluate_config
    from simulation.models.tree_models import XGBoostForecaster

    _sabotage_study(monkeypatch)
    os.environ.pop("MPH_INNER_HP_FAST", None)
    X, y = _toy_xy(n=70, p=5)
    Xtr, ytr, Xva, yva = X[:55], y[:55], X[55:], y[55:]

    # _fast_inner=True → study 미시도 → fit 성공(에러 없음)
    res_fast = _evaluate_config(lambda: XGBoostForecaster(), Xtr, ytr, Xva, yva,
                                transform_name="identity", scaler_name="robust", _fast_inner=True)
    assert isinstance(res_fast, dict)
    assert "error" not in res_fast, f"fast 경로가 study 를 시도/실패: {res_fast.get('error')}"
    assert np.isfinite(res_fast.get("wis", float("inf")))
    assert "MPH_INNER_HP_FAST" not in os.environ, "fast fit 후 플래그 누수 금지"

    # default(_fast_inner=False) → study 시도 → RuntimeError → _evaluate_config 가 catch → error dict
    res_full = _evaluate_config(lambda: XGBoostForecaster(), Xtr, ytr, Xva, yva,
                                transform_name="identity", scaler_name="robust")
    assert "error" in res_full and "STUDY_ATTEMPTED" in res_full["error"], \
        "default _evaluate_config 는 full HP study 를 시도해야 (회귀 가드)"
    assert "MPH_INNER_HP_FAST" not in os.environ


# ──────────────────────────────────────────────────────────────────────────
# 2. feature-stability (_oof_cv_wis) → fast (study 미시도 → 모든 fold 성공 → 유한)
# ──────────────────────────────────────────────────────────────────────────

def test_oof_cv_wis_feature_stability_fast(monkeypatch):
    from simulation.pipeline.per_model_optimize import _oof_cv_wis
    from simulation.models.tree_models import XGBoostForecaster

    _sabotage_study(monkeypatch)
    os.environ.pop("MPH_INNER_HP_FAST", None)
    X, y = _toy_xy(n=80, p=5)
    res = _oof_cv_wis(lambda: XGBoostForecaster(), X, y, "identity", "robust")
    # fast 면 study 미시도 → 각 fold fit 성공 → 유한 WIS. (느린 경로였다면 study→boom→전 fold error→inf)
    assert np.isfinite(res), "feature-stability fast 경로가 동작해야(유한 OOF-WIS, study 미시도)"
    assert "MPH_INNER_HP_FAST" not in os.environ


# ──────────────────────────────────────────────────────────────────────────
# 3. mc-probe (_probe_one_model) → fast
# ──────────────────────────────────────────────────────────────────────────

def test_probe_one_model_mc_fast(monkeypatch):
    from simulation.pipeline.per_model_optimize import _probe_one_model
    from simulation.models.tree_models import XGBoostForecaster

    _sabotage_study(monkeypatch)
    os.environ.pop("MPH_INNER_HP_FAST", None)
    X, y = _toy_xy(n=90, p=5)
    feat_cols = [f"f{i}" for i in range(5)]
    res = _probe_one_model(
        "XGBoost", X, y, transform_name="identity", scaler_name="robust",
        feature_cols=feat_cols, n_folds=2, factory=lambda: XGBoostForecaster(),
    )
    assert isinstance(res, dict) and "cells" in res
    # 'none' method 의 oof_wis 가 유한 = study 미시도(fast). 느린 경로였다면 boom→catch→inf.
    none_oof = res["cells"].get("none", {}).get("oof_wis", float("inf"))
    assert np.isfinite(none_oof), "mc-probe fast 경로가 동작해야('none' OOF-WIS 유한)"
    assert "MPH_INNER_HP_FAST" not in os.environ


# ──────────────────────────────────────────────────────────────────────────
# 3b. RandomForest (G-273d: Optuna 통일) — fast=단일fit, preproc/full=study (XGB/LGB 동형)
# ──────────────────────────────────────────────────────────────────────────

def test_randomforest_fast_bypasses_study(monkeypatch):
    """MPH_INNER_HP_FAST=1 → RF 는 내부 Optuna study 생략(단일 default fit)."""
    import optuna
    from simulation.models.tree_models import RandomForestForecaster

    def _boom(*a, **k):
        raise AssertionError("Optuna study ran despite MPH_INNER_HP_FAST=1")

    monkeypatch.setattr(optuna, "create_study", _boom)
    monkeypatch.setenv("MPH_INNER_HP_FAST", "1")
    X, y = _toy_xy(n=80, p=5)
    m = RandomForestForecaster().fit(X, y)
    pred = m.predict(_toy_xy(n=12, seed=9)[0])
    assert pred.shape == (12,)
    assert np.all(np.isfinite(pred))


def test_randomforest_full_runs_study(monkeypatch):
    """플래그 OFF → RF 는 full Optuna study 실행(XGB/LGB 와 통일된 경로, 더 이상 GridSearchCV 아님)."""
    import optuna
    from simulation.models.tree_models import RandomForestForecaster

    def _boom(*a, **k):
        raise RuntimeError("STUDY_RAN")

    monkeypatch.setattr(optuna, "create_study", _boom)
    monkeypatch.delenv("MPH_INNER_HP_FAST", raising=False)
    monkeypatch.delenv("MPH_INNER_HP_PREPROC_TRIALS", raising=False)
    X, y = _toy_xy(n=90, p=5)
    with pytest.raises(RuntimeError, match="STUDY_RAN"):
        RandomForestForecaster().fit(X, y)


# ──────────────────────────────────────────────────────────────────────────
# 3c. G-273c-B: Stage-1 preproc = 축소 HP 탐색 (단일점 아님), RF 는 단일 fit
# ──────────────────────────────────────────────────────────────────────────

def test_preproc_trials_runs_reduced_study(monkeypatch):
    """MPH_INNER_HP_PREPROC_TRIALS set → XGBoost 는 (단일 fit 아니라) study 실행 — 단 trial 축소.

    transform 선택이 full 을 추종하도록 작은 HP 탐색을 돌린다(단일 고정점 발산 회피)."""
    import optuna
    from simulation.models.tree_models import XGBoostForecaster

    def _boom(*a, **k):
        raise RuntimeError("STUDY_RAN")

    monkeypatch.setattr(optuna, "create_study", _boom)
    monkeypatch.delenv("MPH_INNER_HP_FAST", raising=False)
    monkeypatch.setenv("MPH_INNER_HP_PREPROC_TRIALS", "5")
    X, y = _toy_xy(n=80, p=5)
    # preproc 모드 = study 실행 → create_study 시도 → boom (단일 fit 이었다면 boom 안 남)
    with pytest.raises(RuntimeError, match="STUDY_RAN"):
        XGBoostForecaster().fit(X, y)


def test_preproc_trials_rf_reduced_study(monkeypatch):
    """RandomForest 는 preproc 모드서 (단일fit 아니라) 축소 Optuna study 실행 — G-273d 통일."""
    import optuna
    from simulation.models.tree_models import RandomForestForecaster

    def _boom(*a, **k):
        raise RuntimeError("STUDY_RAN")

    monkeypatch.setattr(optuna, "create_study", _boom)
    monkeypatch.delenv("MPH_INNER_HP_FAST", raising=False)
    monkeypatch.setenv("MPH_INNER_HP_PREPROC_TRIALS", "5")
    X, y = _toy_xy(n=80, p=5)
    with pytest.raises(RuntimeError, match="STUDY_RAN"):
        RandomForestForecaster().fit(X, y)


# ──────────────────────────────────────────────────────────────────────────
# 3d. KRR (G-273d: grid→Optuna 통일) + SVR-RBF (가드 추가) — 3-모드
# ──────────────────────────────────────────────────────────────────────────

def test_krr_fast_bypasses_study(monkeypatch):
    """KRR: MPH_INNER_HP_FAST=1 → 내부 Optuna study 생략(단일 default fit), grid 도 아님."""
    import optuna
    from simulation.models.linear_models import KRRForecaster

    def _boom(*a, **k):
        raise AssertionError("KRR study ran despite MPH_INNER_HP_FAST=1")

    monkeypatch.setattr(optuna, "create_study", _boom)
    monkeypatch.setenv("MPH_INNER_HP_FAST", "1")
    X, y = _toy_xy(n=90, p=10)
    m = KRRForecaster().fit(X, y)
    assert np.all(np.isfinite(m.predict(_toy_xy(n=10, seed=2, p=10)[0])))


def test_krr_full_runs_study(monkeypatch):
    """KRR: 플래그 OFF → 인라인 Optuna study 실행(grid 아님, G-273d 통일)."""
    import optuna
    from simulation.models.linear_models import KRRForecaster

    def _boom(*a, **k):
        raise RuntimeError("STUDY_RAN")

    monkeypatch.setattr(optuna, "create_study", _boom)
    monkeypatch.delenv("MPH_INNER_HP_FAST", raising=False)
    monkeypatch.delenv("MPH_INNER_HP_PREPROC_TRIALS", raising=False)
    X, y = _toy_xy(n=90, p=10)
    with pytest.raises(RuntimeError, match="STUDY_RAN"):
        KRRForecaster().fit(X, y)


def test_svr_rbf_fast_bypasses_study(monkeypatch):
    """SVR-RBF: MPH_INNER_HP_FAST=1 → 20-trial study 생략(단일 강한-default fit)."""
    import optuna
    from simulation.models.linear_models import SVRRBFForecaster

    def _boom(*a, **k):
        raise AssertionError("SVR-RBF study ran despite MPH_INNER_HP_FAST=1")

    monkeypatch.setattr(optuna, "create_study", _boom)
    monkeypatch.setenv("MPH_INNER_HP_FAST", "1")
    X, y = _toy_xy(n=90, p=10)
    m = SVRRBFForecaster().fit(X, y)
    assert np.all(np.isfinite(m.predict(_toy_xy(n=10, seed=2, p=10)[0])))


# ──────────────────────────────────────────────────────────────────────────
# 4. 소스 가드: _fast_inner=True 호출처가 정확히 3곳 (오염 방지)
# ──────────────────────────────────────────────────────────────────────────

def test_fast_inner_callsite_count():
    """_fast_inner=True 는 comparison 3곳(feature-stability·mc-probe OOF·mc-probe insample)뿐이어야.
    실수로 final/_oof_cv_metrics 경로에 추가되면 이 가드가 잡는다."""
    import simulation.pipeline.per_model_optimize as _pm
    src = open(_pm.__file__, encoding="utf-8").read()
    # 콤마 포함 = 실제 호출 인자(주석 안의 설명용 `_fast_inner=True (...` 는 제외)
    n = src.count("_fast_inner=True,")
    assert n == 3, f"_fast_inner=True 호출처는 정확히 3곳이어야 (현재 {n}) — final/metrics 오염 점검"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
