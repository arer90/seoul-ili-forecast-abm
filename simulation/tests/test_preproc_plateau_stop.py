"""Preproc Optuna plateau-stop callback (2026-06-15, 3-LLM 검증 + 사용자 승인).

문제(검증됨): preproc Optuna study 가 pruner 를 배정하지만 _preproc_objective
(_inline_optuna_3stage.py)가 fold 별 trial.report()/should_prune() 를 호출하지 않아
pruner 가 inert → 모든 모델이 best oof_wis 가 trial 1-5 에 plateau 해도 n_trials 전부 소진
(tree_models.py:112-116 은 정확히 trial.report+should_prune 함 — 대조).

해결(안전·additive): study-level plateau-stop 콜백. direction=minimize 에서 best 가
`patience` trial 연속 무개선이면 study.stop(). best trial 은 항상 완료된 뒤 멈추므로
늦은-plateau 챔피언층(TabPFN·NegBinGLM·ElasticNet·XGBoost)은 끝까지, 조기-plateau
모델(FluSight·EARS·KRR·N-BEATS)만 일찍 종료 → 품질 무손상.

Run (macOS 단일 파일):
    KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/python -m pytest \
        simulation/tests/test_preproc_plateau_stop.py -x -q
"""
from __future__ import annotations

import numpy as np
import optuna
import pytest


class _FakeStudy:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


class _FakeTrial:
    def __init__(self, value):
        self.value = value


def test_plateau_stop_fires_after_patience():
    """5→4→3(best) 후 3,3,3 → 3번째 무개선(idx5)에 study.stop()."""
    from simulation.pipeline._inline_optuna_3stage import _make_preproc_plateau_stop
    cb = _make_preproc_plateau_stop(patience=3)
    s = _FakeStudy()
    seq = [5.0, 4.0, 3.0, 3.0, 3.0, 3.0]
    fired_at = None
    for i, v in enumerate(seq):
        cb(s, _FakeTrial(v))
        if s.stopped:
            fired_at = i
            break
    assert fired_at == 5, f"plateau-stop 은 3번째 plateau(idx5)에 발동해야 함, got {fired_at}"


def test_plateau_stop_keeps_going_while_improving():
    """단조 개선 중이면 절대 멈추지 않음 (챔피언층 보호)."""
    from simulation.pipeline._inline_optuna_3stage import _make_preproc_plateau_stop
    cb = _make_preproc_plateau_stop(patience=3)
    s = _FakeStudy()
    for v in [5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.1]:
        cb(s, _FakeTrial(v))
    assert not s.stopped, "개선 중에는 멈추면 안 됨"


def test_plateau_stop_handles_none_value():
    """실패/pruned trial(value=None)은 crash 없이 무개선으로도 안 셈."""
    from simulation.pipeline._inline_optuna_3stage import _make_preproc_plateau_stop
    cb = _make_preproc_plateau_stop(patience=2)
    s = _FakeStudy()
    cb(s, _FakeTrial(None)); cb(s, _FakeTrial(None))
    assert not s.stopped
    cb(s, _FakeTrial(5.0))               # 첫 실값 = best
    cb(s, _FakeTrial(5.0)); cb(s, _FakeTrial(5.0))  # 2 plateau → stop
    assert s.stopped


def test_negative_deep_hp_trials_cut():
    """음성-R² deep(TabularDNN) HP 40→25, 양성-R² deep(Mamba)는 40 유지.
    G-296(b889606): N-BEATS 는 per_model_trials 에서 삭제됨 — REGISTRY 클래스가 PfNBeats(Optuna 0)라
    entry 가 INERT 였고 가짜 '40→25' 결정을 유발했음. 이제 dict 에 없어야 함(get → None)."""
    from simulation.pipeline.config import OptunaConfig
    o = OptunaConfig()
    assert o.per_model_trials.get("TabularDNN") == 25, "TabularDNN HP 40→25 cut 필요"
    assert o.per_model_trials.get("N-BEATS") is None, "N-BEATS 는 G-296 으로 삭제 (Pf=Optuna 0)"
    assert o.per_model_trials.get("Mamba") == 40, "Mamba(양성 R²)는 40 유지 (챔피언 무손상)"
    assert o.per_model_trials.get("iTransformer") == 40, "iTransformer(양성 R²)는 40 유지"


def test_stage1_hier_oof_reports_and_checks_pruner():
    """Fold-level OOF replay must call trial.report and should_prune for the preproc pruner."""
    from simulation.pipeline._inline_optuna_3stage import _oof_cv_wis_hier

    class _MeanReg:
        def fit(self, X, y, **kwargs):
            self.mu = float(np.mean(y))
            return self

        def predict(self, X):
            return np.full(len(X), self.mu, dtype=float)

    class _PruneTrial:
        def __init__(self):
            self.reports = []
            self.should_calls = 0

        def report(self, value, step):
            self.reports.append((float(value), int(step)))

        def should_prune(self):
            self.should_calls += 1
            return bool(self.reports)

    rng = np.random.default_rng(42)
    X = rng.normal(size=(90, 4))
    y = np.linspace(3.0, 18.0, 90) + rng.normal(scale=0.2, size=90)
    trial = _PruneTrial()
    with pytest.raises(optuna.TrialPruned):
        _oof_cv_wis_hier(
            lambda: _MeanReg(), X, y, {"y_mode": "none", "x_mode": "none"},
            n_folds=2, optuna_trial=trial,
        )
    assert trial.reports, "preproc OOF must report at least one intermediate value"
    assert trial.should_calls >= 1, "preproc OOF must call should_prune"


if __name__ == "__main__":
    test_plateau_stop_fires_after_patience(); print("PASS fires_after_patience")
    test_plateau_stop_keeps_going_while_improving(); print("PASS keeps_going")
    test_plateau_stop_handles_none_value(); print("PASS none_value")
    test_negative_deep_hp_trials_cut(); print("PASS hp_cut")
    print("=== ALL PASS ===")
