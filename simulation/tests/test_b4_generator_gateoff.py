"""B4 — frozen pre-stage feature-optuna 생성기 default gate-off.

B2 가 runner LOAD 를 폐기(phase6=STABILITY) → 생성기 산출물(stage2_feature_optuna/*.json)은
기본 경로서 미소비. 따라서 생성기(_rerun_feature_optuna)는 default SKIP(즉시 0).
external 모드(phase5_external JSON 의존)는 MPH_LEGACY_PERMODEL_FEATURES=1 로 opt-in.

run: KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 .venv/bin/python -m pytest <this> -q
"""
import inspect

import pytest

pytestmark = pytest.mark.filterwarnings("ignore")


def test_generator_gated_off_by_default(monkeypatch):
    """MPH_LEGACY_PERMODEL_FEATURES 미설정 → 즉시 0 (subprocess 미실행)."""
    monkeypatch.delenv("MPH_LEGACY_PERMODEL_FEATURES", raising=False)
    from simulation.cli.training_commands import _rerun_feature_optuna
    rc = _rerun_feature_optuna(scope="individual", strategy="all", n_trials=20)
    assert rc == 0, "default 에서 생성기가 skip(0) 돼야 (B4 gate-off)"


def test_generator_gate_is_opt_in_env():
    """소스: gate 가 MPH_LEGACY_PERMODEL_FEATURES opt-in (하드 비활성 아님)."""
    from simulation.cli.training_commands import _rerun_feature_optuna
    src = inspect.getsource(_rerun_feature_optuna)
    assert "MPH_LEGACY_PERMODEL_FEATURES" in src, "opt-in env gate 가 없음"
    assert "return 0" in src, "gate 가 graceful skip(return 0) 이어야"
