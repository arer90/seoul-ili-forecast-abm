"""phase 2-3 = 의도적으로 빔 (2026-06-01, 사용자 명시): feature 선택 + mc 는 phase 13 에서만.

설계 (사용자 명시):
  · phase 1 이 전체 feature(baseline lag/계절성 + 모든 파생변수, ~399) 생성.
  · phase 6~12 는 그 **전체 feature 그대로** 사용 (선택/필터 X). champion gate(12)는 그 위에서 비교.
  · feature 선택(STABILITY) + multicollinearity = **phase 13 (per-model) 에서만**.
  · phase 2-3 = **빔** (feature 계산/LOAD 도, mc 도 없음).

폐기 이력: 옛 frozen pre-stage feature LOAD(B2/B4) + 그걸 대체하려던 global stability stage 도 폐기
  (사용자: "phase 13 에서 하기로 했는데 왜 2-3에서 해").

run: KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 .venv/bin/python -m pytest <this> -q
"""
import inspect

import pytest

pytestmark = pytest.mark.filterwarnings("ignore")


def test_runner_phase23_has_no_feature_compute_or_load():
    """runner phase 2-3: frozen LOAD 도, global stability 계산도, mc 도 없어야 (전부 phase 13)."""
    import simulation.pipeline.runner as r
    src = inspect.getsource(r)
    assert "best_feature_subset" not in src, "frozen LOAD 잔존 (제거돼야)"
    assert "stage2_feature_optuna" not in src, "frozen artifact 경로 잔존 (제거돼야)"
    assert "select_features_stability" not in src, "runner 가 stability 계산하면 안 됨 (phase 13 에서만)"
    assert "global_feature_subset" not in src, "global feature stage 잔존 (제거돼야)"


def test_phase6_defaults_to_full_features():
    """phase 6 = global feature param 없음 → default 전체 feature (선택은 phase 13)."""
    from simulation.pipeline.wfcv import run_wfcv
    sig = inspect.signature(run_wfcv)
    assert "global_feature_subset" not in sig.parameters, "phase 6 에 global feature param 잔존"
    src = inspect.getsource(run_wfcv)
    assert "model_features = None" in src, "phase 6 default 가 None(전체 feature) 이어야"


def test_phase13_still_does_per_model_stability():
    """feature 선택은 phase 13 에서 유지 (per-model STABILITY) — 폐기된 건 phase 2-3 뿐."""
    import simulation.pipeline.per_model_optimize as m
    src = inspect.getsource(m.optimize_one_model)
    assert "select_features_stability" in src, "phase 13 STABILITY 가 사라지면 안 됨"
    assert "make_model_importance_fn" in src, "phase 13 n-adaptive importance_fn 유지돼야"
