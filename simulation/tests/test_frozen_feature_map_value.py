"""frozen pre-stage feature map 의 RETIREMENT regression-guard (B2/B4, 2026-06-01).

배경 (이전 버전의 STRUCTURAL 발견 — 이제 폐기 근거로 보존):
  frozen pre-stage feature map = stage2_feature_optuna/<model>.json 의 best_feature_subset.
  "모델별 feature 를 train-pool OOF 로 1회 선택해 frozen → phase 6/13/14 가 소비" 의도였으나:
    · 6 모델(ARIMA·CatBoost·ElasticNet·SARIMA·SARIMAX·Theta)이 size=87 동일 subset 공유 (가짜 per-model).
    · size 49~111 (n_pool=242 대비 거대 = 과적합 영역, n/p≈2.8).
    · 65 registered 중 12개만 artifact 존재 (대부분 미적용).
  → A≈B 실측(Wilcoxon p=0.52) + codex·gemini·claude 만장일치 → **폐기**.

대체:
  · phase6 = runner 가 train-pool STABILITY(Meinshausen-Bühlmann) 1회 → global feature 기반 (B2).
  · phase13 = per-model STABILITY (n-adaptive: 작은 n=|corr|, massive=model-based) (C).
  · phase14 = per_model_feature_map 미사용 → phase13 산출 자동 반영.
  · 생성기 = default gate-off (MPH_LEGACY_PERMODEL_FEATURES opt-in) (B4).

이 테스트는 그 **폐기가 유지됨**을 regression-guard 로 고정한다.
실행: KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 .venv/bin/python -m pytest <this> -q
"""
import inspect
from pathlib import Path

import pytest

pytestmark = pytest.mark.filterwarnings("ignore")

_ACTIVE_DIR = Path("simulation/results/stage2_feature_optuna")
_ARCHIVE_DIR = Path("simulation/results/_archive/stage2_feature_optuna_RETIRED_20260601")


# ── 폐기 상태 ────────────────────────────────────────────────────────────
def test_active_frozen_dir_absent_or_empty():
    """활성 stage2_feature_optuna/ 는 부재이거나 비어야 (artifacts archived = 폐기)."""
    if _ACTIVE_DIR.exists():
        live = [f for f in _ACTIVE_DIR.glob("*.json") if not f.stem.startswith("_")]
        assert not live, (
            f"활성 frozen artifact 가 아직 {len(live)}개 존재 — B4 archive 미완 또는 재생성됨")


def test_runner_does_not_load_frozen_or_compute_features():
    """runner phase 2-3 = 빔: frozen LOAD 도, feature 계산도 없음 (feature 선택은 phase 13 에서만)."""
    import simulation.pipeline.runner as r
    src = inspect.getsource(r)
    assert "best_feature_subset" not in src, "runner 가 아직 frozen best_feature_subset 를 읽음"
    assert "stage2_feature_optuna" not in src, "frozen artifact 경로 잔존"
    assert "select_features_stability" not in src, "feature 선택은 phase 13 에서만 (runner 계산 X)"


def test_feature_selection_is_phase13_only():
    """feature 선택(STABILITY)은 phase 13 에서만; phase 6 은 전체 feature default (선택 X)."""
    from simulation.pipeline.wfcv import run_wfcv
    assert "global_feature_subset" not in inspect.signature(run_wfcv).parameters, \
        "phase 6 에 global feature stage 잔존 (제거돼야)"
    import simulation.pipeline.per_model_optimize as m
    assert "select_features_stability" in inspect.getsource(m.optimize_one_model), \
        "phase 13 STABILITY 유지돼야"


def test_generator_gated_off_default():
    """생성기 default gate-off (B4) — 재생성으로 폐기가 되돌려지지 않음."""
    from simulation.cli.training_commands import _rerun_feature_optuna
    src = inspect.getsource(_rerun_feature_optuna)
    assert "MPH_LEGACY_PERMODEL_FEATURES" in src and "return 0" in src


# ── 복구 가능성 (artifacts 보존) ──────────────────────────────────────────
def test_retired_artifacts_archived_with_readme():
    """폐기 artifact 가 archive 에 보존 + README (재현·복구 가능)."""
    if not _ARCHIVE_DIR.exists():
        pytest.skip(f"{_ARCHIVE_DIR} 없음 (다른 환경/clean checkout)")
    assert (_ARCHIVE_DIR / "README.md").exists(), "retirement README 없음"
    archived = list(_ARCHIVE_DIR.glob("*.json"))
    assert len(archived) >= 5, f"archive 된 frozen artifact 너무 적음: {len(archived)}"
    print(f"\n  archived frozen artifacts: {len(archived)} (복구 가능, README 동반)")
