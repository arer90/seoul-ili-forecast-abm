"""AUDIT regression (2026-06-01 전수 audit, CRITICAL): full 시나리오는 conformal holdout 예약 필수.

발견: full 에 conformal_holdout_weeks 누락 → config default 0 → phase1 holdout_start=n →
phase10 `has_holdout=False` → "oof_internal_split_OPTIMISTIC" 폴백 → production(--scenario full)
의 PI/WIS/PICP/best-WIS champion 이 in-sample-optimistic. full_light(26) 만 고쳐져 있었음.
이 테스트가 그 회귀를 영구 차단.

run: KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 .venv/bin/python -m pytest <this> -q
"""
import inspect

import pytest

pytestmark = pytest.mark.filterwarnings("ignore")


def test_full_scenario_reserves_conformal_holdout():
    from simulation.cli._scenarios import SCENARIOS
    assert SCENARIOS["full"].get("conformal_holdout_weeks", 0) > 0, (
        "full 시나리오가 conformal holdout 미예약 → phase10 PI optimistic (audit CRITICAL 재발)")


def test_full_light_scenario_reserves_conformal_holdout():
    from simulation.cli._scenarios import SCENARIOS
    assert SCENARIOS["full_light"].get("conformal_holdout_weeks", 0) > 0


def test_phase10_optimistic_fallback_gated_on_holdout_absence():
    """phase10: holdout_start<n → honest split-conformal, ==n → OPTIMISTIC 폴백 (라벨로 구분)."""
    import simulation.pipeline.intervals as m
    src = inspect.getsource(m)
    assert "holdout_start < n" in src, "has_holdout 가 holdout_start<n 조건 잃음"
    assert "oof_internal_split_OPTIMISTIC" in src, "OPTIMISTIC 폴백 라벨 사라짐 (정직성 가드)"
