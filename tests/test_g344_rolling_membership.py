"""G-344 (2026-06-24, 감사 P0-2/P0-3): rolling 평가 라우팅 transform-aware 정정.

P0-2: N-HiTS 가 ROLLING_EVAL(raw y_observed)에 있었으나 실측 transform=HIER_individual(non-identity)
  → raw 면 test R²−13.7 폭발. TRANSFORM_ROLLING 이동(transformed y_observed, N-BEATS/TiDE와 동일).
P0-3: foundation(TiRex/TimesFM-2.5/DLinear) 가 BASELINE_ROLLING(R9 단일원점)이라 oof=inf 로 챔피언
  선택서 제외(TiRex=hold-out test 1위인데 후보조차 안 됨). transform=HIER_none(identity) 라 ROLLING_EVAL
  이동 → _evaluate_config_hierarchical helper(supports_rolling_eval 게이트)가 R9 OOF 도 rolling → 유한.

macOS: per-file.
"""
from simulation.models.base import (
    BASELINE_ROLLING_MODELS,
    ROLLING_EVAL_MODELS,
    TRANSFORM_ROLLING_MODELS,
    supports_baseline_rolling,
    supports_rolling_eval,
    supports_transform_rolling,
)


class _Stub:
    """supports_* helper 는 model.meta.name 으로 판정 → 가벼운 stub."""
    def __init__(self, name):
        self.meta = type("M", (), {"name": name})()


def test_nhits_moved_to_transform_rolling():
    """P0-2: N-HiTS = transform-space(individual) → TRANSFORM_ROLLING, NOT ROLLING_EVAL(raw 폭발)."""
    assert "N-HiTS" in TRANSFORM_ROLLING_MODELS
    assert "N-HiTS" not in ROLLING_EVAL_MODELS, "raw y_observed → test R²−13.7 폭발(G-337 오판)"
    assert {"N-BEATS", "N-HiTS", "TiDE"} <= TRANSFORM_ROLLING_MODELS   # pf 3종 통일
    assert supports_transform_rolling(_Stub("N-HiTS"))
    assert not supports_rolling_eval(_Stub("N-HiTS"))


def test_foundation_moved_to_rolling_eval():
    """P0-3: foundation(identity) → ROLLING_EVAL → R9 OOF 유한(선택 가능). baseline-only(oof=inf) 아님."""
    for m in ("TiRex", "TimesFM-2.5", "DLinear"):
        assert m in ROLLING_EVAL_MODELS, f"{m}: R9 helper rolling → 유한 OOF(챔피언 후보)"
        assert m not in BASELINE_ROLLING_MODELS
        assert supports_rolling_eval(_Stub(m))


def test_baseline_rolling_empty_after_migration():
    """G-344: 전 멤버 migrated → BASELINE_ROLLING 비었음(helper 는 보존)."""
    assert BASELINE_ROLLING_MODELS == frozenset()
    assert not supports_baseline_rolling(_Stub("TiRex"))


def test_rolling_sets_mutually_exclusive():
    """한 모델이 raw-rolling(ROLLING_EVAL)과 transform-rolling(TRANSFORM_ROLLING) 양쪽이면 라우팅 모호."""
    assert ROLLING_EVAL_MODELS.isdisjoint(TRANSFORM_ROLLING_MODELS)


def test_identity_raw_rolling_group_together():
    """foundation 3종 + FusedEpi = identity raw-rolling 동치 그룹(같은 ROLLING_EVAL 처리)."""
    assert {"TiRex", "TimesFM-2.5", "DLinear", "FusedEpi"} <= ROLLING_EVAL_MODELS
