"""G-307 (3자 감사 #1): cross-model champion ranked by R9 OOF-CV WIS, NOT hold-out test WIS.

The bug: per_model_eval ranked rank_wis by the hold-out test 'wis' (refit override), making the
champion the test-best-of-53 = selection-on-test (winner's curse), violating the documented
contract (선정=OOF, 보고=test 분리). Fix: rank_wis = R9 OOF-CV WIS (val_metrics.oof_wis, 5-fold
WF-CV, leakage-free); test WIS becomes rank_wis_test (diagnostic only).

macOS: run PER-FILE.
"""
import numpy as np

from simulation.pipeline.per_model_eval import (
    _assign_oof_and_test_ranks, _designate_best_wis_champion,
)


def test_g307_rank_wis_uses_oof_not_test():
    """OOF-best ≠ test-best: champion(rank_wis==1) follows OOF, NOT test (no winner's curse)."""
    rows = [
        {"model": "A", "oof_wis": 1.0, "wis": 9.0},   # OOF-best, test-WORST
        {"model": "B", "oof_wis": 2.0, "wis": 5.0},
        {"model": "C", "oof_wis": 3.0, "wis": 1.0},   # OOF-worst, test-BEST
    ]
    rows_sorted = _assign_oof_and_test_ranks(rows)
    by = {r["model"]: r for r in rows_sorted}
    assert by["A"]["rank_wis"] == 1, "OOF-best (A) is champion despite worst test"
    assert by["C"]["rank_wis"] == 3, "test-best (C) is NOT champion (curse avoided)"
    # diagnostic test rank is separate (rank_wis=OOF 진단은 G-318 후에도 유지)
    assert by["C"]["rank_wis_test"] == 1
    assert by["A"]["rank_wis_test"] == 3
    # G-339 (2026-06-24, 외부 reviewer #1): champion = LEAK-FREE (OOF 1-SE band 안 fold안정성/parsimony,
    #   hold-out test 미사용). 여기선 A(oof=1.0)만 band 안(B=2.0·C=3.0 은 2% 밖) → 챔피언=A(OOF-best).
    #   옛 G-318 은 shortlist 내 test-best=C 를 골랐으나 그건 winner's curse(C 는 OOF 3배 나쁨).
    #   G-339 는 test 를 안 보므로 이 테스트 제목('uses oof not test')과 오히려 더 일치.
    champ = _designate_best_wis_champion(rows_sorted)
    assert champ["model"] == "A", "G-339 leak-free: OOF 1-SE band(=A only) 내 선정, test-best C 아님"


def test_g307_missing_oof_ranked_last_report_only():
    """META/feature-less model (oof_wis=inf) → ranked last, never champion, even if test-best."""
    rows = [
        {"model": "META", "oof_wis": float("inf"), "wis": 0.5},   # best test, no OOF
        {"model": "Real", "oof_wis": 2.0, "wis": 5.0},
    ]
    rows_sorted = _assign_oof_and_test_ranks(rows)
    by = {r["model"]: r for r in rows_sorted}
    assert by["Real"]["rank_wis"] == 1
    assert by["META"]["rank_wis"] == 2, "no-OOF model ranked last (report-only)"
    assert _designate_best_wis_champion(rows_sorted)["model"] == "Real"


def test_g307_missing_oof_key_treated_as_inf():
    """A row lacking the oof_wis key entirely is treated as +inf (last) — KeyError-safe."""
    rows = [
        {"model": "X", "wis": 1.0},                  # no oof_wis key at all
        {"model": "Y", "oof_wis": 5.0, "wis": 9.0},
    ]
    rows_sorted = _assign_oof_and_test_ranks(rows)
    by = {r["model"]: r for r in rows_sorted}
    assert by["Y"]["rank_wis"] == 1
    assert by["X"]["rank_wis"] == 2


def test_g307_agreement_when_oof_and_test_align():
    """When OOF-best == test-best, both ranks agree (robust champion)."""
    rows = [
        {"model": "Best", "oof_wis": 1.0, "wis": 1.0},
        {"model": "Mid",  "oof_wis": 2.0, "wis": 2.0},
    ]
    rows_sorted = _assign_oof_and_test_ranks(rows)
    by = {r["model"]: r for r in rows_sorted}
    assert by["Best"]["rank_wis"] == 1 and by["Best"]["rank_wis_test"] == 1
    assert _designate_best_wis_champion(rows_sorted)["model"] == "Best"
