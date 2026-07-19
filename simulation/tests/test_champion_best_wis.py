"""Champion 선정 계약 — 2026-06-19 G-320 갱신: G-318 (OOF-shortlist → hold-out 일반화).

이전 계약(2026-06-05, G-307)은 "선정=OOF rank_wis 만, hold-out=보고 전용". 사용자 결정(2026-06-19):
OOF 는 **통계동률 shortlist** 만, 최종 챔피언은 그 안에서 **hold-out 일반화 1위**(G-318) —
winner's curse 는 shortlist 가 통제(test-best-of-53 직접선택 금지), hold-out 은 shortlist 내에서만
개입. + 순수 hold-out best 병기(둘 다 산출). g175/4-criteria gate 는 여전히 없음(2026-06-05 제거).

macOS: run PER-FILE (memory ``test-suite-execution``).
"""
from simulation.pipeline.per_model_eval import (
    _designate_best_wis_champion, select_champion_holdout_best,
)


def _rows():
    # oof_wis = OOF-shortlist 기준, wis = hold-out. B = OOF+hold-out 모두 best → G-318 챔피언.
    return [
        {"model": "A", "oof_wis": 2.0, "wis": 3.0, "pi95_coverage": 0.93},
        {"model": "B", "oof_wis": 1.0, "wis": 1.0, "pi95_coverage": 0.71},   # best 둘 다, under-covered
        {"model": "C", "oof_wis": 3.0, "wis": 4.0, "pi95_coverage": 0.95},
    ]


def test_champion_is_g318_shortlist_holdout_best():
    rows = _rows()
    champ = _designate_best_wis_champion(rows)
    assert champ is not None and champ["model"] == "B", (
        "G-318 챔피언 = OOF-shortlist 내 hold-out best (coverage 나빠도 선정)")
    by = {r["model"]: r for r in rows}
    assert by["B"]["champion_eligible"] is True
    assert by["A"]["champion_eligible"] is False
    assert by["C"]["champion_eligible"] is False


def test_only_one_champion():
    rows = _rows()
    _designate_best_wis_champion(rows)
    assert sum(1 for r in rows if r["champion_eligible"]) == 1


def test_no_g175_gate():
    """champion 선정에 g175/4-criteria gate 없음 — best(OOF+hold-out) 가 champion, coverage 무관."""
    rows = [
        {"model": "A", "oof_wis": 1.0, "wis": 1.0, "pi95_coverage": 0.70, "g175_4criteria_pass": False},
        {"model": "B", "oof_wis": 2.0, "wis": 2.0, "pi95_coverage": 0.95, "g175_4criteria_pass": True},
    ]
    champ = _designate_best_wis_champion(rows)
    assert champ["model"] == "A", "best(둘 다) 가 champion — g175 통과 여부 무관(gate 없음)"


def test_holdout_influences_only_within_shortlist():
    """★ G-318 계약(2026-06-19 변경): hold-out 은 OOF-shortlist '안에서만' 챔피언 결정. shortlist
    밖의 hold-out-lucky(OOF 꼴찌지만 test 최고)는 챔피언 아님 = winner's curse 통제. 이전 G-307의
    'hold-out 전혀 미개입' 에서, 'shortlist 내에서만 개입' 으로 변경(과적합·curse 동시 통제)."""
    rows = [{"model": f"m{i}", "oof_wis": float(i), "wis": 5.0} for i in range(1, 9)]   # OOF-shortlist
    rows.append({"model": "lucky", "oof_wis": 100.0, "wis": 0.1})   # hold-out 최고, OOF 밖
    champ = _designate_best_wis_champion(rows)
    assert champ["model"] != "lucky", "shortlist 밖 hold-out-lucky 는 챔피언 아님(curse 통제)"
    # 단 순수 hold-out best 병기는 lucky 를 가리킴(둘 다 산출 — 사용자 결정)
    assert select_champion_holdout_best(rows)["model"] == "lucky"
