"""M1 — semantic-relation grounding (``semantic_consistency``).

numeric_grounding sees number *presence* only, so a reversed comparison like
"ON R²=0.557 is lower than OFF R²=0.0408" passes it. semantic_consistency
checks the CLAIMED comparison direction against the gold ordering and flags the
confidently-wrong relation. 6-8 cases: matched / reversed / no-comparison /
multi / equal / partial-gold / Korean / NaN-safe.
"""
from simulation.llm_compare.aria_grounding import semantic_consistency

# two gold values whose real ordering is 0.557 > 0.0408
FACTS = ["r2_behavior_on=0.557", "r2_behavior_off=0.0408"]


def test_matched_forward_comparison_is_consistent():
    ans = "행동 ON R²=0.557 is higher than 행동 OFF R²=0.0408."
    r = semantic_consistency(ans, FACTS)
    assert r["n_comparisons"] == 1
    assert r["n_consistent"] == 1
    assert r["n_contradictory"] == 0


def test_reversed_comparison_flags_one_contradiction():
    # the dangerous case: claims ON < OFF, but truth is ON > OFF
    ans = "행동 ON R²=0.557 is lower than 행동 OFF R²=0.0408."
    r = semantic_consistency(ans, FACTS)
    assert r["n_comparisons"] == 1
    assert r["n_contradictory"] == 1
    c = r["contradictions"][0]
    assert c["left"] == "0.557" and c["right"] == "0.0408"
    assert c["claim"] == "left<right" and c["truth"] == "left>right"


def test_no_comparison_returns_zero():
    ans = "행동 ON R²=0.557, 행동 OFF R²=0.0408. 둘 다 보고함."
    r = semantic_consistency(ans, FACTS)
    assert r["n_comparisons"] == 0
    assert r["n_contradictory"] == 0


def test_korean_comparison_direction():
    # "0.0408 보다 0.557 이 더 높다" = 0.557 > 0.0408 → TRUE, consistent.
    ok = semantic_consistency("0.0408 보다 0.557 이 더 높다", FACTS)
    assert ok["n_comparisons"] == 1 and ok["n_consistent"] == 1
    # reversed: "0.557 보다 0.0408 이 더 높다" claims 0.0408 > 0.557 → FALSE.
    bad = semantic_consistency("0.557 보다 0.0408 이 더 높다", FACTS)
    assert bad["n_comparisons"] == 1 and bad["n_contradictory"] == 1


def test_multiple_comparisons_mixed():
    facts = ["a=0.9", "b=0.2", "c=0.5"]
    ans = "0.9 is higher than 0.2, but 0.2 is higher than 0.5."  # 1st ok, 2nd wrong
    r = semantic_consistency(ans, facts)
    assert r["n_comparisons"] == 2
    assert r["n_consistent"] == 1
    assert r["n_contradictory"] == 1


def test_non_gold_comparison_ignored():
    # 0.99 / 0.88 are not gold facts → not judged here (numeric_grounding's job)
    r = semantic_consistency("0.99 is lower than 0.88", FACTS)
    assert r["n_comparisons"] == 0


def test_equal_numbers_no_direction():
    r = semantic_consistency("0.557 is higher than 0.557", ["x=0.557"])
    assert r["n_comparisons"] == 0  # equal carries no direction to contradict


def test_empty_and_none_safe():
    assert semantic_consistency("", [])["n_comparisons"] == 0
    assert semantic_consistency(None, FACTS)["n_contradictory"] == 0
