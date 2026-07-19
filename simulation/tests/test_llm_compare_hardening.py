"""P5 hardening (external review): RAGAS faithfulness + judge-bias + κ bands."""
from simulation.llm_compare.comparison import (
    citation_metrics,
    context_precision,
    context_recall,
    faithfulness,
    harm_summary,
    judge_position_debias,
    judge_tier_agreement,
    landis_koch_band,
    n_for_power,
    repetition_variance,
    repro_manifest,
    risk_coverage_curve,
    verbosity_bias,
)


def test_risk_coverage_abstention():
    # confident answers are correct, unconfident wrong → selective acc drops with coverage
    conf = [0.9, 0.8, 0.7, 0.3, 0.2]
    correct = [True, True, True, False, False]
    r = risk_coverage_curve(conf, correct)
    assert r["selective_accuracy"][0] == 1.0          # most-confident is correct
    assert r["full_accuracy"] == 0.6
    assert r["selective_accuracy"][-1] == r["full_accuracy"]
    assert 0.0 <= r["aurc"] <= 1.0


def test_citation_metrics_alce():
    claims = [
        {"has_citation": True, "citation_supports": True},
        {"has_citation": True, "citation_supports": False},  # cited but wrong source
        {"has_citation": False, "citation_supports": False},  # uncited
    ]
    r = citation_metrics(claims)
    assert r["citation_precision"] == 0.5   # 1 of 2 cited are supported
    assert r["citation_recall"] == round(2 / 3, 3)  # 2 of 3 claims cite
    assert r["n_claims"] == 3


def test_judge_tier_calibration_not_demotion():
    human = ["pass", "pass", "fail", "pass", "fail"]
    rule = ["pass", "pass", "fail", "pass", "fail"]   # perfect agreement
    llm = ["pass", "fail", "fail", "pass", "fail"]    # one disagreement
    out = judge_tier_agreement(human, rule=rule, llm=llm)
    assert out["rule"]["scale_ok"] is True and out["rule"]["kappa"] == 1.0
    assert "kappa" in out["llm"] and "band" in out["llm"]


def test_repro_manifest_deterministic():
    kw = dict(model="claude-haiku-4-5", temperature=0.0, top_p=1.0,
              prompts_sha256="abc123", golden_n=25, golden_freeze_date="2026-06-07",
              law_version="감염병예방법-20250124", seed=42, n_repetitions=5)
    a = repro_manifest(**kw)
    b = repro_manifest(**kw)
    assert a["config_sha256"] == b["config_sha256"] and len(a["config_sha256"]) == 16
    assert a["n_repetitions"] == 5


def test_context_precision_rewards_early_relevant():
    relevant = ["a", "b"]
    early = context_precision(["a", "b", "x", "y"], relevant)
    late = context_precision(["x", "y", "a", "b"], relevant)
    assert early > late
    assert early == 1.0


def test_context_recall():
    assert context_recall(["a", "x"], ["a", "b"]) == 0.5
    assert context_recall(["a", "b"], ["a", "b"]) == 1.0


def test_harm_gate():
    ok = harm_summary([{"severity": "none"}, {"severity": "minor"}])
    bad = harm_summary([{"severity": "none"}, {"severity": "critical"}])
    assert ok["critical_gate_pass"] is True and ok["harm_rate"] == 0.5
    assert bad["critical_gate_pass"] is False and bad["n_critical"] == 1


def test_n_for_power_monotone():
    assert n_for_power(0.2) > n_for_power(0.8)  # smaller effect needs more n
    assert n_for_power(0.5, power=0.9) > n_for_power(0.5, power=0.8)


def test_repetition_variance():
    r = repetition_variance([0.9, 1.0, 0.8])
    assert r["n_runs"] == 3 and r["min"] == 0.8 and r["max"] == 1.0
    assert r["sd"] > 0


def test_landis_koch_bands():
    assert landis_koch_band(-0.1) == "poor"
    assert landis_koch_band(0.15) == "slight"
    assert landis_koch_band(0.5) == "moderate"
    assert landis_koch_band(0.7) == "substantial"
    assert landis_koch_band(0.9) == "almost perfect"


def test_faithfulness_grounded_vs_ungrounded():
    ctx = ["인플루엔자는 제4급감염병으로 표본감시 대상이며 7일 이내 신고한다."]
    grounded = "인플루엔자는 제4급감염병이다 [law:감염병예방법]. 표본감시 대상으로 7일 이내 신고한다."
    ungrounded = "인플루엔자는 제1급감염병이며 즉시 격리해야 한다. 백신은 효과가 없다."
    fg = faithfulness(grounded, ctx)
    fu = faithfulness(ungrounded, ctx)
    assert fg["faithfulness"] > 0.8
    assert fu["faithfulness"] < fg["faithfulness"]
    assert fg["n_claims"] >= 2


def test_judge_position_debias():
    # pair p1 consistent (m1 wins both orders), p2 flips with order
    js = [
        {"pair": "p1", "order": "AB", "winner": "m1"},
        {"pair": "p1", "order": "BA", "winner": "m1"},
        {"pair": "p2", "order": "AB", "winner": "m1"},
        {"pair": "p2", "order": "BA", "winner": "m2"},
    ]
    r = judge_position_debias(js)
    assert r["n_pairs"] == 2
    assert r["consistent_preferences"] == 1
    assert r["position_bias_rate"] == 0.5


def test_verbosity_bias_flag():
    pairs = [{"winner_len": 200, "loser_len": 50}] * 8 + [{"winner_len": 40, "loser_len": 90}] * 2
    r = verbosity_bias(pairs)
    assert r["n"] == 10
    assert r["longer_won_rate"] == 0.8
    assert r["flag"] is True


def test_kr_epi_bench_official_anchored():
    from simulation.llm_compare.kr_epi_bench import load_kr_epi_law, categories
    qa = load_kr_epi_law()
    assert len(qa) >= 40                       # expanded 12 → 40 (clears n_for_power d=0.5 = 32)
    # official-source anchored (NOT thesis sections) — avoids contamination
    assert all("§" not in i.official_source for i in qa)
    assert all(i.must_contain and i.official_source for i in qa)
    assert len({i.id for i in qa}) == len(qa)  # unique ids
    cats = categories()
    for c in ("분류", "신고", "방역", "예방접종", "역학", "데이터"):
        assert c in cats                       # all 6 domains covered


def test_kormedmcqa_helpers_offline():
    # format/score logic is testable without network (synthetic normalized item)
    from simulation.llm_compare.kr_epi_bench import format_mcqa, score_mcqa
    item = {"question": "인플루엔자는 몇 급 감염병인가?",
            "options": {"A": "1급", "B": "2급", "C": "3급", "D": "4급"},
            "answer_letter": "D", "answer_text": "4급"}
    prompt = format_mcqa(item)
    assert "A. 1급" in prompt and "D. 4급" in prompt and "정답" in prompt
    assert score_mcqa(item, "D") is True
    assert score_mcqa(item, " d ") is True     # case/space tolerant
    assert score_mcqa(item, "A") is False
    assert score_mcqa(item, "") is False       # empty → wrong, no crash
