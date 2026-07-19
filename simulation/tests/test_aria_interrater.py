"""m2 — inter-rater honesty.

The bare Fleiss kappa (0.5238) is dragged down by mistral's 2 UNPARSED verdicts
even though the parseable judges agree. The report must surface per-judge
parse-rate + accuracy ALONGSIDE Fleiss so the headline kappa is not read in
isolation. These tests guard (a) the regenerated artifact carries the honesty
fields, and (b) the pure helpers in the generator behave.
"""
import json
from pathlib import Path

import pytest

ART = Path("simulation/results/aria_sci/aria_interrater_kappa.json")


@pytest.mark.skipif(not ART.exists(), reason="kappa artifact not generated")
def test_artifact_has_per_judge_parse_rate_and_accuracy():
    d = json.loads(ART.read_text(encoding="utf-8"))
    ja = d["judge_accuracy_vs_truth"]
    assert ja, "no per-judge block"
    for jm, rec in ja.items():
        assert "parse_rate" in rec, f"{jm} missing parse_rate"
        assert "accuracy_vs_truth" in rec, f"{jm} missing accuracy_vs_truth"
        assert 0.0 <= rec["parse_rate"] <= 1.0


@pytest.mark.skipif(not ART.exists(), reason="kappa artifact not generated")
def test_interrater_summary_reported_alongside_fleiss():
    d = json.loads(ART.read_text(encoding="utf-8"))
    assert "fleiss_kappa" in d
    s = d["interrater_summary"]
    assert "per_judge" in s and s["per_judge"]
    assert "any_unparsed_verdicts" in s and "caveat" in s
    # honesty: if any judge has parse_rate < 1, the caveat must flag it
    has_unparsed = any(r["parse_rate"] < 1.0 for r in s["per_judge"].values())
    assert s["any_unparsed_verdicts"] == has_unparsed
    if has_unparsed:
        assert "UNPARSED" in s["caveat"]


@pytest.mark.skipif(not ART.exists(), reason="kappa artifact not generated")
def test_parse_rate_matches_stored_verdicts():
    """parse_rate must equal (non-UNPARSED verdicts) / n_items — no inflation."""
    d = json.loads(ART.read_text(encoding="utf-8"))
    n_items = len(d["rating_set"])
    for jm, labels in d["verdicts"].items():
        parsed = sum(1 for v in labels if v != "UNPARSED")
        assert d["judge_accuracy_vs_truth"][jm]["parse_rate"] == round(parsed / n_items, 3)


def test_parse_verdict_unparsed_path():
    from simulation.scripts.aria_interrater_kappa import _parse_verdict
    assert _parse_verdict("GROUNDED 입니다") == "GROUNDED"
    assert _parse_verdict("NOT_GROUNDED") == "NOT_GROUNDED"
    assert _parse_verdict("정답을 모르겠음 blah") == "UNPARSED"
    assert _parse_verdict("") == "UNPARSED"
