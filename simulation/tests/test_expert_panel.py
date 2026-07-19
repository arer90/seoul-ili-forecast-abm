"""Expert blinded-panel harness + multi-rater Fleiss κ (P5 #4 calibration)."""
import csv

from simulation.llm_compare.comparison import fleiss_kappa_ratings
from simulation.llm_compare.expert_panel import (
    PanelResponse,
    dry_run,
    load_ratings_csv,
    make_blinded_sheet,
    panel_report,
    rating_template_csv,
)


def test_fleiss_ratings_perfect_and_disagree():
    same = {"A": ["pass", "fail", "pass"], "B": ["pass", "fail", "pass"],
            "C": ["pass", "fail", "pass"]}
    r = fleiss_kappa_ratings(same)
    assert r["kappa"] == 1.0 and r["n_raters"] == 3 and r["band"] == "almost perfect"
    # maximal disagreement (split 50/50 each item) → κ ≤ 0
    opp = {"A": ["pass", "fail", "pass", "fail"], "B": ["fail", "pass", "fail", "pass"]}
    assert fleiss_kappa_ratings(opp)["kappa"] <= 0.0
    assert "error" in fleiss_kappa_ratings({"A": ["pass"]})  # <2 raters


def test_blinded_sheet_hides_source_deterministic():
    resp = [PanelResponse(f"IT{i}", "claude" if i % 2 else "gpt", f"q{i}", f"a{i}")
            for i in range(6)]
    s1 = make_blinded_sheet(resp, seed=7)
    s2 = make_blinded_sheet(resp, seed=7)
    # rows are source-blind (no backend/item_id leak), key holds the unblinding map
    assert all(set(row) == {"sheet_id", "question", "answer"} for row in s1.rows)
    assert all("backend" in s1.key[r["sheet_id"]] for r in s1.rows)
    assert [r["sheet_id"] for r in s1.rows] == [r["sheet_id"] for r in s2.rows]  # deterministic
    assert len(s1.key) == 6


def test_template_roundtrip(tmp_path):
    resp = [PanelResponse(f"IT{i}", "m", f"q{i}", f"a{i}") for i in range(4)]
    sheet = make_blinded_sheet(resp, seed=1)
    tpl = rating_template_csv(sheet, tmp_path / "rater1.csv")
    # a rater fills the verdict column
    rows = list(csv.DictReader(tpl.open(encoding="utf-8")))
    with tpl.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sheet_id", "question", "answer", "verdict(pass/fail)", "notes"])
        for i, row in enumerate(rows):
            w.writerow([row["sheet_id"], row["question"], row["answer"],
                        "pass" if i % 2 else "fail", ""])
    loaded = load_ratings_csv(tpl)
    assert len(loaded) == 4 and set(loaded.values()) <= {"pass", "fail"}


def test_panel_report_calibration():
    resp = [PanelResponse(f"IT{i}", "m", f"q{i}", f"a{i}") for i in range(10)]
    sheet = make_blinded_sheet(resp, seed=3)
    ids = [r["sheet_id"] for r in sheet.rows]
    gold = {sid: ("pass" if i % 2 else "fail") for i, sid in enumerate(ids)}
    humans = {"E1": dict(gold), "E2": dict(gold), "E3": dict(gold)}   # perfect panel
    rule = dict(gold)                                                 # perfect rule judge
    rep = panel_report(sheet, humans, rule_verdicts=rule)
    assert rep["expert_fleiss"] == 1.0 and rep["expert_reliable"] is True
    assert rep["tier_calibration"]["rule"]["scale_ok"] is True
    assert rep["tier_calibration"]["rule"]["kappa"] == 1.0
    assert "error" in panel_report(sheet, {"only": gold})            # <2 raters


def test_dry_run_produces_real_kappa():
    rep = dry_run(n_items=40, seed=42)
    # real κ numbers from the pipeline (synthetic raters — self-test only)
    assert rep["n_items"] == 40 and rep["n_raters"] == 3
    assert isinstance(rep["expert_fleiss"], float)
    # reliable rater/rule config ⇒ rule tier calibrates; structure present
    assert "rule" in rep["tier_calibration"] and "llm" in rep["tier_calibration"]
    assert isinstance(rep["tier_calibration"]["rule"]["scale_ok"], bool)
