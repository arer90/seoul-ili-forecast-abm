"""A4 (M7 SCI-grade): the Phase-11 ad-hoc composite is a diagnostic only.

The composite (R²+RMSE+DM+stability+conformal weighted blend) contradicts the
project's pure best-WIS champion policy, so `_ranking_consolidated` must NOT fold
it into the Borda consolidation — it lives under `diagnostics` and cannot move
the consolidated ranking.
"""
from simulation.pipeline.comprehensive_eval import _ranking_consolidated


def _all_results():
    """Synthetic phase results where composite DISAGREES with WIS/OOF/pairwise."""
    return {
        # OOF R²: A > B > C
        "wfcv": {"wf_results": {"A": {"r2": 0.9}, "B": {"r2": 0.8}, "C": {"r2": 0.7}}},
        # composite: C > B > A  (the contradicting order — would favour C)
        "scoring": {"scores": {"A": {"composite": 0.1}, "B": {"composite": 0.5},
                               "C": {"composite": 0.9}}},
        # WIS + pairwise: A best
        "per_model_eval": {
            "ranking_top10": ["A", "B", "C"],
            "pairwise_relative_wis": {"A": 1.0, "B": 1.5, "C": 2.0},
        },
    }


def test_composite_is_not_a_borda_source(tmp_path):
    res = _ranking_consolidated(_all_results(), tmp_path)
    assert "phase9_composite" not in res["sources"], "composite must not be a Borda source"
    # still reported as a diagnostic for continuity/inspection
    assert res["diagnostics"]["composite_order"][0] == "C"


def test_consolidated_follows_wis_not_composite(tmp_path):
    res = _ranking_consolidated(_all_results(), tmp_path)
    top = res["consolidated"][0]["model"]
    assert top == "A", f"consolidated top should be the WIS/OOF/pairwise winner, got {top}"
    assert top != "C", "composite-favoured model must not be pulled to the top"


def test_borda_sources_are_proper_scoring_only(tmp_path):
    res = _ranking_consolidated(_all_results(), tmp_path)
    # Borda sources = OOF-R² + WIS + pairwise (no composite)
    assert set(res["sources"]) <= {"phase7_oof_r2", "phase11_wis", "phase11_pairwise"}
    assert "phase11_wis" in res["sources"]
