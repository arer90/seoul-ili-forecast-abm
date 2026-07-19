"""End-to-end tests for the real-Seoul ABM proof comparisons."""
from __future__ import annotations

import json
from pathlib import Path

from simulation.abm import epi_proof


RESULT_PATH = (
    Path(__file__).resolve().parents[2]
    / "paper"
    / "_thesis_revision_20260604"
    / "real_runs"
    / "epi_proof.json"
)


def _run_results() -> dict:
    return epi_proof.run_epi_proof()


def test_runs_end_to_end() -> None:
    results = _run_results()
    assert len(results["metadata"]["available_seasons"]) >= 2
    for key in (
        "comparison_1_behaviour",
        "comparison_2_heterogeneity",
        "comparison_3_movement",
    ):
        assert "delta_wis" in results[key]
        assert "dm_t" in results[key]
        assert "hlm_p" in results[key]
        assert isinstance(results[key]["delta_wis"], float)
        assert 0.0 <= results[key]["hlm_p"] <= 1.0


def test_out_of_sample_discipline() -> None:
    results = _run_results()
    assert results["metadata"]["eval_season"] != results["metadata"]["cal_season"]


def test_deterministic() -> None:
    first = _run_results()
    second = _run_results()
    for key in (
        "comparison_1_behaviour",
        "comparison_2_heterogeneity",
        "comparison_3_movement",
    ):
        assert round(first[key]["delta_wis"], 6) == round(second[key]["delta_wis"], 6)


def test_json_written() -> None:
    _run_results()
    assert RESULT_PATH.exists()
    with RESULT_PATH.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["metadata"]["eval_season"] != data["metadata"]["cal_season"]
