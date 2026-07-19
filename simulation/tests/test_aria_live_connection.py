"""M2 connection contract: ARIA (MCP epi tools) read LIVE pipeline outputs (2026-06-06).

Guards that epi.model_compare / epi.lead_time_analysis / epi.shap_features read the
live semantic outputs (per_model_eval/per_model_metrics.csv, shap/_summary.json)
instead of the retired stage3/4 artifact names (which RENUMBER no longer produces,
so the tools always degraded to not_available).
"""
import csv
import json

import pytest

from simulation.server.mcp_epi import EpiMCPServer


@pytest.fixture
def artifacts(tmp_path):
    ad = tmp_path / "results"
    (ad / "shap").mkdir(parents=True)
    (ad / "per_model_eval").mkdir(parents=True)
    (ad / "shap" / "_summary.json").write_text(
        json.dumps({"n_models": 2, "mi_top10": ["temp_avg", "ili_lag1"]}),
        encoding="utf-8",
    )
    cols = ["model", "wis", "mae", "r2", "dm_p_value", "dm_p_value_bh", "lead_time_weeks"]
    with (ad / "per_model_eval" / "per_model_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerow({"model": "NegBinGLM-V7", "wis": "3.25", "mae": "4.2", "r2": "0.92",
                    "dm_p_value": "0.01", "dm_p_value_bh": "0.02", "lead_time_weeks": "2.0"})
        w.writerow({"model": "ARIMA", "wis": "14.9", "mae": "19", "r2": "-0.3",
                    "dm_p_value": "0.5", "dm_p_value_bh": "0.6", "lead_time_weeks": "1.0"})
    return ad


def test_model_compare_reads_live_csv(artifacts):
    srv = EpiMCPServer(artifacts_dir=artifacts)
    rc = srv._h_model_compare({"metric": "wis"})
    assert rc.content["status"] == "live"
    assert rc.content["source"] == "per_model_eval/per_model_metrics.csv"
    # sorted by wis → champion first
    assert rc.content["models"][0]["model"] == "NegBinGLM-V7"
    assert rc.content["models"][0]["dm_p_value"] == 0.01


def test_lead_time_reads_live_csv(artifacts):
    srv = EpiMCPServer(artifacts_dir=artifacts)
    rl = srv._h_lead_time({})
    assert rl.content["status"] == "live"
    assert len(rl.content["models"]) == 2


def test_shap_features_reads_live_summary(artifacts):
    srv = EpiMCPServer(artifacts_dir=artifacts)
    rs = srv._h_shap_features({})
    # my R11 (shap) SHAP summary flows through the artifact view (data present)
    assert "n_models" in json.dumps(rs.content)


def test_gates_report_wired_on_live_outputs(artifacts):
    srv = EpiMCPServer(artifacts_dir=artifacts)
    w = srv._probe_wired()
    assert w["epi.model_compare"]
    assert w["epi.shap_features"]
    assert w["epi.lead_time_analysis"]
