"""M6: end-to-end connection integration (2026-06-06).

Verifies forecast→ABM (M1), ARIA (M2), and Web (M3) all read the SAME live
pipeline outputs (real_eval / per_model_eval / shap) consistently — i.e. the
downstream chain is actually connected, not 4 disconnected seams reading stale
artifacts. Fast integration over a toy fixture (the full fresh smoke run is the
final pre-retrain check, M8/user-gated).
"""
import csv
import json
import sys
from pathlib import Path


def _build_live_results(root: Path) -> Path:
    """Consistent toy pipeline outputs (post-RENUMBER semantic names)."""
    re_dir = root / "real_eval"
    (re_dir / "per_model").mkdir(parents=True)
    (re_dir / "summary.json").write_text(json.dumps({"best_model": "ar1", "real_n": 3}), encoding="utf-8")
    (re_dir / "per_model" / "ar1.json").write_text(
        json.dumps({"predictions": [10.0, 12.0, 11.0]}), encoding="utf-8")

    pm_dir = root / "per_model_eval"
    pm_dir.mkdir(parents=True)
    cols = ["model", "wis", "r2", "rmse", "crps_gaussian", "pi95_coverage", "mape",
            "dm_p_value", "dm_p_value_bh", "lead_time_weeks"]
    with (pm_dir / "per_model_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerow({"model": "NegBinGLM-V7", "wis": "3.25", "r2": "0.92", "rmse": "6.9",
                    "crps_gaussian": "3", "pi95_coverage": "0.97", "mape": "16",
                    "dm_p_value": "0.01", "dm_p_value_bh": "0.02", "lead_time_weeks": "2.0"})

    sh = root / "shap"
    sh.mkdir(parents=True)
    (sh / "_summary.json").write_text(json.dumps({"n_models": 1, "mi_top10": ["temp"]}), encoding="utf-8")
    return root


def test_chain_reads_same_live_outputs(tmp_path):
    root = _build_live_results(tmp_path / "results")

    # M1 — forecast→ABM reads real_eval
    from simulation.abm.forecast_anchor import load_real_forecast
    _weeks, fc = load_real_forecast(root / "real_eval")
    assert list(fc) == [10.0, 12.0, 11.0]

    # M2 — ARIA MCP tools read the live outputs
    from simulation.server.mcp_epi import EpiMCPServer
    srv = EpiMCPServer(artifacts_dir=root)
    assert srv._h_model_compare({}).content["status"] == "live"
    assert srv._h_lead_time({}).content["status"] == "live"
    wired = srv._probe_wired()
    assert wired["epi.model_compare"] and wired["epi.shap_features"] and wired["epi.lead_time_analysis"]

    # M3 — Web trained-models.json generated from the same per_model_eval
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "web" / "scripts"))
    import build_trained_models as btm  # noqa: E402
    out = tmp_path / "trained-models.json"
    btm.build(root / "per_model_eval" / "per_model_metrics.csv", out, timestamp="2026-06-06")
    tm = json.loads(out.read_text(encoding="utf-8"))
    assert tm["version"] == "live" and tm["top"][0]["name"] == "NegBinGLM-V7"
