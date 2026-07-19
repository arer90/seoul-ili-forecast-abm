"""M3: trained-models.json generator from live per_model_eval (replaces v22.6 frozen)."""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "web" / "scripts"))
import build_trained_models as btm  # noqa: E402


def _fake_csv(tmp_path):
    p = tmp_path / "per_model_metrics.csv"
    cols = ["model", "wis", "r2", "rmse", "crps_gaussian", "pi95_coverage", "mape"]
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerow({"model": "ARIMA", "wis": "14.9", "r2": "-0.3", "rmse": "30",
                    "crps_gaussian": "12", "pi95_coverage": "0.6", "mape": "60"})
        w.writerow({"model": "NegBinGLM-V7", "wis": "3.25", "r2": "0.92", "rmse": "6.9",
                    "crps_gaussian": "3", "pi95_coverage": "0.97", "mape": "16"})
    return p


def test_build_sorts_by_wis_and_maps_schema(tmp_path):
    out = tmp_path / "trained-models.json"
    btm.build(_fake_csv(tmp_path), out, timestamp="2026-06-06")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["version"] == "live" and data["total_models"] == 2
    assert data["source"] == "per_model_eval/per_model_metrics.csv"
    top = data["top"]
    assert top[0]["name"] == "NegBinGLM-V7" and top[0]["rank"] == 1 and top[0]["wis"] == 3.25
    assert top[0]["cov95"] == 0.97 and top[0]["crps"] == 3.0
    assert top[1]["name"] == "ARIMA" and top[1]["rank"] == 2
