"""A1 (M7 SCI-grade): operational deployment gate.

An evaluation-best champion can extrapolate-collapse on the real slab (pred ≈1007
vs observed ≈21). The deployment gate REJECTS a contract-violating forecast and
REPLACES it with the stable fallback, and the ABM/ARIA path reads that gated
DEPLOYMENT forecast — separating the retrospective champion from what we deploy.
"""
import json

import numpy as np

from simulation.pipeline.real_eval import _gate_forecast


def test_sane_forecast_passes():
    y_train = np.array([10.0, 12.0, 11.0, 13.0, 9.0, 14.0])
    g = _gate_forecast(np.array([12.0, 13.0, 11.0]), y_train,
                       fallback=np.array([10.0, 10.0, 10.0]))
    assert not g["replaced"] and g["n_violations"] == 0
    assert list(g["pred"]) == [12.0, 13.0, 11.0]


def test_collapsed_forecast_replaced_with_fallback():
    y_train = np.array([10.0, 12.0, 11.0, 13.0, 9.0, 14.0])  # max 14 → cap 42
    g = _gate_forecast(np.array([12.0, 1007.0, 11.0]), y_train,
                       fallback=np.array([11.0, 11.0, 11.0]), k=3.0)
    assert g["replaced"] and g["n_violations"] >= 1
    assert list(g["pred"]) == [11.0, 11.0, 11.0]
    assert "train_max" in g["reason"]


def test_negative_and_nonfinite_flagged():
    y_train = np.arange(1, 20, dtype=float)
    g1 = _gate_forecast(np.array([5.0, -3.0]), y_train, fallback=np.array([5.0, 5.0]))
    assert g1["replaced"] and "negative" in g1["reason"]
    g2 = _gate_forecast(np.array([5.0, np.inf]), y_train, fallback=np.array([5.0, 5.0]))
    assert g2["replaced"] and "non-finite" in g2["reason"]


def test_no_fallback_keeps_pred_but_counts_violations():
    g = _gate_forecast(np.array([5.0, 999.0]), np.arange(1, 20, dtype=float), fallback=None)
    assert not g["replaced"] and g["n_violations"] >= 1  # flagged, not replaced


def test_load_real_forecast_prefers_deployment(tmp_path):
    from simulation.abm.forecast_anchor import load_real_forecast

    re_dir = tmp_path / "real_eval"
    (re_dir / "per_model").mkdir(parents=True)
    # champion per-model holds the COLLAPSED forecast; deployment holds the stable one
    (re_dir / "summary.json").write_text(json.dumps({
        "best_model": "NegBinGLM",
        "deployment": {"model": "median_ensemble", "replaced": True,
                       "forecast": [11.0, 12.0, 10.0]},
    }), encoding="utf-8")
    (re_dir / "per_model" / "NegBinGLM.json").write_text(
        json.dumps({"predictions": [1007.0, 980.0, 1100.0]}), encoding="utf-8")
    _weeks, fc = load_real_forecast(re_dir)
    assert list(fc) == [11.0, 12.0, 10.0]  # deployment, not the collapsed champion
