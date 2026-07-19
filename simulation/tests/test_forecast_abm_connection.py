"""Node 1 connection contract: P1 real_forecaster → ABM forecast anchor (2026-06-06).

Guards that the ABM reads its forecast input from the LIVE real_forecaster
output (operational best_model on the real slab), not the retired R10
per_model_eval CSV. Regression for the forecast→ABM DISCONNECT (forecast_anchor
had 0 production callers + a stale legacy R10 per_model_eval DEFAULT_PATH).
"""
import json

import numpy as np
import pytest

from simulation.abm.forecast_anchor import load_real_forecast


def _fake_real_eval(tmp_path, preds=(5.0, 6.0, 7.0), best="individual_results"):
    """Build a minimal real_eval dir (best_model degenerate on tiny runs)."""
    d = tmp_path / "real_eval"
    (d / "per_model").mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({"best_model": best}), encoding="utf-8")
    (d / "metrics_full.json").write_text(
        json.dumps({"ar1": {"predictions": list(preds)}}), encoding="utf-8"
    )
    (d / "per_model" / "ar1.json").write_text(
        json.dumps({"predictions": list(preds)}), encoding="utf-8"
    )
    return d


def test_falls_back_when_best_model_invalid(tmp_path):
    # summary best_model = "individual_results" (degenerate) → fall through to ar1
    d = _fake_real_eval(tmp_path, (5.0, 6.0, 7.0))
    weeks, fc = load_real_forecast(d)
    assert list(fc) == [5.0, 6.0, 7.0]
    assert list(weeks) == [0, 1, 2]


def test_explicit_model(tmp_path):
    d = _fake_real_eval(tmp_path, (9.0, 8.0))
    weeks, fc = load_real_forecast(d, model_name="ar1")
    assert list(fc) == [9.0, 8.0] and len(weeks) == 2


def test_uses_summary_best_model_when_valid(tmp_path):
    d = _fake_real_eval(tmp_path, (1.0, 2.0, 3.0), best="ar1")
    _weeks, fc = load_real_forecast(d)
    assert list(fc) == [1.0, 2.0, 3.0]


def test_nonfinite_raises(tmp_path):
    d = _fake_real_eval(tmp_path, (5.0, float("nan"), 7.0))
    with pytest.raises(ValueError):
        load_real_forecast(d, model_name="ar1")


def test_missing_predictions_raises(tmp_path):
    d = tmp_path / "empty_real_eval"
    d.mkdir()
    with pytest.raises(ValueError):
        load_real_forecast(d)


# ── M1 CLI wiring: `sim --anchor-forecast` → run_forecast_anchored ──────────
def test_cmd_sim_anchor_forecast_routes(monkeypatch):
    """`sim --anchor-forecast MODEL` drives the ABM from the forecast (forecast→ABM)."""
    import argparse
    import simulation.abm.forecast_anchor as fa
    from simulation.cli.sim_commands import cmd_sim

    captured = {}

    def _fake(model_name, n_agents, output_path, **k):
        captured.update(model=model_name, n=n_agents)
        return {"anchor": {"correlation": 0.9, "degenerate": False}}

    monkeypatch.setattr(fa, "run_forecast_anchored", _fake)
    cmd_sim(argparse.Namespace(anchor_forecast="ar1", n_agents=123))
    assert captured == {"model": "ar1", "n": 123}


def test_cmd_sim_without_anchor_skips_forecast(monkeypatch):
    """No --anchor-forecast → run_forecast_anchored is never called."""
    import argparse
    import simulation.abm.forecast_anchor as fa
    from simulation.cli.sim_commands import cmd_sim

    called = []
    monkeypatch.setattr(fa, "run_forecast_anchored", lambda **k: called.append(1))
    cmd_sim(argparse.Namespace(anchor_forecast=None, list_scenarios=True))
    assert called == []
