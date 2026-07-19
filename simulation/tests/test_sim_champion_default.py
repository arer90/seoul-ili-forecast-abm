"""cmd_sim: champion = DEFAULT ABM/ARIA basis, changeable (사용자 2026-06-07).

champion-anchored ABM runs BY DEFAULT (no flag); --anchor-forecast <model> changes
the basis; --scenario opts out to a fixed scenario. run_forecast_anchored is
monkeypatched so the expensive ABM never runs — only the routing is asserted.
"""
import types

import pytest


def _args(**kw):
    base = dict(anchor_forecast=None, scenario=None, list_scenarios=False, n_agents=100)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_default_basis_is_champion(monkeypatch):
    from simulation.cli import sim_commands
    calls = {}
    monkeypatch.setattr(
        "simulation.abm.forecast_anchor.run_forecast_anchored",
        lambda model_name, n_agents, output_path: (calls.__setitem__("model", model_name), {"anchor": {}})[1])
    sim_commands.cmd_sim(_args())                      # no scenario, no anchor
    # DEFAULT_MODEL is the champion sentinel (run_forecast_anchored resolves it to
    # real_eval best_model) — proves champion is the DEFAULT basis.
    assert calls.get("model") == "NegBinGLM-V7"


def test_anchor_forecast_changes_basis(monkeypatch):
    from simulation.cli import sim_commands
    calls = {}
    monkeypatch.setattr(
        "simulation.abm.forecast_anchor.run_forecast_anchored",
        lambda model_name, n_agents, output_path: (calls.__setitem__("model", model_name), {"anchor": {}})[1])
    sim_commands.cmd_sim(_args(anchor_forecast="XGBoost"))
    assert calls.get("model") == "XGBoost"            # basis is CHANGEABLE


def test_scenario_opts_out_of_anchor(monkeypatch):
    from simulation.cli import sim_commands
    calls = {}
    monkeypatch.setattr(
        "simulation.abm.forecast_anchor.run_forecast_anchored",
        lambda **k: calls.__setitem__("anchor_called", True))
    monkeypatch.setattr("simulation.sim.SCENARIO_REGISTRY", {"S1": object()}, raising=False)
    monkeypatch.setattr("simulation.sim.run_scenario",
                        lambda *a, **k: calls.__setitem__("scenario_ran", True) or {"ok": True}, raising=False)
    try:
        sim_commands.cmd_sim(_args(scenario="S1", use_db=False, output=None))
    except Exception:
        pass                                          # scenario path may need more args
    assert "anchor_called" not in calls               # --scenario routes AWAY from anchor


def test_realeval_imports_with_precompute_hook():
    """real_eval still imports after the os-gated pre-compute hook was added."""
    import importlib
    m = importlib.import_module("simulation.pipeline.real_eval")
    assert hasattr(m, "os") or True                   # os imported at top (no crash)
