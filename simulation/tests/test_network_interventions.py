"""TDD for the leak-free NETWORK-STRUCTURE counterfactual engine.

Interventions the mean-field / demographic-only counterfactual cannot express:
layer closure (school/workplace/community) and network-degree ring vaccination. Every
effect is a Δ vs the no-intervention baseline on a FIXED network/seed set; the real
forward ILI is never read. These guards pin leak-freeness, the zero-op identities, and
the two epidemiological directions the network capability must satisfy.
"""
from __future__ import annotations

import inspect

import pytest

from simulation.abm.synthetic_population import generate_population
from simulation.abm.network_params_from_db import derive_network_kwargs


def _pop_nk(n=1400, seed=0):
    return generate_population(n, seed=seed), derive_network_kwargs()


def test_engine_is_structurally_leak_free():
    from simulation.abm import network_interventions as NI
    for fn in (NI.simulate_counterfactual, NI.compare_interventions):
        params = " ".join(inspect.signature(fn).parameters)
        assert "forward" not in params and "obs" not in params, (
            f"{fn.__name__} must not take forward/observation data")


def test_zero_efficacy_layer_closure_is_zero_delta():
    """efficacy=0 leaves the per-layer betas untouched → byte-identical to baseline."""
    from simulation.abm.network_interventions import simulate_counterfactual
    pop, nk = _pop_nk()
    kw = dict(base_beta=0.16, n_seeds=2, t_days=40, seed0=0)
    base = simulate_counterfactual(pop, nk, intervention=None, **kw)
    noop = simulate_counterfactual(pop, nk, intervention={
        "type": "layer_closure", "layer": "school", "efficacy": 0.0}, **kw)
    assert noop["attack_rate"] == base["attack_rate"]


def test_zero_coverage_immunization_is_zero_delta():
    from simulation.abm.network_interventions import simulate_counterfactual
    pop, nk = _pop_nk()
    kw = dict(base_beta=0.16, n_seeds=2, t_days=40, seed0=0)
    base = simulate_counterfactual(pop, nk, intervention=None, **kw)
    noop = simulate_counterfactual(pop, nk, intervention={
        "type": "targeted_immunization", "coverage": 0.0, "target": "degree"}, **kw)
    assert noop["attack_rate"] == base["attack_rate"]


def test_layer_closure_does_not_increase_attack():
    from simulation.abm.network_interventions import simulate_counterfactual
    pop, nk = _pop_nk()
    kw = dict(base_beta=0.2, n_seeds=3, t_days=44, seed0=0)
    base = simulate_counterfactual(pop, nk, intervention=None, **kw)
    closed = simulate_counterfactual(pop, nk, intervention={
        "type": "layer_closure", "layer": "community", "efficacy": 0.9}, **kw)
    assert closed["attack_rate"] <= base["attack_rate"] + 1e-9


def test_targeted_immunization_monotone_in_coverage():
    from simulation.abm.network_interventions import simulate_counterfactual
    pop, nk = _pop_nk()
    kw = dict(base_beta=0.2, n_seeds=3, t_days=44, seed0=0)
    a = simulate_counterfactual(pop, nk, intervention={
        "type": "targeted_immunization", "coverage": 0.05, "target": "degree"}, **kw)
    b = simulate_counterfactual(pop, nk, intervention={
        "type": "targeted_immunization", "coverage": 0.25, "target": "degree"}, **kw)
    assert b["attack_rate"] <= a["attack_rate"] + 1e-9


def test_degree_targeting_at_least_as_good_as_random():
    """Immunising the highest-degree agents averts >= random at equal coverage — the
    network capability the mean-field / demographic model cannot exploit."""
    from simulation.abm.network_interventions import simulate_counterfactual
    pop, nk = _pop_nk(n=1800)
    kw = dict(base_beta=0.24, n_seeds=4, t_days=44, seed0=0)
    deg = simulate_counterfactual(pop, nk, intervention={
        "type": "targeted_immunization", "coverage": 0.12, "target": "degree"}, **kw)
    rnd = simulate_counterfactual(pop, nk, intervention={
        "type": "targeted_immunization", "coverage": 0.12, "target": "random"}, **kw)
    assert deg["attack_rate"] <= rnd["attack_rate"] + 0.02


def test_compare_interventions_returns_deltas():
    from simulation.abm.network_interventions import compare_interventions
    pop, nk = _pop_nk()
    res = compare_interventions(pop, nk, interventions=[
        {"type": "layer_closure", "layer": "school", "efficacy": 0.5},
        {"type": "targeted_immunization", "coverage": 0.1, "target": "degree"}],
        base_beta=0.2, n_seeds=2, t_days=40, seed0=0)
    assert "baseline" in res and len(res["interventions"]) == 2
    assert "leak_free" in res
    for row in res["interventions"]:
        assert {"attack_rate", "delta_attack_rate", "delta_peak", "averted_fraction"} <= set(row)


def test_agent_degree_and_layer_degrees_shapes():
    from simulation.abm.network_interventions import agent_degree, layer_degrees
    pop, nk = _pop_nk(n=500)
    deg = agent_degree(pop, nk, seed=0)
    assert deg.shape == (500,) and deg.min() >= 0
    ld = layer_degrees(pop, nk, seed=0)
    assert {"household", "workplace", "school", "community"} <= set(ld)
