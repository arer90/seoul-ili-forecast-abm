"""simulation.abm.network_interventions — leak-free NETWORK-STRUCTURE counterfactuals.

The agent-to-agent contact network + who-infected-whom tree let the ABM express
interventions the mean-field model — and the demographic-only counterfactual in
:mod:`simulation.abm.counterfactual` (occupation/age proxy, no contact graph) —
structurally cannot:

  * **Layer closure** — shut a specific contact layer (school / workplace /
    community) by scaling ONLY that layer's per-contact hazard, using the
    data-derived, scale-preserving per-layer betas
    (:func:`simulation.abm.network_params_from_db.derive_beta_by_layer`).
  * **Network-degree ring vaccination** — immunise the highest-degree agents in the
    ACTUAL contact graph (not an occupation/age surrogate), the classic targeted
    strategy that only a network model can evaluate, versus random at equal coverage.

Leak-free by construction: every effect is reported as a Δ against the
no-intervention baseline under the SAME fixed network and seeds; the real forward
ILI is never read (these are forward SCENARIOS, not forecasts — not even the scoring
touches observed truth).
"""
from __future__ import annotations

import numpy as np

from simulation.abm.agent_kernel import (
    STATE_D, STATE_E, STATE_I, STATE_R, run_agent_world)
from simulation.abm.contact_network import build_multilayer_network, degree_summary
from simulation.abm.network_params_from_db import derive_beta_by_layer

__all__ = ["simulate_counterfactual", "compare_interventions", "layer_degrees",
           "agent_degree"]

# season-tail disease params, matched to the ablation harness (variant_ablation)
_DISEASE = dict(sigma=0.45, gamma=0.18, delta=0.002, nu=0.0002)
_EVER = (STATE_E, STATE_I, STATE_R, STATE_D)


def _nk_clean(network_kwargs: dict) -> dict:
    return {k: v for k, v in (network_kwargs or {}).items() if k != "provenance"}


def layer_degrees(pop: dict, network_kwargs: dict, *, seed: int = 0) -> dict:
    """Mean per-agent degree in each contact layer (household/workplace/school/
    community), from the deterministic network at ``seed``. Excludes ``_total``."""
    layers = build_multilayer_network(pop, seed=int(seed), **_nk_clean(network_kwargs))
    deg = degree_summary(layers)
    return {k: float(v) for k, v in deg.items() if k != "_total"}


def agent_degree(pop: dict, network_kwargs: dict, *, seed: int = 0) -> np.ndarray:
    """Per-agent TOTAL contact degree over all layers (for degree-ring targeting)."""
    layers = build_multilayer_network(pop, seed=int(seed), **_nk_clean(network_kwargs))
    n = pop["home_gu"].size
    total = np.zeros(n, dtype=np.float64)
    for name in layers:
        total += np.asarray(layers[name].sum(axis=1)).ravel()
    return total


def _immune_mask(pop: dict, network_kwargs: dict, *, coverage: float, target: str,
                 seed: int) -> np.ndarray:
    """Boolean length-N mask of agents to pre-immunise under ``target`` at ``coverage``."""
    n = pop["home_gu"].size
    k = int(round(float(coverage) * n))
    mask = np.zeros(n, dtype=bool)
    if k <= 0:
        return mask
    if target == "degree":
        deg = agent_degree(pop, network_kwargs, seed=seed)
        idx = np.argsort(deg, kind="stable")[::-1][:k]      # highest-degree first
    elif target == "random":
        idx = np.random.default_rng(seed).permutation(n)[:k]
    else:
        raise ValueError(f"unknown immunisation target: {target!r}")
    mask[idx] = True
    return mask


def _beta_by_layer(pop: dict, network_kwargs: dict, *, base_beta: float, seed: int,
                   intervention: dict | None) -> dict:
    """Data-derived scale-preserving per-layer betas, with a layer-closure override."""
    degs = layer_degrees(pop, network_kwargs, seed=seed)
    bbl = derive_beta_by_layer(base_beta, degs)
    if intervention and intervention.get("type") == "layer_closure":
        layer = intervention["layer"]
        if layer not in bbl:
            raise ValueError(f"layer_closure target {layer!r} not in {list(bbl)}")
        eff = float(intervention.get("efficacy", 1.0))
        bbl = dict(bbl)
        bbl[layer] = bbl[layer] * (1.0 - eff)               # shut only this layer
    return bbl


def _one_run(pop: dict, network_kwargs: dict, *, base_beta: float, beta_by_layer: dict,
             immune_mask: np.ndarray | None, seed: int, net_seed: int, t_days: int) -> dict:
    r = run_agent_world(
        N=pop["home_gu"].size, T_days=t_days, beta=base_beta, global_seed=int(seed),
        import_rate=3.0e-4, population=pop, transmission_mode="hybrid", hybrid_weight=0.5,
        beta_by_layer=beta_by_layer, network_kwargs=network_kwargs, network_seed=int(net_seed),
        initial_vaccinated=(immune_mask if immune_mask is not None and immune_mask.any()
                            else None),
        **_DISEASE)
    state = np.asarray(r["agents"]["state"])
    ever = np.isin(state, _EVER)
    return {"attack": float(ever.mean()), "peak": float(np.asarray(r["I"]).max())}


def simulate_counterfactual(pop: dict, network_kwargs: dict, *, base_beta: float = 0.15,
                            n_seeds: int = 3, t_days: int = 44, seed0: int = 0,
                            intervention: dict | None = None) -> dict:
    """Score one scenario (baseline or a single intervention) on the network ABM.

    The contact network is FIXED (``network_seed = seed0``) across the ``n_seeds``
    epidemic replicates, so the only thing that varies is transmission stochasticity —
    isolating the intervention's effect from network-sampling noise. ``intervention``
    is ``None`` (baseline) or one of::

        {"type": "layer_closure", "layer": "school", "efficacy": 0.5}
        {"type": "targeted_immunization", "coverage": 0.10, "target": "degree"|"random"}

    Args:
        pop: fixed synthetic population (``generate_population``).
        network_kwargs: data-derived contact structure (``derive_network_kwargs``).
        base_beta: aggregate transmission scale (redistributed per layer, leak-free).
        n_seeds: epidemic replicates over the fixed network.
        t_days: forward horizon in days.
        seed0: base seed; also the fixed network seed.
        intervention: the scenario, or ``None`` for the untouched baseline.

    Returns:
        ``{"attack_rate", "peak_prevalence", "n_seeds", "intervention"}`` — means over
        the replicates. ``attack_rate`` is the population share ever infected
        (immunised agents count as not-infected, i.e. averted).

    Side effects: none (reads nothing but the fixed pop/network; no DB, no disk).
    Caller responsibility: this is a counterfactual SCENARIO — no real forward ILI is
    read here or in any downstream Δ; do not feed observed truth in.
    """
    immune = None
    beta_by_layer = _beta_by_layer(pop, network_kwargs, base_beta=base_beta, seed=seed0,
                                   intervention=intervention)
    if intervention and intervention.get("type") == "targeted_immunization":
        immune = _immune_mask(pop, network_kwargs, coverage=intervention.get("coverage", 0.0),
                              target=intervention.get("target", "degree"), seed=seed0)
    atk, pk = [], []
    for s in range(n_seeds):
        out = _one_run(pop, network_kwargs, base_beta=base_beta, beta_by_layer=beta_by_layer,
                       immune_mask=immune, seed=seed0 + s, net_seed=seed0, t_days=t_days)
        atk.append(out["attack"]); pk.append(out["peak"])
    return {"attack_rate": float(np.mean(atk)), "peak_prevalence": float(np.mean(pk)),
            "n_seeds": n_seeds, "intervention": intervention}


def compare_interventions(pop: dict, network_kwargs: dict, interventions: list, *,
                          base_beta: float = 0.15, n_seeds: int = 3, t_days: int = 44,
                          seed0: int = 0) -> dict:
    """Baseline + each intervention, reported as Δ attack rate / Δ peak (averted burden).

    Every arm shares the fixed network and seed set, so each Δ is a within-network
    counterfactual contrast. Leak-free: no real forward ILI enters any arm or Δ.

    Returns:
        ``{"baseline": {...}, "interventions": [{**intervention, attack_rate,
        delta_attack_rate, delta_peak, averted_fraction}, ...], "leak_free": str}``.
    """
    base = simulate_counterfactual(pop, network_kwargs, base_beta=base_beta,
                                   n_seeds=n_seeds, t_days=t_days, seed0=seed0,
                                   intervention=None)
    base_atk = base["attack_rate"]
    rows = []
    for iv in interventions:
        r = simulate_counterfactual(pop, network_kwargs, base_beta=base_beta, n_seeds=n_seeds,
                                    t_days=t_days, seed0=seed0, intervention=iv)
        d_atk = r["attack_rate"] - base_atk
        rows.append({**iv, "attack_rate": round(r["attack_rate"], 4),
                     "delta_attack_rate": round(d_atk, 4),
                     "delta_peak": round(r["peak_prevalence"] - base["peak_prevalence"], 3),
                     "averted_fraction": round(-d_atk / base_atk, 4) if base_atk > 0 else 0.0})
    return {"baseline": {"attack_rate": round(base_atk, 4),
                         "peak_prevalence": round(base["peak_prevalence"], 3)},
            "interventions": rows,
            "leak_free": "counterfactual scenarios vs a no-intervention baseline on a "
                         "fixed network/seed set; real forward ILI never read (not forecasts)"}
