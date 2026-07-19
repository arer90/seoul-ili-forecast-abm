"""Agent-based behavioural ABM (`simulation.abm.agent_based`) unit tests.

Implements + guards the individual-household-agent layer that the thesis §3.4a
(gap G1) recorded but had **not implemented** — the model ran at ~25 mean-field
aggregate agents while the title claims a multi-agent simulation.

Key correctness property: as ``theta_sd -> 0`` with large ``n_agents`` every
agent becomes identical, so the agent model must reproduce the mean-field
``run_coupled_abm`` exactly.
"""
from __future__ import annotations

import numpy as np
import pytest

from simulation.abm.behavioural import BehaviouralParams, run_coupled_abm
from simulation.abm.agent_based import run_agent_abm
from simulation.sim.parameters import DEFAULT_FLU_PARAMS, MetapopParams


def _toy_params(G=6, days=40, dt=0.25, infected0=500.0):
    pops = np.full(G, 100_000.0)
    M = np.full((G, G), 0.05 / (G - 1))
    np.fill_diagonal(M, 0.95)
    return MetapopParams(
        disease=DEFAULT_FLU_PARAMS, populations=pops, mobility=M,
        district_names=[f"d{i}" for i in range(G)],
        initial_infected=np.full(G, infected0), days=days, dt=dt, seed=0,
    )


REBOUND = BehaviouralParams(alpha=2.0, kappa=0.3, tau=90.0, theta=0.1)


# Mild, NON-chaotic config for the equivalence check. The aggressive REBOUND
# regime (high alpha) is deterministically chaotic: the hard compliance
# threshold ``1[R-κF>θ]`` is FP-sensitive at near-critical crossings, so the
# agent ``(G, N)`` arrays and the mean-field ``(G,)`` arrays round one decision
# differently and the trajectories then diverge (documented in
# docs/ABM_NUMERICAL_REMEDIATION_3WAY.md). A mild config stays clear of that
# bifurcation, so the mean-field recovery is bit-exact and a reliable contract.
STABLE_EQUIV = BehaviouralParams(alpha=0.3, kappa=0.1, tau=40.0, theta=0.2)


def test_agent_recovers_mean_field():
    """theta_sd=0 + large N ⇒ the agent model reduces to the mean-field — now EXACT.

    Mathematically exact (homogeneous thresholds ⇒ every agent makes the identical
    decision ⇒ the realised fraction equals the mean-field's). This was previously
    xfail: the explicit-RK4 kernel was cross-process FP-chaotic, amplifying sub-ULP
    (G,N)-vs-(G,) shape rounding into divergence in ~1/3 of runs (docs/ABM_NUMERICAL_
    REMEDIATION_3WAY.md, "needs a stable solver — future work"). 2026-06-10 made the
    mass-conserving exp-Euler the DEFAULT integrator, which removes that chaos: the
    equivalence now holds to machine precision (verified max|Δ|=0.0 across seeds), so
    the xfail marker is retired.
    """
    p = _toy_params()
    mf = run_coupled_abm(p, STABLE_EQUIV)
    ag = run_agent_abm(p, STABLE_EQUIV, n_agents=20_000, theta_sd=0.0, seed=1)
    a, b = mf.city_I(), ag.city_I()
    assert np.max(np.abs(a - b)) < 1e-6, f"max|Δ|={np.max(np.abs(a - b)):.3e}"


def test_agent_count_is_literal():
    """total_agents = n_agents * G — the 'multi-agent' claim is now literal."""
    p = _toy_params(G=6)
    ag = run_agent_abm(p, REBOUND, n_agents=5_000, theta_sd=0.2, seed=1)
    assert ag.total_agents == 5_000 * 6
    assert ag.n_agents == 5_000


def test_heterogeneity_smooths_compliance():
    """With household heterogeneity (theta_sd>0) the compliance fraction takes
    interior values in (0,1), not just the mean-field's hard {0,1}."""
    p = _toy_params()
    ag = run_agent_abm(p, REBOUND, n_agents=10_000, theta_sd=0.3, seed=1)
    frac = ag.compliance_fraction
    assert np.all((frac >= 0.0) & (frac <= 1.0))
    interior = np.mean((frac > 1e-3) & (frac < 1 - 1e-3))
    assert interior > 0.0  # at least some days are partial-compliance


def test_agent_validation():
    p = _toy_params()
    with pytest.raises(ValueError):
        run_agent_abm(p, REBOUND, n_agents=0)
