"""
simulation.abm
==============
Four-parameter behavioural agent-based model layer for the Seoul metapop
SEIR-V-D simulator (/ thesis §3.4a + §4.16).

Public API
----------
- ``BehaviouralParams`` — risk / fatigue ODE parameters (alpha, kappa, tau, theta)
- ``ABMResult`` — extends SimResult with behavioural-state trajectories
- ``run_coupled_abm(metapop_params, behav, *, verbose=False)`` — the coupled run
- ``run_invariant_test(metapop_params)`` — α = 0, κ = 0, τ → ∞ check
- ``run_rebound_scenario(metapop_params, *, behav_off, behav_on)`` — §4.16 headline

The module is pure NumPy. No Rust, no Mesa. District-level aggregation of
the behavioural state (R_i(t), F_i(t), compliance_i(t) per gu) is treated
as the mean-field approximation of a ~10 000-agent population, which is
defensible at 25-district resolution where the empirical variance within
a gu is much smaller than the variance across gu.
"""
from __future__ import annotations

from .behavioural import (
    BehaviouralParams,
    ABMResult,
    run_coupled_abm,
    run_invariant_test,
    run_rebound_scenario,
)

__all__ = [
    "BehaviouralParams",
    "ABMResult",
    "run_coupled_abm",
    "run_invariant_test",
    "run_rebound_scenario",
]
