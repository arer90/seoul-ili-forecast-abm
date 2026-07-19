"""
simulation.sim
==============
Stage 5 — Metapopulation SEIR-V-D simulator (25-gu, commuter-coupled).

Design principles
-----------------
1. **Pure numpy core.** The stepper is RK4; scipy.integrate is an optional
 fallback but not required. Keeps the sim import-light.
2. **6 compartments.** S → E → I → R plus a vaccination branch (V) and a
 terminal deaths compartment (D). V is modeled leaky — vaccinated
 individuals have susceptibility scaled by (1 − VE).
3. **Commuter coupling.** The 25×25 row-stochastic mobility matrix
 ``M[i, j]`` is the fraction of district i's residents present in
 district j during the infectious-contact phase of the day. The
 force-of-infection folds in both the home and work sides.
4. **Interventions as piecewise-constant multipliers.** Any parameter
 (β, ν, γ, …) can be scaled or replaced during a date window via
 ``InterventionSpec``. Scenarios compose interventions.
5. **Validation hook.** After a run, the trajectory is fed to the
 Stage-4 epi-validity gate — compartment conservation, Rt bounds,
 seasonal peak — and the report is attached to the result.

Public API
----------
- ``MetapopParams`` — disease + mobility + initial conditions
- ``InterventionSpec`` — time-windowed parameter modifier
- ``MetapopSEIRVD`` — simulator class
- ``run_scenario(...)`` — entry point that resolves a scenario name
 from the registry and returns a SimResult
- ``SimResult`` — outcome of a run (state trajectory + gate report)

Notes
-----
- State shape is always ``(T, G, 6)`` where G = number of districts.
 The 6-axis order is ``(S, E, I, R, V, D)`` — frozen; compile-time
 consumers should reference the ``COMPARTMENTS`` tuple below.
- The simulator is deterministic given the same ``seed`` argument.
 Leave ``seed=None`` for a stochastic initial perturbation (NumPy
 ``default_rng``).
"""
from __future__ import annotations

from .parameters import (
    COMPARTMENTS,
    DEFAULT_FLU_PARAMS,
    InterventionSpec,
    MetapopParams,
    select_disease_params,
)
from .metapop_seirvd import MetapopSEIRVD, SimResult, run_scenario
from .scenarios import SCENARIO_REGISTRY, register_scenario

# Codex non-bio review #10 fix (sprint 2026-05-06): auto-register the 8
# extended scenarios from `scenarios_extended.py` so the `python -m
# simulation sim --list-scenarios` CLI and the MCP `epi.scenario_run`
# tool both see the full registry. Without this call the eight scenarios
# (school_closure, mask_mandate, school_closure_plus_mask, etc.) lived
# only as a dict and never reached the registry.
from .scenarios_extended import register_extended_scenarios as _register_extended

_register_extended()
del _register_extended

__all__ = [
    "COMPARTMENTS",
    "DEFAULT_FLU_PARAMS",
    "InterventionSpec",
    "MetapopParams",
    "MetapopSEIRVD",
    "SimResult",
    "run_scenario",
    "SCENARIO_REGISTRY",
    "register_scenario",
    "select_disease_params",
]
