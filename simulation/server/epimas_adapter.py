"""EpiMAS thin facade — Phase C8 (2026-05-12).

Background:
    User's "EpiMAS sprint" reset directive (2026-05-11) explicitly stated
    "EpiMAS 새로 만들지 마 — simulation/ 토대 정리 + thin MCP facade".
    The previous EpiMAS scaffolding (untracked `EpiMAS/` directory, 2.5GB)
    is dead-code; the real EpiMAS surface = thin re-export of
    ``simulation.sim`` (Stage 5 SEIR simulator) + ``simulation.server.mcp_epi``
    (Stage 6 ARIA tools).

Design (D-4 deep module, D-5 gray-box):
    Small public surface (3 functions + 4 dataclasses), no new state.
    Internal: delegates to ``simulation.sim.run_scenario`` + scenarios_extended
    registry (already wired to Bracher 2021 / Carrat 2008 calibrated parameters).

Public API:
    - run_metapop_scenario(scenario, **opts) → SimResult — Stage 5 wrap
    - list_scenarios() → list[str] — registered scenarios (6 baseline + 8 extended)
    - epimas_health_check() → dict — quick sanity: sim importable, scenarios loaded
    - re-exports: MetapopParams, MetapopSEIRVD, SimResult, InterventionSpec,
                  COMPARTMENTS, SCENARIO_REGISTRY (from simulation.sim)

Performance: re-export only (~50ms cold import); run_scenario inherits from sim
    (~1-3s for 120-day baseline, ~3-8s for behaviour-coupled extended).
Side effects: none (pure re-export); run_scenario triggers numpy RK4 stepper.
Caller responsibility:
    - scenario name must be in ``list_scenarios()`` (KeyError otherwise).
    - Custom interventions: pass list[InterventionSpec] via run_scenario opts.

References:
    - Plan §C8 (EpiMAS = thin facade)
    - Bracher 2021 PLOS CB (WIS calibration baseline)
    - Carrat 2008 AJE 167:775 (γ = 1/3.0 d, G-175 reference Day-1)
    - User mandate 2026-05-11: "EpiMAS 새로 만들지 마"
"""
from __future__ import annotations

from typing import Any, Optional

# Re-export (D-4 small interface): import-time = sim public API only,
# no torch / duckdb dependency drag-in.
from simulation.sim import (
    COMPARTMENTS,
    DEFAULT_FLU_PARAMS,
    InterventionSpec,
    MetapopParams,
    MetapopSEIRVD,
    SCENARIO_REGISTRY,
    SimResult,
    register_scenario,
    run_scenario,
    select_disease_params,
)


def list_scenarios() -> list[str]:
    """Return registered scenario names (6 baseline + 8 extended).

    Returns:
        Sorted list of canonical scenario names — input to ``run_metapop_scenario``.

    Performance: O(1) registry lookup.
    """
    return sorted(SCENARIO_REGISTRY.keys())


def run_metapop_scenario(
    scenario: str,
    *,
    horizon_days: int = 120,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> SimResult:
    """Run a registered metapop SEIR-V-D scenario (Stage 5 wrap).

    Args:
        scenario: name from ``list_scenarios()``.
        horizon_days: simulation horizon (default 120 = ~17 weeks).
        seed: RNG seed (None = stochastic init perturbation).
        **kwargs: forwarded to ``simulation.sim.run_scenario`` — see
            ``MetapopParams`` for available knobs (β, γ, ν, M, init_state).

    Returns:
        ``SimResult`` with state trajectory shape (T, G=25, 6) and epi-validity
        gate report (Rt bounds, S+E+I+R+V+D=N conservation, seasonal phase).

    Raises:
        KeyError: scenario name not in registry → call ``list_scenarios()`` first.
        ValueError: invalid horizon_days (≤0) or kwargs.

    Performance:
        ~1-3s for baseline (RK4, dt=1d, G=25, T=120).
        ~3-8s for behaviour-coupled extended (4-param ABM hooks).

    Side effects:
        Pure compute; no DB write, no file write. Result is in-memory.

    Caller responsibility:
        - For audit trail, persist ``result.gate_report`` separately.
        - Compartment conservation tolerance is 1e-9 — gate fails if exceeded.
    """
    if scenario not in SCENARIO_REGISTRY:
        raise KeyError(
            f"Unknown scenario: {scenario!r}. "
            f"Available ({len(SCENARIO_REGISTRY)}): {list_scenarios()}"
        )
    # run_scenario(name, params=None, *, overrides) — horizon/seed/param knobs
    # are applied via `overrides` (keys matching MetapopParams fields are set,
    # others ignored). horizon_days maps to the `days` field.
    overrides: dict[str, Any] = dict(kwargs)
    if horizon_days is not None:
        overrides.setdefault("days", horizon_days)
    if seed is not None:
        overrides.setdefault("seed", seed)
    return run_scenario(scenario, overrides=overrides or None)


def epimas_health_check() -> dict[str, Any]:
    """Quick self-check that EpiMAS facade is wired correctly.

    Used by:
        - ``simulation/scripts/preflight_check.sh`` (optional [9] gate)
        - ARIA chain's ``simulation_layer_check`` advisor question

    Returns:
        ``{"ok": bool, "n_scenarios": int, "compartments": tuple,
           "params_default_R0": float, "errors": list[str]}``

    Performance: O(1) registry size + 1 default-param construction.
    Side effects: none.
    """
    errors: list[str] = []
    try:
        scenarios = list_scenarios()
    except (ImportError, KeyError) as e:
        errors.append(f"list_scenarios: {type(e).__name__}: {e}")
        scenarios = []

    try:
        # Parameterizable base: honour MPH_DISEASE if set, else byte-identical
        # DEFAULT_FLU_PARAMS (default flu path unchanged). select_disease_params
        # returns the same object for the flu default → R0 reported below is
        # bit-for-bit identical to the previous DEFAULT_FLU_PARAMS code path.
        params = select_disease_params()
        # Codex Day-1 G-175 anchor: γ = 1/3.0 d (Carrat 2008)
        gamma = float(getattr(params, "gamma", 0.0))
        beta_global = float(getattr(params, "beta", 0.0))
        r0 = beta_global / gamma if gamma > 0 else float("nan")
    except (AttributeError, ValueError) as e:
        errors.append(f"params: {type(e).__name__}: {e}")
        r0 = float("nan")

    return {
        "ok": not errors and len(scenarios) >= 6,
        "n_scenarios": len(scenarios),
        "compartments": COMPARTMENTS,
        "params_default_R0": r0,
        "errors": errors,
    }


__all__ = [
    # Public API (3)
    "list_scenarios",
    "run_metapop_scenario",
    "epimas_health_check",
    # Re-exports (sim public surface)
    "COMPARTMENTS",
    "DEFAULT_FLU_PARAMS",
    "InterventionSpec",
    "MetapopParams",
    "MetapopSEIRVD",
    "SCENARIO_REGISTRY",
    "SimResult",
    "register_scenario",
    "run_scenario",
    "select_disease_params",
]
