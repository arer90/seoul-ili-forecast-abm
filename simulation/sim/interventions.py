"""
simulation.sim.interventions
============================
Resolve ``InterventionSpec`` declarations into per-day parameter views.

The metapop simulator calls ``apply_interventions(...)`` once per step
to pick up the currently-effective disease + vaccination parameters.
Interventions are applied in list order, so when two specs target the
same parameter on overlapping windows the *later* one wins.

Design invariants
-----------------
- ``apply_interventions`` never mutates the input ``DiseaseParams``
  (dataclass is frozen). It returns a new ``DiseaseParams`` with
  possibly-modified fields.
- Per-district vaccination rates are always returned as a length-G
  array, even when the base is a scalar. This removes branching from
  the stepper's hot path.
- No scenarios are defined here — this module only interprets specs.
  See ``simulation.sim.scenarios`` for the registry.
"""
from __future__ import annotations

from dataclasses import replace
from dataclasses import replace
from typing import Iterable

import numpy as np

from .parameters import DiseaseParams, InterventionSpec


__all__ = [
    "apply_interventions",
    "expand_to_district_vector",
    "resolve_active_interventions",
]


_DISEASE_FIELDS: frozenset[str] = frozenset({
    "R0", "beta", "gamma", "sigma", "omega", "VE", "V_waning",
    "ifr", "report_frac",
})


def expand_to_district_vector(x: float | np.ndarray, G: int) -> np.ndarray:
    """Broadcast a scalar or (G,)-array to a length-G float vector."""
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 1:
        return np.full(G, float(arr.item()))
    if arr.size == G:
        return arr.astype(float, copy=True)
    raise ValueError(f"cannot broadcast shape {arr.shape} to ({G},)")


def resolve_active_interventions(
    interventions: Iterable[InterventionSpec],
    day: int,
    metric_values: dict[str, float],
    fired: dict[int, int],
) -> list[InterventionSpec]:
    """Resolve static + reactive interventions to concrete windows for ``day``.

    Static specs (``spec.trigger is None``) pass through unchanged —
    :func:`apply_interventions` handles their :meth:`InterventionSpec.covers`.
    A *reactive* spec (``spec.trigger`` set) activates on the FIRST day its
    trigger metric crosses the threshold (recorded in ``fired``, mutated in
    place) and is then emitted as a concrete ``[fired_day, fired_day +
    duration_days)`` window. This replaces the old fixed-calendar fake-reactive
    scenario (B-P5): the NPI now self-adjusts to epidemic timing.

    Args:
        interventions: full spec list (static and/or reactive).
        day: current absolute day offset (0-indexed).
        metric_values: city-wide metric lookup for this day, e.g.
            ``{"prevalence": 0.012, "incidence": 0.004}`` — computed by the
            caller from the live compartment state.
        fired: spec-index → activation-day. Persists across calls within a run
            (mutated in place); pass the SAME dict every day.

    Returns:
        Specs effective on ``day`` (static + triggered-and-active), each with a
        concrete window, ready for :func:`apply_interventions`.

    Side effects: mutates ``fired`` when a reactive spec first crosses.
    Caller responsibility: ``metric_values`` must contain the key named by
        every reactive spec's ``trigger.metric`` (else KeyError).
    """
    specs = list(interventions)
    out: list[InterventionSpec] = []
    for i, spec in enumerate(specs):
        trig = spec.trigger
        if trig is None:
            out.append(spec)
            continue
        if i not in fired and trig.fires(metric_values[trig.metric]):
            fired[i] = day
        d0 = fired.get(i)
        if d0 is not None and d0 <= day < d0 + trig.duration_days:
            out.append(replace(spec, start_day=d0, end_day=d0 + trig.duration_days))
    return out


def _apply_one(value: float | np.ndarray, spec: InterventionSpec) -> float | np.ndarray:
    """Apply a single op — ``scale``/``set``/``add`` — respecting the
    vectorial/scalar type of ``value``."""
    if spec.op == "scale":
        return value * spec.value
    if spec.op == "set":
        return spec.value if np.isscalar(value) else np.full_like(value, spec.value)
    if spec.op == "add":
        return value + spec.value
    raise ValueError(f"unknown op {spec.op!r}; expected scale|set|add")


def apply_interventions(
    day: int,
    base_disease: DiseaseParams,
    base_vaccination_rate: float | np.ndarray,
    interventions: Iterable[InterventionSpec],
    G: int,
) -> tuple[DiseaseParams, np.ndarray]:
    """Return the disease + vaccination-rate view effective on ``day``.

    Parameters
    ----------
    day : current absolute day offset (0-indexed)
    base_disease : baseline ``DiseaseParams`` for the run
    base_vaccination_rate : baseline vaccination rate (scalar or (G,))
    interventions : list of ``InterventionSpec`` — later entries override
        earlier ones on the same parameter
    G : number of districts (for broadcasting)

    Returns
    -------
    disease : possibly-modified ``DiseaseParams``
    vax_rate : (G,) vaccination-rate vector
    """
    # Start from mutable views
    disease_fields: dict[str, float] = {
        "R0": base_disease.R0,
        "gamma": base_disease.gamma,
        "sigma": base_disease.sigma,
        "omega": base_disease.omega,
        "VE": base_disease.VE,
        "V_waning": base_disease.V_waning,
        "ifr": base_disease.ifr,
        "report_frac": base_disease.report_frac,
    }
    vax_vec = expand_to_district_vector(base_vaccination_rate, G)

    for spec in interventions:
        if not spec.covers(day):
            continue

        if spec.parameter == "vaccination_rate":
            if spec.targets is None:
                vax_vec = _apply_one(vax_vec, spec)
            else:
                tgt = np.asarray(spec.targets, dtype=int)
                vax_vec[tgt] = _apply_one(vax_vec[tgt], spec)
            continue

        # β is a *derived* quantity (R0 · γ). Interventions on "beta" are
        # modelled as interventions on R0 for stability, so downstream
        # code can always recompute β = R0 · γ without a staleness bug.
        if spec.parameter == "beta":
            new_beta = _apply_one(base_disease.beta, spec)
            gamma_cur = disease_fields["gamma"]
            disease_fields["R0"] = float(new_beta / gamma_cur) if gamma_cur > 0 else disease_fields["R0"]
            continue

        if spec.parameter not in _DISEASE_FIELDS:
            raise ValueError(
                f"intervention targets unknown parameter {spec.parameter!r}; "
                f"expected one of: vaccination_rate, beta, {sorted(_DISEASE_FIELDS)}"
            )

        current = disease_fields[spec.parameter]
        disease_fields[spec.parameter] = float(_apply_one(current, spec))

    disease = replace(base_disease, **disease_fields)
    return disease, vax_vec
