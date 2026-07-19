"""P3 double-validation: does the ABM REPRODUCE the observed behavioral signature?

The observational decomposition is confounded (``behavioral_proof.confounding_check``)
and the observed standalone hysteresis is too weak to resolve from a single COVID
wave (``dynamical_signatures`` applied to the panel). So the proof is moved into
the MODEL: the ABM is calibrated to ILI/prevalence and is given **no mobility and
no vaccination input**. If its behavioral state (β_scale = contact reduction) traces
the SAME path-dependent hysteresis loop vs prevalence that the (confounded) observed
mobility hints at — while the behavior-OFF baseline traces NO loop — then the
mechanism, not a confounder, is sufficient to generate the signature.

This is an out-of-sample test: β_scale is never fit to mobility, so a matching loop
is a genuine prediction (PROOF_VALIDATION_PROTOCOL §1.8). Never raises in the
analysis functions; the heavy run wrapper surfaces load/sim errors loudly.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .dynamical_signatures import hysteresis_loop_area

# the post-COVID fatigue regime (matches run_rebound_scenario defaults) and the
# exact behavior-OFF limit (α=κ=0, τ→∞ ⇒ no risk coupling, no fatigue)
_DEFAULT_ON = dict(alpha=1.5, kappa=0.8, tau=60.0, theta=0.15)
_DEFAULT_OFF = dict(alpha=0.0, kappa=0.0, tau=float("inf"))

_METAPOP_CACHE: dict = {}


def load_seoul_metapop(days: int = 180, seed_infected: float = 1000.0):
    """Real 25-gu Seoul MetapopParams (populations + commuter mobility + disease),
    cached (the DB load is ~60 s; the sim itself is ~0.2 s). Returns a fresh
    ``MetapopParams`` with the requested horizon. Side effects: reads the DB once,
    memoises the base load. Raises if the metapop cannot be loaded (loud — a proof
    run must not silently fabricate a population)."""
    from simulation.abm.behavioural import MetapopParams
    from simulation.sim.io import load_metapop_params
    if "base" not in _METAPOP_CACHE:
        _METAPOP_CACHE["base"] = load_metapop_params()
    mp = _METAPOP_CACHE["base"]
    G = int(mp.populations.size)
    return MetapopParams(
        disease=mp.disease, populations=mp.populations, mobility=mp.mobility,
        district_names=mp.district_names,
        initial_infected=np.full(G, float(seed_infected)),
        days=int(days), dt=mp.dt, seed=mp.seed,
    )


def simulate_response(metapop, behaviour_kwargs: dict) -> dict:
    """Run the coupled ABM once; return the prevalence driver and behavioral
    response trajectories.

    Returns ``{days, prevalence, response, compliance, off}`` where
    ``prevalence`` = city-wide I(t), ``response`` = mean β_scale(t) (1.0 = no
    reduction, <1 = behavioral contact reduction — the model's mobility analog),
    and ``off`` flags the behavior-off limit. Raises on invalid params (loud)."""
    from simulation.abm.behavioural import BehaviouralParams, run_coupled_abm
    bp = BehaviouralParams(**behaviour_kwargs)
    res = run_coupled_abm(metapop, bp)
    return {
        "days": np.asarray(res.days, dtype=float),
        "prevalence": res.city_I().astype(float),
        "response": res.mean_beta_scale().astype(float),
        "compliance": res.mean_compliance().astype(float),
        "off": bp.is_behaviour_off(),
    }


def simulated_hysteresis(metapop, *, behaviour_on: Optional[dict] = None,
                         behaviour_off: Optional[dict] = None,
                         n_null: int = 2000, seed: int = 42) -> dict:
    """Core P3 result: the behavior-ON run must trace a hysteresis loop (the
    mechanism's path-dependence) and the behavior-OFF run must NOT.

    Returns ``{on, off, mechanism_produces_loop, verdict}`` where ``on``/``off``
    are :func:`dynamical_signatures.hysteresis_loop_area` dicts on (prevalence,
    response). ``mechanism_produces_loop`` is True iff ON is a significant loop
    AND its area exceeds OFF's — i.e. the loop is attributable to the behavioral
    mechanism, not the epidemic shape. Never raises (returns ``{"error": …}``).

    Performance: 2 sims (~0.4 s) + 2·n_null shoelace evals. Side effects: none
    beyond the sims. Caller responsibility: ``metapop`` is a valid MetapopParams.
    """
    try:
        on = simulate_response(metapop, behaviour_on or _DEFAULT_ON)
        off = simulate_response(metapop, behaviour_off or _DEFAULT_OFF)
    except Exception as exc:  # loud-ish: a proof run reports WHY it failed
        return {"error": f"{type(exc).__name__}: {exc}"}
    h_on = hysteresis_loop_area(on["prevalence"], on["response"], n_null=n_null, seed=seed)
    h_off = hysteresis_loop_area(off["prevalence"], off["response"], n_null=n_null, seed=seed)
    on_sig = bool(h_on.get("significant"))
    on_area = abs(h_on.get("loop_area", 0.0))
    off_area = abs(h_off.get("loop_area", 0.0))
    produces = on_sig and on_area > max(off_area * 1.5, 0.02)
    verdict = (
        f"behavior-ON loop area={h_on.get('loop_area', float('nan')):+.3f} "
        f"(p={h_on.get('null_p', float('nan')):.3f}), behavior-OFF area="
        f"{h_off.get('loop_area', float('nan')):+.3f}. "
        + ("⇒ the behavioral MECHANISM produces the hysteresis (absent in OFF) — "
           "path-dependence is generated by risk/fatigue memory, not the epidemic "
           "shape; β_scale was never fit to mobility (out-of-sample)."
           if produces else
           "⇒ mechanism does NOT produce a loop distinct from baseline at these "
           "params (try a multi-wave horizon or the calibrated regime).")
    )
    return {"on": h_on, "off": h_off, "on_compliance_range":
            [round(float(on["compliance"].min()), 3), round(float(on["compliance"].max()), 3)],
            "mechanism_produces_loop": bool(produces), "verdict": verdict}


def direction_consistent_with_observed(sim_loop_area: float,
                                       observed_loop_area: float) -> dict:
    """Weak corroboration: do the simulated and observed loops circulate the SAME
    way? The observed loop is confounded and non-significant, so this is a
    consistency check (same sign), NOT a significance claim. Returns
    ``{same_direction, verdict}``. Never raises."""
    same = (sim_loop_area > 0) == (observed_loop_area > 0)
    return {
        "same_direction": bool(same),
        "sim_area": round(float(sim_loop_area), 4),
        "observed_area": round(float(observed_loop_area), 4),
        "verdict": (
            f"sim {sim_loop_area:+.3f} vs observed {observed_loop_area:+.3f} — "
            + ("same circulation direction (model consistent with the confounded "
               "observed hint, not a standalone claim)"
               if same else "opposite direction (model does not match observed)")
        ),
    }
