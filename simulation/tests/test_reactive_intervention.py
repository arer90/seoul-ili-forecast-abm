"""B-P5 (M7 SCI-grade): reactive interventions are genuinely state-triggered.

The old `reactive_intervention` scenario hardcoded ``start_day=28`` while its
docstring claimed a WHO-MEM ILI-threshold trigger ("구현: stepper.py 의 trigger
logic 필요... 완전 구현은 Package E"). Now an InterventionSpec can carry a
ReactiveTrigger that fires on the FIRST day a city-wide state metric crosses a
threshold, and the activation day shifts with epidemic timing (not a fixed day).
"""
from dataclasses import replace

import numpy as np


# ── unit: resolver ─────────────────────────────────────────────────────────
def test_resolver_fires_on_crossing_and_holds_for_duration():
    from simulation.sim.interventions import resolve_active_interventions
    from simulation.sim.parameters import InterventionSpec, ReactiveTrigger

    spec = InterventionSpec("beta", 0.65, 0, 0,
                            trigger=ReactiveTrigger("prevalence", 0.01, duration_days=10))
    fired: dict[int, int] = {}

    # below threshold → not active, not fired
    assert resolve_active_interventions([spec], 3, {"prevalence": 0.005}, fired) == []
    assert fired == {}

    # crosses at day 5 → active, with a concrete [5, 15) window recorded
    out = resolve_active_interventions([spec], 5, {"prevalence": 0.02}, fired)
    assert len(out) == 1 and out[0].start_day == 5 and out[0].end_day == 15
    assert fired == {0: 5}

    # stays active through the duration even if the metric later drops
    assert len(resolve_active_interventions([spec], 9, {"prevalence": 0.0}, fired)) == 1

    # expires after the duration window
    assert resolve_active_interventions([spec], 15, {"prevalence": 0.02}, fired) == []


def test_resolver_passes_static_specs_through_untouched():
    from simulation.sim.interventions import resolve_active_interventions
    from simulation.sim.parameters import InterventionSpec

    static = InterventionSpec("beta", 0.6, 20, 40)  # no trigger
    out = resolve_active_interventions([static], 5, {}, {})
    assert out == [static]  # apply_interventions still handles its covers()


# ── integration: full sim ──────────────────────────────────────────────────
def _run_reactive(seed_scale: float, days: int = 160):
    from simulation.sim.metapop_seirvd import MetapopSEIRVD
    from simulation.sim.parameters import InterventionSpec, ReactiveTrigger
    from simulation.sim.scenarios import _default_params

    p = _default_params()
    p = replace(p, days=days, initial_infected=p.initial_infected * seed_scale)
    npi = InterventionSpec("beta", 0.65, 0, 0,
                           trigger=ReactiveTrigger("prevalence", 0.005, duration_days=56))
    return MetapopSEIRVD(p).run(interventions=[npi], run_validator=False)


def test_reactive_npi_activates_and_is_recorded():
    res = _run_reactive(1.0)
    assert res.reactive_activations, "reactive NPI never fired within the run"
    fired_day = res.reactive_activations[0]
    assert 0 < fired_day < 160


def test_trigger_day_is_state_dependent_not_fixed():
    """A faster-seeded epidemic crosses the prevalence threshold earlier."""
    fast = _run_reactive(8.0).reactive_activations.get(0)
    slow = _run_reactive(1.0).reactive_activations.get(0)
    assert fast is not None and slow is not None
    assert fast < slow, f"trigger must move with epidemic timing (fast={fast}, slow={slow})"


def test_scenario_uses_trigger_not_hardcoded_day():
    """The shipped reactive scenario carries a trigger, not a fixed start_day."""
    from simulation.sim.scenarios_extended import _scn_reactive_intervention

    _p, specs = _scn_reactive_intervention(None)
    assert len(specs) == 1
    assert specs[0].trigger is not None, "scenario reverted to a fixed-day NPI"
    assert specs[0].trigger.metric == "prevalence"
