"""P3: sim-vs-observed double-validation — the behavioral MECHANISM must produce
the hysteresis loop that the static (behavior-OFF) baseline cannot.

Uses a tiny synthetic 2-district metapop (the real Seoul load is ~60 s); the
proof is structural: ON traces a significant loop, OFF is constant.
"""
import numpy as np

from simulation.sim.parameters import DiseaseParams, MetapopParams
from simulation.abm.sim_vs_observed import (
    direction_consistent_with_observed,
    simulate_response,
    simulated_hysteresis,
)

_ON = dict(alpha=1.5, kappa=0.8, tau=40.0, theta=0.12)
_OFF = dict(alpha=0.0, kappa=0.0, tau=float("inf"))


def _tiny(days=140, R0=2.5):
    return MetapopParams(
        disease=DiseaseParams(R0=R0),
        populations=np.array([100000.0, 100000.0]),
        mobility=np.array([[0.8, 0.2], [0.2, 0.8]]),
        district_names=["A", "B"],
        initial_infected=np.array([100.0, 100.0]),
        days=days, dt=0.25, seed=1,
    )


def test_behavior_off_response_is_constant():
    off = simulate_response(_tiny(), _OFF)
    assert off["off"] is True
    assert off["response"].std() < 1e-9, "behavior-off must not modulate beta"


def test_behavior_on_response_varies_with_epidemic():
    on = simulate_response(_tiny(), _ON)
    assert on["off"] is False
    assert on["response"].std() > 0.01, "behavior-on must modulate contact"
    assert on["response"].max() <= 1.0 + 1e-9  # beta_scale ≤ baseline


def test_mechanism_produces_loop_absent_in_baseline():
    r = simulated_hysteresis(_tiny(), behaviour_on=_ON, behaviour_off=_OFF, n_null=500)
    assert r["on"]["significant"] is True, r["verdict"]
    # OFF is constant → the signature function returns an error (no loop), which
    # is exactly the point: no behavioral response ⇒ no path-dependence.
    assert "loop_area" not in r["off"] or abs(r["off"]["loop_area"]) < 0.05
    assert r["mechanism_produces_loop"] is True


def test_direction_check_reports_sign_agreement():
    # model clockwise (−) vs a confounded observed (+) → not same direction; the
    # helper must report this honestly rather than claim a match.
    d = direction_consistent_with_observed(-0.30, +0.10)
    assert d["same_direction"] is False
    d2 = direction_consistent_with_observed(-0.30, -0.05)
    assert d2["same_direction"] is True


def test_invalid_params_returns_error_not_raise():
    r = simulated_hysteresis(_tiny(), behaviour_on=dict(alpha=-1.0), n_null=10)
    assert "error" in r
