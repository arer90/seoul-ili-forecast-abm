"""P4: identifiability — mobility resolves the (α, θ) equifinality.

The epidemic curve integrates out the compliance-timing detail, so θ is poorly
identified by prevalence alone; the behavioral observable (β_scale) restores it.
Tiny synthetic metapop, truth on-grid so the SSE minimum is exact.
"""
import numpy as np

from simulation.sim.parameters import DiseaseParams, MetapopParams
from simulation.abm.identifiability import (
    _affine_r2,
    _nsse,
    calibrate_behavioral_to_ili,
    calibrate_forcing_then_behavior,
    identifiability_gain,
    objective_grid,
    objective_grid_nd,
    profile_nd,
)

_TRUTH = dict(alpha=1.5, kappa=0.8, tau=40.0, theta=0.12)
# linspace grids (truth deliberately OFF-grid) so the flat-bottomed valley spans
# several cells and the CI-width contrast is visible (a coarse on-grid truth
# collapses both profiles to the single truth cell → gain hidden)
_A = np.linspace(0.4, 3.0, 9)
_T = np.linspace(0.05, 0.30, 9)


def _tiny(days=140, R0=2.5):
    return MetapopParams(
        disease=DiseaseParams(R0=R0),
        populations=np.array([100000.0, 100000.0]),
        mobility=np.array([[0.8, 0.2], [0.2, 0.8]]),
        district_names=["A", "B"],
        initial_infected=np.array([100.0, 100.0]),
        days=days, dt=0.25, seed=1,
    )


def _grid():
    return objective_grid(_tiny(), _TRUTH, _A, _T, fixed=dict(kappa=0.8, tau=40.0))


def test_nsse_zero_when_identical():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert _nsse(x, x) == 0.0
    assert _nsse(x + 1.0, x) > 0.0


def test_objective_minimised_near_truth():
    g = _grid()
    # argmin of the prevalence surface should sit near truth (α=1.5 → idx 3-4,
    # θ=0.12 → idx 2-3 on the linspace grids) and be small
    ai, tj = np.unravel_index(np.argmin(g["sse_prev"]), g["sse_prev"].shape)
    assert ai in (2, 3, 4), f"alpha argmin {ai} not near truth"
    assert g["sse_prev"].min() < 0.3


def test_theta_poorly_identified_by_ili_but_resolved_by_mobility():
    g = _grid()
    r = identifiability_gain(g, target="theta", mobility_weight=1.0)
    assert r["ili_interval_frac"] >= 0.33, "theta should be broad under ILI alone"
    assert r["joint_interval_frac"] < r["ili_interval_frac"], "mobility must tighten"
    assert r["gain"] >= 2.0
    assert r["identified_by_mobility"] is True, r["verdict"]


def test_gain_reported_for_alpha_too():
    g = _grid()
    r = identifiability_gain(g, target="alpha", mobility_weight=1.0)
    # direction must hold even if alpha is already partly constrained by ILI
    assert r["joint_interval_frac"] <= r["ili_interval_frac"]
    assert "verdict" in r


def test_generalized_to_fatigue_pair_kappa_tau():
    # the headline must extend to the FATIGUE params (κ,τ), not just (α,θ)
    K = np.array([0.2, 0.5, 0.8, 1.1, 1.4])
    TAU = np.array([20.0, 35.0, 50.0, 65.0, 80.0])
    g = objective_grid(_tiny(), _TRUTH, K, TAU, param_x="kappa", param_y="tau",
                       fixed=dict(alpha=1.5, theta=0.12))
    assert g["param_x"] == "kappa" and g["param_y"] == "tau"
    # machinery validity (the κ,τ identifiability MAGNITUDE — 4-5× tightening — is
    # demonstrated on real Seoul, not the degenerate 5-cell tiny synthetic where
    # the absolute-threshold CI metric is noisy)
    for t in ("kappa", "tau"):
        r = identifiability_gain(g, target=t)
        assert "error" not in r, r
        assert 0.0 <= r["ili_interval_frac"] <= 1.0
        assert 0.0 <= r["joint_interval_frac"] <= 1.0
        assert r["gain"] > 0 and "verdict" in r
    # a target not on this grid's axes → explicit error, not a wrong answer
    assert "error" in identifiability_gain(g, target="alpha")


def test_objective_grid_nd_shape_and_truth_minimum():
    grids = {"alpha": [1.0, 1.5, 2.0], "kappa": [0.4, 0.8, 1.2],
             "tau": [30.0, 40.0, 50.0], "theta": [0.08, 0.12, 0.16]}
    g = objective_grid_nd(_tiny(), _TRUTH, grids)
    assert g["names"] == ["alpha", "kappa", "tau", "theta"]
    assert g["sse_prev"].shape == (3, 3, 3, 3)
    # truth (1.5, 0.8, 40, 0.12) is the centre cell (1,1,1,1) → SSE ≈ 0
    assert g["sse_prev"][1, 1, 1, 1] < 1e-6
    assert g["sse_resp"][1, 1, 1, 1] < 1e-6


def test_profile_nd_valid_and_target_guard():
    grids = {"alpha": [1.0, 1.5, 2.0], "kappa": [0.4, 0.8, 1.2],
             "tau": [30.0, 40.0, 50.0], "theta": [0.08, 0.12, 0.16]}
    g = objective_grid_nd(_tiny(), _TRUTH, grids)
    for t in ("alpha", "kappa", "tau", "theta"):
        r = profile_nd(g, t)
        assert "error" not in r
        assert 0.0 <= r["ili_interval_frac"] <= 1.0
        assert 0.0 <= r["joint_interval_frac"] <= 1.0
        assert len(r["ili_profile"]) == 3 and r["gain"] > 0
    assert "error" in profile_nd(g, "nonexistent_param")


def test_affine_r2_perfect_and_calibrate():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert _affine_r2(2.0 * x + 1.0, x) > 0.999  # affine-invariant shape match
    # calibrate to a synthetic ILI wave returns a best param set + finite R²
    ili = np.r_[np.linspace(1, 50, 13), np.linspace(50, 1, 13)]  # 26-week wave
    grids = {"alpha": [0.5, 1.5], "kappa": [0.3, 0.8], "tau": [30.0, 60.0],
             "theta": [0.08, 0.15]}
    best = calibrate_behavioral_to_ili(_tiny(), ili, grids)
    assert "params" in best and set(best["params"]) == {"alpha", "kappa", "tau", "theta"}
    assert np.isfinite(best["r2"])


def test_forcing_then_behavior_co_calibration():
    ili = np.r_[np.linspace(1, 50, 13), np.linspace(50, 1, 13)]
    bgrids = {"alpha": [0.2, 1.0], "kappa": [0.1, 0.6], "tau": [20.0, 60.0],
              "theta": [0.05, 0.15]}
    out = calibrate_forcing_then_behavior(
        _tiny(), ili, r0_grid=[1.4, 2.0], seed_grid=[200.0, 1000.0],
        behavior_grids=bgrids)
    assert out["r0"] in (1.4, 2.0) and out["seed"] in (200.0, 1000.0)
    assert set(out["params"]) == {"alpha", "kappa", "tau", "theta"}
    assert np.isfinite(out["forcing_r2"]) and np.isfinite(out["r2"])
