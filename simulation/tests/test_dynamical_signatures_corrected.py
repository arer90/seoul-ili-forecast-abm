"""Corrected hysteresis metric (Gemini C1/C2 fix): branch-area + permutation null.

The prior shoelace + phase-randomization version (a) MISSED genuine smooth lagged
hysteresis (lagged sinusoid p≈0.46) and fired on waveform sharpness (C1), and (b)
invented loop area for an open single-valued ramp y=x² → 0.166 (C2). These tests
pin the corrected behavior so a regression is caught.
"""
import numpy as np

from simulation.abm.dynamical_signatures import (
    _branch_area, _cycle_bounds, hysteresis_loop_area,
)


def test_branch_area_zero_for_memoryless_single_valued():
    """C2: an open single-valued curve (triangle driver, y=driver²) → ~0, not a
    spurious closure area."""
    drv = np.concatenate([np.linspace(0, 1, 40), np.linspace(1, 0, 40)])
    rsp = drv ** 2                                   # memoryless: y is a function of x
    x = (drv - drv.mean()) / drv.std()
    y = (rsp - rsp.mean()) / rsp.std()
    assert abs(_branch_area(x, y)) < 0.1             # rising≈falling → ~0


def test_lagged_sinusoid_detected_as_hysteresis():
    """C1: a genuine smooth lagged loop is DETECTED (the old phase-randomization
    null missed it at p≈0.46)."""
    t = np.linspace(0, 2 * np.pi, 80)
    r = hysteresis_loop_area(np.sin(t), np.sin(t - 0.8), n_null=2000)
    assert r["null_p"] < 0.05
    assert r["significant"] is True


def test_memoryless_curve_not_significant():
    """C1/C2: a memoryless single-valued response is NOT flagged as hysteresis."""
    drv = np.concatenate([np.linspace(0, 1, 40), np.linspace(1, 0, 40)])
    r = hysteresis_loop_area(drv, drv ** 2, n_null=2000)
    assert r["null_p"] >= 0.05                       # no path-dependence


def test_hard_step_not_spuriously_significant():
    """C1: a memoryless HARD step (sharp waveform) must NOT pass — the old null
    fired on sharpness alone."""
    t = np.linspace(0, 2 * np.pi, 80)
    # single-valued step response of the driver (no lag, no memory)
    r = hysteresis_loop_area(np.sin(t), np.sign(np.sin(t)) * 0.9, n_null=2000)
    # a single-valued (even if sharp) response has rising≈falling → not a loop
    assert r["null_p"] >= 0.05 or abs(r["loop_area"]) < 0.5


def test_keys_preserved_for_callers():
    """Back-compat: sim_vs_observed / multiproxy read these keys."""
    t = np.linspace(0, 2 * np.pi, 60)
    r = hysteresis_loop_area(np.sin(t), np.sin(t - 0.6), n_null=500)
    for k in ("loop_area", "abs_area", "null_p", "n", "circulation",
              "significant", "verdict"):
        assert k in r
    assert r["method"] == "branch_perm"


def test_multicycle_segmentation_and_sum():
    """Multi-cycle: 3 lagged cycles are segmented and the consistent circulation
    reinforces (still significant, n_cycles ≥ 2)."""
    t = np.linspace(0, 6 * np.pi, 180)
    r = hysteresis_loop_area(np.sin(t), np.sin(t - 0.8), n_null=1500)
    assert r["n_cycles"] >= 2
    assert r["null_p"] < 0.05


def test_cycle_bounds_single_when_no_trough():
    """A monotone driver resolves to a single cycle."""
    assert _cycle_bounds(np.linspace(0, 1, 40)) == [(0, 39)]
