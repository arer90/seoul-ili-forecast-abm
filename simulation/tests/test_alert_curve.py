"""A7 (M7): alert operating curve + matched-sensitivity threshold selection."""
import numpy as np

from simulation.analytics.alert_curve import (
    alert_operating_curve,
    threshold_at_sensitivity,
)


def test_curve_sweeps_thresholds_with_wilson_ci():
    y_true = np.array([5.0, 12.0, 8.0, 15.0, 6.0, 11.0])
    y_pred = np.array([6.0, 11.0, 9.0, 14.0, 5.0, 13.0])
    curve = alert_operating_curve(y_true, y_pred, [7.0, 10.0, 13.0])
    assert len(curve) == 3
    for r in curve:
        assert "f1" in r and "sensitivity" in r and len(r["sens_ci"]) == 2
        assert r["tp"] + r["fp"] + r["fn"] + r["tn"] == 6  # confusion sums to n


def test_perfect_separation_gives_f1_one():
    y_true = np.array([1.0, 2.0, 20.0, 21.0])
    y_pred = np.array([1.0, 2.0, 20.0, 21.0])
    curve = alert_operating_curve(y_true, y_pred, [10.0])
    assert curve[0]["f1"] == 1.0 and curve[0]["sensitivity"] == 1.0
    assert curve[0]["n_events"] == 2


def test_threshold_at_sensitivity_picks_most_specific():
    # low threshold → high sens; high threshold → low sens. Want highest thr with sens≥0.9
    y_true = np.array([1.0, 5.0, 9.0, 13.0, 17.0])
    y_pred = np.array([1.0, 5.0, 9.0, 13.0, 17.0])  # perfect
    curve = alert_operating_curve(y_true, y_pred, [2.0, 6.0, 10.0, 14.0])
    thr = threshold_at_sensitivity(curve, target_sens=0.9)
    assert thr is not None and thr == 14.0  # most specific still at 100% sens


def test_no_threshold_meets_target_returns_none():
    y_true = np.array([10.0, 11.0, 12.0])
    y_pred = np.array([1.0, 1.0, 1.0])  # never alerts → 0 sensitivity
    curve = alert_operating_curve(y_true, y_pred, [9.0])
    assert threshold_at_sensitivity(curve, target_sens=0.9) is None
