"""A3 (M7): per-horizon PI coverage table (coverage decays with horizon)."""
import numpy as np

from simulation.analytics.horizon_coverage import per_horizon_coverage


def test_per_horizon_keys_and_nominal():
    preds = {1: np.array([10.0, 11.0, 12.0, 13.0]),
             2: np.array([10.0, 11.0, 12.0])}
    y = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    res = per_horizon_coverage(preds, y, alpha=0.05)
    assert set(res["per_horizon"]) == {1, 2}
    assert res["nominal"] == 0.95
    for h in (1, 2):
        c = res["per_horizon"][h]["coverage"]
        assert 0.0 <= c <= 1.0


def test_alignment_h2_uses_shifted_actuals():
    # h=2 preds[i] forecasts y[i+1]; perfect h=2 preds → high coverage
    preds = {2: np.array([11.0, 12.0, 13.0])}  # forecasts y[1],y[2],y[3]
    y = np.array([10.0, 11.0, 12.0, 13.0])
    res = per_horizon_coverage(preds, y, alpha=0.05)
    assert res["per_horizon"][2]["coverage"] == 1.0  # exact → covered


def test_oof_residuals_override_used():
    preds = {1: np.array([10.0, 10.0, 10.0, 10.0])}
    y = np.array([10.0, 10.0, 10.0, 10.0])
    # huge OOF residuals → wide PI → full coverage; quantile reflects them
    res = per_horizon_coverage(preds, y, alpha=0.05,
                               oof_residuals_by_h={1: np.array([100.0, -100.0, 50.0])})
    assert res["per_horizon"][1]["quantile"] >= 50.0
    assert res["per_horizon"][1]["coverage"] == 1.0


def test_degenerate_horizon_is_nan_not_crash():
    res = per_horizon_coverage({3: np.array([np.nan])}, np.array([1.0, 2.0]), alpha=0.05)
    assert np.isnan(res["per_horizon"][3]["coverage"])
