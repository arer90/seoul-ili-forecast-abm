"""P2: dynamical-signature proof — hysteresis loop / spectral / lag.

Guards that a genuine loop (memory/fatigue) is detected and a memoryless
single-valued response is NOT flagged (the topological signature that
distinguishes the ABM's path-dependent mechanism from a confounder).
"""
import numpy as np

from simulation.abm.dynamical_signatures import (
    hysteresis_loop_area,
    lag_cross_correlation,
    spectral_peak,
)


def test_hysteresis_detects_a_real_loop():
    # an ellipse traversed once = maximal consistent circulation = hysteresis
    t = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    driver = np.cos(t)
    response = np.sin(t)  # 90° out of phase → fat loop
    r = hysteresis_loop_area(driver, response, n_null=1000)
    assert r["significant"] is True, r["verdict"]
    assert r["abs_area"] > 0.2
    assert r["null_p"] < 0.05


def test_memoryless_single_valued_is_not_flagged():
    # driver up then down; response a fixed function of driver → path retraces,
    # no loop. A memoryless confounder must NOT register as hysteresis.
    up = np.linspace(0, 1, 20)
    driver = np.concatenate([up, up[::-1]])
    response = driver ** 2  # single-valued in driver
    r = hysteresis_loop_area(driver, response, n_null=1000)
    assert r["abs_area"] < 0.05
    assert r["significant"] is False, r["verdict"]


def test_spectral_finds_nonannual_oscillation():
    # period = 2 years sampled monthly
    t = np.arange(48)
    s = np.sin(2 * np.pi * t / 24.0)  # 24 months = 2 yr
    r = spectral_peak(s, samples_per_year=12.0)
    assert r["is_annual"] is False
    assert abs(r["period_years"] - 2.0) < 0.3


def test_spectral_finds_annual():
    t = np.arange(48)
    s = np.sin(2 * np.pi * t / 12.0)  # 12 months = 1 yr
    r = spectral_peak(s, samples_per_year=12.0)
    assert r["is_annual"] is True


def test_lag_detects_delay():
    rng = np.random.default_rng(0)
    base = rng.standard_normal(40)
    driver = base.copy()
    response = np.roll(base, 2)  # response lags driver by 2
    r = lag_cross_correlation(driver, response, max_lag=5)
    assert r["best_lag"] == 2


def test_too_few_points_returns_error():
    assert "error" in hysteresis_loop_area([1, 2, 3], [1, 2, 3])
    assert "error" in spectral_peak([1, 2, 3])
