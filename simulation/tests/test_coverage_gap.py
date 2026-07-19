"""Phase C1: coverage_gap_by_regime smoke tests."""
from __future__ import annotations

import numpy as np
import pytest


def _synth_series(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.normal(loc=10.0, scale=2.0, size=n)
    return y


def test_coverage_gap_fallback_shapes():
    from simulation.analytics.diagnostics import coverage_gap_by_regime
    y = _synth_series(200)
    # symmetric PI at 90% nominal from known sigma
    lo = y - 1.645 * 2.0
    hi = y + 1.645 * 2.0
    rows = coverage_gap_by_regime(y, lo, hi, nominal=0.90)
    names = [r["regime"] for r in rows]
    assert names == ["pre_covid", "during_covid", "post_covid", "global"]
    total = sum(r["n"] for r in rows if r["regime"] != "global")
    assert total == 200
    glb = next(r for r in rows if r["regime"] == "global")
    # Full-coverage PI that trivially brackets every y -> coverage == 1.0
    y2 = _synth_series(200, seed=1)
    lo2 = np.full_like(y2, -1e9)
    hi2 = np.full_like(y2, 1e9)
    rows2 = coverage_gap_by_regime(y2, lo2, hi2, nominal=0.90)
    for r in rows2:
        assert r["coverage"] == 1.0
        assert r["gap"] == pytest.approx(0.10)


def test_coverage_gap_zero_coverage_row():
    from simulation.analytics.diagnostics import coverage_gap_by_regime
    y = _synth_series(150)
    # PI that never contains y -> coverage = 0
    lo = y + 100.0
    hi = y + 101.0
    rows = coverage_gap_by_regime(y, lo, hi, nominal=0.90)
    for r in rows:
        assert r["coverage"] == 0.0
        assert r["gap"] == pytest.approx(-0.90)
        assert r["mean_width"] == pytest.approx(1.0)


def test_coverage_gap_with_calendar_dates():
    """When dates span the COVID boundaries, pre/during/post masks
    should carve the series at 2020-03-01 and 2023-01-01."""
    from simulation.analytics.diagnostics import coverage_gap_by_regime
    # 300 weekly dates starting 2018-01-01 -> covers pre/during/post
    dates = np.array(
        [np.datetime64("2018-01-01") + np.timedelta64(7 * i, "D") for i in range(300)]
    )
    y = np.ones(300, dtype=float)
    lo = y - 1.0
    hi = y + 1.0
    rows = coverage_gap_by_regime(y, lo, hi, nominal=0.90, dates=dates)
    by = {r["regime"]: r for r in rows}
    # All three regimes should appear
    assert set(by.keys()) == {"pre_covid", "during_covid", "post_covid", "global"}
    # Partition sanity
    assert by["pre_covid"]["n"] + by["during_covid"]["n"] + by["post_covid"]["n"] == 300
    # Coverage is 100% everywhere because y=1 and PI=[0,2]
    for r in rows:
        assert r["coverage"] == 1.0


def test_coverage_gap_length_mismatch_raises():
    from simulation.analytics.diagnostics import coverage_gap_by_regime
    y = np.zeros(10)
    with pytest.raises(ValueError, match="length mismatch"):
        coverage_gap_by_regime(y, np.zeros(9), np.zeros(10), nominal=0.9)


def test_coverage_gap_empty_inputs_return_empty():
    from simulation.analytics.diagnostics import coverage_gap_by_regime
    rows = coverage_gap_by_regime(
        np.array([], float), np.array([], float), np.array([], float), nominal=0.9
    )
    assert rows == []


def test_coverage_gap_table_multi_level():
    """coverage_gap_table should stack across levels and regimes."""
    from simulation.analytics.diagnostics import coverage_gap_table
    y = _synth_series(100)
    lower_by = {0.80: y - 1.282 * 2.0, 0.95: y - 1.960 * 2.0}
    upper_by = {0.80: y + 1.282 * 2.0, 0.95: y + 1.960 * 2.0}
    rows = coverage_gap_table(y, lower_by, upper_by)
    # 2 levels × 4 regimes = 8 rows (global included per level)
    assert len(rows) == 8
    levels = sorted({r["level"] for r in rows})
    assert levels == [0.80, 0.95]
    for r in rows:
        assert 0.0 <= r["coverage"] <= 1.0
        assert r["mean_width"] > 0
