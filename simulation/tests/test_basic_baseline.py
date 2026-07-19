"""Guard for the basic-feature baseline (G-240, 2026-05-30).

phase 4 baseline now trains on a BASIC feature subset (lag + seasonal), NOT the full
feature set — a "beat a naive lag model" reference (user design, replacing the former
full-feature design). BASIC_FEATURE_COLS is the SSOT for that subset.

macOS: run PER-FILE (memory ``test-suite-execution``).
"""
from simulation.pipeline.baseline import BASIC_FEATURE_COLS


def test_basic_set_is_lag_plus_seasonal_only():
    s = set(BASIC_FEATURE_COLS)
    # lags (user: "lag나 기본적인 것들")
    assert {"ili_rate_lag1", "ili_rate_lag2", "ili_rate_lag4"} <= s
    # seasonal (week-of-year carried by sin/cos + Fourier; no raw integer week exists)
    assert {"sin_month", "cos_month", "season_idx"} <= s
    assert {"fourier_sin_h1", "fourier_cos_h1"} <= s


def test_basic_set_is_small_and_unique():
    # BASIC = small reference set, NOT the full ~262-feature matrix
    assert 5 <= len(BASIC_FEATURE_COLS) <= 20
    assert len(BASIC_FEATURE_COLS) == len(set(BASIC_FEATURE_COLS))


def test_basic_set_excludes_engineered_non_basic():
    # must NOT include interaction / quantile / above_threshold / rolling-heavy cols
    s = set(BASIC_FEATURE_COLS)
    for bad in ("above_threshold", "humid_ili", "school_ili", "age_mixing"):
        assert not any(bad in c for c in s), f"basic set leaked a non-basic feature: {bad}"
