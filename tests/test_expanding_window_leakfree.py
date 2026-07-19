"""Truncation-invariance (leak-free) guard for the expanding-window protocol.

``run_expanding_multihorizon`` builds the feature matrix ONCE on the full series
and then slices ``X_all[:k]`` at each expanding origin (one row per observed
week). That build-once-then-slice is leak-free *iff* every causal feature is
truncation-invariant: building on the full series and slicing ``[:k]`` must equal
building on the prefix ``[:k]`` — i.e. no future week may influence any in-sample
row. polars ``rolling_*(window_size=w)`` is a TRAILING (right-closed) window and
the transforms add a ``.shift(1)``, so the property holds to machine zero.

This test was added after an adversarial correctness review (wf wtnqfhsgp) where a
reviewer wrongly assumed polars rolling was centered. It locks the property in so a
future switch to a centered/look-ahead window (a real leak) fails loudly here
instead of silently inflating forecast skill.

Run (per-file, macOS): .venv/bin/python -m pytest tests/test_expanding_window_leakfree.py -q
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from simulation.models.feature_engine.transforms import (
    _add_diff_features,
    _add_lag_features,
    _add_rolling_features,
)

_N = 355  # ~ the live Seoul ILI series length
_COL = "ili_rate"


def _make_df(n: int = _N) -> pl.DataFrame:
    """Deterministic non-trivial series (random walk, no repeats/flat spans)."""
    rng = np.random.default_rng(42)
    vals = np.cumsum(rng.standard_normal(n)) + 50.0
    return pl.DataFrame({_COL: vals})


def _assert_prefix_equals_slice(full: pl.DataFrame, prefix: pl.DataFrame, k: int, label: str) -> None:
    """Every feature column: full[:k] must equal prefix where both are non-null.

    A nonzero difference (or a null/non-null mismatch) means a future week leaked
    into an in-sample row — the exact failure the expanding window must avoid.
    """
    feat_cols = [c for c in full.columns if c != _COL]
    assert feat_cols, "no feature columns produced"
    for c in feat_cols:
        a = full[c][:k].to_numpy()
        b = prefix[c].to_numpy()
        a_nan, b_nan = np.isnan(a), np.isnan(b)
        # Null pattern must match (a future-dependent feature would become
        # non-null earlier on the full series than on the prefix).
        assert np.array_equal(a_nan, b_nan), f"{label}:{c} null-pattern differs at k={k}"
        both = ~a_nan
        assert both.sum() > 0, f"{label}:{c} produced no comparable values at k={k}"
        assert np.max(np.abs(a[both] - b[both])) == 0.0, f"{label}:{c} leaks future at k={k}"


@pytest.mark.parametrize("k", [40, 100, 200, 301, 340])
def test_rolling_features_truncation_invariant(k: int) -> None:
    df = _make_df()
    full = _add_rolling_features(df, _COL, [4, 8, 13, 26])
    prefix = _add_rolling_features(df[:k], _COL, [4, 8, 13, 26])
    _assert_prefix_equals_slice(full, prefix, k, "rolling")


@pytest.mark.parametrize("k", [40, 100, 200, 301, 340])
def test_lag_features_truncation_invariant(k: int) -> None:
    df = _make_df()
    full = _add_lag_features(df, _COL, [1, 2, 4, 8])
    prefix = _add_lag_features(df[:k], _COL, [1, 2, 4, 8])
    _assert_prefix_equals_slice(full, prefix, k, "lag")


@pytest.mark.parametrize("k", [40, 100, 200, 301, 340])
def test_diff_features_truncation_invariant(k: int) -> None:
    df = _make_df()
    full = _add_diff_features(df, _COL, [1, 2])
    prefix = _add_diff_features(df[:k], _COL, [1, 2])
    _assert_prefix_equals_slice(full, prefix, k, "diff")


def test_centered_window_would_be_caught() -> None:
    """Sanity: a CENTERED rolling window (a real leak) must fail the guard.

    Proves the truncation-invariance check has teeth — if someone swaps the
    trailing window for a centered one, the assertion fires.
    """
    df = _make_df()

    def _centered(d: pl.DataFrame) -> pl.DataFrame:
        # center=True makes row t depend on future rows -> NOT truncation-invariant
        return d.with_columns(
            pl.col(_COL).rolling_mean(window_size=5, center=True).alias("leaky")
        )

    full = _centered(df)
    prefix = _centered(df[:200])
    with pytest.raises(AssertionError):
        _assert_prefix_equals_slice(full, prefix, 200, "centered")
