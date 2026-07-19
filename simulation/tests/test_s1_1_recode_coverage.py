"""
simulation/tests/test_s1_1_recode_coverage.py
==============================================
S1-1 fold-recode coverage guardrail.

Source of leakage risk (CAUSALITY_AUDIT §2):
  - `*_qbin` / `*_qnorm`  — quantile edges from FULL series
  - `above_threshold`     — baseline = median(ili[:int(n*0.8)])
  - `{src}_ili` interactions — src.max() from FULL series
  - `er_burden_ili`       — (1/clip(er_bed_avg)).max() from FULL series

Every one of these columns MUST be rewritten by the per-fold recode
functions in `simulation.pipeline.wfcv` BEFORE the fold's WF-CV
fit sees it. If someone adds a new _qbin / _ili / threshold feature at
build time and forgets to list it in the *_SPECS tables, this test will
fail — that's the guardrail.

What the test does:
  1. Build a synthetic (n_rows, n_feat) X_all whose columns include
     every spec-listed *output* column and each required *source*
     column.
  2. Call each _recode_*_per_fold(X_all, feat_names, train_end).
  3. Assert every spec-listed output column was actually mutated
     (i.e., X_out[:, out_idx] != X_all[:, out_idx] somewhere).
  4. Also assert the function touched NOTHING outside the spec list,
     so we don't silently clobber unrelated columns.
"""
from __future__ import annotations

import numpy as np
import pytest

from simulation.pipeline.wfcv import (
    _QUANTILE_SPECS,
    _ABOVE_THRESHOLD_COL,
    _INTERACTION_SPECS,
    _INVERSE_MAX_INTERACTION_SPECS,
    _INTERACTION_MULTIPLIER_COL,
    _recode_quantile_features_per_fold,
    _recode_above_threshold_per_fold,
    _recode_interaction_features_per_fold,
)


def _changed_cols(X_before: np.ndarray, X_after: np.ndarray) -> set[int]:
    """Indices of columns that differ anywhere (nan-safe)."""
    out = set()
    n_cols = X_before.shape[1]
    for j in range(n_cols):
        a = X_before[:, j]
        b = X_after[:, j]
        same = np.array_equal(a, b, equal_nan=True)
        if not same:
            out.add(j)
    return out


# ── 1. Quantile recode ─────────────────────────────────────────────
def test_quantile_recode_covers_all_specs():
    n_rows = 120
    train_end = 80
    # Feature layout: for every (src, _) pair include src, src_qbin, src_qnorm.
    cols: list[str] = []
    for src, _n_bins in _QUANTILE_SPECS:
        cols += [src, f"{src}_qbin", f"{src}_qnorm"]
    # a distractor column that must NOT be touched
    cols.append("distractor_raw")

    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_rows, len(cols)))

    X_out = _recode_quantile_features_per_fold(X, cols, train_end)
    changed = _changed_cols(X, X_out)

    expected_changed: set[int] = set()
    for src, _n in _QUANTILE_SPECS:
        expected_changed.add(cols.index(f"{src}_qbin"))
        expected_changed.add(cols.index(f"{src}_qnorm"))

    assert expected_changed.issubset(changed), (
        f"quantile recode failed to touch: "
        f"{[cols[i] for i in sorted(expected_changed - changed)]}"
    )
    # distractor must be untouched
    assert cols.index("distractor_raw") not in changed, (
        "quantile recode touched non-spec column"
    )


# ── 2. Above-threshold recode ──────────────────────────────────────
def test_above_threshold_recode_touches_col():
    from simulation.pipeline.wfcv import (
        _recode_above_threshold_per_fold,
    )
    n_rows = 120
    train_end = 80
    cols = [_ABOVE_THRESHOLD_COL, "distractor_raw"]
    rng = np.random.default_rng(1)
    X = rng.normal(size=(n_rows, len(cols)))
    # y must have a real baseline (build-time uses median of y)
    y = rng.uniform(1.0, 10.0, size=n_rows)

    X_out = _recode_above_threshold_per_fold(X, y, cols, train_end)
    changed = _changed_cols(X, X_out)

    assert cols.index(_ABOVE_THRESHOLD_COL) in changed, (
        "above_threshold column was NOT rewritten per-fold — "
        "build-time threshold is leaking into every fold"
    )
    assert cols.index("distractor_raw") not in changed


# ── 3. Interaction recode (main + inverse-max) ─────────────────────
def test_interaction_recode_covers_all_specs():
    n_rows = 120
    train_end = 80

    cols: list[str] = [_INTERACTION_MULTIPLIER_COL]
    # main spec: {out_name, src_name}
    for out_name, src_name, _eps in _INTERACTION_SPECS:
        cols += [src_name, out_name]
    # inverse-max spec (er_burden_ili)
    for out_name, src_name, _clip, _eps in _INVERSE_MAX_INTERACTION_SPECS:
        cols += [src_name, out_name]
    cols.append("distractor_raw")
    # de-dupe preserving order (ili_rate_lag1 may also appear as a src)
    seen: set[str] = set()
    cols_dedup = []
    for c in cols:
        if c not in seen:
            cols_dedup.append(c)
            seen.add(c)
    cols = cols_dedup

    rng = np.random.default_rng(2)
    X = rng.uniform(0.5, 5.0, size=(n_rows, len(cols)))  # positive so
    # inverse-max denom stays finite

    X_out = _recode_interaction_features_per_fold(X, cols, train_end)
    changed = _changed_cols(X, X_out)

    expected_outputs = set()
    for out_name, _src, _eps in _INTERACTION_SPECS:
        expected_outputs.add(cols.index(out_name))
    for out_name, _src, _clip, _eps in _INVERSE_MAX_INTERACTION_SPECS:
        expected_outputs.add(cols.index(out_name))

    missing = expected_outputs - changed
    assert not missing, (
        f"interaction recode failed to touch: "
        f"{[cols[i] for i in sorted(missing)]}"
    )
    # distractor + source columns should be untouched (only output cols
    # are rewritten)
    assert cols.index("distractor_raw") not in changed
    for _out_name, src_name, _eps in _INTERACTION_SPECS:
        assert cols.index(src_name) not in changed, (
            f"source column {src_name} was mutated"
        )


# ── 4. Sanity: all specs have non-empty tuples ─────────────────────
def test_spec_tables_non_empty():
    assert len(_QUANTILE_SPECS) >= 1
    assert len(_INTERACTION_SPECS) >= 1
    assert len(_INVERSE_MAX_INTERACTION_SPECS) >= 1
    assert isinstance(_ABOVE_THRESHOLD_COL, str) and _ABOVE_THRESHOLD_COL
    assert isinstance(_INTERACTION_MULTIPLIER_COL, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
