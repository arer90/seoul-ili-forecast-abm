"""G-309 (audit #4): phase-13 OOF/refit must recode global-summary features fold-locally.

quantile (*_qbin/*_qnorm), above_threshold, interaction features are coded ONCE at build time with
GLOBAL (test+real era) statistics. baseline/wfcv/real_eval recode them per-fold (train_end); phase-13
(per_model_optimize/_inline) did not → OOF selection + test/real refit used future-informed coding
(leakage). _recode_advanced_per_fold applies the 3 wfcv recoders; no-op if columns absent.

macOS: run PER-FILE.
"""
import numpy as np

from simulation.pipeline.per_model_optimize import _recode_advanced_per_fold
from simulation.pipeline.wfcv import _QUANTILE_SPECS


def test_g309_noop_when_quantile_columns_absent():
    """Non-MPH_ADVANCED run (no quantile/threshold/interaction cols) → X unchanged (safe)."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(50, 5))
    y = np.abs(rng.normal(size=50))
    fc = ["lag1", "lag2", "sin_week", "cos_week", "trend"]
    Xf = _recode_advanced_per_fold(X, y, fc, train_end=30)
    assert np.array_equal(Xf, X), "absent columns → no-op (existing runs unaffected)"


def test_g309_noop_when_feature_cols_none():
    X = np.random.RandomState(1).normal(size=(20, 3))
    assert _recode_advanced_per_fold(X, np.ones(20), None, 10) is X


def test_g309_recodes_quantile_cols_when_present():
    """quantile cols present → *_qbin/*_qnorm recomputed from [:train_end] (differ from build-time),
    while the source + unrelated columns stay untouched."""
    src, _ = _QUANTILE_SPECS[0]
    fc = [src, f"{src}_qbin", f"{src}_qnorm", "lag1"]
    rng = np.random.RandomState(0)
    X = rng.normal(size=(60, 4))
    y = np.abs(rng.normal(size=60))
    Xf = _recode_advanced_per_fold(X, y, fc, train_end=40)
    assert not np.array_equal(X[:, 1:3], Xf[:, 1:3]), "qbin/qnorm recoded (fold-local)"
    assert np.array_equal(X[:, 0], Xf[:, 0]), "source column untouched"
    assert np.array_equal(X[:, 3], Xf[:, 3]), "unrelated column untouched"


def test_g309_does_not_mutate_input():
    src, _ = _QUANTILE_SPECS[0]
    fc = [src, f"{src}_qbin", f"{src}_qnorm"]
    X = np.random.RandomState(2).normal(size=(40, 3))
    X_copy = X.copy()
    _recode_advanced_per_fold(X, np.abs(X[:, 0]), fc, 25)
    assert np.array_equal(X, X_copy), "input X must never be mutated"
