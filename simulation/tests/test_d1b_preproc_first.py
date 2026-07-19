"""D1=b: preproc-FIRST staged selection helper (`_preproc_first_select`).

Verifies the isolated preproc→feature logic (the user's order, replacing the pre-stage
feature load) runs and returns a valid (best_preproc, feature_indices). Wiring into
optimize_one_model is gated behind MPH_PREPROC_FIRST (default 0) — this test covers the
new logic in isolation before that integration.

macOS: run PER-FILE.
"""
import numpy as np
import pytest

from simulation.tests.test_d4_preproc_oof_blind import _skewed_data, _elasticnet_factory

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def test_preproc_first_select_returns_preproc_then_features():
    from simulation.pipeline._inline_optuna_3stage import _preproc_first_select

    X, y = _skewed_data(n=180)
    fac = _elasticnet_factory()
    cols = [f"f{i}" for i in range(X.shape[1])]
    ntr = 120

    best_preproc, feat_idx = _preproc_first_select(
        "ElasticNet", fac, X[:ntr], y[:ntr], X[ntr:], y[ntr:],
        feature_cols=cols, n_trials_preproc=3, n_trials_feature=3,
    )

    # Stage 1 produced a preproc cell with transform/scaler chosen
    assert isinstance(best_preproc, dict)
    assert "transform" in best_preproc and "scaler" in best_preproc
    # Stage 2 produced either all-columns (None) or a valid index subset
    assert feat_idx is None or (
        isinstance(feat_idx, list) and all(0 <= int(i) < X.shape[1] for i in feat_idx)
    )
