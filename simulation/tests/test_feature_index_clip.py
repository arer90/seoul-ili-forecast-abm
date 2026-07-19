"""G-242: feature_indices clip guard — stale original-space indices on an mc-reduced X.

codex+gemini review (2026-05-30) of the Optuna order found a real latent bug: in
run_per_model_optimize the multicollinearity 4-method is applied GLOBALLY (once, before the per-model
loop), reducing X. Then per-model, optimize_one_model rebuilds Stage-2 indices by matching
pre-selected feature NAMES against the (reduced) feature_cols:
    stage2_indices = [i for i, c in enumerate(feature_cols) if c in best_features]
- none/vif/corr: names still match survivors → valid reduced-space indices (safe).
- pca:           feature_cols are now PC1..PCk → intersection EMPTY → the incoming
                 original-space feature_indices survives → X[:, idx] IndexError on PCs.

`_clip_feature_indices` drops out-of-range indices (None ⇒ use all columns), making the
pca path graceful + visible. This file is the TDD for that guard + documents the
mc×stage2 interaction the user asked to verify.

macOS: run PER-FILE (memory ``test-suite-execution``).
"""
import numpy as np

from simulation.pipeline.per_model_optimize import _clip_feature_indices


def test_clip_none_passthrough():
    assert _clip_feature_indices(None, 10) is None


def test_clip_in_range_unchanged():
    # none/vif/corr path: indices already valid for the (reduced) X
    assert _clip_feature_indices([0, 2, 5], 10) == [0, 2, 5]


def test_clip_drops_out_of_range():
    assert _clip_feature_indices([0, 3, 12, 20], 10) == [0, 3]


def test_clip_all_out_of_range_returns_none():
    # pca scenario: original-space indices on a small PC array → none survive → all-cols
    assert _clip_feature_indices([40, 41, 100], 8) is None


def test_pca_intersection_empties_then_clip_prevents_indexerror():
    """Mini-repro of the run_per_model_optimize → optimize_one_model interaction under mc=pca."""
    X = np.random.default_rng(0).normal(size=(50, 6))           # post-pca: 6 PC columns
    feature_cols = [f"PC{i+1}" for i in range(6)]
    best_features = {"ari_rate", "ili_rate_lag1", "fourier_h1"}  # original Stage-2 names

    # the exact intersection optimize_one_model does:
    stage2_indices = [i for i, c in enumerate(feature_cols) if c in best_features]
    assert stage2_indices == [], "pca name-intersection MUST be empty (the latent bug)"

    feature_indices = [10, 20, 30]                              # stale original-space idx
    feature_indices = _clip_feature_indices(feature_indices, X.shape[1])
    assert feature_indices is None                              # all out of range → all-cols

    # downstream slice emulation must NOT raise:
    X_use = X if feature_indices is None else X[:, feature_indices]
    assert X_use.shape == (50, 6)


def test_vif_corr_path_indices_unchanged():
    """vif/corr: Stage-2 names still match survivors → valid reduced-space indices kept."""
    assert _clip_feature_indices([1, 4, 9], 12) == [1, 4, 9]
