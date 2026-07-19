"""③ gated feature-neighborhood (MPH_FEAT_NEIGHBORHOOD_K, default 0=off) — G-242.

K=0 → frozen pre-stage subset unchanged (default behaviour, ZERO pipeline change).
K>0 → subset enlarged by its k nearest |corr-with-target| neighbours, pre-stage set
always preserved as a subset (the frozen choice is never lost).

macOS: run PER-FILE (memory ``test-suite-execution``).
"""
import numpy as np

from simulation.pipeline.per_model_optimize import _expand_feature_neighborhood


def _data(n=120, p=12, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    # features 5 & 7 strongly drive y → they are the "neighbours" worth picking up
    y = 2.0 * X[:, 5] + 1.5 * X[:, 7] + rng.normal(0, 0.3, n)
    return X, y


def test_k_zero_is_noop():
    X, y = _data()
    base = [0, 1, 2]
    assert _expand_feature_neighborhood(base, X, y, k=0) == base


def test_k_negative_is_noop():
    X, y = _data()
    base = [3, 4]
    assert _expand_feature_neighborhood(base, X, y, k=-5) == base


def test_k_positive_is_superset_bounded():
    X, y = _data()
    base = [0, 1, 2]
    out = _expand_feature_neighborhood(base, X, y, k=3)
    assert set(base).issubset(set(out)), f"pre-stage set lost: {out}"
    assert len(out) <= len(base) + 3, f"grew beyond k: {out}"
    assert len(out) > len(base), "k>0 should add neighbours when candidates exist"


def test_neighbours_are_highest_corr():
    """Added neighbours include the strongly-correlated features (5, 7)."""
    X, y = _data()
    base = [0, 1]  # 5,7 left out → must be picked up as top-corr neighbours
    out = _expand_feature_neighborhood(base, X, y, k=2)
    added = set(out) - set(base)
    assert added == {5, 7}, f"expected high-corr feats 5,7; added {added}"


def test_k_capped_by_available_candidates():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 5))
    y = X[:, 0] + rng.normal(0, 0.3, 120)
    base = [0, 1, 2, 3, 4]  # everything already selected → no candidates
    assert sorted(_expand_feature_neighborhood(base, X, y, k=3)) == sorted(base)
