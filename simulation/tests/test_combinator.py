"""Smoke test: feature_engine.combinator runs through all 4 stages."""
from __future__ import annotations

import numpy as np


def test_combinator_produces_augmented_X():
    from simulation.models.feature_engine.combinator import combinate_features

    rng = np.random.default_rng(0)
    n, p = 120, 6
    X = rng.normal(size=(n, p))
    # True relationship: y depends on x0*x1 + x2
    y = 2.0 * X[:, 0] * X[:, 1] + X[:, 2] + rng.normal(0, 0.2, size=n)
    feature_names = [f"x{i}" for i in range(p)]

    X_aug, new_names, report = combinate_features(
        X, y, feature_names,
        orders=(2,),
        max_candidates_per_order=15,
        percentile_mi_keep=50.0,
        use_pcmci=False,      # skip — tigramite optional
        use_optuna=False,     # skip — speeds up test
    )
    assert X_aug.shape[0] == n
    assert X_aug.shape[1] >= p
    assert report.n_main_effects == p
    assert report.n_candidates_by_order.get(2, 0) > 0


def test_combinator_prefers_interaction_x0_x1():
    """When y = x0*x1, the combinator should retain x0×x1."""
    from simulation.models.feature_engine.combinator import combinate_features

    rng = np.random.default_rng(1)
    n = 150
    X = rng.normal(size=(n, 4))
    y = 3.0 * X[:, 0] * X[:, 1] + 0.1 * rng.normal(size=n)
    feature_names = ["x0", "x1", "x2", "x3"]

    X_aug, new_names, report = combinate_features(
        X, y, feature_names,
        orders=(2,),
        use_pcmci=False,
        use_optuna=False,
    )
    # MI filter should keep x0×x1 in the shortlist
    assert any("x0" in n and "x1" in n for n in new_names)
