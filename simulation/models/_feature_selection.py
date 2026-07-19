"""_feature_selection.py
========================
Shared MI top-K feature-selection helper with training-size-aware K capping.

Background (per_model_pipeline_isolated, 2026-04-22):
 A hard-coded K that wins on train+val (n=214) often breaks at train-only
 (n=180) or at leftmost WF-CV folds (n < 180) because of p>n rank deficiency.
 Hence factory defaults are kept conservative (K ∈ {20, 40, 80}).

This helper lets each factory request its "desired K", but the helper caps
it by `n_train // divisor` so the effective K stays a safe fraction of the
training set size. Divisor defaults to 4 (K ≤ n/4, a common rule of thumb
for regularized regression to avoid p>n collapse).

Usage inside a factory's fit:
 from simulation.models._feature_selection import mi_top_k_adaptive
 idx = mi_top_k_adaptive(X_train, y_train, K_desired=40)
 X_sel = X_train[: idx]
 # stash idx as self._feat_idx for predict time.
"""
from __future__ import annotations

import numpy as np


def mi_top_k_adaptive(
    X: np.ndarray,
    y: np.ndarray,
    K_desired: int = 40,
    divisor: int = 4,
    min_k: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """Return feature indices for MI top-K with K capped by n_train // divisor.

    Args:
        X, y: training features and targets (MI is computed on these ONLY —
              no test leakage).
        K_desired: the K the factory would ideally use.
        divisor: K is capped by n_train // divisor. Default 4 means K ≤ n/4.
        min_k: minimum K even if n is tiny (ensures at least `min_k` features).
        random_state: seed for sklearn's MI estimator.

    Returns:
        1-D int array of length min(K_desired, n_train // divisor, X.shape[1]),
        containing the indices of the top-MI features in descending MI order.
    """
    from sklearn.feature_selection import mutual_info_regression

    n, p = X.shape
    K_cap = max(min_k, n // divisor)
    K_eff = min(K_desired, K_cap, p)

    mi = mutual_info_regression(X, y, random_state=random_state)
    idx = np.argsort(-mi)[:K_eff]
    return idx
