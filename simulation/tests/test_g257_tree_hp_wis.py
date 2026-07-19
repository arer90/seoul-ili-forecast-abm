"""Internal tree-HP fold score = WIS, not R² (G-257, codex).

The per-fold R² (1 − SS_res/SS_tot) divides by the fold variance, so on a low-variance quiet
fold a small error explodes R² to ≈−99 → HP selection dominated by quiet-fold NOISE and
inconsistent with the outer OOF WIS objective. `_fold_wis` uses a stable, sigma-scaled WIS.
"""
from __future__ import annotations

import numpy as np

from simulation.models.tree_models import _fold_wis


def test_fold_wis_stable_where_r2_explodes() -> None:
    """On a low-variance quiet fold, R² explodes negative but WIS stays bounded."""
    y_val = np.array([5.0, 5.1, 4.9, 5.2])     # quiet fold, tiny variance
    pred = np.array([6.0, 6.1, 5.9, 6.2])      # off by ~1
    y_tr = np.array([4.0, 5.0, 6.0, 5.0, 4.5])
    ss_res = float(np.sum((y_val - pred) ** 2))
    ss_tot = float(np.sum((y_val - y_val.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot
    assert r2 < -10.0, "setup: R² should explode on this low-variance fold"
    wis = _fold_wis(y_val, pred, y_tr)
    assert 0.0 < wis < 50.0, f"WIS should stay bounded/stable, got {wis}"


def test_fold_wis_lower_is_better() -> None:
    y_val = np.array([10.0, 20.0, 30.0, 40.0])
    y_tr = np.array([10.0, 20.0, 30.0, 40.0, 25.0])
    good = _fold_wis(y_val, np.array([11.0, 19.0, 31.0, 39.0]), y_tr)   # accurate
    bad = _fold_wis(y_val, np.array([25.0, 25.0, 25.0, 25.0]), y_tr)    # flat/wrong
    assert good < bad, f"accurate fold WIS {good} should beat inaccurate {bad}"
