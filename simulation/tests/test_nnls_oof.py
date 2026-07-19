"""Phase D1: NNLS weights on the OOF slab."""
from __future__ import annotations

import numpy as np
import pytest


def test_oof_nnls_weights_respects_holdout_start():
    """Weights must be computed from [0, holdout_start) only — injecting
    wild errors into the holdout slab must not change the output."""
    from simulation.pipeline.ensemble_weights import oof_nnls_weights

    rng = np.random.default_rng(0)
    n = 200
    ho = 160
    y = np.sin(np.linspace(0, 6 * np.pi, n))
    oof = {
        "A": y + rng.normal(scale=0.1, size=n),
        "B": y + rng.normal(scale=0.3, size=n),
    }
    w1 = oof_nnls_weights(oof, y, holdout_start=ho)
    # perturb only the holdout slab
    oof2 = {k: v.copy() for k, v in oof.items()}
    oof2["A"][ho:] += 1e3
    oof2["B"][ho:] -= 1e3
    w2 = oof_nnls_weights(oof2, y, holdout_start=ho)
    assert w1 == w2
    # weights sum to 1
    assert sum(w1.values()) == pytest.approx(1.0, abs=1e-9)
    # non-negative
    assert all(v >= 0 for v in w1.values())


def test_oof_nnls_weights_r2_floor_drops_bad_models():
    """Models below r2_floor should be filtered and receive weight=0."""
    from simulation.pipeline.ensemble_weights import oof_nnls_weights

    rng = np.random.default_rng(1)
    n = 150
    y = np.sin(np.linspace(0, 5 * np.pi, n))
    good = y + rng.normal(scale=0.05, size=n)
    noise = rng.normal(scale=3.0, size=n)           # R² ≪ 0
    oof = {"Good": good, "Noise": noise}

    w = oof_nnls_weights(oof, y, holdout_start=n, r2_floor=0.3)
    assert "Noise" in w and w["Noise"] == 0.0
    assert w["Good"] > 0


def test_oof_nnls_weights_all_rejected_returns_empty():
    from simulation.pipeline.ensemble_weights import oof_nnls_weights
    rng = np.random.default_rng(2)
    n = 120
    y = np.ones(n)
    bad = {
        "X": rng.normal(scale=10.0, size=n),
        "Y": rng.normal(scale=10.0, size=n),
    }
    w = oof_nnls_weights(bad, y, r2_floor=0.3)
    assert w == {}


def test_oof_nnls_weights_handles_nan_gaps():
    """NaN positions in OOF preds should not crash nnls; joint-valid rows
    are used and output weights still sum to 1."""
    from simulation.pipeline.ensemble_weights import oof_nnls_weights

    rng = np.random.default_rng(3)
    n = 180
    y = np.sin(np.linspace(0, 6 * np.pi, n))
    a = y + rng.normal(scale=0.1, size=n)
    b = y + rng.normal(scale=0.15, size=n)
    a[:5] = np.nan   # early-fold gap
    b[170:] = np.nan # late-fold gap (still inside cal end if holdout=None)
    oof = {"A": a, "B": b}
    w = oof_nnls_weights(oof, y)
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-9)


def test_blend_oof_predictions_applies_weights():
    from simulation.pipeline.ensemble_weights import blend_oof_predictions

    oof = {
        "A": np.array([1.0, 2.0, 3.0, 4.0]),
        "B": np.array([2.0, 4.0, 6.0, 8.0]),
    }
    weights = {"A": 0.25, "B": 0.75}
    blend = blend_oof_predictions(oof, weights)
    # 0.25·A + 0.75·B = [1.75, 3.5, 5.25, 7.0]
    np.testing.assert_allclose(blend, [1.75, 3.5, 5.25, 7.0])


def test_blend_oof_predictions_skips_zero_and_missing_weights():
    from simulation.pipeline.ensemble_weights import blend_oof_predictions
    oof = {"A": np.ones(3), "B": 2 * np.ones(3)}
    weights = {"A": 0.0, "B": 1.0, "C": 0.5}   # C missing from oof
    blend = blend_oof_predictions(oof, weights)
    # Only B contributes; renormalized to 1.0
    np.testing.assert_allclose(blend, [2.0, 2.0, 2.0])
