"""Phase C2: jackknife+ / CV+ conformal methods (Barber+2021)."""
from __future__ import annotations

import numpy as np
import pytest


def test_split_conformal_reduces_to_ceil_order_statistic():
    """split_conformal_interval must produce the same quantile as
    phase10_intervals._conformal_pi (the established implementation)."""
    from simulation.models.conformal import split_conformal_interval
    from simulation.pipeline.intervals import _conformal_pi

    rng = np.random.default_rng(0)
    y_cal = rng.normal(size=80)
    p_cal = rng.normal(size=80)
    y_test = rng.normal(size=30)
    p_test = rng.normal(size=30)
    residuals_cal = np.abs(y_cal - p_cal)

    # New function
    lo, hi = split_conformal_interval(p_test, residuals_cal, alpha=0.1)
    # Reference
    ref = _conformal_pi(y_cal, p_cal, y_test, p_test, alpha=0.1)
    q_new = float((hi[0] - lo[0]) / 2)
    # phase10_intervals._conformal_pi rounds `quantile` to 4 dp on output.
    q_ref = float(ref["quantile"])
    assert q_new == pytest.approx(q_ref, abs=5e-4)


def test_qplus_and_qminus_endpoints():
    """q⁺ at level=1 returns the max; q⁻ at level close to 0 picks the min."""
    from simulation.models.conformal import _qplus, _qminus
    v = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0])
    assert _qplus(v, 1.0) == 9.0
    # level·(n+1) = 0.1·7 = 0.7 → ⌈0.7⌉=1 → v_sorted[0] = 1.0
    assert _qplus(v, 0.1) == 1.0
    assert _qminus(v, 1.0) == 9.0   # floor(7)=7, clip→6 → max
    assert _qminus(v, 0.01) == 1.0  # floor(0.07)=0, clip→1 → min


def test_jackknife_plus_shape_and_coverage_vs_split():
    """Jackknife+ intervals must have correct shape and empirical coverage
    at the nominal level on a synthetic i.i.d. problem — and typically
    a tighter or comparable interval vs. split conformal."""
    from simulation.models.conformal import (
        jackknife_plus_interval,
        split_conformal_interval,
    )
    rng = np.random.default_rng(42)
    n_cal, n_test = 200, 100
    y_test = rng.normal(scale=1.0, size=n_test)

    # Simulate LOO predictions: each row i has a slightly different prediction
    # for every test point, with residuals_cal iid from the same scale.
    base_pred = rng.normal(scale=0.1, size=n_test)            # shared mean
    fp = base_pred[None, :] + rng.normal(scale=0.05, size=(n_cal, n_test))
    residuals_cal = np.abs(rng.normal(scale=1.0, size=n_cal))

    lo_j, hi_j = jackknife_plus_interval(fp, residuals_cal, alpha=0.1)
    assert lo_j.shape == (n_test,)
    assert hi_j.shape == (n_test,)
    assert np.all(hi_j >= lo_j)

    cov_j = float(np.mean((y_test >= lo_j) & (y_test <= hi_j)))
    # Theoretical guarantee: ≥ 1 - 2α = 0.80. With n=200 and N(0,1) data,
    # empirical coverage should be comfortably above that.
    assert cov_j >= 0.80

    # Sanity: split-conformal on the same residuals produces a PI of
    # comparable width (both methods apply the same scale-1 residuals).
    lo_s, hi_s = split_conformal_interval(base_pred, residuals_cal, alpha=0.1)
    split_width = float(np.mean(hi_s - lo_s))
    j_width = float(np.mean(hi_j - lo_j))
    # J+ can be up to ~2× split width in the worst case (Barber+2021 §3).
    assert j_width <= 2.5 * split_width


def test_jackknife_plus_rejects_shape_mismatch():
    from simulation.models.conformal import jackknife_plus_interval
    with pytest.raises(ValueError, match="n_cal mismatch"):
        jackknife_plus_interval(
            np.zeros((10, 5)), np.zeros(7), alpha=0.1
        )
    with pytest.raises(ValueError, match="must be 2-D"):
        jackknife_plus_interval(
            np.zeros(10), np.zeros(10), alpha=0.1
        )


def test_cv_plus_expands_fold_representation():
    """cv_plus_interval should recover the same output as jackknife+
    when we expand the fold representation by hand."""
    from simulation.models.conformal import (
        cv_plus_interval,
        jackknife_plus_interval,
    )
    rng = np.random.default_rng(7)
    n_test = 12
    # 3 folds × 5 indices each = 15 cal points
    fold_preds = {
        f"f{k}": rng.normal(scale=1.0, size=n_test) for k in range(3)
    }
    fold_indices = {
        "f0": [0, 1, 2, 3, 4],
        "f1": [5, 6, 7, 8, 9],
        "f2": [10, 11, 12, 13, 14],
    }
    residuals = np.abs(rng.normal(scale=0.5, size=15))

    lo_cv, hi_cv = cv_plus_interval(fold_preds, fold_indices, residuals, alpha=0.1)

    # Hand-expand into jackknife+ input
    fp = np.empty((15, n_test))
    for k, ids in fold_indices.items():
        for i in ids:
            fp[i] = fold_preds[k]
    lo_j, hi_j = jackknife_plus_interval(fp, residuals, alpha=0.1)

    np.testing.assert_allclose(lo_cv, lo_j)
    np.testing.assert_allclose(hi_cv, hi_j)


def test_conformal_interval_dispatch():
    """Dispatcher must route correctly and raise on unknown method."""
    from simulation.models.conformal import conformal_interval

    rng = np.random.default_rng(0)
    pt = rng.normal(size=10)
    r = np.abs(rng.normal(size=50))
    lo, hi = conformal_interval(
        method="split", pred_test=pt, residuals_cal=r, alpha=0.1
    )
    assert lo.shape == hi.shape == (10,)

    with pytest.raises(ValueError, match="unknown conformal method"):
        conformal_interval(method="bogus", pred_test=pt, residuals_cal=r)


# ══════════════════════════════════════════════════════════════════════════
# F3 — CV+ wiring through phase6 from phase7-shaped inputs.
# The upstream phase7 loop now ships per-model `(K, H)` holdout matrices
# and per-fold `(val_start, val_end)` windows. The helper below aggregates
# those into the form `conformal.cv_plus_interval` expects.
# ══════════════════════════════════════════════════════════════════════════
def test_cv_plus_pi_from_folds_happy_path():
    """With well-calibrated per-fold holdout preds, CV+ should reach ~95%
    coverage on the held-out slab."""
    from simulation.pipeline.intervals import _cv_plus_pi_from_folds

    rng = np.random.default_rng(11)
    n = 200
    ho = 160
    y = np.sin(np.linspace(0, 6 * np.pi, n)) + rng.normal(scale=0.2, size=n)
    oof = y + rng.normal(scale=0.25, size=n)
    oof[:40] = np.nan  # min-train cold start

    fold_val_indices = [(40 + 20 * k, 60 + 20 * k) for k in range(6)]
    h = n - ho
    fold_holdout = np.vstack([
        y[ho:] + rng.normal(scale=0.3, size=h) for _ in fold_val_indices
    ])
    out = _cv_plus_pi_from_folds(
        y_all=y, oof_pred=oof,
        fold_holdout_preds=fold_holdout,
        fold_val_indices=fold_val_indices,
        y_test=y[ho:], alpha=0.05,
    )
    assert out["n_folds"] == len(fold_val_indices)
    assert out["coverage"] >= 0.80   # ≥ 1 - 2α Barber+2021 guarantee
    assert out["width"] > 0


def test_cv_plus_pi_from_folds_returns_empty_without_data():
    from simulation.pipeline.intervals import _cv_plus_pi_from_folds
    out = _cv_plus_pi_from_folds(
        y_all=np.zeros(10), oof_pred=np.zeros(10),
        fold_holdout_preds=None,
        fold_val_indices=[],
        y_test=np.zeros(3), alpha=0.05,
    )
    assert out == {}


def test_run_phase6_surfaces_cv_plus_entry():
    """Smoke: run_intervals with fold_holdout_predictions populated must
    attach a non-empty `cv_plus` dict to each model entry."""
    from simulation.pipeline.intervals import run_intervals

    class _Cfg:
        class data: dates = None
        class scoring: residual_space_mode = "off"

    rng = np.random.default_rng(13)
    n = 260
    ho = 234
    y = np.sin(np.linspace(0, 12, n)) + rng.normal(scale=0.2, size=n)
    oof_pred = y + rng.normal(scale=0.25, size=n)
    oof = {"M": oof_pred}
    ho_preds = {"M": oof_pred[ho:]}
    fold_val = [(40 + 20 * k, 60 + 20 * k) for k in range(5)]
    h = n - ho
    fold_mat = np.vstack([
        oof_pred[ho:] + rng.normal(scale=0.2, size=h) for _ in fold_val
    ])
    r = run_intervals(
        y, oof, _Cfg(),
        holdout_predictions=ho_preds, holdout_start=ho,
        fold_holdout_predictions={"M": fold_mat},
        fold_val_indices={"M": fold_val},
    )
    entry = r["pi_results"]["M"]
    assert "cv_plus" in entry
    assert entry["cv_plus"]  # non-empty dict
    assert entry["cv_plus"]["n_folds"] == len(fold_val)
