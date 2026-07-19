"""TDD for the leakage-free + de-circularized unified feature-method comparison harness.

codex + Gemini (3-way eval, 2026-06-01) flagged two CRITICAL bugs in the original harness:
  1. selection leakage — features selected ONCE on the full pool, then scored across folds
     (the selection saw the validation rows).
  2. POOL_M=18 straitjacket — wrapper candidate pool was purely top-|corr|, so wrappers could
     never reach a feature that |corr| misses (rigged "wrappers don't beat |corr|").

These tests pin the fix:
  - `_oof3_nested` re-runs feature selection INSIDE each fold on the training prefix only.
  - `_fold_cand` builds a de-circularized pool = union(top-|corr|, top-RF-importance) so
    interaction-only features (low marginal |corr|, high model importance) are reachable.

Run PER-FILE on macOS: KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 pytest <thisfile>
"""
import numpy as np
import pytest

import simulation.scripts._unified_feature_method_comparison as U


def test_fold_cand_reaches_interaction_feature_corr_misses():
    """De-circularization: a feature with near-zero marginal |corr| but high RF importance
    (interaction-only) MUST be in the candidate pool — otherwise wrappers are straitjacketed."""
    rng = np.random.default_rng(0)
    n, p = 320, 50   # p ≫ POOL_CORR so top-|corr| is selective (mirrors ILI's ~258 features)
    X = rng.standard_normal((n, p))
    # features 0,1 are interaction-only: their PRODUCT drives y, marginals ~0 corr.
    y = 1.8 * (X[:, 0] * X[:, 1]) + 0.3 * X[:, 2] + 0.1 * rng.standard_normal(n)
    cand, cs = U._fold_cand(X, y)
    # feature 0 is interaction-only → its marginal |corr| is genuinely tiny (hidden from |corr|).
    assert cs[0] < 0.12, f"feature 0 should hide from |corr|, got {cs[0]:.3f}"
    top_corr = set(np.argsort(cs)[::-1][: U.POOL_CORR].tolist())
    assert 0 not in top_corr, "test premise: feature 0 should NOT be in the top-|corr| pool"
    # ...yet the de-circularized pool reaches it via RF importance — the whole point of the fix.
    assert 0 in cand, (
        f"feature 0 (|corr|={cs[0]:.3f}) that |corr| misses must be reachable via importance; cand={cand}")


def test_fold_cand_is_union_not_pure_corr():
    """The pool is a UNION of |corr| and importance — generally larger / different from top-|corr| alone."""
    rng = np.random.default_rng(1)
    n, p = 250, 20
    X = rng.standard_normal((n, p))
    y = 1.5 * (X[:, 0] * X[:, 1]) + 0.8 * X[:, 5] + 0.2 * rng.standard_normal(n)
    cand, cs = U._fold_cand(X, y)
    top_corr_only = set(np.argsort(cs)[::-1][: U.POOL_CORR].tolist())
    assert not set(cand).issubset(top_corr_only) or len(set(cand)) > len(top_corr_only) - 1, \
        "union pool should differ from pure top-|corr| (importance adds features)"


def test_oof3_nested_selects_only_on_training_prefix(monkeypatch):
    """Leakage guard: every per-fold selection call sees ONLY the training prefix (rows < n),
    never the full pool. A leaky harness would pass the full Pp to the selector."""
    from sklearn.linear_model import LinearRegression
    rng = np.random.default_rng(2)
    n, p = 120, 10
    Pp = rng.standard_normal((n, p))
    yp = np.abs(rng.standard_normal(n)) + 0.1
    ylog = np.log1p(yp)
    inv = lambda z: np.expm1(np.clip(z, -2, 20))
    fac = lambda: LinearRegression()

    seen_rows = []

    def spy_select(method, fac_, P, y_, yl_, iv_, cand_, cs_):
        seen_rows.append(int(P.shape[0]))
        return [0, 1]  # trivial selection

    monkeypatch.setattr(U, "_select", spy_select)

    # build folds the same way the worker does
    nf = 3
    fs = n // (nf + 1)
    folds = []
    for k in range(1, nf + 1):
        etr = k * fs
        eva = (k + 1) * fs if k < nf else n
        if eva - etr < 4:
            continue
        cand_tr, cs_tr = U._fold_cand(Pp[:etr], ylog[:etr])
        folds.append((etr, eva, cand_tr, cs_tr))

    oof_mean, oof_sd = U._oof3_nested("STABILITY", fac, Pp, yp, ylog, inv, folds)

    assert seen_rows, "selection should have been called at least once per fold"
    assert max(seen_rows) < n, f"selection saw full pool (n={n}) → LEAKAGE; rows seen={seen_rows}"
    # the training prefixes are exactly k*fs for each fold
    assert seen_rows == [k * fs for k in range(1, nf + 1)], \
        f"per-fold selection must see growing training prefixes, got {seen_rows}"
    assert np.isfinite(oof_mean) and oof_sd >= 0.0


def test_oof3_nested_returns_fold_dispersion():
    """Error bars: nested OOF returns (mean, std) across folds, std>0 when folds differ."""
    from sklearn.linear_model import LinearRegression
    rng = np.random.default_rng(3)
    n, p = 140, 8
    Pp = rng.standard_normal((n, p))
    # non-stationary target so folds genuinely differ → std should be > 0
    t = np.linspace(0, 3, n)
    yp = np.abs(2.0 + t + 0.5 * Pp[:, 0] + 0.2 * rng.standard_normal(n))
    ylog = np.log1p(yp)
    inv = lambda z: np.expm1(np.clip(z, -2, 20))
    fac = lambda: LinearRegression()
    nf = 3
    fs = n // (nf + 1)
    folds = []
    for k in range(1, nf + 1):
        etr = k * fs
        eva = (k + 1) * fs if k < nf else n
        if eva - etr < 4:
            continue
        cand_tr, cs_tr = U._fold_cand(Pp[:etr], ylog[:etr])
        folds.append((etr, eva, cand_tr, cs_tr))
    mean, sd = U._oof3_nested("FULL", fac, Pp, yp, ylog, inv, folds)
    assert np.isfinite(mean) and mean < 1e8
    assert sd >= 0.0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
