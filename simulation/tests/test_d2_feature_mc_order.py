"""D2: feature↔mc order — A (feature→mc) vs B (mc→feature) vs HYBRID, empirically.

Synthetic data where the UNIQUE signal carrier f0 is collinear with several NOISE features,
so target-blind vif gives f0 a high VIF and tends to drop it. Which order keeps the signal
and forecasts better?
  A:      target-aware select (top-k |corr y|) on full → vif on the selected subset.
  B:      vif on the FULL set (target-blind) → select on survivors.
  HYBRID: |corr|>0.98 dedup → select → vif on survivors.
Selection/mc are fit on TRAIN only; R² measured on a held-out tail. Averaged over seeds.

macOS: run PER-FILE.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _data(n=240, seed=0):
    rng = np.random.default_rng(seed)
    f0 = rng.normal(size=n)                       # unique signal carrier
    f4 = rng.normal(size=n)                       # independent signal
    f1 = f0 + rng.normal(0, 0.55, n)             # noise features collinear with f0
    f2 = f0 + rng.normal(0, 0.55, n)
    f3 = f0 + rng.normal(0, 0.55, n)
    noise = rng.normal(size=(n, 5))
    X = np.column_stack([f0, f1, f2, f3, f4, noise])
    y = 3.0 * f0 + 2.0 * f4 + rng.normal(0, 0.5, n)
    return X, y, [f"f{i}" for i in range(X.shape[1])]


def _select_topk(X, y, cols, k):
    sc = []
    for i in range(X.shape[1]):
        c = X[:, i]
        r = 0.0 if np.std(c) < 1e-9 else abs(float(np.corrcoef(c, y)[0, 1]))
        sc.append((0.0 if not np.isfinite(r) else r, i))
    sc.sort(reverse=True)
    return sorted(cols[i] for _, i in sc[:k])


def _vif(X, y, cols):
    from simulation.pipeline.mc_filter_stage3 import apply_multicollinearity_filter
    d = np.zeros((1, X.shape[1]))
    _, _, _, kept, _ = apply_multicollinearity_filter(X, X, d, y, feature_cols=cols, method="vif")
    return list(cols) if kept is None else [cols[i] for i in kept]


def _dedup(X, cols, thr=0.98):
    drop = set()
    for i in range(X.shape[1]):
        if i in drop:
            continue
        for j in range(i + 1, X.shape[1]):
            if j in drop or np.std(X[:, i]) < 1e-9 or np.std(X[:, j]) < 1e-9:
                continue
            if abs(np.corrcoef(X[:, i], X[:, j])[0, 1]) > thr:
                drop.add(j)
    return [cols[i] for i in range(X.shape[1]) if i not in drop]


def _holdout_r2(X, y, cols, final_names):
    from sklearn.linear_model import Ridge
    idx = [cols.index(nm) for nm in final_names if nm in cols]
    if not idx:
        return -1.0
    ntr = int(len(y) * 0.7)
    Xs = X[:, idx]
    return float(Ridge(alpha=0.01).fit(Xs[:ntr], y[:ntr]).score(Xs[ntr:], y[ntr:]))


def _orders_once(seed):
    X, y, cols = _data(seed=seed)
    ntr = int(len(y) * 0.7)
    Xtr, ytr = X[:ntr], y[:ntr]
    # A: select → vif
    selA = _select_topk(Xtr, ytr, cols, k=5)
    finalA = _vif(Xtr[:, [cols.index(n) for n in selA]], ytr, selA)
    # B: vif → select
    keptB = _vif(Xtr, ytr, cols)
    finalB = _select_topk(Xtr[:, [cols.index(n) for n in keptB]], ytr, keptB, min(5, len(keptB)))
    # HYBRID: dedup → select → vif
    dd = _dedup(Xtr, cols, 0.98)
    selH = _select_topk(Xtr[:, [cols.index(n) for n in dd]], ytr, dd, min(5, len(dd)))
    finalH = _vif(Xtr[:, [cols.index(n) for n in selH]], ytr, selH)
    return {
        "A": (_holdout_r2(X, y, cols, finalA), "f0" in finalA),
        "B": (_holdout_r2(X, y, cols, finalB), "f0" in finalB),
        "HYBRID": (_holdout_r2(X, y, cols, finalH), "f0" in finalH),
    }


def summarize(seeds=range(8)):
    res = [_orders_once(s) for s in seeds]
    r2 = {k: float(np.mean([r[k][0] for r in res])) for k in ("A", "B", "HYBRID")}
    keep = {k: float(np.mean([r[k][1] for r in res])) for k in ("A", "B", "HYBRID")}
    return r2, keep


def test_order_B_loses_signal_more_than_A_and_hybrid():
    r2, keep = summarize()
    # A and HYBRID retain the signal carrier f0 at least as often as the target-blind B:
    assert keep["A"] >= keep["B"], f"f0-retention A={keep['A']} vs B={keep['B']}"
    assert keep["HYBRID"] >= keep["B"], f"f0-retention HYBRID={keep['HYBRID']} vs B={keep['B']}"
    # and forecast at least as well on average (target-aware-first ≥ target-blind-first):
    assert r2["A"] >= r2["B"] - 0.02, f"R2 A={r2['A']:.3f} vs B={r2['B']:.3f}"
    assert r2["HYBRID"] >= r2["B"] - 0.02, f"R2 HYBRID={r2['HYBRID']:.3f} vs B={r2['B']:.3f}"
