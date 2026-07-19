"""D3: per-model mc — the decision must be DATA-DRIVEN, not a family heuristic (empirical).

Two empirical facts pinned here:

1. vif does NOT improve holdout R² for regularized (Ridge) or tree (RF) models on
   signal-carrying collinear data — they prefer ``none`` on point accuracy.

2. vif's effect on the train−test OVERFIT GAP is **data-dependent, with opposite sign**:
   - Regime A (many near-duplicate NOISE features = overfit fuel): vif REDUCES the gap.
   - Regime B (a few collinear copies that CARRY signal): vif does NOT reduce (can increase)
     the gap, because it throws away usable information.

Same method, opposite effect depending on the data → you cannot pick mc by a fixed
"linear→vif / tree→none" FAMILY rule. The mc method must be chosen **per-model from the
data**, which is exactly what ④ `mc_per_model_selection.csv` (OOF WIS + overfit_gap)
measures. ``none`` remains a sound default for point accuracy (matches live config).

macOS: run PER-FILE.
"""
import numpy as np
import pytest

from simulation.tests.test_mc_overfitting_control import _collinear_data  # regime A: noise dups

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _data(n=240, seed=0):
    """Regime B: a few collinear copies that CARRY the f0 signal (not pure noise)."""
    rng = np.random.default_rng(seed)
    f0 = rng.normal(size=n)
    f1 = rng.normal(size=n)
    copies = [f0 + rng.normal(0, 0.3, n) for _ in range(4)]
    noise = rng.normal(size=(n, 4))
    X = np.column_stack([f0, f1] + copies + [noise])
    y = 2.5 * f0 + 1.5 * f1 + rng.normal(0, 0.5, n)
    return X, y, [f"f{i}" for i in range(X.shape[1])]


def _apply(X, y, cols, method):
    if method == "none":
        return X
    from simulation.pipeline.mc_filter_stage3 import apply_multicollinearity_filter
    d = np.zeros((1, X.shape[1]))
    Xtr_f, _, _, _, _ = apply_multicollinearity_filter(X, X, d, y, feature_cols=cols, method=method)
    return np.asarray(Xtr_f)


def _model(name):
    if name == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=0.01)
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=60, random_state=0)


def _test_r2(name, X, y, cols, method):
    Xf = _apply(X, y, cols, method)
    ntr = int(len(y) * 0.7)
    m = _model(name).fit(Xf[:ntr], y[:ntr])
    return float(m.score(Xf[ntr:], y[ntr:]))


def _ridge_gap(X, y, cols, method):
    Xf = _apply(X, y, cols, method)
    ntr = int(len(y) * 0.7)
    from sklearn.linear_model import Ridge
    m = Ridge(alpha=0.01).fit(Xf[:ntr], y[:ntr])
    return float(m.score(Xf[:ntr], y[:ntr]) - m.score(Xf[ntr:], y[ntr:]))


def test_vif_does_not_improve_test_r2_for_regularized_or_tree():
    for name in ("ridge", "rf"):
        te_none = float(np.mean([_test_r2(name, *_data(seed=s), "none") for s in range(5)]))
        te_vif = float(np.mean([_test_r2(name, *_data(seed=s), "vif") for s in range(5)]))
        assert te_none >= te_vif - 0.01, (
            f"{name}: vif should not improve test R²: none={te_none:.3f} vif={te_vif:.3f}")


def test_mc_gap_benefit_is_data_dependent_not_heuristic():
    # Regime A: many near-duplicate NOISE features → vif REDUCES the overfit gap
    a_none = float(np.mean([_ridge_gap(*_collinear_data(seed=s), "none") for s in range(5)]))
    a_vif = float(np.mean([_ridge_gap(*_collinear_data(seed=s), "vif") for s in range(5)]))
    deltaA = a_vif - a_none
    # Regime B: signal-carrying collinear copies → vif does NOT reduce (can increase) the gap
    b_none = float(np.mean([_ridge_gap(*_data(seed=s), "none") for s in range(5)]))
    b_vif = float(np.mean([_ridge_gap(*_data(seed=s), "vif") for s in range(5)]))
    deltaB = b_vif - b_none
    # Same method, materially different effect across data regimes → can't use a family rule:
    assert deltaA < deltaB, (
        f"vif's gap-effect should be data-dependent: regimeA(noise-dup) Δ={deltaA:+.3f} "
        f"< regimeB(signal-dup) Δ={deltaB:+.3f} — if not, the demonstration fails")
    # and in the noise regime vif genuinely helps (gap reduction):
    assert deltaA < 0, f"vif should reduce gap on noise-dup data: Δ={deltaA:+.3f}"
