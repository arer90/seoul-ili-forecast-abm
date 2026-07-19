"""G-362: PermutationExplainer fallback for native-SHAP-gap models.

Reproduction (2026-06-25): 5 models had native_shap_importance=NaN in importance.csv —
SVR-RBF (kernel 348-feat, KernelExplainer skipped by G-348), Mamba/N-BEATS/iTransformer
(deep, DeepExplainer+GradientExplainer fail on custom nets), GAM-Spline (LinearExplainer
fail, GAM not sklearn-linear). They had permutation importance but no per-sample SHAP.

Fix: _permutation_shap_fallback — model-agnostic shap.PermutationExplainer in the original
feature space. For p>k, attribute only the top-k features (by perm importance) with the
rest held at the background median → bounds cost to ~2k+1 predict calls.
"""
import numpy as np

from simulation.pipeline.shap_analysis import _permutation_shap_fallback


def _data(n=50, p=20, signal=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    X[:, signal] *= 4.0
    cols = [f"f{i}" for i in range(p)]
    return X, cols


def test_fallback_low_dim_full():
    """p<=k → full PermutationExplainer; signal feature ranks top, sv is (rows,p)."""
    X, cols = _data(p=20)
    fn = lambda Z: Z[:, 3] * 2.0
    ranked, sv = _permutation_shap_fallback(fn, X, cols, perm_ranked=[], k=30)
    assert ranked and ranked[0][0] == "f3"
    assert sv is not None and sv.shape[1] == 20


def test_fallback_high_dim_subset_bounded():
    """p>k → top-k subset (by perm importance); sv still full-width, signal ranks top."""
    X, cols = _data(p=100)
    fn = lambda Z: Z[:, 3] * 2.0
    perm = [("f3", 99.0), ("f7", 10.0)] + [(f"f{i}", 0.1) for i in range(100) if i not in (3, 7)]
    ranked, sv = _permutation_shap_fallback(fn, X, cols, perm_ranked=perm, k=30)
    assert ranked and ranked[0][0] == "f3"     # signal in top-k survives
    assert sv is not None and sv.shape[1] == 100  # mapped back to full width
    # features outside top-k are zero (not attributed)
    nz = int((np.abs(sv).mean(axis=0) > 1e-12).sum())
    assert nz <= 30


def test_fallback_predict_fail_returns_empty():
    X, cols = _data()
    bad = lambda Z: (_ for _ in ()).throw(RuntimeError("boom"))
    ranked, sv = _permutation_shap_fallback(bad, X, cols, perm_ranked=[])
    assert ranked == [] and sv is None


def test_fallback_skips_ultra_slow_predict():
    """G-362b: ultra-slow predict (TabPFN류) → time-budget 초과 → fallback skip (perm 유지)."""
    import time as _t
    X, cols = _data(p=100)
    slow = lambda Z: (_t.sleep(0.05), Z[:, 3] * 2.0)[1]   # 0.05s/call → 2k+1 evals 폭발
    ranked, sv = _permutation_shap_fallback(slow, X, cols, perm_ranked=[], k=30, time_budget=0.01)
    assert ranked == [] and sv is None       # k_eff<3 → skip, R11 무한정체 방지


def test_fallback_variance_fallback_when_no_perm():
    """No perm ranking → top-k by variance (high-variance signal col survives)."""
    X, cols = _data(p=60)
    fn = lambda Z: Z[:, 3] * 2.0
    ranked, sv = _permutation_shap_fallback(fn, X, cols, perm_ranked=[], k=20)
    assert ranked and ranked[0][0] == "f3"   # f3 has 4x variance → in top-k
    assert sv is not None and sv.shape[1] == 60
