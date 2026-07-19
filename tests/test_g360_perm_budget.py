"""G-360: permutation importance predict-time budget cap + X-independent skip.

Reproduction (2026-06-25): R11 SHAP stalled 1.5h+ on SeirCount-TabPFN —
permutation = O(p·n_repeats) = 348×3 = 1044 TabPFN predicts (slow per-call).
G-348 only capped native *kernel* SHAP, NOT the universal permutation backbone.

Fix:
  Part B (_permutation_importance): measure base-predict dt; if dt·p·n_repeats >
    time_budget, fall back to n_repeats=1 over the top-K highest-variance features.
    Fast models (dt≈ms) are unaffected (full 3-repeat).
  Part A (_explain_one): X-independent foundation (inner forecaster
    USES_FEATURES=False: TiRex/TimesFM-2.5/DLinear/TiRex-LoRA) → skip permutation
    (shuffling X cannot change the prediction → meaningless + slow). Read via
    art.model since the ChampionArtifact wrapper does not expose USES_FEATURES.
"""
import time

import numpy as np

from simulation.pipeline.shap_analysis import _permutation_importance


def _data(n=60, p=20, signal_col=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    X[:, signal_col] *= 5.0  # high variance + the only signal
    y = X[:, signal_col] * 2.0 + rng.normal(scale=0.1, size=n)
    return X, y, [f"f{i}" for i in range(p)]


def test_no_cap_for_fast_predict():
    """dt≈0 → est ≪ budget → full p features scored, signal ranks top."""
    X, y, cols = _data()
    fast = lambda Z: Z[:, 3] * 2.0
    r = _permutation_importance(fast, X, y, cols, time_budget=90.0)
    assert r[0][0] == "f3"
    assert len(r) == len(cols)


def test_cap_triggers_for_slow_predict():
    """dt·p·n_repeats > budget → n_repeats=1 + top-K(=max(5,budget/dt)) features."""
    X, y, cols = _data()
    slow = lambda Z: (time.sleep(0.005), Z[:, 3] * 2.0)[1]
    r = _permutation_importance(slow, X, y, cols, time_budget=0.0005)
    nz = sum(1 for _, v in r if abs(v) > 1e-9)
    assert nz <= 5  # budget/dt ≈ 0 → floor max_feats=5


def test_cap_preserves_top_signal_feature():
    """High-variance signal feature survives the variance-based top-K cut."""
    X, y, cols = _data()
    slow = lambda Z: (time.sleep(0.005), Z[:, 3] * 2.0)[1]
    r = _permutation_importance(slow, X, y, cols, time_budget=0.0005)
    assert r[0][0] == "f3"


def test_predict_fail_returns_empty():
    X, y, cols = _data()
    bad = lambda Z: (_ for _ in ()).throw(RuntimeError("boom"))
    assert _permutation_importance(bad, X, y, cols) == []


class _InnerNoFeat:
    USES_FEATURES = False


class _InnerFeat:
    USES_FEATURES = True


class _Art:
    def __init__(self, inner):
        self.model = inner


def test_part_a_skip_predicate_x_independent():
    """_explain_one branch reads USES_FEATURES from the inner forecaster (art.model)."""
    skip = lambda art: not getattr(getattr(art, "model", None), "USES_FEATURES", True)
    assert skip(_Art(_InnerNoFeat())) is True    # X-independent → skip
    assert skip(_Art(_InnerFeat())) is False     # uses features → run permutation
    assert skip(_Art(None)) is False             # no inner model → default to run
