"""Order TDD (D1=b): the staged per-model order is preproc → feature (the user's order).

`test_d2_feature_mc_order` already covers the ②feature ↔ ③mc sub-order empirically. This
file pins the ①preproc → ②feature ordering inside `_preproc_first_select` as a RUNNABLE
assertion (it was previously only verified by codex+gemini code-tracing + docs §3-bis):
Stage-1 preproc MUST run before Stage-2 feature, and the feature subset MUST be searched
*after* the preproc is fixed — never the reverse.

macOS: run PER-FILE.
"""
import numpy as np
import pytest


def test_preproc_first_select_runs_preproc_then_feature(monkeypatch):
    import simulation.pipeline._inline_optuna_3stage as m

    calls: list[str] = []

    def _fake_preproc(*a, **k):
        calls.append("preproc")
        # mimic _stage1_preproc_optuna_inline return: (best_cell, trial_results)
        return ({"transform": "identity", "scaler": "none",
                 "preproc_optuna_params": {}}, [])

    def _fake_feature(*a, **k):
        calls.append("feature")
        # mimic _stage2_feature_optuna_per_model return: (feature_indices, meta)
        return ([0, 1, 2], {"n_selected": 3})

    monkeypatch.setattr(m, "_stage1_preproc_optuna_inline", _fake_preproc)
    monkeypatch.setattr(m, "_stage2_feature_optuna_per_model", _fake_feature)

    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 6))
    y = X[:, 0] + rng.normal(0, 0.3, 120)
    cols = [f"f{i}" for i in range(6)]

    best, feat = m._preproc_first_select(
        "M", lambda: None, X[:80], y[:80], X[80:], y[80:],
        feature_cols=cols, n_trials_preproc=1, n_trials_feature=1,
    )

    # THE ORDER: preproc first, then feature (never feature→preproc):
    assert calls == ["preproc", "feature"], (
        f"staged order must be preproc→feature, got {calls}")
    # and it returns the preproc choice + the feature subset it then selected:
    assert isinstance(best, dict) and "transform" in best
    assert feat == [0, 1, 2]


def test_d2_d3_d4_order_tests_present():
    """Sanity: the other order/criterion TDDs exist (so 'order TDD' is the full set)."""
    import importlib.util
    from pathlib import Path
    base = Path(__file__).parent
    for name in ("test_d2_feature_mc_order", "test_d3_mc_family_vs_global",
                 "test_d4_preproc_oof_blind"):
        assert (base / f"{name}.py").exists(), f"missing order/criterion TDD: {name}"
