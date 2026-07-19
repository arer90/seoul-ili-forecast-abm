"""A (per-model mc apply): the user wants multicollinearity handled PER-MODEL, not one
global method for all 53. The ④ comparison already MEASURES per-model OOF WIS; this wires a
margin-guarded selection so each model uses its OWN best mc — but only when the benefit is
clear (an overfit guard at n≈349, since the mc choice itself can overfit a noisy 2-fold OOF).
'none' (no filter) is the simplest option and is preferred on ties / unclear benefit.

D3 refuted clean FAMILY rules (mc effect is data-dependent) → the selection must be
data-measured, not a tree/linear heuristic. This tests the data-measured margin guard.

macOS: run PER-FILE.
"""
import numpy as np
import pytest


def _rows():
    """④ comparison rows: model × method → oof_wis (the CSV schema)."""
    return [
        # clear: vif beats none by 20% → choose vif
        {"model": "A", "method": "none", "oof_wis": "1.000"},
        {"model": "A", "method": "vif", "oof_wis": "0.800"},
        {"model": "A", "method": "corr", "oof_wis": "0.950"},
        {"model": "A", "method": "pca", "oof_wis": "1.200"},
        # marginal: best (corr) beats none by only 0.5% → guard → none
        {"model": "B", "method": "none", "oof_wis": "1.000"},
        {"model": "B", "method": "vif", "oof_wis": "1.010"},
        {"model": "B", "method": "corr", "oof_wis": "0.995"},
        {"model": "B", "method": "pca", "oof_wis": "1.100"},
        # none is genuinely best → none
        {"model": "C", "method": "none", "oof_wis": "0.700"},
        {"model": "C", "method": "vif", "oof_wis": "0.900"},
        {"model": "C", "method": "pca", "oof_wis": "1.500"},
    ]


def test_clear_benefit_picks_per_model_method():
    from simulation.pipeline.per_model_optimize import _per_model_mc_choice
    assert _per_model_mc_choice(_rows(), "A", fallback="none", rel_margin=0.02) == "vif"


def test_marginal_benefit_guards_to_none():
    """corr beats none by 0.5% < 2% margin → keep 'none' (don't overfit the mc choice)."""
    from simulation.pipeline.per_model_optimize import _per_model_mc_choice
    assert _per_model_mc_choice(_rows(), "B", fallback="none", rel_margin=0.02) == "none"


def test_none_best_returns_none():
    from simulation.pipeline.per_model_optimize import _per_model_mc_choice
    assert _per_model_mc_choice(_rows(), "C", fallback="vif", rel_margin=0.02) == "none"


def test_unknown_model_uses_fallback():
    """A model absent from the ④ rows falls back to the global choice."""
    from simulation.pipeline.per_model_optimize import _per_model_mc_choice
    assert _per_model_mc_choice(_rows(), "ZZZ", fallback="corr", rel_margin=0.02) == "corr"


def test_per_model_diverges_across_models():
    """The whole point: different models get different mc (not one global method)."""
    from simulation.pipeline.per_model_optimize import _per_model_mc_choice
    rows = _rows()
    chosen = {m: _per_model_mc_choice(rows, m, fallback="none", rel_margin=0.02)
              for m in ("A", "B", "C")}
    assert chosen == {"A": "vif", "B": "none", "C": "none"}
    assert len(set(chosen.values())) >= 2, "per-model selection collapsed to one method"


def test_apply_mc_columns_none_is_passthrough():
    from simulation.pipeline.per_model_optimize import _apply_mc_columns
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 5))
    cols = [f"f{i}" for i in range(5)]
    Xtr, Xva, Xte, Xreal, fc, state, meta = _apply_mc_columns(
        "none", X[:40], X[40:50], X[50:], X[:10], np.arange(40.0), cols)
    assert Xtr.shape[1] == 5 and fc == cols and state is None
    assert Xreal.shape[1] == 5


def test_apply_mc_columns_vif_drops_collinear_and_remaps_real():
    from simulation.pipeline.per_model_optimize import _apply_mc_columns
    rng = np.random.default_rng(1)
    base = rng.normal(size=(80, 1))
    # 3 near-collinear (base, ~base, 2·base) + 2 independent
    X = np.hstack([base, base + 1e-6 * rng.normal(size=(80, 1)), 2 * base,
                   rng.normal(size=(80, 2))])
    cols = [f"f{i}" for i in range(5)]
    Xtr, Xva, Xte, Xreal, fc, state, meta = _apply_mc_columns(
        "vif", X[:60], X[60:70], X[70:], X[:10], np.arange(60.0), cols)
    assert Xtr.shape[1] < 5, "vif must drop collinear columns"
    assert len(fc) == Xtr.shape[1], "feature_cols must track kept columns"
    assert Xreal.shape[1] == Xtr.shape[1], "X_real must be remapped to the same kept columns"
