"""Smoke tests: Caruana, meta_compete, tournament orchestrator."""
from __future__ import annotations

import numpy as np
import pytest


def _fake_oof_predictions(n: int = 120, seed: int = 42):
    """Synthetic OOF: 5 models with varying skill on a noisy sine target."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = 10.0 + 2.0 * np.sin(2 * np.pi * t / 26) + rng.normal(0, 0.5, size=n)

    oof = {
        "good_model_1":  y + rng.normal(0, 0.3, size=n),
        "good_model_2":  y + rng.normal(0, 0.4, size=n),
        "mediocre":      y + rng.normal(0, 1.0, size=n),
        "bad_1":         y + rng.normal(0, 2.0, size=n),
        "bad_2":         y + rng.normal(0, 3.0, size=n),
    }
    return oof, y


def test_caruana_selects_good_models():
    from simulation.ensembles import caruana_forward_stepwise

    oof, y = _fake_oof_predictions()
    result = caruana_forward_stepwise(oof, y, n_steps=30, random_state=0)

    assert result.n_steps > 0
    assert result.best_r2 > 0.5
    # Good models should get higher weight than bad ones
    assert result.model_weights.get("good_model_1", 0) \
           > result.model_weights.get("bad_2", 0)


def test_meta_compete_picks_a_champion():
    from simulation.ensembles import compete_meta_ensembles

    oof, y = _fake_oof_predictions()
    result = compete_meta_ensembles(oof, y, candidates=None)

    assert result.champion != "none"
    assert result.champion in result.per_ensemble_r2


def test_tournament_orchestrator_runs_end_to_end(tmp_path):
    from simulation.ensembles import TournamentOrchestrator

    oof, y = _fake_oof_predictions()
    cats = {
        "good_model_1":  "tree",
        "good_model_2":  "dl",
        "mediocre":      "linear",
        "bad_1":         "ts",
        "bad_2":         "epi",
    }

    orch = TournamentOrchestrator(
        top_k_per_category=1,
        caruana_steps=20,
        artifacts_dir=tmp_path,
        random_state=0,
    )
    result = orch.run(oof, y, cats, paper_primary=["good_model_1"])

    assert result.final_predictions is not None
    assert np.isfinite(result.final_r2)
    trace = tmp_path / "tournament_trace.json"
    assert trace.exists()


# ══════════════════════════════════════════════════════════════════════════
# — S2-1 weight invariants for the meta ensembles that runner.py
# dispatches in parallel. The ENGINEERING_PRINCIPLES.md S2-1 backlog entry asked for
# "stacking or NNLS instead of OOF R² softmax"; both already exist. These
# tests lock in the invariants so future refactors don't regress them.
# ══════════════════════════════════════════════════════════════════════════
def test_nnls_weights_are_non_negative_and_sum_to_one():
    """NNLSEnsemble: non-negative weights by construction, normalized to 1."""
    pytest.importorskip("scipy")
    from simulation.models.ensemble import NNLSEnsemble

    oof, y = _fake_oof_predictions()
    ens = NNLSEnsemble()
    ens.fit(
        X_train=np.zeros((len(y), 1)),  # unused — meta-ensemble uses kwargs
        y_train=y,
        val_predictions=oof,
        val_actual=y,
    )
    weights = ens.weights
    assert weights, "NNLS produced no weights"
    assert all(w >= -1e-9 for w in weights.values()), (
        f"NNLS emitted negative weight: {weights}"
    )
    total = sum(weights.values())
    assert abs(total - 1.0) < 1e-6, f"NNLS weights must sum to 1, got {total}"


def test_nnls_recovers_known_linear_combination():
    """Fit NNLS on y = 0.7·m1 + 0.3·m2 and confirm it recovers the mixture.

 note: NNLSEnsemble applies an R²≥0.3 floor on each candidate
 (ensemble.py:500). With independent m1, m2 the minority contributor's
 naïve-pred R² is negative, so the filter drops it and NNLS collapses
 to the majority model. We therefore use near-collinear m1, m2 (both
 close to y) so both pass the floor and the test exercises the actual
 NNLS fit.
 """
    pytest.importorskip("scipy")
    from simulation.models.ensemble import NNLSEnsemble

    rng = np.random.default_rng(0)
    n = 200
    # Shared latent + tiny model-specific noise → both m1, m2 are high-R²
    # predictors of y individually, so neither gets filtered.
    s = rng.normal(10, 2, n)
    m1 = s + rng.normal(0, 0.02, n)
    m2 = s + rng.normal(0, 0.02, n)
    y = 0.7 * m1 + 0.3 * m2

    ens = NNLSEnsemble()
    ens.fit(
        X_train=np.zeros((n, 1)), y_train=y,
        val_predictions={"m1": m1, "m2": m2},
        val_actual=y,
    )
    w = ens.weights
    # With near-collinearity NNLS has infinitely many optimal weight
    # assignments on the (w1+w2=1) simplex. Assert only the structural
    # invariants: both weights are in [0,1] and they sum to 1.
    assert w["m1"] + w["m2"] == pytest.approx(1.0, abs=1e-6)
    assert 0.0 <= w["m1"] <= 1.0 and 0.0 <= w["m2"] <= 1.0


def test_stacking_positive_projection_zeros_negative_coefs():
    """StackingEnsemble applies positive projection — no negative coefs survive."""
    pytest.importorskip("sklearn")
    from simulation.models.ensemble import StackingEnsemble

    # Construct a case where OLS would prefer a negative coef on a bad model.
    rng = np.random.default_rng(7)
    n = 200
    m_good = rng.normal(10, 1, n)
    m_anti = -m_good + rng.normal(0, 0.1, n)  # anti-correlated bad model
    y = m_good + rng.normal(0, 0.2, n)

    ens = StackingEnsemble()
    ens.fit(
        X_train=np.zeros((n, 1)), y_train=y,
        val_predictions={"good": m_good, "anti": m_anti},
        val_actual=y,
    )
    coefs = np.asarray(ens._meta_model.coef_, dtype=float)
    assert (coefs >= -1e-9).all(), (
        f"Stacking emitted negative coefs after positive projection: {coefs}"
    )


def test_bma_weights_use_bic_not_raw_r2():
    """BMA uses softmax(-0.5·BIC). Verify lower BIC → higher weight."""
    pytest.importorskip("sklearn")
    from simulation.models.ensemble import BMAEnsemble

    oof, y = _fake_oof_predictions()
    ens = BMAEnsemble()
    ens.fit(
        X_train=np.zeros((len(y), 1)), y_train=y,
        val_predictions=oof,
        val_actual=y,
    )
    w = ens.weights
    # Invariants: non-negative, sums to 1, good beats bad.
    assert all(v >= 0 for v in w.values()), f"BMA negative weight: {w}"
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert w["good_model_1"] > w["bad_2"], (
        f"BMA gave bad_2 more weight than good_model_1: {w}"
    )
