"""OOF objective explosion guard (G-256c).

User-preferred fix (2026-06-12): instead of HARD-restricting the y-transform pool for
extrapolating models, keep the full pool (including identity) and make the OOF objective SEE a
blow-up — a nonlinear-inverse transform (log1p→expm1) on a linear/NN model overshoots to
100s–1000s× the data. The plain OOF mean averages that rare blow-up away (one bad fold
outweighed by in-range gains), so Optuna keeps picking the exploding transform even though
identity is available. ``_sanity_penalize_wis`` adds a dominating penalty when predictions
exceed ``sanity_mult × train max`` so the exploding config loses selection and Optuna picks a
safe transform on its own. Diagnostic (exp_why_optuna_picks_log1p.py): plain OOF mean MAE
identity 11.62 / log1p 10.97 (Optuna → log1p); sanity-penalized → log1p 2010 (Optuna → identity).
"""
from __future__ import annotations

import numpy as np

from simulation.pipeline.per_model_optimize import (
    _in_range_fold_metrics, _oof_selection_score, _sanity_penalize_wis,
)

TRAIN_MAX = 67.0   # cap at 3× = 201 (a legit epidemic peak ~1.5× = ~100 stays unpenalized)


def test_in_range_predictions_unpenalized() -> None:
    # predictions up to 100 (~1.5× train max = a real peak) must NOT be penalized
    assert _sanity_penalize_wis(5.0, np.array([10.0, 50.0, 100.0]), TRAIN_MAX) == 5.0


def test_exploded_predictions_penalized() -> None:
    w = _sanity_penalize_wis(5.0, np.array([10.0, 50.0, 250.0]), TRAIN_MAX)  # 250 > 3×67=201
    assert w > 1_000.0, "an exploded prediction must dominate the WIS"


def test_penalty_scales_with_explosion_magnitude() -> None:
    mild = _sanity_penalize_wis(5.0, np.array([250.0]), TRAIN_MAX)
    severe = _sanity_penalize_wis(5.0, np.array([990_796.0]), TRAIN_MAX)
    assert severe > mild


def test_penalty_flips_selection_to_identity() -> None:
    """The diagnostic scenario: log1p wins the plain mean but loses once its fold-4 blow-up
    (max pred 201.6) is penalized — so Optuna would pick identity itself."""
    identity_folds = [20.93, 0.84, 4.63, 12.16, 19.52]                  # no explosion
    log1p_folds = [(1.58, 1.3), (0.71, 2.8), (6.89, 40.0),
                   (41.03, 201.6), (4.65, 31.0)]                        # (mae, max_pred)
    id_mean = float(np.mean(identity_folds))
    log_mean = float(np.mean([
        _sanity_penalize_wis(mae, np.array([mx]), TRAIN_MAX) for mae, mx in log1p_folds
    ]))
    assert log_mean > id_mean, f"penalty failed to flip selection: log1p {log_mean} vs identity {id_mean}"


def test_non_finite_predictions_safe() -> None:
    # inf/nan max → guarded (no crash), returns original wis
    assert _sanity_penalize_wis(5.0, np.array([np.inf]), TRAIN_MAX) == 5.0
    assert _sanity_penalize_wis(5.0, np.array([]), TRAIN_MAX) == 5.0


def test_in_range_fold_metrics_exclude_novel_extrapolation_rows() -> None:
    m = _in_range_fold_metrics(
        y_train=np.array([1.0, 2.0, 3.0]),
        y_val=np.array([1.0, 2.0, 4.0]),
        y_pred=np.array([1.0, 2.0, 100.0]),
        residuals=np.array([0.1, -0.1, 0.2]),
        fallback_wis=50.0,
    )
    assert m["n_in_range"] == 2
    assert m["r2_in_range"] == 1.0
    assert m["wis_in_range"] < 50.0


def test_selection_score_prefers_in_range_fit_over_full_wis_noise(monkeypatch) -> None:
    monkeypatch.setenv("MPH_OOF_IN_RANGE_ALPHA", "0.7")
    monkeypatch.setenv("MPH_OOF_IN_RANGE_R2_FLOOR", "0.90")
    monkeypatch.setenv("MPH_OOF_IN_RANGE_R2_PENALTY", "0.25")
    good_in_range = {"wis": 10.0, "wis_in_range": 2.0, "r2_in_range": 0.95, "n_in_range": 12}
    bad_in_range = {"wis": 5.0, "wis_in_range": 20.0, "r2_in_range": 0.0, "n_in_range": 12}
    assert _oof_selection_score(good_in_range) < _oof_selection_score(bad_in_range)
