"""OOF fold aggregation must not discard outbreak folds (G-255).

ROOT CAUSE (2026-06-12, codex+gemini converged): phase-13 OOF-CV selection aggregated
per-fold scores with ``np.median`` (_inline_optuna_3stage.py:130/426, added 2026-05-30 for
single-bad-fold robustness). With 5 expanding walk-forward folds, only ~2 contain an epidemic
peak (fold val maxes ≈ [3.0, 3.4, 6.1, 66.9, 64.1]); the median therefore lands on a low-flu
fold and a *peak-blind* config (great on quiet weeks, terrible on the outbreak folds) wins
selection. The Seoul ILI test slab IS a peak (100.7 > train 66.9), so the median-selected
config collapses on test (LightGBM r2=0.313, CatBoost -0.306).

Fix: aggregate with ``mean`` (every fold counts, outbreak folds included). The original
single-glitch concern is already covered upstream by the finite-score filter (a non-finite
fold is dropped before aggregation), so a finite outlier carries only 1/n_folds weight.
"""
from __future__ import annotations

import numpy as np

from simulation.pipeline._inline_optuna_3stage import _aggregate_oof_folds


# fold val maxes ≈ [3, 3, 6, 67, 64]: 3 quiet folds + 2 outbreak folds.
PEAK_BLIND = [0.5, 0.5, 0.6, 12.0, 11.0]   # aces the quiet folds, fails the outbreak folds
BALANCED = [1.5, 1.5, 1.6, 5.0, 4.5]        # moderate everywhere, far better on the outbreaks


def test_peak_blind_config_loses_to_balanced() -> None:
    """The robust aggregation must rank the balanced (peak-handling) config better (lower)."""
    assert _aggregate_oof_folds(BALANCED) < _aggregate_oof_folds(PEAK_BLIND), (
        "aggregation let a peak-blind config win — it is discarding the outbreak folds"
    )


def test_median_would_have_picked_the_peak_blind_one() -> None:
    """CHARACTERIZATION: documents WHY median was wrong (it ranks peak-blind better)."""
    assert np.median(PEAK_BLIND) < np.median(BALANCED)          # median: peak-blind 'wins' (bug)
    assert np.mean(PEAK_BLIND) > np.mean(BALANCED)              # mean: balanced wins (correct)


def test_non_finite_folds_dropped() -> None:
    """Single-glitch robustness is preserved by dropping non-finite folds before aggregating."""
    assert _aggregate_oof_folds([1.0, float("inf"), 2.0, float("nan")]) == 1.5  # mean(1.0, 2.0)


def test_empty_returns_inf() -> None:
    assert _aggregate_oof_folds([]) == float("inf")
    assert _aggregate_oof_folds([float("nan"), float("inf")]) == float("inf")


# ── G-256b: regime-conditional aggregation (fold_maxes + outbreak_level) ──────
_FOLD_MAXES = [3.0, 4.0, 6.0, 67.0, 64.0]   # 3 quiet folds + 2 elevated (outbreak) folds
_OUTBREAK_LEVEL = 30.0                       # quiet = max ≤ 30, elevated = max > 30


def test_regime_balanced_does_not_ignore_outbreak_folds() -> None:
    """The median bug reborn: a config that aces the quiet folds but fails the outbreak folds
    must NOT win once regimes are balanced (median ranked it best by ignoring the 2 peak folds)."""
    peak_ignorer = [0.5, 0.5, 0.5, 20.0, 20.0]   # great quiet, terrible outbreak
    balanced = [2.0, 2.0, 2.0, 4.0, 4.0]          # moderate everywhere
    # median would pick the peak-ignorer (0.5 < 2.0) — the original collapse:
    assert np.median(peak_ignorer) < np.median(balanced)
    # regime-balanced gives the 2 outbreak folds equal weight to the 3 quiet folds → picks balanced:
    r_ignorer = _aggregate_oof_folds(peak_ignorer, _FOLD_MAXES, _OUTBREAK_LEVEL)
    r_balanced = _aggregate_oof_folds(balanced, _FOLD_MAXES, _OUTBREAK_LEVEL)
    assert r_balanced < r_ignorer, f"regime aggregation ignored the outbreak folds: {r_balanced} vs {r_ignorer}"


def test_regime_balanced_equal_weight_formula() -> None:
    """0.5·mean(quiet) + 0.5·mean(elevated): neither regime dominated by fold count."""
    scores = [1.0, 1.0, 1.0, 5.0, 7.0]   # quiet mean 1.0, elevated mean 6.0
    assert _aggregate_oof_folds(scores, _FOLD_MAXES, _OUTBREAK_LEVEL) == 0.5 * 1.0 + 0.5 * 6.0


def test_regime_falls_back_to_mean_when_single_regime() -> None:
    """All-quiet folds (no elevated) → plain mean (no spurious split)."""
    quiet_only_maxes = [3.0, 4.0, 5.0, 6.0, 7.0]
    scores = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _aggregate_oof_folds(scores, quiet_only_maxes, _OUTBREAK_LEVEL) == 3.0  # mean


def test_regime_falls_back_to_mean_without_context() -> None:
    """No fold_maxes / no outbreak_level → G-255 plain mean (backward compatible)."""
    scores = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _aggregate_oof_folds(scores) == 3.0
    assert _aggregate_oof_folds(scores, _FOLD_MAXES, None) == 3.0
