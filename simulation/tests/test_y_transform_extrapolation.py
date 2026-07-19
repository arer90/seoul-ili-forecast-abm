"""Y-transform extrapolation guard — forecasting targets MUST extrapolate above train history.

ROOT CAUSE (2026-06-12, G-254): phase-13 preproc Optuna picked the ``rank`` y-transform
for LightGBM (chain=['rank']) and CatBoost (individual rank). ``rank``'s inverse maps
predictions back into the *sorted training values*, so a tree model (whose leaf averages
cannot exceed the training target range) is HARD-CAPPED at the train maximum. The Seoul ILI
test slab peaks at 100.7 while the train pool maxes near ~30 → the model physically cannot
predict the epidemic peak. Worse, the cap sits *below the data mean* for CatBoost (ceiling
19.1 < mean 28.0), so even average-and-above weeks (not just the peak) are under-predicted →
test r2 collapses to 0.313 (LightGBM) / -0.306 (CatBoost), and even the non-peak r2 is bad.

An empirical headroom sweep found THREE train-bounded transforms: ``rank`` (1.06×),
``arcsine_sqrt`` (0.03× — clips ILI to [0,1], destroying all magnitude info), and
``gaussian`` (1.00× — QuantileTransformer→normal, bounded to the train empirical CDF).
A forecasting y-transform must let an additive model reach values meaningfully above the
training maximum; these three cannot. They are removed from the y-target pools (they remain
valid for X features, where bounded scaling is fine).
"""
from __future__ import annotations

import numpy as np
import pytest

from simulation.pipeline.preproc_optuna_hierarchical import (
    _apply_single_y_transform,
    METRIC_Y_TRANSFORMS,
    CATEGORICAL_Y_TRANSFORMS,
)

# A forecasting transform must produce >1.10× the train max when an additive model
# overshoots the transformed training span by 30%. Bounded transforms give ~1.0×.
HEADROOM_THRESHOLD = 1.10

# Transforms that MUST NOT appear in the y-target pools (train-bounded inverse).
KNOWN_BOUNDED = ("rank", "arcsine_sqrt", "gaussian")


def _headroom(name: str, train_max: float = 30.0, seed: int = 42) -> float:
    """Ratio of inverse(overshoot) to train max for a single y-transform.

    Simulates an additive model predicting 30% beyond the transformed training span,
    then inverts. >1.0 means the transform can extrapolate above the training maximum;
    ~1.0 means it is train-bounded (caps predictions, cannot forecast novel peaks).
    """
    rng = np.random.RandomState(seed)
    y = rng.uniform(4.0, train_max, 200)
    yt, inv, _state = _apply_single_y_transform(y.copy(), name)
    yt = np.asarray(yt, dtype=np.float64)
    span = float(yt.max() - yt.min())
    probe = yt.max() + 0.3 * max(span, 1e-9)
    out = float(np.asarray(inv(np.array([probe]))).ravel()[0])
    return out / float(y.max())


@pytest.mark.parametrize("name", METRIC_Y_TRANSFORMS + CATEGORICAL_Y_TRANSFORMS)
def test_every_y_transform_can_extrapolate(name: str) -> None:
    """PROPERTY: every y-target transform in the active pools must extrapolate.

    Red until rank/arcsine_sqrt/gaussian are removed from the pools.
    """
    h = _headroom(name)
    assert h > HEADROOM_THRESHOLD, (
        f"{name!r} is train-bounded (headroom {h:.2f}×) — a tree/additive model cannot "
        f"forecast peaks above training history with it. Remove from the y-target pool."
    )


def test_known_bounded_transforms_excluded_from_y_pools() -> None:
    """GUARD: the three train-bounded transforms are not selectable as y-targets."""
    assert "rank" not in METRIC_Y_TRANSFORMS
    assert "arcsine_sqrt" not in METRIC_Y_TRANSFORMS
    assert "gaussian" not in CATEGORICAL_Y_TRANSFORMS


def test_known_bounded_transforms_really_are_bounded() -> None:
    """CHARACTERIZATION: confirm each excluded transform actually caps (documents WHY)."""
    for name in KNOWN_BOUNDED:
        assert _headroom(name) <= HEADROOM_THRESHOLD, (
            f"{name!r} unexpectedly extrapolates — re-evaluate its exclusion."
        )


def test_rank_destroys_upper_tail_magnitude() -> None:
    """REPRODUCTION (user's concern): rank is magnitude-blind, so it muddles the WHOLE
    upper range, not just the single peak.

    ``rank`` assigns *uniform* spacing to sorted values, so the jump from an average week to
    an epidemic peak looks identical to the jump between two adjacent average weeks — the
    model loses the information that "100 is far above 30". A model minimizing squared error
    in rank space therefore predicts a muted, mid-range rank for every above-average week,
    which inverts to a value near the middle of the training distribution (the 19-30 ceiling
    we observed). ``log1p`` preserves the magnitude gap, so the peak stays separable.

    This is the deterministic core of the bug — no model, no feature extrapolation.
    """
    y = np.array([10.0, 20.0, 28.0, 30.0, 100.0])  # 100 is a far upper-tail outlier
    yt_rank = np.sort(np.asarray(_apply_single_y_transform(y.copy(), "rank")[0]))
    yt_log = np.sort(np.asarray(_apply_single_y_transform(y.copy(), "log1p")[0]))

    # Spacing of the top gap (30→100) relative to the gap just below it (28→30).
    rank_ratio = (yt_rank[-1] - yt_rank[-2]) / (yt_rank[-2] - yt_rank[-3])
    log_ratio = (yt_log[-1] - yt_log[-2]) / (yt_log[-2] - yt_log[-3])

    # rank: ~1× — the 30→100 epidemic jump is compressed to the same step as 28→30.
    assert rank_ratio < 2.0, (
        f"rank should give ~uniform spacing (magnitude-blind), got top-gap ratio {rank_ratio:.1f}"
    )
    # log1p: the epidemic jump is preserved as a much larger step (peak stays separable).
    assert log_ratio > 5.0, (
        f"log1p should preserve the upper-tail magnitude, got top-gap ratio {log_ratio:.1f}"
    )
