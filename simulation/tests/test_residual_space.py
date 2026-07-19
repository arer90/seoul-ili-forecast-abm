"""Residual-space guard for R7 (intervals) conformal PI."""
from __future__ import annotations

import numpy as np
import pytest


def test_residual_space_guard_accepts_raw_predictions():
    from simulation.pipeline.intervals import _assert_same_residual_space
    rng = np.random.default_rng(0)
    y = rng.normal(loc=10.0, scale=2.0, size=200)
    preds = {
        "A": y + rng.normal(scale=0.4, size=200),
        "B": y + rng.normal(scale=1.0, size=200),
    }
    diag = _assert_same_residual_space(y, preds, mode="raise")
    for row in diag.values():
        assert row["flag"] == "ok"


def test_residual_space_guard_flags_log_transformed_model():
    """A model that returns log1p(y) instead of y should be flagged."""
    from simulation.pipeline.intervals import _assert_same_residual_space
    rng = np.random.default_rng(1)
    y = np.exp(rng.normal(loc=2.5, scale=0.4, size=200)) + 5.0   # raw ILI-like
    raw_pred = y + rng.normal(scale=0.5, size=200)
    log_pred = np.log1p(raw_pred)                                # wrong space
    preds = {"OK": raw_pred, "Broken": log_pred}
    diag = _assert_same_residual_space(y, preds, mode="warn")
    assert diag["OK"]["flag"] == "ok"
    assert diag["Broken"]["flag"] in ("scale", "location")


def test_residual_space_guard_raises_in_raise_mode():
    from simulation.pipeline.intervals import _assert_same_residual_space
    rng = np.random.default_rng(2)
    y = np.exp(rng.normal(loc=3.0, scale=0.3, size=200))
    bad = np.log1p(y)
    with pytest.raises(ValueError, match="residual-space mismatch"):
        _assert_same_residual_space(y, {"Bad": bad}, mode="raise")


def test_residual_space_guard_off_returns_diag_only():
    """mode='off' must never raise or warn, only compute the diagnostic."""
    from simulation.pipeline.intervals import _assert_same_residual_space
    y = np.arange(100, dtype=float)
    broken = y * 100.0    # wildly different scale
    diag = _assert_same_residual_space(y, {"Bad": broken}, mode="off")
    # scale ratio is ~100 — well outside tol — but no raise
    assert diag["Bad"]["flag"] == "scale"


def test_residual_space_guard_empty_model_tag():
    from simulation.pipeline.intervals import _assert_same_residual_space
    y = np.arange(100, dtype=float)
    preds = {"Empty": np.array([np.nan] * 100)}
    diag = _assert_same_residual_space(y, preds, mode="off")
    assert diag["Empty"] == {"flag": "empty"}


def test_phase6_exposes_residual_space_diag():
    """run_intervals must surface the diagnostic in its return dict."""
    from simulation.pipeline.intervals import run_intervals

    class _Cfg:
        class data: dates = None
        class scoring: residual_space_mode = "off"

    rng = np.random.default_rng(3)
    n = 260
    y = np.sin(np.linspace(0, 12, n)) + rng.normal(scale=0.2, size=n)
    oof = {"M": y + rng.normal(scale=0.3, size=n)}
    ho = 240
    ho_pred = {"M": oof["M"][ho:]}
    r = run_intervals(y, oof, _Cfg(), holdout_predictions=ho_pred, holdout_start=ho)
    assert "residual_space_diag" in r
    assert r["residual_space_diag"]["M"]["flag"] == "ok"
