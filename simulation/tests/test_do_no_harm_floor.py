"""PART E (transform-fix reconciliation, 2026-06-21): leak-free do-no-harm floor.

_do_no_harm_select compares the R9-optimized config against an identity baseline on the most-recent
TAIL of the train+val pool (NEVER the sealed test / y_test), and falls back to the baseline config
if the baseline beats R9 by MPH_DO_NO_HARM_MARGIN (default 0.05). It skips ensembles + pools too
small to evaluate, and logs tail_max vs train_max (vs test context if available) so the
extrapolation-proxy quality is auditable. It augments the existing G-328c baseline-floor.

The scoring is injected (score_fn) so the unit test exercises the SELECTION logic without fitting
real models — matching/mismatch/empty/edge cases per the smoke-test standard.
"""
from __future__ import annotations

import numpy as np


def _pool(n=60):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, 5))
    y = np.maximum(5 + 3 * np.sin(np.arange(n) / 8.0) + rng.normal(0, 0.5, n), 0.1)
    return X, y


def test_falls_back_when_baseline_beats_r9_on_tail():
    from simulation.pipeline.per_model_optimize import _do_no_harm_select
    X, y = _pool()
    r9 = {"transform": "log1p", "scaler": "robust", "preproc_optuna_params": {"y_mode": "individual"}}
    base = {"transform": "identity", "scaler": "none", "preproc_optuna_params": None}
    # baseline WIS clearly lower (better) than R9 → fall back
    score_fn = lambda cfg, *_: 1.0 if cfg is base else 2.0
    chosen, fell_back = _do_no_harm_select(
        "NegBinGLM", r9, base, X, y, X, y,
        score_fn=score_fn, tail_frac=0.25)
    assert fell_back is True
    assert chosen is base


def test_keeps_r9_when_r9_better_or_within_margin():
    from simulation.pipeline.per_model_optimize import _do_no_harm_select
    X, y = _pool()
    r9 = {"transform": "log1p", "scaler": "robust"}
    base = {"transform": "identity", "scaler": "none"}
    # R9 better → keep R9
    score_fn = lambda cfg, *_: 1.0 if cfg is r9 else 2.0
    chosen, fell_back = _do_no_harm_select(
        "NegBinGLM", r9, base, X, y, X, y, score_fn=score_fn, tail_frac=0.25)
    assert fell_back is False and chosen is r9

    # baseline better but only within the margin (default 0.05 = 5%) → keep R9 (overfit guard)
    score_fn2 = lambda cfg, *_: (0.98 if cfg is base else 1.0)  # 2% better, < 5% margin
    chosen2, fb2 = _do_no_harm_select(
        "NegBinGLM", r9, base, X, y, X, y, score_fn=score_fn2, tail_frac=0.25,
        margin=0.05)
    assert fb2 is False and chosen2 is r9


def test_skips_ensembles():
    from simulation.pipeline.per_model_optimize import _do_no_harm_select
    X, y = _pool()
    r9 = {"transform": "log1p"}
    base = {"transform": "identity"}
    calls = []
    score_fn = lambda cfg, *_: (calls.append(cfg), 1.0)[1]
    chosen, fell_back = _do_no_harm_select(
        "Ensemble-NNLS", r9, base, X, y, X, y, score_fn=score_fn, tail_frac=0.25)
    assert chosen is r9 and fell_back is False
    assert calls == [], "ensembles must be skipped (no scoring)"


def test_skips_when_pool_too_small():
    from simulation.pipeline.per_model_optimize import _do_no_harm_select
    rng = np.random.default_rng(1)
    X = rng.standard_normal((6, 5)); y = np.abs(rng.normal(5, 1, 6))
    r9 = {"transform": "log1p"}
    base = {"transform": "identity"}
    score_fn = lambda cfg, *_: 1.0
    chosen, fell_back = _do_no_harm_select(
        "NegBinGLM", r9, base, X, y, X, y, score_fn=score_fn, tail_frac=0.25, min_tail=8)
    assert chosen is r9 and fell_back is False


def test_never_uses_test_y_and_logs_tail_context(caplog):
    """Leak guard: the helper must only score on the train+val tail, and must log tail_max/train_max."""
    import logging
    from simulation.pipeline.per_model_optimize import _do_no_harm_select
    X, y = _pool()
    seen_lengths = []
    # score_fn receives (cfg, Xtr, ytr, Xev, yev); record the eval-window size it is asked to score
    def score_fn(cfg, Xtr, ytr, Xev, yev):
        seen_lengths.append(len(yev))
        return 1.0
    with caplog.at_level(logging.INFO):
        _do_no_harm_select("NegBinGLM", {"transform": "log1p"}, {"transform": "identity"},
                           X, y, X, y, score_fn=score_fn, tail_frac=0.25)
    # eval window must be the TAIL (a fraction of the pool), never the full pool blindly
    assert all(0 < n < len(y) for n in seen_lengths), "eval window must be a strict tail subset"
    # audit log line present
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "tail_max" in text and "train_max" in text, "must log tail_max vs train_max for audit"
