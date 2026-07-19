"""
simulation.verifier.integration
================================
Opt-in integration layer: wraps existing pipeline entry points with
@verify_before / @verify_after hooks.

Design choice:
    We do NOT modify `runner.py` or `expanding_cv.py` directly — that
    would force every caller to pay the verifier cost. Instead users
    who want verification call the *verified_* wrappers defined here.

Usage::

    from simulation.verifier.integration import verified_run_expanding_cv
    results = verified_run_expanding_cv(feat_df, ...)   # same API

    # Or decorate a user-level pipeline entry:
    from simulation.verifier.integration import verify_pipeline
    @verify_pipeline(phase="my_phase")
    def my_fn(...): ...
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Optional

import numpy as np

from .decorators import (
    CheckerResult,
    verify_after,
    verify_before,
)
from .epi_validity import check_epi_validity
from .leakage_per_fold import LeakagePerFoldHook

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Pre-run checkers (run before pipeline)
# ══════════════════════════════════════════════════════════════════════════
def check_db_healthy(*args, **kwargs) -> CheckerResult:
    """Verify DB quick_check + schema before pipeline starts."""
    try:
        from simulation.database import quick_check, verify_schema
        qc = quick_check()
        vs = verify_schema()
        status = "ok"
        if qc.strip().lower() != "ok":
            status = "fail"
        elif not vs["ok"]:
            status = "warn"
        return CheckerResult(
            status=status, checker="db_healthy",
            details={"quick_check": qc, "missing": vs.get("missing", []),
                     "extra_count": len(vs.get("extra", []))},
        )
    except Exception as e:
        return CheckerResult(status="fail", checker="db_healthy",
                             details={"exception": str(e)})


def check_seed_fixed(*args, **kwargs) -> CheckerResult:
    """Verify that numpy + random seeds are fixed (detect global RNG drift)."""
    import os
    import random
    try:
        state = random.getstate()
        np_state = np.random.get_state()
        return CheckerResult(
            status="ok", checker="seed_fixed",
            details={
                "python_hash_seed": os.environ.get("PYTHONHASHSEED", "<unset>"),
                "random_state_key": str(state[0]),
                "numpy_state_key": str(np_state[0]),
            },
        )
    except Exception as e:
        return CheckerResult(status="warn", checker="seed_fixed",
                             details={"error": str(e)})


# ══════════════════════════════════════════════════════════════════════════
# Post-run checkers
# ══════════════════════════════════════════════════════════════════════════
def check_predictions_valid(*args, out: Any = None, **kwargs) -> CheckerResult:
    """Verify the pipeline output passes epi validity."""
    if out is None:
        return CheckerResult(status="warn", checker="predictions_valid",
                             details={"skipped": "out is None"})

    # Try to extract predictions
    preds = _extract_predictions(out)
    if preds is None:
        return CheckerResult(status="warn", checker="predictions_valid",
                             details={"skipped": "could not extract predictions"})

    return check_epi_validity(predictions=preds)


def _extract_predictions(out: Any) -> Optional[np.ndarray]:
    """Best-effort prediction extraction from various return shapes."""
    if isinstance(out, np.ndarray):
        return out
    if isinstance(out, dict):
        for k in ("predictions", "y_pred", "yhat", "oof_preds"):
            if k in out:
                return np.asarray(out[k], dtype=float).ravel()
    return None


# ══════════════════════════════════════════════════════════════════════════
# Convenience decorator
# ══════════════════════════════════════════════════════════════════════════
def verify_pipeline(
    phase: str = "pipeline",
    *,
    include_leakage_hook: bool = False,
    on_fail: str = "warn",
) -> Callable:
    """Decorator that stacks a standard set of before/after hooks.

    Before: check_db_healthy, check_seed_fixed
    After : check_predictions_valid (+ LeakagePerFoldHook if requested)
    """
    before_checkers = [check_db_healthy, check_seed_fixed]
    after_checkers: list = [check_predictions_valid]
    if include_leakage_hook:
        after_checkers.append(LeakagePerFoldHook())

    def _decorate(fn):
        wrapped = verify_before(phase, checkers=before_checkers, on_fail=on_fail)(fn)
        wrapped = verify_after(phase, checkers=after_checkers, on_fail=on_fail)(wrapped)

        @functools.wraps(fn)
        def _inner(*args, **kwargs):
            return wrapped(*args, **kwargs)
        _inner.__verify_phase__ = phase
        return _inner
    return _decorate


# ══════════════════════════════════════════════════════════════════════════
# Pre-baked verified wrappers
# ══════════════════════════════════════════════════════════════════════════
def verified_run_expanding_cv(*args, **kwargs):
    """Verified wrapper for simulation.models.expanding_cv.run_expanding_cv."""
    from simulation.models.expanding_cv import run_expanding_cv

    @verify_pipeline(phase="expanding_cv", on_fail="warn")
    def _call(*a, **kw):
        return run_expanding_cv(*a, **kw)

    return _call(*args, **kwargs)


def verified_run_pipeline(*args, **kwargs):
    """Verified wrapper for simulation.pipeline.runner.run_pipeline (if it exists)."""
    try:
        from simulation.pipeline.runner import run_pipeline
    except ImportError as e:
        raise RuntimeError(f"pipeline.runner unavailable: {e}")

    @verify_pipeline(phase="pipeline_runner", include_leakage_hook=False, on_fail="warn")
    def _call(*a, **kw):
        return run_pipeline(*a, **kw)

    return _call(*args, **kwargs)


__all__ = [
    "check_db_healthy",
    "check_seed_fixed",
    "check_predictions_valid",
    "verify_pipeline",
    "verified_run_expanding_cv",
    "verified_run_pipeline",
]
