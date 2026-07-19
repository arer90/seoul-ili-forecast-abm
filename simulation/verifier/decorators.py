"""
simulation.verifier.decorators
==============================
@verify_before / @verify_after — zero-subprocess hook layer.

Core contract (§9.4, RECOMMENDED_PIPELINE.md ):
 1. Decorators wrap *pipeline phase entry points* (phase1_data, phase4_ar,
 expanding_cv, runner.fit_one_model, ...).
 2. Each hook runs a list of lightweight `checker(ctx, args, kwargs, out)`
 callables and writes one row into `verifier_audit` per checker.
 3. Failures raise `VerifierError`; warnings get logged + persisted but
 do not abort the run.
 4. Hooks are opt-in: decorator accepts `checkers=[...]` explicitly so
 we never silently inject behavior.

Threading model:
 A process-local `_CURRENT_RUN_ID` is set by the orchestrator at the
 start of every pipeline run; decorators read it to tag audit rows.
"""
from __future__ import annotations

import functools
import json
import logging
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Run-context (per-thread) — orchestrator sets this once at run start
# ══════════════════════════════════════════════════════════════════════════
_CURRENT_RUN_ID: ContextVar[Optional[str]] = ContextVar("_CURRENT_RUN_ID", default=None)
_PHASE_STACK: ContextVar[list[str]] = ContextVar("_PHASE_STACK", default=[])


def get_current_run_id() -> Optional[str]:
    return _CURRENT_RUN_ID.get()


def set_current_run_id(run_id: Optional[str]) -> None:
    _CURRENT_RUN_ID.set(run_id)


# ══════════════════════════════════════════════════════════════════════════
# Errors
# ══════════════════════════════════════════════════════════════════════════
class VerifierError(RuntimeError):
    """Raised when a `before` or `after` checker returns status='fail'."""


# ══════════════════════════════════════════════════════════════════════════
# Checker protocol (duck-typed)
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class CheckerResult:
    """Return value of a checker callable."""
    status: str               # 'ok' / 'warn' / 'fail'
    checker: str              # human-readable name
    details: dict = field(default_factory=dict)

    def to_audit_row(self, phase: str, hook: str, elapsed_ms: int, run_id: Optional[str]) -> dict:
        import datetime as _dt
        return {
            "ts": _dt.datetime.now().isoformat(),
            "run_id": run_id,
            "phase": phase,
            "hook": hook,
            "checker": self.checker,
            "status": self.status,
            "details_json": json.dumps(self.details, default=_json_default),
            "elapsed_ms": elapsed_ms,
        }


def _json_default(o):
    """numpy / pandas types → JSON-safe."""
    try:
        import numpy as np
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
    except Exception:
        pass
    return str(o)


# ══════════════════════════════════════════════════════════════════════════
# Audit writer (lazy DB connection)
# ══════════════════════════════════════════════════════════════════════════
_AUDIT_LOCK = threading.Lock()


def audit_write(row: dict, db_path: Optional[str] = None) -> None:
    """Persist one verifier_audit row. Idempotent under the UNIQUE
    constraint (none — table is append-only)."""
    try:
        from simulation.database import insert_rows  # lazy
        with _AUDIT_LOCK:
            insert_rows("verifier_audit", [row])
    except Exception as e:
        # Never let audit write kill the pipeline
        log.warning("verifier_audit write failed: %s", e)


# ══════════════════════════════════════════════════════════════════════════
# Decorators
# ══════════════════════════════════════════════════════════════════════════
CheckerFn = Callable[..., CheckerResult]


def verify_before(
    phase: str,
    checkers: Optional[list[CheckerFn]] = None,
    *,
    on_fail: str = "raise",   # 'raise' / 'warn' / 'ignore'
    persist: bool = True,
):
    """Pre-call hook.

    Checker signature::

        def my_checker(*args, **kwargs) -> CheckerResult: ...

    The checker sees the same positional + keyword args as the wrapped
    function.  Use `kwargs` to pass context like `run_id`, `fold_idx`.
    """
    checkers = checkers or []

    def _decorate(fn):
        @functools.wraps(fn)
        def _wrapped(*args, **kwargs):
            _run_hooks(
                phase=phase, hook="before", checkers=checkers,
                args=args, kwargs=kwargs, out=None,
                on_fail=on_fail, persist=persist,
            )
            # Track phase stack so nested calls get correct phase label
            stack = list(_PHASE_STACK.get())
            stack.append(phase)
            token = _PHASE_STACK.set(stack)
            try:
                return fn(*args, **kwargs)
            finally:
                _PHASE_STACK.reset(token)
        _wrapped.__verify_phase__ = phase
        _wrapped.__verify_checkers_before__ = list(checkers)
        return _wrapped
    return _decorate


def verify_after(
    phase: str,
    checkers: Optional[list[CheckerFn]] = None,
    *,
    on_fail: str = "raise",
    persist: bool = True,
):
    """Post-call hook.

    Checker signature includes `out=<return value>`::

        def my_checker(*args, out=None, **kwargs) -> CheckerResult: ...
    """
    checkers = checkers or []

    def _decorate(fn):
        @functools.wraps(fn)
        def _wrapped(*args, **kwargs):
            stack = list(_PHASE_STACK.get())
            stack.append(phase)
            token = _PHASE_STACK.set(stack)
            try:
                out = fn(*args, **kwargs)
            finally:
                _PHASE_STACK.reset(token)
            _run_hooks(
                phase=phase, hook="after", checkers=checkers,
                args=args, kwargs=kwargs, out=out,
                on_fail=on_fail, persist=persist,
            )
            return out
        _wrapped.__verify_phase__ = phase
        _wrapped.__verify_checkers_after__ = list(checkers)
        return _wrapped
    return _decorate


def _run_hooks(
    *,
    phase: str, hook: str,
    checkers: list[CheckerFn],
    args: tuple, kwargs: dict, out: Any,
    on_fail: str, persist: bool,
) -> None:
    run_id = get_current_run_id()
    for ck in checkers:
        t0 = time.perf_counter()
        try:
            if hook == "before":
                res = ck(*args, **kwargs)
            else:
                res = ck(*args, out=out, **kwargs)
            if not isinstance(res, CheckerResult):
                res = CheckerResult(
                    status="ok" if res else "fail",
                    checker=getattr(ck, "__name__", "unnamed"),
                    details={"raw": str(res)},
                )
        except Exception as e:
            res = CheckerResult(
                status="fail",
                checker=getattr(ck, "__name__", "unnamed"),
                details={"exception": str(e), "type": type(e).__name__},
            )
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        if persist:
            audit_write(res.to_audit_row(phase=phase, hook=hook,
                                         elapsed_ms=elapsed_ms, run_id=run_id))

        if res.status == "fail":
            msg = f"[{phase}:{hook}] {res.checker} FAILED: {res.details}"
            if on_fail == "raise":
                raise VerifierError(msg)
            elif on_fail == "warn":
                log.warning(msg)
            # on_fail == "ignore": do nothing
        elif res.status == "warn":
            log.warning("[%s:%s] %s WARN: %s", phase, hook, res.checker, res.details)
