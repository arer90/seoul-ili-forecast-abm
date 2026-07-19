"""
simulation.verifier.leakage_per_fold
=====================================
Per-fold leakage hook (§15.3, RECOMMENDED_PIPELINE.md ).

Wraps the existing simulation.models.leakage_checker.LeakageChecker so
every walk-forward fold fires the same 6-check battery automatically,
and every result is persisted into `verifier_audit`.

Design rationale:
 The existing `LeakageChecker` is well-tested (models/leakage_checker.py,
 lines 70-264), so we do NOT reimplement it. We wrap it + the
 decorators.CheckerResult protocol so it plugs into @verify_after.

Usage (inside expanding_cv or phase6_wfcv)::

 from simulation.verifier.leakage_per_fold import LeakagePerFoldHook

 hook = LeakagePerFoldHook

 @verify_after("phase7_fold", checkers=[hook], on_fail="warn")
 def run_fold(X_tr, X_te, y_tr, y_te, feature_names, out=None):
 ...

Or call directly::

 result = hook(X_tr=..., X_te=..., y_tr=..., y_te=..., feature_names=[...])
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from .decorators import CheckerResult

log = logging.getLogger(__name__)


class LeakagePerFoldHook:
    """Callable hook that runs LeakageChecker on a single fold."""

    def __init__(self, *, min_level: str = "WARNING"):
        """
        Parameters
        ----------
        min_level : str
            Only propagate violations at this level or higher.
            Levels: "INFO" < "WARNING" < "CRITICAL"
        """
        self.min_level = min_level

    def __call__(self, *args, out: Any = None, **kwargs) -> CheckerResult:
        """
        Accepted kwargs (any of the following naming conventions):
            X_train / X_tr / X
            X_test  / X_te
            y_train / y_tr / y
            y_test  / y_te
            feature_names / features
        """
        X_train = _first_of(kwargs, "X_train", "X_tr")
        X_test  = _first_of(kwargs, "X_test", "X_te")
        y_train = _first_of(kwargs, "y_train", "y_tr")
        y_test  = _first_of(kwargs, "y_test", "y_te")
        feature_names = _first_of(kwargs, "feature_names", "features")

        # Fallback: if args were passed positionally in a known order
        if X_train is None and len(args) >= 4:
            X_train, X_test, y_train, y_test = args[:4]
            if len(args) >= 5:
                feature_names = args[4]

        if any(v is None for v in (X_train, X_test, y_train, y_test)):
            return CheckerResult(
                status="warn",
                checker="leakage_per_fold",
                details={"skipped": "insufficient kwargs",
                         "have": list(kwargs.keys())},
            )

        try:
            from simulation.models.leakage_checker import LeakageChecker
        except Exception as e:
            return CheckerResult(
                status="warn",
                checker="leakage_per_fold",
                details={"skipped": "LeakageChecker unavailable",
                         "error": str(e)},
            )

        feats = list(feature_names) if feature_names is not None else [
            f"f{i}" for i in range(np.asarray(X_train).shape[1])
        ]
        checker = LeakageChecker(
            X_train=np.asarray(X_train),
            X_test=np.asarray(X_test),
            y_train=np.asarray(y_train),
            y_test=np.asarray(y_test),
            feature_names=feats,
        )
        try:
            report = checker.run_all_checks()
        except Exception as e:
            return CheckerResult(
                status="warn",
                checker="leakage_per_fold",
                details={"run_all_checks_failed": str(e)},
            )

        # Determine status from report severity
        if report.n_critical > 0:
            status = "fail"
        elif report.n_warning > 0:
            status = "warn"
        else:
            status = "ok"

        details = {
            "n_critical": int(report.n_critical),
            "n_warning": int(report.n_warning),
            "n_info": int(report.n_info),
            "passed": bool(report.passed),
            "findings": [
                {
                    "level": w.level,
                    "category": w.category,
                    "feature": w.feature,
                    "message": w.message,
                    "value": float(w.value),
                }
                for w in report.warnings
                if _level_rank(w.level) >= _level_rank(self.min_level)
            ],
        }
        return CheckerResult(status=status, checker="leakage_per_fold", details=details)


def _first_of(d: dict, *keys: str) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _level_rank(level: str) -> int:
    return {"INFO": 0, "WARNING": 1, "CRITICAL": 2}.get(level.upper(), 0)
