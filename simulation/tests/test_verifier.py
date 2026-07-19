"""Smoke tests: verifier decorators, AST checker, epi validity."""
from __future__ import annotations

import numpy as np
import pytest


# ══════════════════════════════════════════════════════════════════════════
# Decorators
# ══════════════════════════════════════════════════════════════════════════
def test_verify_before_runs_checker():
    from simulation.verifier import verify_before
    from simulation.verifier.decorators import CheckerResult

    calls = []

    def my_checker(*args, **kwargs) -> CheckerResult:
        calls.append(args)
        return CheckerResult(status="ok", checker="my_checker", details={})

    @verify_before("test_phase", checkers=[my_checker], persist=False)
    def add(a, b):
        return a + b

    assert add(2, 3) == 5
    assert len(calls) == 1


def test_verify_after_sees_output():
    from simulation.verifier import verify_after
    from simulation.verifier.decorators import CheckerResult

    seen = []

    def check_output(*args, out=None, **kwargs) -> CheckerResult:
        seen.append(out)
        return CheckerResult(status="ok", checker="check_output", details={})

    @verify_after("test_phase", checkers=[check_output], persist=False)
    def double(x):
        return x * 2

    assert double(5) == 10
    assert seen == [10]


def test_verifier_error_on_fail():
    from simulation.verifier import verify_before, VerifierError
    from simulation.verifier.decorators import CheckerResult

    def always_fail(*args, **kwargs) -> CheckerResult:
        return CheckerResult(status="fail", checker="always_fail", details={})

    @verify_before("test", checkers=[always_fail], persist=False, on_fail="raise")
    def fn():
        return 1

    with pytest.raises(VerifierError):
        fn()


# ══════════════════════════════════════════════════════════════════════════
# AST checker
# ══════════════════════════════════════════════════════════════════════════
def test_ast_checker_detects_sqlite3_connect():
    from simulation.verifier import AstChecker
    src = """
import sqlite3
conn = sqlite3.connect("foo.db")
"""
    report = AstChecker().scan_source(src, filepath="user_code.py")
    names = {v.pattern for v in report.violations}
    assert "sqlite3_connect_bypass" in names


def test_ast_checker_detects_n_jobs_minus_one():
    from simulation.verifier import AstChecker
    src = """
from sklearn.ensemble import RandomForestRegressor
m = RandomForestRegressor(n_jobs=-1)
"""
    report = AstChecker().scan_source(src, filepath="mod.py")
    assert any(v.pattern == "n_jobs_minus_one" for v in report.violations)


def test_ast_checker_detects_bare_except():
    from simulation.verifier import AstChecker
    src = """
try:
    x = 1
except:
    pass
"""
    report = AstChecker().scan_source(src)
    assert any(v.pattern == "bare_except" for v in report.violations)


def test_ast_checker_allows_safe_connect():
    from simulation.verifier import AstChecker
    src = """
from simulation.database import safe_connect
conn = safe_connect()
"""
    report = AstChecker().scan_source(src, filepath="user_code.py")
    assert report.n_fail == 0


def test_ast_checker_flags_fit_transform_at_module_level():
    """Module-level fit_transform on ambiguous variable → flagged (real leakage pattern)."""
    from simulation.verifier import AstChecker
    src = """
from sklearn.preprocessing import StandardScaler
X = [[1, 2], [3, 4]]
X_scaled = StandardScaler().fit_transform(X)
"""
    report = AstChecker().scan_source(src, filepath="leaky.py")
    assert any(v.pattern == "fit_transform_before_split" for v in report.violations)


def test_ast_checker_exempts_fit_transform_inside_fit_method():
    """fit_transform inside .fit() / _pretrain / train is train-only by contract."""
    from simulation.verifier import AstChecker
    src = """
from sklearn.preprocessing import StandardScaler

class MyModel:
    def fit(self, X, y):
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        return self

    def _pretrain_on_overseas(self, X_overseas, y_overseas):
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_overseas)
        return X_s

    def train_one_epoch(self, X_batch):
        sc = StandardScaler()
        return sc.fit_transform(X_batch)
"""
    report = AstChecker().scan_source(src, filepath="model.py")
    # All three calls are inside train-scoped methods → no leakage flag
    leakage_violations = [v for v in report.violations
                          if v.pattern == "fit_transform_before_split"]
    assert len(leakage_violations) == 0, (
        f"Expected 0 leakage flags inside fit/pretrain/train methods, "
        f"got {len(leakage_violations)}: {leakage_violations}"
    )


# ══════════════════════════════════════════════════════════════════════════
# Epi validity
# ══════════════════════════════════════════════════════════════════════════
def test_epi_validity_accepts_reasonable_R0():
    from simulation.verifier import check_epi_validity
    res = check_epi_validity(params={"R0": 1.3, "gamma": 0.25, "sigma": 0.5})
    assert res.status == "ok"


def test_epi_validity_rejects_huge_R0():
    from simulation.verifier import check_epi_validity
    res = check_epi_validity(params={"R0": 50.0})
    assert res.status == "fail"


def test_epi_validity_catches_negative_predictions():
    from simulation.verifier import check_epi_validity
    preds = np.array([1.0, -0.5, 2.0, 3.0])
    res = check_epi_validity(predictions=preds)
    assert res.status == "fail"


def test_epi_validity_catches_nan_predictions():
    from simulation.verifier import check_epi_validity
    preds = np.array([1.0, np.nan, 2.0])
    res = check_epi_validity(predictions=preds)
    assert res.status == "fail"
