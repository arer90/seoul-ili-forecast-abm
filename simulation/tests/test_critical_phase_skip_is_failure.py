"""Guard: a critical phase that skips involuntarily must not report success.

The fail-loud gate was written to stop a voided run from printing
"Pipeline Complete!". It checked one shape only — ``{"error": ...}`` — so a
critical phase that returned ``{"skipped": True, "reason": ...}`` instead
sailed straight through it.

That is reachable, and the way you reach it is by restarting an interrupted
run. ``all_results`` starts empty at each launch and is never rehydrated from
the phase checkpoints on disk, so resuming at R10 leaves the SSOT evaluation
with no predictions to score. It returns ``{"skipped": True, "reason": "no
predictions"}``, the gate finds no ``"error"`` key, and the run reports
success having evaluated nothing.

The one skip that is legitimate is ``reason="disabled"`` — the operator did not
ask for that phase. Everything else means the phase intended to run and could
not, which is exactly the silent void the gate exists to catch.

Run standalone — macOS needs per-file pytest runs (LightGBM/OpenMP):
    .venv/bin/python -m pytest simulation/tests/test_critical_phase_skip_is_failure.py -q
"""

import pytest

from simulation.pipeline.runner import _CRITICAL_PHASES, _collect_critical_failures


def test_clean_run_reports_no_failure():
    ok = {k: {"elapsed": 1.0} for k in _CRITICAL_PHASES}
    assert _collect_critical_failures(ok) == []


def test_error_is_still_caught():
    """The original behaviour must not regress."""
    res = {"per_model_eval": {"error": "boom"}}
    failures = _collect_critical_failures(res)
    assert len(failures) == 1
    assert "boom" in failures[0][1]


@pytest.mark.parametrize("reason", [
    "no predictions",          # resume at R10 with an empty all_results
    "no test predictions",
    "filter excluded all",
    "n_test=4 < 10",
    "factory_unavailable: ImportError",
])
def test_involuntary_skip_of_a_critical_phase_is_a_failure(reason):
    res = {"per_model_eval": {"skipped": True, "reason": reason}}
    failures = _collect_critical_failures(res)
    assert failures, (
        f"a critical phase skipped with reason={reason!r} passed the fail-loud "
        f"gate — the run would print 'Pipeline Complete!' having evaluated nothing"
    )
    assert reason in failures[0][1]


def test_deliberately_disabled_phase_is_not_a_failure():
    """`--per-model-optimize` off is an operator choice, not a void."""
    res = {"per_model_optimize": {"skipped": True, "reason": "disabled"}}
    assert _collect_critical_failures(res) == []


def test_every_critical_phase_is_covered():
    """The gate must apply to all of P1/R9/R10, not just the one that regressed."""
    for key in _CRITICAL_PHASES:
        res = {key: {"skipped": True, "reason": "no predictions"}}
        assert _collect_critical_failures(res), f"{key} not gated"


def test_non_critical_phase_may_skip_freely():
    res = {"diagnostics": {"skipped": True, "reason": "no predictions"}}
    assert _collect_critical_failures(res) == []
