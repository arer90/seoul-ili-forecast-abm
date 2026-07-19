"""Guard: the portability baselines in CI must match what the checker actually reports.

Two defects this pins, both found when CI first ran (2026-07-19):

1. The baseline was hand-counted and wrong. ``posix-tmp`` was recorded as 10
   while the checker reported 11, because the checker was scanning its own
   source — its docstring and regexes necessarily contain the patterns it looks
   for. CI failed on Linux and macOS on the very first run.

2. The gate silently did nothing on Windows. All three checks lived in one
   multi-line ``run:`` block; PowerShell surfaces only the last command's exit
   code, so an early failure was invisible there. Windows reported success on
   the same commit that failed elsewhere. Each check now gets its own step.

This test reads the numbers straight out of the workflow file and compares them
against a live run, so the two cannot drift apart again.

Run standalone — macOS needs per-file pytest runs (LightGBM/OpenMP):
    .venv/bin/python -m pytest simulation/tests/test_portability_baseline.py -q
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "portability.yml"
CHECKER = ROOT / "scripts" / "check_portability.py"


def _declared_baselines() -> dict[str, int]:
    """Parse `--only <check> ... --max <n>` out of the workflow."""
    text = WORKFLOW.read_text(encoding="utf-8")
    found: dict[str, int] = {}
    for m in re.finditer(r"--only\s+(\S+)[^\n]*?--max\s+(\d+)", text):
        found[m.group(1)] = int(m.group(2))
    return found


def _live_count(check: str) -> int:
    out = subprocess.run(
        [sys.executable, str(CHECKER), "--only", check, "--max", "999999"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
    ).stdout
    m = re.search(rf"^\s*{re.escape(check)}\s+(\d+)\s*$", out, re.M)
    assert m, f"could not parse count for {check} from:\n{out}"
    return int(m.group(1))


def test_workflow_exists_and_declares_baselines():
    assert WORKFLOW.exists(), "portability workflow is missing"
    assert _declared_baselines(), "no --only/--max baselines found in the workflow"


@pytest.mark.parametrize("check", sorted(_declared_baselines()))
def test_baseline_is_not_below_reality(check):
    """CI would fail on a clean tree if the declared budget were too tight."""
    declared = _declared_baselines()[check]
    live = _live_count(check)
    assert live <= declared, (
        f"{check}: checker reports {live} but the workflow allows {declared}. "
        f"CI fails on an unchanged tree. Fix the finding or raise the baseline."
    )


@pytest.mark.parametrize("check", sorted(_declared_baselines()))
def test_baseline_is_not_stale(check):
    """A budget far above reality stops catching regressions."""
    declared = _declared_baselines()[check]
    live = _live_count(check)
    assert declared - live <= 5, (
        f"{check}: workflow allows {declared} but only {live} remain. "
        f"Lower the baseline to {live} so new findings are still caught."
    )


def test_each_check_is_its_own_step():
    """A multi-line run: block does not fail the step on Windows PowerShell."""
    text = WORKFLOW.read_text(encoding="utf-8")
    for block in re.findall(r"run:\s*\|(.*?)(?=\n      - |\n\Z)", text, re.S):
        calls = re.findall(r"check_portability\.py", block)
        assert len(calls) <= 1, (
            "a single run: block invokes check_portability.py more than once; "
            "on Windows only the last exit code counts, so earlier failures are "
            "silently ignored. Give each check its own step."
        )


def test_checker_excludes_itself():
    """Its own docstring and regexes contain every pattern it searches for.

    Asserted on behaviour, not on the source text: an earlier version of this
    test grepped for the literal `!= SELF`, so rewriting the exclusion into an
    equivalent form broke the test while the checker still worked correctly.
    """
    for check in _declared_baselines():
        out = subprocess.run(
            [sys.executable, str(CHECKER), "--only", check, "--list", "--max", "999999"],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        ).stdout
        assert "check_portability.py:" not in out, (
            f"{check} reports findings in the checker's own source; it must "
            f"exclude itself, or its docstring and regex literals count as findings"
        )
