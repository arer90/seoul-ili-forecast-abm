"""Guard: --resume-from help text must not advertise values the parser rejects.

Phase references moved from numbers to R/P labels, and the argument's help strings
did not all follow. ``simulation/__main__.py`` still offered "번호(예: 13)" as the
example while ``resolve_resume_from("13")`` raises — so the one value the help
showed was the one value guaranteed to fail. The shell wrappers drifted the other
way and claimed numbers were rejected outright, which is also wrong.

The real contract has a seam that makes both mistakes easy:

    "0" / 0          -> 0,    meaning "run every phase" (a sentinel, not a phase number)
    None / ""        -> None, the argparse default — flag omitted entirely
    "R9" / "per_model_optimize" -> that phase
    "9" / "13" / anything else  -> argparse.ArgumentTypeError

``0`` staying valid is not an oversight — ``run_pipeline.sh`` passes ``--resume-from 0``
on an ordinary full run, so rejecting it would break the default path. Note that ``0``
and ``None`` are distinct return values that both mean "start at the beginning";
``resolve_resume_from`` short-circuits None/"" before it ever reaches the sentinel
handling in ``phases.resume_index``.

Run standalone — macOS needs per-file pytest runs (LightGBM/OpenMP):
    .venv/bin/python -m pytest simulation/tests/test_resume_from_argparse.py -q
"""

import argparse
import re
from pathlib import Path

import pytest

from simulation.pipeline import phases
from simulation.pipeline.runner import resolve_resume_from

ROOT = Path(__file__).resolve().parents[2]


# ── behaviour ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("value", ["0", 0])
def test_zero_is_the_run_all_sentinel(value):
    """run_pipeline.sh passes --resume-from 0 on a normal full run."""
    assert resolve_resume_from(value) == 0


@pytest.mark.parametrize("value", [None, ""])
def test_absent_value_stays_none(value):
    """Distinct from 0: the flag was omitted, so argparse's default survives."""
    assert resolve_resume_from(value) is None


@pytest.mark.parametrize("value", ["R9", "r9", " R9 ", "per_model_optimize"])
def test_labels_and_names_are_accepted(value):
    assert resolve_resume_from(value) == phases.order("R9")


@pytest.mark.parametrize("value", ["9", "13", "1", "R99", "bogus", "phase13"])
def test_everything_else_is_rejected(value):
    with pytest.raises(argparse.ArgumentTypeError):
        resolve_resume_from(value)


def test_every_registered_label_resolves():
    for label in phases.all_labels():
        assert resolve_resume_from(label) == phases.order(label)


# ── help text must match behaviour ───────────────────────────────────────────
def _help_strings() -> list[tuple[str, str]]:
    """Every --resume-from help/docstring shipped to a user, as (where, text)."""
    out = []
    for rel in ("simulation/__main__.py", "simulation/pipeline/runner.py"):
        src = (ROOT / rel).read_text(encoding="utf-8")
        for m in re.finditer(
            r'add_argument\(\s*"--resume-from".*?\)', src, re.S
        ):
            out.append((rel, m.group(0)))
    for rel in ("run_pipeline.sh", "scripts/launch_full_run.sh"):
        p = ROOT / rel
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if "--resume-from" in line and line.lstrip().startswith("#"):
                out.append((rel, line))
    return out


def test_help_sites_were_found():
    """A regex that silently matches nothing would make the guard below vacuous."""
    sites = _help_strings()
    assert len(sites) >= 4, f"only found {len(sites)} help sites: {[s[0] for s in sites]}"


@pytest.mark.parametrize("where,text", _help_strings())
def test_help_does_not_advertise_a_rejected_number(where, text):
    """Any bare number shown as an example must be one the parser accepts."""
    for num in re.findall(r"(?<![\w.-])(\d{1,3})(?![\w.-])", text):
        if num == "0":
            continue          # the documented run-all sentinel
        try:
            resolve_resume_from(num)
        except argparse.ArgumentTypeError:
            pytest.fail(
                f"{where}: --resume-from help mentions {num!r}, which the parser "
                f"rejects. A user copying the example gets an argparse error."
            )
