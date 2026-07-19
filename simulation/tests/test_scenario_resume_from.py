"""Guard: a scenario's ``resume_from`` must resolve to the phase its description names.

The defect this catches (found 2026-07-19): ``diagnostics-only`` carried
``resume_from = 7``. ``training_commands.py`` resolves an int straight through
``PHASES[i][0]``, and ``PHASES[7]`` is R8 (scoring) — so the scenario silently
started at R8 and skipped R5 diagnostics, R6 dm_test and R7 intervals, which is
the opposite of what its name and description promise. ``wfcv-only`` had the
same shape: ``6`` resolved to R7, skipping R4, R5 and R6.

Integer indices are the trap: they drift silently whenever PHASES is reordered.
Scenarios now carry R/P labels, which ``training_commands.py`` passes through
unchanged. This test fails if an index creeps back in, or if a label stops
matching the phase its description claims.

Run standalone — macOS needs per-file pytest runs (LightGBM/OpenMP):
    .venv/bin/python -m pytest simulation/tests/test_scenario_resume_from.py -q
"""

import re

import pytest

from simulation.cli._scenarios import SCENARIOS
from simulation.pipeline.phases import PHASES

_LABELS = [p[0] for p in PHASES]
_WITH_RESUME = sorted(k for k, v in SCENARIOS.items() if v.get("resume_from") is not None)


def _resolve(rf):
    """Mirror training_commands.py: int -> PHASES[i][0]; str passes through."""
    if isinstance(rf, int) and 0 <= rf < len(PHASES):
        return PHASES[rf][0]
    return rf


def test_at_least_one_scenario_uses_resume_from():
    """Keeps the suite honest — an empty parametrize would pass vacuously."""
    assert _WITH_RESUME, "no scenario defines resume_from; this guard would be vacuous"


@pytest.mark.parametrize("name", _WITH_RESUME)
def test_resume_from_is_a_label_not_an_index(name):
    rf = SCENARIOS[name]["resume_from"]
    assert isinstance(rf, str), (
        f"{name}: resume_from={rf!r} is an integer index. Indices drift silently when "
        f"PHASES is reordered — use the R/P label instead (e.g. 'R5')."
    )


@pytest.mark.parametrize("name", _WITH_RESUME)
def test_resume_from_is_a_known_phase(name):
    resolved = _resolve(SCENARIOS[name]["resume_from"])
    assert resolved in _LABELS, f"{name}: resume_from resolves to {resolved!r}, not in {_LABELS}"


@pytest.mark.parametrize("name", _WITH_RESUME)
def test_resume_from_matches_its_description(name):
    """The description names a phase; resume_from must land on that phase."""
    scn = SCENARIOS[name]
    resolved = _resolve(scn["resume_from"])
    claimed = re.search(r"\b([RP]\d{1,2})\b", scn.get("desc", ""))
    if claimed is None:
        pytest.skip(f"{name}: description names no phase")
    assert claimed.group(1) == resolved, (
        f"{name}: description says {claimed.group(1)} but resume_from resolves to "
        f"{resolved}. The scenario would start at the wrong phase and silently skip work."
    )


def test_old_integer_values_would_now_fail():
    """The specific regression, pinned: 7 is R8 and 6 is R7, not R5 and R4."""
    assert _resolve(7) == "R8"
    assert _resolve(6) == "R7"
    assert _resolve("R5") == "R5"
    assert _resolve("R4") == "R4"
