"""Canonical R/P pipeline phase registry — single source of truth (SSOT).

Replaces the old non-sequential magic phase numbers (1, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16
with real_eval=12 running *after* 14) with an explicit, ordered **R(research) / P(production)** layout.

- **R (research)** track: frozen-split training, selection, evaluation, reporting.
- **P (production)** track: deployment + operational forecasting (runs AFTER all research).

The ORDER of ``PHASES`` defines dispatch/resume ordering via list index — there are no magic
numbers in the dispatch logic any more. ``resolve()`` still accepts a legacy number or the old
semantic name so existing scripts / ``--resume-from`` keep working during the transition.

Design discipline: this is a deep module — small interface (``order``/``resolve``/``label_of``/
``name_of``/``track_of``), rich internal lookup. Callers ask by label ("R9"), semantic name
("per_model_optimize"), or legacy number (13) and get a consistent answer.
"""
from __future__ import annotations

from typing import Optional

# (label, track, semantic_name, is_cli)
#   is_cli = invoked as a standalone CLI, not part of the main `train` dispatch loop.
#
# Phase numbers are GONE — only R/P labels + semantic names exist (no legacy-number resolution).
# ORDER = actual train-dispatch order. real_forecaster (P1) is production-track but still runs
# mid-pipeline (after R10, before R11) because comprehensive_eval(R12) currently consumes it;
# the physical move to the end + decoupling is Phase B.
#
# Production e2e (P1 real → P2 family → P3 ABM → P4 ARIA → P5 web): P2..P5 are declared here so
# the labels exist, but they are wired/orchestrated in Phase B (not yet in the train dispatch).
PHASES: list[tuple[str, str, str, bool]] = [
    ("R1",  "research",   "data",               False),
    ("R2",  "research",   "baseline",           False),
    ("R3",  "research",   "external",           False),
    ("R4",  "research",   "wfcv",               False),
    ("R5",  "research",   "diagnostics",        False),
    ("R6",  "research",   "dm_test",            False),
    ("R7",  "research",   "intervals",          False),
    ("R8",  "research",   "scoring",            False),
    ("R9",  "research",   "per_model_optimize", False),
    ("R10", "research",   "per_model_eval",     False),
    ("R11", "research",   "shap",               False),
    ("R12", "research",   "comprehensive_eval", False),
    ("P1",  "production", "real_forecaster",    False),  # post-R12 (2026-06-20): R-track 전부 뒤 = production track 시작. R9 챔피언 사용 + ABM/ARIA gate. R12는 real_eval 디커플(R9/R10만 보고)
    ("P2",  "production", "family_deploy",      True),   # Phase B (per-family #1 deploy set)
    ("P3",  "production", "abm",                True),   # Phase B (SEIR simulation)
    ("P4",  "production", "aria",               True),   # Phase B (LLM layer)
    ("P5",  "production", "web",                True),   # Phase B (serving/visualization)
    ("Pinf","production", "inference",          True),   # serving CLI for P1
    ("Pov", "production", "overseas",           True),   # auxiliary overseas-ILI CLI
]

# Back-compat aliases: old semantic names → canonical name.
_NAME_ALIASES = {
    "shap_analysis": "shap",
    "xai": "shap",
    "real_eval": "real_forecaster",   # old real_eval → P1 real_forecaster
    "family_curation": "family_deploy",
    "comprehensive_eval": "comprehensive_eval",
}

_BY_LABEL = {p[0].upper(): i for i, p in enumerate(PHASES)}
_BY_NAME = {p[2]: i for i, p in enumerate(PHASES)}


def _index(value) -> int:
    """Resolve a label or semantic name to the canonical ordered index.

    Args:
        value: "R9" | "per_model_optimize" | alias. Phase NUMBERS are no longer accepted.

    Returns:
        0-based index into PHASES.

    Raises:
        KeyError: value is not a known label/name.
    """
    s = str(value).strip()
    if s.upper() in _BY_LABEL:            # R/P labels are case-insensitive (R9 == r9)
        return _BY_LABEL[s.upper()]
    low = s.lower()                       # semantic names are case-insensitive too
    key = _NAME_ALIASES.get(low, low)
    if key in _BY_NAME:
        return _BY_NAME[key]
    raise KeyError(f"unknown phase: {value!r} (use an R/P label or semantic name; numbers removed)")


def order(value) -> int:
    """Ordered position of a phase (label/name/legacy number). Lower = earlier."""
    return _index(value)


def label_of(value) -> str:
    """Canonical R/P label (e.g. 'R9') for any phase reference."""
    return PHASES[_index(value)][0]


def name_of(value) -> str:
    """Canonical semantic name (e.g. 'per_model_optimize') for any phase reference."""
    return PHASES[_index(value)][2]


def track_of(value) -> str:
    """'research' or 'production'."""
    return PHASES[_index(value)][1]


def resolve(value) -> str:
    """Alias of label_of — canonical label for a label/name/legacy number."""
    return label_of(value)


def display(value) -> str:
    """Human/log display, e.g. 'R9 per_model_optimize'."""
    i = _index(value)
    return f"{PHASES[i][0]} {PHASES[i][2]}"


def should_run(phase, resume_from_index: int) -> bool:
    """Dispatch gate: run ``phase`` when resuming from ordered index ``resume_from_index``.

    Replaces the old ``if resume_from <= <magic number>`` gates. A phase runs iff its ordered
    position is at or after the resume point.

    Args:
        phase: label / semantic name / legacy number of the phase being gated.
        resume_from_index: ordered index to resume from (0 = run everything). See
            :func:`resume_index`.

    Returns:
        True if the phase should execute on this (re)run.
    """
    return order(phase) >= int(resume_from_index)


def resume_index(value) -> int:
    """Map a ``--resume-from`` value to an ordered index. None/0/"" → 0 (run all)."""
    if value in (None, 0, "", "0"):
        return 0
    return order(value)


def is_known(value) -> bool:
    try:
        _index(value)
        return True
    except KeyError:
        return False


def all_labels() -> list[str]:
    return [p[0] for p in PHASES]
