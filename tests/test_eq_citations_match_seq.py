"""Guard: every "Equation (3.x)" cited in the prose must be an equation that exists.

The thesis numbers its display equations with Word ``SEQ Equation`` auto-fields, but
the "(3." prefix and ")" suffix are literal text and no bookmark wraps any field — so
a cross-reference typed into the prose is *static text*. Insert or delete one display
equation and Word silently renumbers the fields while the typed citations keep pointing
at the old numbers.

This guard closes that gap: it reads the numbers actually printed next to the display
equations, reads every citation in the body prose, and asserts the citations resolve.
It is the same class of check as ``tests/test_docx_numbers_match_results.py`` — catch a
silent drift that renders correctly and reads plausibly while being wrong.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

docx = pytest.importorskip("docx")

_ROOT = Path(__file__).resolve().parents[1]
_DOCX = _ROOT / "paper" / "고려대_보건석사학위논문_이승진_2026.docx"

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_MATH = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"

# "Equation (3.13)" / "Equations (3.7)–(3.12)" — en-dash or hyphen ranges.
_CITE = re.compile(r"Equations?\s+\((\d+)\.(\d+)\)(?:\s*[–-]\s*\((\d+)\.(\d+)\))?")
_LABEL = re.compile(r"\((\d+)\.(\d+)\)")


def _paragraph_text(p) -> str:
    return "".join(t.text or "" for t in p.iter(f"{_W}t"))


def _load():
    if not _DOCX.exists():
        pytest.skip(f"{_DOCX.name} absent")
    d = docx.Document(str(_DOCX))
    numbered: set[tuple[int, int]] = set()
    cited: list[tuple[int, int]] = []
    for p in d.element.body.iter(f"{_W}p"):
        text = _paragraph_text(p)
        has_math = p.find(f".//{_MATH}oMath") is not None
        # A numbered display equation = a paragraph holding OMML whose plain text is
        # exactly the label, e.g. "(3.7)".
        if has_math:
            label = _LABEL.fullmatch(text.strip())
            if label:
                numbered.add((int(label.group(1)), int(label.group(2))))
                continue
        for match in _CITE.finditer(text):
            cited.append((int(match.group(1)), int(match.group(2))))
            if match.group(3):
                cited.append((int(match.group(3)), int(match.group(4))))
    return numbered, cited


def test_every_cited_equation_exists():
    numbered, cited = _load()
    assert cited, "no 'Equation (x.y)' citations found — the numbered equations are orphaned again"
    missing = sorted({c for c in cited if c not in numbered})
    assert not missing, (
        "prose cites equations that do not exist: "
        + ", ".join(f"({a}.{b})" for a, b in missing)
        + f"\nnumbered equations present: {sorted(numbered)}"
    )


def test_citation_ranges_are_ascending():
    """'Equations (3.7)–(3.12)' must not be inverted or empty."""
    if not _DOCX.exists():
        pytest.skip(f"{_DOCX.name} absent")
    d = docx.Document(str(_DOCX))
    bad = []
    for p in d.element.body.iter(f"{_W}p"):
        for match in _CITE.finditer(_paragraph_text(p)):
            if not match.group(3):
                continue
            lo = (int(match.group(1)), int(match.group(2)))
            hi = (int(match.group(3)), int(match.group(4)))
            if hi <= lo:
                bad.append(match.group(0))
    assert not bad, f"inverted or empty equation ranges: {bad}"
