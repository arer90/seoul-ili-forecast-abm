#!/usr/bin/env python3
"""Page-lock guard for the thesis docx — total pages AND every chapter/appendix start page must not move.

The 2026-07-14 submission baseline froze the layout: **Word 369 pages**, with each
front-matter section, chapter, and appendix starting on a fixed printed page. Any
later content edit (wording, spelling, fixing a broken sentence) is allowed ONLY if
it leaves that map untouched — the printed TOC/LoT/LoF page numbers and the bound
pagination must stay valid.

This guard renders the docx with LibreOffice (Word 369 == LibreOffice 370; the
one-page offset is a renderer difference, not a document change) and asserts every
anchor in ``paper/PAGE_LOCK_20260714.json`` still lands on the same printed page.

Run:
    .venv/bin/python scripts/check_page_lock.py
Exit 0 = lock intact. Exit 1 = an anchor moved (roll the edit back or shorten it).
"""

from __future__ import annotations

import json
import re
from collections import Counter
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_DOCX = _ROOT / "paper" / "고려대_보건석사학위논문_이승진_2026.docx"
_LOCK = _ROOT / "paper" / "PAGE_LOCK_20260714.json"

_ANCHORS: list[tuple[str, str]] = [
    ("ABSTRACT", r"^ABSTRACT$"),
    ("국문 초록", r"^국문 초록$"),
    ("TABLE OF CONTENTS", r"^TABLE OF CONTENTS$"),
    ("LIST OF TABLES", r"^LIST OF TABLES$"),
    ("LIST OF FIGURES", r"^LIST OF FIGURES$"),
    ("NOMENCLATURE", r"^NOMENCLATURE$"),
    *[(f"CHAPTER {n}", rf"^CHAPTER {n}\.") for n in range(1, 6)],
    ("DATA AND CODE AVAILABILITY", r"^DATA AND CODE AVAILABILITY"),
    ("REFERENCES", r"^REFERENCES$"),
    ("APPENDICES", r"^APPENDICES$"),
    *[(f"Appendix {L}", rf"^Appendix {L}\. ") for L in "ABCDEFGHIJKLMNOPQ"],
]
# Front-matter anchors legitimately appear before the TOC-cache pages; body anchors
# must be searched after it, or the TOC's own cached lines match first.
_FRONT = {
    "ABSTRACT", "국문 초록", "TABLE OF CONTENTS",
    "LIST OF TABLES", "LIST OF FIGURES", "NOMENCLATURE",
}
_TOC_CACHE_PAGES = 18


def _printed_page(page_text: str) -> str:
    """Return the page number printed in the footer (roman or arabic), or '-'."""
    lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
    for line in reversed(lines[-4:]):
        if re.fullmatch(r"[ivxlcdm]+|\d{1,3}", line):
            return line
    return "-"


_RENDERS = 7   # LibreOffice is not deterministic — see _render_pages


def _render_once() -> list[str]:
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as profile:
        subprocess.run(
            ["soffice", "--headless", f"-env:UserInstallation=file://{profile}",
             "--convert-to", "pdf", "--outdir", tmp, str(_DOCX)],
            check=True, capture_output=True,
        )
        pdf = next(Path(tmp).glob("*.pdf"))
        text = subprocess.run(
            ["pdftotext", "-layout", str(pdf), "-"],
            check=True, capture_output=True, text=True,
        ).stdout
    return text.split("\f")


def measure() -> tuple[int, dict[str, str], dict[str, Counter]]:
    """Render ``_RENDERS`` times and return the MODAL total and the MODAL page per anchor.

    LibreOffice does not lay the same file out the same way twice. Rendering an unchanged docx
    seven times (md5 identical every time, 2026-07-15) gave 371, 370, 369, 369, 369, 370, 369.
    A single render therefore cannot tell a real one-page shift — precisely what this guard
    exists to catch — from renderer jitter, and it was producing both false passes and false
    failures. The original lock file was itself baselined from one unlucky render and carried
    five wrong anchors.

    Voting per anchor (not picking one "modal layout") is what makes this robust: the jitter is
    local, so an anchor that genuinely moved moves in every render, while a jittery one splits.

    Returns:
        (modal_total, {anchor: modal_printed_page}, {anchor: vote_counter}) — the counters are
        returned so a caller can show *how* close a call was.
    """
    runs = [_render_once() for _ in range(_RENDERS)]
    total = Counter(len(r) - 1 for r in runs).most_common(1)[0][0]

    votes: dict[str, Counter] = {}
    for pages in runs:
        for name, page in _current_map(pages).items():
            votes.setdefault(name, Counter())[page] += 1
    modal = {name: c.most_common(1)[0][0] for name, c in votes.items()}
    return total, modal, votes


def _current_map(pages: list[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for name, pattern in _ANCHORS:
        for i, page in enumerate(pages):
            if name not in _FRONT and i < _TOC_CACHE_PAGES:
                continue  # skip the TOC's own cached entry lines
            if any(re.match(pattern, ln.strip()) for ln in page.splitlines()):
                found[name] = _printed_page(page)
                break
    return found


def main() -> int:
    if not _DOCX.exists() or not _LOCK.exists():
        print(f"SKIP: {_DOCX.name} or {_LOCK.name} absent")
        return 0

    lock = json.loads(_LOCK.read_text(encoding="utf-8"))
    expected = {k: v["printed"] for k, v in lock["anchors"].items()}
    expected_total = lock["total_pages_libreoffice"]

    total, current, votes = measure()

    problems: list[str] = []
    unresolved: list[str] = []
    if total != expected_total:
        problems.append(
            f"TOTAL PAGES moved: {expected_total} -> {total} "
            f"(Word {lock['total_pages_word']} is the frozen count)"
        )
    for name, want in expected.items():
        got = current.get(name)
        tally = votes.get(name, Counter())
        if got is None:
            problems.append(f"{name}: anchor not found (heading text changed?)")
        elif got == want:
            continue
        elif len(tally) > 1 and want in tally:
            # The renderer itself cannot place this anchor: it voted for BOTH pages on the
            # same unchanged file. Appendix L/M/N/P/Q sit on such a boundary — their votes
            # flipped 5:2 -> 3:4 across two runs of an identical document. A guard that fails
            # at random is worse than one that says it has no signal, so this is advisory.
            unresolved.append(f"{name}: {want} vs {got} — renderer split {dict(tally)}")
        else:
            problems.append(f"{name}: printed page {want} -> {got}   (votes {dict(tally)})")

    if unresolved:
        print(f"⚠ {len(unresolved)} anchor(s) LibreOffice cannot resolve "
              f"(it renders both pages on an unchanged file — check these in Word):")
        for u in unresolved:
            print(f"    {u}")

    if problems:
        print(f"PAGE LOCK BROKEN — modal over {_RENDERS} renders:")
        for p in problems:
            print(f"  ✗ {p}")
        return 1

    firm = len(expected) - len(unresolved)
    print(f"✅ page lock intact — modal over {_RENDERS} renders: {total} pages "
          f"(Word {lock['total_pages_word']}); {firm}/{len(expected)} anchors verified exactly"
          + (f", {len(unresolved)} unresolvable by this renderer" if unresolved else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
