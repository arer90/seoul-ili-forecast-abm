#!/usr/bin/env python3
"""Assert the thesis PDF reproduces the frozen table of contents — every heading, not just chapters.

The submitted layout is fixed down to the subsection: ``3.5.2`` must print on page 45 and
``Appendix O.4`` on page 266, because the bound TOC promises exactly that. Checking only chapter
and appendix starts (31 anchors) leaves ~37 subsection lines unverified, and a length-changing
edit inside §4.2 can move ``4.2.1`` without touching any chapter boundary.

The reference is ``paper/THESIS_TOC_LOCK.tsv`` (``printed_page <TAB> heading``), transcribed from
the submitted PDF's own contents pages.

Matching is by the heading's *body* occurrence, not its TOC line: the contents pages repeat every
heading verbatim, so a naive search finds page v before page 45. Body pages start after the TOC
block, so the scan skips it.

Run:
    .venv/bin/python scripts/thesis_toc_check.py                     # check the working PDF
    .venv/bin/python scripts/thesis_toc_check.py --pdf other.pdf
Exit 0 = every heading on its frozen page. Exit 1 = something moved.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_PDF = _ROOT / "paper" / "고려대_보건석사학위논문_이승진_2026.pdf"
_LOCK = _ROOT / "paper" / "THESIS_TOC_LOCK.tsv"

_TOC_PAGES = 20          # physical pages of front matter that repeat every heading
_TOTAL = 369


def _pages(pdf: Path) -> list[str]:
    out = subprocess.run(["pdftotext", "-layout", str(pdf), "-"],
                         check=True, capture_output=True, text=True).stdout
    return out.split("\f")


def _printed(page: str) -> str | None:
    """The number printed in the footer, or None when the page carries none."""
    lines = [ln.strip() for ln in page.splitlines() if ln.strip()]
    for line in reversed(lines[-4:]):
        if re.fullmatch(r"\d{1,3}", line):
            return line
    return None


def _find(pages: list[str], heading: str) -> str | None:
    """Printed page of the first BODY page whose text opens a line with ``heading``.

    Long headings wrap: Appendix P's title breaks after "full evaluation", so requiring the whole
    string on one line finds nothing. Match a leading slice instead — long enough to be unique
    (``Appendix H.`` and ``Appendix H.1`` diverge at character 11) and short enough to survive the
    wrap.
    """
    prefix = heading[:48]
    pat = re.compile(r"^\s*" + re.escape(prefix).replace(r"\ ", r"\s+"), re.IGNORECASE)
    for i, page in enumerate(pages):
        if i < _TOC_PAGES:
            continue
        for line in page.splitlines():
            if pat.match(line):
                return _printed(page)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=str(_PDF))
    args = ap.parse_args()

    pdf = Path(args.pdf)
    if not _LOCK.exists():
        print(f"lock missing: {_LOCK}")
        return 1

    pages = _pages(pdf)
    total = len(pages) - 1

    lock = []
    for raw in _LOCK.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        page, heading = raw.split("\t", 1)
        lock.append((page.strip(), heading.strip()))

    print(f"■ {pdf.name}")
    print(f"  총 {total} 페이지  " + ("일치 ✓" if total == _TOTAL else f"✗ 기준 {_TOTAL}"))
    print()

    bad, missing = [], []
    for want, heading in lock:
        got = _find(pages, heading)
        if got is None:
            missing.append(heading)
            print(f"  {'?':>4} ← {want:>4}  ✗ 못 찾음   {heading}")
        elif got != want:
            bad.append((heading, want, got))
            print(f"  {got:>4} ← {want:>4}  ✗ 이동     {heading}")

    ok = len(lock) - len(bad) - len(missing)
    print(f"\n  {ok}/{len(lock)} 항목이 고정 페이지에 있음")

    if bad or missing or total != _TOTAL:
        print("✗ 목차 고정이 깨졌습니다.")
        return 1
    print(f"✅ 목차 고정 유지 — 총 {total}p, {len(lock)}개 항목 전부 원래 페이지")
    return 0


if __name__ == "__main__":
    sys.exit(main())
