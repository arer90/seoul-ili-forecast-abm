#!/usr/bin/env python3
"""Extract the thesis' page map — every chapter/appendix start page — and compare two documents.

The submission PDF is the AUTHORITY for layout (Word/Quartz output, 369 pages). The docx must
reproduce that map exactly: same total, same printed page for every front-matter section, chapter,
and appendix. Anything else means the bound thesis and its own table of contents disagree.

Two facts make a naive check useless, and this script exists to handle both:

  * A heading's *printed* page (the number in the footer — roman ``iv``, arabic ``62``) is what the
    TOC promises the reader. The *physical* sheet index is what a PDF viewer shows. They differ by
    the unnumbered front matter, so both are reported.
  * LibreOffice does not lay the same file out twice the same way — an unchanged docx rendered seven
    times gave 371/370/369/369/369/370/369. A single render cannot distinguish a real one-page shift
    from renderer jitter, so the docx side is rendered N times and every anchor is decided by MODAL
    VOTE. An anchor the renderer itself splits is reported as unresolvable, not as a failure.

The only fully trustworthy check is PDF-vs-PDF: re-export from Word, then run this with two PDFs and
the renderer drops out of the loop entirely.

Run:
    .venv/bin/python scripts/thesis_page_map.py                       # PDF map + docx comparison
    .venv/bin/python scripts/thesis_page_map.py --pdf new.pdf         # PDF vs PDF (definitive)
    .venv/bin/python scripts/thesis_page_map.py --renders 3           # faster, noisier docx check
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_DOCX = _ROOT / "paper" / "고려대_보건석사학위논문_이승진_2026.docx"
_PDF = _ROOT / "paper" / "고려대_보건석사학위논문_이승진_2026.pdf"

# (label, regex matched against a stripped line). Order = document order.
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

# Front-matter headings legitimately precede the TOC's own cached lines; body headings must be
# searched after it or the TOC entry "CHAPTER 1. Introduction ...... 1" matches before the chapter.
_FRONT = {"ABSTRACT", "국문 초록", "TABLE OF CONTENTS",
          "LIST OF TABLES", "LIST OF FIGURES", "NOMENCLATURE"}
_TOC_CACHE_PAGES = 18


def _printed(page_text: str) -> str:
    """The page number printed in the footer (roman or arabic), or '-' when the page carries none."""
    lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
    for line in reversed(lines[-4:]):
        if re.fullmatch(r"[ivxlcdm]+|\d{1,3}", line):
            return line
    return "-"


def _pages_of_pdf(pdf: Path) -> list[str]:
    out = subprocess.run(["pdftotext", "-layout", str(pdf), "-"],
                         check=True, capture_output=True, text=True).stdout
    return out.split("\f")


def _pages_of_docx(docx: Path) -> list[str]:
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as profile:
        subprocess.run(
            ["soffice", "--headless", f"-env:UserInstallation=file://{profile}",
             "--convert-to", "pdf", "--outdir", tmp, str(docx)],
            check=True, capture_output=True,
        )
        return _pages_of_pdf(next(Path(tmp).glob("*.pdf")))


def _map(pages: list[str]) -> dict[str, tuple[str, int]]:
    """{anchor: (printed page, physical sheet 1-indexed)} for every anchor found."""
    found: dict[str, tuple[str, int]] = {}
    for name, pattern in _ANCHORS:
        for i, page in enumerate(pages):
            if name not in _FRONT and i < _TOC_CACHE_PAGES:
                continue
            if any(re.match(pattern, ln.strip()) for ln in page.splitlines()):
                found[name] = (_printed(page), i + 1)
                break
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", help="compare the reference PDF against THIS pdf instead of the docx")
    ap.add_argument("--ref", help="reference PDF (default: the submission PDF next to the docx). "
                                  "Point this at the archived as-submitted copy once the working "
                                  "PDF has been re-exported, or the check compares a file to itself.")
    ap.add_argument("--renders", type=int, default=7,
                    help="docx renders to vote over (LibreOffice is non-deterministic)")
    args = ap.parse_args()

    ref_pdf = Path(args.ref) if args.ref else _PDF
    if not ref_pdf.exists():
        print(f"reference PDF missing: {ref_pdf}")
        return 1

    ref_pages = _pages_of_pdf(ref_pdf)
    ref = _map(ref_pages)
    ref_total = len(ref_pages) - 1

    print(f"■ 기준 = 제출본 PDF  ({ref_pdf.name})")
    print(f"  총 {ref_total} 페이지\n")

    if args.pdf:
        cmp_path = Path(args.pdf)
        cmp_pages = _pages_of_pdf(cmp_path)
        cur = {k: v for k, v in _map(cmp_pages).items()}
        cur_total = len(cmp_pages) - 1
        votes: dict[str, Counter] = {}
        label = f"새 PDF ({cmp_path.name})"
        jitter_ok = False
    else:
        runs = [_pages_of_docx(_DOCX) for _ in range(args.renders)]
        cur_total = Counter(len(r) - 1 for r in runs).most_common(1)[0][0]
        votes = {}
        for pages in runs:
            for name, (printed, _phys) in _map(pages).items():
                votes.setdefault(name, Counter())[printed] += 1
        modal = _map(runs[0])
        cur = {}
        for name, counter in votes.items():
            printed = counter.most_common(1)[0][0]
            cur[name] = (printed, modal.get(name, (printed, 0))[1])
        label = f"현재 docx (LibreOffice {args.renders}회 렌더 최빈값)"
        jitter_ok = True

    print(f"■ 대조 = {label}")
    print(f"  총 {cur_total} 페이지  "
          + ("일치 ✓" if cur_total == ref_total else f"✗ 기준 {ref_total} 과 다름") + "\n")

    hdr = f"  {'구성 요소':<28} {'제출본':>8} {'현재':>8}   {'물리쪽':>6}  판정"
    print(hdr)
    print("  " + "─" * (len(hdr) + 6))

    bad, split = [], []
    for name, _ in _ANCHORS:
        want = ref.get(name)
        got = cur.get(name)
        if want is None:
            print(f"  {name:<28} {'(없음)':>8}")
            continue
        wp, wphys = want
        if got is None:
            print(f"  {name:<28} {wp:>8} {'못찾음':>8}          ✗ 제목이 바뀌었나?")
            bad.append(name)
            continue
        gp, _ = got
        tally = votes.get(name, Counter())
        if gp == wp:
            print(f"  {name:<28} {wp:>8} {gp:>8}   {wphys:>6}  ✓")
        elif jitter_ok and len(tally) > 1 and wp in tally:
            print(f"  {name:<28} {wp:>8} {gp:>8}   {wphys:>6}  ~ 렌더러가 갈림 {dict(tally)}")
            split.append(name)
        else:
            print(f"  {name:<28} {wp:>8} {gp:>8}   {wphys:>6}  ✗ 이동")
            bad.append(name)

    print()
    ok = len(ref) - len(bad) - len(split)
    if bad or cur_total != ref_total:
        print(f"✗ 페이지 구성이 제출본과 다릅니다 — 앵커 {len(bad)}개 이동"
              + (f", 총페이지 {ref_total}→{cur_total}" if cur_total != ref_total else ""))
        return 1
    msg = f"✅ 제출본 PDF 와 페이지 구성 일치 — 총 {ref_total}p, 앵커 {ok}/{len(ref)} 정확 일치"
    if split:
        msg += f", {len(split)}개는 LibreOffice가 판정 불가(Word 에서 확인 필요)"
    print(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
