#!/usr/bin/env python3
"""Apply a JSON list of exact text edits to the thesis docx, atomically.

Every edit is ``{"block": int, "old": str, "new": str}`` — ``old`` must exist inside a
single ``w:r`` of that body block. Runs are the unit because §3.3/§3.4 prose is now
interleaved with inline ``m:oMath`` nodes: a substring that straddles a math node lives
in no single run and cannot be matched, so we fail loudly rather than silently skipping
(the exact silent-void failure mode ENGINEERING_PRINCIPLES.md G-237 forbids).

Nothing is written unless *every* edit resolves. On success the previous file is kept
as ``*_pre_<tag>.docx`` so a broken page lock can be rolled back in one move.

Run:
    .venv/bin/python scripts/thesis_apply_edits.py edits.json --tag codealign
    .venv/bin/python scripts/check_page_lock.py        # MUST stay 369p / 31 anchors
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import docx
from lxml import etree

_ROOT = Path(__file__).resolve().parents[1]
_DOCX = _ROOT / "paper" / "고려대_보건석사학위논문_이승진_2026.docx"

_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_XML = "http://www.w3.org/XML/1998/namespace"


def w(tag: str) -> str:
    return f"{{{_W}}}{tag}"


def _run_text(r) -> str:
    return "".join(t.text or "" for t in r.iter(w("t")))


def _candidate_runs(block) -> list:
    """Every ``w:r`` the edit may target, whether ``block`` is a paragraph or a table.

    Body blocks are not all paragraphs: 74 of them are ``w:tbl``, whose runs live several
    levels down (tbl > tr > tc > p > r). A paragraph-children-only scan silently missed
    those, so table-cell corrections looked like "string not found" rather than "wrong
    place to look" — descend instead.
    """
    if block.tag == w("tbl"):
        return list(block.iter(w("r")))
    return [r for r in block.iterchildren() if r.tag == w("r")]


def apply_edit(p, old: str, new: str, *, all_runs: bool = False) -> int:
    """Replace ``old`` with ``new`` inside the run(s) that contain it.

    Args:
        all_runs: replace in EVERY matching run of the block. Needed when a symbol is
            repeated — e.g. Appendix A writes the commuter index ``j→k`` three times and
            all three are transposed relative to the code. Without it a repeated token is
            rejected as ambiguous, which is the right default: silently editing the first
            of several identical strings is how a half-applied correction happens.

    Returns:
        Number of runs edited.

    Raises:
        LookupError: ``old`` is in no single run (missing, or split across runs / m:oMath).
        ValueError: ``old`` is in several runs and ``all_runs`` is False — ambiguous, so
            the caller must add context rather than guess.
    """
    hits = [r for r in _candidate_runs(p) if old in _run_text(r)]
    if not hits:
        raise LookupError(f"not found in any single run: {old!r}")
    if len(hits) > 1 and not all_runs:
        raise ValueError(f"ambiguous — {len(hits)} runs contain: {old!r}")

    for r in hits:
        _swap(r, old, new)
    return len(hits)


def _swap(r, old: str, new: str) -> None:
    text = _run_text(r)
    head, _, tail = text.partition(old)

    rpr = r.find(w("rPr"))
    parent = r.getparent()
    idx = parent.index(r)

    def _mk(s: str):
        nr = etree.Element(w("r"))
        if rpr is not None:
            nr.append(etree.fromstring(etree.tostring(rpr)))
        t = etree.SubElement(nr, w("t"))
        t.text = s
        t.set(f"{{{_XML}}}space", "preserve")
        return nr

    parent.insert(idx, _mk(head + new + tail))
    parent.remove(r)


def set_cell(tbl, row: int, col: int, new: str) -> None:
    """Replace a table cell's text by ADDRESS, keeping its first run's formatting.

    Some cells cannot be reached by string match: Table 2 prints ``3.90`` as both the
    champion's MAE and the runner-up's WIS, and ``0.904`` as two different models' scores, so
    an ``old`` string is ambiguous inside the block. Addressing the cell (row, col) is exact.

    Raises:
        IndexError: the address does not exist — fail loud rather than edit the wrong cell.
    """
    rows = list(tbl.iter(w("tr")))
    cells = list(rows[row].iter(w("tc")))
    tc = cells[col]

    paras = list(tc.iter(w("p")))
    if not paras:
        raise IndexError(f"cell ({row},{col}) has no paragraph")
    target = paras[0]

    runs = [r for r in target.iterchildren() if r.tag == w("r")]
    if not runs:
        raise IndexError(f"cell ({row},{col}) has no run")

    keep, rest = runs[0], runs[1:]
    for t in keep.iter(w("t")):
        t.text = new
        t.set(f"{{{_XML}}}space", "preserve")
        break
    else:
        t = etree.SubElement(keep, w("t"))
        t.text = new
    # drop trailing runs so a rsid-split cell does not keep its old tail
    for r in rest:
        r.getparent().remove(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("plan", help="JSON file: [{block, old, new, ...}, ...] or [{block, row, col, new}]")
    ap.add_argument("--tag", default="edits", help="backup suffix")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    edits = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    if isinstance(edits, dict):
        edits = edits.get("edits", [])

    d = docx.Document(str(_DOCX))
    kids = list(d.element.body.iterchildren())

    net = 0
    for i, e in enumerate(edits, 1):
        block = e["block"]
        # cell-addressed edit (row/col) — used where an `old` string is ambiguous in-block
        if "row" in e and "col" in e:
            try:
                blk = kids[block]
                if blk.tag != w("tbl"):
                    raise IndexError("block is not a table")
                before = "".join(
                    t.text or "" for t in
                    list(list(blk.iter(w("tr")))[e["row"]].iter(w("tc")))[e["col"]].iter(w("t"))
                )
                set_cell(blk, e["row"], e["col"], e["new"])
            except IndexError as exc:
                print(f"✗ [{i}/{len(edits)}] {e.get('id', '')} block {block} "
                      f"cell({e['row']},{e['col']}): {exc}")
                print("  ABORT — nothing written; docx untouched.")
                return 1
            delta = len(e["new"]) - len(before)
            net += delta
            print(f"✓ [{i}/{len(edits)}] {e.get('id', ''):<16} block {block:<5} "
                  f"cell({e['row']},{e['col']}) {before!r}→{e['new']!r} Δ{delta:+d}")
            continue

        old, new = e["old"], e["new"]
        try:
            n = apply_edit(kids[block], old, new, all_runs=e.get("all", False))
            e["_n"] = n
        except (LookupError, ValueError) as exc:
            print(f"✗ [{i}/{len(edits)}] {e.get('id', '')} block {block}: {exc}")
            print("  ABORT — nothing written; docx untouched.")
            return 1
        delta = (len(new) - len(old)) * e["_n"]
        net += delta
        rep = f" ×{e['_n']}" if e["_n"] > 1 else ""
        print(f"✓ [{i}/{len(edits)}] {e.get('id', ''):<16} block {block:<5} Δ{delta:+d}{rep}")

    print(f"\nnet char delta: {net:+d}")
    if args.dry_run:
        print("(dry run — not saved)")
        return 0

    backup = _DOCX.with_name(f"{_DOCX.stem}_pre_{args.tag}.docx")
    shutil.copy2(_DOCX, backup)
    d.save(str(_DOCX))
    print(f"✅ saved. backup: {backup.name}")
    print("   next: .venv/bin/python scripts/check_page_lock.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
