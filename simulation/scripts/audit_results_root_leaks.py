#!/usr/bin/env python3
"""Audit: enumerate every NON-docstring ``simulation/results`` string literal in
the live code tree (the MPH_OUTPUT_ROOT sweep scope).

Why AST, not grep: grep cannot distinguish a docstring/comment mention from a
real code literal, and the per_model_research miss (2026-05-29) proved that
"scan the files I edited" is the wrong frame — sibling writers hide in untouched
caller files. This walks the whole live tree and reports code literals only.

Excludes: _archive/, tests/, __pycache__, *_backup*. Reports grouped by the
first path segment after ``simulation/results/`` so data-output writers
(predictions, per_model_*) can be prioritized over figures/MD reports.

Run:  .venv/bin/python -m simulation.scripts.audit_results_root_leaks
"""
from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NEEDLE = "simulation/results"
SKIP_PARTS = {"_archive", "tests", "__pycache__"}


def _docstring_node_ids(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                ids.add(id(body[0].value))
    return ids


def _classify(fp: Path, line_src: str, literal: str) -> str:
    """Heuristic: which non-docstring literals are REAL path-construction leaks
    vs. safe config.save_dir fallbacks / log+help messages / the SSOT definition."""
    if fp.name == "config_global.py":
        return "SSOT_DEF"          # the project-local default definition itself
    if "get_results_dir" in line_src:
        return "SAFE"              # already routed (literal is a dead fallback string)
    if "getattr(config" in line_src and "save_dir" in line_src:
        return "SAFE"              # config.save_dir wins; literal only if attr missing (never)
    if "config.save_dir" in line_src or "config.get_save_dir" in line_src or "get_save_dir(" in line_src:
        return "SAFE"
    # message/help/comment-ish strings (not a path passed to Path()/open())
    if any(tok in literal for tok in ("`", "→", "\\", "<", "Wrote:", "default:", "Format:", "Dims:")):
        return "MESSAGE"
    if literal.strip().count(" ") >= 2:
        return "MESSAGE"
    return "CANDIDATE"             # real hardcoded path literal — review per-site


def _scan_file(fp: Path) -> list[tuple[int, str, str]]:
    try:
        text = fp.read_text(encoding="utf-8")
        tree = ast.parse(text)
    except Exception:
        return []
    src_lines = text.splitlines()
    doc_ids = _docstring_node_ids(tree)
    hits: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and NEEDLE in node.value
            and id(node) not in doc_ids
        ):
            ln = getattr(node, "lineno", -1)
            line_src = src_lines[ln - 1] if 0 < ln <= len(src_lines) else ""
            hits.append((ln, node.value, _classify(fp, line_src, node.value)))
    return hits


def _first_segment(literal: str) -> str:
    # ".../simulation/results/<seg>/..." → <seg>  (or "<root>" if bare)
    after = literal.split("simulation/results", 1)[1].lstrip("/")
    seg = after.split("/", 1)[0] if after else "<results root>"
    return seg or "<results root>"


def main() -> int:
    roots = [REPO / "simulation", REPO / "scripts"]
    cand_by_seg: dict[str, list[str]] = defaultdict(list)
    klass_count: dict[str, int] = defaultdict(int)
    site_total = 0
    files_total = 0
    for root in roots:
        if not root.exists():
            continue
        for fp in sorted(root.rglob("*.py")):
            if any(part in SKIP_PARTS for part in fp.parts):
                continue
            if "_backup" in str(fp) or fp.name == "audit_results_root_leaks.py":
                continue
            hits = _scan_file(fp)
            if not hits:
                continue
            files_total += 1
            site_total += len(hits)
            rel = fp.relative_to(REPO)
            for lineno, lit, klass in hits:
                klass_count[klass] += 1
                if klass == "CANDIDATE":
                    cand_by_seg[_first_segment(lit)].append(f"{rel}:{lineno}  {lit!r}")

    print(f"=== results-root code literals: {site_total} sites in {files_total} files ===")
    for k in ("CANDIDATE", "SAFE", "MESSAGE", "SSOT_DEF"):
        print(f"    {k:10s}: {klass_count.get(k, 0)}")
    print(f"\n=== CANDIDATE (real path literals — review per-site) ===\n")
    n_cand = sum(len(v) for v in cand_by_seg.values())
    for seg in sorted(cand_by_seg, key=lambda s: -len(cand_by_seg[s])):
        sites = cand_by_seg[seg]
        print(f"── '{seg}'  ({len(sites)} site(s)) ──")
        for s in sites:
            print(f"    {s}")
        print()
    print(f"TOTAL CANDIDATE sites: {n_cand}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
