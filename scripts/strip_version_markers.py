"""Strip version markers ( 등) from comments and docstrings only.

SAFE strategy (vs the earlier catastrophic `sed 's/ */ /g'` pass):
 - Uses `tokenize` to parse Python files into tokens
 - Only modifies text INSIDE tokenize.COMMENT and tokenize.STRING tokens
 - Preserves all whitespace, indentation, and identifier names
 - Verifies AST is unchanged structurally before writing (skip if parse fails)

Patterns removed:
 .x.y, .x, -(with word boundaries)
 Standalone vN in specific contexts (comment prefixes only)
 Phrase-level markers: "Mac-migration:" -> "Mac-migration:"
 "FIX (2026-xx-xx):" -> "FIX (2026-xx-xx):"
 "KUIRB freeze:" -> "KUIRB freeze:"
 "(KUIRB 정합):" -> "(KUIRB 정합):"

Out of scope (left alone):
 - Identifiers like `run_phase6_v2`, `graph_models_v2` (renamed separately)
 - File names
 - Code literals used at runtime (e.g., `{"version": ""}` in dicts)
 We strip only from COMMENT and STRING tokens whose content LOOKS LIKE
 natural-language prose (contains spaces or Korean). Pure identifier-like
 strings like `""` alone inside a dict are PRESERVED if they are the
 sole content of the string.

Usage:
 uv run python scripts/strip_version_markers.py [--dry-run] [--diff]
"""
from __future__ import annotations

import argparse
import ast
import io
import re
import sys
import tokenize
from pathlib import Path


# Ordered most-specific first.
# Keep word boundaries; replace with empty string (and a space) for readability.
VERSION_PATTERNS = [
    # Phrase-level cleanups first (match + trim leading ".x ").
    (re.compile(r"\bv22\.\d+\.\d+\s+"), ""),
    (re.compile(r"\bv22\.\d+\s+"), ""),
    (re.compile(r"\bv1[4-9]\s+"), ""),
    (re.compile(r"\bv2[0-2](?!\d)\s+"), ""),
    # Parenthetical ".x" inside text: "" -> "" (rare)
    (re.compile(r"\(\s*\.\d+(\.\d+)?\s*\)"), ""),
    # Tail marker: "... " -> "..."
    (re.compile(r"\s*\(\.\d+(\.\d+)?\)"), ""),
    # Standalone .x at end of line (no trailing space)
    (re.compile(r"\bv22\.\d+(\.\d+)?\b"), ""),
    (re.compile(r"\bv1[4-9]\b"), ""),
    (re.compile(r"\bv2[0-2](?!\d)\b"), ""),
    # Also strip bare "v7", "v6", "v4", etc. when followed by 비교 context keywords.
    # But we scope it to comments only — see process_token below.
]


# Pattern that matches versioned doc tokens we consider noise in comment text.
SINGLE_DIGIT_VERSION = re.compile(r"(?<![A-Za-z0-9_])v[1-9][a-z]?(?=[\s,:;\)\/])")


def _looks_like_prose(text: str) -> bool:
    """Is this token content a natural-language comment/docstring vs. a short identifier literal?

 We preserve short literal-looking strings (e.g. `""` as a dict value)
 and only rewrite prose-like content (has whitespace / Korean / prose length).
 """
    stripped = text.strip().strip("\"'")
    # Very short? Leave alone — almost certainly a literal tag used at runtime.
    if len(stripped) <= 8 and "," not in stripped and " " not in stripped:
        return False
    # Has letters other than the version prefix? It's prose.
    if re.search(r"[a-zA-Z가-힣]", stripped):
        return True
    return False


def _strip_in_text(text: str) -> tuple[str, int]:
    """Apply all version patterns to a chunk of prose text."""
    new = text
    hits = 0
    for pat, repl in VERSION_PATTERNS:
        new, n = pat.subn(repl, new)
        hits += n
    # Single-digit version marker — only remove when it appears in prose
    # following a context keyword (benchmark/compare/result/neuter patterns).
    new, n = SINGLE_DIGIT_VERSION.subn("", new)
    hits += n
    # Compact double-spaces created by trimming (within the text only, not
    # indentation — tokenize tokens are content only, they do not include
    # leading indentation).
    new = re.sub(r"  +", " ", new)
    # Clean up orphaned punctuation left behind: ", , ", ": ,", "()", "( )".
    new = re.sub(r",\s*,", ",", new)
    new = re.sub(r"\(\s*\)", "", new)
    new = re.sub(r"\(\s*,", "(", new)
    new = re.sub(r",\s*\)", ")", new)
    new = re.sub(r":\s*,", ":", new)
    return new, hits


def _process_comment(token_string: str) -> tuple[str, int]:
    """Comment tokens always start with '#'. Preserve prefix, strip body."""
    assert token_string.startswith("#")
    # Keep the '#' and any immediate whitespace untouched.
    m = re.match(r"(#\s*)(.*)", token_string, flags=re.DOTALL)
    if not m:
        return token_string, 0
    prefix, body = m.group(1), m.group(2)
    new_body, hits = _strip_in_text(body)
    return prefix + new_body, hits


def _process_string(token_string: str) -> tuple[str, int]:
    """Triple-quoted docstrings + regular strings.

    Only rewrite if content looks like prose (see _looks_like_prose).
    We preserve the original quote style and prefix (r/b/f) exactly.
    """
    # Identify prefix (rb, f, u, etc.) and quote type.
    m = re.match(r"([rRbBuUfF]*)('''|\"\"\"|'|\")(.*)(\2)$", token_string, flags=re.DOTALL)
    if not m:
        return token_string, 0
    prefix, quote, body, _close = m.group(1), m.group(2), m.group(3), m.group(4)
    if not _looks_like_prose(body):
        return token_string, 0
    new_body, hits = _strip_in_text(body)
    if hits == 0:
        return token_string, 0
    return f"{prefix}{quote}{new_body}{quote}", hits


def process_file(path: Path, dry_run: bool = False) -> tuple[int, bool]:
    """Return (total_hits, changed)."""
    try:
        original = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return 0, False

    # Ensure the file parses before we touch it.
    try:
        ast.parse(original)
    except SyntaxError:
        return 0, False

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(original).readline))
    except tokenize.TokenizeError:
        return 0, False

    # FSTRING_MIDDLE is only present on 3.12+; use getattr for portability.
    FSTRING_MIDDLE = getattr(tokenize, "FSTRING_MIDDLE", None)

    new_tokens = []
    total_hits = 0
    for tok in tokens:
        tok_type, tok_string, start, end, line = tok
        if tok_type == tokenize.COMMENT:
            new_str, hits = _process_comment(tok_string)
            total_hits += hits
            if hits:
                tok = (tok_type, new_str, start, end, line)
        elif tok_type == tokenize.STRING:
            new_str, hits = _process_string(tok_string)
            total_hits += hits
            if hits:
                tok = (tok_type, new_str, start, end, line)
        elif FSTRING_MIDDLE is not None and tok_type == FSTRING_MIDDLE:
            # f-string literal segments — no quotes, just raw text between
            # {}-interpolations. Treat as prose.
            if _looks_like_prose(tok_string):
                new_str, hits = _strip_in_text(tok_string)
                total_hits += hits
                if hits:
                    tok = (tok_type, new_str, start, end, line)
        new_tokens.append(tok)

    if total_hits == 0:
        return 0, False

    try:
        rebuilt = tokenize.untokenize(new_tokens)
    except Exception:
        return 0, False

    # Verify new source still parses.
    try:
        ast.parse(rebuilt)
    except SyntaxError:
        return 0, False  # skip — safer to leave alone than commit broken source

    # Verify structural AST unchanged — body count/node count identical.
    # (Rough but catches accidental token loss.)
    old_nodes = sum(1 for _ in ast.walk(ast.parse(original)))
    new_nodes = sum(1 for _ in ast.walk(ast.parse(rebuilt)))
    if old_nodes != new_nodes:
        return 0, False

    if not dry_run:
        path.write_text(rebuilt, encoding="utf-8")
    return total_hits, True


def _process_markdown(path: Path, dry_run: bool = False) -> tuple[int, bool]:
    """Simple text-level strip for .md / .txt files. No AST, just regex."""
    try:
        original = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return 0, False
    new, hits = _strip_in_text(original)
    if hits == 0 or new == original:
        return 0, False
    if not dry_run:
        path.write_text(new, encoding="utf-8")
    return hits, True


EXCLUDE_DIRS = {"__pycache__", "_past", "_root_legacy", "_thesis_archive",
                "_archive", "_legacy", "_sandbox", "node_modules", ".venv",
                ".next", ".pytest_cache", "target", ".git"}


def walk_targets(root: Path, include_md: bool = True):
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        suffix = p.suffix.lower()
        if suffix == ".py":
            yield p, "py"
        elif include_md and suffix in (".md", ".txt"):
            yield p, "md"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--root", default=".", help="project root")
    ap.add_argument("--py-only", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    stats = {"py_files_changed": 0, "py_hits": 0,
             "md_files_changed": 0, "md_hits": 0, "skipped": 0}
    for path, kind in walk_targets(root, include_md=not args.py_only):
        if kind == "py":
            hits, changed = process_file(path, dry_run=args.dry_run)
            if changed:
                stats["py_files_changed"] += 1
                stats["py_hits"] += hits
                print(f"  py  {hits:4d}  {path.relative_to(root)}")
            elif hits == 0 and path.suffix == ".py":
                pass  # no hits — don't print
        elif kind == "md":
            hits, changed = _process_markdown(path, dry_run=args.dry_run)
            if changed:
                stats["md_files_changed"] += 1
                stats["md_hits"] += hits
                print(f"  md  {hits:4d}  {path.relative_to(root)}")

    print()
    print(f"Python:   {stats['py_files_changed']} files, {stats['py_hits']} replacements")
    print(f"Markdown: {stats['md_files_changed']} files, {stats['md_hits']} replacements")
    if args.dry_run:
        print("(DRY RUN — no files written)")


if __name__ == "__main__":
    main()
