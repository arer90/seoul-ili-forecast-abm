#!/usr/bin/env python3
"""Static portability checks — run on Linux, Windows and macOS without installing deps.

A full dependency install is not viable in CI: pyproject pins torch to the
pytorch-cu128 index on linux/win32, so `uv sync` on a GPU-less runner pulls
multi-gigabyte CUDA wheels. These checks need nothing but the standard library,
so they can run on all three platforms in seconds and still catch the class of
bug that actually breaks non-macOS users.

Each check corresponds to a defect found in this repository on 2026-07-19:

  abs-path   scripts/*.py hard-coded /Users/arer90/... pointing at a different
             checkout — instant FileNotFoundError anywhere else.
  posix-tmp  Path("/tmp") glob instead of tempfile.gettempdir(); on Windows this
             resolves to <drive>:\\tmp, yields nothing, and the caller reports
             "no training log" while training is running.
  encoding   read_text()/open() with no encoding= — the repo is full of Korean,
             and a Korean-locale Windows defaults to cp949, so UTF-8 sources
             raise UnicodeDecodeError.
  posix-proc pgrep/ps calls that do not exist on Windows, caught only by
             CalledProcessError so FileNotFoundError escapes.

Exit code 0 = clean, 1 = findings. Use --list to print every finding.
"""
from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Files that are allowed to fail a given check, with the reason.
ALLOW: dict[str, set[str]] = {
    # The detached launcher is POSIX by design; it is documented as such and is
    # not the only way to start training.
    "posix-proc": {"scripts/double_fork_launch.py"},
    # These "/tmp/cellA" strings are argument values compared for equality —
    # cell_run_env() builds an env dict and never touches the filesystem, so
    # the path is opaque to the test and portable as written.
    "posix-tmp": {"simulation/tests/test_ablation_factorial.py"},
}

ABS_PATH = re.compile(r'["\'](/(?:Users|home)/[A-Za-z0-9_.-]+/[^"\']*)["\']')
POSIX_TMP = re.compile(r'Path\(\s*["\']/tmp["\']\s*\)|["\']/tmp/[^"\']*["\']')
POSIX_PROC = re.compile(r'["\'](pgrep|pkill|ps)["\']')


# The checker's own source necessarily contains the very patterns it looks for
# (in its docstring and regexes), so it excludes itself.
SELF = Path(__file__).resolve()

# Text file types worth scanning. The path checks used to look at .py only,
# which left the shell scripts — the layer most likely to hard-code a path —
# entirely unexamined: run_pipeline.sh writes its log to a literal /tmp and no
# check ever saw it. Config and web sources are included for the same reason.
TEXT_SUFFIXES = {
    ".py", ".sh", ".bash", ".zsh",
    ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    ".json", ".yml", ".yaml", ".toml", ".cfg", ".ini",
    ".rs", ".c", ".h", ".sql", ".md", ".txt",
}


def tracked(suffixes: set[str] | None = None) -> list[Path]:
    """Tracked files, optionally narrowed to a set of suffixes.

    Untracked files are excluded on purpose: they are not part of what a user
    receives, so a finding in one is not a distribution defect.
    """
    # -z gives NUL-separated, UNQUOTED paths. Plain `git ls-files` output is
    # split on whitespace here, which breaks any path containing a space, and
    # git quotes non-ASCII names as "\352\260\220..." unless core.quotepath is
    # false — a local setting, so the same command returns different paths on a
    # runner than on the machine the baselines were measured on.
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT,
                         capture_output=True, text=True,
                         encoding="utf-8").stdout.split("\0")
    want = suffixes if suffixes is not None else TEXT_SUFFIXES
    files = []
    for f in out:
        if not f:
            continue
        p = ROOT / f
        # An empty `want` means "every tracked file" — the filename check has to
        # see binaries and data files too, since a bad NAME is bad regardless of
        # what is inside it.
        if want and p.suffix not in want:
            continue
        if not p.is_file() or p.resolve() == SELF:
            continue
        files.append(p)
    return files


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def allowed(check: str, p: Path) -> bool:
    return rel(p) in ALLOW.get(check, set())


def check_abs_path(p: Path, src: str) -> list[str]:
    hits = []
    for i, line in enumerate(src.splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        m = ABS_PATH.search(line)
        if m:
            hits.append(f"{rel(p)}:{i}: hard-coded home path {m.group(1)!r}")
    return hits


def check_posix_tmp(p: Path, src: str) -> list[str]:
    hits = []
    for i, line in enumerate(src.splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        if POSIX_TMP.search(line):
            hits.append(f"{rel(p)}:{i}: hard-coded /tmp — use tempfile.gettempdir()")
    return hits


def check_encoding(p: Path, src: str) -> list[str]:
    """open()/read_text()/write_text() on a text path with no encoding=."""
    hits = []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return hits
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
        if name not in {"open", "read_text", "write_text"}:
            continue
        kw = {k.arg for k in node.keywords}
        if "encoding" in kw:
            continue
        # binary mode needs no encoding
        mode = ""
        for a in node.args[1:2]:
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                mode = a.value
        for k in node.keywords:
            if k.arg == "mode" and isinstance(k.value, ast.Constant):
                mode = str(k.value.value)
        if "b" in mode:
            continue
        hits.append(f"{rel(p)}:{node.lineno}: {name}() without encoding= "
                    f"(Windows ko-KR defaults to cp949)")
    return hits


def check_posix_proc(p: Path, src: str) -> list[str]:
    hits = []
    for i, line in enumerate(src.splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        if POSIX_PROC.search(line) and ("subprocess" in src or "check_output" in line or "run(" in line):
            hits.append(f"{rel(p)}:{i}: POSIX-only process tool — absent on Windows")
    return hits


def check_filename(p: Path, src: str) -> list[str]:
    """Non-ASCII or space-bearing tracked path.

    Two failure modes, both hit on 2026-07-20. `git ls-files` quotes non-ASCII
    names as "\\352\\260\\220..." unless core.quotepath is false — a per-machine
    setting — so the same tooling sees a different file set on a runner than on
    the machine a baseline was measured on. And a path containing a space is
    split apart by any consumer that parses that output on whitespace.

    Checked once per file via its own path; `src` is unused.
    """
    r = rel(p)
    hits = []
    if any(ord(c) > 127 for c in r):
        hits.append(f"{r}: non-ASCII path — git quotes it unless core.quotepath=false")
    if " " in r:
        hits.append(f"{r}: path contains a space — breaks whitespace-split consumers")
    return hits


CHECKS = {
    "filename": check_filename,
    "abs-path": check_abs_path,
    "posix-tmp": check_posix_tmp,
    "encoding": check_encoding,
    "posix-proc": check_posix_proc,
}

# Which file types each check reads.
#
# The two path checks scan every text type: a hard-coded path is just as broken
# in a shell script or a JSON config as in a module, and the shell layer is
# where they cluster.
#
# The other two are Python-specific by nature. `encoding` parses an AST, and
# `posix-proc` is about Python shelling out to a tool that does not exist on
# Windows — a .sh calling pgrep is not a defect, since the shell script is
# POSIX-only by definition and is documented as such.
SCOPE: dict[str, set[str] | None] = {
    "filename": set(),         # empty set = every tracked file, text or not
    "abs-path": None,          # None = every text type
    "posix-tmp": None,
    "encoding": {".py"},
    "posix-proc": {".py"},
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="print every finding")
    ap.add_argument("--only", choices=sorted(CHECKS), help="run a single check")
    ap.add_argument("--max", type=int, default=0,
                    help="allowed findings before failing (baseline)")
    args = ap.parse_args()

    names = [args.only] if args.only else list(CHECKS)
    found: dict[str, list[str]] = {n: [] for n in names}

    for n in names:
        for p in tracked(SCOPE[n]):
            if allowed(n, p):
                continue
            try:
                src = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            found[n].extend(CHECKS[n](p, src))

    total = sum(len(v) for v in found.values())
    print(f"portability checks on {sys.platform}")
    for n in names:
        print(f"  {n:<11} {len(found[n])}")
        if args.list:
            for h in found[n]:
                print(f"      {h}")
    print(f"  {'total':<11} {total}  (allowed {args.max})")

    return 1 if total > args.max else 0


if __name__ == "__main__":
    raise SystemExit(main())
