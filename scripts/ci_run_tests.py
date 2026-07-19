#!/usr/bin/env python3
"""Run every test file CI can run, and only skip the ones it demonstrably cannot.

The list in `scripts/ci_test_exclusions.txt` is EXCLUSIONS, not inclusions. A new
test file is therefore covered the moment it lands — nobody has to remember to
register it. The trade-off is that the exclusion list must be re-measured when
the environment changes, which `--survey` does.

Runs one pytest process per file. That is not a style choice: macOS segfaults
when LightGBM/OpenMP is initialised more than once in a process, so the whole
repository is run per-file. It also means a failure names the file that broke
instead of one wall of output.

    python scripts/ci_run_tests.py            # run everything not excluded
    python scripts/ci_run_tests.py --list     # show what would run, run nothing
    python scripts/ci_run_tests.py --survey   # re-measure and print a fresh
                                              # exclusion list for this env

Exit 0 if every non-excluded file passes.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUSIONS = ROOT / "scripts" / "ci_test_exclusions.txt"

# Why a file could not run, when surveying.
DATA = re.compile(r"25 Seoul gu names|no such table|epi_real_seoul|OperationalError"
                  r"|models/champion|pubmed_abstracts|sentinel_influenza|feature_cache")
DEP = re.compile(r"ModuleNotFoundError|No module named", re.I)


def test_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z",
                          "simulation/tests/test_*.py", "tests/test_*.py"],
                         cwd=ROOT, capture_output=True, text=True,
                         encoding="utf-8").stdout.split("\0")
    # split("\0") leaves a trailing empty string; `pytest ""` collects the whole
    # suite rather than one file, so every DB-dependent test runs and fails.
    return sorted(f for f in out if f)


def excluded() -> dict[str, str]:
    """path -> category. Missing file = the list is stale, which is worth knowing."""
    if not EXCLUSIONS.exists():
        return {}
    out: dict[str, str] = {}
    for line in EXCLUSIONS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "\t" not in line:
            continue
        cat, path = line.split("\t", 1)
        out[path.strip()] = cat.strip()
    return out


def run_one(path: str, timeout: int = 300) -> tuple[bool, str]:
    try:
        p = subprocess.run([sys.executable, "-m", "pytest", path, "-q"],
                           cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"
    lines = [ln for ln in (p.stdout or "").splitlines() if ln.strip()]
    summary = lines[-1] if lines else "(no output)"
    if p.returncode == 0:
        return True, summary
    # Carry the assertion text into the summary. Downloading a job log needs
    # admin rights on the repository, so without this a platform-specific
    # failure is a bare "1 failed" to anyone who does not have them — which is
    # exactly how the first Windows failure cost two rounds of guessing.
    detail = [ln.strip() for ln in lines
              if ln.lstrip().startswith(("E ", "FAILED", "assert "))][:6]
    if detail:
        summary = f"{summary} || " + " | ".join(d[:160] for d in detail)
    # pytest exits 5 for "no tests collected", which it ALSO returns when every
    # test in the file skipped at module level — an environment gate (no Rust
    # toolchain, no optional backend), not a failure. Distinguish the two by
    # whether anything was actually skipped; a file with no tests at all is a
    # stale exclusion worth surfacing.
    if p.returncode == 5:
        return ("skipped" in summary), summary
    return False, summary


def survey() -> int:
    print("# re-measured exclusion list - replace scripts/ci_test_exclusions.txt body\n")
    for f in test_files():
        p = subprocess.run([sys.executable, "-m", "pytest", f, "-q"],
                           cwd=ROOT, capture_output=True, text=True)
        if p.returncode == 0:
            continue
        tail = (p.stdout or "")[-6000:]
        cat = ("EMPTY" if p.returncode == 5 else
               "DEP" if DEP.search(tail) else
               "DATA" if DATA.search(tail) else "FAIL")
        print(f"{cat}\t{f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the plan, run nothing")
    ap.add_argument("--survey", action="store_true", help="re-measure exclusions")
    args = ap.parse_args()

    if args.survey:
        return survey()

    skip = excluded()
    allf = test_files()
    stale = sorted(set(skip) - set(allf))
    todo = [f for f in allf if f not in skip]

    print(f"test files tracked : {len(allf)}")
    print(f"excluded           : {len(skip)}  "
          f"({', '.join(f'{c}={sum(1 for v in skip.values() if v == c)}' for c in sorted(set(skip.values())))})")
    print(f"to run             : {len(todo)}")
    if stale:
        print(f"\nWARNING: {len(stale)} excluded path(s) no longer exist - the list is stale:")
        for f in stale:
            print(f"  {f}")
    if args.list:
        for f in todo:
            print(f"  {f}")
        return 1 if stale else 0

    # ::error:: lines become GitHub annotations, which the API exposes without
    # the admin rights that downloading a job log requires. Without them a
    # Windows-only failure is invisible to anyone who cannot open the log.
    import os
    annotate = os.environ.get("GITHUB_ACTIONS") == "true"

    failed = []
    for i, f in enumerate(todo, 1):
        ok, summary = run_one(f)
        if not ok:
            failed.append((f, summary))
            print(f"  [{i}/{len(todo)}] FAIL {f}: {summary}", flush=True)
            if annotate:
                print(f"::error file={f}::{summary}", flush=True)

    print(f"\n{len(todo) - len(failed)} passed, {len(failed)} failed")
    if annotate and failed:
        names = ", ".join(f for f, _ in failed[:10])
        print(f"::error::{len(failed)} test file(s) failed: {names}", flush=True)
    for f, s in failed:
        print(f"  FAIL {f}: {s}")
    if stale:
        print(f"\n{len(stale)} stale exclusion(s); re-run with --survey")
    return 1 if (failed or stale) else 0


if __name__ == "__main__":
    raise SystemExit(main())
