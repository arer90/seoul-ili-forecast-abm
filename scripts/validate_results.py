#!/usr/bin/env python3
"""Validate the shipped analysis results without recomputing them.

A stranger cloning this repository gets the result files but not the 13 GB
database, so they cannot re-run the pipeline to check that the published
numbers are coherent. This script does the next best thing: it verifies that
the shipped artifacts are internally consistent, structurally intact, and agree
with the numbers quoted in the README.

It deliberately uses the standard library only, so CI can run it on Linux,
Windows and macOS without installing torch (pyproject pins torch to a CUDA
index on linux/win32, which a GPU-less runner should not download).

What it proves: the results are present, parseable, correctly shaped, and the
headline claims trace to a file.
What it does NOT prove: that re-running the pipeline reproduces them. That
needs the database and a training run — see SETUP.md.

Exit code 0 = all checks pass, 1 = at least one failure.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "simulation" / "results"


def _set_root(root: Path) -> None:
    """Point the checks at another checkout — used by the guard test."""
    global ROOT, RESULTS, failures, passes
    ROOT = Path(root).resolve()
    RESULTS = ROOT / "simulation" / "results"
    failures, passes = [], 0

failures: list[str] = []
passes = 0

# Known-degenerate rows, pinned so the gate still catches anything new.
#
# These seven combination forecasters carry neither a test WIS nor an
# out-of-fold WIS, and their relative-WIS comes out as inf rather than a ratio.
# They are therefore unscored on the interval metric and are not part of any
# reported ranking — the champion and every quoted comparison come from the
# individually scored models. Listing them here documents the gap instead of
# hiding it; the check below fails if the set ever grows.
ENSEMBLES_WITHOUT_WIS = {
    "Ensemble-InvRMSE",
    "Ensemble-BMA",
    "Ensemble-NNLS",
    "Ensemble-NNLS-Filtered",
    "Ensemble-Diversity",
    "Ensemble-ResidualAR",
    "Ensemble-Adaptive",
}


def check(name: str, ok: bool, detail: str = "") -> None:
    global passes
    if ok:
        passes += 1
        print(f"  PASS  {name}")
    else:
        failures.append(f"{name}: {detail}")
        print(f"  FAIL  {name}  {detail}")


def load_json(rel: str):
    p = RESULTS / rel
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        failures.append(f"{rel}: invalid JSON ({e})")
        return None


def finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


# ── 1. the per-model metric table ────────────────────────────────────────────
def check_metrics_table() -> None:
    p = RESULTS / "per_model_eval" / "per_model_metrics.csv"
    if not p.exists():
        check("metrics table present", False, f"{p} missing")
        return
    with open(p, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    check("metrics table non-empty", len(rows) > 0, f"{len(rows)} rows")
    if not rows:
        # Nothing below can say anything meaningful about an empty table, and
        # every later check would index rows[0].
        return
    check("metrics table has a model column", "model" in rows[0], "no 'model' column")

    models = [r["model"] for r in rows]
    check("model names unique", len(models) == len(set(models)),
          f"{len(models)} rows, {len(set(models))} unique")

    # A blank test WIS is deliberate, not a defect. A model with no native
    # prediction interval gets NaN rather than an interval reconstructed from
    # test residuals, because that reconstruction would leak the test set. Such
    # models are still ranked, on their leak-free out-of-fold WIS. So the
    # invariant is on oof_wis, not wis.
    no_oof = [r["model"] for r in rows
              if not finite(_num(r.get("oof_wis"))) or _num(r.get("oof_wis")) <= 0]
    check("every model outside the ensemble family has a leak-free OOF WIS",
          set(no_oof) <= ENSEMBLES_WITHOUT_WIS,
          f"unexpected: {sorted(set(no_oof) - ENSEMBLES_WITHOUT_WIS)[:5]}")

    scored = [r["model"] for r in rows if finite(_num(r.get("wis")))]
    check("enough models carry a native-interval test WIS",
          len(scored) >= 20, f"only {len(scored)} of {len(rows)}")

    # A test WIS, where present, is a positive loss.
    nonpos = [r["model"] for r in rows
              if finite(_num(r.get("wis"))) and _num(r.get("wis")) <= 0]
    check("every present test WIS is positive", not nonpos, f"offenders: {nonpos[:5]}")

    # relative-WIS is a ratio against the FluSight baseline: finite and positive
    # wherever the model has a WIS to divide.
    col = "relative_wis_vs_baseline"
    if col in rows[0]:
        bad = [r["model"] for r in rows
               if (v := _num(r.get(col))) is not None and not finite(v)]
        check("relative-WIS is finite outside the known-degenerate ensembles",
              set(bad) <= ENSEMBLES_WITHOUT_WIS,
              f"unexpected: {sorted(set(bad) - ENSEMBLES_WITHOUT_WIS)[:5]}")

        ranked = sorted(
            ((_num(r.get(col)), r["model"]) for r in rows if finite(_num(r.get(col)))),
        )
        check("champion FusedEpi tops the relative-WIS leaderboard",
              bool(ranked) and ranked[0][1] == "FusedEpi",
              f"top-3: {[m for _, m in ranked[:3]]}")


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ── 2. phase checkpoints ─────────────────────────────────────────────────────
def check_checkpoints() -> None:
    d = RESULTS / "checkpoints"
    if not d.is_dir():
        check("checkpoints directory present", False, f"{d} missing")
        return
    files = sorted(d.glob("checkpoint_*.json"))
    check("checkpoints present", len(files) > 0, f"{len(files)} files")

    broken = []
    for f in files:
        try:
            json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            broken.append(f"{f.name} ({e})")
    check("every checkpoint parses", not broken, "; ".join(broken[:3]))

    r1 = load_json("checkpoints/checkpoint_R1.json")
    if r1 and isinstance(r1.get("data"), dict):
        n = r1["data"].get("n")
        cols = r1["data"].get("feature_cols")
        check("R1 records the sample size", isinstance(n, int) and n > 0, f"n={n}")
        check("R1 records the feature columns",
              isinstance(cols, list) and len(cols) > 0,
              f"{len(cols) if isinstance(cols, list) else 'missing'} features")


# ── 3. ABM forward validation ────────────────────────────────────────────────
def check_abm() -> None:
    d = load_json("abm_forward_validation/result.json")
    if d is None:
        check("ABM forward result present", False, "abm_forward_validation/result.json missing")
        return
    check("ABM forward result present", True)

    for key in ("forward_r2", "forward_r2_behavior_on", "forward_r2_behavior_off"):
        v = d.get(key)
        check(f"ABM {key} is finite", finite(v), f"{key}={v!r}")

    on, off = d.get("forward_r2_behavior_on"), d.get("forward_r2_behavior_off")
    if finite(on) and finite(off):
        check("behaviour-on beats behaviour-off", on > off, f"on={on:.4f} off={off:.4f}")

    for key in ("forward_r2", "forward_r2_behavior_on", "forward_r2_behavior_off"):
        v = d.get(key)
        if finite(v):
            check(f"ABM {key} within (-1, 1]", -1.0 < v <= 1.0, f"{key}={v}")


# ── 4. README headline numbers trace to a file ───────────────────────────────
def check_readme_traceability() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    abm = load_json("abm_forward_validation/result.json") or {}
    claims: list[tuple[str, float | None]] = [
        ("0.722", abm.get("forward_r2")),
        ("0.557", abm.get("forward_r2_behavior_on")),
        ("0.041", abm.get("forward_r2_behavior_off")),
    ]
    for text, actual in claims:
        if text not in readme:
            continue
        ok = actual is not None and abs(round(actual, 3) - float(text)) < 5e-4
        check(f"README {text} matches the result file",
              ok, f"file value {actual!r}")

    p = RESULTS / "per_model_eval" / "per_model_metrics.csv"
    if p.exists() and "3.28" in readme:
        with open(p, encoding="utf-8") as fh:
            fused = [r for r in csv.DictReader(fh) if r.get("model") == "FusedEpi"]
        wis = _num(fused[0].get("wis")) if fused else None
        check("README WIS 3.28 matches the champion row",
              wis is not None and abs(round(wis, 2) - 3.28) < 5e-3,
              f"file value {wis!r}")


# ── 5. web aggregates the dashboard reads ────────────────────────────────────
def check_aggregates() -> None:
    d = ROOT / "web" / "public" / "aggregates"
    if not d.is_dir():
        check("web aggregates present", False, f"{d} missing")
        return
    files = sorted(d.glob("*.json"))
    check("web aggregates present", len(files) > 0, f"{len(files)} files")
    broken = []
    for f in files:
        try:
            json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            broken.append(f"{f.name} ({e})")
    check("every aggregate parses", not broken, "; ".join(broken[:3]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=None,
                    help="checkout to validate (default: this one)")
    args = ap.parse_args(argv)
    if args.root is not None:
        _set_root(args.root)

    print(f"validating shipped results under {RESULTS.relative_to(ROOT)}\n")
    check_metrics_table()
    check_checkpoints()
    check_abm()
    check_readme_traceability()
    check_aggregates()
    print(f"\n  {passes} passed, {len(failures)} failed")
    for f in failures:
        print(f"    - {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
