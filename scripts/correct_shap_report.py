#!/usr/bin/env python3
"""Re-derive the R11 SHAP summary and report from the attributions on disk.

`shap_analysis.py` decided whether a model had been explained by testing a
ranking for truthiness. `_permutation_importance` starts from `np.zeros(p)` and
returns one entry per feature no matter what happened, so a run that measured
nothing still produced a full list of `(feature, 0.0)` pairs — and the flag came
back True. A stable sort over that all-zero vector then emitted the original
column order, so `temp_avg, humidity, wind_speed, rainfall` was written out as a
top-feature ranking for models nothing had been measured on.

The code now applies `_measured()` at each of those decision points, which fixes
future runs. This script applies the same rule to the artifacts already shipped:
recount the summary, drop all-zero SHAP matrices, and rewrite the report so a
model that was not explained says so.

    python scripts/correct_shap_report.py [--dry-run]

Exit 0 on success.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SHAP = ROOT / "simulation" / "results" / "shap"

NOT_MEASURED = "not measured"


def _column(rows: list[dict], key: str) -> list[float]:
    out = []
    for r in rows:
        try:
            out.append(float(r[key]))
        except (TypeError, ValueError, KeyError):
            pass
    return out


def _measured(values: list[float]) -> bool:
    """Same rule as shap_analysis._measured: at least one non-zero attribution."""
    return bool(values) and any(v != 0.0 and np.isfinite(v) for v in values)


def survey() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for d in sorted(x for x in SHAP.iterdir() if x.is_dir()):
        f = d / "importance.csv"
        if not f.exists():
            continue
        rows = list(csv.DictReader(f.open(encoding="utf-8")))
        perm_ok = _measured(_column(rows, "permutation_importance"))
        nat_ok = _measured(_column(rows, "native_shap_importance"))
        npy = d / "shap_values.npy"
        sv_zero = False
        if npy.exists():
            try:
                sv_zero = not np.any(np.load(npy) != 0.0)
            except Exception:
                sv_zero = False
        out[d.name] = {"perm": perm_ok, "native": nat_ok,
                       "npy": npy if npy.exists() else None, "npy_zero": sv_zero}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not SHAP.is_dir():
        print(f"{SHAP} absent")
        return 1
    st = survey()
    n_perm = sum(1 for v in st.values() if v["perm"])
    n_nat = sum(1 for v in st.values() if v["native"])
    zero_npy = [k for k, v in st.items() if v["npy_zero"]]

    summary_path = SHAP / "_summary.json"
    old = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    print(f"models with an importance.csv : {len(st)}")
    print(f"  permutation measured        : {n_perm}   (summary said {old.get('n_with_permutation')})")
    print(f"  native SHAP measured        : {n_nat}   (summary said {old.get('n_with_native_shap')})")
    print(f"  all-zero shap_values.npy    : {len(zero_npy)}  {zero_npy}")
    if args.dry_run:
        print("(--dry-run) nothing written")
        return 0

    # 1) all-zero SHAP matrices explain nothing; a reader loading one gets a
    #    correctly shaped matrix of zeros with no way to tell.
    for name in zero_npy:
        st[name]["npy"].unlink()
        print(f"  removed {name}/shap_values.npy (all zero)")

    # 2) summary counts
    old.update({
        "n_with_permutation": n_perm,
        "n_with_native_shap": n_nat,
        "counting_rule": (
            "a ranking counts as measured only if at least one attribution is "
            "non-zero (shap_analysis._measured); an all-zero result means the "
            "attribution could not be measured, not that every feature is "
            "irrelevant"
        ),
        "corrected": "scripts/correct_shap_report.py, 2026-07-19",
    })
    summary_path.write_text(json.dumps(old, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  rewrote {summary_path.name}")

    # 3) report rows
    rp = SHAP / "REPORT.md"
    if not rp.exists():
        return 0
    lines = rp.read_text(encoding="utf-8").splitlines()
    out, changed = [], 0
    row = re.compile(r"^\|\s*([^|]+?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|$")
    for ln in lines:
        m = row.match(ln)
        if not m or m.group(1) in ("model", "-------"):
            if ln.startswith("- Models explained:"):
                ln = (f"- Models explained: **{len(st)}** "
                      f"(permutation {n_perm}, native SHAP {n_nat})")
            out.append(ln)
            continue
        name, fam, perm_c, nat_c, top = m.groups()
        s = st.get(name)
        if not s:
            out.append(ln)
            continue
        new_perm = "✓" if s["perm"] else NOT_MEASURED
        new_nat = "✓" if s["native"] else NOT_MEASURED
        new_top = top if (s["perm"] or s["native"]) else "—"
        if (new_perm, new_nat, new_top) != (perm_c, nat_c, top):
            changed += 1
        out.append(f"| {name} | {fam} | {new_perm} | {new_nat} | {new_top} |")

    note = (
        "\n> **Corrected 2026-07-19.** Rows previously showed ✓ for models whose "
        "attributions were entirely zero, with the first four feature columns "
        "(`temp_avg, temp_min, humidity, wind_speed` and similar) printed as their "
        "top drivers — an artifact of sorting a zero vector, not a measurement. "
        "Those rows now read *not measured*. See `scripts/correct_shap_report.py`.\n"
    )
    text = "\n".join(out)
    if "Corrected 2026-07-19" not in text:
        text = text.replace("## Per-model", note + "\n## Per-model", 1)
    rp.write_text(text + "\n", encoding="utf-8")
    print(f"  rewrote REPORT.md ({changed} rows corrected)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
