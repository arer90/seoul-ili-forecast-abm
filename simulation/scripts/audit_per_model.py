"""Per-model problem auditor for phase-13 per_model_optimal/ JSONs.

Flags GENUINE problems only (refit-null on non-ensemble, OOF/WIS blow-up, feature collapse,
JSON read failure). Does NOT alert on negative hold-out test R² — that is the documented
structural limitation of deep/classic-TS models on the 68-step single-extrapolation stress
test (selection is on OOF-WIS), not a per-model bug. Use --all to also print those.

Usage:
    python -m simulation.scripts.audit_per_model [DIR] [--all]
Exit code: 1 if any genuine problem, else 0.
"""
import glob
import json
import math
import os
import sys


def audit(d: str, show_all: bool = False):
    problems, structural, done = [], [], []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        n = os.path.basename(f)[:-5]
        if n == "summary":
            continue
        done.append(n)
        try:
            with open(f, encoding="utf-8") as _fh:
                j = json.load(_fh)
        except Exception as e:  # noqa: BLE001
            problems.append((n, f"JSON_READ_FAIL:{type(e).__name__}"))
            continue
        vm = j.get("val_metrics") or {}
        tm = j.get("test_metrics") or {}
        oof = vm.get("oof_wis") if isinstance(vm, dict) else None
        tr2 = tm.get("r2") if isinstance(tm, dict) else None
        twis = tm.get("wis") if isinstance(tm, dict) else None
        fi = j.get("feature_indices")
        nf = len(fi) if isinstance(fi, list) else None
        flags = []
        # GENUINE problems (alert):
        if not n.startswith("Ensemble-") and tr2 is None:
            flags.append("REFIT-NULL(test없음)")
        if isinstance(oof, (int, float)) and (math.isinf(oof) or oof > 1e4):
            flags.append("OOF발산")
        if isinstance(twis, (int, float)) and twis > 1e3:
            flags.append(f"WIS폭발={twis:.0f}")
        if isinstance(nf, int) and nf <= 3:
            flags.append(f"feature붕괴(n={nf})")
        if flags:
            problems.append((n, "; ".join(flags)))
        # structural (negative test R² — expected, not alerted):
        elif isinstance(tr2, (int, float)) and tr2 < 0:
            structural.append((n, f"음수testR2={tr2:.2f}(구조적)"))
    return problems, structural, done


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    show_all = "--all" in sys.argv
    d = args[0] if args else "simulation/results/per_model_optimal"
    problems, structural, done = audit(d, show_all)
    for n, fl in problems:
        print(f"⚠ {n}: {fl}")
    if show_all:
        for n, fl in structural:
            print(f"· {n}: {fl}")
    print(f"[audit] {len(done)}/53 완료 · 진짜문제 {len(problems)} · 구조적음수R2 {len(structural)}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
