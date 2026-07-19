#!/usr/bin/env python3
"""Recompute ONLY the two rewritten count-family GLMs, in place, in the SSOT metrics CSV.

The thesis quotes ``simulation/results/per_model_eval/per_model_metrics.csv`` (48 models).
``NegBinGLM`` and ``PoissonAutoreg`` were reimplemented as the true log-link GLMs their names
claim, so exactly those two rows are stale — the other 46 must not move by a single digit.

A full pipeline re-run would move all 48 (Optuna is not bit-reproducible across runs, and the
active per_model_optimal JSONs are from a *different* run than the CSV). So this script
recomputes the two rows through the same code paths R9/R10 use, and rewrites only them.

FusedEpi is recomputed too — not to change it, but as a CONTROL: if this harness reproduces
the champion's committed row, it is faithful, and the 46 untouched rows are safe by
construction. The control row is compared, never written.

Env: the SSOT CSV was produced with static (non-adaptive) conformal WIS. MPH_ADAPTIVE_CONFORMAL
is pinned to 0 here — leaving it on gives NegBinGLM 3.62 where the CSV says 3.90.

Run:
    .venv/bin/python scripts/reeval_two_glms.py            # verify only (no write)
    .venv/bin/python scripts/reeval_two_glms.py --write    # rewrite the two rows
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

os.environ["MPH_ADAPTIVE_CONFORMAL"] = "0"   # must match the run that produced the CSV
os.environ.setdefault("MPH_EVAL_FEATURES", "basic")

import numpy as np  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from simulation.pipeline.config import PipelineConfig  # noqa: E402
from simulation.pipeline.data import run_data  # noqa: E402
from simulation.pipeline.per_model_optimize import _oof_cv_wis  # noqa: E402
from simulation.pipeline.per_model_eval import select_champion_g318  # noqa: E402
from simulation.pipeline.phase_evaluator import evaluate_predictions_full  # noqa: E402
from simulation.models.epi_models import (  # noqa: E402
    NegBinGLMForecaster, PoissonAutoregForecaster,
)

_CSV = _ROOT / "simulation" / "results" / "per_model_eval" / "per_model_metrics.csv"
_OPT = _ROOT / "simulation" / "results" / "per_model_optimal"
_TARGETS = ("NegBinGLM", "PoissonAutoreg")
_CONTROL = "FusedEpi"
_THRESHOLD = 8.6          # KDCA season epidemic threshold (real_eval.py)


def _factories():
    """Class factories. The two targets come from the rewritten module; the control comes from
    the live registry so it is exactly the class the pipeline itself would have fitted."""
    from simulation.models.fused_epi import FusedEpiForecaster

    return {
        "NegBinGLM": NegBinGLMForecaster,
        "PoissonAutoreg": PoissonAutoregForecaster,
        _CONTROL: FusedEpiForecaster,   # champion, imported directly (registry is lazy)
    }


def _row_for(name, fac, d):
    """Recompute one model's selection signals + evaluation row, exactly as R9/R10 do."""
    X = np.asarray(d["X_all"], float)
    y = np.asarray(d["y_all"], float).ravel()
    n_train = int(d["n_train"])
    pool_end = int(n_train + d.get("n_val", 0))
    n_test = int(d.get("n_test") or (len(y) - pool_end))

    # ── R9: OOF-WIS on the TRAIN block only (identity × none — both models are META-pinned)
    oof, folds = _oof_cv_wis(
        fac, X[:n_train], y[:n_train], "identity", "none",
        feature_indices=None, n_folds=5, return_folds=True,
    )

    # ── R9 refit: fit on train+val pool, predict the sealed test slab (1-step, observations fed)
    m = fac().fit(X[:pool_end], y[:pool_end])
    pred = np.asarray(m.predict(X[pool_end:pool_end + n_test]), float)
    y_test = y[pool_end:pool_end + n_test]

    # ── R10: leak-free residuals = in-sample (pool) residuals; sigma = their SD
    resid = y[:pool_end] - np.asarray(m.predict(X[:pool_end]), float)
    resid = resid[np.isfinite(resid)]
    sigma = float(max(np.std(resid), 1e-3))

    full = evaluate_predictions_full(
        y_test=y_test, y_pred=pred,
        residuals=resid, sigma=sigma,
        y_train_pool=None, threshold=_THRESHOLD, phase_id="R10",
    )
    full["oof_wis"] = float(oof)
    full["oof_wis_folds"] = list(map(float, folds)) if folds is not None else None
    full["pi_source"] = "r9_leakfree"
    return full, resid, pred


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    d = run_data(PipelineConfig())
    facs = _factories()
    rows = list(csv.DictReader(_CSV.open(encoding="utf-8")))
    by = {r["model"]: r for r in rows}
    print(f"SSOT CSV: {len(rows)} models × {len(rows[0])} columns\n")

    # ── CONTROL: the harness must reproduce the champion's committed row ──────
    if _CONTROL in facs:
        try:
            full, _, _ = _row_for(_CONTROL, facs[_CONTROL], d)
            old = by[_CONTROL]
            print(f"=== CONTROL {_CONTROL} (must match SSOT) ===")
            for k in ("oof_wis", "wis", "r2", "mae", "rmse"):
                o = float(old.get(k, "nan") or "nan")
                n = float(full.get(k, float("nan")))
                ok = np.isclose(o, n, rtol=2e-3, atol=2e-3, equal_nan=True)
                print(f"  {k:<10} SSOT={o:<10.4f} harness={n:<10.4f} {'✓' if ok else '✗ DRIFT'}")
        except Exception as exc:  # control is diagnostic only — never blocks
            print(f"=== CONTROL {_CONTROL}: could not refit ({type(exc).__name__}: {exc}) ===")
        print()

    # ── the two rows we actually replace ─────────────────────────────────────
    new_rows = {}
    for name in _TARGETS:
        full, resid, pred = _row_for(name, facs[name], d)
        old = by[name]
        print(f"=== {name} ===")
        for k in ("oof_wis", "wis", "r2", "mae", "rmse", "picp_95"):
            o = old.get(k, "")
            n = full.get(k, "")
            fo = f"{float(o):.4f}" if o not in ("", None) and str(o) != "nan" else "nan"
            fn = f"{float(n):.4f}" if isinstance(n, (int, float)) and np.isfinite(n) else "nan"
            print(f"  {k:<10} old={fo:<10} new={fn}")
        merged = dict(old)
        for k, v in full.items():
            if k in merged:
                merged[k] = v
        new_rows[name] = (merged, resid, pred)
        print()

    # ── champion re-selection on the spliced table ───────────────────────────
    spliced = []
    for r in rows:
        rr = dict(new_rows[r["model"]][0]) if r["model"] in new_rows else dict(r)
        for k in ("oof_wis", "wis", "n_features"):
            try:
                rr[k] = float(rr.get(k, "nan"))
            except (TypeError, ValueError):
                rr[k] = float("nan")
        fj = _OPT / f"{r['model']}.json"
        if r["model"] in new_rows:
            rr["oof_wis_folds"] = new_rows[r["model"]][0].get("oof_wis_folds")
        elif fj.exists():
            try:
                rr["oof_wis_folds"] = (json.load(fj.open()).get("val_metrics") or {}).get("oof_wis_folds")
            except Exception:
                rr["oof_wis_folds"] = None
        spliced.append(rr)

    ranked = sorted((r for r in spliced if np.isfinite(r["oof_wis"])), key=lambda r: r["oof_wis"])
    ch = select_champion_g318(spliced)
    print("=== 재선정 (spliced 48행) ===")
    print(f"  champion = {ch['model'] if ch else None}")
    for i, r in enumerate(ranked[:10], 1):
        mark = " ←NEW" if r["model"] in new_rows else ""
        print(f"  {i:2d}. {r['model']:<22} oof_wis={r['oof_wis']:.4f}{mark}")

    if not args.write:
        print("\n(verify only — nothing written; pass --write to commit)")
        return 0

    shutil.copy2(_CSV, _CSV.with_suffix(".csv.pre_glm"))
    cols = list(rows[0].keys())
    with _CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (new_rows[r["model"]][0].get(k, r.get(k)) if r["model"] in new_rows
                            else r.get(k)) for k in cols})
    print(f"\n✅ 2 rows rewritten; 46 untouched. backup: {_CSV.name}.pre_glm")

    for name, (merged, resid, pred) in new_rows.items():
        fj = _OPT / f"{name}.json"
        j = json.loads(fj.read_text(encoding="utf-8")) if fj.exists() else {"model": name}
        j.setdefault("val_metrics", {})
        j["val_metrics"]["oof_wis"] = merged["oof_wis"]
        j["val_metrics"]["oof_wis_folds"] = merged.get("oof_wis_folds")
        j["val_metrics"]["insample_residuals"] = [float(x) for x in resid]
        j["refit_test_predictions"] = [float(x) for x in pred]
        j["best_config"] = {"transform": "identity", "scaler": "none",
                            "n_features": None, "feature_indices": None}
        fj.write_text(json.dumps(j, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"✅ {fj.name} updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
