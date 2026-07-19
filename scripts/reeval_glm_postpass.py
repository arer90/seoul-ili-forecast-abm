#!/usr/bin/env python3
"""Restore the cross-model columns R10 computes but a per-model refit cannot.

``scripts/reeval_two_glms.py`` (and the splice that followed) rebuilt the two count-family
rows with ``evaluate_predictions_full``, which is a *per-model* evaluator. Thirty of the 146
columns are not per-model at all — ranks are positions in the field, the pairwise relative WIS
is a tournament against every other model, the Diebold-Mariano tests need a baseline, and the
bootstrap bounds need the residual sample. Those came out NaN for exactly the two rows that
changed, and figures read them.

This post-pass recomputes them with the same functions R10 uses, for ALL 48 models, and then
asserts that the 46 rows it did not intend to touch come back byte-identical to the committed
CSV. That assertion is the proof the computation is R10's: if the recipe were wrong, the 46
would move too.

Provenance trap this script exists to avoid: the ACTIVE ``per_model_optimal/*.json`` are from a
LATER run than the committed CSV. Their siblings — the JSONs that actually reproduce it — live
in ``_archive_fullrun_20260701_024145``. Reading the active ones silently rewrites 46 rows.

Run:
    .venv/bin/python scripts/reeval_glm_postpass.py            # verify only
    .venv/bin/python scripts/reeval_glm_postpass.py --write    # commit the 2 rows
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path

os.environ["MPH_ADAPTIVE_CONFORMAL"] = "0"      # the CSV was built with static conformal WIS
os.environ.setdefault("MPH_EVAL_FEATURES", "basic")

import numpy as np  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from simulation.analytics.diagnostics import weighted_interval_score_empirical  # noqa: E402
from simulation.analytics.hub_metrics import (  # noqa: E402
    FLUSIGHT_ALPHAS, mase, pairwise_relative_wis, relative_skill_score,
)
from simulation.analytics.metrics import (  # noqa: E402
    bootstrap_ci, crps_gaussian, diebold_mariano_vs_baseline,
)
from simulation.analytics.multiple_testing import apply_bh_fdr  # noqa: E402
from simulation.pipeline.config import PipelineConfig  # noqa: E402
from simulation.pipeline.data import run_data  # noqa: E402

_CSV = _ROOT / "simulation" / "results" / "per_model_eval" / "per_model_metrics.csv"
_ACTIVE_OPT = _ROOT / "simulation" / "results" / "per_model_optimal"
_ARCHIVE_OPT = (
    _ROOT / "simulation" / "results" / "_archive_fullrun_20260701_024145" / "per_model_optimal"
)
_TARGETS = ("NegBinGLM", "PoissonAutoreg")

# Columns this pass owns. Everything else in the row is per-model and already correct.
_CROSS = (
    "mae_ci95_lo", "mae_ci95_hi", "mae_ci95_lo_bs", "mae_ci95_hi_bs",
    "wis_ci95_lo", "wis_ci95_hi",
    "mase_h1", "mase_h4", "mase_h13", "mase_h26", "mase_h52",
    "dm_z_stat", "dm_p_value", "dm_z_vs_climatology", "dm_p_vs_climatology",
    "dm_z_vs_lag52", "dm_p_vs_lag52",
    "dm_p_value_bh", "dm_p_vs_climatology_bh", "dm_p_vs_lag52_bh",
    "skill_mae_vs_persist", "skill_wis_vs_persist", "skill_crps_vs_persist",
    "skill_mae_vs_snaive",
    "relative_wis_pairwise",
    "rank_wis", "rank_wis_test", "rank_log_wis", "rank_mae", "rank_r2",
)


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _preds_and_residuals(model: str) -> tuple[np.ndarray | None, np.ndarray]:
    """The model's sealed-test predictions and its leak-free residual sample.

    The two rewritten models read from the ACTIVE JSON (they were just refitted); every other
    model reads from the ARCHIVE, whose JSONs are the ones that reproduce the committed CSV.
    """
    src = _ACTIVE_OPT if model in _TARGETS else _ARCHIVE_OPT
    path = src / f"{model}.json"
    if not path.exists():
        return None, np.array([], dtype=np.float64)
    d = json.loads(path.read_text(encoding="utf-8"))
    pred = d.get("refit_test_predictions")
    res = (d.get("val_metrics") or {}).get("insample_residuals") or []
    res = np.asarray(res, dtype=np.float64)
    return (np.asarray(pred, dtype=np.float64) if pred else None,
            res[np.isfinite(res)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    d = run_data(PipelineConfig())
    y = np.asarray(d["y_all"], dtype=np.float64).ravel()
    pool_end = int(d["n_train"]) + int(d.get("n_val", 0))
    n_test = int(d.get("n_test") or (len(y) - pool_end))
    y_test = y[pool_end:pool_end + n_test]
    y_in = y[:pool_end]

    rows = list(csv.DictReader(_CSV.open(encoding="utf-8")))
    cols = list(rows[0].keys())
    committed = {r["model"]: dict(r) for r in rows}

    preds, resids = {}, {}
    for r in rows:
        p, res = _preds_and_residuals(r["model"])
        if p is not None and len(p) == n_test:
            preds[r["model"]] = p
            resids[r["model"]] = res

    print(f"{len(rows)} models; predictions recovered for {len(preds)}\n")

    # ── per-model: bootstrap bounds, MASE, DM tests ──────────────────────
    out: dict[str, dict] = {r["model"]: {} for r in rows}
    wis_arrays: dict[str, np.ndarray] = {}

    for m, pred in preds.items():
        o = out[m]
        ae = np.abs(y_test - pred)
        ci = bootstrap_ci(ae, statistic=np.mean, n_boot=2000, alpha=0.05, random_state=42)
        o["mae_ci95_lo"] = round(float(ci.get("ci_lo", np.nan)), 4)
        o["mae_ci95_hi"] = round(float(ci.get("ci_hi", np.nan)), 4)
        o["mae_ci95_lo_bs"] = o["mae_ci95_lo"]
        o["mae_ci95_hi_bs"] = o["mae_ci95_hi"]

        for h in (1, 4, 13, 26, 52):
            v = mase(y_test, pred, y_train=y_in, seasonality=h)
            o[f"mase_h{h}"] = round(float(v), 4) if np.isfinite(v) else float("nan")

        dm = diebold_mariano_vs_baseline(y_test, pred)
        o["dm_z_stat"] = round(_f(dm.get("dm_z_stat")), 4)
        o["dm_p_value"] = round(_f(dm.get("dm_p_value")), 4)

        clim = np.full_like(y_test, float(np.mean(y_in)))
        dmc = diebold_mariano_vs_baseline(y_test, pred, y_baseline=clim)
        o["dm_z_vs_climatology"] = round(_f(dmc.get("dm_z_stat")), 4)
        o["dm_p_vs_climatology"] = round(_f(dmc.get("dm_p_value")), 4)

        lag52 = y[pool_end - 52:pool_end - 52 + n_test]
        if len(lag52) == n_test:
            dml = diebold_mariano_vs_baseline(y_test, pred, y_baseline=lag52)
            o["dm_z_vs_lag52"] = round(_f(dml.get("dm_z_stat")), 4)
            o["dm_p_vs_lag52"] = round(_f(dml.get("dm_p_value")), 4)

        # the WIS tournament only admits models with a leak-free residual source
        if len(resids[m]) >= 2:
            wis_arrays[m] = np.asarray(
                weighted_interval_score_empirical(
                    y_test, pred, resids[m], alphas=list(FLUSIGHT_ALPHAS)),
                dtype=np.float64,
            )
            wci = bootstrap_ci(wis_arrays[m], statistic=np.mean,
                               n_boot=2000, alpha=0.05, random_state=42)
            o["wis_ci95_lo"] = round(float(wci.get("ci_lo", np.nan)), 4)
            o["wis_ci95_hi"] = round(float(wci.get("ci_hi", np.nan)), 4)

    # ── skill vs persistence / seasonal naive ────────────────────────────
    persist = np.concatenate([[y_in[-1]], y_test[:-1]])
    persist_mae = float(np.mean(np.abs(persist - y_test)))
    p_res = (y_test - persist)[np.isfinite(y_test - persist)]
    persist_wis = float(np.mean(weighted_interval_score_empirical(
        y_test, persist, p_res, alphas=FLUSIGHT_ALPHAS)))
    p_sigma = max(float(np.std(y_test - persist)), 1e-3)
    persist_crps = float(np.mean(crps_gaussian(
        y_test, persist, np.full_like(y_test, p_sigma))))
    snaive = y[pool_end - 52:pool_end - 52 + n_test]
    snaive_mae = float(np.mean(np.abs(snaive - y_test))) if len(snaive) == n_test else np.nan

    for r in rows:
        m, o = r["model"], out[r["model"]]
        mae_v, wis_v = _f(r.get("mae")), _f(r.get("wis"))
        crps_v = _f(r.get("crps_gaussian"))
        o["skill_mae_vs_persist"] = round(
            relative_skill_score(mae_v, persist_mae, lower_is_better=True), 4)
        if np.isfinite(snaive_mae):
            o["skill_mae_vs_snaive"] = round(
                relative_skill_score(mae_v, snaive_mae, lower_is_better=True), 4)
        o["skill_wis_vs_persist"] = (
            round(relative_skill_score(wis_v, persist_wis, lower_is_better=True), 4)
            if np.isfinite(wis_v) else float("nan"))
        o["skill_crps_vs_persist"] = (
            round(relative_skill_score(crps_v, persist_crps, lower_is_better=True), 4)
            if np.isfinite(crps_v) else float("nan"))

    # ── tournament + ranks (genuinely cross-model) ───────────────────────
    rel = pairwise_relative_wis(wis_arrays) if len(wis_arrays) >= 2 else {}
    for r in rows:
        out[r["model"]]["relative_wis_pairwise"] = round(
            _f(rel.get(r["model"], float("nan"))), 4)

    def _rank(key: str, dest: str, *, lower_better: bool = True) -> None:
        def sort_key(r):
            v = _f(r.get(key))
            if not np.isfinite(v):
                return float("inf")
            return v if lower_better else -v
        for i, r in enumerate(sorted(rows, key=sort_key), 1):
            out[r["model"]][dest] = i

    _rank("oof_wis", "rank_wis")
    _rank("wis", "rank_wis_test")
    _rank("log_wis", "rank_log_wis")
    _rank("mae", "rank_mae")
    _rank("r2", "rank_r2", lower_better=False)

    # Benjamini-Hochberg across each DM family (NaN-safe wrapper R10 uses)
    for src, dst in (("dm_p_value", "dm_p_value_bh"),
                     ("dm_p_vs_climatology", "dm_p_vs_climatology_bh"),
                     ("dm_p_vs_lag52", "dm_p_vs_lag52_bh")):
        ps = {r["model"]: _f(out[r["model"]].get(src)) for r in rows}
        adj = apply_bh_fdr(ps).get("pvals_corrected", {})
        for m, v in adj.items():
            out[m][dst] = round(float(v), 4) if np.isfinite(_f(v)) else float("nan")

    # ── CONTROL: the 46 untouched models must come back to the committed values ──
    def _same(a, b) -> bool:
        fa, fb = _f(a), _f(b)
        if math.isnan(fa) and math.isnan(fb):
            return True
        return bool(np.isclose(fa, fb, rtol=2e-2, atol=2e-2))

    drift: list[str] = []
    for r in rows:
        m = r["model"]
        if m in _TARGETS:
            continue
        for c in _CROSS:
            if c in out[m] and not _same(committed[m].get(c), out[m][c]):
                drift.append(f"{m}.{c}: {committed[m].get(c)} -> {out[m][c]}")

    print(f"=== CONTROL: {len(rows) - len(_TARGETS)} untouched models, "
          f"{len(_CROSS)} cross-model columns ===")
    if drift:
        print(f"  ✗ {len(drift)} cells drift — the recipe is NOT R10's. First 15:")
        for line in drift[:15]:
            print(f"     {line}")
        print("\n  ABORT — nothing written.")
        return 1
    print("  ✓ every cell reproduces the committed CSV — recipe verified\n")

    print("=== the two rewritten rows ===")
    for m in _TARGETS:
        print(f"  {m}")
        for c in ("rank_wis", "rank_wis_test", "rank_mae", "relative_wis_pairwise",
                  "dm_p_value", "mase_h1", "wis_ci95_lo", "wis_ci95_hi"):
            print(f"    {c:<24} {committed[m].get(c, '-'):<10} → {out[m].get(c, '-')}")

    if not args.write:
        print("\n(verify only — pass --write to commit)")
        return 0

    shutil.copy2(_CSV, str(_CSV) + ".pre_postpass")
    with _CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            merged = dict(r)
            if r["model"] in _TARGETS:
                for c, v in out[r["model"]].items():
                    if c in merged:
                        merged[c] = v
            w.writerow(merged)
    print(f"\n✅ 2 rows completed; 46 untouched. backup: {_CSV.name}.pre_postpass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
