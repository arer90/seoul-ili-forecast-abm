#!/usr/bin/env python3
"""Re-run R10 itself over the 48 models, with only the two count GLMs changed.

Reimplementing R10's cross-model post-pass by hand does not work: ``y_in`` is the whole
in-sample array (so the persistence benchmark starts from ``y_all[-1]``, not the last training
week), the ranks are assigned by re-sorting an already-OOF-sorted list (so ties break by OOF),
and the WIS confidence bound is a block bootstrap that recomputes the empirical WIS on
resampled indices. A hand copy drifted on 286 cells and the control caught it.

So call the real thing. ``run_per_model_eval`` needs three inputs:

  * ``phase1``  — from ``run_data``.
  * ``all_results["wfcv"]["oof_predictions"]`` — R10's Source 1. Each model's sealed-test
    predictions are injected here, on a full-length array whose in-sample slots are NaN.
  * ``all_results["per_model_optimize"]["per_model_configs"]`` — carries ``val_metrics``
    (OOF-WIS, its folds, and the leak-free residuals R10 builds the intervals from).

``refit_test_predictions`` is deliberately NOT put in ``per_model_configs``: with it present
the current code also emits 41 ``name[fs]`` rows, and the committed 48-row table has none.

Provenance: the 46 unchanged models are read from ``_archive_fullrun_20260701_024145`` — the
JSONs that actually reproduce the committed CSV. The ACTIVE ``per_model_optimal`` is a later
run and reading it would silently move every row.

Run:
    .venv/bin/python scripts/reeval_glm_r10.py            # verify (writes to a scratch dir)
    .venv/bin/python scripts/reeval_glm_r10.py --write    # commit the two rows
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ["MPH_ADAPTIVE_CONFORMAL"] = "0"      # the committed CSV uses static conformal WIS
os.environ.setdefault("MPH_EVAL_FEATURES", "basic")

import numpy as np  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from simulation.pipeline.config import PipelineConfig  # noqa: E402
from simulation.pipeline.data import run_data  # noqa: E402
from simulation.pipeline.per_model_eval import run_per_model_eval  # noqa: E402

_CSV = _ROOT / "simulation" / "results" / "per_model_eval" / "per_model_metrics.csv"
_ACTIVE = _ROOT / "simulation" / "results" / "per_model_optimal"
_ARCHIVE = (
    _ROOT / "simulation" / "results" / "_archive_fullrun_20260701_024145" / "per_model_optimal"
)
_TARGETS = ("NegBinGLM", "PoissonAutoreg")

# Columns a per-model refit cannot produce — the ones this run exists to restore.
_CROSS = (
    "rank_wis", "rank_wis_test", "rank_log_wis", "rank_mae", "rank_r2",
    "relative_wis_pairwise", "wis_ci95_lo", "wis_ci95_hi",
    "mae_ci95_lo", "mae_ci95_hi", "mae_ci95_lo_bs", "mae_ci95_hi_bs",
    "mase_h1", "mase_h4", "mase_h13", "mase_h26", "mase_h52",
    "dm_z_stat", "dm_p_value", "dm_p_value_bh",
    "dm_z_vs_climatology", "dm_p_vs_climatology", "dm_p_vs_climatology_bh",
    "dm_z_vs_lag52", "dm_p_vs_lag52", "dm_p_vs_lag52_bh",
    "skill_mae_vs_persist", "skill_wis_vs_persist", "skill_crps_vs_persist",
    "skill_mae_vs_snaive",
)


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _load(model: str) -> dict | None:
    src = _ACTIVE if model in _TARGETS else _ARCHIVE
    path = src / f"{model}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    d = run_data(PipelineConfig())
    y = np.asarray(d["y_all"], dtype=np.float64).ravel()
    n_total = len(y)
    pool_end = int(d["n_train"]) + int(d.get("n_val", 0))
    n_test = int(d.get("n_test") or (n_total - pool_end))

    committed = {r["model"]: dict(r)
                 for r in csv.DictReader(_CSV.open(encoding="utf-8"))}
    models = list(committed)

    oof_preds: dict[str, np.ndarray] = {}
    configs: dict[str, dict] = {}
    missing: list[str] = []

    for m in models:
        j = _load(m)
        pred = (j or {}).get("refit_test_predictions")
        if not pred or len(pred) != n_test:
            missing.append(m)
            continue
        # Source 1 shape: a full-length array; R10 slices [test_start:test_end] out of it.
        arr = np.full(n_total, np.nan, dtype=np.float64)
        arr[pool_end:pool_end + n_test] = np.asarray(pred, dtype=np.float64)
        oof_preds[m] = arr

        vm = (j.get("val_metrics") or {})
        configs[m] = {
            # NO refit_test_predictions here — it would spawn 41 "[fs]" rows the committed
            # table does not have.
            "val_metrics": {
                "oof_wis": vm.get("oof_wis"),
                "oof_wis_folds": vm.get("oof_wis_folds"),
                "insample_residuals": vm.get("insample_residuals"),
            },
            "best_config": j.get("best_config", {}),
        }

    print(f"{len(models)} models; predictions injected for {len(oof_preds)}"
          + (f"; no R9 artifact for {missing}" if missing else ""))

    all_results = {
        "wfcv": {"oof_predictions": oof_preds},
        "per_model_optimize": {"per_model_configs": configs},
    }

    out_dir = Path(tempfile.mkdtemp(prefix="r10_"))
    cfg = PipelineConfig()
    cfg.save_dir = str(out_dir)              # never write into results/ while verifying
    res = run_per_model_eval(d, all_results, cfg)

    fresh_path = Path(res.get("metrics_csv") or (out_dir / "per_model_eval" / "per_model_metrics.csv"))
    fresh = {r["model"]: dict(r) for r in csv.DictReader(fresh_path.open(encoding="utf-8"))}
    print(f"R10 produced {len(fresh)} rows -> {fresh_path}\n")

    extra = [m for m in fresh if m not in committed]
    if extra:
        print(f"  ⚠ unexpected extra rows ({len(extra)}): {extra[:6]}")

    # ── CONTROL: the 46 models we did not touch must land on the committed numbers ──
    def _same(a, b) -> bool:
        fa, fb = _f(a), _f(b)
        if math.isnan(fa) and math.isnan(fb):
            return True
        return bool(np.isclose(fa, fb, rtol=2e-2, atol=2e-2))

    drift = [
        f"{m}.{c}: {committed[m].get(c)} -> {fresh[m].get(c)}"
        for m in committed if m not in _TARGETS and m in fresh
        for c in _CROSS
        if c in fresh[m] and not _same(committed[m].get(c), fresh[m].get(c))
    ]
    checked = len([m for m in committed if m not in _TARGETS and m in fresh])
    print(f"=== CONTROL: {checked} untouched models x {len(_CROSS)} cross-model columns ===")
    if drift:
        print(f"  ✗ {len(drift)} cells drift — this is not the committed run's recipe. First 12:")
        for line in drift[:12]:
            print(f"     {line}")
        print("\n  ABORT — results/ untouched.")
        return 1
    print("  ✓ all reproduce the committed CSV — R10 wiring verified\n")

    print("=== the two rewritten rows ===")
    for m in _TARGETS:
        print(f"  {m}")
        for c in ("oof_wis", "wis", "r2", "rank_wis", "rank_wis_test", "rank_mae",
                  "relative_wis_pairwise", "dm_p_value", "wis_ci95_lo", "wis_ci95_hi"):
            print(f"    {c:<24} {committed[m].get(c, '-'):<12} → {fresh[m].get(c, '-')}")

    if not args.write:
        print(f"\n(verify only — scratch output in {out_dir})")
        return 0

    shutil.copy2(_CSV, str(_CSV) + ".pre_r10")
    cols = list(next(iter(committed.values())).keys())
    with _CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for m, row in committed.items():
            src = fresh[m] if m in _TARGETS and m in fresh else row
            w.writerow({c: src.get(c, row.get(c)) for c in cols})
    print(f"\n✅ 2 rows written from R10; 46 untouched. backup: {_CSV.name}.pre_r10")
    return 0


if __name__ == "__main__":
    sys.exit(main())
