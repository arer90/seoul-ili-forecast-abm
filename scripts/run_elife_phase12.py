"""
Phase 12 KR — eLife Format Per-Model Observed vs Predicted (54 models).

Uses `refit_test_predictions` from per_model_optimal/<MODEL>.json (n=68 hold-out).
Ground truth = last 68 weeks of KDCA sentinel_influenza.

OUTPUTS:
    simulation/results/phase12_elife/
    ├── predictions_{MODEL}.csv             # per-model predictions
    ├── plots/elife_{MODEL}.png             # eLife Figure J per model
    ├── elife_phase12_grid.png              # 54-panel grid
    └── elife_phase12_table2_master.csv     # Table 2 master (54 rows)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulation.analytics.elife_format import (
    adapt_phase12_json, compute_elife_metrics,
    plot_elife_grid, plot_elife_single,
    write_elife_table2_csv, write_predictions_csv,
)
from simulation.database import safe_connect  # G-116 SSOT
from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

REPO = Path(__file__).parent.parent
DB = REPO / "simulation" / "data" / "db" / "epi_real_seoul.db"  # DB 는 redirect 안 됨 (project-local)
# results 는 MPH_OUTPUT_ROOT 존중 — pipeline 이 per_model_optimal 을 save_dir 로 redirect 하므로 read 도 동일 base.
_RESULTS = get_results_dir()
P12_DIR = _RESULTS / "per_model_optimal"   # read: Phase 12 산출 (redirect 됨)
OUT = _RESULTS / "phase12_elife"           # write: 누수 방지


def load_kr_holdout(n: int = 68) -> tuple[np.ndarray, list[str]]:
    """Load last n weeks of KR sentinel as ground truth (test set)."""
    conn = safe_connect(str(DB))
    rows = conn.execute(
        "SELECT season_start, week_seq, AVG(ili_rate) FROM sentinel_influenza "
        "WHERE ili_rate IS NOT NULL "
        "GROUP BY season_start, week_seq ORDER BY season_start, week_seq"
    ).fetchall()
    conn.close()
    if len(rows) < n:
        raise ValueError(f"sentinel_influenza only has {len(rows)} weeks, need >= {n}")
    last_n = rows[-n:]
    y_obs = np.array([r[2] for r in last_n], dtype=float)
    labels = [f"{r[0]}W{r[1]:02d}" for r in last_n]
    return y_obs, labels


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plots_dir = OUT / "plots"
    plots_dir.mkdir(exist_ok=True)
    preds_dir = OUT / "predictions"
    preds_dir.mkdir(exist_ok=True)

    # KR ground truth (last 68 weeks)
    y_obs, x_labels = load_kr_holdout(68)
    log.info(f"KR ground truth: {len(y_obs)} weeks ({x_labels[0]} → {x_labels[-1]})")

    # Iterate Phase 12 JSON files
    items = []
    failed = []
    for json_path in sorted(P12_DIR.glob("*.json")):
        if json_path.name.startswith("_") or json_path.name == "summary.json":
            continue
        try:
            d = json.load(open(json_path, encoding="utf-8"))
        except Exception as e:
            log.warning(f"  {json_path.name}: parse fail ({e})")
            failed.append(json_path.name)
            continue

        result = adapt_phase12_json(d)
        if result is None:
            failed.append(json_path.name)
            continue
        y_pred, model_name = result

        if len(y_pred) != len(y_obs):
            log.warning(f"  {model_name}: pred len={len(y_pred)} != obs len={len(y_obs)}, SKIP")
            failed.append(model_name)
            continue

        # Sanitize NaN/Inf
        if not np.all(np.isfinite(y_pred)):
            log.warning(f"  {model_name}: NaN/Inf in predictions, SKIP")
            failed.append(model_name)
            continue

        metrics = compute_elife_metrics(y_obs, y_pred)
        items.append({
            "name": model_name,
            "y_obs": y_obs, "y_pred": y_pred,
            "metrics": metrics,
        })

        write_predictions_csv(
            model_name, y_obs, y_pred,
            preds_dir / f"predictions_{model_name}.csv",
            x_labels=x_labels,
        )
        # Per-model plot
        is_kr_champion = model_name == "SVR-Linear"  # paper main
        plot_elife_single(
            y_obs, y_pred,
            f"KR {model_name} — Phase 12 Optuna refit predictions (n=68 hold-out)",
            plots_dir / f"elife_{model_name}.png",
            x_labels=x_labels[::8] if len(x_labels) > 16 else x_labels,  # subsample for readability
            metrics=metrics,
            highlight=is_kr_champion,
        )
        log.info(f"  {model_name}: MAE={metrics['MAE']:.3f}, RMSE={metrics['RMSE']:.3f}, "
                  f"MAPE={metrics['MAPE']:.1f}%, SMAPE={metrics['SMAPE']:.1f}%")

    log.info(f"\nProcessed {len(items)} models, failed {len(failed)}: {failed[:10]}")

    # Sort by MAPE (best first) for ranking display
    items_sorted = sorted(items, key=lambda x: x["metrics"]["MAPE"])

    # Master Table 2 CSV
    write_elife_table2_csv(items_sorted, OUT / "elife_phase12_table2_master.csv")

    # Top-20 grid (most readable)
    top20 = items_sorted[:20]
    plot_elife_grid(
        top20,
        f"Phase 12 eLife Format — Top-20 KR Models by MAPE (BayesianRidge / SVR-Linear / ElasticNet etc., n=68 hold-out)",
        OUT / "elife_phase12_top20_grid.png",
        ncol=4, highlight_key="SVR-Linear",
    )
    # Full grid (54 models)
    plot_elife_grid(
        items_sorted,
        f"Phase 12 eLife Format — All 54 KR Models (sorted by MAPE)",
        OUT / "elife_phase12_full_grid.png",
        ncol=6, highlight_key="SVR-Linear",
    )

    # Summary stats
    maes = [x["metrics"]["MAE"] for x in items]
    rmses = [x["metrics"]["RMSE"] for x in items]
    mapes = [x["metrics"]["MAPE"] for x in items if not np.isnan(x["metrics"]["MAPE"])]
    log.info(f"\n=== Summary across {len(items)} models ===")
    log.info(f"MAE:   median={np.median(maes):.3f}, mean={np.mean(maes):.3f}, min={min(maes):.3f}, max={max(maes):.3f}")
    log.info(f"RMSE:  median={np.median(rmses):.3f}, mean={np.mean(rmses):.3f}")
    log.info(f"MAPE%: median={np.median(mapes):.2f}, mean={np.mean(mapes):.2f}")
    log.info(f"\nTop-5 by MAPE:")
    for x in items_sorted[:5]:
        log.info(f"  {x['name']:<30} MAPE={x['metrics']['MAPE']:.2f}% MAE={x['metrics']['MAE']:.3f}")
    log.info(f"\n✓ Output: {OUT}")


if __name__ == "__main__":
    main()
