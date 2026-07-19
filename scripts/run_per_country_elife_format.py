"""
28-country per-country observed-vs-predicted (eLife 107767 Figure I/J format).

For EACH country in TRUE ILI cohort I-B:
  1. Load ILI series (raw scale, NOT z-score, for plot)
  2. Fit baseline BayesianRidge (lag-1 + seasonal)
  3. Last 8 weeks hold-out
  4. Generate eLife Figure J style plot (red=observation, blue=prediction)
  5. Compute Table 2 metrics (MAE, RMSE, MAPE, SMAPE)

OUTPUTS:
  simulation/results/phase15_cross_country/elife_format/
  ├── predictions_{COUNTRY}.csv          # week | y_obs | y_pred
  ├── plots/elife_{COUNTRY}.png           # red vs blue time-series
  ├── elife_summary_grid.png              # 28 panel grid (paper Figure 양식)
  └── elife_table2_master.csv             # country × {MAE, RMSE, MAPE, SMAPE}
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.dates import DateFormatter, MonthLocator
from datetime import datetime, timedelta

from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

from simulation.pipeline.phase15_true_ili_cohort import (
    get_cohort_ib, load_country_ili,
)
from simulation.analytics.eda_equal_footing import cluster_of
from simulation.database import safe_connect  # G-116 SSOT

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

REPO = Path(__file__).parent.parent
DB = REPO / "simulation" / "data" / "db" / "epi_real_seoul.db"
OUT = REPO / "simulation" / "results" / "phase15_cross_country" / "elife_format"
OUT_PLOTS = OUT / "plots"
OUT_CSV = OUT / "predictions"


def metrics_4(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """eLife Table 2 metrics."""
    eps = 1e-9
    diff = y_true - y_pred
    abs_diff = np.abs(diff)
    mae = float(np.mean(abs_diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    nonzero = np.abs(y_true) > eps
    mape = float(np.mean(abs_diff[nonzero] / np.abs(y_true[nonzero])) * 100) if nonzero.any() else float("nan")
    smape = float(np.mean(2 * abs_diff / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "SMAPE": smape}


def build_features(values: list[float], weeks: list[int]):
    """Lag-1 + seasonal sin/cos + week features."""
    X, y, idx = [], [], []
    for i in range(1, len(values)):
        prev = values[i-1]
        w = weeks[i]
        X.append([prev, w, math.sin(2*math.pi*w/52), math.cos(2*math.pi*w/52)])
        y.append(values[i])
        idx.append(i)
    return np.array(X), np.array(y), idx


def run_one_country(conn, country: str, year_min: int, year_max: int, n_hold: int = 8):
    rows, src = load_country_ili(conn, country, year_min, year_max)
    if len(rows) < 60:
        return None
    # Aggregate (year, week) -> value (KR sentinel has season_start+week_seq)
    df = sorted(rows, key=lambda r: (r[0], r[1]))
    values = [r[2] for r in df]
    weeks = [r[1] for r in df]
    years = [r[0] for r in df]
    labels = [f"{y}W{w:02d}" for y, w in zip(years, weeks)]

    X, y, idx = build_features(values, weeks)
    if len(y) < 30:
        return None
    n_te = min(n_hold, max(1, len(y) // 8))
    n_tr = len(y) - n_te

    scaler = StandardScaler().fit(X[:n_tr])
    model = BayesianRidge()
    model.fit(scaler.transform(X[:n_tr]), y[:n_tr])
    y_pred = model.predict(scaler.transform(X[n_tr:]))
    y_obs = y[n_tr:]
    test_labels = [labels[i] for i in idx[n_tr:]]
    test_years = [years[i] for i in idx[n_tr:]]
    test_weeks = [weeks[i] for i in idx[n_tr:]]

    mets = metrics_4(y_obs, y_pred)
    return {
        "country": country, "source": src, "n_train": n_tr, "n_test": n_te,
        "y_obs": y_obs, "y_pred": y_pred,
        "labels": test_labels, "years": test_years, "weeks": test_weeks,
        "metrics": mets,
    }


def plot_one(result: dict, outpath: Path):
    """eLife Figure I/J format: red=obs, blue=pred."""
    fig, ax = plt.subplots(figsize=(10, 5))
    obs = result["y_obs"]; pred = result["y_pred"]
    x = list(range(len(obs)))
    ax.plot(x, obs, color="#dc2626", linewidth=2.2, marker="o", markersize=7, label="Observations")
    ax.plot(x, pred, color="#2563eb", linewidth=2.2, marker="s", markersize=7, label="Predictions")
    ax.set_xticks(x); ax.set_xticklabels(result["labels"], rotation=30, ha="right", fontsize=9)
    ax.set_xlabel(f"Hold-out weeks (n={len(obs)})", fontsize=11)
    ax.set_ylabel(f"{result['country']} ILI ({result['source']})", fontsize=11)
    ax.set_title(f"{result['country']} — TRUE ILI Observed vs Predicted (BayesianRidge)\n"
                  f"MAE={result['metrics']['MAE']:.3f}, RMSE={result['metrics']['RMSE']:.3f}, "
                  f"MAPE={result['metrics']['MAPE']:.1f}%, SMAPE={result['metrics']['SMAPE']:.1f}%",
                  fontsize=11, fontweight="bold")
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_grid(results: list[dict], outpath: Path, ncol: int = 4):
    """28-panel grid (eLife paper Figure 양식)."""
    N = len(results)
    nrow = (N + ncol - 1) // ncol
    fig = plt.figure(figsize=(5 * ncol, 2.6 * nrow))
    gs = gridspec.GridSpec(nrow, ncol, hspace=0.55, wspace=0.22)
    for i, r in enumerate(results):
        ax = fig.add_subplot(gs[i // ncol, i % ncol])
        x = list(range(len(r["y_obs"])))
        ax.plot(x, r["y_obs"], color="#dc2626", linewidth=1.5, marker="o", markersize=4, label="Obs")
        ax.plot(x, r["y_pred"], color="#2563eb", linewidth=1.5, marker="s", markersize=4, label="Pred")
        _, color = cluster_of(r["country"])
        title = f"{r['country']} | MAE={r['metrics']['MAE']:.2f} RMSE={r['metrics']['RMSE']:.2f}"
        ax.set_title(title, fontsize=10, fontweight="bold", color=color)
        if r["country"] == "KR":
            ax.set_facecolor("#fef3c7")
            for sp in ax.spines.values():
                sp.set_linewidth(2.0); sp.set_edgecolor("#dc2626")
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"eLife Figure J Format — 28-Country Observed vs Predicted "
                  f"(BayesianRidge, 2021-2025, last 8-week hold-out)",
                  fontsize=15, fontweight="bold", y=0.998)
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=12,
               bbox_to_anchor=(0.5, -0.005), frameon=True)
    plt.savefig(outpath, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year-min", type=int, default=2021)
    ap.add_argument("--year-max", type=int, default=2025)
    ap.add_argument("--n-hold", type=int, default=8)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    OUT_PLOTS.mkdir(parents=True, exist_ok=True)
    OUT_CSV.mkdir(parents=True, exist_ok=True)

    conn = safe_connect(str(DB))
    countries = get_cohort_ib(conn)
    log.info(f"Cohort I-B: {len(countries)} countries — {countries}")

    results = []
    failed = []
    for c in countries:
        r = run_one_country(conn, c, args.year_min, args.year_max, args.n_hold)
        if r is None:
            failed.append(c)
            log.warning(f"  {c}: SKIP (insufficient data)")
            continue
        results.append(r)
        # Save predictions CSV
        with open(OUT_CSV / f"predictions_{c}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["year", "week_no", "week_label", "y_obs", "y_pred"])
            for i in range(len(r["y_obs"])):
                w.writerow([r["years"][i], r["weeks"][i], r["labels"][i],
                            f"{r['y_obs'][i]:.6f}", f"{r['y_pred'][i]:.6f}"])
        # Plot per country
        plot_one(r, OUT_PLOTS / f"elife_{c}.png")
        m = r["metrics"]
        log.info(f"  {c} ({r['source']:<15}): MAE={m['MAE']:.3f}, RMSE={m['RMSE']:.3f}, "
                  f"MAPE={m['MAPE']:.1f}%, SMAPE={m['SMAPE']:.1f}%")
    conn.close()

    # Master Table 2 CSV (paper Table 2 양식)
    with open(OUT / "elife_table2_master.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["country", "cluster", "source", "n_train", "n_test", "MAE", "RMSE", "MAPE", "SMAPE"])
        for r in sorted(results, key=lambda x: x["country"]):
            cl, _ = cluster_of(r["country"])
            m = r["metrics"]
            w.writerow([r["country"], cl, r["source"], r["n_train"], r["n_test"],
                         f"{m['MAE']:.4f}", f"{m['RMSE']:.4f}", f"{m['MAPE']:.4f}", f"{m['SMAPE']:.4f}"])
    log.info(f"✓ elife_table2_master.csv ({len(results)} countries)")

    # Summary grid figure
    plot_grid(sorted(results, key=lambda x: x["country"]), OUT / "elife_summary_grid.png")
    log.info(f"✓ elife_summary_grid.png (28 panel)")

    # Stats summary
    if results:
        all_mae = [r["metrics"]["MAE"] for r in results]
        all_rmse = [r["metrics"]["RMSE"] for r in results]
        all_mape = [r["metrics"]["MAPE"] for r in results if not math.isnan(r["metrics"]["MAPE"])]
        log.info(f"\n=== Summary stats across {len(results)} countries ===")
        log.info(f"MAE:   median={np.median(all_mae):.3f}, mean={np.mean(all_mae):.3f}, min={min(all_mae):.3f}, max={max(all_mae):.3f}")
        log.info(f"RMSE:  median={np.median(all_rmse):.3f}, mean={np.mean(all_rmse):.3f}")
        log.info(f"MAPE%: median={np.median(all_mape):.2f}, mean={np.mean(all_mape):.2f}")
        log.info(f"Failed: {failed}")

    log.info(f"\n✓ All output: {OUT}")


if __name__ == "__main__":
    main()
