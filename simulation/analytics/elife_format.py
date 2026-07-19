"""
eLife 107767 Format — Generic Cross-Phase Observed-vs-Predicted Helper.

PURPOSE:
    Single source of truth for eLife Figure I/J style plots + Table 2 metrics.
    Pluggable into any phase that produces (y_obs, y_pred) arrays.

USAGE:
    from simulation.analytics.elife_format import (
        compute_elife_metrics,   # Table 2 metrics (MAE/RMSE/MAPE/SMAPE)
        plot_elife_single,        # Figure J style per-model/country/region
        plot_elife_grid,          # Multi-panel summary
        write_elife_table2_csv,   # Paper Table 2 CSV
        write_predictions_csv,    # Per-unit prediction CSV
    )

WHO USES:
    - scripts/run_per_country_elife_format.py  (28-country overseas (Pov) cross-country)
    - scripts/run_elife_phase12.py             (54-model per_model_optimize (R9) KR)
    - any future phase with (y_obs, y_pred)

REFERENCE:
    Wang et al. (2025) eLife 107767 — LSTM ILI forecasting (Putian/Sanming)
    https://elifesciences.org/reviewed-preprints/107767
"""
from __future__ import annotations

import csv
import logging
import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

log = logging.getLogger(__name__)

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# Standard eLife Table 2 metrics
ELIFE_METRICS = ["MAE", "RMSE", "MAPE", "SMAPE"]


def compute_elife_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute eLife Table 2 metrics (MAE/RMSE/MAPE/SMAPE).

    Args:
        y_true: ground truth array.
        y_pred: prediction array.

    Returns:
        dict with keys "MAE", "RMSE", "MAPE", "SMAPE" (% scaled).

    Performance: O(n).
    Side effects: None.
    Caller responsibility: y_true.shape == y_pred.shape.

    Reference: eLife 107767 Table 2 — 4 metrics for LSTM ILI forecast.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    assert y_true.shape == y_pred.shape, f"shape mismatch {y_true.shape} vs {y_pred.shape}"

    eps = 1e-9
    diff = y_true - y_pred
    abs_diff = np.abs(diff)
    mae = float(np.mean(abs_diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    nonzero = np.abs(y_true) > eps
    mape = float(np.mean(abs_diff[nonzero] / np.abs(y_true[nonzero])) * 100) if nonzero.any() else float("nan")
    smape = float(np.mean(2 * abs_diff / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "SMAPE": smape}


def plot_elife_single(
    y_obs: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    outpath: Path,
    *,
    x_labels: Optional[list[str]] = None,
    obs_label: str = "Observations",
    pred_label: str = "Predictions",
    metrics: Optional[dict] = None,
    figsize: tuple = (10, 5),
    highlight: bool = False,
):
    """Generate single-unit eLife Figure J style plot (red=obs, blue=pred).

    Args:
        y_obs: observation array.
        y_pred: prediction array (same len as y_obs).
        title: figure title.
        outpath: output PNG path.
        x_labels: optional x-axis labels (week labels). Default: 1..N.
        obs_label/pred_label: legend labels.
        metrics: optional dict to display in subtitle.
        figsize: figure size.
        highlight: if True, yellow background + red frame (KR highlight).
    """
    fig, ax = plt.subplots(figsize=figsize)
    n = len(y_obs)
    x = list(range(n))
    ax.plot(x, y_obs, color="#dc2626", linewidth=2.2, marker="o", markersize=7, label=obs_label)
    ax.plot(x, y_pred, color="#2563eb", linewidth=2.2, marker="s", markersize=7, label=pred_label)
    if x_labels and len(x_labels) == n:
        # Full labels: every point
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=9)
    elif x_labels:
        # Subsampled labels — use linearly spaced ticks matching label count
        n_lab = len(x_labels)
        ticks = np.linspace(0, n - 1, n_lab).astype(int).tolist()
        ax.set_xticks(ticks)
        ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=9)
    ax.set_xlabel(f"Hold-out points (n={n})", fontsize=11)
    if metrics:
        sub = f" — MAE={metrics.get('MAE', 0):.3f}, RMSE={metrics.get('RMSE', 0):.3f}, MAPE={metrics.get('MAPE', 0):.1f}%, SMAPE={metrics.get('SMAPE', 0):.1f}%"
        ax.set_title(title + sub, fontsize=11, fontweight="bold")
    else:
        ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)
    if highlight:
        ax.set_facecolor("#fef3c7")
        for sp in ax.spines.values():
            sp.set_linewidth(2.5); sp.set_edgecolor("#dc2626")
    plt.tight_layout()
    plt.savefig(outpath, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_elife_grid(
    items: list[dict],
    suptitle: str,
    outpath: Path,
    *,
    ncol: int = 4,
    highlight_key: Optional[str] = "KR",
    highlight_field: str = "name",
):
    """Generate multi-panel eLife grid (paper Figure 양식).

    Args:
        items: list of dicts {name, y_obs, y_pred, metrics}.
        suptitle: figure-level title.
        outpath: output PNG path.
        ncol: panels per row.
        highlight_key: name value to highlight (e.g. "KR" or "SVR-Linear").
        highlight_field: dict field to match for highlight.
    """
    N = len(items)
    nrow = (N + ncol - 1) // ncol
    fig = plt.figure(figsize=(5 * ncol, 2.6 * nrow))
    gs = gridspec.GridSpec(nrow, ncol, hspace=0.55, wspace=0.22)
    for i, item in enumerate(items):
        ax = fig.add_subplot(gs[i // ncol, i % ncol])
        n = len(item["y_obs"])
        x = list(range(n))
        ax.plot(x, item["y_obs"], color="#dc2626", linewidth=1.4, marker="o", markersize=4, label="Obs")
        ax.plot(x, item["y_pred"], color="#2563eb", linewidth=1.4, marker="s", markersize=4, label="Pred")
        m = item.get("metrics", {})
        title = f"{item['name']} | MAE={m.get('MAE', 0):.2f} RMSE={m.get('RMSE', 0):.2f}"
        ax.set_title(title, fontsize=10, fontweight="bold")
        if highlight_key and item.get(highlight_field) == highlight_key:
            ax.set_facecolor("#fef3c7")
            for sp in ax.spines.values():
                sp.set_linewidth(2.0); sp.set_edgecolor("#dc2626")
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle(suptitle, fontsize=15, fontweight="bold", y=0.998)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=12,
               bbox_to_anchor=(0.5, -0.005), frameon=True)
    plt.savefig(outpath, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close()


def write_elife_table2_csv(
    items: list[dict],
    outpath: Path,
    *,
    extra_cols: Optional[list[str]] = None,
) -> None:
    """Write eLife Table 2 master CSV (paper 양식).

    Args:
        items: list of {name, metrics, ...} dicts.
        outpath: CSV output path.
        extra_cols: additional columns to include from item dict.
    """
    extra_cols = extra_cols or []
    with open(outpath, "w", encoding="utf-8") as f:  # G-057: encoding 명시
        wr = csv.writer(f)
        wr.writerow(["name"] + extra_cols + ELIFE_METRICS)
        for item in items:
            m = item.get("metrics", {})
            wr.writerow(
                [item["name"]]
                + [item.get(c, "") for c in extra_cols]
                + [f"{m.get(k, float('nan')):.4f}" for k in ELIFE_METRICS]
            )
    log.info(f"✓ {outpath.name} — {len(items)} rows")


def write_predictions_csv(
    name: str,
    y_obs: np.ndarray,
    y_pred: np.ndarray,
    outpath: Path,
    *,
    x_labels: Optional[list[str]] = None,
    extra_cols: Optional[dict[str, list]] = None,
) -> None:
    """Write per-unit predictions CSV.

    Args:
        name: unit identifier (model/country/region).
        y_obs/y_pred: arrays.
        outpath: CSV path.
        x_labels: optional time labels.
        extra_cols: dict {col_name: list_of_values}.
    """
    extra_cols = extra_cols or {}
    n = len(y_obs)
    with open(outpath, "w", encoding="utf-8") as f:  # G-057: encoding 명시
        wr = csv.writer(f)
        headers = ["idx", "label"] + list(extra_cols.keys()) + ["y_obs", "y_pred"]
        wr.writerow(headers)
        for i in range(n):
            row = [i, (x_labels[i] if x_labels else str(i + 1))]
            for k, vals in extra_cols.items():
                row.append(vals[i] if i < len(vals) else "")
            row += [f"{y_obs[i]:.6f}", f"{y_pred[i]:.6f}"]
            wr.writerow(row)


# ── Phase-specific Adapters ──

def adapt_phase12_json(json_data: dict) -> Optional[tuple[np.ndarray, str]]:
    """Extract predictions from per_model_optimize (R9) per_model_optimal/*.json structure.

    Args:
        json_data: parsed JSON dict from per_model_optimal/<MODEL>.json.

    Returns:
        (y_pred np.array, model_name) or None if no predictions found.
    """
    preds = json_data.get("refit_test_predictions")
    if preds and isinstance(preds, list) and len(preds) > 0:
        return np.array(preds, dtype=float), json_data.get("model", "unknown")
    return None


def adapt_phase15_csv(csv_path: Path) -> Optional[tuple[np.ndarray, np.ndarray, list[str]]]:
    """Extract (y_obs, y_pred, labels) from overseas (Pov) cross-country predictions_{COUNTRY}.csv.

    Args:
        csv_path: path to predictions CSV.

    Returns:
        (y_obs, y_pred, week_labels) or None if read failure.
    """
    try:
        with open(csv_path, encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            rows = list(rdr)
        y_obs = np.array([float(r["y_obs"]) for r in rows])
        y_pred = np.array([float(r["y_pred"]) for r in rows])
        labels = [r.get("week_label", r.get("label", str(i))) for i, r in enumerate(rows)]
        return y_obs, y_pred, labels
    except Exception as e:
        log.error(f"Failed to read {csv_path}: {e}")
        return None
