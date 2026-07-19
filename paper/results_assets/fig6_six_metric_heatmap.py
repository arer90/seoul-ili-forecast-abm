#!/usr/bin/env python3
"""Figure 6 — unified 6-metric model x metric heatmap (matplotlib, PNG + this source).

Replaces the earlier 9-metric heatmap (which mixed mape/smape/crps/pi95_coverage/
direction_acc) with EXACTLY the 6 headline metrics requested for the thesis:

    columns (left -> right):  WIS  |  R^2  |  RMSE  |  MAE  |  AUC-ROC  |  C-index
    rows:                     models, sorted by the canonical rel-WIS / WIS rank
                              (SSOT column `rank_wis`: test-WIS where available,
                               OOF-WIS fallback for point/baseline models),
                              best (FusedEpi champion) on top.

Direction unification (so green = better everywhere):
    higher-is-better : R^2, AUC-ROC, C-index   -> z-score(value)
    lower-is-better  : WIS, RMSE, MAE          -> z-score(-value)

Honesty rule (NO fabrication):
    AUC-ROC and WIS are genuinely NaN for the ~22 point / baseline models that emit
    no probabilistic forecast and no alert-classification. Those cells are drawn as
    GREY with diagonal HATCHING and excluded from each column's z-score normalization.
    (C-index IS available for all 48 models, so its column is fully colored.)

SSOT (read-only, no retraining):
    simulation/results/per_model_eval/per_model_metrics.csv  (48 model rows)

Output (B5-width figure):
    paper/results_assets/fig6_six_metric_heatmap.png
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "savefig.dpi": 200,
    "figure.dpi": 140,
})

ROOT = Path(__file__).resolve().parents[2]
METRICS_CSV = (
    ROOT / "simulation" / "results" / "per_model_eval" / "per_model_metrics.csv"
)
OUT_PNG = Path(__file__).resolve().with_suffix(".png")

TEAL = "#0f766e"   # FusedEpi champion
AMBER = "#b45309"  # NegBinGLM co-champion

# (csv_column, display_label, higher_is_better)
METRIC_COLS = [
    ("wis", "WIS", False),
    ("r2", "R$^2$", True),
    ("rmse", "RMSE", False),
    ("mae", "MAE", False),
    ("roc_auc", "AUC-ROC", True),
    ("c_index", "C-index", True),
]


def _rows() -> list[dict]:
    with METRICS_CSV.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(v) -> float:
    """Parse a cell to float; '', None, 'nan', 'inf' -> NaN (treated as missing)."""
    if v in (None, "", "nan", "NaN"):
        return float("nan")
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x if np.isfinite(x) else float("nan")


def build_heatmap() -> str:
    rows = _rows()
    # Canonical rel-WIS / WIS ordering = SSOT `rank_wis` (ascending = best first).
    rows = sorted(rows, key=lambda r: _f(r.get("rank_wis")))
    names = [r.get("model", "?") for r in rows]

    # Raw value matrix (rows x 6), NaN preserved.
    raw = np.full((len(rows), len(METRIC_COLS)), np.nan, dtype=float)
    for i, r in enumerate(rows):
        for j, (col, _, _) in enumerate(METRIC_COLS):
            raw[i, j] = _f(r.get(col))

    # Per-column z-score on the direction-unified "effective goodness", NaN-safe.
    Z = np.full_like(raw, np.nan)
    for j, (_, _, higher_better) in enumerate(METRIC_COLS):
        col = raw[:, j]
        eff = col if higher_better else -col  # so larger eff = better
        mask = np.isfinite(eff)
        if mask.sum() >= 2:
            mu = np.nanmean(eff[mask])
            sd = np.nanstd(eff[mask])
            if sd > 0:
                Z[mask, j] = (eff[mask] - mu) / sd
            else:
                Z[mask, j] = 0.0
        elif mask.sum() == 1:
            Z[mask, j] = 0.0

    # Masked array: missing cells rendered via cmap.set_bad -> grey.
    Zm = np.ma.masked_invalid(Z)

    n = len(names)
    fig, ax = plt.subplots(figsize=(5.1, max(5.5, n * 0.27 + 1.4)))  # ~B5 width

    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="#cfcfcf")  # grey for NaN / not-applicable cells
    im = ax.imshow(Zm, cmap=cmap, aspect="auto", vmin=-2.0, vmax=2.0)

    # Diagonal hatching overlay on every missing cell (honest "not available").
    for i in range(n):
        for j in range(len(METRIC_COLS)):
            if not np.isfinite(Z[i, j]):
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, hatch="///", edgecolor="#7a7a7a",
                        linewidth=0.0,
                    )
                )

    # Axes / ticks.
    ax.set_xticks(range(len(METRIC_COLS)))
    ax.set_xticklabels(
        [lbl for _, lbl, _ in METRIC_COLS], rotation=20, fontsize=8.5, ha="right"
    )
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=6.8)
    for tick, nm in zip(ax.get_yticklabels(), names):
        if nm == "FusedEpi":
            tick.set_color(TEAL)
            tick.set_fontweight("bold")
        elif nm == "NegBinGLM":
            tick.set_color(AMBER)
            tick.set_fontweight("bold")
    ax.tick_params(length=0)

    # Light gridlines between cells.
    ax.set_xticks(np.arange(-0.5, len(METRIC_COLS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)

    ax.set_title(
        "Six-metric model performance heatmap\n"
        "(per-column z-score, direction-unified so green = better)",
        fontsize=10.5, pad=8,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("z-score (green = better, red = worse)", fontsize=9)

    # Legend patch explaining the grey/hatched not-applicable cells.
    na_patch = plt.Rectangle(
        (0, 0), 1, 1, facecolor="#cfcfcf", hatch="///",
        edgecolor="#7a7a7a", label="N/A (point/baseline: no PI or alert class)",
    )
    ax.legend(
        handles=[na_patch], loc="upper center",
        bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=7.8,
    )

    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight")
    plt.close(fig)
    return str(OUT_PNG)


if __name__ == "__main__":
    out = build_heatmap()
    print(f"wrote {out}")
