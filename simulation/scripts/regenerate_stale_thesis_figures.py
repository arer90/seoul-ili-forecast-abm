#!/usr/bin/env python3
"""Regenerate the 5 STALE thesis figures that still named the RETIRED champion
NegBinGLM-V7 (banned). Current champion (SSOT) = FusedEpi.

SSOT sources (read-only, no retraining):
  - simulation/results/per_model_eval/per_model_metrics.csv  (48 rows, FusedEpi champion)
  - simulation/results/wis_ssot.csv                           (WIS reconciliation)
  - simulation/results/csv/predictions_FusedEpi.csv           (68 test obs vs pred)
  - simulation/results/abm_forward_validation/result.json     (2026 forward window)

Outputs (PNG only — DOES NOT touch the docx):
  simulation/results/figures/fig1_system_overview.png
  simulation/results/figures/fig3_obs_vs_pred_champion.png
  simulation/results/figures/fig4_forest_plot_wis.png
  simulation/results/figures/fig5_heatmap_model_x_metric.png
  simulation/results/figures/fig7_abm_forward_2026.png

English labels, DejaVu Sans, real values only.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "savefig.dpi": 160,
    "figure.dpi": 130,
})

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "simulation" / "results"
OUT = RES / "figures"
OUT.mkdir(parents=True, exist_ok=True)

METRICS_CSV = RES / "per_model_eval" / "per_model_metrics.csv"
FUSED_PRED_CSV = RES / "csv" / "predictions_FusedEpi.csv"
ABM_JSON = RES / "abm_forward_validation" / "result.json"

TEAL = "#0f766e"
NAVY = "#1e3a5f"


def _rows():
    with METRICS_CSV.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(v, default=float("nan")):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------- Figure 1
def fig1_system_overview():
    """System overview diagram. Corrected text: 48-model registry, FusedEpi WIS
    champion, 124-metric battery (was 53 / NegBinGLM-V7; matches the body's
    124-metric / 138-column accounting — the earlier 124->129 edit is reverted)."""
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    boxes = [
        ("KDCA sentinel\nILI surveillance\n(Seoul, 2019-2025)", 8, "#e0f2f1"),
        ("8-stage\npreprocessing\n& leak control", 25, "#e0f2f1"),
        ("48-model registry\n-> FusedEpi\nWIS champion", 43, "#b2dfdb"),
        ("25-district SEIR-V-D\nmetapopulation\n+ behavioral ABM", 62, "#e0f2f1"),
        ("ARIA audited\nadvisory layer\n(human-in-loop)", 81, "#e0f2f1"),
    ]
    w, h, y0 = 15, 14, 14
    centers = []
    for label, cx, color in boxes:
        box = FancyBboxPatch(
            (cx - w / 2, y0), w, h,
            boxstyle="round,pad=0.4,rounding_size=1.2",
            linewidth=1.6, edgecolor=NAVY, facecolor=color,
        )
        ax.add_patch(box)
        ax.text(cx, y0 + h / 2, label, ha="center", va="center",
                fontsize=10, color="#0b2545", weight="bold")
        centers.append(cx)

    for a, b in zip(centers[:-1], centers[1:]):
        ax.add_patch(FancyArrowPatch(
            (a + w / 2, y0 + h / 2), (b - w / 2, y0 + h / 2),
            arrowstyle="-|>", mutation_scale=18, linewidth=1.8, color=TEAL))

    # Footer band of evaluation facts (corrected numbers)
    ax.text(50, 6.5,
            "Leakage-controlled evaluation: 124-metric battery | "
            "champion FusedEpi (WIS 3.28, R2 0.936) | one champion; NegBinGLM = interpretable",
            ha="center", va="center", fontsize=10.5, style="italic", color=NAVY)
    ax.text(50, 36,
            "Figure 1. System overview: surveillance data to audited "
            "district-resolution pipeline.",
            ha="center", va="center", fontsize=11.5, weight="bold", color="#0b2545")

    fig.tight_layout()
    p = OUT / "fig1_system_overview.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return str(p)


# ---------------------------------------------------------------- Figure 3
def fig3_obs_vs_pred():
    """Champion (FusedEpi) one-step-ahead forecast vs observed on the held-out
    test span (n=68). Real obs/pred from predictions_FusedEpi.csv."""
    rows = [r for r in csv.DictReader(FUSED_PRED_CSV.open(encoding="utf-8"))
            if r["split"] == "test"]
    yt = np.array([_f(r["y_true"]) for r in rows])
    yp = np.array([_f(r["y_pred"]) for r in rows])
    x = np.arange(1, len(yt) + 1)

    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    mae = np.mean(np.abs(yt - yp))
    rmse = np.sqrt(np.mean((yt - yp) ** 2))

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.plot(x, yt, "-o", color=NAVY, ms=3.5, lw=1.6, label="Observed ILI")
    ax.plot(x, yp, "-s", color=TEAL, ms=3.0, lw=1.6, alpha=0.9,
            label="FusedEpi 1-step forecast")
    ax.set_xlabel("Held-out test week (n = 68)")
    ax.set_ylabel("ILI rate (per 1,000 outpatient visits)")
    ax.set_title("Observed vs predicted Seoul ILI — champion FusedEpi "
                 "1-step-ahead forecast")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", frameon=True)
    ax.text(0.02, 0.95,
            f"R2 = {r2:.3f}   MAE = {mae:.2f}   RMSE = {rmse:.2f}   WIS = 3.28",
            transform=ax.transAxes, va="top", ha="left", fontsize=10.5,
            bbox=dict(boxstyle="round", fc="#f0fdfa", ec=TEAL))
    fig.tight_layout()
    p = OUT / "fig3_obs_vs_pred_champion.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return str(p)


# ---------------------------------------------------------------- Figure 4
def fig4_forest_plot():
    """WIS forest plot, top-20 by WIS. Values and order come from the metrics CSV.

    Only the champion is named in the axis labels. The figure used to hard-code a second model as
    a joint title-holder, which stopped being true when the thesis settled on a single champion —
    and the label outlived the prose edit that dropped the idea everywhere else, because a grep of
    the manuscript text cannot see inside a PNG.
    """
    rows = [r for r in _rows() if r.get("wis") not in (None, "", "nan")]
    rows.sort(key=lambda r: _f(r["wis"]))
    top = rows[:20]

    wis = [_f(r["wis"]) for r in top]
    lo = [_f(r.get("wis_ci95_lo"), w) for r, w in zip(top, wis)]
    hi = [_f(r.get("wis_ci95_hi"), w) for r, w in zip(top, wis)]
    names = [r["model"] for r in top]
    ys = list(range(len(top)))
    xerr = [[max(w - l, 0) for w, l in zip(wis, lo)],
            [max(h - w, 0) for w, h in zip(wis, hi)]]

    fig, ax = plt.subplots(figsize=(8, max(4.5, len(top) * 0.34)))
    colors = [TEAL if n == "FusedEpi" else "#555" for n in names]
    for yi, w, le, he, c in zip(ys, wis, xerr[0], xerr[1], colors):
        ax.errorbar(w, yi, xerr=[[le], [he]], fmt="o", capsize=3,
                    color=c, ms=6 if c != "#555" else 4.5)
    ax.set_yticks(ys)
    labels = [f"{n} (champion)" if n == "FusedEpi" else n for n in names]
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("WIS (held-out test, n=68) — lower is better, 95% CI bars")
    ax.set_title("Multi-model ranking by weighted interval score (top 20)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    p = OUT / "fig4_forest_plot_wis.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return str(p)


# ---------------------------------------------------------------- Figure 5
def fig5_heatmap():
    """Model x metric heatmap (z-score, green=better). Best models on top —
    FusedEpi row at/near top. Same metric set + sign logic as comprehensive_eval."""
    rows = _rows()
    metric_cols = [
        ("r2", True), ("mae", False), ("rmse", False), ("mape", False),
        ("smape", False), ("wis", False), ("crps_gaussian", False),
        ("pi95_coverage", "calib"), ("direction_acc", True),
    ]
    mat, names = [], []
    for r in rows:
        vec, ok = [], True
        for col, _ in metric_cols:
            v = r.get(col)
            if v in (None, "", "nan"):
                ok = False
                break
            try:
                vec.append(float(v))
            except (TypeError, ValueError):
                ok = False
                break
        if ok:
            mat.append(vec)
            names.append(r.get("model", "?"))
    M = np.array(mat, dtype=float)
    Mn = np.zeros_like(M)
    for j, (_, dirn) in enumerate(metric_cols):
        col = M[:, j]
        if dirn == "calib":
            eff = -np.abs(col - 0.95)
        elif dirn is True:
            eff = col
        else:
            eff = -col
        sd = np.std(eff)
        Mn[:, j] = (eff - eff.mean()) / sd if sd > 0 else 0.0
    order = np.argsort(-np.nanmean(Mn, axis=1))
    Mn = Mn[order]
    names = [names[i] for i in order]

    fig, ax = plt.subplots(figsize=(8.5, max(4, len(names) * 0.26 + 1)))
    im = ax.imshow(Mn, cmap="RdYlGn", aspect="auto", vmin=-2, vmax=2)
    ax.set_xticks(range(len(metric_cols)))
    ax.set_xticklabels([c for c, _ in metric_cols], rotation=45, ha="right",
                       fontsize=9)
    ax.set_yticks(range(len(names)))
    ylabels = [f"{n}  *" if n == "FusedEpi" else n for n in names]
    ax.set_yticklabels(ylabels, fontsize=7.5)
    # Bold-highlight the champion tick
    for tick, n in zip(ax.get_yticklabels(), names):
        if n == "FusedEpi":
            tick.set_color(TEAL)
            tick.set_fontweight("bold")
        elif n == "NegBinGLM":
            tick.set_color("#b45309")
            tick.set_fontweight("bold")
    ax.set_title("Model x metric heatmap across the evaluation battery "
                 "(z-score, green = better)")
    plt.colorbar(im, ax=ax, label="z-score (higher = better)")
    fig.tight_layout()
    p = OUT / "fig5_heatmap_model_x_metric.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return str(p)


# ---------------------------------------------------------------- Figure 7
def fig7_abm_forward():
    """ABM forecast-anchored 2026 FORWARD-window validation (NOT retrospective).
    Real arrays from abm_forward_validation/result.json: forward_r2=0.722,
    behavior-on 0.557 vs off 0.041, dates 2026-02-16 .. 2026-06-01 (n=16)."""
    d = json.loads(ABM_JSON.read_text(encoding="utf-8"))
    dates = d["forward_dates"]
    real = np.array(d["real_forward_ili"], dtype=float)
    abm = np.array(d["abm_anchored_forward"], dtype=float)
    n = len(dates)
    x = np.arange(n)
    r2 = d["forward_r2"]
    on = d["forward_r2_behavior_on"]
    off = d["forward_r2_behavior_off"]
    rmse = d["forward_rmse"]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.plot(x, real, "-o", color=NAVY, ms=4, lw=1.8,
            label="Observed ILI (real 2026 forward)")
    ax.plot(x, abm, "-s", color=TEAL, ms=3.5, lw=1.8, alpha=0.9,
            label="Behavioral ABM (forecast-anchored)")
    ax.set_xticks(x[::2])
    ax.set_xticklabels([dt[5:] for dt in dates[::2]], rotation=45, ha="right",
                       fontsize=9)
    ax.set_xlabel("Forward window (2026-02-16 .. 2026-06-01, n=16)")
    ax.set_ylabel("ILI rate (per 1,000 outpatient visits)")
    ax.set_title("Behavioral ABM — forecast-anchored 2026 forward-window "
                 "prediction (not retrospective)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", frameon=True)
    ax.text(0.02, 0.95,
            f"Forward R2 = {r2:.3f}   RMSE = {rmse:.2f}\n"
            f"Behavior ON R2 = {on:.3f}  vs  OFF R2 = {off:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=10.5,
            bbox=dict(boxstyle="round", fc="#f0fdfa", ec=TEAL))
    fig.tight_layout()
    p = OUT / "fig7_abm_forward_2026.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return str(p)


if __name__ == "__main__":
    outputs = {
        "Figure 1": fig1_system_overview(),
        "Figure 3": fig3_obs_vs_pred(),
        "Figure 4": fig4_forest_plot(),
        "Figure 5": fig5_heatmap(),
        "Figure 7": fig7_abm_forward(),
    }
    for k, v in outputs.items():
        print(f"{k}: {v}")
