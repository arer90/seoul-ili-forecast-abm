"""Methods figure: data split + leakage-control timeline.

Renders the temporal data split as a calendar-axis timeline:
    train -> validation (OOF / WF-CV) -> test (sealed, opened once) -> forward (real, operational)
with a walk-forward rolling-origin schematic below, and an explicit leakage barrier carrying the
central guarantee: model/transform/feature/hyper-parameter/champion selection happens on
out-of-fold WF-CV data only; the sealed test slab is opened exactly once for reporting; the forward
slab is true future-data and is never used for selection.

SSOT (no retraining, no DB reads — static schematic of documented split):
  simulation/pipeline/data.py  (R1 run_data)
    - HWP 4-way split: train | val | test | real(forward)         (~L388-396)
    - conformal holdout reserved from the TAIL of in-sample (S0-1) (~L315-324)
  simulation/pipeline/config.py
    - in_sample_test_ratio = 0.20 (HWP 68/337), in_sample_val_ratio = 0.10
  R1 checkpoint (results/checkpoints/checkpoint_R1.json) — ACTUAL run values:
    n(in-sample)=337  n_train=242  n_val=27  n_test=68
    pool_end/test_start=269  holdout_start=311  holdout_weeks=26
    real(forward)_weeks=16  real_dates 2026-02-22 .. 2026-06-07 (strict 7-day weekly, Sun)
  Reconstructed calendar boundaries (weekly grid back from real_first):
    in-sample : 2019-09-08 .. 2026-02-15
    train     : 2019-09-08 .. 2024-04-21   (242 wk)
    val       : 2024-04-28 .. 2024-10-27   ( 27 wk)
    test      : 2024-11-03 .. 2026-02-15   ( 68 wk, sealed)
    conformal holdout : 2025-08-24 .. 2026-02-15  (last 26 wk of in-sample, PI-only)
    forward   : 2026-02-22 .. 2026-06-07   ( 16 wk, operational rolling-origin 1-step)
  Selection guarantee: G-339 leak-free (per_model_eval.py:select_champion_g318),
    MPH_BEST_BY=oof_cv  (ENGINEERING_PRINCIPLES.md learning env).

Style matches paper/methods_assets/fig_selection_hierarchy.py
(DejaVu Sans, savefig.dpi=160, TEAL/NAVY/AMBER palette, BAND_OOF/BAND_TEST tints).

Constraints honoured: REAL documented split only (no fabricated dates/counts — all from
checkpoint_R1.json + config.py); seed=42 (no stochastic step — schematic); NO uv sync. B5 width.

Run:
    .venv/bin/python paper/methods_assets/fig_leakage_split.py
Output:
    paper/methods_assets/fig_leakage_split.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SCS = "𝒜ℬ𝒞𝒟ℰℱ𝒢ℋℐ𝒥𝒦ℒℳ𝒩𝒪𝒫𝒬ℛ𝒮𝒯𝒰𝒱𝒲𝒳𝒴𝒵"
SCF = "STIX Two Math"
def sc(s):
    return "".join(_SCS[ord(c) - 65] if "A" <= c <= "Z" else c for c in s)
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "savefig.dpi": 160,
    "figure.dpi": 130,
})

# ── palette (matches thesis figure SSOT) ─────────────────────────────────────
TEAL = "#0f766e"     # train (fit)
TEAL_L = "#5eb0a8"   # validation (OOF / WF-CV)
NAVY = "#1e3a5f"     # ink / forward
AMBER = "#b45309"    # sealed test (touched once)
PURPLE = "#6d28d9"   # forward / operational (true future)
GREY = "#6b7280"
INK = "#1f2937"
BAND_OOF = "#d1fae5"    # selection (OOF) region tint
BAND_TEST = "#fde7c9"   # sealed-test region tint
BAND_FWD = "#ede9fe"    # forward region tint
EDGE = "#0b3d3a"

np.random.seed(42)  # determinism contract (no stochastic op; documented)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "methods_assets" / "fig_leakage_split.png"

# ── SSOT split (weeks) — from checkpoint_R1.json ─────────────────────────────
N_TRAIN, N_VAL, N_TEST, N_FWD = 242, 27, 68, 16
N_HOLD = 26                       # conformal holdout (tail of in-sample, PI-only)
N_INSAMPLE = N_TRAIN + N_VAL + N_TEST   # = 337
N_TOTAL = N_INSAMPLE + N_FWD            # = 353

# calendar labels (reconstructed from weekly grid, verified vs real_dates)
DATES = {
    "in_first":   "2019-09-08",
    "train_end":  "2024-04-21",
    "val_start":  "2024-04-28",
    "val_end":    "2024-10-27",
    "test_start": "2024-11-03",
    "test_end":   "2026-02-15",
    "hold_start": "2025-08-24",
    "fwd_start":  "2026-02-22",
    "fwd_end":    "2026-06-07",
}


# ── figure geometry ──────────────────────────────────────────────────────────
# Two stacked panels: (top) the four-way calendar split, (bottom) the walk-forward schematic.
# Modifiers, region banners, and the guarantee box were removed on request; the caption in §3.5
# carries the leakage guarantee. What stays is the split itself, real dates, and how WF-CV rolls.
fig, ax = plt.subplots(figsize=(13.8, 6.6))
ax.set_xlim(-14, N_TOTAL + 2)
ax.set_ylim(0, 100)
ax.axis("off")

x0 = 0
x_train_end = N_TRAIN
x_val_end = N_TRAIN + N_VAL
x_test_end = N_INSAMPLE
x_fwd_end = N_TOTAL

# ── 1. the four-way split bar, each segment its own colour + real date range ──
bar_y, bar_h = 66.0, 15.0
# The validation band is only 27 weeks wide — "VALIDATION" overflows it, so it gets the short
# label "Val." while the wide bands keep their full names.
segments = [
    (x0,          x_train_end, TEAL,   "TRAIN",      f"{N_TRAIN} wk", DATES["in_first"],  DATES["train_end"]),
    (x_train_end, x_val_end,   TEAL_L, "Val.",       f"{N_VAL} wk",   DATES["val_start"], DATES["val_end"]),
    (x_val_end,   x_test_end,  AMBER,  "TEST",       f"{N_TEST} wk",  DATES["test_start"], DATES["test_end"]),
    (x_test_end,  x_fwd_end,   PURPLE, "FORWARD",    f"{N_FWD} wk",   DATES["fwd_start"], DATES["fwd_end"]),
]
for xa, xb, fill, name, nwk, d0, d1 in segments:
    w = xb - xa
    cx = xa + w / 2.0
    ax.add_patch(Rectangle((xa, bar_y), w, bar_h, facecolor=fill,
                           edgecolor="white", linewidth=2.0, zorder=3))
    if w >= 20:                                   # label inside the band
        ax.text(cx, bar_y + bar_h * 0.60, name, ha="center", va="center",
                fontsize=12.5, fontweight="bold", color="white", zorder=4)
        ax.text(cx, bar_y + bar_h * 0.24, nwk, ha="center", va="center",
                fontsize=9.5, color="white", zorder=4)
    else:                                         # FORWARD is narrow: label above, with its dates
        ax.annotate(f"{name}  ({nwk})\n{d0} .. {d1}", xy=(cx, bar_y + bar_h),
                    xytext=(cx, bar_y + bar_h + 7.5), ha="center", va="bottom",
                    fontsize=10.5, fontweight="bold", color=fill, linespacing=1.35,
                    arrowprops=dict(arrowstyle="-", color=fill, lw=1.2), zorder=6)

# Date under each boundary, all on one row. The train-end (2024-04-21) and val-end (2024-11-03)
# ticks are only ~27 weeks apart, so their labels are pushed apart horizontally — right-anchored
# just left of one tick, left-anchored just right of the other — to clear each other on one line.
# FORWARD's dates ride in its callout above the bar.
boundaries = [
    (x0,          DATES["in_first"],   "left",   INK,    0.0),
    (x_train_end, DATES["train_end"],  "right",  TEAL,  -1.0),
    (x_val_end,   DATES["test_start"], "left",   AMBER,  1.0),
    (x_test_end,  DATES["test_end"],   "right",  AMBER,  0.0),
]
for xt, lab, ha, col, dx in boundaries:
    ax.plot([xt, xt], [bar_y - 2.0, bar_y], color=col, lw=1.5, zorder=3)
    ax.text(xt + dx, bar_y - 4.2, lab, ha=ha, va="top", fontsize=11.5, color=col,
            fontweight="bold", zorder=4)

# ── 2. walk-forward schematic: each fold trains on all prior weeks, predicts the next one ─
# Three explicit folds make the "expanding window, predict 1 step ahead" idea readable at a glance.
wf_y0 = 42.0
row_h, row_gap = 8.5, 3.4                  # taller rows so the in-bar labels read at print size
pool_end = x_val_end                      # selection uses train+val only (weeks 0..269)
origins = [110, 175, 240]                 # illustrative expanding-window cut points
for k, origin in enumerate(origins):
    yk = wf_y0 - k * (row_h + row_gap)
    ax.add_patch(Rectangle((0, yk), origin, row_h, facecolor=TEAL, alpha=0.85,
                           edgecolor="white", linewidth=1.0, zorder=3))       # train
    ax.add_patch(Rectangle((origin, yk), 7, row_h, facecolor=AMBER,
                           edgecolor="white", linewidth=1.0, zorder=4))       # predict 1 step
    ax.text(origin / 2.0, yk + row_h / 2.0, "train on all prior weeks",
            ha="center", va="center", fontsize=11.0, color="white",
            fontweight="bold", zorder=5)
    ax.text(origin + 11, yk + row_h / 2.0, "predict\nnext week", ha="left", va="center",
            fontsize=10.5, color=AMBER, fontweight="bold", linespacing=1.05, zorder=5)
    ax.text(-4, yk + row_h / 2.0, f"fold {k+1}", ha="right", va="center",
            fontsize=11.5, color=GREY, fontweight="bold", zorder=4)
# "..." to say the roll continues to the end of the selectable pool
last_y = wf_y0 - (len(origins) - 1) * (row_h + row_gap)
ax.text(pool_end / 2.0, last_y - row_gap - 3.0, ". . .  window rolls forward, one week at a time",
        ha="center", va="center", fontsize=11.0, style="italic", color=TEAL)

# ── 3. two boundaries that matter, drawn as thin lines at the bar only ────────
for bx, col in [(x_val_end, AMBER), (x_test_end, PURPLE)]:
    ax.plot([bx, bx], [bar_y - 1.0, bar_y + bar_h + 1.0], color=col, lw=1.6,
            ls=(0, (4, 3)), zorder=5)
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")
