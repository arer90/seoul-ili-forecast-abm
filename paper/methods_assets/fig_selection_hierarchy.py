"""Methods figure: per-model selection hierarchy (preproc -> MC/feature -> stability -> HPO -> refit -> sealed test).

Renders the model-selection pipeline as a single linear flow of labelled boxes, one box per
stage, with the key parameters of each stage annotated underneath. A horizontal "leakage
barrier" makes the central methodological guarantee explicit: every selection decision is made on
out-of-fold (OOF) WF-CV data only; the held-out test slab is touched exactly once, at the end, for
reporting (it is never used to pick a transform, feature set, hyper-parameter, or champion).

SSOT (no retraining, no DB reads — this is a static schematic of the documented pipeline):
  simulation/pipeline/per_model_optimize.py  (R9 per_model_optimize)
    - PREPROC-FIRST + STABILITY + GUARD ordering  (~L2375-2483)
    - feature_guard margin  MPH_FEAT_MARGIN=0.02  (L2370)
    - nested size-path 1-SE / parsimony           (L2421-2456)
    - leakage-free champion score = OOF-WIS, not test_wis (L2835, L2889)
  simulation/pipeline/preproc_optuna_hierarchical.py
    - flat-grid 7-transform + 1-SE (G-333/G-335), STABLE_Y_TRANSFORMS (L99)
  ENGINEERING_PRINCIPLES.md  R9 line: "실제순서 preproc->mc->STABILITY feature->HP"
             learning env: MPH_MC_MARGIN=0.02, MPH_BEST_BY=oof_cv
  Champion selection: G-339 LEAK-FREE (per_model_eval.py:select_champion_g318)
             OOF 1-SE band -> fold stability -> parsimony -> OOF-WIS; hold-out test NOT used to select.

Style matches simulation/scripts/regenerate_stale_thesis_figures.py and
paper/ch4_new_assets/*.py (DejaVu Sans, savefig.dpi=160, TEAL/NAVY/amber palette).

Constraints honoured: REAL documented pipeline only (no fabricated numbers); seed=42 (no
stochastic step here — schematic); NO uv sync.

Run:
    .venv/bin/python paper/methods_assets/fig_selection_hierarchy.py
Output:
    paper/methods_assets/fig_selection_hierarchy.png
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "savefig.dpi": 160,
    "figure.dpi": 130,
})

# ── palette (matches thesis figure SSOT) ─────────────────────────────────────
TEAL = "#0f766e"    # selection / OOF stages
NAVY = "#1e3a5f"    # input / refit
AMBER = "#b45309"   # sealed test (touched once)
GREY = "#6b7280"
INK = "#1f2937"     # box text
BAND_OOF = "#d1fae5"   # OOF-only region tint
BAND_TEST = "#fde7c9"  # sealed-test region tint
EDGE = "#0b3d3a"

np.random.seed(42)  # determinism contract (no stochastic op; documented)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "methods_assets" / "fig_selection_hierarchy.png"

# ── stage definitions (SSOT-sourced) ─────────────────────────────────────────
# Each: (short title, fill colour, text colour, kind). Parameter captions, the region banners, the
# title, and the footnote were removed on request: the figure is now the bare sequence of stage
# boxes, and the surrounding prose in §3.5 carries every detail those labels used to spell out.
#   kind: "io" = data box (input/output), "sel" = OOF selection stage, "test" = sealed test
STAGES = [
    ("Raw features", "#eef2ff", NAVY, "io"),
    ("Preprocessing", TEAL, "white", "sel"),
    ("Multicollinearity /\nfeature selection", TEAL, "white", "sel"),
    ("Feature stability\nselection", TEAL, "white", "sel"),
    ("Hyper-parameter\noptimisation", TEAL, "white", "sel"),
    ("Final refit", NAVY, "white", "io"),
    ("Sealed test", AMBER, "white", "test"),
]

# ── geometry ─────────────────────────────────────────────────────────────────
N = len(STAGES)
fig_w, fig_h = 13.6, 5.9   # B5-width friendly aspect
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_xlim(0, N)
ax.set_ylim(0, 10)
ax.axis("off")

box_w = 0.80
box_h = 3.35
cy = 5.35                       # vertical centre of the box row
xs = [i + 0.5 for i in range(N)]  # box centres

# ── leakage barrier: OOF-only region (stages 0..5) vs sealed-test region (stage 6) ──
barrier_x = xs[-1] - 0.5 - 0.06   # just left of the sealed-test box
ax.axvspan(0.04, barrier_x, ymin=0.04, ymax=0.965, color=BAND_OOF, alpha=0.45, zorder=0)
ax.axvspan(barrier_x, N - 0.04, ymin=0.04, ymax=0.965, color=BAND_TEST, alpha=0.55, zorder=0)
ax.plot([barrier_x, barrier_x], [0.55, 9.45], color=AMBER, lw=2.2, ls=(0, (6, 4)), zorder=1)

# ── draw boxes: title only, centred; no parameter captions ───────────────────
for i, (title, fill, txt, kind) in enumerate(STAGES):
    x = xs[i]
    half_w = box_w / 2.0
    bb = FancyBboxPatch(
        (x - half_w, cy - box_h / 2.0), box_w, box_h,
        boxstyle="round,pad=0.012,rounding_size=0.07",
        linewidth=1.6, edgecolor=EDGE if kind != "test" else "#7c3a00",
        facecolor=fill, zorder=3,
    )
    ax.add_patch(bb)
    ax.text(x, cy, title,
            ha="center", va="center", fontsize=11.5, fontweight="bold",
            color=txt, zorder=4, linespacing=1.15)

# ── arrows between boxes ─────────────────────────────────────────────────────
for i in range(N - 1):
    x0 = xs[i] + box_w / 2.0
    x1 = xs[i + 1] - box_w / 2.0
    crosses = (i == N - 2)  # arrow into the sealed-test box crosses the barrier
    arr = FancyArrowPatch(
        (x0 + 0.005, cy), (x1 - 0.005, cy),
        arrowstyle="-|>", mutation_scale=15,
        linewidth=2.0 if crosses else 1.8,
        color=AMBER if crosses else GREY, zorder=2,
    )
    ax.add_patch(arr)

# Title, the "once" arrow annotation, the region banners, and the footnote strip are all removed on
# request — the boxes and the leakage-barrier tint are the whole figure now.

fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")
