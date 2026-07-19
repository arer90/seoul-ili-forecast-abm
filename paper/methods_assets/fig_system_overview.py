"""Figure 1 — System overview: the seven stages from surveillance to advisory.

Each stage is a filled card whose hue marks which layer it belongs to — data (navy/teal), the
forecasting benchmark (teal/green), the mechanistic simulator (purple/amber), the advisory
surface (navy) — so the three-layer architecture of the thesis is legible before a word is read.
The palette is the restrained methods-section set shared with Figures 2, 6 and 7.

The figure is drawn to the aspect ratio of the frame it occupies in the manuscript
(``wp:extent`` 4846320 x 1236398 EMU = 3.920). Word stretches a picture to fill that frame
regardless of the file's pixel size, so a figure authored at any other ratio arrives subtly
distorted; matching it here is what keeps the type upright.

Run:    .venv/bin/python paper/methods_assets/fig_system_overview.py
Output: paper/methods_assets/fig_system_overview.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 220})

OUT = Path(__file__).resolve().parent / "fig_system_overview.png"

# Restrained methods-section palette, shared with Figures 2, 6 and 7.
TEAL = "#0f766e"
NAVY = "#1e3a5f"
AMBER = "#b45309"
GREEN = "#15803d"
PURPLE = "#6d28d9"

# (title, subtitle, fill) — one card per stage, left to right.
# Fill groups the seven stages by layer: data (navy/teal), forecast (teal/green),
# mechanistic (purple/amber), advisory (navy). All fills are dark enough for white type.
STAGES = [
    ("Surveillance\nData", "Weekly Seoul\nILI, 2019–25", NAVY),
    ("Preprocess\n(features)", "Calendar,\nmeteorology", TEAL),
    ("Model\nRegistry", "48 models,\nleak-free", TEAL),
    ("WIS\nChampion", "FusedEpi\nforecaster", GREEN),
    ("SEIR-V-D\nMetapop.", "25 districts", PURPLE),
    ("Behavioral\nABM", "Agent\ncontacts", AMBER),
    ("ARIA\nAdvisory", "Audited\ninterface", NAVY),
]

CAPTION = ("A leakage-controlled forecasting benchmark anchors a district-resolution SEIR-V-D "
           "model with a\nbehavioral ABM contact layer, exposed through the audited ARIA "
           "advisory layer.")

ARROW = "#3f4a56"
INK = "#2b3138"

# The manuscript frame's aspect. Height follows from it, so the figure fills the frame with no
# letterbox — the frame is 4930140 x 1737360 EMU in the docx, and matching it keeps the page map.
_FRAME_ASPECT = 4930140 / 1737360   # = 2.838
W_IN = 13.6
fig, ax = plt.subplots(figsize=(W_IN, W_IN / _FRAME_ASPECT))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# Cards only — the descriptive caption below the row is dropped so the boxes can fill almost the
# whole frame, which makes the text inside them read at the scale the figure is printed. The thesis
# caption already carries the sentence that used to sit under the row.
n = len(STAGES)
BW = 11.6                     # card width in axes units
GAP = (100 - n * BW) / (n + 1)
BH = 82.0                     # tall cards: fill the frame vertically now the caption is gone
TOP = 92.0
y = TOP - BH

for i, (title, sub, fill) in enumerate(STAGES):
    x = GAP + i * (BW + GAP)
    ax.add_patch(FancyBboxPatch(
        (x, y), BW, BH, boxstyle="round,pad=0.4,rounding_size=1.8",
        linewidth=0, facecolor=fill, zorder=3))
    ax.text(x + BW / 2, y + BH * 0.62, title, ha="center", va="center",
            fontsize=15.0, fontweight="bold", color="white", zorder=4, linespacing=1.2)
    ax.text(x + BW / 2, y + BH * 0.34, sub, ha="center", va="center",
            fontsize=12.0, color="white", zorder=4, linespacing=1.25)

    if i < n - 1:                                   # solid triangle, centred in the gap
        cx, cy = x + BW + GAP / 2, y + BH / 2
        hw = GAP * 0.34                             # half-width: stays clear of both cards
        ax.add_patch(Polygon(
            [(cx - hw, cy + 3.2), (cx + hw, cy), (cx - hw, cy - 3.2)],
            closed=True, facecolor=ARROW, edgecolor="none", zorder=4))

# No bbox_inches="tight": cropping to content would change the aspect the frame expects.
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
fig.savefig(OUT, facecolor="white")
plt.close(fig)
print(f"wrote {OUT}")
