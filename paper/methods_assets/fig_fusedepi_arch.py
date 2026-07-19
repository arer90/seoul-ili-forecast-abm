"""Figure 5 — FusedEpi model architecture, CVPR/ICCV-style detailed design diagram.

Left-to-right data flow with tensor shapes and the internal structure of each block:
  Feature tensor  ->  TiRex xLSTM foundation encoder (base)  ->  TabPFN in-context residual
  ->  dynamic-alpha fusion  ->  negative-binomial likelihood head  ->  adaptive-conformal PI head
The design is the documented implementation (simulation/models/fused_epi.py); no fabricated blocks.

Run: .venv/bin/python paper/methods_assets/fig_fusedepi_arch.py
Out: paper/methods_assets/fig_fusedepi_arch.png
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

_SCS = "𝒜ℬ𝒞𝒟ℰℱ𝒢ℋℐ𝒥𝒦ℒℳ𝒩𝒪𝒫𝒬ℛ𝒮𝒯𝒰𝒱𝒲𝒳𝒴𝒵"
SCF = "STIX Two Math"
def sc(s):
    return "".join(_SCS[ord(c) - 65] if "A" <= c <= "Z" else c for c in s)

plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 200})

INK = "#1f2937"; GREY = "#6b7280"
BLUE = "#1e3a5f"; TEAL = "#0f766e"; AMBER = "#b45309"; PLUM = "#6d28d9"; GREEN = "#2f8f7d"
C_IN = "#dbe4f0"; C_TIREX = "#d4ede8"; C_TABPFN = "#d9efe9"; C_FUSE = "#fde8cf"
C_HEAD = "#fdecd6"; C_CONF = "#ece3f9"; C_OUT = "#d7efe0"

OUT = Path(__file__).resolve().parent / "fig_fusedepi_arch.png"
fig, ax = plt.subplots(figsize=(14.5, 7.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def block(x, y, w, h, title, lines, fill, edge, *, fs_t=11.5, fs_l=8.4, tcol=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.4",
                                linewidth=1.8, edgecolor=edge, facecolor=fill, zorder=3))
    ax.text(x + w / 2, y + h - 3.0, title, ha="center", va="center",
            fontsize=fs_t, fontweight="bold", color=tcol or edge, zorder=4)
    for j, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 6.2 - j * 3.0, ln, ha="center", va="center",
                fontsize=fs_l, color=INK, zorder=4)


def tensor(x, y, label, color):
    for k, off in enumerate((1.6, 0.8, 0.0)):
        ax.add_patch(Rectangle((x + off, y + off), 6.0, 9, facecolor=color,
                               edgecolor=INK, linewidth=1.0, alpha=0.92 - k * 0.12, zorder=3))
    ax.text(x + 3.8, y - 2.0, label, ha="center", va="top", fontsize=7.6, color=GREY, style="italic")


def arrow(x0, y0, x1, y1, color=GREY, label=None, lw=2.2, rad=0.0, ldy=1.1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=15,
                                 linewidth=lw, color=color, zorder=2,
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + ldy, label, ha="center", va="bottom",
                fontsize=7.6, color=color, style="italic", zorder=5)


ax.text(50, 97, sc("FusedEpi — model architecture (TiRex xLSTM base + TabPFN residual fusion)"),
        ha="center", va="center", fontsize=15.5, color=INK, fontfamily=SCF)
ax.text(50, 92.4, "one fit · one predict · one deployed artifact   (simulation/models/fused_epi.py)",
        ha="center", va="center", fontsize=9.0, color=GREY, style="italic")

# 0 input tensor
tensor(2.0, 56, "X  ∈  ℝ^(T×D)\nD ≈ 13 lag+seasonal+mech.", C_IN)
ax.text(5.0, 47.5, "leak-free\nrolling features", ha="center", va="top", fontsize=7.2,
        color=BLUE, fontweight="bold")

# 1 TiRex encoder
block(15, 62, 22, 26, "TiRex encoder (foundation)",
      ["pretrained xLSTM · 35M params", "sLSTM ▸ mLSTM ▸ sLSTM blocks",
       "rolling 1-step base forecast"], C_TIREX, TEAL)
for yy in (70.4, 66.9):
    ax.add_patch(FancyBboxPatch((18, yy), 16, 2.4, boxstyle="round,pad=0.1,rounding_size=0.4",
                                linewidth=0.9, edgecolor=TEAL, facecolor="#bfe3db", zorder=4))
ax.text(26, 71.6, "xLSTM block × N", ha="center", va="center", fontsize=6.8, color="#0b3d3a", zorder=5)

# 2 TabPFN residual
block(15, 30, 22, 26, "TabPFN residual corrector",
      ["tabular transformer (in-context)", "fits residual  r = y − ŷ_TiRex",
       "small-n prior-data-fit"], C_TABPFN, GREEN)

# 3 mc prune + mechanistic anchor
block(15, 8, 22, 17, "mc prune + mechanistic anchor",
      ["do-no-harm collinearity prune (train-only)", "Rt · S/N · FoI = Rt·(1−S/N) channels"],
      "#eef1f4", GREY, fs_t=9.6, fs_l=7.6, tcol=INK)

# 4 dynamic-alpha fusion
block(42, 46, 22, 24, "dynamic-α fusion",
      ["ŷ = ŷ_TiRex + α · g(x)", "α = h · clip(n/n_ref, α_min, 1)",
       "do-no-harm gate (α→0 if worse)"], C_FUSE, AMBER, fs_t=12)

# 5 NegBin head
block(42, 16, 22, 23, "NegBin likelihood head",
      ["μ = fused ŷ, over-dispersed", "Y ~ NegBin(μ, disp)",
       "cal-split dispersion estimate"], C_HEAD, AMBER)

# 6 adaptive conformal
block(69, 40, 24, 30, "adaptive-conformal PI head",
      ["CQR + Conformal-PID (P+I)", "online widen / narrow · auto-skew",
       "leak-free: past residuals only"], C_CONF, PLUM)

# 7 output
block(69, 8, 24, 23, "output",
      ["point forecast  ŷ_t  ∈ ℝ", "PI 50 / 80 / 95 (calibrated)",
       "WIS-scored, rolling 1-step"], C_OUT, GREEN, fs_t=12)

# data-flow arrows
arrow(9.5, 63, 15, 74, TEAL, "sequence", rad=0.16)
arrow(9.5, 60, 15, 43, GREEN, "features", rad=-0.16)
arrow(26, 25.1, 26, 30, GREY)
arrow(37, 74, 42, 62, TEAL, "ŷ_TiRex", rad=0.10)
arrow(37, 43, 42, 54, GREEN, "g(x)", rad=-0.10)
arrow(53, 45.9, 53, 39, AMBER, "μ")
arrow(64, 27, 69, 42, AMBER, "quantiles", rad=0.14)
arrow(64, 58, 69, 55, PLUM, "point")
arrow(81, 39.9, 81, 31, GREEN, "calibrated")

fig.text(0.5, 0.015,
         "Central conformal intervals: 95% (0.025, 0.975) and 50% (0.25, 0.75) via CQR; "
         "80% interpolated from the same NegBin family. Champion selected by leak-free OOF-WIS (G-339).",
         ha="center", va="bottom", fontsize=7.4, color=GREY)

fig.savefig(OUT, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"wrote {OUT}")
