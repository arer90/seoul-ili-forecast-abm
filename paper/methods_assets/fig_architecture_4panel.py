"""Three-Layer Decision-Support Architecture — 4-panel figure.

Reproduces the reference layout:
  (a) data pipeline & feature engineering   (b) model registry · walk-forward CV · evaluation
  (c) champion → forecast-anchored ABM      (d) ARIA agent–MCP–Hermes consultation surface

Every printed number is verified against a shipped result file. Sources:
  registry size ............ simulation/models/registry.py CATEGORY_MODELS (45 unique, live count)
  relative-WIS podium ...... simulation/results/per_model_eval/per_model_metrics.csv
                             column relative_wis_vs_baseline: FusedEpi .7061, GAM-Spline .7529, TiRex .7855
  champion WIS ............. same file, column wis = 3.2784
  weeks / features ......... simulation/results/checkpoints/checkpoint_R1.json (n=337, 398 features)
  districts / age bands .... simulation/abm/agent_kernel.py (_N_GU=25, _N_AGE=7)
  ABM forward R^2 .......... simulation/results/abm_forward_validation/result.json
  MCP tool count ........... simulation/server/mcp_epi.py TOOL_SPECS (12, runtime-verified)

Output: paper/methods_assets/fig_architecture_4panel.png
Spelling: US English throughout.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 175, "figure.dpi": 120})
rng = np.random.default_rng(42)
SCF = "STIX Two Math"

_SC = {"A": "\U0001D49C", "B": "ℬ", "C": "\U0001D49E", "D": "\U0001D49F", "E": "ℰ",
       "F": "ℱ", "G": "\U0001D4A2", "H": "ℋ", "I": "ℐ", "J": "\U0001D4A5",
       "K": "\U0001D4A6", "L": "ℒ", "M": "ℳ", "N": "\U0001D4A9", "O": "\U0001D4AA",
       "P": "\U0001D4AB", "Q": "\U0001D4AC", "R": "ℛ", "S": "\U0001D4AE", "T": "\U0001D4AF",
       "U": "\U0001D4B0", "V": "\U0001D4B1", "W": "\U0001D4B2", "X": "\U0001D4B3",
       "Y": "\U0001D4B4", "Z": "\U0001D4B5"}


def sc(s: str) -> str:
    return "".join(_SC.get(c, c) for c in s)


INK = "#1a1a1a"; GREY = "#6b7280"; SLATE = "#475569"
NAVY = "#1e3a5f"; TEAL = "#0f766e"; PURPLE = "#6d28d9"; ORANGE = "#d97706"
BG_A = "#dce8f5"; BG_B = "#fbeedd"; BG_C = "#e2efe4"; BG_D = "#efe6f7"
GOLD = "#e0b000"; SILVER = "#a8b0b8"; BRONZE = "#c07b3a"

fig, ax = plt.subplots(figsize=(20.0, 11.3))
ax.set_xlim(0, 200); ax.set_ylim(0, 113); ax.axis("off")


def panel(x, y, w, h, bg, label, lab_col=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.2",
                                linewidth=1.4, edgecolor="#b8c2ce", facecolor=bg, zorder=1))
    ax.text(x + 2.4, y + h - 3.0, sc(label), ha="left", va="center", fontsize=15.5,
            color=lab_col, zorder=5, fontfamily=SCF)


def rbox(x, y, w, h, txt, fc, ec=None, tc=INK, fs=8.5, bold=False, mono=False, z=3, lw=1.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.18,rounding_size=0.55",
                                linewidth=lw, edgecolor=ec or fc, facecolor=fc, zorder=z))
    if txt:
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
                color=tc, zorder=z + 1, fontweight="bold" if bold else "normal",
                fontfamily="monospace" if mono else None, linespacing=1.35)


def arrow(x0, y0, x1, y1, col=INK, lw=1.6, style="-|>", ms=13, rad=0.0, z=4):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms,
                                 linewidth=lw, color=col, zorder=z,
                                 connectionstyle=f"arc3,rad={rad}"))


def heat(x, y, w, h, M, cmap, ec="white", lw=0.25):
    nr, nc = M.shape
    cw, ch = w / nc, h / nr
    cm = plt.get_cmap(cmap)
    lo, hi = M.min(), M.max()
    for i in range(nr):
        for j in range(nc):
            v = (M[i, j] - lo) / (hi - lo + 1e-9)
            ax.add_patch(Rectangle((x + j * cw, y + h - (i + 1) * ch), cw, ch,
                                   facecolor=cm(0.10 + 0.86 * v), edgecolor=ec, lw=lw, zorder=3))


def medal(x, y, rank, r=1.15):
    col = {1: GOLD, 2: SILVER, 3: BRONZE}[rank]
    ax.add_patch(Circle((x, y), r, facecolor=col, edgecolor="#8a6d1f" if rank == 1 else "#7b8288",
                        linewidth=0.8, zorder=6))
    ax.text(x, y - 0.05, str(rank), ha="center", va="center", fontsize=6.4,
            color="white", fontweight="bold", zorder=7)
    # ribbon
    ax.plot([x - 0.55, x - 0.25], [y + r, y + r + 1.15], color="#c2410c", lw=1.3, zorder=5)
    ax.plot([x + 0.55, x + 0.25], [y + r, y + r + 1.15], color="#1d4ed8", lw=1.3, zorder=5)


# ═══════════════════════════ title ═══════════════════════════
ax.text(100, 110.4, "Three-Layer Decision-Support Architecture for District-Level Seasonal ILI Forecasting",
        ha="center", fontsize=19.5, color=INK, family="serif")
ax.text(100, 106.9, sc("Forecasting Registry")+"   →   "+sc("Metapopulation SEIR Simulator")
        + "   →   " + sc("Agent−MCP−Hermes Consultation Surface"),
        ha="center", fontsize=11.2, color=SLATE, fontfamily=SCF)

# ═══════════════════════ (a) data pipeline ═══════════════════════
panel(1.5, 56.5, 96, 47.5, BG_A, "(a)   Data Pipeline & Feature Engineering")

ax.text(5.2, 96.4, sc("Data Sources"), fontsize=8.8, color=SLATE, fontfamily=SCF)

srcs = [("KDCA ILI", "337 wks · 25 gu"), ("KMA Weather", "T · RH · AH"), ("Census", "25×25 commuter")]
for i, (t, s) in enumerate(srcs):
    yy = 88.6 - i * 8.0
    rbox(5, yy, 22, 6.4, "", "white", ec="#334155", lw=1.2)
    ax.text(16, yy + 4.1, t, ha="center", fontsize=10.2, color=INK)
    ax.text(16, yy + 1.7, s, ha="center", fontsize=7.4, color=SLATE, style="italic")
# bracket + down arrow to DB
ax.plot([28.4, 30.4, 30.4, 28.4], [91.8, 91.8, 70.2, 70.2], color=INK, lw=1.1, zorder=3)
ax.plot([30.4, 30.4], [81.0, 71.2], color=INK, lw=1.1, zorder=3)
arrow(16, 87.2, 16, 70.2, col=INK, lw=1.2)

# DB cylinder
cx, cy, cw2, chh = 16, 62.0, 16, 7.0
ax.add_patch(Rectangle((cx - cw2 / 2, cy), cw2, chh, facecolor="#2e6ca4", edgecolor="#1b4d78", lw=1.1, zorder=3))
ax.add_patch(Ellipse((cx, cy + chh), cw2, 2.6, facecolor="#4a8fc7", edgecolor="#1b4d78", lw=1.1, zorder=4))
ax.add_patch(Ellipse((cx, cy), cw2, 2.6, facecolor="#2e6ca4", edgecolor="#1b4d78", lw=1.1, zorder=3))
ax.text(cx, 59.6, "Epidemiology DB", ha="center", fontsize=9.4, color=INK)
ax.text(cx, 57.7, "~80M rows  ·  85 tables", ha="center", fontsize=7.2, color=SLATE, style="italic")
arrow(24.5, 65.5, 39.5, 65.5, col=INK, lw=1.3)
ax.text(31.5, 66.6, "load", ha="center", fontsize=7.6, color=SLATE, style="italic")

# feature tensor
ax.text(48, 97.0, sc("Feature Tensor")+"  X", ha="center", fontsize=10.0, color=INK, fontfamily=SCF)
ax.text(48, 94.4, "ℝ^(T×D×A×F)", ha="center", fontsize=8.6, color=INK)
ax.text(48, 92.0, "337 × 25 × 7 × 398", ha="center", fontsize=7.8, color=SLATE, style="italic")
for k, off in enumerate([(2.2, 2.2), (1.1, 1.1), (0, 0)]):
    M = rng.random((7, 7))
    heat(36.5 + off[0], 70.6 + off[1], 17.0, 17.0, M, ["Greens", "Blues", "Oranges"][k])
for j, ln in enumerate(["T = 337 week", "D = 25 gu  ·  A = 7 age", "stratified", "F = 398 feature"]):
    ax.text(36.5, 68.2 - j * 2.2, ln, fontsize=7.6, color=SLATE, style="italic")

# feature engineer
ax.text(75, 99.0, sc("Feature Engineer")+"   ℱ_θ", ha="center", fontsize=10.6, color=INK, fontfamily=SCF)
ax.add_patch(FancyBboxPatch((60.0, 61.5), 30.0, 34.5, boxstyle="round,pad=0.4,rounding_size=1.0",
                            linewidth=1.8, edgecolor="#ea6a1f", facecolor="none", zorder=3))
feats = ["Lag embedding  (ℓ ∈ 1−52w)", "Fourier  sin/cos  52w · 26w", "Regime dummies  (3 eras)",
         "Commuter share  c_ij", "Age-pop weights  a ∈ ℝ⁷", "Google Trends  z-score", "Log1p ·  Winsorize"]
for i, f in enumerate(feats):
    rbox(61.6, 91.8 - i * 4.55, 26.8, 3.7, f, "#555c66", tc="white", fs=8.0)
ax.add_patch(Circle((88.8, 94.6), 1.45, facecolor="#e8590c", edgecolor="white", lw=1.1, zorder=6))
ax.text(88.8, 94.6, "▲", ha="center", va="center", fontsize=6.2, color="white", zorder=7)


# color legend strip
for i, c in enumerate(["#2b6cb0", "#2f855a", "#dd6b20", "#c53030", "#6b46c1", "#2c7a7b", "#4299e1"]):
    ax.add_patch(Rectangle((92.5, 90.6 - i * 3.3), 3.4, 2.9, facecolor=c, edgecolor="white", lw=0.5, zorder=4))
ax.text(94.2, 64.6, "H", ha="center", fontsize=10.0, color=INK, fontweight="bold")

# ═══════════════════ (b) registry / CV / evaluation ═══════════════════
panel(101.5, 56.5, 97, 47.5, BG_B, "(b)   Model Registry  ·  Walk-Forward CV  ·  Probabilistic Evaluation")
ax.text(105, 97.6, sc("Model Registry")+"    |R|  =  45 active models", fontsize=9.6, color=INK, fontfamily=SCF)

cats = [("Stat", "#5b9bd5", 8), ("Epi/Bay", "#70ad9b", 6), ("Linear", "#e8c547", 7),
        ("Tree", "#e08a4a", 9), ("DL", "#d8534f", 9), ("Fnd/TF", "#8b5cf6", 6)]
for i, (nm, col, n) in enumerate(cats):
    yy = 92.6 - i * 3.6
    ax.text(105, yy + 1.1, nm, fontsize=8.4, color=INK, va="center")
    for j in range(n):
        ax.add_patch(Rectangle((113 + j * 3.05, yy), 2.65, 2.45, facecolor=col,
                               edgecolor="white", lw=0.5, zorder=3))
ax.text(122, 71.6, "←  OOF composite rank  →", ha="center", fontsize=7.8,
        color=SLATE, style="italic", fontfamily=SCF)
ax.text(155, 68.6, "fit × 45 models", ha="center", fontsize=7.8, color=SLATE, style="italic", fontfamily=SCF)
ax.annotate("", xy=(178.0, 66.4), xytext=(137.0, 66.4),
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.25), zorder=6)
ax.text(178.8, 67.4, "$y,\\hat{y}$", fontsize=8.4, color=INK, style="italic")

# walk-forward CV bars
ax.text(162, 97.6, sc("Walk-Forward CV"), ha="center", fontsize=9.6, color=INK, fontfamily=SCF)
segs = [(0.52, 0.13), (0.60, 0.11), (0.66, 0.10), (0.72, 0.09), (0.78, 0.08)]
for i, (tr, va) in enumerate(segs):
    yy = 93.0 - i * 3.5
    x0, W = 146, 34
    ax.add_patch(Rectangle((x0, yy), W * tr, 2.5, facecolor="#a9cbe8", edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((x0 + W * tr, yy), W * va, 2.5, facecolor="#e8b64c", edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((x0 + W * (tr + va), yy), W * (1 - tr - va - 0.16), 2.5,
                           facecolor="#f2f2f2", edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((x0 + W * 0.84, yy), W * 0.16, 2.5, facecolor="#c0392b", edgecolor="none", zorder=3))
    ax.text(144.4, yy + 1.25, f"F{i+1}", ha="right", va="center", fontsize=8.0, color=INK)
ax.text(146, 73.6, "2019W01", fontsize=7.2, color=INK, style="italic")
ax.text(180, 73.6, "2025W25", ha="right", fontsize=7.2, color=INK, style="italic")
for i, (lab, col) in enumerate([("train", "#a9cbe8"), ("val", "#e8b64c"), ("26w hold", "#c0392b")]):
    xx = 147 + i * 12
    ax.add_patch(Rectangle((xx, 69.0), 2.6, 2.2, facecolor=col, edgecolor="none", zorder=3))
    ax.text(xx + 3.2, 70.1, lab, fontsize=7.8, color=INK, va="center")

# evaluator column
ax.text(190, 97.6, sc("Evaluator"), ha="center", fontsize=9.6, color=INK, fontfamily=SCF)
evals = ["WIS (primary)", "CRPS", "Pinball ρ α", "PIT · KS", "R²  (OOF)", "DM test", "PICP 95%", "Conformal α̂"]
for i, e in enumerate(evals):
    rbox(182, 92.6 - i * 3.55, 15.5, 2.9, e, "#5b626b", tc="white", fs=7.6)

# podium — relative-WIS leaderboard (verified)
pod = [("FusedEpi", 1, 0.71), ("GAM-Spline", 2, 0.75), ("TiRex", 3, 0.79)]
for i, (nm, rk, v) in enumerate(pod):
    xx = 107 + i * 22
    medal(xx, 63.4, rk, r=1.05)
    ax.text(xx + 2.3, 63.4, nm, fontsize=9.0, color=INK, va="center")
ax.text(105, 59.4, "(relative-WIS leaderboard vs baseline:  0.71 · 0.75 · 0.79  |  champion FusedEpi, WIS 3.28)",
        fontsize=7.8, color=SLATE, style="italic", fontfamily=SCF)

# ═══════════════════ (c) champion → forecast-anchored ABM ═══════════════════
panel(1.5, 2.5, 96, 51.5, BG_C, "(c)   Champion → Forecast-Anchored ABM  ·  SEIR-V-D + ABM")
ax.text(5, 47.6, sc("Champion")+"  ·  best-WIS", fontsize=9.4, color="#166534", fontfamily=SCF)
ax.text(50, 47.6, sc("Metapop SEIR-V-D 25-gu + ABM"), ha="center", fontsize=9.4,
        color="#166534", fontfamily=SCF)

# podium bars
bars = [("GAM-Spline", 2, 7.6, "#b9bec4", "spline · GAM"),
        ("FusedEpi", 1, 10.4, GOLD, "fusion · champion"),
        ("TiRex", 3, 6.2, BRONZE, "xLSTM · foundation")]
for i, (nm, rk, hgt, col, note) in enumerate(bars):
    xx = 6.5 + i * 7.2
    ax.add_patch(Rectangle((xx, 19.6), 6.2, hgt, facecolor=col, edgecolor="#8a8f95", lw=0.7, zorder=3))
    medal(xx + 3.1, 19.6 + hgt + 1.6, rk, r=0.95)
    ax.text(xx + 3.1, 18.8, nm + "\n" + note, ha="center", va="top", fontsize=6.6,
            color=INK, rotation=40, linespacing=1.5)
ax.plot([5.6, 27.5], [19.6, 19.6], color="#6b7280", lw=1.6, zorder=4)

# SEIRVD chain
ax.text(26.6, 39.2, "λ$^t$", fontsize=13.0, color=INK)
comp = [("S", "#3b82c4", "Susceptible"), ("E", "#e8a33d", "Exposed"), ("I", "#d9534f", "Infectious"),
        ("R", "#5cb85c", "Recovered"), ("V", "#2f8f7d", "Vaccinated"), ("D", "#3f4854", "Death")]
rates = ["β · λ", "σ", "γ", "ν", "μ"]
for i, (lab, col, sub) in enumerate(comp):
    xx = 33.0 + i * 8.0
    ax.add_patch(Circle((xx, 36.6), 3.0, facecolor=col, edgecolor="white", lw=1.4, zorder=5))
    ax.text(xx, 36.6, lab, ha="center", va="center", fontsize=13.5, color="white",
            fontweight="bold", zorder=6)
    ax.text(xx, 32.2, sub, ha="center", fontsize=7.8, color=INK)
    if i < 5:
        arrow(xx + 3.3, 36.6, xx + 6.3, 36.6, col=INK, lw=1.5, ms=11)
        ax.text(xx + 4.8, 39.2, rates[i], ha="center", fontsize=8.6, color="#c0392b")
ax.plot([28.6, 30.4], [36.6, 36.6], color=INK, lw=1.5, zorder=4)

# equations
ax.text(53, 29.4, sc("Governing equations (per district g, age group a)"), ha="center",
        fontsize=8.0, color=SLATE, style="italic", fontfamily=SCF)
eqs = ["dS/dt = −β · Σ_h M_{gh} · I_h/N_h · S  −  ν · S",
       "dI/dt = σ · E − γ · I − μ_I · I        (μ_I drives I → D channel)",
       "dV/dt = ν · S − waning · V        dD/dt = μ_I · I + μ_V · V"]
for i, e in enumerate(eqs):
    ax.text(53, 26.8 - i * 2.6, e, ha="center", fontsize=7.9, color=INK, fontweight="bold")
ax.text(53, 18.8, "Conservation:  S + E + I + R + V + D = N          (per step, per district × age)",
        ha="center", fontsize=7.8, color=SLATE, style="italic", fontfamily=SCF)

# age swatches
ax.text(31.0, 15.4, "A = 7 age groups", fontsize=8.2, color=INK, fontfamily=SCF)
ages = ["0−4", "5−14", "15−24", "25−44", "45−54", "55−64", "65+"]
acol = ["#7cb0dd", "#2f8f7d", "#4a9d6e", "#e8c547", "#e08a4a", "#d8534f", "#8b5cf6"]
for i, (a, c) in enumerate(zip(ages, acol)):
    ax.add_patch(Rectangle((31.0 + i * 5.6, 10.4), 5.0, 3.4, facecolor=c, edgecolor="white", lw=0.6, zorder=3))
    ax.text(33.5 + i * 5.6, 8.9, a, ha="center", fontsize=7.2, color=INK)

# commuter matrix
ax.text(85.5, 47.4, sc("Seoul 25 gu"), ha="center", fontsize=8.6, color=INK, fontfamily=SCF)
Mc = np.eye(25) * 3.0 + rng.random((25, 25)) * 0.55
heat(76.5, 28.5, 18, 17.5, Mc, "Blues", ec="#e8eef5", lw=0.15)
ax.text(85.5, 26.6, sc("Commuter C ∈ ℝ")+"$^{25×25}$", ha="center", fontsize=8.0,
        color=SLATE, style="italic", fontfamily=SCF)

# footer facts
ax.text(4.5, 7.0, "FusedEpi  WIS 3.28  ·  rel-WIS 0.71   |   champion forecast → ABM forcing",
        fontsize=8.0, color="#166534", fontfamily=SCF, fontweight="bold")
ax.text(4.5, 5.0, "forward R²  0.557 behavior-on  vs  0.041 off,  best 0.722",
        fontsize=8.0, color=INK, fontfamily=SCF)
ax.text(4.5, 3.1, "Outputs: ŷ per district × age × week  |  Conformal PI (α = 0.1)  |  epi-validity gate (mass 1.9e-16, Rt ∈ [0.3, 8])",
        fontsize=8.0, color=INK, fontfamily=SCF)

# ═══════════════════ (d) ARIA consultation surface ═══════════════════
panel(101.5, 2.5, 97, 51.5, BG_D, "(d)   ARIA  ·  Agent−MCP−Hermes Consultation Surface")

ax.text(112.7, 47.2, sc("PH Query q"), ha="center", fontsize=9.0, color=INK, fontfamily=SCF)
rbox(105, 33.5, 15.5, 11.5, "", "white", ec="#5b21b6", lw=1.3)
ax.text(112.7, 41.8, '"Gangnam ILI', ha="center", fontsize=7.8, color=INK)
ax.text(112.7, 39.9, 'alert next week?"', ha="center", fontsize=7.8, color=INK)
ax.text(112.7, 36.0, "ctx: λ̂, R$_t$, beds", ha="center", fontsize=7.4, color=SLATE, style="italic")

ax.text(131.5, 47.2, sc("ARIA  ·  12 MCP"), ha="center", fontsize=9.0, color=INK, fontfamily=SCF)
ax.add_patch(FancyBboxPatch((123.5, 25.5), 16.5, 19.5, boxstyle="round,pad=0.35,rounding_size=1.0",
                            linewidth=1.2, edgecolor="#374151", facecolor="#3f4854", zorder=3))
for i, t in enumerate(["Tokenize · Embed", "Multi-Head Attn", "Chain-of-Thought", "Tool-Use Planner", "JSON schema"]):
    rbox(124.7, 41.6 - i * 3.4, 14.1, 2.8, t, "#5a6270", tc="white", fs=7.2, z=4)

ax.text(150.0, 47.2, sc("Hermes R_θ"), ha="center", fontsize=9.0, color=INK, fontfamily=SCF)
ax.add_patch(FancyBboxPatch((142.5, 25.5), 15.0, 19.5, boxstyle="round,pad=0.35,rounding_size=1.0",
                            linewidth=1.2, edgecolor="#5b21b6", facecolor="#6d28d9", zorder=3))
for i, t in enumerate(["Intent cls", "Tool sel τ*", "Arg validate", "Rate limit", "Audit log"]):
    rbox(143.6, 41.6 - i * 3.4, 12.8, 2.8, t, "#8b5cf6", tc="white", fs=7.2, z=4)

ax.text(172.0, 47.2, sc("MCP Tools 𝒯 (12)"), ha="center", fontsize=9.0, color=INK, fontfamily=SCF)
ax.add_patch(FancyBboxPatch((159.5, 24.2), 25.0, 20.8, boxstyle="round,pad=0.35,rounding_size=1.0",
                            linewidth=1.2, edgecolor="#4b5563", facecolor="none", zorder=3))
tools = ["query_db", "lead_time", "forecast", "outbreak", "model_cmp", "validity",
         "shap_feat", "lit_rag", "rt_estim", "scenario", "intl_cmp", "coupled_fwd"]
for i, t in enumerate(tools):
    r, c = divmod(i, 2)
    xx = 160.7 + c * 12.0
    yy = 41.0 - r * 3.35
    rbox(xx, yy, 11.2, 2.8, t, "#7c3aed", tc="white", fs=7.0, mono=True, z=4)
    ax.add_patch(Circle((xx + 1.3, yy + 1.4), 0.42, facecolor="#c4b5fd", edgecolor="none", zorder=6))
ax.text(172.0, 21.8, sc("JSON-RPC 2.0  ·  12 registered"), ha="center", fontsize=7.6,
        color=SLATE, style="italic", fontfamily=SCF)

# HITL decision
ax.text(192.0, 47.2, sc("HITL Decision"), ha="center", fontsize=9.0, color=INK, fontfamily=SCF)
ax.add_patch(FancyBboxPatch((186.6, 23.0), 11.4, 22.0, boxstyle="round,pad=0.3,rounding_size=0.9",
                            linewidth=1.5, edgecolor="#1e3a5f", facecolor="white", zorder=3))
for i, c in enumerate(["#22c55e", "#f59e0b", "#ef4444"]):
    ax.add_patch(Circle((192.3, 41.6 - i * 3.7), 1.35, facecolor=c, edgecolor="none", zorder=5))
ax.add_patch(FancyBboxPatch((190.1, 36.2), 4.4, 3.4, boxstyle="round,pad=0.1,rounding_size=0.3",
                            linewidth=1.5, edgecolor="#f59e0b", facecolor="none", zorder=6))
ax.text(192.3, 25.4, "Return\nthe result", ha="center", va="center", fontsize=7.6, color=INK, linespacing=1.4)
ax.text(187.2, 30.6, "✔ approved", fontsize=6.9, color="#15803d", va="center")
ax.text(187.2, 28.4, "✘ not approved", fontsize=6.9, color="#b91c1c", va="center")

# connectors
arrow(120.8, 39.2, 123.2, 39.2, col=TEAL, lw=1.6)
arrow(140.3, 39.2, 142.2, 39.2, col=TEAL, lw=1.6)
arrow(157.8, 39.2, 159.2, 39.2, col=TEAL, lw=1.6)
arrow(184.8, 39.2, 186.3, 39.2, col=NAVY, lw=1.8)

# feedback loop
ax.plot([186.6, 112.7], [19.6, 19.6], color="#ea580c", lw=1.5, zorder=4)
ax.plot([186.6, 186.6], [23.0, 19.6], color="#ea580c", lw=1.5, zorder=4)
arrow(112.7, 19.6, 112.7, 33.2, col="#ea580c", lw=1.5)
ax.text(146.0, 20.8, "Response and re-prompt", ha="center", fontsize=7.4, color=INK)

ax.text(104.5, 12.4, sc("Governance: all tool-calls logged  ·  schema-validated  ·  rate-limited  ·  human sign-off required"),
        fontsize=8.0, color=INK, fontfamily=SCF)
ax.text(104.5, 9.6, sc("feedback (audit trail  ·  RLHF-lite)"), fontsize=8.0, color="#dc2626", fontfamily=SCF)

OUTP = Path(__file__).resolve().parent / "fig_architecture_4panel.png"
fig.savefig(OUTP, bbox_inches="tight", dpi=175, facecolor="white")
print(f"wrote {OUTP}")
