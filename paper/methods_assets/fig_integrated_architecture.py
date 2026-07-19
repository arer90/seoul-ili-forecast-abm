"""Figure 27 — Integrated decision-support architecture and headline results (4-panel, high-fidelity).

Panels: (a) data/features, (b) 48-model registry + walk-forward CV + probabilistic evaluation,
(c) champion -> forecast-anchored SEIR-V-D + ABM, (d) ARIA advisory layer.
Panel (d) shows ARIA's dual role — PERFORMANCE + multi-LLM COMPARISON over an interchangeable
backend set (own from-scratch LM and its updated version + Claude, ChatGPT, Gemini, local Ollama),
with the grounding gate as the arbiter (grounding, not model capacity, carries skill).

Elegant math-script headers via STIX Two Math (matches the original); feature-correlation and
commuter-matrix heatmaps for visual parity. Documented implementation only; seed=42.

Run:    .venv/bin/python paper/methods_assets/fig_integrated_architecture.py
Output: paper/methods_assets/fig_integrated_architecture.png
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle

plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 175, "figure.dpi": 120})
rng = np.random.default_rng(42)
SCF = "STIX Two Math"   # math-script capable

# math-script uppercase map (matches original 𝒯hree-𝓛ayer look)
_SC = {"A": "\U0001D49C", "B": "ℬ", "C": "\U0001D49E", "D": "\U0001D49F", "E": "ℰ",
       "F": "ℱ", "G": "\U0001D4A2", "H": "ℋ", "I": "ℐ", "J": "\U0001D4A5",
       "K": "\U0001D4A6", "L": "ℒ", "M": "ℳ", "N": "\U0001D4A9", "O": "\U0001D4AA",
       "P": "\U0001D4AB", "Q": "\U0001D4AC", "R": "ℛ", "S": "\U0001D4AE", "T": "\U0001D4AF",
       "U": "\U0001D4B0", "V": "\U0001D4B1", "W": "\U0001D4B2", "X": "\U0001D4B3", "Y": "\U0001D4B4",
       "Z": "\U0001D4B5"}
def sc(s):  # convert uppercase to math-script (elegant caps)
    return "".join(_SC.get(ch, ch) for ch in s)

NAVY = "#1e3a5f"; TEAL = "#0f766e"; AMBER = "#b45309"; PURPLE = "#6d28d9"
GREEN = "#2f8f7d"; INK = "#1f2937"; GREY = "#64748b"
BG_A = "#dbe6f4"; BG_B = "#fbf1e2"; BG_C = "#e3efe5"; BG_D = "#ece3f6"

OUTP = Path(__file__).resolve().parents[2] / "paper" / "methods_assets" / "fig_integrated_architecture.png"
fig, ax = plt.subplots(figsize=(20.6, 11.7))
ax.set_xlim(0, 200); ax.set_ylim(0, 116); ax.axis("off")


def panel(x, y, w, h, bg, label):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.6",
                                linewidth=1.5, edgecolor="#c3ccd8", facecolor=bg, zorder=1))
    ax.text(x + 2.6, y + h - 3.2, sc(label), ha="left", va="center", fontsize=17,
            color=INK, zorder=4, fontfamily=SCF)


def box(x, y, w, h, title, fill, lines=None, fs_t=10.5, fs_l=8.4, tcol="white", edge=None, mono=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.8",
                                linewidth=1.3, edgecolor=edge or fill, facecolor=fill, zorder=3))
    ax.text(x + w / 2, y + h - (2.3 if lines else h / 2), title, ha="center", va="center",
            fontsize=fs_t, fontweight="bold", color=tcol, zorder=4,
            fontfamily="monospace" if mono else None)
    for j, ln in enumerate(lines or []):
        ax.text(x + w / 2, y + h - 4.7 - j * 2.2, ln, ha="center", va="center",
                fontsize=fs_l, color="#eaf0f7" if tcol == "white" else INK, zorder=4)


def arr(x0, y0, x1, y1, color=GREY, lw=2.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=15,
                                 linewidth=lw, color=color, zorder=2))


def heat(x, y, w, h, M, cmap):
    """draw matrix M as a grid of colored cells within (x,y,w,h)."""
    nr, ncc = M.shape
    cw, ch = w / ncc, h / nr
    cm = plt.get_cmap(cmap)
    vmin, vmax = M.min(), M.max()
    for i in range(nr):
        for j in range(ncc):
            v = (M[i, j] - vmin) / (vmax - vmin + 1e-9)
            ax.add_patch(Rectangle((x + j * cw, y + h - (i + 1) * ch), cw, ch,
                                   facecolor=cm(0.12 + 0.85 * v), edgecolor="white", lw=0.3, zorder=3))


# ── overall title ─────────────────────────────────────────────
ax.text(100, 113.6, sc("Integrated Three-Layer Decision-Support Architecture for District-Level Seasonal ILI"),
        ha="center", va="center", fontsize=19, color=INK, fontfamily=SCF)
ax.text(100, 109.6, sc("Forecasting Registry")+"   →   "+sc("Metapopulation SEIR-V-D Simulator")
        + "   →   " + sc("ARIA Agent-MCP-Hermes Advisory Surface"),
        ha="center", va="center", fontsize=12, color=GREY, fontfamily=SCF)

# ============================ (a) DATA ============================
panel(1, 60, 97, 46, BG_A, "(a)  Data pipeline & feature engineering")
for i, (t, s) in enumerate([("KDCA ILI", "337 wk · 25 districts"), ("KMA Weather", "T · RH · AH"),
                            ("Census", "25×25 commuter")]):
    box(5, 92 - i * 8.0, 24, 6.4, t, "white", [s], fs_t=10, fs_l=7.8, tcol=INK, edge=NAVY)
box(5, 63.5, 24, 6.0, "Epidemiology DB", NAVY, ["~80M rows · 85 tables"], fs_t=9.5, fs_l=7.6)
arr(17, 71.6, 17, 69.7, NAVY)
# feature-correlation heatmap (visual parity with original)
ax.text(40, 97.4, "Feature tensor  X = 337×25×7×398",
        ha="center", fontsize=8.6, color=INK, style="italic")
Hc = rng.normal(0, 1, (9, 9)); Hc = (Hc + Hc.T) / 2; np.fill_diagonal(Hc, 2.2)
heat(33, 78.5, 15, 15, Hc, "RdYlBu_r")
ax.text(40.5, 77.0, "feature corr.", ha="center", fontsize=7.4, color=GREY)
arr(49, 86, 55, 86, NAVY)
feats = ["Lag embedding (ℓ 1–52w)", "Fourier sin/cos 52w·26w", "Regime dummies (3 eras)",
         "Commuter share c_ij", "Age-pop weights a∈ℝ⁷", "Google Trends z-score", "Log1p · Winsorize"]
ax.text(78, 98.4, sc("Feature engineer")+"  F_θ", ha="center", fontsize=10.5, color=AMBER, fontfamily=SCF)
for i, f in enumerate(feats):
    box(62, 93.6 - i * 4.05, 33, 3.3, f, "#4b5563", fs_t=8.2)

# ============================ (b) MODELS ============================
panel(102, 60, 97, 46, BG_B, "(b)  Model registry · walk-forward CV · probabilistic evaluation")
ax.text(120, 100.5, "Registry  |R| = 45 active models", ha="center", fontsize=9.8, fontweight="bold", color=INK)
fams = [("Stat", "#3b82c4", 5), ("Epi/Bay", "#4fa07a", 6), ("Linear", "#e0a83d", 6),
        ("Tree", "#c85450", 8), ("DL", "#8b3a3a", 9), ("Fnd/TF", "#7c53b3", 8)]
for r, (fn, col, cnt) in enumerate(fams):
    ax.text(106, 96.6 - r * 2.9, fn, ha="left", va="center", fontsize=8.2, color=INK)
    for c in range(cnt):
        ax.add_patch(Rectangle((114 + c * 2.35, 95.5 - r * 2.9), 2.05, 2.05, facecolor=col,
                               edgecolor="white", lw=0.6, zorder=3))
ax.text(120, 78.3, "fit × 45 models", ha="center", fontsize=8.0, color=GREY, style="italic")
ax.text(158, 100.5, "Walk-forward CV", ha="center", fontsize=9.8, fontweight="bold", color=INK)
for i in range(5):
    y = 96.2 - i * 3.0
    ax.text(143, y, f"F{i+1}", ha="right", va="center", fontsize=8, color=INK)
    tr = 18 + i * 3.5
    ax.add_patch(Rectangle((145, y - 1.0), tr, 2.0, facecolor="#9cc3e8", edgecolor="white", lw=0.5, zorder=3))
    ax.add_patch(Rectangle((145 + tr, y - 1.0), 4.5, 2.0, facecolor="#e0a83d", edgecolor="white", lw=0.5, zorder=3))
    ax.add_patch(Rectangle((145 + tr + 4.5, y - 1.0), 6.0, 2.0, facecolor="#b23b3b", edgecolor="white", lw=0.5, zorder=3))
ax.text(145, 78.6, "2019W01", ha="left", fontsize=7, color=GREY)
ax.text(178, 78.6, "2025W25", ha="right", fontsize=7, color=GREY)
for lab, cx2, cc in [("train", 149, "#9cc3e8"), ("val", 162, "#e0a83d"), ("26w hold", 171, "#b23b3b")]:
    ax.add_patch(Rectangle((cx2, 74.8), 1.6, 1.6, facecolor=cc, zorder=3))
    ax.text(cx2 + 2.2, 75.6, lab, ha="left", va="center", fontsize=7, color=INK)
evals = ["WIS (primary)", "CRPS", "R² (OOF)", "PIT · KS", "DM test", "PICP 95%", "Conformal"]
for i, e in enumerate(evals):
    box(184, 96.6 - i * 4.35, 13.5, 3.5, e, "white", fs_t=7.6, tcol=INK, edge=AMBER)
for i, (nm, mc) in enumerate([("FusedEpi", "#e0b000"), ("GAM-Spline", "#9aa5b1"), ("TiRex", "#c07b3a")]):
    ax.add_patch(Circle((105.5 + i * 16.5, 64.6), 0.95, facecolor=mc, edgecolor="white", lw=0.9, zorder=4))
    ax.text(107.0 + i * 16.5, 64.6, nm, ha="left", va="center", fontsize=9.2, fontweight="bold", color=INK, zorder=4)
ax.text(154, 64.6, "(top-3 by relative-WIS vs baseline: 0.71 · 0.75 · 0.79;  champion FusedEpi, WIS 3.28)", ha="left", va="center",
        fontsize=8.4, color=GREY, style="italic", zorder=4)

# ============================ (c) ABM ============================
panel(1, 3, 97, 45, BG_C, "(c)  Champion → forecast-anchored SEIR-V-D + ABM")
ax.text(22, 40.0, sc("Champion FusedEpi")+"  →  ABM forcing", ha="center", fontsize=9.6,
        fontweight="bold", color=GREEN, fontfamily=SCF)
comps = [("S", "#3b82c4"), ("E", "#e0a83d"), ("I", "#c85450"), ("R", "#4fa07a"), ("V", "#2f8f7d"), ("D", "#4b5563")]
labs = ["Suscept.", "Exposed", "Infect.", "Recov.", "Vacc.", "Death"]
greek = ["β·λ", "σ", "γ", "ν", "μ"]
cxs = np.linspace(9, 74, 6)
for i, (c, col) in enumerate(comps):
    ax.add_patch(Circle((cxs[i], 31), 3.3, facecolor=col, edgecolor="white", lw=1.5, zorder=4))
    ax.text(cxs[i], 31, c, ha="center", va="center", fontsize=12, fontweight="bold", color="white", zorder=5)
    ax.text(cxs[i], 26.6, labs[i], ha="center", va="center", fontsize=7.0, color=INK)
    if i < 5:
        arr(cxs[i] + 3.4, 31, cxs[i + 1] - 3.4, 31, INK, lw=1.5)
        ax.text((cxs[i] + cxs[i + 1]) / 2, 33.2, greek[i], ha="center", fontsize=9, color="#b23b3b", style="italic")
# commuter matrix heatmap (Seoul, 25 districts)
Cm = np.abs(rng.normal(0, 0.25, (14, 14))); np.fill_diagonal(Cm, 2.4)
for k in range(1, 14):
    Cm[k, k-1] = Cm[k-1, k] = 1.1
heat(82, 22.5, 14, 14, Cm, "Blues")
ax.text(89, 21.0, "commuter c ∈ 25 districts", ha="center", fontsize=7.2, color=GREY)
ax.text(42, 21.5, "dS/dt = −β·Σ_h M_gh·I_h/N_h·S − ν·S     dI/dt = σE − γI − μ_I·I",
        ha="center", fontsize=8.2, color=INK, family="monospace")
ax.text(42, 18.6, "conservation:  S+E+I+R+V+D = N   (per step, per district × age)",
        ha="center", fontsize=7.8, color=GREY, family="monospace")
ax.text(42, 14.6, "A = 7 age groups (0–4 · 5–14 · 15–24 · 25–44 · 45–54 · 55–64 · 65+)",
        ha="center", fontsize=7.8, color=INK)
# Two lines, not one: as a single line this ran off panel (c) and printed over panel (d).
ax.text(50, 10.2, "forward R²  0.557 behavior-on  vs  0.041 off  ·  best 0.722",
        ha="center", fontsize=8.6, fontweight="bold", color=GREEN)
ax.text(50, 7.4, "Conformal PI (α=0.1)  ·  epi-validity gate (mass 1.9e-16, Rt ∈ [0.3, 8])",
        ha="center", fontsize=8.2, fontweight="bold", color=GREEN)
ax.text(50, 4.8, "outputs:  ŷ per district × age × week", ha="center", fontsize=8.0, color=GREY)

# ============================ (d) ARIA — CORRECTED ============================
panel(102, 3, 97, 45, BG_D, "(d)  ARIA · performance + multi-LLM comparison")
box(105, 33.5, 20, 7.6, "PH query", "white", ['"Gangnam ILI', 'alert next week?"'], fs_t=9, fs_l=7.4,
    tcol=INK, edge=PURPLE)
arr(125, 37.3, 129, 37.3, PURPLE)
# ★ interchangeable LLM backends (compared, weak->strong)
backs = [("from-scratch", "#8b3a3a"), ("from-scratch v2", "#a24b4b"), ("Ollama ×N", "#334155"),
         ("Gemini", "#b45309"), ("ChatGPT", "#0f766e"), ("Claude", "#6d28d9")]
for i, (bn, bc) in enumerate(backs):
    r, c = divmod(i, 3)
    xx = 129 + c * 22.7; yy = 41.8 - r * 4.4
    ax.add_patch(FancyBboxPatch((xx, yy - 3.3), 21.2, 3.3, boxstyle="round,pad=0.13,rounding_size=0.6",
                                linewidth=1.1, edgecolor=bc, facecolor="white", zorder=4))
    ax.text(xx + 10.6, yy - 1.65, bn, ha="center", va="center", fontsize=7.7, color=bc, fontweight="bold", zorder=5)
ax.annotate("", xy=(196, 30.0), xytext=(129, 30.0),
            arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=1.4), zorder=4)
ax.text(162, 31.0, "grounding recall  weak → strong   (from-scratch 0.44 … Claude 1.0)  ·  grounding gate = arbiter",
        ha="center", fontsize=7.2, color=INK, style="italic")
arr(134, 27.4, 134, 24.6, NAVY)
# MCP tools
box(105, 15.5, 57, 11.6, "MCP epi server  —  12 read-only tools", TEAL, fs_t=9.4)
tools = ["query_db", "forecast", "model_cmp", "rt_estimate", "outbreak", "lead_time",
         "validity", "ili_rag", "scenario", "shap_feat", "intl_cmp", "coupled_fwd"]
for i, t in enumerate(tools):
    r, c = divmod(i, 4)
    xx = 107 + c * 13.7; yy = 22.4 - r * 3.1
    ax.add_patch(FancyBboxPatch((xx, yy - 2.5), 12.8, 2.5, boxstyle="round,pad=0.1,rounding_size=0.5",
                                linewidth=0.8, edgecolor="#0b3d3a", facecolor="#e6fbf5", zorder=4))
    ax.text(xx + 6.4, yy - 1.25, t, ha="center", va="center", fontsize=6.7, color=INK, family="monospace", zorder=5)
box(165, 21.9, 32, 5.2, "Grounding · CoVe", GREEN, ["numeric + semantic · Self-Ask · verify"], fs_t=8.6, fs_l=7.0)
box(165, 15.5, 32, 5.2, "Hermes audit", AMBER, ["hash-chain · provenance · rate-limit"], fs_t=8.6, fs_l=7.0)
arr(133, 15.4, 133, 12.6, NAVY)
box(105, 6.0, 92, 5.4, "Human-in-the-loop output", NAVY,
    ["approved / not approved → the epidemiologist decides   ·   all tool-calls logged · schema-validated · human sign-off"],
    fs_t=9.4, fs_l=7.2)

fig.savefig(OUTP, bbox_inches="tight", dpi=175, facecolor="white")
plt.close(fig)
print(f"wrote {OUTP}")
