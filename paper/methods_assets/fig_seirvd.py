#!/usr/bin/env python3
"""SEIR-V-D compartmental schematic for the Methods chapter.

Renders the metapopulation SEIR-V-D model used by the agent kernel
(``simulation/abm/agent_kernel.py``) and behavioural layer
(``simulation/abm/behavioural.py``). Compartment and rate names are kept
identical to the SSOT code so the figure and the implementation agree:

    Compartments : S, E, I, R, V, D   (agent_kernel._COMPARTMENT_NAMES)
    beta(t)      : time-varying force of infection (seasonal + behavioural)
    sigma        : E -> I  (1 / latent_period)        disease_params.sigma
    gamma        : I -> R  (1 / infectious_period)     disease_params.gamma
    nu           : S -> V  (vaccination)               agent_kernel arg ``nu``
    delta        : I -> D  (mortality / I->D rate)     agent_kernel arg ``delta``
    (1-VE) lambda: V -> E  (leaky-vaccine breakthrough) disease_params VE
    omega        : R -> S  (waning immunity)            agent_kernel arg ``waning``
    omega_V      : V -> S  (waning vaccine protection)  vaccine-waning rate

The schematic also shows the two couplings that make the model a
*metapopulation behavioural* model rather than a single-patch SEIR:

    1. 25-gu metapopulation with commuter mixing  (M, 25x25 mixing matrix)
       => district force of infection lambda_i = sum_j M_ij * beta(t) * I_j / N_j
    2. Behavioural feedback (prevalence -> beta reduction):
       dR_i/dt = alpha (I_i/N_i) - lambda_R R_i ; compliant if R_i - kappa F_i > theta
       => beta_i(t) = beta_0 (c_i / c0)^2   (quadratic contact reduction)

Outputs (B5-width, print-ready):
    fig_seirvd.png   (300 dpi raster)
    fig_seirvd.pdf   (vector, optional)

Run:
    .venv/bin/python paper/methods_assets/fig_seirvd.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # portability: headless render on macOS/Linux/Windows
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D

# ── Palette (colour-blind-safe, muted; print-friendly) ─────────────────────
C_S = "#4878CF"   # Susceptible   (blue)
C_E = "#EE9D34"   # Exposed       (amber)
C_I = "#D1453B"   # Infectious    (red)
C_R = "#59A14F"   # Recovered     (green)
C_V = "#7B68B6"   # Vaccinated    (purple)
C_D = "#595959"   # Deceased      (grey)
EDGE = "#2B2B2B"
FLOW = "#2B2B2B"
FEEDBACK = "#B5651D"   # behavioural feedback arrow (warm brown)
COUPLE = "#1F6F8B"     # metapopulation coupling (teal)
TEXT = "#1A1A1A"

# B5 page is 176 mm wide; usable text width ~ 132 mm ~ 5.2 in.
FIG_W = 7.2   # render slightly wider, scaled into B5 column in docx
FIG_H = 4.35


def _box(ax, x, y, label, sub, color, w=1.05, h=0.92, fontsize=20):
    """Draw a rounded compartment box centred at (x, y)."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.8, edgecolor=EDGE, facecolor=color, alpha=0.92,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y + 0.10, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="white", zorder=4)
    ax.text(x, y - 0.27, sub, ha="center", va="center",
            fontsize=8.0, color="white", zorder=4)
    return (x, y, w, h)


def _flow(ax, p0, p1, label, *, color=FLOW, rad=0.0, lw=2.0,
          lab_dx=0.0, lab_dy=0.20, fontsize=12.5, ls="-", label_box=True):
    """Curved arrow from p0 to p1 with a rate label near its midpoint."""
    arr = FancyArrowPatch(
        p0, p1,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>", mutation_scale=18,
        linewidth=lw, color=color, linestyle=ls, zorder=2,
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(arr)
    mx = (p0[0] + p1[0]) / 2 + lab_dx
    my = (p0[1] + p1[1]) / 2 + lab_dy
    bbox = dict(boxstyle="round,pad=0.18", facecolor="white",
                edgecolor="none", alpha=0.85) if label_box else None
    ax.text(mx, my, label, ha="center", va="center",
            fontsize=fontsize, color=color, zorder=5, bbox=bbox)


def build(out_dir: Path) -> tuple[Path, Path]:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "axes.linewidth": 0.0,
    })

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Compartment positions ──────────────────────────────────────────────
    yC = 3.7   # main S->E->I->R row
    xS, xE, xI, xR = 1.5, 4.1, 6.7, 9.3
    yV = 5.9   # V above S
    yD = 1.4   # D below I

    bS = _box(ax, xS, yC, "S", "Susceptible", C_S)
    bE = _box(ax, xE, yC, "E", "Exposed", C_E)
    bI = _box(ax, xI, yC, "I", "Infectious", C_I)
    bR = _box(ax, xR, yC, "R", "Recovered", C_R)
    bV = _box(ax, xS, yV, "V", "Vaccinated", C_V)
    bD = _box(ax, xI, yD, "D", "Deceased", C_D)

    hw = bS[2] / 2  # half-width
    hh = bS[3] / 2  # half-height

    # ── Main horizontal flows S -> E -> I -> R ──────────────────────────────
    _flow(ax, (xS + hw, yC), (xE - hw, yC),
          r"$\beta(t)\,\frac{I}{N}$", lab_dy=0.34, fontsize=14)
    _flow(ax, (xE + hw, yC), (xI - hw, yC),
          r"$\sigma$", lab_dy=0.30, fontsize=15)
    _flow(ax, (xI + hw, yC), (xR - hw, yC),
          r"$\gamma$", lab_dy=0.30, fontsize=15)

    # ── Vaccination S -> V (up) and vaccine waning V -> S (down) ────────────
    _flow(ax, (xS - 0.24, yC + hh), (xS - 0.24, yV - hh),
          r"$\nu$", lab_dx=-0.42, lab_dy=0.0, fontsize=15)
    _flow(ax, (xS + 0.24, yV - hh), (xS + 0.24, yC + hh),
          r"$\omega_V$", lab_dx=0.44, lab_dy=-0.02, fontsize=12, color=C_V)

    # ── Mortality I -> D (vertical, down) ───────────────────────────────────
    _flow(ax, (xI, yC - hh), (xI, yD + hh),
          r"$\delta$", lab_dx=0.42, lab_dy=0.0, fontsize=15)

    # ── Annotate flow meanings under the row (small print) ──────────────────
    ax.text(xS + (xE - xS) / 2, yC - 0.62, "force of infection",
            ha="center", va="center", fontsize=7.2, color=TEXT, style="italic")
    ax.text(xE + (xI - xE) / 2, yC - 0.62, "latent\n($1/$latent period)",
            ha="center", va="center", fontsize=7.2, color=TEXT, style="italic")
    ax.text(xI + (xR - xI) / 2, yC - 0.62, "recovery\n($1/$infectious period)",
            ha="center", va="center", fontsize=7.2, color=TEXT, style="italic")
    ax.text(xS - 1.02, (yC + yV) / 2, "vaccination",
            ha="center", va="center", fontsize=7.2, color=TEXT,
            style="italic", rotation=90)
    ax.text(xI + 1.02, (yC + yD) / 2 - 0.05, "mortality\n($I\\to D$)",
            ha="left", va="center", fontsize=7.2, color=TEXT,
            style="italic", rotation=0)

    # ── Behavioural feedback: prevalence (I) -> reduce beta(t) ──────────────
    # Curved dashed arrow from I (high prevalence) back onto the S->E beta
    # flow, landing just above the beta(t) rate label (the damped transmission).
    beta_mid_x = xS + (xE - xS) / 2
    fb = FancyArrowPatch(
        (xI - 0.18, yC + hh + 0.02),          # start at TOP edge of I
        (beta_mid_x + 0.14, yC + 0.30),       # land just above the beta arrow
        connectionstyle="arc3,rad=0.46",      # arc UP and over E (not below)
        arrowstyle="-|>", mutation_scale=16,
        linewidth=1.9, color=FEEDBACK, linestyle=(0, (5, 2)), zorder=6,
        shrinkA=2, shrinkB=1,
    )
    ax.add_patch(fb)
    ax.text(xE - 0.10, yC + 2.05,
            "behavioural feedback",
            ha="center", va="center", fontsize=8.6, color=FEEDBACK,
            fontweight="bold")
    ax.text(xE - 0.10, yC + 1.66,
            r"prevalence $\dfrac{I}{N}\;\downarrow\;\beta(t)$",
            ha="center", va="center", fontsize=9.2, color=FEEDBACK)

    # ── Leaky-vaccine breakthrough  V -> E  at  (1-VE) * force of infection ──
    bt = FancyArrowPatch(
        (xS + 0.52, yV - hh + 0.04), (xE - 0.60, yC + hh - 0.02),
        connectionstyle="arc3,rad=-0.16",
        arrowstyle="-|>", mutation_scale=14,
        linewidth=1.6, color=C_V, linestyle=(0, (5, 2)), zorder=5,
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(bt)
    ax.text(3.30, 5.02, r"$(1-\mathrm{VE})\,\lambda$",
            ha="center", va="center", fontsize=9.3, color=C_V,
            bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.9),
            zorder=7)
    ax.text(3.30, 4.74, "breakthrough", ha="center", va="center",
            fontsize=6.8, color=C_V, style="italic", zorder=7)

    # ── Waning immunity  R -> S  (rate omega): long arc below the main row ───
    wn = FancyArrowPatch(
        (xR, yC - hh), (xS, yC - hh),
        connectionstyle="arc3,rad=-0.34",
        arrowstyle="-|>", mutation_scale=16,
        linewidth=1.6, color=C_S, zorder=1,
        shrinkA=3, shrinkB=3,
    )
    ax.add_patch(wn)
    ax.text((xS + xR) / 2, 0.62, r"$\omega$   (waning $R\to S$)",
            ha="center", va="center", fontsize=9.0, color=C_S, style="italic",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9),
            zorder=6)

    # ── Metapopulation coupling banner (25-gu commuter mixing) ──────────────
    ax.text(xR + 0.55, yV + 0.05,
            "25-district metapopulation",
            ha="center", va="center", fontsize=9.0, color=COUPLE,
            fontweight="bold")
    ax.text(xR + 0.55, yV - 0.40,
            r"$\lambda_i = \sum_j M_{ij}\,\beta(t)\dfrac{I_j}{N_j}$",
            ha="center", va="center", fontsize=9.2, color=COUPLE)
    ax.text(xR + 0.55, yV - 0.92,
            "commuter coupling $M$",
            ha="center", va="center", fontsize=7.6, color=COUPLE,
            style="italic")

    # Small inter-district coupling glyph on the I box (between-gu arrows)
    cg_x, cg_y = xI, yC + hh + 0.06
    for dx in (-0.30, 0.30):
        ax.add_patch(FancyArrowPatch(
            (cg_x, cg_y), (cg_x + dx, cg_y + 0.42),
            connectionstyle=f"arc3,rad={0.3 if dx > 0 else -0.3}",
            arrowstyle="<|-|>", mutation_scale=8,
            linewidth=1.2, color=COUPLE, zorder=1, shrinkA=1, shrinkB=1,
            alpha=0.0,  # kept invisible; coupling already labelled above
        ))

    # ── Legend (compartment colours) ────────────────────────────────────────
    legend_items = [
        ("S  Susceptible", C_S), ("E  Exposed", C_E),
        ("I  Infectious", C_I), ("R  Recovered", C_R),
        ("V  Vaccinated", C_V), ("D  Deceased", C_D),
    ]
    handles = [Line2D([0], [0], marker="s", color="none",
                      markerfacecolor=c, markeredgecolor=EDGE,
                      markersize=10, label=t) for t, c in legend_items]
    leg = ax.legend(handles=handles, loc="lower center",
                    bbox_to_anchor=(0.5, -0.045), ncol=6,
                    frameon=False, fontsize=8.0, handletextpad=0.35,
                    columnspacing=1.0)
    for txt in leg.get_texts():
        txt.set_color(TEXT)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.07)

    png = out_dir / "fig_seirvd.png"
    pdf = out_dir / "fig_seirvd.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


if __name__ == "__main__":
    out = Path(__file__).resolve().parent
    png, pdf = build(out)
    print(f"wrote {png}")
    print(f"wrote {pdf}")
