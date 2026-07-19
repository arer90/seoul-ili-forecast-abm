"""Methods figure: R/P (Research / Production) full pipeline flowchart.

Renders the complete two-track pipeline as an honest flow diagram. Every stage box
is sourced 1:1 from the SSOT (simulation/pipeline/phases.py) — label, semantic name,
and track ('research' vs 'production') — so the figure can never drift from the code.

Layout (2026-07-14 redesign: vertical)
--------------------------------------
The pipeline is strictly sequential (R1 -> ... -> R12 -> handoff -> P1 -> ... -> P5),
and the thesis embeds this figure in a PORTRAIT box, so the flow runs straight DOWN a
single spine: seventeen full-width stage bars, one per phase, each a "label | name |
descriptor" row. Nothing wraps, nothing reverses, and the reader's eye never leaves
one column. (Earlier revisions laid the stages out in wrapped horizontal rows, which
either reversed the numbering or needed carriage-return connectors across the page.)

- RESEARCH lane (teal): R1..R12 stacked top to bottom.
- R9 detail panel: preproc -> mc -> stability -> HP (ENGINEERING_PRINCIPLES.md R9 internal order),
  hung off the R9 bar in the right margin so it never crosses the spine.
- PRODUCTION lane (amber): P1..P5, entered from R12 by the navy handoff arrow.

The figure is emitted at a FIXED 0.9007 width/height ratio (see ASPECT) because it is
embedded at a frozen display box (14.06 x 15.61 cm). Any other ratio would be stretched
by Word and would move the page-locked layout (scripts/check_page_lock.py).

Honesty (ENGINEERING_PRINCIPLES.md #5 / G-163): bars list ONLY the phases declared in PHASES. The two
CLI-only auxiliary entries (Pinf inference, Pov overseas) are drawn as a footnote, NOT
in the P1..P5 chain, because phases.py marks them is_cli=True and they sit outside the
end-to-end production chain (phases.py:49-50).

Run:
    .venv/bin/python paper/methods_assets/fig_rp_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

# --- SSOT import: the figure's content is the code's content -----------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from simulation.pipeline.phases import PHASES  # noqa: E402  (path tweak above)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "savefig.dpi": 170,
    "figure.dpi": 130,
})

# --- palette -----------------------------------------------------------------
TEAL = "#0f766e"        # research accent
TEAL_FILL = "#ecfdf5"   # research bar fill
TEAL_LANE = "#f7fdfb"   # research lane wash
TEAL_RULE = "#a7d9d1"   # in-bar divider
AMBER = "#b45309"       # production accent
AMBER_FILL = "#fffaf0"  # production bar fill
AMBER_LANE = "#fffdf7"  # production lane wash
AMBER_RULE = "#e8c69a"
NAVY = "#1e3a5f"        # handoff
GREY = "#6b7280"
FAINT = "#cbd5e1"
INK = "#1f2937"

# The thesis embeds this PNG in a frozen 14.06 x 15.61 cm box. Emit exactly that ratio
# so Word never stretches it and the page lock holds.
ASPECT = 14.06 / 15.61  # 0.9007

OUT = Path(__file__).resolve().parent / "fig_rp_pipeline.png"

# --- terse descriptors -------------------------------------------------------
DESC = {
    "data": "load / feature engineering",
    "baseline": "BASIC-feature fit",
    "external": "exogenous covariates",
    "wfcv": "walk-forward CV",
    "diagnostics": "residual checks",
    "dm_test": "Diebold-Mariano",
    "intervals": "prediction intervals",
    "scoring": "WIS scoring",
    "per_model_optimize": "select + tune",
    "per_model_eval": "129-metric evaluation",
    "shap": "SHAP / XAI",
    "comprehensive_eval": "R9 / R10 summary",
    "real_forecaster": "rolling 1-step",
    "family_deploy": "per-family champion",
    "abm": "SEIR-V-D simulation",
    "aria": "LLM layer",
    "web": "serve / visualise",
}
SHORT = {
    "per_model_optimize": "model opt",
    "per_model_eval": "model eval",
    "comprehensive_eval": "comprehensive",
}

# --- geometry (0-100 canvas) -------------------------------------------------
BAR_X, BAR_W = 5.0, 67.0          # stage bars
CX = BAR_X + BAR_W / 2            # 38.5 — the single flow spine
GUT = 11.0                        # label gutter inside each bar
BAR_H = 3.15

R_LANE = (37.0, 92.5)             # research lane (bottom, top)
R_TOP = 88.2                      # top edge of the R1 bar
R_PITCH = 4.13                    # bar-to-bar spacing (12 bars -> R12 bottom ~39.4)

P_LANE = (6.5, 33.0)              # production lane
P_TOP = 29.3
P_PITCH = 4.55

PANEL = (75.5, 37.5, 23.0, 30.0)  # R9 detail: x, y, w, h  (right margin, clear of spine)


def _split_tracks():
    """Partition PHASES into the research chain, the P1..P5 production chain, and aux CLIs.

    Returns:
        (research, production, aux) — each a list of (label, name) tuples in PHASES order.
    """
    research, production, aux = [], [], []
    for label, track, name, _is_cli in PHASES:
        if track == "research":
            research.append((label, name))
        elif label[1:].isdigit():        # P1..P5 = the end-to-end production chain
            production.append((label, name))
        else:                            # Pinf, Pov = standalone serving CLIs
            aux.append((label, name))
    return research, production, aux


def _bar(ax, y, label, name, *, fill, accent, rule):
    """One stage bar: [label] | semantic name ............ descriptor.

    Args:
        y: bottom edge of the bar on the 0-100 canvas.
    """
    ax.add_patch(FancyBboxPatch(
        (BAR_X, y), BAR_W, BAR_H,
        boxstyle="round,pad=0.006,rounding_size=0.035",
        linewidth=1.0, edgecolor=accent, facecolor=fill, zorder=3,
    ))
    cy = y + BAR_H / 2
    ax.text(BAR_X + GUT / 2, cy, label, ha="center", va="center",
            fontsize=9.4, fontweight="bold", color=accent, zorder=4)
    ax.plot([BAR_X + GUT, BAR_X + GUT], [y + 0.45, y + BAR_H - 0.45],
            color=rule, linewidth=0.8, zorder=4)
    ax.text(BAR_X + GUT + 3.0, cy, SHORT.get(name, name), ha="left", va="center",
            fontsize=8.0, color=INK, zorder=4)
    if DESC.get(name):
        ax.text(BAR_X + BAR_W - 3.0, cy, DESC[name], ha="right", va="center",
                fontsize=6.5, color=GREY, style="italic", zorder=4)


def _lane(ax, bottom, top, wash, accent, name, note):
    """Lane wash + accent rail + header row (name left, note right — spine stays clear)."""
    ax.add_patch(FancyBboxPatch(
        (0, bottom), 100, top - bottom,
        boxstyle="round,pad=0,rounding_size=0.6",
        linewidth=0, facecolor=wash, zorder=0,
    ))
    ax.add_patch(FancyBboxPatch(
        (0, bottom), 0.85, top - bottom,
        boxstyle="square,pad=0", linewidth=0, facecolor=accent, zorder=1,
    ))
    hy = top - 2.3
    ax.text(BAR_X, hy, name, ha="left", va="center",
            fontsize=8.6, fontweight="bold", color=accent)
    ax.text(98.5, hy, note, ha="right", va="center", fontsize=7.0, color=GREY)


def _down(ax, y0, y1, color, lw=1.15, ms=8, x=CX):
    ax.add_patch(FancyArrowPatch(
        (x, y0), (x, y1), arrowstyle="-|>", mutation_scale=ms,
        linewidth=lw, color=color, zorder=2, shrinkA=0, shrinkB=0,
    ))


def main():
    research, production, aux = _split_tracks()

    # Honesty guard: the figure must match the SSOT exactly.
    assert [l for l, _ in research] == [f"R{i}" for i in range(1, 13)], research
    assert [l for l, _ in production] == ["P1", "P2", "P3", "P4", "P5"], production

    fig_w = 6.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_w / ASPECT))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # ---- title --------------------------------------------------------------
    ax.text(0, 97.6, "R/P pipeline", ha="left", va="center",
            fontsize=13.0, fontweight="bold", color=INK)
    ax.text(23.5, 97.4, "Research (R1–R12) → Production (P1–P5)", ha="left", va="center",
            fontsize=9.4, color=GREY)
    ax.plot([0, 100], [94.9, 94.9], color=FAINT, linewidth=0.9, zorder=1)

    # ---- RESEARCH lane: R1..R12 straight down -------------------------------
    _lane(ax, R_LANE[0], R_LANE[1], TEAL_LANE, TEAL,
          "RESEARCH", "frozen split — train → select → evaluate → report")

    ybot = {}
    for i, (label, name) in enumerate(research):
        y = R_TOP - i * R_PITCH - BAR_H
        _bar(ax, y, label, name, fill=TEAL_FILL, accent=TEAL, rule=TEAL_RULE)
        ybot[label] = (y, y + BAR_H)
        if i:                                    # arrow down from the previous bar
            _down(ax, R_TOP - (i - 1) * R_PITCH - BAR_H, y + BAR_H, TEAL)

    # ---- R9 detail panel (right margin — never crosses the spine) -----------
    px, py, pw, ph = PANEL
    ax.add_patch(FancyBboxPatch(
        (px, py), pw, ph,
        boxstyle="round,pad=0,rounding_size=0.8",
        linewidth=0.9, edgecolor=TEAL, facecolor="#f0fdfa",
        linestyle=(0, (4, 2.4)), zorder=1,
    ))
    ax.text(px + pw / 2, py + ph - 3.0, "R9  internal order",
            ha="center", va="center", fontsize=7.4, fontweight="bold", color=TEAL)

    inner = ["preproc", "mc", "stability", "HP"]
    ibx, ibw, ibh, ipitch = px + 2.5, pw - 5.0, 4.2, 5.8
    itop = py + ph - 6.5
    for j, step in enumerate(inner):
        iy = itop - j * ipitch - ibh
        ax.add_patch(FancyBboxPatch(
            (ibx, iy), ibw, ibh,
            boxstyle="round,pad=0.006,rounding_size=0.04",
            linewidth=1.0, edgecolor=TEAL, facecolor="white", zorder=3,
        ))
        ax.text(ibx + ibw / 2, iy + ibh / 2, step, ha="center", va="center",
                fontsize=7.4, color=INK, zorder=4)
        if j:
            _down(ax, itop - (j - 1) * ipitch - ibh, iy + ibh, TEAL,
                  lw=1.0, ms=6, x=ibx + ibw / 2)

    # dotted leader: R9 bar -> panel
    r9b, r9t = ybot["R9"]
    ax.plot([BAR_X + BAR_W, px], [(r9b + r9t) / 2, (r9b + r9t) / 2],
            color=TEAL, linewidth=0.9, linestyle=(0, (1.4, 1.8)), zorder=2)

    # ---- PRODUCTION lane: P1..P5 straight down ------------------------------
    _lane(ax, P_LANE[0], P_LANE[1], AMBER_LANE, AMBER,
          "PRODUCTION", "after all research — deploy + operational forecast")

    for i, (label, name) in enumerate(production):
        y = P_TOP - i * P_PITCH - BAR_H
        _bar(ax, y, label, name, fill=AMBER_FILL, accent=AMBER, rule=AMBER_RULE)
        if i:
            _down(ax, P_TOP - (i - 1) * P_PITCH - BAR_H, y + BAR_H, AMBER)

    # ---- handoff: R12 -> P1, straight down the same spine -------------------
    r12b, _ = ybot["R12"]
    _down(ax, r12b, P_TOP, NAVY, lw=1.6, ms=11)
    ax.text(CX - 2.2, (R_LANE[0] + P_LANE[1]) / 2,   # centred in the inter-lane band
            "handoff: R9 champion + ABM/ARIA gate",
            ha="right", va="center", fontsize=6.2, color=NAVY, style="italic")

    # ---- auxiliary CLIs (honest: outside the P1–P5 chain) -------------------
    aux_str = ",  ".join(f"{l} {n}" for l, n in aux)
    ax.text(50, 3.2,
            f"Auxiliary serving CLIs — standalone, outside the P1–P5 chain:  {aux_str}",
            ha="center", va="center", fontsize=6.3, color=GREY)

    # No bbox_inches='tight': the frozen display box needs the exact ASPECT above.
    fig.subplots_adjust(left=0.012, right=0.988, top=0.988, bottom=0.012)
    fig.savefig(OUT, facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
