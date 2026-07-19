"""§4 TIER-2 figure: full per-model relative-WIS leaderboard (all 41 finite-WIS models).

Reads the SSOT relative-WIS leaderboard produced by per_model_eval / sci_supplement
(REAL artifact, no retraining) and the live registry family map. Renders a thesis-style
horizontal forest of relative-WIS vs the FluSight-Baseline, with the 0.6-0.9 skill band
shaded, champion (FusedEpi) / co-champion (NegBinGLM) / baseline highlighted, and the
skill cutoff (relative-WIS = 1.0) marked.

Style matches simulation/scripts/regenerate_stale_thesis_figures.py
(DejaVu Sans, savefig.dpi=160, TEAL/NAVY/amber palette).

Constraints honoured: REAL data only; seed=42 (no stochastic step here); DB reads via
simulation.database.read_only_connect (not needed — leaderboard already on disk); NO uv sync.

Run:
    .venv/bin/python paper/ch4_new_assets/fig_relwis_leaderboard.py
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
    "font.size": 11,
    "savefig.dpi": 160,
    "figure.dpi": 130,
})

TEAL = "#0f766e"   # champion / skillful
NAVY = "#1e3a5f"
AMBER = "#b45309"  # co-champion (NegBinGLM)
GREY = "#6b7280"
BAND = "#d1fae5"   # 0.6-0.9 skill band fill

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "simulation" / "results"
LEADERBOARD = RES / "sci_supplement" / "sci_relative_wis_leaderboard.csv"
OUT = ROOT / "paper" / "ch4_new_assets" / "fig_relwis_leaderboard.png"

np.random.seed(42)  # determinism contract (no stochastic op; documented)


def _family_map() -> dict[str, str]:
    """model -> broad family from the live registry SSOT (CATEGORY_MODELS)."""
    from simulation.models.registry import CATEGORY_MODELS

    m2c: dict[str, str] = {}
    for cat, models in CATEGORY_MODELS.items():
        for m in models:
            m2c[m] = cat
    return m2c


def main() -> None:
    rows = []
    with LEADERBOARD.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)

    fam = _family_map()
    # ascending relative-WIS (best at top of the forest)
    rows.sort(key=lambda r: float(r["relative_wis_vs_baseline"]))

    names = [r["model"] for r in rows]
    rel = [float(r["relative_wis_vs_baseline"]) for r in rows]

    ys = np.arange(len(names))[::-1]  # top = best
    colors = []
    for n in names:
        if n == "FusedEpi":
            colors.append(TEAL)
        elif n == "NegBinGLM":
            colors.append(AMBER)
        elif n == "FluSight-Baseline":
            colors.append(NAVY)
        else:
            colors.append(GREY)

    fig, ax = plt.subplots(figsize=(8.4, max(5, len(names) * 0.27 + 1)))

    # 0.6-0.9 skill band + skill cutoff line
    ax.axvspan(0.6, 0.9, color=BAND, alpha=0.7, zorder=0,
               label="skill band 0.6-0.9")
    ax.axvline(1.0, color=NAVY, lw=1.3, ls="--", zorder=1,
               label="baseline (rel-WIS = 1.0)")

    ax.barh(ys, rel, color=colors, height=0.62, zorder=2, alpha=0.92)
    for y, v, c in zip(ys, rel, colors):
        ax.text(v + 0.07, y, f"{v:.3f}", va="center", ha="left",
                fontsize=7.2, color=c, zorder=3)

    labels = [f"{n}  [{fam.get(n, '?')}]" for n in names]
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=7.6)
    for tick, n in zip(ax.get_yticklabels(), names):
        if n == "FusedEpi":
            tick.set_color(TEAL); tick.set_fontweight("bold")
        elif n == "NegBinGLM":
            tick.set_color(AMBER); tick.set_fontweight("bold")
        elif n == "FluSight-Baseline":
            tick.set_color(NAVY); tick.set_fontweight("bold")

    ax.set_xlabel("Relative WIS vs FluSight-Baseline  (lower = better; < 1.0 = skillful)",
                  fontsize=10)
    ax.set_title("Full per-model leaderboard — relative WIS (OOF), 41 finite-WIS models",
                 fontsize=11.5, color=NAVY, weight="bold")
    ax.set_xlim(0, max(rel) * 1.08)
    ax.margins(y=0.01)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    ax.grid(axis="x", color="#e5e7eb", lw=0.7, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
