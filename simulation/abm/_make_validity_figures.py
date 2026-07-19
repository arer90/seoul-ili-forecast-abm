"""Generate validity / significance figures for thesis §6.0.

Produces three PNG files under paper/presentation/assets/:
  fig_validity_trail.png       — 6-tier validity flow (conceptual)
  fig_wis_ranking.png          — top/bottom 10 models by post-E WIS
  fig_policy_shift.png         — S1–S6 peak shift bar chart

Run: python3 -m simulation.abm._make_validity_figures
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


OUT = Path("paper/presentation/assets")
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Fig A — Validity trail
# ---------------------------------------------------------------------------
def fig_validity_trail() -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
    ax.set_xlim(0, 11); ax.set_ylim(0, 5.5); ax.axis("off")
    ax.set_aspect("equal")
    ax.text(5.5, 5.2, "Six-tier validity trail (§6.0)", ha="center", fontsize=14,
            fontweight="bold", color="#0f172a")
    ax.text(5.5, 4.85, "each tier is claimed independently and is traceable to its supporting section",
            ha="center", fontsize=9, color="#64748b", style="italic")
    tiers = [
        ("Construct",  "KDCA ILI = per 1 000\noutpatients",       "§3.1",     "#1e40af"),
        ("Internal",   "Back-door adjustment\n+ shift(1) guard",   "§3.1a [71]", "#0f766e"),
        ("External",   "25-gu commuter matrix\n(Census 2020)",      "§3.4",     "#b45309"),
        ("Temporal",   "Regime-split DM +\nBH-FDR q<0.05",           "§4.8, §4.12n", "#be185d"),
        ("Spatial",    "4-D tensor preserved\n337×25×7×309",         "§3.3",     "#6d28d9"),
        ("Predictive", "26-week split-conformal\nholdout (S0-1)",   "§3.5, §3.6", "#0e7490"),
    ]
    x0, y0, w, h = 0.3, 2.7, 1.65, 1.5
    for i, (name, desc, ref, color) in enumerate(tiers):
        x = x0 + i * 1.76
        box = FancyBboxPatch((x, y0), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.2, edgecolor="#334155", facecolor=color)
        ax.add_patch(box)
        ax.text(x + w / 2, y0 + h - 0.22, name, ha="center", va="center",
            color="white", fontsize=11, fontweight="bold")
        ax.text(x + w / 2, y0 + h / 2 - 0.05, desc, ha="center", va="center",
            color="white", fontsize=8.5, alpha=0.95)
        ax.text(x + w / 2, y0 + 0.18, ref, ha="center", va="center",
            color="white", fontsize=8, alpha=0.85, style="italic")
        if i < len(tiers) - 1:
            arrow = FancyArrowPatch((x + w, y0 + h / 2), (x + w + 0.11, y0 + h / 2),
                arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color="#475569")
            ax.add_patch(arrow)

    ax.text(0.3, 2.1, "Input:", ha="left", fontsize=9, fontweight="bold", color="#1e293b")
    ax.text(1.15, 2.1, "KDCA sentinel ILI (2019 W01 – 2025 W25)  +  KOSIS commuter matrix  +  KMA weather",
            ha="left", fontsize=9, color="#1e293b")
    ax.text(0.3, 1.6, "Output:", ha="left", fontsize=9, fontweight="bold", color="#1e293b")
    ax.text(1.15, 1.6, "defensible claim about Seoul 25-gu ILI forecasting + behavioural-ABM counterfactual",
            ha="left", fontsize=9, color="#1e293b")

    ax.text(0.3, 0.9, "Each tier decouples a distinct failure mode: construct = mis-measurement; internal = confounding;",
            ha="left", fontsize=8, color="#64748b")
    ax.text(0.3, 0.6, "external = generalisability; temporal = regime shift; spatial = aggregation bias; predictive = leakage.",
            ha="left", fontsize=8, color="#64748b")

    fig.savefig(OUT / "fig_validity_trail.png", bbox_inches="tight", dpi=160, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig B — WIS ranking (top/bottom 10) from post_E_eval.json
# ---------------------------------------------------------------------------
def fig_wis_ranking() -> None:
    from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
    with open(get_results_dir() / "post_E_eval.json", encoding="utf-8") as f:
        d = json.load(f)
    details = [x for x in d["details"] if "wis" in x and "model" in x]
    rows = sorted([(x["model"], float(x["wis"])) for x in details], key=lambda r: r[1])
    top10 = rows[:10]
    bot10 = rows[-10:][::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    def _draw(ax, data, title, base_color):
        names = [n for n, _ in data]
        vals = [v for _, v in data]
        bars = ax.barh(range(len(names)), vals, color=base_color, edgecolor="#1e293b")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("WIS (post-E conformal, lower is better)", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.25)
        # value labels
        for bar, v in zip(bars, vals):
            ax.text(v + max(vals) * 0.008, bar.get_y() + bar.get_height() / 2,
                    f"{v:.2f}", va="center", fontsize=8, color="#334155")

    _draw(ax1, top10, "Top 10 models by WIS (best)", "#0f766e")
    _draw(ax2, bot10, "Bottom 10 models by WIS (worst)", "#b91c1c")

    fig.suptitle("Fig 6.0b — Post-E WIS ranking with 95 % bootstrap CIs available in ledger",
                 fontsize=11, fontweight="bold", y=1.02, color="#0f172a")
    fig.tight_layout()
    fig.savefig(OUT / "fig_wis_ranking.png", bbox_inches="tight", dpi=160, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig C — S1-S6 policy peak-shift bar chart with public-health interpretation
# ---------------------------------------------------------------------------
def fig_policy_shift() -> None:
    import csv
    rows = []
    from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
    with open(get_results_dir() / "abm_scenarios_v1" / "policy_table.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    ids = [r["id"] for r in rows]
    names = [r["name"].replace("(behaviour off)", "(off)").replace("Vaccination game (Bauch imitation)", "Vax game")
             for r in rows]
    shifts = [float(r["peak_shift_pct_vs_S1"]) for r in rows]
    compliance = [float(r["mean_compliance"]) for r in rows]
    attack = [float(r["attack_rate_city"]) for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=150)

    # Peak shift
    axes[0].bar(ids, shifts,
                color=["#94a3b8", "#16a34a", "#f59e0b", "#0284c7", "#a855f7", "#ec4899"],
                edgecolor="#1e293b")
    axes[0].set_title("Peak shift % vs S1 baseline", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("% change in city peak I", fontsize=9)
    axes[0].axhline(0, color="#334155", lw=0.8)
    axes[0].grid(True, axis="y", alpha=0.25)
    for i, (v, n) in enumerate(zip(shifts, names)):
        label = f"{v:+.1f}%" if i > 0 else "baseline"
        axes[0].text(i, v - 3, label, ha="center", fontsize=8.5, color="#0f172a")

    # Compliance
    axes[1].bar(ids, compliance,
                color=["#94a3b8", "#16a34a", "#f59e0b", "#0284c7", "#a855f7", "#ec4899"],
                edgecolor="#1e293b")
    axes[1].set_title("Mean compliance-day fraction", fontsize=11, fontweight="bold")
    axes[1].set_ylim(0, 0.22)
    axes[1].set_ylabel("fraction", fontsize=9)
    axes[1].grid(True, axis="y", alpha=0.25)
    for i, v in enumerate(compliance):
        axes[1].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8.5, color="#0f172a")

    # Attack rate
    axes[2].bar(ids, attack,
                color=["#94a3b8", "#16a34a", "#f59e0b", "#0284c7", "#a855f7", "#ec4899"],
                edgecolor="#1e293b")
    axes[2].set_title("City attack rate", fontsize=11, fontweight="bold")
    axes[2].set_ylim(0, 0.42)
    axes[2].set_ylabel("fraction", fontsize=9)
    axes[2].grid(True, axis="y", alpha=0.25)
    for i, v in enumerate(attack):
        axes[2].text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=8.5, color="#0f172a")

    fig.suptitle("Fig 4.16 — S1–S6 policy comparison (behavioural multi-agent layer)",
                 fontsize=12, fontweight="bold", y=1.02, color="#0f172a")
    fig.tight_layout()
    fig.savefig(OUT / "fig_policy_shift.png", bbox_inches="tight", dpi=160, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    fig_validity_trail()
    print(f"wrote {OUT / 'fig_validity_trail.png'}")
    fig_wis_ranking()
    print(f"wrote {OUT / 'fig_wis_ranking.png'}")
    fig_policy_shift()
    print(f"wrote {OUT / 'fig_policy_shift.png'}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
