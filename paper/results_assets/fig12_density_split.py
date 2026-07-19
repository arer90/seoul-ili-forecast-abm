"""fig12_density_split.py — Figure 12 split + enlarged (density down-scaling validation).

PURPOSE
    The thesis Figure 12 ("Indirect validation of the density down-scaling")
    packed two unrelated panels plus a long footnote into one small image, so
    neither the scatter nor the per-disease bars were legible at print size.
    This regenerator SPLITS the same real-data SSOT into two enlarged figures:

      Figure 19.1  (A) Down-scaling MECHANISM: per-district daytime-density ->
                   allocated agents (Spearman rho = 1.00, perfect monotone map).
      Figure 19.2  (B) Indirect CHECK: Spearman rho of per-district density vs
                   the annual notifiable respiratory-disease distribution, with
                   the honest-limitation note rendered as readable body text.

DATA SSOT (read-only, measured — no DB, no sim, no fabrication):
    simulation/results/abm_density_allocation/district_weights.csv   (density, n_agents)
    simulation/results/abm_density_allocation/validation.json        (per-disease Spearman rho)

This script reuses the loaders/labels from the original source figure
``simulation/scripts/fig_density_downscale_aria.py`` (single SSOT) and only
changes the LAYOUT (split + enlarge). No values are altered.

Output (PNG, white bg, dpi=200):
    paper/results_assets/fig12_1_downscale_mechanism.png
    paper/results_assets/fig12_2_downscale_check.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parents[2]  # paper/results_assets/ -> repo root
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the original figure's loaders + display-label maps (single SSOT).
from simulation.scripts.fig_density_downscale_aria import (  # noqa: E402
    GU_ENG,
    _DISEASE_ENG,
    _load_geojson_polygons,
    _load_validation,
    _load_weights,
    _spearman,
)

OUT_DIR = _THIS.parent
OUT_A = OUT_DIR / "fig12_1_downscale_mechanism.png"
OUT_B = OUT_DIR / "fig12_2_downscale_check.png"


def _font() -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False


def make_panel_a(weights: dict) -> None:
    """Figure 19.1 — density -> agent allocation scatter (enlarged, standalone)."""
    gus = list(weights.keys())
    dens = np.array([weights[g]["density"] for g in gus])
    nag = np.array([weights[g]["n_agents"] for g in gus])
    rho_alloc = _spearman(dens, nag)

    fig, ax = plt.subplots(figsize=(10.5, 8.0))
    ax.scatter(dens / 1e3, nag, s=130, c="#2c7fb8", edgecolor="white",
               linewidth=1.2, zorder=3)
    ax.axhline(4000, color="#c0392b", ls="--", lw=1.6, zorder=1,
               label="uniform baseline (4,000 agents / district)")
    # Label every district (room now that the panel is full-size). Alternate the
    # label side by density rank so the dense mid-cluster (~3000-4000 agents)
    # does not overprint; leaders to the right, the rest staggered left/right.
    order = sorted(gus, key=lambda g: weights[g]["density"])
    for rank, g in enumerate(order):
        right = rank % 2 == 0
        ax.annotate(GU_ENG.get(g, g),
                    (weights[g]["density"] / 1e3, weights[g]["n_agents"]),
                    fontsize=9.5,
                    xytext=(6, -3) if right else (-6, 4),
                    ha="left" if right else "right",
                    textcoords="offset points", color="#333333")
    ax.set_xlabel("Daytime living-population density (thousand persons / km^2)",
                  fontsize=14)
    ax.set_ylabel("Allocated agents (n_agents)", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_title(
        "Figure 19.1  Down-scaling mechanism: density -> agent allocation\n"
        f"Spearman rho = {rho_alloc:.2f} (perfect monotone map; "
        "agents proportional to density; n = 25 districts, total 100,000 agents)",
        fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=12, framealpha=0.9)
    ax.grid(alpha=0.25, zorder=0)
    fig.tight_layout()
    fig.savefig(OUT_A, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] {OUT_A}  (rho={rho_alloc:.3f}, n={len(gus)})")


def make_panel_b(validation: dict) -> None:
    """Figure 19.2 — per-disease Spearman rho bars (enlarged, readable note)."""
    vdict = validation["validation"]["validations"]
    items = sorted(vdict.items(),
                   key=lambda kv: kv[1]["spearman_density_vs_cases"], reverse=True)
    labels = [_DISEASE_ENG.get(k, k) for k, _ in items]
    rhos = [v["spearman_density_vs_cases"] for _, v in items]
    cases = [v["total_cases"] for _, v in items]
    is_neg = [v.get("is_primary", False) for _, v in items]
    ypos = np.arange(len(labels))
    colors = ["#41ab5d" if r >= 0.4 else ("#fdae61" if r >= 0.2 else "#d7d7d7")
              for r in rhos]

    fig, ax = plt.subplots(figsize=(11.5, 7.5))
    bars = ax.barh(ypos, rhos, color=colors, edgecolor="#333333", linewidth=0.8,
                   zorder=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{lab}\n(n={c:,.0f} cases)" for lab, c in zip(labels, cases)],
                       fontsize=12)
    ax.invert_yaxis()
    for b, r, neg in zip(bars, rhos, is_neg):
        tag = "  (weak negative control)" if neg else ""
        ax.text(r + 0.012, b.get_y() + b.get_height() / 2, f"rho = {r:.2f}{tag}",
                va="center", fontsize=12,
                fontweight="bold" if r >= 0.4 else "normal",
                color="#777777" if neg else "#1a1a1a")
    ax.axvline(0.4, color="#888", ls="--", lw=1.2, zorder=1)
    ax.set_xlim(0, max(rhos) * 1.55 + 0.05)
    ax.set_xlabel("Spearman rho (per-district density vs annual notifiable-disease cases)",
                  fontsize=13.5)
    ax.tick_params(axis="x", labelsize=12)
    ax.set_title(
        "Figure 19.2  Indirect down-scaling check: density vs annual\n"
        "notifiable respiratory-disease distribution (2020-2024 census, 25 districts)",
        fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.25, zorder=0)

    note = (
        "Honest limitation: per-district weekly ILI (2025/26) does not exist (KDCA ILI sentinels are city-level). The spatial weights are "
        "density / mechanism-based and are validated only INDIRECTLY against the per-district ANNUAL distribution of notifiable respiratory "
        "diseases (2020-2024) - not a direct weekly-ILI calibration; rho values are as measured. CONFOUND: childhood diseases (varicella, "
        "mumps, pertussis) track RESIDENTIAL child density, not daytime commuter density, so a positive rho here is suggestive but not a clean "
        "validation of the commuter-weighted allocation; pneumococcal disease (rho ~ 0.07) is shown as a weak negative control."
    )
    fig.text(0.5, -0.04, note, ha="center", va="top", fontsize=10.5, color="#555555",
             wrap=True)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(OUT_B, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] {OUT_B}  (n_disease={len(labels)})")


def main() -> int:
    _font()
    weights = _load_weights()
    validation = _load_validation()
    # Populate GU_ENG (Korean gu name -> English) so scatter labels are legible
    # in an English-only font (DejaVu Sans has no Hangul glyphs). Same SSOT loader
    # the original figure uses; geojson is read-only.
    try:
        _load_geojson_polygons()
    except Exception as exc:  # pragma: no cover - geojson should exist
        print(f"[warn] geojson load failed ({exc}); falling back to romanized keys")
    make_panel_a(weights)
    make_panel_b(validation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
