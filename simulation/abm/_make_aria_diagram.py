"""Generate the ARIA architecture diagram for the thesis.

Produces paper/presentation/assets/fig_aria_architecture.png, a single
layered-architecture image that the thesis §3.5a refers to.

Run with:
    python3 -m simulation.abm._make_aria_diagram
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def _box(ax, x, y, w, h, label, color, text_color="white", fontsize=10, weight="bold", subtitle=None):
    """Draw a rounded-rectangle box with label at top and subtitle below.

    Places the label near the top of the box and the (multi-line) subtitle
    beneath it so they do not overlap even for tall boxes with 2-line
    subtitles.
    """
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2, edgecolor="#334155",
        facecolor=color, zorder=3,
    )
    ax.add_patch(box)
    if subtitle:
        # Label near top; subtitle in lower 2/3 of box
        n_lines = subtitle.count("\n") + 1
        label_y = y + h - 0.25
        sub_y = y + h * 0.38 if n_lines <= 1 else y + h * 0.30
        ax.text(x + w / 2, label_y, label,
                ha="center", va="center", color=text_color,
                fontsize=fontsize, fontweight=weight, zorder=4)
        ax.text(x + w / 2, sub_y, subtitle,
                ha="center", va="center", color=text_color,
                fontsize=fontsize - 2, alpha=0.92, zorder=4)
    else:
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center", color=text_color,
                fontsize=fontsize, fontweight=weight, zorder=4)


def _arrow(ax, xy1, xy2, color="#475569", width=1.0):
    arrow = FancyArrowPatch(
        xy1, xy2, arrowstyle="-|>", mutation_scale=14,
        linewidth=width, color=color, zorder=2,
    )
    ax.add_patch(arrow)


def main() -> int:
    fig, ax = plt.subplots(figsize=(12, 7), dpi=140)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")

    # Palette
    c_model  = "#0f766e"
    c_sim    = "#7e22ce"
    c_abm    = "#c2410c"
    c_aria   = "#b91c1c"
    c_user   = "#475569"

    # Title
    ax.text(6, 7.55, "ARIA — Adaptive Response Integrated Advisor",
            ha="center", va="center", fontsize=16, fontweight="bold", color="#0f172a")
    ax.text(6, 7.2, "Seoul district-level ILI · three-layer decision-support stack",
            ha="center", va="center", fontsize=10, color="#64748b", style="italic")

    # Layer 0 — DATA (bottom)
    _box(ax, 0.3, 0.3, 11.4, 0.9, "Layer 0 — Data substrate",
         "#1e293b", fontsize=11,
         subtitle="epi_real_seoul.db (71 tables · 4.49M rows) · KDCA ILI · KMA weather · KOSIS 25×25 commuter · HIRA")

    # Layer 1 — Forecasting + SEIR + ABM (three side-by-side boxes)
    _box(ax, 0.3, 1.5, 3.6, 1.5, "Layer 1a — Forecasting registry",
         c_model, fontsize=11,
         subtitle="66 models · 8 families · WIS/CRPS/PICP\nRank 1 NegBinGLM · Rank 2 ElasticNet · Rank 3 BayesRidge")
    _box(ax, 4.2, 1.5, 3.6, 1.5, "Layer 1b — SEIR-V-D kernel",
         c_sim, fontsize=11,
         subtitle="25-gu commuter-coupled · 150 ODEs\nR₀=1.37 · RK4 Δt=0.25d · mass-conservation 10⁻¹⁵")
    _box(ax, 8.1, 1.5, 3.6, 1.5, "Layer 1c — Adaptive ABM",
         c_abm, fontsize=11,
         subtitle="25 district agents · (α, κ, τ, θ)\nβᵢ(t) = β₀(1−0.6c)² · α=0 ≡ kernel")

    # Arrows from Data → Layer 1
    for xm in (2.1, 6.0, 9.9):
        _arrow(ax, (xm, 1.2), (xm, 1.5))

    # Layer 2 — ARIA (top)
    _box(ax, 1.0, 3.4, 10.0, 1.7, "Layer 2 — ARIA consultation surface",
         c_aria, fontsize=13,
         subtitle="Agent orchestrator · 12 MCP tools · Hermes SHA-256 hash-chained audit log\nProviders: Anthropic · Google · OpenAI · Ollama (gemma3 · qwen2.5 · phi3.5 · mistral · llama3.2 · deepseek-r1)")

    # Arrows Layer 1 → ARIA
    for xm in (2.1, 6.0, 9.9):
        _arrow(ax, (xm, 3.0), (xm, 3.4))

    # User query panel (top)
    _box(ax, 1.5, 5.6, 4.0, 1.0, "Epidemiologist query",
         c_user, fontsize=10,
         subtitle='"Gangnam alert next week?"\n"S3 fatigue vs S2 — policy trade-off?"')
    _box(ax, 6.5, 5.6, 4.0, 1.0, "ARIA response",
         "#0369a1", fontsize=10,
         subtitle="Grounded, hedged, structured\n+ audit trail (§7.4)")

    # Bidirectional arrows user ↔ ARIA
    _arrow(ax, (3.5, 5.6), (3.5, 5.1), color="#3b82f6", width=1.4)
    _arrow(ax, (8.5, 5.1), (8.5, 5.6), color="#0369a1", width=1.4)

    # Side annotations
    ax.text(0.1, 2.25, "Layer 1", ha="left", va="center",
            fontsize=9, color="#64748b", rotation=90, style="italic")
    ax.text(0.1, 4.25, "Layer 2 (ARIA)", ha="left", va="center",
            fontsize=9, color="#b91c1c", rotation=90, fontweight="bold")
    ax.text(0.1, 6.1, "User", ha="left", va="center",
            fontsize=9, color="#64748b", rotation=90, style="italic")

    # Bottom-right provenance
    ax.text(11.85, 0.05, "ARIA · 2026-04",
            ha="right", va="bottom", fontsize=7, color="#94a3b8", style="italic")

    out_dir = Path("paper/presentation/assets")
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "fig_aria_architecture.png"
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight", dpi=160, facecolor="white")
    print(f"wrote {png_path}")
    plt.close(fig)

    # Also copy to a location the docx can embed (image in word/media/)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
