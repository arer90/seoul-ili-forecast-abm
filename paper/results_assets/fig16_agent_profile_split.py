"""fig16_agent_profile_split.py — Figure 16 split + enlarged (per-agent ABM profiles).

PURPOSE
    The thesis Figure 16 ("Per-agent profiles in the behavioural ABM") packed
    eight sub-panels (two representative-agent cards+timelines on top, four
    population-distribution charts on the bottom) into one 2x4 grid, so each
    sub-panel was too small to read. This regenerator SPLITS the same
    deterministic ABM run into two enlarged figures, grouped by what they show:

      Figure 23.1  Representative INDIVIDUAL agents: agent A and agent B, each as
                   a static-attribute card (who they are - observed) plus an
                   SEIR state timeline (how they were infected - model-derived).
      Figure 23.2  POPULATION distributions among the infected: age x sex
                   pyramid, comorbidity, occupation, and attack rate by age.

DATA SSOT (deterministic, seed=42 — identical to the source figure):
    ``simulate_with_history(N=3000, T_days=120, seed=42)`` over the Seoul
    25-district synthetic population (static attrs = observed KOSIS/HIRA/commute/
    business; SEIR path = model-derived agent-based kernel). The same run that
    produced the original docx image (representative agents 857 / 2471).

This reuses the draw helpers from the original source figure
``simulation/scripts/fig_agent_profile.py`` (single SSOT). Only the LAYOUT
changes (split into two enlarged figures). No values are altered.

Output (PNG, white bg):
    paper/results_assets/fig16_1_representative_agents.png
    paper/results_assets/fig16_2_population_distributions.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from simulation.abm.agent_history import (  # noqa: E402
    extract_agent_trajectory,
    population_summary,
    simulate_with_history,
)
from simulation.scripts.fig_agent_profile import (  # noqa: E402
    N_AGENTS,
    SEED,
    SIM_KWARGS,
    T_DAYS,
    _draw_age_sex_pyramid,
    _draw_agent_card,
    _draw_agent_timeline,
    _draw_attack_rate_by_age,
    _draw_occupation_dist,
    _draw_severity_dist,
    _infected_mask,
    _pick_representative_infected,
    _setup_korean_font,
    _state_legend_handles,
)

OUT_DIR = _THIS.parent
OUT_1 = OUT_DIR / "fig16_1_representative_agents.png"
OUT_2 = OUT_DIR / "fig16_2_population_distributions.png"


def make_agents(traj_a: dict, traj_b: dict | None, summary: dict) -> None:
    """Figure 23.1 — two representative agents (card + timeline each), enlarged."""
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.3],
                          hspace=0.30, wspace=0.18,
                          left=0.04, right=0.985, top=0.88, bottom=0.10)
    # Row 0 = agent A, row 1 = agent B (each: card | timeline).
    _draw_agent_card(fig.add_subplot(gs[0, 0]), traj_a, "Representative agent A")
    _draw_agent_timeline(fig.add_subplot(gs[0, 1]), traj_a, "Representative agent A")
    ax_b_card = fig.add_subplot(gs[1, 0])
    ax_b_tl = fig.add_subplot(gs[1, 1])
    if traj_b is not None:
        _draw_agent_card(ax_b_card, traj_b, "Representative agent B")
        _draw_agent_timeline(ax_b_tl, traj_b, "Representative agent B")
    else:
        for ax in (ax_b_card, ax_b_tl):
            ax.axis("off")
        ax_b_card.text(0.5, 0.5, "No second representative infected agent",
                       ha="center", va="center", fontsize=13)
    fig.legend(handles=_state_legend_handles(), loc="lower center", ncol=6,
               fontsize=11, frameon=True, framealpha=0.9,
               title="State color (timeline)", title_fontsize=11,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        "Figure 23.1  Representative individual agents in the behavioural ABM\n"
        f"Seoul 25-district synthetic population N={summary['n_agents']:,} | T={T_DAYS} days | seed={SEED}  |  "
        "static attrs = observed (KOSIS/HIRA/commute/business) / SEIR path = model-derived",
        fontsize=15, fontweight="bold", y=0.975)
    fig.savefig(OUT_1, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] {OUT_1}")


def make_population(result: dict, inf_mask, summary: dict) -> None:
    """Figure 23.2 — four population distributions among the infected, enlarged."""
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11))
    _draw_age_sex_pyramid(axes[0, 0], result, inf_mask)
    _draw_severity_dist(axes[0, 1], result, inf_mask)
    _draw_occupation_dist(axes[1, 0], result, inf_mask)
    _draw_attack_rate_by_age(axes[1, 1], result, inf_mask)
    for ax in axes.ravel():
        ax.title.set_fontsize(13)
    fig.suptitle(
        "Figure 23.2  Population distributions among the infected (behavioural ABM)\n"
        f"N={summary['n_agents']:,} agents, attack rate {summary['attack_rate']:.3f} | "
        f"T={T_DAYS} days | seed={SEED}  (observed-vs-infected, model-derived)",
        fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT_2, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] {OUT_2}")


def main() -> int:
    _setup_korean_font()
    print(f"[sim] simulate_with_history(N={N_AGENTS}, T_days={T_DAYS}, seed={SEED})")
    result = simulate_with_history(N_AGENTS, T_DAYS, seed=SEED, **SIM_KWARGS)
    inf_mask = _infected_mask(result)
    if int(inf_mask.sum()) == 0:
        raise SystemExit("[fig16] no infected agents — honest skip (no fabricated data).")
    summary = population_summary(result)
    reps = _pick_representative_infected(result, k=2)
    print(f"[sim] infected={int(inf_mask.sum()):,}/{summary['n_agents']:,} "
          f"(attack_rate={summary['attack_rate']:.3f}); representatives={reps}")
    traj_a = extract_agent_trajectory(result, reps[0]) if reps else None
    traj_b = extract_agent_trajectory(result, reps[1]) if len(reps) >= 2 else None
    if traj_a is None:
        raise SystemExit("[fig16] no representative infected agent — honest skip.")
    make_agents(traj_a, traj_b, summary)
    make_population(result, inf_mask, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
