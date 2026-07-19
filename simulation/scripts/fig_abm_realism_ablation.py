"""Publication figures + table for the ABM realism ablation (A / B / hybrid / EnKF).

Generates, from the leak-free anchored ablation + real-time EnKF results:
  Fig 1  realism-ablation stack (forward R² of A/B/hybrid/hybrid+EnKF vs champion)
  Fig 2  EnKF real-time trajectory (real / champion / hybrid / hybrid+EnKF)
  Fig 3  person-like panels the mean-field model cannot produce
          (offspring distribution · per-layer transmission · occupation & entity attack)
  Fig 4  hybrid force-of-infection architecture (schematic)
  Table  results table (CSV + rendered PNG)

Run: .venv/bin/python -m simulation.scripts.fig_abm_realism_ablation
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("simulation/results/figures")
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = Path("simulation/results")
_C = {"A": "#4C72B0", "B": "#DD8452", "H": "#55A868", "E": "#8172B3",
      "champ": "#C44E52", "real": "#333333"}

# SINGLE config for the whole ablation set — ablation, EnKF, and person-like are all
# generated at (N_AGENTS, N_SEEDS) so H-alone is one number everywhere (Fig Q.1 == the
# EnKF baseline), the person-like block has one authoritative config, and the EnKF lift
# is a within-run comparison. (Previously mixed 20000/5, 16000/6, 16000/3.)
N_AGENTS, N_SEEDS = 20000, 5


def _gen_data() -> dict:
    """Compute the figure data (ablation R², EnKF trajectories, person-like, entity C).

    All arms are recomputed at the single (N_AGENTS, N_SEEDS) config and the ablation
    + EnKF JSONs are rewritten so the on-disk artifacts, the figures, and the docx all
    reference one consistent run. The ablation is deterministic (fixed population +
    seeds), so re-running reproduces the A/B/H forward-R² headline exactly.
    """
    from simulation.abm.variant_ablation import (
        compare_anchored_variants, enkf_couple_forward, entity_metrics, tree_metrics)
    from simulation.abm.synthetic_population import generate_population
    from simulation.abm.network_params_from_db import derive_network_kwargs
    from simulation.abm.agent_kernel import run_agent_world

    # anchored ablation (A/B/H + ensemble) — recompute fresh at the single config and
    # persist (drops any stale person_like the old artifact carried).
    ablation = compare_anchored_variants(variants=("A", "B", "H"),
                                         n_agents=N_AGENTS, n_seeds=N_SEEDS)
    (RESULTS / "abm_variant_ablation_anchored.json").write_text(
        json.dumps(ablation, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # real-time EnKF with trajectories — same config → variant_alone == ablation H.
    enkf = enkf_couple_forward(variant="H", n_agents=N_AGENTS, n_seeds=N_SEEDS)
    (RESULTS / "abm_hybrid_enkf.json").write_text(
        json.dumps(enkf, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # person-like from hybrid runs: transmission tree + per-entity attack (variant C).
    # This is the single authoritative person-like block (offspring/layer/occupation),
    # counted per-seed by the corrected tree_metrics.
    pop = generate_population(N_AGENTS, seed=0)
    nk = derive_network_kwargs()
    trees, last_state = [], None
    for s in range(N_SEEDS):
        r = run_agent_world(N=N_AGENTS, T_days=44, beta=0.15, sigma=0.45, gamma=0.18,
                            delta=0.002, nu=0.0002, global_seed=s, import_rate=3e-4,
                            population=pop, transmission_mode="hybrid", hybrid_weight=0.5,
                            network_kwargs=nk, beta_amp=0.45, beta_phase=105.0)
        trees.append(r["transmission_tree"])
        last_state = r["agents"]["state"]
    person = tree_metrics(trees, pop)
    entity = entity_metrics(pop, nk, last_state, seed=0)      # variant C
    return {"config": {"n_agents": N_AGENTS, "n_seeds": N_SEEDS}, "ablation": ablation,
            "enkf": enkf, "person": person, "entity": entity,
            "network_provenance": nk.get("provenance")}


def fig1_ablation(d: dict) -> None:
    V = d["ablation"]["variants"]
    E = d["ablation"].get("ensemble_AB", {})
    champ = d["enkf"]["champion_alone_forward_r2"]
    rows = [("A: mean-field", V["A"]["forward_r2"], _C["A"]),
            ("B: agent-to-agent", V["B"]["forward_r2"], _C["B"]),
            ("A+B ensemble", E.get("forward_r2", np.nan), _C["E"]),
            ("H: hybrid fusion", V["H"]["forward_r2"], _C["H"]),
            ("H + EnKF (real-time)", d["enkf"]["variant_plus_enkf_forward_r2"], _C["H"])]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ys = np.arange(len(rows))[::-1]
    for y, (lab, val, c) in zip(ys, rows):
        ax.barh(y, val, color=c, edgecolor="white", height=0.66)
        ax.text(val + 0.008, y, f"{val:.3f}", va="center", fontsize=10, fontweight="bold")
    ax.axvline(champ, ls="--", color=_C["champ"], lw=1.6)
    ax.text(champ - 0.012, 2.0, f"FusedEpi ceiling {champ:.3f}", color=_C["champ"],
            fontsize=9, rotation=90, va="center", ha="right")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=10)
    ax.set_xlabel("Forward $R^2$ vs held-out 2026 Seoul ILI (leak-free)")
    ax.set_xlim(0, max(champ + 0.08, 0.98))
    ax.set_title("ABM transmission-mechanism ablation: hybrid fusion + EnKF are best",
                 fontsize=11, fontweight="bold", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "abm_realism_ablation.png", dpi=200); plt.close(fig)


def fig2_enkf(d: dict) -> None:
    t = d["enkf"]["trajectories"]
    x = np.arange(len(t["real"]))
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(x, t["real"], "o-", color=_C["real"], lw=2.2, ms=5, label="Real ILI (held-out)")
    ax.plot(x, t["champion_forecast"], "s--", color=_C["champ"], lw=1.6, ms=4,
            label=f"FusedEpi forecast ($R^2$={d['enkf']['champion_alone_forward_r2']:.3f})")
    ax.plot(x, t["variant_alone"], "^-", color=_C["H"], lw=1.6, ms=4, alpha=0.85,
            label=f"Hybrid ABM ($R^2$={d['enkf']['variant_alone_forward_r2']:.3f})")
    ax.plot(x, t["variant_plus_enkf"], "d-", color=_C["E"], lw=2.0, ms=5,
            label=f"Hybrid + EnKF ($R^2$={d['enkf']['variant_plus_enkf_forward_r2']:.3f})")
    ax.set_xticks(x[::2])
    ax.set_xticklabels([t["forward_dates"][i][5:] for i in range(0, len(x), 2)],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Weekly ILI rate (per 1,000)")
    ax.set_xlabel("Forward week (2026)")
    ax.set_title("Real-time EnKF: weekly champion nowcasts correct the hybrid ABM state",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5, framealpha=0.9); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "abm_realtime_enkf_trajectory.png", dpi=200); plt.close(fig)


def fig3_personlike(d: dict) -> None:
    p, ent = d["person"], d["entity"]
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2))
    # (a) per-layer transmission share
    ls = p.get("layer_share", {})
    names = list(ls); vals = [ls[n] for n in names]
    axs[0].bar(names, vals, color=[_C["H"]] * len(names), edgecolor="white")
    for i, v in enumerate(vals):
        axs[0].text(i, v + 0.005, f"{v:.2f}", ha="center", fontsize=9)
    axs[0].set_title("Transmission share by contact layer", fontsize=10, fontweight="bold")
    axs[0].set_ylabel("share of agent-to-agent infections")
    # (b) occupation attack rate — real KOSIS occupation categories (code order from
    #     _load_industry_names). Agriculture/fishery is dropped: Seoul is a city with
    #     almost no farmers (a few dozen of the synthetic population), so its rate is
    #     noise rather than a structural ~0. High-contact occupations (above the mean
    #     attack rate) are highlighted to make the office/professional contrast legible
    #     regardless of the run's absolute attack level.
    oa = p.get("occupation_attack_rate", {})
    occ_lbl = {0: "manager/prof.", 1: "office", 2: "service/sales",
               3: "agriculture", 4: "technical", 5: "manual"}
    oa = {k: v for k, v in oa.items()}
    keys = [k for k in sorted(oa, key=lambda k: -oa[k]) if int(k) != 3]  # drop tiny agri.
    _hi = float(np.mean([oa[k] for k in keys])) if keys else 0.0
    axs[1].bar([occ_lbl.get(int(k), str(k)) for k in keys], [oa[k] for k in keys],
               color=[_C["champ"] if oa[k] >= _hi else _C["B"] for k in keys], edgecolor="white")
    for i, k in enumerate(keys):
        axs[1].text(i, oa[k] + 0.005, f"{oa[k]:.2f}", ha="center", fontsize=8)
    axs[1].set_title("Attack rate by occupation\n(high-contact jobs > office/professional)",
                     fontsize=10, fontweight="bold")
    axs[1].set_ylabel("share ever infected"); axs[1].tick_params(axis="x", rotation=30)
    # (c) per-entity (variant C) attack. The second bar is the SIZE-FAIR outbreak
    #     metric — share of units with within-unit spread (≥2 infected) — not the
    #     size-confounded "≥50% infected" (which reads ~0 for large classes and so
    #     hid that schools, the highest-mean-attack units, amplify spread the most).
    elabs = ["household", "workplace", "school"]
    mean_ar = [ent[e]["mean_attack_rate"] for e in elabs]
    spread = [ent[e].get("share_entities_with_within_spread",
                         ent[e].get("share_entities_majority_infected", 0.0)) for e in elabs]
    xx = np.arange(len(elabs)); w = 0.38
    axs[2].bar(xx - w / 2, mean_ar, w, label="mean attack rate", color=_C["A"])
    axs[2].bar(xx + w / 2, spread, w, label="share with within-unit spread (≥2)",
               color=_C["champ"])
    for i, (m, s) in enumerate(zip(mean_ar, spread)):
        axs[2].text(i - w / 2, m + 0.006, f"{m:.2f}", ha="center", fontsize=7)
        axs[2].text(i + w / 2, s + 0.006, f"{s:.2f}", ha="center", fontsize=7)
    axs[2].set_xticks(xx); axs[2].set_xticklabels(elabs)
    axs[2].set_title("Per-entity outbreak (variant C)", fontsize=10, fontweight="bold")
    axs[2].legend(fontsize=7.5, loc="upper left")
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
    _ss = p.get("mean_superspreaders_per_run_k_ge5", p.get("superspreaders_k_ge5"))
    fig.suptitle("Person-like outputs a mean-field model cannot produce "
                 f"(offspring k mean {p.get('offspring_k_mean')}, dispersion "
                 f"{p.get('offspring_k_dispersion')}; ~{_ss:g} superspreaders per run)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "abm_person_like_metrics.png", dpi=200); plt.close(fig)


def fig4_architecture(d: dict) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.8)); ax.axis("off")

    def box(x, y, w, h, text, c, fs=9):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=c, edgecolor="#333",
                                   lw=1.2, alpha=0.9, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                zorder=3, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", (x2, y2), (x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.6), zorder=1)
    box(0.02, 0.62, 0.24, 0.24, "Mean-field FoI\n(district prevalence)\n→ smooth aggregate", "#cfe0f3")
    box(0.02, 0.14, 0.24, 0.24, "Network FoI\n(household·work·school·\ncommunity edges)\n→ who-infects-whom", "#f3ddcf")
    box(0.36, 0.38, 0.24, 0.24, "HYBRID\nw·mean-field +\n(1−w)·network", "#d6ecd9", fs=10)
    box(0.68, 0.60, 0.30, 0.26, "Forward simulation\n+ person-like tree,\nsuperspreading, attack", "#e6dff2")
    box(0.68, 0.14, 0.30, 0.26, "EnKF real-time:\nweekly champion nowcast\n→ correct ABM state", "#f6d9dc")
    arrow(0.26, 0.72, 0.36, 0.55); arrow(0.26, 0.26, 0.36, 0.45)
    arrow(0.60, 0.55, 0.68, 0.66); arrow(0.60, 0.45, 0.68, 0.32)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Hybrid force-of-infection: mean-field ⊕ agent-network, coupled to the forecast",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "abm_hybrid_architecture.png", dpi=200); plt.close(fig)


def fig5_occupation_schedule_illustrative(d: dict) -> None:
    """Variant D — ILLUSTRATIVE occupation daily time-budget (NOT calibrated).

    The audit is explicit: no Seoul agent-resolution time-use survey exists, and a
    full schedule/sleep/environment layer is weakly identifiable from aggregate ILI
    and would degrade forward accuracy. So D is shown as a labelled conceptual
    sketch of how occupation could drive where-and-when contacts, alongside the
    occupation effect that IS already data-grounded (the differential attack rate).
    """
    blocks = ["sleep", "home", "commute", "work/school", "community"]
    bc = {"sleep": "#31356e", "home": "#6a8caf", "commute": "#e0a458",
          "work/school": "#c0553b", "community": "#5a9e6f"}
    # illustrative daily hours (sum 24) per occupation — a documented assumption
    sched = {
        "office worker": [7, 5, 2, 8, 2],
        "service/sales": [6, 4, 2, 9, 3],
        "student": [8, 5, 1, 7, 3],
        "manual/technical": [7, 4, 2, 9, 2],
        "retired/home": [8, 11, 0, 0, 5],
    }
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ys = np.arange(len(sched))[::-1]
    for y, (occ, hrs) in zip(ys, sched.items()):
        left = 0
        for b, h in zip(blocks, hrs):
            ax.barh(y, h, left=left, color=bc[b], edgecolor="white", height=0.62,
                    label=b if y == ys[0] else None)
            left += h
    ax.set_yticks(ys); ax.set_yticklabels(list(sched), fontsize=10)
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 4))
    ax.set_xlabel("hours of the day (illustrative)")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              frameon=False)
    ax.set_title("Variant D (ILLUSTRATIVE — not calibrated): occupation-driven daily "
                 "contact time-budget", fontsize=10.5, fontweight="bold", pad=10)
    ax.text(0.5, 1.11, "No Seoul agent-resolution time-use data exists; schedule/sleep is a "
            "documented sketch. The occupation EFFECT is already data-grounded (see attack-rate panel).",
            transform=ax.transAxes, ha="center", fontsize=8, color="#555", style="italic")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "abm_occupation_schedule_illustrative.png",
                                    dpi=200, bbox_inches="tight"); plt.close(fig)


def make_table(d: dict) -> None:
    import csv
    V = d["ablation"]["variants"]; E = d["ablation"].get("ensemble_AB", {})
    rows = [
        ["Variant", "Mechanism", "Forward R2", "RMSE", "Person-like metrics"],
        ["A", "mean-field (district FoI)", V["A"]["forward_r2"], V["A"]["forward_rmse"], "no"],
        ["B", "agent-to-agent (contact network)", V["B"]["forward_r2"], V["B"]["forward_rmse"], "yes"],
        ["A+B", "prediction ensemble", E.get("forward_r2"), E.get("forward_rmse"), "yes"],
        ["H", "hybrid fusion (mean-field ⊕ network)", V["H"]["forward_r2"], V["H"]["forward_rmse"], "yes"],
        ["H+EnKF", "hybrid + real-time assimilation", d["enkf"]["variant_plus_enkf_forward_r2"],
         d["enkf"]["variant_plus_enkf_forward_rmse"], "yes"],
        ["(ceiling)", "FusedEpi forecast alone", d["enkf"]["champion_alone_forward_r2"], "-", "-"],
    ]
    with (RESULTS / "abm_realism_ablation_table.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    fig, ax = plt.subplots(figsize=(10, 2.6)); ax.axis("off")
    tbl = ax.table(cellText=[[str(c) for c in r] for r in rows[1:]],
                   colLabels=rows[0], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.5)
    for j in range(len(rows[0])):
        tbl[0, j].set_facecolor("#4C72B0"); tbl[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Table. ABM realism ablation vs the real held-out 2026 forward window (leak-free)",
                 fontsize=10, fontweight="bold", pad=14)
    fig.tight_layout(); fig.savefig(OUT / "abm_realism_ablation_table.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    d = _gen_data()
    (RESULTS / "abm_realism_figure_data.json").write_text(
        json.dumps(d, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    fig1_ablation(d); fig2_enkf(d); fig3_personlike(d); fig4_architecture(d); make_table(d)
    print("figures →", OUT)
    for f in ("abm_realism_ablation", "abm_realtime_enkf_trajectory", "abm_person_like_metrics",
              "abm_hybrid_architecture", "abm_realism_ablation_table"):
        print("  ", f + ".png")


if __name__ == "__main__":
    main()
