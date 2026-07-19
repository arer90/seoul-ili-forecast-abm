"""Seed-ensemble CI + paired test for vaccination counterfactual (M.4 fix).

The policy headline — high-contact targeting averts ~1.86 infections/dose vs
uniform ~1.42 vs elderly ~1.13 — was reported as bare POINT estimates with no
CI and no test, while a sibling ablation (threshold-dispersion) reported a proper
bootstrap half-width. An external reviewer flagged this as a §M.4 violation
(policy headline without uncertainty). This script honesty-fixes it:

  1. Re-run the counterfactual over many seeds (read-only ABM; no retrain, no DB
     write). For EACH seed, build the population once and run all four strategy
     arms on that SAME population (paired by seed).
  2. averted-per-dose is computed PER SEED:  (none[seed] - strategy[seed]) / doses.
     The per-seed values form a distribution -> 95% CI per strategy (t-interval).
  3. Paired test target_high_contact vs uniform on the per-seed averted/dose
     differences (paired t-test + Wilcoxon signed-rank). Same-seed populations
     make this a clean paired design.

Outputs (no DB write, no model retrain):
  simulation/results/abm_counterfactual_ci/strategy_ci.csv
  simulation/results/abm_counterfactual_ci/test.json
  simulation/results/figures/counterfactual_ci.png

Reuses the exact production machinery from simulation.abm.counterfactual so the
numbers reproduce the headline (N=12500, budget_frac=0.10) — only the
aggregation differs (per-seed retained instead of mean-collapsed).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from simulation.abm.counterfactual import _outcomes, _vaccination_mask
from simulation.abm.agent_kernel import run_agent_world
from simulation.abm.epi_proof import (
    DEFAULT_DISEASE,
    BEHAVIOUR_OFF,
    DB_PATH,
    _make_population,
    _load_ili_seasons,
)

STRATEGIES = ("none", "uniform", "target_elderly", "target_high_contact")
TARGETED = ("uniform", "target_elderly", "target_high_contact")
_POP_KIND = {"heterogeneous": "rich_movement", "homogeneous": "homogeneous_static"}

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "abm_counterfactual_ci"
FIG_PATH = (
    Path(__file__).resolve().parents[1] / "results" / "figures" / "counterfactual_ci.png"
)
T_DAYS = 364

# The thesis quotes THIS configuration (both populations, 30 paired seeds, 12,500 agents,
# 10% dose budget). Anything else is a robustness sweep and must never land on top of it.
HEADLINE = {
    "n_seeds": 30,
    "n_agents": 12_500,
    "budget_frac": 0.10,
    "populations": ("heterogeneous", "homogeneous"),
}


def _paths(n_seeds: int, n_agents: int, budget_frac: float,
           populations: tuple[str, ...]) -> tuple[Path, Path]:
    """Where this run's artifacts go: the canonical dir only for the headline config.

    ``OUT_DIR`` used to be a constant, so ``run(n_seeds=20, n_agents=8000,
    budget_frac=0.2)`` — a legitimate robustness sweep — silently overwrote the headline
    ``test.json`` the thesis cites. The overwrite was invisible: the file still parsed,
    still had every key, and simply held different numbers under a different metadata
    block. Sweeps now get their own directory, so the headline artifact can only be
    replaced by re-running the headline.

    Returns:
        (out_dir, fig_path) — both under results/, neither created here.
    """
    cfg = {
        "n_seeds": n_seeds, "n_agents": n_agents,
        "budget_frac": budget_frac, "populations": tuple(populations),
    }
    if cfg == HEADLINE:
        return OUT_DIR, FIG_PATH
    tag = f"sweep_n{n_agents}_s{n_seeds}_b{budget_frac:g}_{'-'.join(populations)}"
    return OUT_DIR / tag, FIG_PATH.with_name(f"counterfactual_ci_{tag}.png")


def _run_one(pop, *, strategy: str, budget: int, seed: int, n_agents: int,
             disease: dict, behaviour: dict) -> dict[str, float]:
    """One arm, one seed, on a pre-built population (paired design)."""
    mask = _vaccination_mask(pop, strategy, budget, int(seed))
    result = run_agent_world(
        N=n_agents, T_days=T_DAYS,
        beta=float(disease["beta"]), sigma=float(disease["sigma"]),
        gamma=float(disease["gamma"]), delta=float(disease["delta"]), nu=0.0,
        population=pop, global_seed=int(seed),
        theta_mean=float(behaviour["theta"]), alpha_mean=float(behaviour["alpha"]),
        kappa_mean=float(behaviour["kappa"]), tau_mean=float(behaviour["tau"]),
        beta_amp=float(disease.get("beta_amp", 0.0)),
        beta_phase=float(disease.get("beta_phase", 0.0)),
        import_rate=float(disease.get("import_rate", 0.0)),
        initial_vaccinated=(mask if strategy != "none" else None),
    )
    o = _outcomes(result, n_agents)
    o["doses"] = int(mask.sum()) if strategy != "none" else 0
    return o


def _t_ci(vals: np.ndarray) -> tuple[float, float, float]:
    """mean, lo95, hi95 via Student-t (paired/one-sample interval)."""
    m = float(vals.mean())
    if vals.size <= 1:
        return m, m, m
    half = float(stats.t.ppf(0.975, df=vals.size - 1) * stats.sem(vals))
    return m, m - half, m + half


def run(*, n_seeds: int = 30, n_agents: int = 12_500, budget_frac: float = 0.10,
        populations=("heterogeneous", "homogeneous"), year: int | None = None) -> dict:
    out_dir, fig_path = _paths(n_seeds, n_agents, budget_frac, tuple(populations))
    disease = dict(DEFAULT_DISEASE)
    behaviour = dict(BEHAVIOUR_OFF)
    budget = int(round(n_agents * budget_frac))
    if year is None:
        seasons = _load_ili_seasons(Path(DB_PATH))
        year = int(max(s.season for s in seasons))

    # per_seed[pop][strategy] = list over seeds of {infections, deaths, ...}
    per_seed: dict[str, dict[str, list[dict[str, float]]]] = {
        p: {s: [] for s in STRATEGIES} for p in populations
    }
    for pop_label in populations:
        for seed in range(n_seeds):
            # ONE population per (pop, seed); all arms share it -> paired.
            pop = _make_population(_POP_KIND[pop_label], N=n_agents,
                                   seed=int(seed), year=int(year))
            for strategy in STRATEGIES:
                per_seed[pop_label][strategy].append(
                    _run_one(pop, strategy=strategy, budget=budget, seed=seed,
                             n_agents=n_agents, disease=disease, behaviour=behaviour)
                )

    # ---- per-seed averted-per-dose distributions ----
    # averted/dose[seed] = (none[seed].infections - strat[seed].infections)/doses
    rows: list[dict[str, Any]] = []
    averted_dist: dict[str, dict[str, np.ndarray]] = {}
    for pop_label in populations:
        base_inf = np.array(
            [r["infections"] for r in per_seed[pop_label]["none"]], dtype=np.float64
        )
        base_dth = np.array(
            [r["deaths"] for r in per_seed[pop_label]["none"]], dtype=np.float64
        )
        averted_dist[pop_label] = {}
        for st in TARGETED:
            inf = np.array(
                [r["infections"] for r in per_seed[pop_label][st]], dtype=np.float64
            )
            dth = np.array(
                [r["deaths"] for r in per_seed[pop_label][st]], dtype=np.float64
            )
            doses = np.array(
                [r["doses"] for r in per_seed[pop_label][st]], dtype=np.float64
            )
            doses = np.where(doses <= 0, 1.0, doses)
            inf_apd = (base_inf - inf) / doses           # per-seed averted/dose
            dth_apd = (base_dth - dth) / doses
            averted_dist[pop_label][st] = inf_apd
            m, lo, hi = _t_ci(inf_apd)
            dm, dlo, dhi = _t_ci(dth_apd)
            rows.append({
                "population": pop_label,
                "strategy": st,
                "n_seed": int(inf_apd.size),
                "inf_averted_per_dose_mean": m,
                "inf_averted_per_dose_ci_lo": lo,
                "inf_averted_per_dose_ci_hi": hi,
                "inf_averted_per_dose_sd": float(inf_apd.std(ddof=1)),
                "deaths_averted_per_dose_mean": dm,
                "deaths_averted_per_dose_ci_lo": dlo,
                "deaths_averted_per_dose_ci_hi": dhi,
            })

    # ---- paired test: high_contact vs uniform (heterogeneous headline) ----
    tests: dict[str, Any] = {}
    for pop_label in populations:
        hc = averted_dist[pop_label]["target_high_contact"]
        un = averted_dist[pop_label]["uniform"]
        diff = hc - un                                   # paired, per-seed
        # paired t-test on the per-seed differences
        t_stat, t_p = stats.ttest_rel(hc, un)
        # Wilcoxon signed-rank (non-parametric backup); guard all-zero diffs
        try:
            w_stat, w_p = stats.wilcoxon(hc, un)
            w_stat, w_p = float(w_stat), float(w_p)
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
        dm, dlo, dhi = _t_ci(diff)
        # CI-overlap honesty check between the two strategy CIs
        hc_m, hc_lo, hc_hi = _t_ci(hc)
        un_m, un_lo, un_hi = _t_ci(un)
        ci_overlap = not (hc_lo > un_hi or un_lo > hc_hi)
        cohen_dz = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else float("nan")
        tests[pop_label] = {
            "comparison": "target_high_contact vs uniform (infections averted/dose)",
            "n_seed": int(diff.size),
            "high_contact_mean": hc_m,
            "high_contact_ci95": [hc_lo, hc_hi],
            "uniform_mean": un_m,
            "uniform_ci95": [un_lo, un_hi],
            "paired_diff_mean": dm,
            "paired_diff_ci95": [dlo, dhi],
            "paired_ttest_t": float(t_stat),
            "paired_ttest_p": float(t_p),
            "wilcoxon_stat": w_stat,
            "wilcoxon_p": w_p,
            "cohen_dz": cohen_dz,
            "strategy_ci_overlap": bool(ci_overlap),
            "significant_at_0.05": bool(t_p < 0.05),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "strategy_ci.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    test_path = out_dir / "test.json"
    with test_path.open("w", encoding="utf-8") as fh:
        json.dump({
            "metadata": {
                "n_seeds": int(n_seeds), "n_agents": int(n_agents),
                "budget": int(budget), "budget_frac": float(budget_frac),
                "year": int(year), "design": "paired-by-seed (same population per seed across arms)",
                "populations": list(populations),
                "is_headline": out_dir == OUT_DIR,
                "disease": disease, "behaviour": behaviour,
                "note": "read-only ABM; no model retrain; no sqlite write",
            },
            "tests": tests,
        }, fh, indent=2, sort_keys=True)

    _make_figure(rows, tests, populations, fig_path)
    return {"rows": rows, "tests": tests,
            "csv": str(csv_path), "test_json": str(test_path), "fig": str(fig_path)}


def _make_figure(rows, tests, populations, fig_path: Path = FIG_PATH) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_pop = {p: [r for r in rows if r["population"] == p] for p in populations}
    npop = len(populations)
    fig, axes = plt.subplots(1, npop, figsize=(6.2 * npop, 5.0), sharey=True)
    if npop == 1:
        axes = [axes]
    colors = {"uniform": "#4C72B0", "target_elderly": "#55A868",
              "target_high_contact": "#C44E52"}
    labels = {"uniform": "uniform", "target_elderly": "elderly",
              "target_high_contact": "high-contact"}

    for ax, pop_label in zip(axes, populations):
        prows = sorted(by_pop[pop_label], key=lambda r: TARGETED.index(r["strategy"]))
        xs = np.arange(len(prows))
        means = [r["inf_averted_per_dose_mean"] for r in prows]
        los = [r["inf_averted_per_dose_mean"] - r["inf_averted_per_dose_ci_lo"] for r in prows]
        his = [r["inf_averted_per_dose_ci_hi"] - r["inf_averted_per_dose_mean"] for r in prows]
        cols = [colors[r["strategy"]] for r in prows]
        ax.bar(xs, means, color=cols, alpha=0.85, width=0.6, zorder=2)
        ax.errorbar(xs, means, yerr=[los, his], fmt="none", ecolor="black",
                    elinewidth=1.6, capsize=6, capthick=1.6, zorder=3)
        for x, m, r in zip(xs, means, prows):
            ax.text(x, r["inf_averted_per_dose_ci_hi"] + 0.03, f"{m:.2f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels([labels[r["strategy"]] for r in prows], fontsize=11)
        t = tests[pop_label]
        sig = "p={:.3g}{}".format(
            t["paired_ttest_p"],
            " *" if t["significant_at_0.05"] else " (n.s.)",
        )
        ovl = "CIs overlap" if t["strategy_ci_overlap"] else "CIs disjoint"
        ax.set_title(
            f"{pop_label}\nhigh-contact vs uniform: {sig}; {ovl}",
            fontsize=11,
        )
        ax.axhline(0, color="grey", lw=0.8)
        ax.grid(axis="y", alpha=0.3, zorder=0)

    axes[0].set_ylabel("infections averted per dose\n(per-seed mean ± 95% CI)", fontsize=11)
    fig.suptitle(
        f"Vaccination counterfactual: averted-per-dose with seed-ensemble 95% CI "
        f"(n_seed={tests[populations[0]]['n_seed']}, paired by seed)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=30)
    ap.add_argument("--n-agents", type=int, default=12_500)
    ap.add_argument("--budget-frac", type=float, default=0.10)
    args = ap.parse_args()
    out = run(n_seeds=args.n_seeds, n_agents=args.n_agents, budget_frac=args.budget_frac)
    print(json.dumps(out["tests"], indent=2))
    print("\nCSV :", out["csv"])
    print("JSON:", out["test_json"])
    print("FIG :", out["fig"])
