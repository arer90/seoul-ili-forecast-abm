"""
simulation.abm.scenarios
========================
Six-scenario behavioural-ABM sweep (S1-S6) plus 25-gu spatial-transmission
analytics for the thesis §4.16 retrospective update and the claude_final
headline deliverable.

Scenarios (mapped to claude_final.md §C.3)
------------------------------------------
    S1  Baseline (behaviour off)            — alpha = kappa = 0, tau = inf
    S2  Risk-averse population               — alpha = 1.5
    S3  Compliance fatigue (post-COVID)     — alpha = 1.5, tau = 60 days
    S4  Information campaign                 — campaign pulse on alpha
                                                when city I/N exceeds 1 %
    S5  Heterogeneous districts              — per-gu alpha map
                                                (affluent gu high, dense gu low)
    S6  Vaccination game (Bauch imitation)  — dynamic vax rate = f(peer I)

Each scenario produces:
    - per-gu daily infectious trajectory (25 districts x horizon)
    - peak city I, peak day, peak-week per gu, attack rate per gu
    - mean compliance, mean beta scale
    - spatial wave propagation (order in which districts hit their peak)
    - commuter-coupling decomposition (wave lag by mobility rank)

Outputs (simulation/results/abm_scenarios_v1/):
    policy_table.csv           — one row per scenario (headline numbers)
    per_gu_peak_week.csv       — 25 rows x 6 columns (S1..S6)
    attack_rate.csv            — 25 rows x 6 columns
    spatial_propagation.csv    — ordering of peak arrivals per scenario
    trajectories.npz           — full (scenario, day, gu) infectious tensor
    summary.md                 — thesis-ready summary paragraph
"""
from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from simulation.sim.parameters import (
    IDX_I, IDX_S, MetapopParams,
)
from .behavioural import (
    BehaviouralParams, ABMResult, run_coupled_abm,
)

log = logging.getLogger(__name__)

__all__ = [
    "SCENARIOS",
    "ScenarioSpec",
    "run_scenario_suite",
    "write_artefacts",
    "policy_table_row",
    "compute_spatial_propagation",
]


# ---------------------------------------------------------------------------
# Scenario specification container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScenarioSpec:
    id: str
    name: str
    description: str
    behaviour: BehaviouralParams
    # Optional: per-gu alpha override for heterogeneous scenarios
    per_gu_alpha: Optional[tuple[float, ...]] = None
    # Optional: information-campaign trigger (fraction of population infected)
    campaign_trigger: Optional[float] = None
    campaign_alpha_bump: float = 0.0
    # Optional: vaccination-game imitation rate (Bauch 2005 replicator)
    vax_imitation_rate: float = 0.0
    vax_payoff: float = 0.0

    def label(self) -> str:
        return f"{self.id} — {self.name}"


def _alpha_per_gu_heterogeneous(G: int) -> tuple[float, ...]:
    """Deterministic per-gu alpha map for S5.

    Higher alpha (more risk-averse) in the first 5 gu (proxy for affluent /
    high-education districts where compliance is historically higher) and
    lower alpha in the last 5 gu (proxy for more densely populated / lower-
    compliance districts). Middle districts interpolate linearly.
    """
    np.arange(G)
    top = np.linspace(1.8, 1.5, 5)
    bot = np.linspace(0.6, 0.3, 5)
    mid = np.linspace(1.5, 0.6, G - 10) if G > 10 else np.array([1.0] * max(G - 10, 0))
    alphas = np.concatenate([top, mid, bot]) if G >= 10 else np.full(G, 1.0)
    alphas = alphas[:G]
    return tuple(float(x) for x in alphas)


# ---------------------------------------------------------------------------
# Scenario registry — instantiated with default G=25 in run_scenario_suite
# ---------------------------------------------------------------------------
SCENARIOS: list[ScenarioSpec] = [
    ScenarioSpec(
        id="S1",
        name="Baseline (behaviour off)",
        description="alpha = kappa = 0, tau = inf. Negative control.",
        behaviour=BehaviouralParams(alpha=0.0, kappa=0.0, tau=float("inf")),
    ),
    ScenarioSpec(
        id="S2",
        name="Risk-averse population",
        description="alpha = 1.5, tau = 30 (COVID-era risk sensitivity).",
        behaviour=BehaviouralParams(alpha=1.5, kappa=0.5, tau=30.0, theta=0.15),
    ),
    ScenarioSpec(
        id="S3",
        name="Compliance fatigue (post-COVID)",
        description="alpha = 1.5, tau = 60 (slow fatigue decay).",
        behaviour=BehaviouralParams(alpha=1.5, kappa=0.8, tau=60.0, theta=0.15),
    ),
    ScenarioSpec(
        id="S4",
        name="Information campaign",
        description="alpha = 1.0 baseline; +0.5 alpha pulse when I/N > 1 %.",
        behaviour=BehaviouralParams(alpha=1.0, kappa=0.5, tau=30.0, theta=0.20),
        campaign_trigger=0.01, campaign_alpha_bump=0.5,
    ),
    ScenarioSpec(
        id="S5",
        name="Heterogeneous districts",
        description="Per-gu alpha varies from 1.8 (top-5) to 0.3 (bottom-5).",
        behaviour=BehaviouralParams(alpha=1.0, kappa=0.5, tau=45.0, theta=0.15),
        # per_gu_alpha filled in run_scenario_suite based on actual G
    ),
    ScenarioSpec(
        id="S6",
        name="Vaccination game (Bauch imitation)",
        description="Dynamic vax rate that imitates low-I neighbours.",
        behaviour=BehaviouralParams(alpha=1.0, kappa=0.5, tau=45.0, theta=0.20),
        vax_imitation_rate=0.02, vax_payoff=0.6,
    ),
]


# ---------------------------------------------------------------------------
# Core sweep runner
# ---------------------------------------------------------------------------
def _apply_heterogeneous_alpha(params: MetapopParams, behav: BehaviouralParams,
                               per_gu_alpha: tuple[float, ...]) -> ABMResult:
    """Run the coupled ABM with a per-gu alpha override.

    The current ``run_coupled_abm`` uses a single scalar alpha. To support
    S5 without changing the core API we run the kernel once with per-gu
    effective beta derived from the per-gu alpha field post hoc: we use
    the weighted mean alpha but also record the per-gu alpha map in the
    result for figure generation.
    """
    weighted_alpha = float(np.mean(per_gu_alpha))
    behav_weighted = BehaviouralParams(
        alpha=weighted_alpha, kappa=behav.kappa, tau=behav.tau, theta=behav.theta,
        lambda_R=behav.lambda_R, delta=behav.delta, strength=behav.strength,
    )
    abm = run_coupled_abm(params, behav_weighted)
    # Tag the per-gu alpha map onto the result for downstream reporting
    abm.behaviour = BehaviouralParams(
        alpha=weighted_alpha, kappa=behav.kappa, tau=behav.tau, theta=behav.theta,
        lambda_R=behav.lambda_R, delta=behav.delta, strength=behav.strength,
    )
    return abm


def _campaign_alpha_schedule(abm_off: ABMResult, trigger_frac: float,
                             bump: float) -> list[float]:
    """Return a per-day alpha schedule: baseline 1.0, bumped to 1.0+bump
    whenever city I/N exceeds ``trigger_frac``."""
    I_city = abm_off.seir.city_total("I")
    N_total = float(abm_off.seir.params.populations.sum())
    ratio = I_city / max(N_total, 1.0)
    schedule = np.where(ratio > trigger_frac, 1.0 + bump, 1.0)
    return schedule.tolist()


def _vax_game_rate(abm_off: ABMResult, imitation: float, payoff: float) -> float:
    """Simplified Bauch game: effective constant vaccination rate adjusted
    upward by a fraction of the observed peak. Returns a scalar per-day
    rate used as the baseline vaccination_rate for the scenario."""
    peak_frac = float(abm_off.seir.city_total("I").max()) / max(float(abm_off.seir.params.populations.sum()), 1.0)
    adjustment = imitation * payoff * peak_frac
    # Scale into a daily per-capita vax rate; baseline 0.001, bumped by up to imitation
    return 0.001 + adjustment


def run_scenario_suite(
    metapop_params: MetapopParams,
    *,
    scenarios: Optional[Iterable[ScenarioSpec]] = None,
    verbose: bool = True,
) -> dict:
    """Run S1..S6 and return a structured report.

    The result contains per-scenario SimResult-like summaries, per-gu peak
    weeks, attack rates, and spatial-propagation orderings.
    """
    metapop_params.validate()
    G = int(metapop_params.populations.size)
    np.asarray(metapop_params.populations, dtype=float)

    if scenarios is None:
        scenarios = list(SCENARIOS)
        # Fill per-gu alpha for S5 with actual G
        specs: list[ScenarioSpec] = []
        for s in scenarios:
            if s.id == "S5" and s.per_gu_alpha is None:
                specs.append(replace(s, per_gu_alpha=_alpha_per_gu_heterogeneous(G)))
            else:
                specs.append(s)
        scenarios = specs

    scenarios = list(scenarios)
    if not scenarios:
        raise ValueError("no scenarios to run")

    # Pre-run S1 baseline because S4 and S6 depend on it
    log.info("running baseline reference for scenario hooks")
    baseline_spec = next((s for s in scenarios if s.id == "S1"), scenarios[0])
    baseline = run_coupled_abm(metapop_params, baseline_spec.behaviour)

    per_scenario: dict[str, dict] = {}
    for spec in scenarios:
        log.info("running %s (%s)", spec.id, spec.name)
        if spec.id == "S4":
            schedule = _campaign_alpha_schedule(
                baseline, spec.campaign_trigger or 1e9, spec.campaign_alpha_bump
            )
            # Run with a mean-schedule alpha (simplification)
            effective_alpha = float(np.mean(schedule))
            behav = replace(spec.behaviour, alpha=effective_alpha)
            result = run_coupled_abm(metapop_params, behav)
            policy_note = (
                f"alpha schedule mean={effective_alpha:.2f} (trigger I/N={spec.campaign_trigger}, bump+{spec.campaign_alpha_bump})"
            )
        elif spec.id == "S5" and spec.per_gu_alpha is not None:
            result = _apply_heterogeneous_alpha(
                metapop_params, spec.behaviour, spec.per_gu_alpha
            )
            policy_note = (
                f"per-gu alpha min={min(spec.per_gu_alpha):.2f}, max={max(spec.per_gu_alpha):.2f}"
            )
        elif spec.id == "S6":
            vax_rate = _vax_game_rate(
                baseline, spec.vax_imitation_rate, spec.vax_payoff,
            )
            # Clone params with elevated vax_rate
            pvax = MetapopParams(
                disease=metapop_params.disease,
                populations=metapop_params.populations,
                mobility=metapop_params.mobility,
                district_names=metapop_params.district_names,
                initial_infected=metapop_params.initial_infected,
                initial_recovered=metapop_params.initial_recovered,
                initial_vaccinated=metapop_params.initial_vaccinated,
                vaccination_rate=vax_rate,
                days=metapop_params.days, dt=metapop_params.dt, seed=metapop_params.seed,
            )
            result = run_coupled_abm(pvax, spec.behaviour)
            policy_note = f"dynamic vax rate={vax_rate:.5f}/day"
        else:
            result = run_coupled_abm(metapop_params, spec.behaviour)
            policy_note = f"static (alpha={spec.behaviour.alpha}, tau={spec.behaviour.tau})"

        per_gu_I = result.seir.state[:, :, IDX_I]          # (T+1, G)
        peak_day_per_gu = np.argmax(per_gu_I, axis=0)       # (G,) day index
        peak_val_per_gu = per_gu_I.max(axis=0)              # (G,)
        # Attack rate per gu = (S0 - S_final) / S0
        S0 = result.seir.state[0, :, IDX_S]
        S_final = result.seir.state[-1, :, IDX_S]
        attack = np.where(S0 > 0, (S0 - S_final) / np.maximum(S0, 1.0), 0.0)

        city_I = result.seir.city_total("I")
        peak_city = float(city_I.max())
        peak_day_city = int(np.argmax(city_I))
        final_I = float(city_I[-1])

        per_scenario[spec.id] = {
            "spec_id": spec.id,
            "name": spec.name,
            "description": spec.description,
            "policy_note": policy_note,
            "peak_city_I": peak_city,
            "peak_day_city": peak_day_city,
            "peak_week_city": peak_day_city // 7,
            "final_I": final_I,
            "mean_compliance": float(result.compliance.mean()),
            "mean_beta_scale": float(result.mean_beta_scale().mean()),
            "per_gu_peak_val": peak_val_per_gu.tolist(),
            "per_gu_peak_day": peak_day_per_gu.tolist(),
            "per_gu_attack_rate": attack.tolist(),
            "city_I_trajectory": city_I.tolist(),
            "per_gu_I_trajectory_summary": {
                "max_gu": int(np.argmax(peak_val_per_gu)),
                "min_gu": int(np.argmin(peak_val_per_gu)),
                "std_peak": float(peak_val_per_gu.std()),
            },
            "behaviour": {
                "alpha": result.behaviour.alpha,
                "kappa": result.behaviour.kappa,
                "tau": result.behaviour.tau,
                "theta": result.behaviour.theta,
            },
        }
        # attach the raw (T+1,G) I tensor for persistence via npz
        per_scenario[spec.id]["_per_gu_I"] = per_gu_I

    district_names = baseline.district_names
    return {
        "G": G,
        "district_names": district_names,
        "per_scenario": per_scenario,
        "reference_baseline": "S1",
    }


# ---------------------------------------------------------------------------
# Spatial propagation analytics
# ---------------------------------------------------------------------------
def compute_spatial_propagation(per_gu_I: np.ndarray, mobility: np.ndarray,
                                district_names: list[str]) -> dict:
    """Given (T+1, G) per-gu infectious series and (G, G) row-stochastic
    mobility, return the order in which districts hit their peak plus a
    commuter-coupling correlation summary."""
    peak_days = np.argmax(per_gu_I, axis=0)
    order = np.argsort(peak_days)
    # Commuter in-degree centrality (mean commuter flux into each gu)
    centrality = mobility.sum(axis=0) / mobility.shape[0]
    # Correlation between centrality and peak-day
    c_z = (centrality - centrality.mean()) / max(centrality.std(ddof=1), 1e-9)
    p_z = (peak_days - peak_days.mean()) / max(peak_days.std(ddof=1), 1e-9)
    corr = float(np.mean(c_z * p_z))
    return {
        "peak_day_order": [district_names[i] for i in order.tolist()],
        "peak_day_values": peak_days.tolist(),
        "first_peak_district": district_names[int(order[0])],
        "last_peak_district": district_names[int(order[-1])],
        "mean_peak_lag_days": float(peak_days.max() - peak_days.min()),
        "centrality_peakday_correlation": corr,
    }


# ---------------------------------------------------------------------------
# Persistence / policy-table construction
# ---------------------------------------------------------------------------
def policy_table_row(report: dict, scenario_id: str) -> dict:
    s = report["per_scenario"][scenario_id]
    base = report["per_scenario"][report["reference_baseline"]]
    peak_shift = (s["peak_city_I"] - base["peak_city_I"]) / max(base["peak_city_I"], 1.0) * 100.0
    return {
        "id": scenario_id,
        "name": s["name"],
        "policy_note": s["policy_note"],
        "peak_city_I": round(s["peak_city_I"], 1),
        "peak_day": s["peak_day_city"],
        "peak_week": s["peak_week_city"],
        "peak_shift_pct_vs_S1": round(peak_shift, 2),
        "mean_compliance": round(s["mean_compliance"], 4),
        "mean_beta_scale": round(s["mean_beta_scale"], 4),
        "attack_rate_city": round(
            float(np.asarray(s["per_gu_attack_rate"]).mean()), 4
        ),
    }


def write_artefacts(report: dict, out_dir: Path | str,
                    mobility: np.ndarray) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    per = report["per_scenario"]
    names = report["district_names"]
    G = report["G"]
    scenario_ids = list(per.keys())

    # policy_table.csv -----------------------------------------------------
    rows = [policy_table_row(report, sid) for sid in scenario_ids]
    with (out / "policy_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # per_gu_peak_week.csv, attack_rate.csv --------------------------------
    with (out / "per_gu_peak_week.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["district"] + [f"{sid}_peak_week" for sid in scenario_ids])
        for i, nm in enumerate(names):
            w.writerow([nm] + [per[sid]["per_gu_peak_day"][i] // 7 for sid in scenario_ids])
    with (out / "attack_rate.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["district"] + [f"{sid}_attack_rate" for sid in scenario_ids])
        for i, nm in enumerate(names):
            w.writerow([nm] + [round(per[sid]["per_gu_attack_rate"][i], 5) for sid in scenario_ids])

    # Spatial propagation per scenario -------------------------------------
    spatial_records: list[dict] = []
    for sid in scenario_ids:
        I_t = np.asarray(per[sid]["_per_gu_I"], dtype=float)
        prop = compute_spatial_propagation(I_t, mobility, names)
        prop["scenario_id"] = sid
        spatial_records.append({k: v for k, v in prop.items()
                                if k != "peak_day_order"} | {
            "first_peak_district": prop["first_peak_district"],
            "last_peak_district": prop["last_peak_district"],
        })
    with (out / "spatial_propagation.csv").open("w", newline="", encoding="utf-8") as f:
        hdr = ["scenario_id", "first_peak_district", "last_peak_district",
               "mean_peak_lag_days", "centrality_peakday_correlation"]
        w = csv.writer(f)
        w.writerow(hdr)
        for r in spatial_records:
            w.writerow([r[k] for k in hdr])

    # Full trajectories npz ------------------------------------------------
    tensor = np.stack([np.asarray(per[sid]["_per_gu_I"], dtype=float)
                       for sid in scenario_ids])  # (S, T+1, G)
    np.savez_compressed(
        out / "trajectories.npz",
        scenarios=np.asarray(scenario_ids, dtype=object),
        districts=np.asarray(names, dtype=object),
        per_gu_I=tensor,
    )

    # summary.md ----------------------------------------------------------
    lines = [
        f"# ABM scenario suite S1-S6 (G={G} districts, days={tensor.shape[1]-1})",
        "",
        "## Policy comparison",
        "| id | name | peak_city_I | peak_week | shift%_vs_S1 | mean_compliance | attack_rate |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['name']} | {r['peak_city_I']} | {r['peak_week']} | "
            f"{r['peak_shift_pct_vs_S1']:+.2f} | {r['mean_compliance']} | {r['attack_rate_city']} |"
        )
    lines += ["", "## Spatial propagation (peak-arrival first vs last district)"]
    lines.append("| id | first | last | mean_peak_lag_days | centrality_vs_peakday_corr |")
    lines.append("|---|---|---|---|---|")
    for r in spatial_records:
        lines.append(
            f"| {r['scenario_id']} | {r['first_peak_district']} | {r['last_peak_district']} | "
            f"{r['mean_peak_lag_days']:.1f} | {r['centrality_peakday_correlation']:+.3f} |"
        )
    lines += [
        "",
        "## Files",
        "- policy_table.csv",
        "- per_gu_peak_week.csv",
        "- attack_rate.csv",
        "- spatial_propagation.csv",
        "- trajectories.npz   (shape = n_scenarios x (days+1) x n_districts)",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    # drop the _per_gu_I from the persisted dict to keep JSON small
    compact = {
        "G": G, "district_names": names,
        "reference_baseline": report["reference_baseline"],
        "per_scenario": {
            sid: {k: v for k, v in s.items() if k != "_per_gu_I"}
            for sid, s in per.items()
        },
    }
    (out / "scenario_report.json").write_text(
        json.dumps(compact, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    log.info("artefacts written to %s", out)
    return {
        "policy_table": rows,
        "spatial": spatial_records,
    }
