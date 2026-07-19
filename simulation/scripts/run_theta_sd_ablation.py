"""theta_sd ablation: does household compliance-threshold heterogeneity change the ABM?

ABM enhancement #4. This script quantifies, head-on, whether the per-agent
compliance-threshold heterogeneity (``theta_sd``) that the attribute ABM hardcodes
to 0.15 actually changes the fit / WIS / trajectory against real Seoul ILI, or
whether heterogeneity is dormant under observation (only finite-N seed noise).

Design (no live-code edits; all dynamics imported from the unmodified
``simulation.abm.epi_proof`` and ``simulation.abm.agent_kernel``):

  1. Reuse ``epi_proof`` to load the real ILI seasons and pick the same
     calibration / evaluation split it uses by default, then calibrate the
     seasonal forcing (disease) and the behaviour grid ONCE. The calibrated
     disease and behaviour (alpha, kappa, tau, theta_mean) are then FROZEN.
  2. Sweep ``theta_sd in {0.0, 0.05, 0.10, 0.15, 0.20}`` with everything else
     held fixed, running K seeds per point through ``run_agent_world`` directly
     (``epi_proof._simulate_one`` hardcodes theta_sd=0.15, so the kernel is
     called here instead). theta_sd=0.0 is the mean-field reference; every agent
     gets the identical threshold ``theta_mean``.
  3. For each theta_sd, measure against real eval-season ILI: held-out WIS, fit
     R^2, peak size + timing, between-replicate trajectory spread, and the
     per-agent compliance variance the heterogeneity actually induces. All are
     reported as differences from the theta_sd=0 (mean-field) row so a reader can
     see immediately whether heterogeneity is material or noise.

Determinism: the seed set is ``range(K)``; ``run_agent_world`` keys all random
streams off ``global_seed``; identical inputs reproduce identical curves. The
only external read is the ILI series, loaded via ``read_only_connect`` (lock-free,
never writes).

Run:
    .venv/bin/python -m simulation.scripts.run_theta_sd_ablation
    .venv/bin/python -m simulation.scripts.run_theta_sd_ablation --k 12 --n-agents 30000

Output: ``simulation/results/theta_sd_ablation/result.json``.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from simulation.abm import epi_proof as ep
from simulation.abm.agent_kernel import run_agent_world
from simulation.database import read_only_connect

THETA_SD_GRID: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20)
DEFAULT_K = 16
DEFAULT_N_AGENTS = 30_000
DEFAULT_CAL_SEEDS = 8

DB_PATH = ep.DB_PATH
RESULT_PATH = (
    Path(__file__).resolve().parents[1] / "results" / "theta_sd_ablation" / "result.json"
)


def _load_seasons_read_only(db_path: Path) -> list[ep.SeasonSeries]:
    """Load the real Seoul ILI seasons through a lock-free read-only connection.

    Mirrors ``epi_proof._load_ili_seasons`` (same query, same >=20-week filter and
    finite check) but opens the DB with ``read_only_connect`` so the ablation can
    run while a writer holds the lock and can never mutate the source.

    Args:
        db_path: path to the project SQLite DB.

    Returns:
        Calendar-ordered list of ``SeasonSeries`` (>= 2, else ValueError).

    Raises:
        ValueError: if fewer than two seasons have >= 20 finite weekly points.

    Side effects: opens (and closes) one read-only fd; no writes.
    """
    con = read_only_connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """
            SELECT season_start, week_seq, AVG(ili_rate) AS ili_rate
            FROM sentinel_influenza
            WHERE ili_rate IS NOT NULL
              AND ili_rate >= 0
            GROUP BY season_start, week_seq
            ORDER BY season_start, week_seq
            """
        ).fetchall()
    finally:
        con.close()

    grouped: dict[int, list[tuple[int, float]]] = {}
    for row in rows:
        season = int(row["season_start"])
        grouped.setdefault(season, []).append(
            (int(row["week_seq"]), float(row["ili_rate"]))
        )

    seasons: list[ep.SeasonSeries] = []
    for season, values in sorted(grouped.items()):
        if len(values) < 20:
            continue
        week_seq = np.array([w for w, _ in values], dtype=np.int16)
        ili_rate = np.array([v for _, v in values], dtype=np.float64)
        if np.all(np.isfinite(ili_rate)):
            seasons.append(
                ep.SeasonSeries(season=season, week_seq=week_seq, ili_rate=ili_rate)
            )
    if len(seasons) < 2:
        raise ValueError("at least two real Seoul ILI seasons are required")
    return seasons


def _simulate_theta_sd(
    season: ep.SeasonSeries,
    *,
    seeds: Sequence[int],
    n_agents: int,
    disease: dict[str, float],
    behaviour: dict[str, float],
    theta_sd: float,
) -> tuple[np.ndarray, list[float]]:
    """Run K replicate ABM curves for one ``theta_sd``, holding all else fixed.

    Uses the same rich-movement population and disease/behaviour the frozen
    calibration produced; only ``theta_sd`` varies. ``theta_sd=0`` makes every
    agent's threshold exactly ``theta_mean`` (mean-field).

    Args:
        season: the season whose length / year drive the simulation horizon.
        seeds: replicate seeds; each is the kernel ``global_seed``.
        n_agents: number of agents (>= 25).
        disease: frozen disease dict (beta, sigma, ... forcing, import_rate).
        behaviour: frozen behaviour dict (alpha, kappa, tau, theta=theta_mean).
        theta_sd: relative SD of the per-agent compliance threshold.

    Returns:
        ``(weekly_replicates, compliance_var_per_seed)`` where the first is a
        ``(K, n_weeks)`` weekly-incidence array and the second is the final-day
        per-agent compliance variance for each replicate.

    Performance: O(K * n_weeks * 7 * n_agents). Side effects: none.
    """
    curves: list[np.ndarray] = []
    compliance_var: list[float] = []
    n_weeks = len(season.ili_rate)
    for seed in seeds:
        population = ep._make_population(
            "rich_movement", N=n_agents, seed=int(seed), year=int(season.season)
        )
        result = run_agent_world(
            N=n_agents,
            T_days=int(n_weeks * 7),
            beta=float(disease["beta"]),
            sigma=float(disease["sigma"]),
            gamma=float(disease["gamma"]),
            delta=float(disease["delta"]),
            nu=float(disease["nu"]),
            population=population,
            global_seed=int(seed),
            theta_mean=float(behaviour["theta"]),
            theta_sd=float(theta_sd),
            alpha_mean=float(behaviour["alpha"]),
            kappa_mean=float(behaviour["kappa"]),
            tau_mean=float(behaviour["tau"]),
            beta_amp=float(disease.get("beta_amp", 0.0)),
            beta_phase=float(disease.get("beta_phase", 0.0)),
            import_rate=float(disease.get("import_rate", 0.0)),
        )
        curves.append(ep._weekly_incidence(result, n_weeks))
        compliance_var.append(float(np.var(result["agents"]["compliance"])))
    return np.vstack(curves).astype(np.float64, copy=False), compliance_var


def _r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Coefficient of determination of ``predicted`` against ``observed``."""
    y = np.asarray(observed, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _arm_for_theta_sd(
    cal: ep.SeasonSeries,
    ev: ep.SeasonSeries,
    *,
    seeds: Sequence[int],
    n_agents: int,
    disease: dict[str, float],
    behaviour: dict[str, float],
    theta_sd: float,
) -> dict[str, Any]:
    """Calibrate the affine map on ``cal`` and score one theta_sd on held-out ``ev``.

    The affine map (offset + scale) is fit on the calibration-season replicate mean
    exactly as ``epi_proof._evaluate_arm`` does, then applied to the eval-season
    replicates before scoring — so the comparison is genuinely out-of-sample.

    Returns a dict of metrics: held-out WIS (+95% bootstrap CI), log score, fit
    R^2, observed/predicted peak size and ISO-week timing, the between-replicate
    trajectory spread (mean per-week SD of the mapped eval ensemble), and the
    mean final-day per-agent compliance variance.
    """
    cal_reps, _ = _simulate_theta_sd(
        cal,
        seeds=seeds,
        n_agents=n_agents,
        disease=disease,
        behaviour=behaviour,
        theta_sd=theta_sd,
    )
    affine = ep._fit_affine(cal_reps.mean(axis=0), cal.ili_rate)
    ev_reps, comp_var = _simulate_theta_sd(
        ev,
        seeds=seeds,
        n_agents=n_agents,
        disease=disease,
        behaviour=behaviour,
        theta_sd=theta_sd,
    )
    mapped = ep._apply_affine(ev_reps, affine)
    score = ep._score_forecast(ev.ili_rate, mapped)

    mapped_mean = mapped.mean(axis=0)
    obs = np.asarray(ev.ili_rate, dtype=np.float64)
    week_seq = np.asarray(ev.week_seq, dtype=np.int64)

    pred_peak_idx = int(np.argmax(mapped_mean))
    obs_peak_idx = int(np.argmax(obs))
    # Between-replicate trajectory spread: how much the K seeds disagree week by
    # week (mean over weeks of the per-week SD of the mapped ensemble).
    traj_spread = float(np.mean(mapped.std(axis=0, ddof=1 if mapped.shape[0] > 1 else 0)))

    return {
        "theta_sd": float(theta_sd),
        "wis": float(score["wis"]),
        "wis_ci95": [float(score["wis_ci95"][0]), float(score["wis_ci95"][1])],
        "log_score": float(score["log_score"]),
        "fit_r2": _r2(obs, mapped_mean),
        "affine": {"offset": float(affine.offset), "scale": float(affine.scale)},
        "degenerate": bool(abs(float(affine.scale)) < 1e-9),
        "pred_peak_value": float(mapped_mean[pred_peak_idx]),
        "pred_peak_week_seq": int(week_seq[pred_peak_idx]),
        "obs_peak_value": float(obs[obs_peak_idx]),
        "obs_peak_week_seq": int(week_seq[obs_peak_idx]),
        "peak_timing_error_weeks": int(pred_peak_idx - obs_peak_idx),
        "trajectory_spread": traj_spread,
        "mean_compliance_var": float(np.mean(comp_var)),
        "wis_per_week": [float(v) for v in score["wis_per_week"]],
        "mapped_mean": [float(v) for v in mapped_mean],
    }


def _summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the deltas-vs-mean-field table and the honest verdict.

    The verdict treats heterogeneity as material only if any theta_sd>0 row moves
    held-out WIS beyond the half-width of the mean-field row's 95% bootstrap CI;
    otherwise the differences are within seed noise and heterogeneity is dormant
    under observation.
    """
    base = next(r for r in rows if r["theta_sd"] == 0.0)
    base_wis = base["wis"]
    base_ci = base["wis_ci95"]
    base_ci_halfwidth = 0.5 * (base_ci[1] - base_ci[0])
    base_r2 = base["fit_r2"]
    base_peak = base["pred_peak_value"]
    base_spread = base["trajectory_spread"]

    table: list[dict[str, Any]] = []
    max_abs_wis_delta = 0.0
    for r in rows:
        d_wis = r["wis"] - base_wis
        max_abs_wis_delta = max(max_abs_wis_delta, abs(d_wis))
        table.append(
            {
                "theta_sd": r["theta_sd"],
                "wis": round(r["wis"], 6),
                "delta_wis_vs_meanfield": round(d_wis, 6),
                "fit_r2": round(r["fit_r2"], 6),
                "delta_r2_vs_meanfield": round(r["fit_r2"] - base_r2, 6),
                "pred_peak_value": round(r["pred_peak_value"], 4),
                "delta_peak_vs_meanfield": round(r["pred_peak_value"] - base_peak, 4),
                "pred_peak_week_seq": r["pred_peak_week_seq"],
                "peak_timing_error_weeks": r["peak_timing_error_weeks"],
                "trajectory_spread": round(r["trajectory_spread"], 6),
                "delta_spread_vs_meanfield": round(
                    r["trajectory_spread"] - base_spread, 6
                ),
                "mean_compliance_var": r["mean_compliance_var"],
            }
        )

    material = max_abs_wis_delta > base_ci_halfwidth
    verdict = (
        "MATERIAL: heterogeneity moves held-out WIS beyond the mean-field seed-noise "
        "band — theta_sd is an essential, observable degree of freedom."
        if material
        else "DORMANT-UNDER-OBSERVATION: every theta_sd>0 row stays inside the "
        "mean-field 95% bootstrap CI half-width, so on this aggregate ILI fit "
        "heterogeneity is finite-N seed noise, not a material driver of the mean "
        "trajectory. (Consistent with the mean-field limit: theta_sd->0 and "
        "N->inf collapse to the homogeneous curve.) Heterogeneity is dormant "
        "under aggregate observation but remains essential under targeting, where "
        "the per-agent compliance distribution — not just its mean — decides who "
        "is reached."
    )
    return {
        "table": table,
        "meanfield_wis": round(base_wis, 6),
        "meanfield_wis_ci95": base_ci,
        "meanfield_wis_ci_halfwidth": round(base_ci_halfwidth, 6),
        "max_abs_delta_wis": round(max_abs_wis_delta, 6),
        "heterogeneity_material": bool(material),
        "verdict": verdict,
    }


def run_theta_sd_ablation(
    *,
    K: int = DEFAULT_K,
    n_agents: int = DEFAULT_N_AGENTS,
    cal_seeds: int = DEFAULT_CAL_SEEDS,
    cal_season: int | None = None,
    eval_season: int | None = None,
    db_path: str | Path = DB_PATH,
    output_path: str | Path = RESULT_PATH,
) -> dict[str, Any]:
    """Run the full theta_sd ablation and write ``output_path``.

    Calibrates disease + behaviour once (epi_proof helpers), freezes them, then
    sweeps ``THETA_SD_GRID`` over K seeds and writes a deltas-vs-mean-field table
    plus an honest material/dormant verdict.

    Args:
        K: replicate seeds per theta_sd point (>= 2 for CIs).
        n_agents: agents per replicate (>= 25; large enough to avoid fizzle).
        cal_seeds: seeds used inside the one-time calibration.
        cal_season / eval_season: optional explicit split; default = epi_proof's.
        db_path / output_path: source DB and JSON destination.

    Returns:
        The full result dict (also written to ``output_path``).

    Side effects: writes one JSON file; reads the DB read-only.
    """
    db_path = Path(db_path)
    output_path = Path(output_path)
    seeds = tuple(range(int(K)))
    cal_seed_tuple = tuple(range(int(cal_seeds)))

    seasons = _load_seasons_read_only(db_path)
    cal, ev = ep._select_cal_eval(
        seasons, cal_season=cal_season, eval_season=eval_season
    )
    if cal.season == ev.season:
        raise ValueError("eval_season must differ from cal_season")

    # One-time frozen calibration (unmodified epi_proof helpers; internally uses
    # the live theta_sd=0.15 — that is the operating point we sweep around).
    disease, forcing_cal = ep._calibrate_forcing(
        cal, seeds=cal_seed_tuple, n_agents=n_agents
    )
    behaviour, behaviour_cal = ep._calibrate_behaviour(
        cal, seeds=cal_seed_tuple, n_agents=n_agents, disease=disease
    )

    rows = [
        _arm_for_theta_sd(
            cal,
            ev,
            seeds=seeds,
            n_agents=n_agents,
            disease=disease,
            behaviour=behaviour,
            theta_sd=theta_sd,
        )
        for theta_sd in THETA_SD_GRID
    ]
    summary = _summarise(rows)

    results = {
        "ablation": "theta_sd (household compliance-threshold heterogeneity SD)",
        "summary": summary,
        "rows": rows,
        "frozen": {
            "disease": {k: float(v) for k, v in disease.items()},
            "behaviour": {k: float(v) for k, v in behaviour.items()},
            "theta_mean": float(behaviour["theta"]),
            "note": (
                "theta is sampled as theta_mean*(1+theta_sd*N(0,1)) clipped at 0, "
                "so theta_sd is RELATIVE; absolute SD = theta_mean*theta_sd."
            ),
        },
        "metadata": {
            "theta_sd_grid": list(THETA_SD_GRID),
            "live_default_theta_sd": 0.15,
            "K": int(K),
            "seeds": [int(s) for s in seeds],
            "n_agents": int(n_agents),
            "cal_seeds": int(cal_seeds),
            "cal_season": int(cal.season),
            "eval_season": int(ev.season),
            "available_seasons": [
                {"season": int(s.season), "weeks": int(len(s.ili_rate))}
                for s in seasons
            ],
            "forcing_calibration": forcing_cal,
            "behaviour_calibration": behaviour_cal,
            "db_path": str(db_path),
            "output_path": str(output_path),
            "read_only_note": (
                "ILI loaded via read_only_connect (mode=ro, lock-free, never writes); "
                "no live code modified — all dynamics imported from epi_proof / "
                "agent_kernel."
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
    return results


def _print_table(results: dict[str, Any]) -> None:
    summary = results["summary"]
    meta = results["metadata"]
    frozen = results["frozen"]
    print(
        f"theta_sd ablation  cal={meta['cal_season']} -> eval={meta['eval_season']}  "
        f"K={meta['K']}  N={meta['n_agents']}"
    )
    print(
        f"frozen behaviour: theta_mean={frozen['theta_mean']:.3f}  "
        f"alpha={frozen['behaviour']['alpha']}  kappa={frozen['behaviour']['kappa']}  "
        f"tau={frozen['behaviour']['tau']}"
    )
    header = (
        f"{'theta_sd':>9} {'WIS':>9} {'dWIS':>9} {'R2':>8} {'peak':>9} "
        f"{'peak_wk':>8} {'spread':>9} {'compl_var':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in summary["table"]:
        print(
            f"{r['theta_sd']:>9.2f} {r['wis']:>9.4f} {r['delta_wis_vs_meanfield']:>9.4f} "
            f"{r['fit_r2']:>8.4f} {r['pred_peak_value']:>9.3f} "
            f"{r['pred_peak_week_seq']:>8d} {r['trajectory_spread']:>9.4f} "
            f"{r['mean_compliance_var']:>12.3e}"
        )
    print("-" * len(header))
    print(
        f"mean-field WIS={summary['meanfield_wis']:.4f}  "
        f"CI95={summary['meanfield_wis_ci95']}  "
        f"CI half-width={summary['meanfield_wis_ci_halfwidth']:.4f}  "
        f"max|dWIS|={summary['max_abs_delta_wis']:.4f}"
    )
    print(f"VERDICT: {summary['verdict']}")
    print(f"written: {meta['output_path']}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="theta_sd heterogeneity ablation")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="replicate seeds per point")
    parser.add_argument("--n-agents", type=int, default=DEFAULT_N_AGENTS, help="agents per replicate")
    parser.add_argument("--cal-seeds", type=int, default=DEFAULT_CAL_SEEDS, help="seeds for one-time calibration")
    parser.add_argument("--cal-season", type=int, default=None)
    parser.add_argument("--eval-season", type=int, default=None)
    args = parser.parse_args(argv)
    results = run_theta_sd_ablation(
        K=args.k,
        n_agents=args.n_agents,
        cal_seeds=args.cal_seeds,
        cal_season=args.cal_season,
        eval_season=args.eval_season,
    )
    _print_table(results)


if __name__ == "__main__":
    main()
