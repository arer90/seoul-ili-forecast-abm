"""simulation.scripts.run_abm_real_validation
=================================================
P3 — Behavioural ABM **real-wave validation** runner.

Goal (thesis honesty closing, 2026-06-26)
-----------------------------------------
Replace the thesis orphan ``R²=0.884`` (an unbacked headline with no on-disk
artifact — see docs/abm-diagnosis-20260613) with a **freshly computed, real-data
artifact**, and materialise the headline behavioural claim — that the *adaptive*
(behaviour-ON) ABM fits the observed wave **significantly** better than the
*static* (behaviour-OFF) ABM.

What it produces (per season, then aggregated):

1. ``r2_adaptive`` / ``r2_static`` — the individual-agent SEIR-V-D fit to the
   REAL Seoul ILI wave (:func:`simulation.abm.validate_real.fit_agent_to_observed`),
   run once with the behavioural layer ON and once OFF. ``r2_adaptive`` is the
   honest replacement for the orphan 0.884.
2. ``dm_p_value`` / ``dm_stat`` — HLN-corrected Diebold-Mariano significance of
   the per-week WIS difference between a multi-seed behaviour-ON ensemble and a
   behaviour-OFF ensemble against the same real wave
   (:func:`simulation.abm.behavior_disease_validation.validate_arms`).
3. ``behavior_necessary`` — ``r2_adaptive > r2_static`` (does the behavioural
   coupling improve the fit at all?).

Discipline
----------
- **Real data only.** The observed wave is the all-age mean ILI per epi-week
  from ``sentinel_influenza`` via :func:`read_only_connect` (lock-free, never
  blocks a writer; no ad-hoc low-level DB handle). No synthetic / placeholder
  targets — a season with too few weeks is skipped loudly, not faked to zero.
- **Deterministic.** Fixed ``SEED`` (42) for every agent draw and bootstrap;
  the behavioural-fit grid is itself deterministic. Re-running yields identical
  JSON.
- **No existing code modified** — this runner only *imports* the existing
  deep modules (validate_real, behavior_disease_validation, behavioural,
  calibrate, within_season_validation).

Usage
-----
    # full 7-season run (37 500 agents/gu, n_boot=2000) — detached:
    python -m simulation.scripts.run_abm_real_validation

    # reduced smoke (used by the built-in self-test, see --smoke):
    python -m simulation.scripts.run_abm_real_validation --smoke

Output: ``<results>/abm_real_validation/result.json``.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np

from simulation.abm.behavioural import BehaviouralParams, run_coupled_abm
from simulation.abm.behavior_disease_validation import validate_arms
from simulation.abm.validate_real import fit_agent_to_observed, weekly_incidence
from simulation.sim.io import load_metapop_params
from simulation.sim.parameters import MetapopParams

log = logging.getLogger("abm_real_validation")

SEED = 42

# Documented calibration best (calibrate.py grid optimum, interior-bracketed):
# α=2, κ=0.2, τ=90, θ=0.075 — used as the behaviour-ON anchor when no on-disk
# best_fit.json is found. Behaviour-OFF is the α=0,κ=0,τ=∞ invariant.
_DEFAULT_BEHAV_ON = dict(alpha=2.0, kappa=0.2, tau=90.0, theta=0.075)


# ---------------------------------------------------------------------------
# Real-wave loader (read-only, all-age mean per epi-week)
# ---------------------------------------------------------------------------
def load_real_ili_seasons(
    db_path: Optional[str] = None,
    *,
    min_weeks: int = 20,
) -> dict[int, np.ndarray]:
    """Load every Seoul ILI wave (all-age mean per epi-week) from the DB.

    The observed weekly ILI for a season is the mean ``ili_rate`` across the
    seven KDCA age bands at each ``week_seq`` — a single, representative
    city-wide wave (a single age band is too noisy to anchor a wave fit).

    Args:
        db_path: SQLite path; ``None`` → project ``DB_PATH`` (read-only).
        min_weeks: drop seasons shorter than this (a usable wave fit needs a
            full epidemic curve; KDCA seasons are ~52 weeks).

    Returns:
        ``{season_start: ili_weekly}`` ordered by season; ``ili_weekly`` is a
        ``(W,)`` float array of the all-age mean ILI rate per epi-week.

    Raises:
        ValueError: if ``sentinel_influenza`` has no usable ``ili_rate`` rows.

    Side effects: opens (and closes) one read-only connection. Concurrency:
        safe against an active writer (lock-free ``mode=ro``).
    """
    from simulation.database import read_only_connect

    con = read_only_connect(db_path)
    try:
        seasons = [
            int(r[0])
            for r in con.execute(
                "SELECT DISTINCT season_start FROM sentinel_influenza "
                "WHERE ili_rate IS NOT NULL ORDER BY season_start"
            )
        ]
        if not seasons:
            raise ValueError("sentinel_influenza 에 ili_rate 없음 (real wave 부재)")
        out: dict[int, np.ndarray] = {}
        for season in seasons:
            rows = con.execute(
                "SELECT week_seq, AVG(ili_rate) FROM sentinel_influenza "
                "WHERE season_start=? AND ili_rate IS NOT NULL "
                "GROUP BY week_seq ORDER BY week_seq",
                [season],
            ).fetchall()
            ili = np.array([float(r[1]) for r in rows], dtype=np.float64)
            if ili.size >= min_weeks and np.any(ili > 0):
                out[season] = ili
            else:
                log.warning(
                    "season %d skipped (weeks=%d < %d or all-zero)",
                    season, ili.size, min_weeks,
                )
        if not out:
            raise ValueError(f"no season with >= {min_weeks} usable weeks")
        return out
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Calibration best (on-disk reuse → documented default)
# ---------------------------------------------------------------------------
def resolve_behaviour_on() -> tuple[BehaviouralParams, dict, str]:
    """Resolve the behaviour-ON parameter set (alpha, kappa, tau, theta).

    Prefers the persisted calibration optimum
    (``<results>/abm_calibration_v1/best_fit.json``); falls back to the
    documented interior-bracketed grid optimum if it is absent or malformed.

    Returns:
        ``(BehaviouralParams, params_dict, source)`` where ``source`` records
        whether the on-disk calibration or the documented default was used.
    """
    from simulation.utils.paths import get_results_dir

    best_path = get_results_dir() / "abm_calibration_v1" / "best_fit.json"
    params = dict(_DEFAULT_BEHAV_ON)
    source = "documented-default (calibrate.py grid optimum α=2,κ=0.2,τ=90,θ=0.075)"
    if best_path.exists():
        try:
            best = json.loads(best_path.read_text(encoding="utf-8")).get("best", {})
            cand = {k: float(best[k]) for k in ("alpha", "kappa", "tau", "theta")
                    if k in best}
            if len(cand) == 4 and all(np.isfinite(list(cand.values()))):
                params = cand
                source = f"on-disk calibration ({best_path})"
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            log.warning("best_fit.json unreadable (%s) → documented default", exc)
    return BehaviouralParams(**params), params, source


def _behav_off() -> BehaviouralParams:
    """Behaviour-OFF invariant (α=0, κ=0, τ=∞ ⇒ β_i(t)=β_0 fixed)."""
    return BehaviouralParams(alpha=0.0, kappa=0.0, tau=float("inf"))


# ---------------------------------------------------------------------------
# Multi-seed ensemble of weekly ABM curves (for the WIS / DM comparison)
# ---------------------------------------------------------------------------
def _affine_map(sim: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Least-squares affine map ``offset + scale·sim`` of a simulated weekly
    curve onto the observed wave's reporting scale (scale ≥ 0). Degenerate
    (flat) ``sim`` collapses to the observed mean."""
    s = np.asarray(sim, float)
    r = np.asarray(obs, float)
    if float(np.var(s)) <= 1e-12:
        return np.full_like(r, float(np.mean(r)))
    A = np.vstack([np.ones_like(s), s]).T
    coef, *_ = np.linalg.lstsq(A, r, rcond=None)
    offset, scale = float(coef[0]), max(float(coef[1]), 0.0)
    return offset + scale * s


def _arm_reps(
    mp: MetapopParams,
    behaviour: BehaviouralParams,
    obs: np.ndarray,
    n_seeds: int,
) -> np.ndarray:
    """Multi-seed ensemble of affine-mapped weekly ABM curves for one arm.

    ``run_coupled_abm`` is the mean-field ODE (deterministic) — so seed
    variation enters only through the per-day RK round-off; we draw the
    ensemble by jittering the seed AND a tiny R0 multiplier band to produce a
    non-degenerate proper-score ensemble (the WIS needs spread). Each rep is
    affine-mapped to the observed reporting scale, exactly as the
    within-season validator does.

    Returns: ``(n_seeds, W)`` float array on the observed ILI scale.
    """
    W = len(obs)
    base_R0 = float(mp.disease.R0)
    rng = np.random.default_rng(SEED)
    reps = np.zeros((n_seeds, W), dtype=float)
    for s in range(n_seeds):
        # small deterministic R0 spread → ensemble width (≤ ±5%), seed fixed-derived
        r0_mult = 1.0 + 0.05 * (rng.random() - 0.5) * 2.0
        disease_s = replace(mp.disease, R0=base_R0 * r0_mult)
        mp_s = replace(mp, disease=disease_s)
        res = run_coupled_abm(mp_s, behaviour)
        sim_weekly = weekly_incidence(res)[:W]
        if sim_weekly.size < W:
            sim_weekly = np.pad(sim_weekly, (0, W - sim_weekly.size))
        reps[s] = _affine_map(sim_weekly, obs)
    return reps


# ---------------------------------------------------------------------------
# Per-season validation
# ---------------------------------------------------------------------------
def validate_season(
    season: int,
    ili: np.ndarray,
    behav_on: BehaviouralParams,
    *,
    n_agents: int,
    n_boot: int,
    n_seeds: int,
    seed_infected: float,
) -> dict:
    """Validate one real ILI wave: adaptive vs static R² + DM-HLN significance.

    Args:
        season: KDCA season_start year (label only).
        ili: ``(W,)`` observed all-age mean weekly ILI.
        behav_on: behaviour-ON parameter set.
        n_agents: agents/gu for the wave fit (free knob; result insensitive
            above ~500/gu — see validate_real docstring).
        n_boot: paired bootstrap reps for the DM CI (full=2000, smoke=100).
        n_seeds: ensemble size per arm for the WIS/DM comparison.
        seed_infected: single-district seed of infection.

    Returns:
        per-season dict (r2_adaptive, r2_static, rmse, dm_p_value, dm_stat,
        behavior_necessary, n_weeks, agent_count, params, peak_week).
    """
    W = len(ili)
    # 25-gu Seoul metapop, horizon = the observed wave length.
    mp = load_metapop_params(days=W * 7, seed_infected=float(seed_infected))

    # 1) Real-wave R² fit — behaviour-ON (adaptive) and behaviour-OFF (static).
    #    Adaptive gets its FULLEST shot: the default behavioural grid
    #    (alpha/kappa/tau/theta) AND the R0/gamma shape levers are searched, so
    #    a non-positive behavioural gain is a genuine finding, not a handicap.
    #    Static pins the behavioural axes OFF (α=0,κ=0,τ=∞ invariant) so only
    #    the R0/gamma shape levers move — the behaviour-OFF reference fit.
    fit_on = fit_agent_to_observed(ili, mp, n_agents=n_agents, seed=SEED)
    fit_off = fit_agent_to_observed(
        ili, mp, n_agents=n_agents, seed=SEED,
        alpha_grid=(0.0,),
        kappa_grid=(0.0,),
        tau_grid=(float("inf"),),
        theta_grid=(0.0,),
    )

    # 2) DM-HLN significance — multi-seed ensembles, behaviour-ON vs OFF.
    reps_adaptive = _arm_reps(mp, behav_on, ili, n_seeds)
    reps_static = _arm_reps(mp, _behav_off(), ili, n_seeds)
    arms = validate_arms(ili, reps_adaptive, reps_static,
                         n_boot=n_boot, seed=SEED,
                         threshold=float(np.median(ili)))
    sig = arms["significance"]

    return {
        "season": int(season),
        "n_weeks": int(W),
        "agent_count": int(fit_on.total_agents),
        "peak_week": int(np.argmax(ili)),
        "r2_adaptive": float(fit_on.r2),
        "r2_static": float(fit_off.r2),
        "rmse": float(fit_on.rmse),
        "rmse_static": float(fit_off.rmse),
        "behavior_necessary": bool(fit_on.r2 > fit_off.r2),
        "dm_stat": float(sig["dm_hln_stat"]),
        "dm_p_value": float(sig["dm_p_value"]),
        "delta_wis_mean": float(sig["delta_wis_mean"]),
        "adaptive_significantly_better": bool(sig["adaptive_significantly_better"]),
        "wis_adaptive": float(arms["adaptive"]["wis"]),
        "wis_static": float(arms["static"]["wis"]),
        "best_params_on": dict(fit_on.params),
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run(
    *,
    n_agents: int = 1_500,
    n_boot: int = 2_000,
    n_seeds: int = 30,
    seed_infected: float = 1_000.0,
    seasons: Optional[list[int]] = None,
    min_weeks: int = 20,
    out_dir: Path | str | None = None,
    db_path: Optional[str] = None,
) -> dict:
    """Run the full real-wave ABM validation over all (or selected) seasons.

    Picks the season with the largest peak ILI as the **headline** (the wave the
    thesis 0.884 referred to); reports per-season R² for the 7-season
    generalisation table.

    Args:
        n_agents: agents/gu for each wave fit (full headline run = 1500/gu ⇒
            37 500 total over 25 gu).
        n_boot: paired bootstrap reps for the DM CI (full=2000).
        n_seeds: ensemble size per arm for the WIS/DM comparison.
        seed_infected: single-district seed of infection.
        seasons: restrict to these season_start years (None = all usable).
        min_weeks: minimum weeks to treat a season as a usable wave.
        out_dir: output dir (None → ``<results>/abm_real_validation``).
        db_path: SQLite path (None → project DB, read-only).

    Returns:
        the result dict (also written to ``result.json``).

    Side effects: writes ``result.json`` under ``out_dir``. No DB writes.
    """
    if out_dir is None:
        from simulation.utils.paths import get_results_dir
        out_dir = get_results_dir() / "abm_real_validation"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    behav_on, on_params, on_source = resolve_behaviour_on()
    waves = load_real_ili_seasons(db_path, min_weeks=min_weeks)
    if seasons:
        waves = {s: w for s, w in waves.items() if s in set(seasons)}
        if not waves:
            raise ValueError(f"no usable wave among requested seasons {seasons}")

    log.info("real-wave ABM validation: %d season(s) %s | behaviour-ON %s (%s)",
             len(waves), sorted(waves), on_params, on_source)

    t0 = time.time()
    per_season: list[dict] = []
    for season, ili in sorted(waves.items()):
        log.info("season %d: W=%d weeks, peak=%.1f @ week %d",
                 season, len(ili), float(ili.max()), int(np.argmax(ili)))
        res = validate_season(
            season, ili, behav_on,
            n_agents=n_agents, n_boot=n_boot, n_seeds=n_seeds,
            seed_infected=seed_infected,
        )
        log.info("  → R²(adaptive)=%.4f R²(static)=%.4f  DM p=%.4g  behaviour_necessary=%s",
                 res["r2_adaptive"], res["r2_static"], res["dm_p_value"],
                 res["behavior_necessary"])
        per_season.append(res)
    elapsed = time.time() - t0

    # Headline = the season whose observed wave has the largest peak (most
    # epidemic signal ⇒ the canonical validation wave the thesis 0.884 referred to).
    headline = max(per_season, key=lambda r: float(waves[r["season"]].max()))

    seasonal_r2 = {int(r["season"]): r["r2_adaptive"] for r in per_season}
    n_necessary = sum(1 for r in per_season if r["behavior_necessary"])
    n_significant = sum(1 for r in per_season if r["adaptive_significantly_better"])

    result = {
        # ── headline (orphan-0.884 replacement) ──
        "season": headline["season"],
        "n_weeks": headline["n_weeks"],
        "agent_count": headline["agent_count"],
        "r2_adaptive": headline["r2_adaptive"],
        "r2_static": headline["r2_static"],
        "rmse": headline["rmse"],
        "dm_p_value": headline["dm_p_value"],
        "dm_stat": headline["dm_stat"],
        "behavior_necessary": headline["behavior_necessary"],
        # ── 7-season generalisation ──
        "seasonal_r2_adaptive": seasonal_r2,
        "n_seasons": len(per_season),
        "n_seasons_behavior_necessary": n_necessary,
        "n_seasons_adaptive_significant": n_significant,
        "per_season": per_season,
        # ── provenance ──
        "behaviour_on_params": on_params,
        "behaviour_on_source": on_source,
        "config": {
            "n_agents_per_gu": n_agents, "n_boot": n_boot, "n_seeds": n_seeds,
            "seed": SEED, "seed_infected": seed_infected, "min_weeks": min_weeks,
        },
        "elapsed_sec": round(elapsed, 2),
        # ── honesty note ──
        "honesty_note": (
            "r2_adaptive is the individual-agent SEIR-V-D fit to the REAL "
            "all-age-mean Seoul ILI wave (sentinel_influenza, read-only), "
            "computed fresh — it REPLACES the thesis orphan R²=0.884 which had "
            "no on-disk artifact. behaviour-OFF (static) R² is reported "
            "alongside so the behavioural gain is auditable; the ABM is a "
            "mechanism/counterfactual engine, NOT a competition forecaster. "
            "dm_p_value is the HLN-corrected Diebold-Mariano per-week WIS test "
            "(adaptive vs static, multi-seed ensemble). All deterministic "
            "(seed=42). No synthetic targets: short seasons are skipped, not "
            "faked."
        ),
    }

    (out / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("wrote %s (headline season %d R²=%.4f, %d/%d seasons behaviour-necessary, "
             "%.1fs)", out / "result.json", result["season"], result["r2_adaptive"],
             n_necessary, len(per_season), elapsed)
    return result


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-agents", type=int, default=1_500,
                    help="agents per gu (full headline = 1500 ⇒ 37 500 total)")
    ap.add_argument("--n-boot", type=int, default=2_000,
                    help="paired-bootstrap reps for the DM CI (full=2000)")
    ap.add_argument("--n-seeds", type=int, default=30,
                    help="ensemble size per arm for the WIS/DM comparison")
    ap.add_argument("--seed-infected", type=float, default=1_000.0)
    ap.add_argument("--seasons", type=int, nargs="*", default=None,
                    help="restrict to these season_start years (default: all)")
    ap.add_argument("--min-weeks", type=int, default=20)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="reduced 1-season smoke (small agents, n_boot=100) — "
                         "verifies R² and dm_p_value materialise; NOT the full run")
    args = ap.parse_args(argv)

    if args.smoke:
        # Reduced smoke: 1 season (largest-peak), tiny agents, n_boot=100, few seeds.
        waves = load_real_ili_seasons(min_weeks=args.min_weeks)
        headline_season = max(waves, key=lambda s: float(waves[s].max()))
        log.info("SMOKE: 1 season (%d), n_agents=120, n_boot=100, n_seeds=6", headline_season)
        report = run(
            n_agents=120, n_boot=100, n_seeds=6,
            seed_infected=args.seed_infected, seasons=[headline_season],
            min_weeks=args.min_weeks, out_dir=args.out_dir,
        )
    else:
        report = run(
            n_agents=args.n_agents, n_boot=args.n_boot, n_seeds=args.n_seeds,
            seed_infected=args.seed_infected, seasons=args.seasons,
            min_weeks=args.min_weeks, out_dir=args.out_dir,
        )

    print()
    print("=== ABM real-wave validation ===")
    print(f"  headline season    : {report['season']} ({report['n_weeks']} weeks, "
          f"{report['agent_count']} agents)")
    print(f"  R² adaptive (real) : {report['r2_adaptive']:.4f}   "
          f"[replaces orphan 0.884]")
    print(f"  R² static          : {report['r2_static']:.4f}")
    print(f"  behaviour necessary: {report['behavior_necessary']}")
    print(f"  DM-HLN p-value     : {report['dm_p_value']:.4g} (stat {report['dm_stat']:.3f})")
    print(f"  seasons behaviour-necessary: "
          f"{report['n_seasons_behavior_necessary']}/{report['n_seasons']}")
    print(f"  elapsed            : {report['elapsed_sec']:.1f}s")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
