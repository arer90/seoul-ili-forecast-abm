"""Out-of-sample real-Seoul ILI comparisons for the attribute ABM."""
from __future__ import annotations

import copy
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import stats

from simulation.abm.agent_kernel import run_agent_world
from simulation.abm.synthetic_population import generate_population
from simulation.database import safe_connect


DB_PATH = Path(__file__).resolve().parents[1] / "data" / "db" / "epi_real_seoul.db"
RESULT_PATH = (
    Path(__file__).resolve().parents[2]
    / "paper"
    / "_thesis_revision_20260604"
    / "real_runs"
    / "epi_proof.json"
)

QUANTILE_LEVELS = np.array([0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975])
INTERVAL_SPECS = (
    (0, 6, 0.05),
    (1, 5, 0.20),
    (2, 4, 0.50),
)
DEFAULT_DISEASE = {
    "beta": 0.18,
    "sigma": 0.45,
    "gamma": 0.18,
    "delta": 0.002,
    "nu": 0.0002,
    # Seasonal forcing (calibrated on 2023 by _calibrate_forcing). A constant
    # beta burns out in ~3 weeks and anti-correlates with the real ~20-week ILI
    # season; beta(t)=beta*(1+amp*cos(2pi(t-phase)/365.25)) plus a tiny
    # importation reproduces the observed seasonal curve (corr~0.97 vs -0.07).
    "beta_amp": 0.65,
    "beta_phase": 120.0,
    "import_rate": 3.0e-4,
}
FORCING_GRID = {
    "beta": (0.15, 0.18, 0.22),
    "beta_amp": (0.45, 0.55, 0.65),
    "beta_phase": (90.0, 105.0, 120.0),
}
BEHAVIOUR_OFF = {
    "alpha": 0.0,
    "kappa": 0.0,
    "tau": 1.0e9,
    "theta": 0.0,
}
BEHAVIOUR_GRID = {
    "alpha": (0.15, 0.45, 0.90),
    "kappa": (0.0, 0.30),
    "tau": (14.0, 60.0),
    "theta": (0.05, 0.20),
}
_RUN_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
_POPULATION_CACHE: dict[tuple[int, int, int], dict[str, np.ndarray]] = {}
_SIM_CACHE: dict[tuple[Any, ...], np.ndarray] = {}


@dataclass(frozen=True)
class SeasonSeries:
    season: int
    week_seq: np.ndarray
    ili_rate: np.ndarray


@dataclass(frozen=True)
class AffineMap:
    offset: float
    scale: float


def run_epi_proof(
    *,
    K: int = 30,
    seeds: Sequence[int] | None = None,
    n_agents: int = 37_500,
    cal_season: int | None = None,
    eval_season: int | None = None,
    db_path: str | Path = DB_PATH,
    output_path: str | Path = RESULT_PATH,
) -> dict[str, Any]:
    """Run held-out ABM comparisons and write ``output_path``.

    Raises ``ValueError`` when fewer than two real seasons exist or the selected
    calibration and evaluation seasons are the same. Performance is
    O(configs*K*season_days*n_agents); side effect is the JSON write.
    """
    seed_tuple = tuple(int(s) for s in (range(K) if seeds is None else seeds))
    if not seed_tuple:
        raise ValueError("at least one seed is required")
    if n_agents < 25:
        raise ValueError("n_agents must be >= 25")

    db_path = Path(db_path)
    output_path = Path(output_path)
    cache_key = (str(db_path.resolve()), str(output_path.resolve()), seed_tuple, int(n_agents), cal_season, eval_season)
    if cache_key in _RUN_CACHE:
        cached = copy.deepcopy(_RUN_CACHE[cache_key])
        _write_json(cached, output_path)
        return cached

    seasons = _load_ili_seasons(db_path)
    cal, ev = _select_cal_eval(seasons, cal_season=cal_season, eval_season=eval_season)
    if cal.season == ev.season:
        raise ValueError("eval_season must differ from cal_season")

    cal_seeds = seed_tuple[: min(len(seed_tuple), 8)]
    disease, forcing_cal = _calibrate_forcing(
        cal, seeds=cal_seeds, n_agents=n_agents
    )
    fitted_behaviour, calibration = _calibrate_behaviour(
        cal, seeds=cal_seeds, n_agents=n_agents, disease=disease
    )
    calibration["forcing"] = forcing_cal

    common = {
        "seeds": seed_tuple,
        "n_agents": int(n_agents),
        "disease": disease,
    }
    rich_on = _evaluate_arm(
        cal,
        ev,
        behaviour=fitted_behaviour,
        population_kind="rich_movement",
        **common,
    )
    behaviour_off = _evaluate_arm(
        cal,
        ev,
        behaviour=BEHAVIOUR_OFF,
        population_kind="rich_movement",
        **common,
    )
    homogeneous = _evaluate_arm(
        cal,
        ev,
        behaviour=fitted_behaviour,
        population_kind="homogeneous_static",
        **common,
    )
    static_movement = _evaluate_arm(
        cal,
        ev,
        behaviour=fitted_behaviour,
        population_kind="rich_static",
        **common,
    )

    results = {
        "comparison_1_behaviour": _comparison(
            on=rich_on,
            off=behaviour_off,
            on_label="behaviour ON",
            off_label="behaviour OFF",
            ev=ev,
        ),
        "comparison_2_heterogeneity": _comparison(
            on=rich_on,
            off=homogeneous,
            on_label="heterogeneity ON",
            off_label="heterogeneity OFF",
            ev=ev,
        ),
        "comparison_3_movement": _comparison(
            on=rich_on,
            off=static_movement,
            on_label="scheduled movement ON",
            off_label="scheduled movement OFF",
            ev=ev,
        ),
        "metadata": {
            "cal_season": int(cal.season),
            "eval_season": int(ev.season),
            "K": int(len(seed_tuple)),
            "seeds": [int(s) for s in seed_tuple],
            "n_agents": int(n_agents),
            "available_seasons": [
                {"season": int(s.season), "weeks": int(len(s.ili_rate))}
                for s in seasons
            ],
            "calibration": calibration,
            "behaviour_off": dict(BEHAVIOUR_OFF),
            "disease_params": dict(disease),
            "quantile_levels": [float(q) for q in QUANTILE_LEVELS],
            "db_path": str(db_path),
            "output_path": str(output_path),
            "read_only_note": "safe_connect has no read_only keyword here; only SELECT queries are issued.",
        },
    }
    _write_json(results, output_path)
    _RUN_CACHE[cache_key] = copy.deepcopy(results)
    return copy.deepcopy(results)


def _load_ili_seasons(db_path: Path) -> list[SeasonSeries]:
    with safe_connect(str(db_path), verify=False) as conn:
        rows = conn.execute(
            """
            SELECT season_start, week_seq, AVG(ili_rate) AS ili_rate
            FROM sentinel_influenza
            WHERE ili_rate IS NOT NULL
              AND ili_rate >= 0
            GROUP BY season_start, week_seq
            ORDER BY season_start, week_seq
            """
        ).fetchall()

    grouped: dict[int, list[tuple[int, float]]] = {}
    for row in rows:
        season = int(row["season_start"])
        grouped.setdefault(season, []).append((int(row["week_seq"]), float(row["ili_rate"])))

    seasons: list[SeasonSeries] = []
    for season, values in sorted(grouped.items()):
        if len(values) < 20:
            continue
        week_seq = np.array([w for w, _ in values], dtype=np.int16)
        ili_rate = np.array([v for _, v in values], dtype=np.float64)
        if np.all(np.isfinite(ili_rate)):
            seasons.append(SeasonSeries(season=season, week_seq=week_seq, ili_rate=ili_rate))
    if len(seasons) < 2:
        raise ValueError("at least two real Seoul ILI seasons are required")
    return seasons


def _select_cal_eval(
    seasons: list[SeasonSeries],
    *,
    cal_season: int | None,
    eval_season: int | None,
) -> tuple[SeasonSeries, SeasonSeries]:
    by_season = {s.season: s for s in seasons}
    if cal_season is not None and cal_season not in by_season:
        raise ValueError(f"cal_season {cal_season} not found in DB")
    if eval_season is not None and eval_season not in by_season:
        raise ValueError(f"eval_season {eval_season} not found in DB")
    if cal_season is not None and eval_season is not None:
        if cal_season == eval_season:
            raise ValueError("eval_season must differ from cal_season")
        return by_season[int(cal_season)], by_season[int(eval_season)]
    pool = [s for s in seasons if len(s.ili_rate) >= 50]
    if len(pool) < 2:
        pool = seasons
    if eval_season is not None:
        ev = by_season[int(eval_season)]
        candidates = [s for s in pool if s.season != ev.season]
        if not candidates:
            candidates = [s for s in seasons if s.season != ev.season]
        return candidates[-1], ev
    if cal_season is not None:
        cal = by_season[int(cal_season)]
        candidates = [s for s in pool if s.season != cal.season]
        if not candidates:
            candidates = [s for s in seasons if s.season != cal.season]
        return cal, candidates[-1]
    return pool[-2], pool[-1]


def _calibrate_forcing(
    season: SeasonSeries,
    *,
    seeds: Sequence[int],
    n_agents: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Grid-search seasonal forcing (beta, amplitude, phase) on the calibration
    season with behaviour disabled, minimising calibration WIS after the affine
    map. A constant beta burns out in ~3 weeks and anti-correlates with the real
    season; this stage finds the beta(t) that reproduces the observed curve."""
    best: tuple[float, dict[str, float], float] | None = None
    tried = 0
    for beta, amp, phase in itertools.product(
        FORCING_GRID["beta"],
        FORCING_GRID["beta_amp"],
        FORCING_GRID["beta_phase"],
    ):
        disease = {
            **DEFAULT_DISEASE,
            "beta": float(beta),
            "beta_amp": float(amp),
            "beta_phase": float(phase),
        }
        reps = _simulate_replicates(
            season,
            seeds=seeds,
            n_agents=n_agents,
            behaviour=BEHAVIOUR_OFF,
            population_kind="rich_movement",
            disease=disease,
        )
        sim_mean = reps.mean(axis=0)
        affine = _fit_affine(sim_mean, season.ili_rate)
        wis = float(np.mean(_wis_per_week(season.ili_rate, _apply_affine(reps, affine))))
        corr = (
            float(np.corrcoef(sim_mean, season.ili_rate)[0, 1])
            if sim_mean.std() > 1e-12 else 0.0
        )
        tried += 1
        if best is None or wis < best[0]:
            best = (wis, disease, corr)
    if best is None:
        raise RuntimeError("forcing calibration produced no valid candidate")
    best_wis, disease, corr = best
    return disease, {
        "season": int(season.season),
        "grid_size": int(tried),
        "objective": "calibration-season WIS after affine map (behaviour off)",
        "best_wis": float(best_wis),
        "corr_sim_vs_obs": float(corr),
        "fitted_forcing": {
            "beta": float(disease["beta"]),
            "beta_amp": float(disease["beta_amp"]),
            "beta_phase": float(disease["beta_phase"]),
            "import_rate": float(disease["import_rate"]),
        },
    }


def _calibrate_behaviour(
    season: SeasonSeries,
    *,
    seeds: Sequence[int],
    n_agents: int,
    disease: dict[str, float],
) -> tuple[dict[str, float], dict[str, Any]]:
    best: tuple[float, dict[str, float], AffineMap] | None = None
    tried = 0
    for alpha, kappa, tau, theta in itertools.product(
        BEHAVIOUR_GRID["alpha"],
        BEHAVIOUR_GRID["kappa"],
        BEHAVIOUR_GRID["tau"],
        BEHAVIOUR_GRID["theta"],
    ):
        behaviour = {
            "alpha": float(alpha),
            "kappa": float(kappa),
            "tau": float(tau),
            "theta": float(theta),
        }
        reps = _simulate_replicates(
            season,
            seeds=seeds,
            n_agents=n_agents,
            behaviour=behaviour,
            population_kind="rich_movement",
            disease=disease,
        )
        affine = _fit_affine(reps.mean(axis=0), season.ili_rate)
        mapped = _apply_affine(reps, affine)
        wis = float(np.mean(_wis_per_week(season.ili_rate, mapped)))
        tried += 1
        if best is None or wis < best[0]:
            best = (wis, behaviour, affine)
    if best is None:
        raise RuntimeError("behaviour calibration produced no valid candidate")
    best_wis, behaviour, affine = best
    return behaviour, {
        "season": int(season.season),
        "seeds": [int(s) for s in seeds],
        "grid_size": int(tried),
        "objective": "calibration-season WIS after affine map",
        "best_wis": float(best_wis),
        "fitted_behaviour": dict(behaviour),
        "affine_on_calibration_mean": {
            "offset": float(affine.offset),
            "scale": float(affine.scale),
        },
    }


def _evaluate_arm(
    cal: SeasonSeries,
    ev: SeasonSeries,
    *,
    seeds: Sequence[int],
    n_agents: int,
    disease: dict[str, float],
    behaviour: dict[str, float],
    population_kind: str,
) -> dict[str, Any]:
    cal_reps = _simulate_replicates(
        cal,
        seeds=seeds,
        n_agents=n_agents,
        disease=disease,
        behaviour=behaviour,
        population_kind=population_kind,
    )
    affine = _fit_affine(cal_reps.mean(axis=0), cal.ili_rate)
    ev_reps = _simulate_replicates(
        ev,
        seeds=seeds,
        n_agents=n_agents,
        disease=disease,
        behaviour=behaviour,
        population_kind=population_kind,
    )
    mapped = _apply_affine(ev_reps, affine)
    score = _score_forecast(ev.ili_rate, mapped)
    return {
        "population_kind": population_kind,
        "behaviour": dict(behaviour),
        "affine": {"offset": float(affine.offset), "scale": float(affine.scale)},
        "mapped_replicates": mapped,
        "wis_per_week": score.pop("wis_per_week"),
        "log_score_per_week": score.pop("log_score_per_week"),
        "metrics": score,
    }


def _simulate_replicates(
    season: SeasonSeries,
    *,
    seeds: Sequence[int],
    n_agents: int,
    disease: dict[str, float],
    behaviour: dict[str, float],
    population_kind: str,
    transmission_mode: str = "meanfield",
    network_kwargs: dict | None = None,
    beta_by_layer: dict | None = None,
) -> np.ndarray:
    curves = [
        _simulate_one(
            season,
            seed=int(seed),
            n_agents=n_agents,
            disease=disease,
            behaviour=behaviour,
            population_kind=population_kind,
            transmission_mode=transmission_mode,
            network_kwargs=network_kwargs,
            beta_by_layer=beta_by_layer,
        )
        for seed in seeds
    ]
    return np.vstack(curves).astype(np.float64, copy=False)


def _simulate_one(
    season: SeasonSeries,
    *,
    seed: int,
    n_agents: int,
    disease: dict[str, float],
    behaviour: dict[str, float],
    population_kind: str,
    transmission_mode: str = "meanfield",
    network_kwargs: dict | None = None,
    beta_by_layer: dict | None = None,
) -> np.ndarray:
    behaviour_key = tuple((k, float(behaviour[k])) for k in ("alpha", "kappa", "tau", "theta"))
    disease_key = tuple(
        (k, float(disease.get(k, 0.0)))
        for k in ("beta", "sigma", "gamma", "delta", "nu",
                  "beta_amp", "beta_phase", "import_rate")
    )
    def _hkey(d: dict | None) -> tuple:
        out = []
        for k in sorted(d or {}):
            if k == "provenance":
                continue  # doc-only; not a simulation input
            v = d[k]
            if isinstance(v, np.ndarray):
                v = ("arr", v.shape, hash(v.tobytes()))
            elif isinstance(v, dict):
                v = tuple(sorted(v.items()))
            out.append((k, v))
        return tuple(out)

    net_key = (_hkey(network_kwargs), _hkey(beta_by_layer))
    key = (season.season, len(season.ili_rate), seed, n_agents, population_kind,
           behaviour_key, disease_key, transmission_mode, net_key)
    if key in _SIM_CACHE:
        return _SIM_CACHE[key].copy()

    population = _make_population(
        population_kind,
        N=n_agents,
        seed=seed,
        year=int(season.season),
    )
    result = run_agent_world(
        N=n_agents,
        T_days=int(len(season.ili_rate) * 7),
        beta=float(disease["beta"]),
        sigma=float(disease["sigma"]),
        gamma=float(disease["gamma"]),
        delta=float(disease["delta"]),
        nu=float(disease["nu"]),
        population=population,
        global_seed=int(seed),
        theta_mean=float(behaviour["theta"]),
        theta_sd=0.15,
        alpha_mean=float(behaviour["alpha"]),
        kappa_mean=float(behaviour["kappa"]),
        tau_mean=float(behaviour["tau"]),
        beta_amp=float(disease.get("beta_amp", 0.0)),
        beta_phase=float(disease.get("beta_phase", 0.0)),
        import_rate=float(disease.get("import_rate", 0.0)),
        transmission_mode=transmission_mode,
        network_kwargs=network_kwargs,
        beta_by_layer=beta_by_layer,
    )
    weekly = _weekly_incidence(result, len(season.ili_rate))
    _SIM_CACHE[key] = weekly.copy()
    return weekly


def _make_population(kind: str, *, N: int, seed: int, year: int) -> dict[str, np.ndarray]:
    base = _base_population(N=N, seed=seed, year=year)
    if kind == "rich_movement":
        return base
    if kind == "rich_static":
        base["work_gu"] = base["home_gu"].copy()
        return base
    if kind == "homogeneous_static":
        base["age_band"] = np.full(N, 3, dtype=np.int8)
        base["sex"] = np.zeros(N, dtype=np.int8)
        base["occupation"] = np.full(N, "office", dtype=object)
        base["severity"] = np.zeros(N, dtype=np.int8)
        base["work_gu"] = base["home_gu"].copy()
        return base
    raise ValueError(f"unknown population_kind: {kind}")


def _base_population(*, N: int, seed: int, year: int) -> dict[str, np.ndarray]:
    key = (int(N), int(seed), int(year))
    if key not in _POPULATION_CACHE:
        _POPULATION_CACHE[key] = generate_population(N, seed=seed, year=year)
    return {name: values.copy() for name, values in _POPULATION_CACHE[key].items()}


def _weekly_incidence(result: dict[str, Any], n_weeks: int) -> np.ndarray:
    cumulative = (
        np.asarray(result["E"], dtype=np.float64)
        + np.asarray(result["I"], dtype=np.float64)
        + np.asarray(result["R"], dtype=np.float64)
        + np.asarray(result["D"], dtype=np.float64)
    )
    daily = np.diff(cumulative, prepend=cumulative[0])
    daily = np.clip(daily, 0.0, None)
    return daily[: n_weeks * 7].reshape(n_weeks, 7).sum(axis=1)


def _fit_affine(sim_mean: np.ndarray, observed: np.ndarray) -> AffineMap:
    x = np.asarray(sim_mean, dtype=np.float64)
    y = np.asarray(observed, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2 or float(np.var(x[mask])) <= 1e-12:
        return AffineMap(offset=float(np.nanmean(y)), scale=0.0)
    xv = x[mask]
    yv = y[mask]
    scale = float(np.cov(xv, yv, ddof=0)[0, 1] / np.var(xv))
    if not np.isfinite(scale):
        scale = 0.0
    offset = float(np.mean(yv) - scale * np.mean(xv))
    if not np.isfinite(offset):
        offset = float(np.nanmean(y))
    return AffineMap(offset=offset, scale=scale)


def _apply_affine(replicates: np.ndarray, affine: AffineMap) -> np.ndarray:
    mapped = affine.offset + affine.scale * np.asarray(replicates, dtype=np.float64)
    return np.clip(mapped, 0.0, None)


def _score_forecast(observed: np.ndarray, mapped_replicates: np.ndarray) -> dict[str, Any]:
    y = np.asarray(observed, dtype=np.float64)
    reps = np.asarray(mapped_replicates, dtype=np.float64)
    wis_week = _wis_per_week(y, reps)
    log_week = _log_score_per_week(y, reps)
    wis_ci, log_ci = _bootstrap_metric_ci(y, reps)
    return {
        "wis": float(np.mean(wis_week)),
        "wis_ci95": [float(wis_ci[0]), float(wis_ci[1])],
        "log_score": float(np.mean(log_week)),
        "log_score_ci95": [float(log_ci[0]), float(log_ci[1])],
        "replicate_mean_ci95": _replicate_mean_ci(reps),
        "wis_per_week": wis_week,
        "log_score_per_week": log_week,
    }


def _wis_per_week(observed: np.ndarray, mapped_replicates: np.ndarray) -> np.ndarray:
    y = np.asarray(observed, dtype=np.float64)
    reps = np.asarray(mapped_replicates, dtype=np.float64)
    q = np.quantile(reps, QUANTILE_LEVELS, axis=0)
    total = 0.5 * np.abs(y - q[3])
    for lo_idx, hi_idx, alpha in INTERVAL_SPECS:
        lower = q[lo_idx]
        upper = q[hi_idx]
        total += (alpha / 2.0) * (upper - lower)
        total += np.maximum(lower - y, 0.0)
        total += np.maximum(y - upper, 0.0)
    return total / (len(INTERVAL_SPECS) + 0.5)


def _log_score_per_week(observed: np.ndarray, mapped_replicates: np.ndarray) -> np.ndarray:
    y = np.asarray(observed, dtype=np.float64)
    reps = np.asarray(mapped_replicates, dtype=np.float64)
    mean = reps.mean(axis=0)
    ddof = 1 if reps.shape[0] > 1 else 0
    sd = reps.std(axis=0, ddof=ddof)
    sd = np.maximum(sd, 1.0e-6)
    return 0.5 * np.log(2.0 * np.pi * sd * sd) + ((y - mean) ** 2) / (2.0 * sd * sd)


def _bootstrap_metric_ci(observed: np.ndarray, mapped_replicates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reps = np.asarray(mapped_replicates, dtype=np.float64)
    if reps.shape[0] < 2:
        wis = np.array([float(np.mean(_wis_per_week(observed, reps)))] * 2)
        log = np.array([float(np.mean(_log_score_per_week(observed, reps)))] * 2)
        return wis, log
    rng = np.random.default_rng(20_260_604)
    wis_values = np.empty(200, dtype=np.float64)
    log_values = np.empty(200, dtype=np.float64)
    for i in range(200):
        idx = rng.integers(0, reps.shape[0], size=reps.shape[0])
        sample = reps[idx]
        wis_values[i] = float(np.mean(_wis_per_week(observed, sample)))
        log_values[i] = float(np.mean(_log_score_per_week(observed, sample)))
    return np.quantile(wis_values, [0.025, 0.975]), np.quantile(log_values, [0.025, 0.975])


def _replicate_mean_ci(mapped_replicates: np.ndarray) -> dict[str, float]:
    reps = np.asarray(mapped_replicates, dtype=np.float64)
    mean_per_rep = reps.mean(axis=1)
    mean = float(np.mean(mean_per_rep))
    if reps.shape[0] < 2:
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean}
    half = float(stats.t.ppf(0.975, df=reps.shape[0] - 1) * stats.sem(mean_per_rep))
    return {"mean": mean, "ci95_low": mean - half, "ci95_high": mean + half}


def _comparison(
    *,
    on: dict[str, Any],
    off: dict[str, Any],
    on_label: str,
    off_label: str,
    ev: Any = None,
) -> dict[str, Any]:
    delta = float(on["metrics"]["wis"] - off["metrics"]["wis"])
    delta_log = float(on["metrics"]["log_score"] - off["metrics"]["log_score"])
    dm_t, p_value = _hln_dm(on["wis_per_week"], off["wis_per_week"], h=1)
    # G-237 fail-loud: a collapsed affine (scale≈0) makes both arms predict the
    # climatological mean, so the comparison is structurally blind to any
    # difference. Flag it instead of letting it masquerade as an honest null.
    degenerate = (
        abs(float(on["affine"]["scale"])) < 1e-9
        or abs(float(off["affine"]["scale"])) < 1e-9
    )
    result = {
        "delta_wis": delta,
        "delta_log_score": delta_log,
        "dm_t": float(dm_t),
        "hlm_p": float(p_value),
        "degenerate": bool(degenerate),
        "interpretation": _interpret(
            delta, p_value, on_label, off_label, degenerate=degenerate
        ),
        "on": _arm_public_summary(on),
        "off": _arm_public_summary(off),
    }
    # SCI-급 추가 (2026-06-05): real ILI 대비 WIS·RMSE·MAE·CRPS·coverage 전체 +
    # bootstrap CI (앙상블=mapped_replicates). DM(위 hlm_p)에 더해 효과크기+CI 제시.
    if ev is not None and not degenerate:
        try:
            from simulation.abm.behavior_disease_validation import validate_arms_calibrated
            # 관측노이즈 fold (seed-spread 앙상블의 coverage 붕괴 해결, n_agents-안정):
            #   ABM 평균 μ → NegBin(φ̂) 예측 앙상블 → calibrated WIS/coverage/AUC/C-index.
            result["sci_validation"] = validate_arms_calibrated(
                np.asarray(ev.ili_rate, dtype=np.float64),
                np.asarray(on["mapped_replicates"], dtype=np.float64).mean(axis=0),
                np.asarray(off["mapped_replicates"], dtype=np.float64).mean(axis=0),
                n_draws=500, n_boot=1000,
            )
        except Exception as _e:
            result["sci_validation"] = {"error": str(_e)}
    return result


def _hln_dm(loss_on: np.ndarray, loss_off: np.ndarray, *, h: int = 1) -> tuple[float, float]:
    d = np.asarray(loss_on, dtype=np.float64) - np.asarray(loss_off, dtype=np.float64)
    d = d[np.isfinite(d)]
    T = int(d.size)
    if T < 3:
        return 0.0, 1.0
    dbar = float(np.mean(d))
    var = float(np.var(d, ddof=0))
    for lag in range(1, h):
        if lag < T:
            cov = float(np.mean((d[lag:] - dbar) * (d[:-lag] - dbar)))
            var += 2.0 * (1.0 - lag / h) * cov
    if var <= 0.0 or not np.isfinite(var):
        return 0.0, 1.0
    dm = dbar / np.sqrt(var / T)
    correction = np.sqrt(max((T + 1 - 2 * h + h * (h - 1) / T) / T, 1.0e-12))
    dm_hln = float(dm * correction)
    p_value = 2.0 * (1.0 - stats.t.cdf(abs(dm_hln), df=T - 1))
    return dm_hln, float(np.clip(p_value, 0.0, 1.0))


def _interpret(
    delta_wis: float,
    p_value: float,
    on_label: str,
    off_label: str,
    *,
    degenerate: bool = False,
) -> str:
    if degenerate:
        return (
            "DEGENERATE: the affine map collapsed to a constant (scale≈0) for at "
            "least one arm, so both arms predict the climatological mean and the "
            "comparison cannot detect any difference — this result is INVALID. "
            "Increase n_agents until the simulated epidemic curve has non-zero "
            "variance (N=300 fizzles; N>=10,000 produces a real curve)."
        )
    if delta_wis < 0.0 and p_value < 0.05:
        return f"{on_label} beats {off_label} on held-out WIS (HLN-DM p<0.05)."
    if delta_wis < 0.0:
        return f"{on_label} has lower held-out WIS, but the HLN-DM test is not significant."
    if delta_wis > 0.0 and p_value < 0.05:
        return f"{on_label} does not beat {off_label}; held-out WIS is significantly worse."
    return f"{on_label} does not beat {off_label}; held-out WIS difference is not significant."


def _arm_public_summary(arm: dict[str, Any]) -> dict[str, Any]:
    return {
        "population_kind": arm["population_kind"],
        "behaviour": dict(arm["behaviour"]),
        "affine": dict(arm["affine"]),
        "metrics": copy.deepcopy(arm["metrics"]),
    }


def _write_json(results: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
