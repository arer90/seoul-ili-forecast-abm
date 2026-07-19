"""Agent-world fit quality: R² + WIS, agent-count and grid refinement.

The metapopulation surrogate used for the identifiability study fits seasonal flu
only crudely (R²≈0.49); the AGENT-world (``run_agent_world`` via epi_proof's
rich-movement replicates) fits it well — R²≈0.95 / WIS 2.24 on season 2023. This
module makes that fit quality a first-class, reusable measurement and exposes the
two knobs the refinement needs:

  • the seasonal-forcing grid (β, β_amp, β_phase) — refine it when the default
    27-point grid bottoms out at an edge (e.g. the harder 2024 season);
  • the agent count N ("수") — more agents reduce stochastic ensemble noise, so
    the prediction-interval score (WIS) stabilises.

Both R² (affine-mapped point fit) AND WIS (probabilistic, multi-seed ensemble
quantiles) are reported, because a point R² alone is not a probabilistic-forecast
metric. Reuses epi_proof's validated rich-movement simulator + affine obs-map +
WIS so the numbers are comparable to the thesis. Never raises in the analysis
layer (a failed cell drops out of the grid). See PROOF_VALIDATION_PROTOCOL §4.5.
"""
from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np


def _r2_of_affine(mean: np.ndarray, ili: np.ndarray, affine) -> float:
    mapped = affine.offset + affine.scale * np.asarray(mean, dtype=np.float64)
    y = np.asarray(ili, dtype=np.float64)
    sse = float(np.sum((y - mapped) ** 2))
    var = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - sse / var if var > 0 else float("-inf")


def calibrate_agent_world(season, *, n_agents: int = 37500,
                          seeds: Sequence[int] = (1, 2, 3),
                          beta_grid=None, amp_grid=None, phase_grid=None,
                          behaviour=None) -> dict:
    """Calibrate the agent-world seasonal forcing to a real ILI season and report
    BOTH R² (affine point fit) and WIS (ensemble probabilistic score).

    Selection is by WIS (the principled probabilistic objective, as in epi_proof);
    R² is reported alongside. Grids default to a refined span centred on the
    epi_proof default (wider than the built-in 27-point ``FORCING_GRID`` so the
    optimum is not pinned to an edge). Returns ``{r2, wis, corr, n_agents,
    n_seeds, forcing, grid_size, hit_0p8}``. Never raises (failed cells skipped).
    """
    from .epi_proof import (BEHAVIOUR_OFF, DEFAULT_DISEASE, _apply_affine,
                            _fit_affine, _simulate_replicates, _wis_per_week)
    behaviour = behaviour if behaviour is not None else BEHAVIOUR_OFF
    beta_grid = beta_grid if beta_grid is not None else (0.12, 0.15, 0.18, 0.22)
    amp_grid = amp_grid if amp_grid is not None else (0.45, 0.60, 0.75, 0.90)
    phase_grid = phase_grid if phase_grid is not None else (90.0, 105.0, 120.0, 135.0)
    ili = np.asarray(season.ili_rate, dtype=np.float64)
    best, tried = None, 0
    for beta, amp, phase in itertools.product(beta_grid, amp_grid, phase_grid):
        disease = {**DEFAULT_DISEASE, "beta": float(beta),
                   "beta_amp": float(amp), "beta_phase": float(phase)}
        try:
            reps = _simulate_replicates(season, seeds=list(seeds), n_agents=int(n_agents),
                                        behaviour=behaviour, disease=disease,
                                        population_kind="rich_movement")
        except Exception:
            continue
        tried += 1
        mean = reps.mean(axis=0)
        affine = _fit_affine(mean, ili)
        wis = float(np.mean(_wis_per_week(ili, _apply_affine(reps, affine))))
        if best is None or wis < best["_wis_raw"]:
            corr = (float(np.corrcoef(mean, ili)[0, 1]) if mean.std() > 1e-12 else 0.0)
            best = {"_wis_raw": wis, "r2": round(_r2_of_affine(mean, ili, affine), 4),
                    "wis": round(wis, 3), "corr": round(corr, 4),
                    "forcing": {"beta": float(beta), "beta_amp": float(amp),
                                "beta_phase": float(phase)}}
    if best is None:
        return {"error": "no successful agent-world calibration run"}
    best.pop("_wis_raw")
    best.update({"n_agents": int(n_agents), "n_seeds": len(list(seeds)),
                 "grid_size": tried, "hit_0p8": bool(best["r2"] >= 0.8),
                 "season": int(season.season)})
    return best


def agent_count_effect(season, forcing: dict, *, agent_counts=(5000, 15000, 37500, 75000),
                       seeds: Sequence[int] = (1, 2, 3, 4, 5), behaviour=None) -> list[dict]:
    """Hold the forcing fixed and vary the agent count N ("수"); report R², WIS and
    the inter-seed ensemble noise at each N. More agents → less stochastic noise →
    a more stable probabilistic forecast. Returns one dict per N. Never raises."""
    from .epi_proof import (BEHAVIOUR_OFF, DEFAULT_DISEASE, _apply_affine,
                            _fit_affine, _simulate_replicates, _wis_per_week)
    behaviour = behaviour if behaviour is not None else BEHAVIOUR_OFF
    disease = {**DEFAULT_DISEASE, **{k: float(v) for k, v in forcing.items()}}
    ili = np.asarray(season.ili_rate, dtype=np.float64)
    out = []
    for n in agent_counts:
        try:
            reps = _simulate_replicates(season, seeds=list(seeds), n_agents=int(n),
                                        behaviour=behaviour, disease=disease,
                                        population_kind="rich_movement")
        except Exception:
            continue
        mean = reps.mean(axis=0)
        affine = _fit_affine(mean, ili)
        wis = float(np.mean(_wis_per_week(ili, _apply_affine(reps, affine))))
        # ensemble noise = mean across weeks of the inter-seed coefficient of variation
        with np.errstate(invalid="ignore", divide="ignore"):
            cv = reps.std(axis=0) / np.where(mean > 0, mean, np.nan)
        noise = float(np.nanmean(cv))
        out.append({"n_agents": int(n), "r2": round(_r2_of_affine(mean, ili, affine), 4),
                    "wis": round(wis, 3), "ensemble_noise_cv": round(noise, 4)})
    return out


def agent_world_behavioral_sensitivity(season, forcing: dict, *,
                                       n_agents: int = 15000,
                                       seeds: Sequence[int] = (1, 2, 3),
                                       behaviour_grid=None) -> dict:
    """⑩b: at a fixed (well-fitting) forcing, vary the behavioral params (α,κ,τ,θ)
    and measure how much the agent-world ILI fit moves. A tiny correlation range ⇒
    the behavioral parameters are NON-identifiable from the flu fit on the precise
    agent-world too — behavior barely affects the seasonal wave (the agent-world
    counterpart of the metapop τ-only / weak-flu-behavior finding, and consistent
    with the behaviour-OFF R²=0.95). Returns ``{n_configs, corr_min, corr_max,
    corr_range, behaviorally_identifiable, verdict}``. Never raises."""
    import itertools
    from .epi_proof import (BEHAVIOUR_GRID, BEHAVIOUR_OFF, DEFAULT_DISEASE,
                            _simulate_replicates)
    grid = behaviour_grid if behaviour_grid is not None else BEHAVIOUR_GRID
    disease = {**DEFAULT_DISEASE, **{k: float(v) for k, v in forcing.items()}}
    ili = np.asarray(season.ili_rate, dtype=np.float64)

    def _corr(beh):
        reps = _simulate_replicates(season, seeds=list(seeds), n_agents=int(n_agents),
                                    behaviour=beh, disease=disease,
                                    population_kind="rich_movement")
        m = reps.mean(axis=0)
        return float(np.corrcoef(m, ili)[0, 1]) if m.std() > 1e-12 else 0.0
    try:
        off_corr = _corr(BEHAVIOUR_OFF)
    except Exception:
        off_corr = float("nan")
    corrs, configs = [], []
    for a, k, tau, th in itertools.product(grid["alpha"], grid["kappa"],
                                           grid["tau"], grid["theta"]):
        beh = {"alpha": float(a), "kappa": float(k), "tau": float(tau), "theta": float(th)}
        try:
            reps = _simulate_replicates(season, seeds=list(seeds), n_agents=int(n_agents),
                                        behaviour=beh, disease=disease,
                                        population_kind="rich_movement")
        except Exception:
            continue
        mean = reps.mean(axis=0)
        corrs.append(float(np.corrcoef(mean, ili)[0, 1]) if mean.std() > 1e-12 else 0.0)
        configs.append(beh)
    if not corrs:
        return {"error": "no successful behavioral-grid run"}
    c = np.array(corrs)
    rng = float(c.max() - c.min())
    sensitive = rng > 0.1  # the fit must move >0.1 corr for the params to be identifiable
    best = configs[int(np.argmax(c))]
    # does ANY behavior-ON config beat behaviour-OFF? (does flu DEMAND behavior?)
    behavior_helps = np.isfinite(off_corr) and (c.max() > off_corr + 0.01)
    verdict = (
        f"{len(corrs)} (α,κ,τ,θ) configs at fixed forcing: ILI-fit corr "
        f"{c.min():.3f}–{c.max():.3f} (span {rng:.3f}); behaviour-OFF corr={off_corr:.3f}. "
        + (("⇒ the precise agent-world IS sensitive to the behavioral params "
            "(span >0.1) — identifiable-CAPABLE, unlike the metapop which was "
            "insensitive (so the agent-world is the right identifiability tool). "
            + ("But behaviour-OFF still fits best, so flu does NOT demand behavior — "
               "the behavioral mechanism is identifiable yet identified as inactive "
               "for seasonal flu; it would be identified as active in a strong-"
               "behavior (pandemic) regime."
               if not behavior_helps else
               "And some behavior-ON config beats OFF — flu shows a (weak) behavioral "
               "signal the agent-world can identify."))
           if sensitive else
           "⇒ insensitive — params non-identifiable, as on the metapop."))
    return {"n_configs": len(corrs), "corr_min": round(float(c.min()), 4),
            "corr_max": round(float(c.max()), 4), "corr_range": round(rng, 4),
            "off_corr": round(float(off_corr), 4) if np.isfinite(off_corr) else None,
            "behavior_helps_fit": bool(behavior_helps), "best_config": best,
            "behaviorally_sensitive": bool(sensitive), "verdict": verdict}


def evaluate_agent_world_full(season, forcing: dict, *, n_agents: int = 37500,
                              seeds: Sequence[int] = (1, 2, 3, 4, 5),
                              behaviour=None, threshold: float = 8.6) -> dict:
    """Full **134-metric SSOT** evaluation of the agent-world ILI fit — not just the
    5 surface metrics.

    Runs the rich-movement ensemble, affine-maps it to the ILI scale, and feeds it
    to the forecasting battery ``phase_evaluator.evaluate_predictions_full`` (the
    same SSOT the 53 forecasters use), so the agent-world and the forecasters are
    scored on ONE comparable metric set. The 5 named metrics (R²/RMSE/WIS/c-index/
    AUC-ROC) are highlighted under ``surface`` while ``metrics`` holds the whole
    battery (the rest NaN only where multi-model context is absent).

    Returns ``{n_metrics, surface, metrics}``. Never raises (failure → ``{error}``).
    The outbreak ``threshold`` (default 8.6, the KDCA ILI epidemic threshold)
    binarises ILI for the classification metrics (AUC-ROC / c-index).
    """
    from .epi_proof import (BEHAVIOUR_OFF, DEFAULT_DISEASE, _fit_affine, _simulate_replicates)
    from ..pipeline.phase_evaluator import evaluate_predictions_full
    behaviour = behaviour if behaviour is not None else BEHAVIOUR_OFF
    disease = {**DEFAULT_DISEASE, **{k: float(v) for k, v in forcing.items()}}
    ili = np.asarray(season.ili_rate, dtype=np.float64)
    try:
        reps = _simulate_replicates(season, seeds=list(seeds), n_agents=int(n_agents),
                                    behaviour=behaviour, disease=disease,
                                    population_kind="rich_movement")
        mean = reps.mean(axis=0)
        affine = _fit_affine(mean, ili)
        y_pred = affine.offset + affine.scale * mean
        metrics = evaluate_predictions_full(
            y_test=ili, y_pred=y_pred, residuals=(y_pred - ili),
            threshold=float(threshold), phase_id="agent_world")
    except Exception as exc:  # loud-ish: a proof eval reports why it failed
        return {"error": f"{type(exc).__name__}: {exc}"}
    surf = {k: (round(float(metrics[k]), 4) if metrics.get(k) is not None
                and np.isfinite(metrics.get(k, np.nan)) else metrics.get(k))
            for k in ("r2", "rmse", "wis", "c_index", "roc_auc")}
    n_finite = sum(1 for v in metrics.values()
                   if isinstance(v, (int, float)) and np.isfinite(v))
    return {"n_metrics": len(metrics), "n_finite": n_finite,
            "surface": surf, "metrics": metrics,
            "season": int(season.season), "n_agents": int(n_agents)}
