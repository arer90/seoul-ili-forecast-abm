"""
simulation.abm.validate_real
============================
Validate the individual-agent ABM (:func:`simulation.abm.agent_based.run_agent_abm`)
against a REAL observed weekly ILI wave, and report the achievable error
(RMSE / MSE / MAE / R²).

This answers the reviewer/committee question "how well does the agent simulation
match reality?" — and does so honestly: the mechanistic agent SEIR-V-D is fit to
an observed epidemic wave by (i) a small behavioural-parameter grid (curve
shape), (ii) peak-timing alignment (phase), and (iii) a single amplitude scale
(reporting/ascertainment factor). The residual RMSE/MSE is the honest mismatch.

Fewer agents are used here than the 10 000-per-district headline run because the
fit re-runs the simulation across a parameter grid; the agent count is a free
knob (``n_agents``) and the result is insensitive to it above ~500/district.
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, replace

import numpy as np

from simulation.abm.agent_based import run_agent_abm
from simulation.abm.behavioural import BehaviouralParams
from simulation.sim.parameters import MetapopParams

log = logging.getLogger(__name__)

__all__ = ["AgentRealFit", "weekly_incidence", "fit_agent_to_observed"]


@dataclass
class AgentRealFit:
    """Best agent-ABM fit to an observed weekly ILI wave."""
    rmse: float
    mse: float
    mae: float
    r2: float
    scale: float            # amplitude (reporting/ascertainment) factor
    offset: float           # endemic / non-influenza ILI baseline (affine intercept)
    shift_weeks: int        # phase alignment applied to the simulated curve
    beta_mult: float        # transmission (R0) multiplier of the best fit
    params: dict            # best behavioural parameters
    n_agents: int
    total_agents: int
    observed: np.ndarray    # (W,) observed weekly ILI
    fitted: np.ndarray      # (W,) offset + scale * aligned simulated weekly curve


def weekly_incidence(agent_result) -> np.ndarray:
    """City-wide weekly incidence (sum of daily new infections per 7-day block)."""
    daily = np.asarray(agent_result.seir.incidence).sum(axis=1)  # (T+1,)
    n_weeks = len(daily) // 7
    return np.array([daily[w * 7:(w + 1) * 7].sum() for w in range(n_weeks)])


def _align_and_score(obs: np.ndarray, sim: np.ndarray, max_shift: int = 6):
    """Peak-align ``sim`` to ``obs`` (search +/- max_shift weeks) and fit an
    AFFINE map ``pred = offset + scale * sim`` by least squares.

    The ``offset`` is the endemic / non-influenza ILI baseline that a single
    influenza wave does not model (ILI = background + influenza-attributable);
    ``scale`` is the reporting/ascertainment amplitude. Returns
    ``(rmse, scale, offset, shift, pred)`` for the best alignment.
    """
    W = len(obs)
    best = None
    op = int(np.argmax(obs))
    sp = int(np.argmax(sim))
    base = op - sp
    for shift in range(base - max_shift, base + max_shift + 1):
        s = np.zeros(W)
        src0, dst0 = max(0, -shift), max(0, shift)
        n = min(W - dst0, len(sim) - src0)
        if n <= 3:
            continue
        s[dst0:dst0 + n] = sim[src0:src0 + n]
        # require a non-degenerate epidemic curve (variance in s), else the
        # affine fit collapses to a flat offset≈mean — a spurious low-variance
        # "fit" with R²≈0 that must not win.
        if not np.all(np.isfinite(s)) or float(np.var(s)) <= 1e-12:
            continue
        # affine least squares: obs ~ offset + scale * s
        A = np.vstack([np.ones(W), s]).T
        coef, *_ = np.linalg.lstsq(A, obs, rcond=None)
        offset, scale = float(coef[0]), float(coef[1])
        if not (np.isfinite(offset) and np.isfinite(scale)) or scale <= 0:
            continue
        pred = offset + scale * s
        if not np.all(np.isfinite(pred)):
            continue
        rmse = float(np.sqrt(np.mean((pred - obs) ** 2)))
        if np.isfinite(rmse) and (best is None or rmse < best[0]):
            best = (rmse, scale, offset, shift, pred)
    return best


def fit_agent_to_observed(
    observed_weekly,
    metapop_params: MetapopParams,
    *,
    n_agents: int = 1_000,
    theta_sd: float = 0.25,
    seed: int = 42,
    alpha_grid=(1.0, 2.0, 3.0),
    kappa_grid=(0.1, 0.2, 0.3),
    tau_grid=(60.0, 90.0, 120.0),
    theta_grid=(0.05, 0.10, 0.15),
    beta_mult_grid=(0.8, 0.9, 1.0),  # R0 in the influenza range (1.1-1.4) and below
                                     # the SEIR overflow regime ⇒ a deterministic fit
    gamma_mult_grid=(0.8, 1.0, 1.3),  # 1/gamma = infectious period ⇒ wave width/sharpness
    max_shift: int = 8,
    compliance_steepness: float = float("inf"),
) -> AgentRealFit:
    """Fit the agent ABM to an observed weekly ILI wave; report error metrics.

    Args:
        observed_weekly: (W,) observed weekly ILI series (a single wave).
        metapop_params: kernel inputs (days is overridden to W*7).
        n_agents: agents per district for the fit (reduced for grid speed).
        compliance_steepness: +inf (default) = hard threshold; a finite value
            (e.g. 30) uses the smooth logistic compliance, which suppresses the
            FP knife-edge and so reduces grid points lost to the SEIR blow-up.

    Returns:
        AgentRealFit (rmse, mse, mae, r2, scale, shift, best params, curves).

    Performance: O(|grid| * W*7 * G * n_agents). Side effects: none.
    """
    obs = np.asarray(observed_weekly, dtype=float)
    obs = np.nan_to_num(obs, nan=float(np.nanmean(obs)))
    W = len(obs)
    mp0 = replace(metapop_params, days=W * 7)
    base_R0 = float(mp0.disease.R0)  # beta = R0 * gamma is a derived property
    base_gamma = float(mp0.disease.gamma)

    best_fit = None
    for bm, gm in itertools.product(beta_mult_grid, gamma_mult_grid):
        # vary R0 (epidemic intensity) and gamma (1/infectious-period ⇒ wave
        # width/sharpness) — the two shape levers for matching a real wave
        disease_b = replace(mp0.disease, R0=base_R0 * float(bm), gamma=base_gamma * float(gm))
        mp = replace(mp0, disease=disease_b)
        for a, k, tau, th in itertools.product(alpha_grid, kappa_grid, tau_grid, theta_grid):
            behav = BehaviouralParams(alpha=a, kappa=k, tau=tau, theta=th,
                                      compliance_steepness=compliance_steepness)
            res = run_agent_abm(mp, behav, n_agents=n_agents, theta_sd=theta_sd, seed=seed)
            sim = weekly_incidence(res)
            # skip non-finite / overflowing (high-R0 SEIR blow-up) — these would
            # make the grid selection non-deterministic; the degenerate flat case
            # (no epidemic ⇒ var≈0) is rejected inside _align_and_score.
            if not np.all(np.isfinite(sim)) or sim.sum() <= 0 or sim.max() > 1e12:
                continue
            scored = _align_and_score(obs, sim, max_shift=max_shift)
            if scored is None:
                continue
            rmse, scale, offset, shift, pred = scored
            if best_fit is None or rmse < best_fit.rmse:
                mse = float(np.mean((pred - obs) ** 2))
                mae = float(np.mean(np.abs(pred - obs)))
                ss = float(np.sum((obs - obs.mean()) ** 2))
                r2 = 1.0 - float(np.sum((pred - obs) ** 2)) / ss if ss > 0 else float("nan")
                best_fit = AgentRealFit(
                    rmse=rmse, mse=mse, mae=mae, r2=r2, scale=scale, offset=offset,
                    shift_weeks=int(shift), beta_mult=float(bm),
                    params=dict(alpha=a, kappa=k, tau=tau, theta=th, gamma_mult=float(gm)),
                    n_agents=int(n_agents), total_agents=int(n_agents) * int(mp.populations.size),
                    observed=obs, fitted=pred,
                )
    if best_fit is None:
        raise RuntimeError("no agent-ABM grid point produced a usable epidemic curve")
    return best_fit
