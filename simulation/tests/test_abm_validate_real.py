"""Agent-ABM real-data validation (`simulation.abm.validate_real`) unit tests.

Guards the fit-to-observed machinery that reports how well the agent simulation
reproduces a real ILI wave (RMSE / MSE / MAE / R²).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from simulation.abm.agent_based import run_agent_abm
from simulation.abm.behavioural import BehaviouralParams
from simulation.abm.validate_real import (
    fit_agent_to_observed, weekly_incidence,
)
from simulation.sim.parameters import DEFAULT_FLU_PARAMS, MetapopParams


def _toy_params(G=4):
    pops = np.full(G, 200_000.0)
    M = np.full((G, G), 0.05 / (G - 1))
    np.fill_diagonal(M, 0.95)
    return MetapopParams(
        disease=DEFAULT_FLU_PARAMS, populations=pops, mobility=M,
        district_names=[f"d{i}" for i in range(G)],
        initial_infected=np.full(G, 400.0), days=10, dt=0.25, seed=0,
    )


def test_weekly_incidence_shape():
    p = replace(_toy_params(), days=21)
    res = run_agent_abm(p, BehaviouralParams(alpha=2.0, kappa=0.2, tau=90.0, theta=0.1),
                        n_agents=300, seed=1)
    wk = weekly_incidence(res)
    assert wk.ndim == 1 and len(wk) == 3 and np.all(wk >= 0)


def test_fit_recovers_self_generated_wave():
    """Fitting the agent ABM to a curve the agent ABM itself generated (same
    behavioural params) recovers it with high R² — the fit machinery works."""
    G = 4
    p = _toy_params(G)
    truth = BehaviouralParams(alpha=2.0, kappa=0.2, tau=90.0, theta=0.1)
    obs_res = run_agent_abm(replace(p, days=18 * 7), truth, n_agents=600, seed=1)
    obs = weekly_incidence(obs_res) * 1e-3  # a 'rate-like' rescale
    fit = fit_agent_to_observed(
        obs, p, n_agents=600, seed=1,
        alpha_grid=(2.0,), kappa_grid=(0.2,), tau_grid=(90.0,), theta_grid=(0.1,),
    )
    assert np.isfinite(fit.rmse) and np.isfinite(fit.r2)
    assert fit.r2 > 0.8          # same shape ⇒ near-perfect after scale+align
    assert fit.total_agents == 600 * G
    assert fit.scale > 0


def test_fit_metrics_finite_on_arbitrary_wave():
    p = _toy_params()
    obs = np.array([5, 8, 14, 28, 45, 60, 52, 33, 20, 12, 8, 6], dtype=float)
    fit = fit_agent_to_observed(obs, p, n_agents=400, seed=2,
                                alpha_grid=(1.0, 2.0), tau_grid=(60.0, 90.0))
    for v in (fit.rmse, fit.mse, fit.mae, fit.r2):
        assert np.isfinite(v)
    assert fit.mse == pytest.approx(fit.rmse ** 2, rel=1e-6)
