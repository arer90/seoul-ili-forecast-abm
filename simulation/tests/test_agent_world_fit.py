"""⑩: agent-world fit harness — R² + WIS, agent-count precision effect.

Uses a small synthetic season + small N so the test is fast; the real R²=0.95 /
WIS=2.17 (season 2023) and the agent-count noise sweep are demonstration runs.
"""
import numpy as np

from simulation.abm.agent_world_fit import (
    agent_count_effect,
    agent_world_behavioral_sensitivity,
    calibrate_agent_world,
    evaluate_agent_world_full,
)
from simulation.abm.epi_proof import SeasonSeries


def _season():
    weeks = np.arange(20)
    ili = np.r_[np.linspace(1.0, 40.0, 10), np.linspace(40.0, 1.0, 10)]
    return SeasonSeries(season=2023, week_seq=weeks, ili_rate=ili)


def test_calibrate_reports_r2_and_wis():
    r = calibrate_agent_world(_season(), n_agents=2000, seeds=(1,),
                              beta_grid=(0.18,), amp_grid=(0.65,), phase_grid=(120.0,))
    assert "r2" in r and "wis" in r and "corr" in r
    assert np.isfinite(r["r2"]) and r["r2"] <= 1.0
    assert r["wis"] >= 0.0
    assert isinstance(r["hit_0p8"], bool)
    assert set(r["forcing"]) == {"beta", "beta_amp", "beta_phase"}
    assert r["n_agents"] == 2000


def test_more_agents_reduce_ensemble_noise():
    forcing = {"beta": 0.18, "beta_amp": 0.65, "beta_phase": 120.0}
    sweep = agent_count_effect(_season(), forcing,
                               agent_counts=(2000, 20000), seeds=(1, 2, 3))
    assert len(sweep) == 2
    # more agents → smaller inter-seed coefficient of variation (less stochastic)
    assert sweep[1]["ensemble_noise_cv"] <= sweep[0]["ensemble_noise_cv"]
    assert all(np.isfinite(row["wis"]) for row in sweep)


def test_full_134_metric_battery_includes_the_five():
    forcing = {"beta": 0.18, "beta_amp": 0.65, "beta_phase": 120.0}
    r = evaluate_agent_world_full(_season(), forcing, n_agents=3000, seeds=(1, 2, 3))
    assert "error" not in r, r
    assert r["n_metrics"] > 50  # the full SSOT battery, not just 5
    # the 5 surface metrics the user named must all be present
    assert set(r["surface"]) == {"r2", "rmse", "wis", "c_index", "roc_auc"}
    assert np.isfinite(r["surface"]["r2"]) and np.isfinite(r["surface"]["rmse"])


def test_behavioral_sensitivity_reports_off_baseline_and_span():
    grid = {"alpha": [0.2, 0.9], "kappa": [0.3], "tau": [30.0], "theta": [0.1]}
    r = agent_world_behavioral_sensitivity(
        _season(), {"beta": 0.18, "beta_amp": 0.65, "beta_phase": 120.0},
        n_agents=2000, seeds=(1, 2), behaviour_grid=grid)
    assert "error" not in r, r
    assert r["n_configs"] == 2
    assert "off_corr" in r and "behaviorally_sensitive" in r
    assert "behavior_helps_fit" in r and "verdict" in r
    assert r["corr_range"] >= 0.0


def test_calibrate_empty_grid_returns_error():
    # a grid whose every cell fails → explicit error, not a crash
    r = calibrate_agent_world(_season(), n_agents=2000, seeds=(1,),
                              beta_grid=(), amp_grid=(0.65,), phase_grid=(120.0,))
    assert "error" in r
