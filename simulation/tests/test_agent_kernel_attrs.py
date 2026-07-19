"""Attribute-driven dynamics tests for the NumPy agent kernel."""
from __future__ import annotations

import numpy as np

from simulation.abm.agent_kernel import run_agent_world
from simulation.abm.synthetic_population import generate_population


N = 2_000
T_DAYS = 60
SEED = 42
PARAMS = {
    "N": N,
    "T_days": T_DAYS,
    "beta": 0.75,
    "sigma": 0.35,
    "gamma": 0.12,
    "delta": 0.08,
    "nu": 0.0,
    "global_seed": SEED,
    "theta_mean": 10.0,
    "theta_sd": 1e-6,
    "alpha_mean": 0.0,
    "kappa_mean": 0.0,
    "tau_mean": 1e9,
}


def _population() -> dict[str, np.ndarray]:
    return generate_population(N, seed=SEED)


def _copy_population(population: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: value.copy() for key, value in population.items()}


def _total(result: dict) -> np.ndarray:
    return result["S"] + result["E"] + result["I"] + result["R"] + result["V"] + result["D"]


def _cumulative_infections(result: dict) -> int:
    return int(result["E"][-1] + result["I"][-1] + result["R"][-1] + result["D"][-1])


def _assert_no_nan_inf(value) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _assert_no_nan_inf(child)
        return
    if isinstance(value, np.ndarray):
        assert np.all(np.isfinite(value.astype(np.float64)))


def test_mass_conservation_with_attrs() -> None:
    result = run_agent_world(**PARAMS, population=_population())
    assert result["attr_dynamics_active"] is True
    np.testing.assert_array_equal(_total(result), np.full(T_DAYS, N))


def test_severity_matters() -> None:
    base = _population()
    high = _copy_population(base)
    low = _copy_population(base)
    high["severity"] = np.ones(N, dtype=np.int8)
    low["severity"] = np.zeros(N, dtype=np.int8)

    high_result = run_agent_world(**PARAMS, population=high)
    low_result = run_agent_world(**PARAMS, population=low)

    high_deaths = int(high_result["D"][-1])
    low_deaths = int(low_result["D"][-1])
    ratio = high_deaths / max(low_deaths, 1)
    print(f"EFFECT severity_death_ratio={ratio:.3f}")
    assert high_deaths > low_deaths * 1.5


def test_scheduled_movement_matters() -> None:
    movement = _population()
    no_commute = _copy_population(movement)
    no_commute["work_gu"] = no_commute["home_gu"].copy()

    movement_result = run_agent_world(**PARAMS, population=movement)
    no_commute_result = run_agent_world(**PARAMS, population=no_commute)

    peak_movement = float(np.max(movement_result["I"]))
    peak_no_commute = float(np.max(no_commute_result["I"]))
    peak_diff = abs(peak_movement - peak_no_commute) / max(peak_no_commute, 1.0)
    trajectory_diff = float(np.max(np.abs(movement_result["I"] - no_commute_result["I"])))
    trajectory_diff /= max(peak_no_commute, 1.0)
    diff = max(peak_diff, trajectory_diff)
    print(f"EFFECT movement_trajectory_diff_pct={100.0 * diff:.2f}")
    assert diff >= 0.05


def test_occupation_attack_rate() -> None:
    base = _population()
    service = _copy_population(base)
    unemployed = _copy_population(base)
    service["occupation"] = np.full(N, "service", dtype=object)
    unemployed["occupation"] = np.full(N, "unemployed", dtype=object)

    service_result = run_agent_world(**PARAMS, population=service)
    unemployed_result = run_agent_world(**PARAMS, population=unemployed)

    service_infections = _cumulative_infections(service_result)
    unemployed_infections = _cumulative_infections(unemployed_result)
    ratio = service_infections / max(unemployed_infections, 1)
    print(f"EFFECT occupation_attack_rate_ratio={ratio:.3f}")
    assert service_infections > unemployed_infections


def test_no_nan_inf() -> None:
    result = run_agent_world(**PARAMS, population=_population())
    _assert_no_nan_inf(result)


def test_determinism_with_attrs() -> None:
    population = _population()
    first = run_agent_world(**PARAMS, population=population)
    second = run_agent_world(**PARAMS, population=population)
    for key in "SEIRVD":
        np.testing.assert_array_equal(first[key], second[key])
    for key, value in first["agents"].items():
        np.testing.assert_array_equal(value, second["agents"][key])


def test_back_compat_no_population() -> None:
    without_kwarg = run_agent_world(**PARAMS)
    explicit_none = run_agent_world(**PARAMS, population=None)
    assert without_kwarg["attr_dynamics_active"] is False
    assert explicit_none["attr_dynamics_active"] is False
    for key in "SEIRVD":
        np.testing.assert_array_equal(without_kwarg[key], explicit_none[key])
    for key, value in without_kwarg["agents"].items():
        np.testing.assert_array_equal(value, explicit_none["agents"][key])
