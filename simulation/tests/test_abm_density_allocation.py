"""Regression guard for density-proportional spatial agent allocation (G-388).

Covers the surgical ``run_agent_abm`` per-district extension (uniform path must
stay byte-identical; heterogeneous path must mask padded agents) and the pure
allocation / validation helpers in ``simulation.abm.density_allocation``.

Pure compute + one read-only DB read; no model is retrained, no DB is mutated.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from simulation.abm.agent_based import run_agent_abm
from simulation.abm.behavioural import BehaviouralParams
from simulation.abm.density_allocation import (
    allocate_agents_by_density,
    spearman_rho,
)
from simulation.sim.parameters import DEFAULT_FLU_PARAMS, MetapopParams


def _toy(G: int = 4, days: int = 30) -> MetapopParams:
    pops = np.array([100000.0, 200000.0, 150000.0, 300000.0])[:G]
    M = np.eye(G) * 0.9 + (0.1 / (G - 1)) * (1 - np.eye(G))
    I0 = np.zeros(G)
    I0[0] = 20.0
    return MetapopParams(
        disease=DEFAULT_FLU_PARAMS, populations=pops, mobility=M,
        district_names=[f"g{i}" for i in range(G)], initial_infected=I0,
        days=days, dt=0.25, seed=42,
    )


# ── run_agent_abm: uniform path unchanged ────────────────────────────────
def test_uniform_array_matches_scalar_byte_identical():
    """n_agents_per_district = [k,k,k,k] must reproduce scalar n_agents=k exactly."""
    mp = _toy()
    b = BehaviouralParams(alpha=2.0, kappa=0.2, tau=90.0, theta=0.1)
    r_scalar = run_agent_abm(mp, b, n_agents=1500, theta_sd=0.25, seed=7)
    r_array = run_agent_abm(
        mp, b, n_agents_per_district=np.full(4, 1500), theta_sd=0.25, seed=7
    )
    assert np.array_equal(r_scalar.city_I(), r_array.city_I())
    assert np.array_equal(r_scalar.compliance_fraction, r_array.compliance_fraction)
    assert r_scalar.total_agents == r_array.total_agents == 6000


# ── run_agent_abm: heterogeneous path ────────────────────────────────────
def test_heterogeneous_allocation_valid_and_masked():
    mp = _toy()
    b = BehaviouralParams(alpha=2.0, kappa=0.2, tau=90.0, theta=0.1)
    counts = np.array([300, 800, 1500, 3000])
    r = run_agent_abm(mp, b, n_agents_per_district=counts, theta_sd=0.25, seed=7)
    assert r.total_agents == int(counts.sum())
    assert r.n_agents == int(counts.max())            # n_max
    assert np.array_equal(r.n_agents_per_district, counts)
    cf = r.compliance_fraction
    assert np.all(np.isfinite(cf)) and cf.min() >= 0.0 and cf.max() <= 1.0
    assert np.all(np.isfinite(r.city_I())) and r.city_I().min() >= 0.0


def test_padding_does_not_dilute_compliance():
    """A district with FEW agents must not get its compliance pulled toward the
    padded (inactive) columns — masked mean is over active agents only."""
    mp = _toy()
    # theta_sd=0 ⇒ deterministic threshold; with the SAME behaviour the realised
    # compliance fraction of a district depends only on its R_d trajectory, not on
    # how many padding columns sit behind it. So a small district run alongside a
    # large one must give the same compliance as if it were sized uniformly small.
    b = BehaviouralParams(alpha=3.0, kappa=0.1, tau=90.0, theta=0.05)
    hetero = run_agent_abm(
        mp, b, n_agents_per_district=np.array([50, 50, 50, 4000]),
        theta_sd=0.0, seed=3,
    )
    uniform_small = run_agent_abm(mp, b, n_agents=50, theta_sd=0.0, seed=3)
    # districts 0..2 have the same agent count (50) in both runs; with theta_sd=0
    # the compliance fraction for those districts must match (padding-independent).
    assert np.allclose(
        hetero.compliance_fraction[:, :3],
        uniform_small.compliance_fraction[:, :3],
        atol=1e-12,
    )


def test_zero_count_rejected():
    mp = _toy()
    b = BehaviouralParams(alpha=2.0, kappa=0.2, tau=90.0, theta=0.1)
    with pytest.raises(ValueError):
        run_agent_abm(mp, b, n_agents_per_district=np.array([0, 1, 2, 3]))
    with pytest.raises(ValueError):
        run_agent_abm(mp, b, n_agents_per_district=np.array([1, 2, 3]))  # wrong len


# ── allocate_agents_by_density ───────────────────────────────────────────
def test_allocation_sums_exactly_and_respects_floor():
    d = np.array([1000.0, 4000.0, 2000.0, 3000.0, 250.0])
    out = allocate_agents_by_density(d, 100_000, floor=100)
    assert out.sum() == 100_000
    assert out.min() >= 100
    # monotone: higher density ⇒ at least as many agents
    order = np.argsort(d)
    assert np.all(np.diff(out[order]) >= 0)


def test_allocation_proportionality():
    d = np.array([1000.0, 2000.0])           # 2:1 density
    out = allocate_agents_by_density(d, 30_000, floor=0 + 1)
    # ratio should be close to 2:1 (floor of 1 is negligible at this budget)
    assert abs(out[1] / out[0] - 2.0) < 0.01


def test_allocation_budget_too_small_raises():
    d = np.ones(25)
    with pytest.raises(ValueError):
        allocate_agents_by_density(d, 100, floor=100)  # 25*100 > 100


def test_allocation_zero_density_uniform_fallback():
    d = np.zeros(4)
    out = allocate_agents_by_density(d, 4000, floor=100)
    assert out.sum() == 4000
    assert out.max() - out.min() <= 1   # ~uniform


# ── spearman_rho ─────────────────────────────────────────────────────────
def test_spearman_basic():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert spearman_rho(x, x) == pytest.approx(1.0)
    assert spearman_rho(x, -x) == pytest.approx(-1.0)


def test_spearman_ties_and_degenerate():
    assert np.isnan(spearman_rho(np.ones(5), np.arange(5.0)))  # zero variance
    # monotone-with-ties stays high
    x = np.array([1.0, 1.0, 2.0, 3.0, 3.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert spearman_rho(x, y) > 0.8
