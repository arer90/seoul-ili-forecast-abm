"""Behavioural ABM (`simulation.abm.behavioural`) unit tests.

Closes a real coverage gap: before this file the four-parameter behavioural
ABM had **no tests**, which is why a Numba-path bug (`scale_district`
``UnboundLocalError`` at the city-mean beta step) shipped undetected — the
non-Numba path defined ``scale_district`` but the Numba path did not, and no
test exercised the Numba path or compared the two.

Covers:
  - Numba vs non-Numba path EQUIVALENCE (the regression guard for that bug class)
  - Numba path runs without crashing and returns finite, valid output
  - behaviour-off parameters reproduce the kernel-only baseline (beta == beta_0)
  - compliance is a {0,1} indicator and effective beta stays non-negative

Synthetic MetapopParams only (no DB) so CI runs in seconds.
"""
from __future__ import annotations

import numpy as np
import pytest

import simulation.abm.behavioural as bh
from simulation.abm.behavioural import BehaviouralParams, run_coupled_abm
from simulation.sim.parameters import DEFAULT_FLU_PARAMS, MetapopParams


def _toy_params(G=6, days=40, dt=0.25, infected0=500.0):
    pops = np.full(G, 100_000.0)
    M = np.full((G, G), 0.05 / (G - 1))
    np.fill_diagonal(M, 0.95)
    I0 = np.full(G, infected0)
    return MetapopParams(
        disease=DEFAULT_FLU_PARAMS, populations=pops, mobility=M,
        district_names=[f"d{i}" for i in range(G)],
        initial_infected=I0, days=days, dt=dt, seed=0,
    )


REBOUND = BehaviouralParams(alpha=2.0, kappa=0.3, tau=90.0, theta=0.1)


def _run_with_numba_flag(params, behav, flag):
    orig = bh._HAS_NUMBA
    try:
        bh._HAS_NUMBA = flag
        return run_coupled_abm(params, behav)
    finally:
        bh._HAS_NUMBA = orig


def test_numba_path_runs_and_is_finite():
    """Direct regression for the scale_district UnboundLocalError: the Numba
    path must run end-to-end and return finite trajectories."""
    if not bh._HAS_NUMBA:
        pytest.skip("numba not installed")
    out = _run_with_numba_flag(_toy_params(), REBOUND, True)
    assert np.all(np.isfinite(out.city_I()))
    assert np.all(np.isfinite(out.beta_eff))
    assert out.city_I().max() > 0


def test_numba_nonnumba_equivalence():
    """The Numba and pure-numpy paths must produce identical trajectories
    (the equivalence the shipped bug silently broke)."""
    if not bh._HAS_NUMBA:
        pytest.skip("numba not installed — only one path available")
    params = _toy_params()
    a = _run_with_numba_flag(params, REBOUND, True)
    b = _run_with_numba_flag(params, REBOUND, False)
    for name in ("city_I",):
        assert np.allclose(getattr(a, name)(), getattr(b, name)(), atol=1e-9, rtol=0)
    for attr in ("compliance", "beta_eff", "risk", "fatigue"):
        assert np.allclose(getattr(a, attr), getattr(b, attr), atol=1e-9, rtol=0), attr


def test_behaviour_off_reproduces_kernel_baseline():
    """alpha=kappa=0, tau->inf => no behavioural coupling => beta_eff == beta_0."""
    off = BehaviouralParams(alpha=0.0, kappa=0.0, tau=float("inf"))
    assert off.is_behaviour_off()
    out = run_coupled_abm(_toy_params(), off)
    beta_0 = out.behaviour.__class__  # noqa: F841  (sanity)
    assert np.allclose(out.beta_eff, DEFAULT_FLU_PARAMS.beta, atol=1e-12)
    assert np.allclose(out.compliance, 0.0, atol=1e-12)


def test_compliance_indicator_and_beta_nonneg():
    """Compliance is a {0,1} indicator; effective beta stays non-negative."""
    out = run_coupled_abm(_toy_params(), REBOUND)
    uniq = np.unique(out.compliance)
    assert set(np.round(uniq, 9)).issubset({0.0, 1.0})
    assert np.all(out.beta_eff >= 0.0)


def test_behavioural_params_validate():
    """BehaviouralParams.validate accepts valid sets (incl. behaviour-off) and
    rejects out-of-range tau/alpha/kappa/theta/strength."""
    REBOUND.validate()  # valid
    BehaviouralParams(alpha=0.0, kappa=0.0, tau=float("inf")).validate()  # behaviour-off
    for bad in (dict(alpha=-1.0), dict(kappa=-0.1), dict(theta=-0.1),
                dict(tau=0.0), dict(tau=-5.0), dict(strength=1.5), dict(strength=-0.1)):
        with pytest.raises(ValueError):
            BehaviouralParams(**bad).validate()


def test_metapop_validate_rejects_nonfinite_inputs():
    """MetapopParams.validate fails fast on non-finite / non-positive inputs
    instead of letting NaN/Inf reach the FoI matmul (overflow / abort)."""
    from dataclasses import replace
    base = _toy_params()
    pops = np.asarray(base.populations, dtype=float).copy()
    pops[0] = np.nan
    with pytest.raises(ValueError):
        replace(base, populations=pops).validate()
    pops2 = np.asarray(base.populations, dtype=float).copy()
    pops2[0] = -1.0
    with pytest.raises(ValueError):
        replace(base, populations=pops2).validate()
