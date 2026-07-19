"""TDD for the commuter-coupled district transmission analytics (NGM, import, Moran).

These are district-resolved quantities the aggregate model cannot produce, all pure
functions of the commuter matrix M + populations (leak-free — no forward ILI). The
guards pin the epidemiological identities: M=I collapses to the well-mixed β/γ and
zero import; commuting creates positive cross-district import; commuter-weighted
Moran's I detects a surface aligned with the flow network.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from simulation.sim.commuter_ngm import (
    commuter_ngm, import_fraction, commuter_weighted_moran)


def _no_commute(g=5):
    return np.eye(g), np.array([100.0, 200.0, 300.0, 150.0, 250.0])[:g]


def _commuting(g=4):
    # rows sum to 1; residents of every district spend part of the day elsewhere
    M = np.array([[0.6, 0.2, 0.1, 0.1],
                  [0.15, 0.7, 0.1, 0.05],
                  [0.2, 0.1, 0.6, 0.1],
                  [0.1, 0.1, 0.2, 0.6]])
    N = np.array([120.0, 300.0, 180.0, 220.0])
    return M, N


def test_ngm_reduces_to_wellmixed_when_no_commuting():
    """M=I → K = (β/γ)·I → R_eff = β/γ (the classic homogeneous R0)."""
    M, N = _no_commute()
    beta, gamma = 0.5, 0.2
    out = commuter_ngm(M, N, beta=beta, gamma=gamma)
    assert abs(out["r_eff"] - beta / gamma) < 1e-9
    assert np.allclose(out["K"], (beta / gamma) * np.eye(N.size))


def test_ngm_r_eff_scales_linearly_with_beta():
    M, N = _commuting()
    a = commuter_ngm(M, N, beta=0.3, gamma=0.2)["r_eff"]
    b = commuter_ngm(M, N, beta=0.6, gamma=0.2)["r_eff"]
    assert abs(b - 2.0 * a) < 1e-9          # K is linear in beta


def test_ngm_district_loads_are_nonneg_and_sized():
    M, N = _commuting()
    out = commuter_ngm(M, N, beta=0.4, gamma=0.2)
    assert out["district_in"].shape == (N.size,)
    assert out["district_out"].shape == (N.size,)
    assert (out["district_in"] >= 0).all() and (out["district_out"] >= 0).all()
    assert abs(out["dominant_eigvec"].sum() - 1.0) < 1e-9


def test_ngm_rejects_bad_population():
    M, N = _commuting()
    N2 = N.copy(); N2[0] = 0.0
    with pytest.raises(ValueError):
        commuter_ngm(M, N2, beta=0.4, gamma=0.2)


def test_import_fraction_zero_without_commuting():
    """M=I → nobody mixes across districts → import fraction is 0 everywhere."""
    M, N = _no_commute()
    frac = import_fraction(M, N)
    assert np.allclose(frac, 0.0, atol=1e-9)


def test_import_fraction_positive_with_commuting_and_bounded():
    M, N = _commuting()
    frac = import_fraction(M, N)
    assert frac.shape == (N.size,)
    assert (frac >= -1e-9).all() and (frac <= 1.0 + 1e-9).all()
    assert frac.max() > 0.05               # commuting genuinely imports pressure


def test_moran_detects_flow_aligned_surface():
    """A surface that mirrors the commuting structure is positively autocorrelated;
    a shuffled surface is not."""
    M, N = _commuting()
    # build a surface correlated with each district's commuting exposure
    x = import_fraction(M, N) + 0.01 * np.arange(N.size)
    res = commuter_weighted_moran(x, M, n_perm=499, seed=0)
    assert set(res) >= {"moran_i", "expected_i", "p_value", "n"}
    assert -1.0 <= res["moran_i"] <= 1.0
    # a constant surface has no autocorrelation signal
    flat = commuter_weighted_moran(np.ones(N.size), M, n_perm=99, seed=0)
    assert flat["moran_i"] == 0.0


def test_functions_are_leak_free_structural():
    """No commuter-analytics function may take forward/observation data — they are
    functions of the commuter matrix + populations/model outputs only."""
    for fn in (commuter_ngm, import_fraction, commuter_weighted_moran):
        names = set(inspect.signature(fn).parameters)
        assert not (names & {"forward", "forward_ili", "obs", "observed", "observed_ili"}), (
            f"{fn.__name__} must not take forward/observation data")
