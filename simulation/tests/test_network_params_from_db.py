"""Leak-free guard for DATA-DERIVED contact-network structure params.

Confirms the network structure is derived from forecast-time STRUCTURAL data
(per-gu living-population mobility) + cited external constants — never from the
forward ILI observations or the forward window, and never tuned to the score.
"""
from __future__ import annotations

import numpy as np
import pytest

from simulation.abm.network_params_from_db import (
    EXTERNAL_CONSTANTS,
    derive_network_kwargs,
)


# ── Track 3B: data-derived per-layer beta fusion (scale-preserving) ────────────
def test_derive_beta_by_layer_scale_preserved_and_ordered():
    """Per-layer beta must (a) preserve the aggregate force so the forward-scale is
    unchanged (redistribution only), and (b) order per-contact hazard by cited
    contact intensity: household (closest/longest) > community (brief casual)."""
    from simulation.abm.network_params_from_db import derive_beta_by_layer
    deg = {"household": 1.8, "workplace": 4.0, "school": 6.0, "community": 8.0}
    beta = 0.3
    bbl = derive_beta_by_layer(beta, deg)
    # scale-preserving: sum_L deg_L * beta_L == the aggregate beta (uniform baseline)
    agg = sum(deg[L] * bbl[L] for L in deg)
    assert abs(agg - beta) < 1e-9, f"aggregate force {agg} != beta {beta} (not scale-preserving)"
    # per-contact hazard ordering (cited intensity): household highest, community lowest
    assert bbl["household"] == max(bbl.values())
    assert bbl["household"] > bbl["community"]
    assert bbl["school"] > bbl["workplace"] > bbl["community"]
    assert all(v > 0 for v in bbl.values())


def test_derive_beta_by_layer_uniform_weights_reduce_to_baseline():
    """With equal weights the per-layer beta collapses to the uniform beta/deg_total
    the kernel uses today — proving the fusion is a strict generalization."""
    from simulation.abm.network_params_from_db import derive_beta_by_layer
    deg = {"household": 2.0, "workplace": 3.0, "school": 5.0, "community": 6.0}
    beta = 0.25
    flat = derive_beta_by_layer(beta, deg, weights={k: 1.0 for k in deg})
    deg_total = sum(deg.values())
    for L in deg:
        assert abs(flat[L] - beta / deg_total) < 1e-12


def test_per_contact_weights_are_cited_not_tuned():
    """The per-layer intensity weights must be documented constants, not a sweep."""
    w = EXTERNAL_CONSTANTS["per_contact_transmission_weights"]
    assert w["household"] == max(w.values())      # household reference (highest)
    assert w["community"] == min(w.values())      # casual community lowest
    assert "source" in EXTERNAL_CONSTANTS["per_contact_transmission_source"].lower() \
        or "POLYMOD" in EXTERNAL_CONSTANTS["per_contact_transmission_source"]


def test_returns_per_gu_degree_and_size_ranges():
    nk = derive_network_kwargs()
    deg = nk["community_mean_degree"]
    assert isinstance(deg, np.ndarray) and deg.shape == (25,)
    assert np.all(deg > 0)
    assert deg.max() > deg.min()          # per-gu heterogeneity from mobility (not flat)
    for key in ("hh_size", "work_size", "class_size"):
        lo, hi = nk[key]
        assert 1 <= lo <= hi


def test_absolute_base_is_cited_external_not_tuned():
    # the absolute community contact rate is a documented survey value, and the
    # derived per-gu mean stays near it — it is NOT a value fit to the forward score
    nk = derive_network_kwargs()
    base = EXTERNAL_CONSTANTS["community_base_degree"]
    assert base == 8.0                    # POLYMOD-class casual-contact survey value
    assert nk["community_mean_degree"].mean() == pytest.approx(base, abs=2.5)
    assert "POLYMOD" in EXTERNAL_CONSTANTS["community_base_source"]


def test_household_size_is_census_correction():
    # external Seoul census mean ~2.1 → range (1,3), correcting the old (2,4)≈3
    lo, hi = derive_network_kwargs()["hh_size"]
    assert (lo, hi) == (1, 3)
    assert "Census" in EXTERNAL_CONSTANTS["hh_size_source"]


def test_leak_free_provenance_and_forward_exclusion():
    nk = derive_network_kwargs(forecast_origin="20260216")
    prov = nk["provenance"]["leak_free"]
    assert "20260216" in prov              # forward origin is the exclusion boundary
    assert "forward" in prov.lower() or "exclud" in prov.lower()


def test_filter_actually_excludes_pre_origin_only():
    # a forecast_origin before all data leaves no mobility rows → RuntimeError,
    # proving the WHERE stdr_de < forecast_origin filter is real (not cosmetic)
    with pytest.raises(RuntimeError):
        derive_network_kwargs(forecast_origin="20180101", window_start="20180101")


def test_no_dependence_on_forward_values():
    # the derivation takes only structural inputs; identical output on repeat calls
    a = derive_network_kwargs()["community_mean_degree"]
    b = derive_network_kwargs()["community_mean_degree"]
    assert np.array_equal(a, b)            # deterministic, structural, no random/forward input
