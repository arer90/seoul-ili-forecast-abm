"""Tests for the targeted-vs-uniform vaccination counterfactual.

The publishable claim is structural: demographic targeting separates strategies
in the heterogeneous agent model but not in the homogeneous mean-field control.
"""
from __future__ import annotations

import pathlib
import tempfile

import numpy as np

from simulation.abm import counterfactual as cf


def _small_run() -> dict:
    out = pathlib.Path(tempfile.gettempdir()) / "cf_test.json"
    # cached by (K, n_agents, budget, year, disease, behaviour) so repeated
    # calls across test functions reuse the first computation.
    return cf.run_counterfactual(K=4, n_agents=8000, budget_frac=0.10, output_path=out)


def test_heterogeneous_separates_homogeneous_flat() -> None:
    s = _small_run()["analysis"]["summary"]
    assert s["het_infection_strategy_spread"] > 5.0 * s["hom_infection_strategy_spread"] + 1e-9
    assert s["claim_supported"] is True


def test_high_contact_is_optimal_in_heterogeneous() -> None:
    het = _small_run()["analysis"]["heterogeneous"]
    # transmission-blocking: vaccinating spreaders is best for BOTH endpoints
    assert het["best_for_infections"] == "target_high_contact"


def test_homogeneous_strategies_collapse() -> None:
    hom = _small_run()["results"]["homogeneous"]
    u = hom["uniform"]["infections"]
    hc = hom["target_high_contact"]["infections"]
    el = hom["target_elderly"]["infections"]
    assert abs(u - hc) / max(u, 1.0) < 0.10
    assert abs(u - el) / max(u, 1.0) < 0.10


def test_targeted_mask_concentrates_on_group() -> None:
    from simulation.abm.epi_proof import _make_population

    pop = _make_population("rich_movement", N=5000, seed=0, year=2024)
    elderly = cf._vaccination_mask(pop, "target_elderly", 500, seed=0)
    assert int(elderly.sum()) == 500
    assert pop["age_band"][elderly].astype(float).mean() > pop["age_band"].astype(float).mean()

    none = cf._vaccination_mask(pop, "none", 500, seed=0)
    assert int(none.sum()) == 0
