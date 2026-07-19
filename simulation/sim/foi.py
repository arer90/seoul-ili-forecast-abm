"""
simulation.sim.foi
==================
Force-of-infection (FoI) for the commuter-coupled metapopulation.

Derivation
----------
Every day is split into two phases: "home" and "away". During the
away phase each district i distributes its residents across all
districts j according to the row-stochastic mobility matrix M.
Infections happen at both phases, proportional to the *local mixing
pool* at each location.

Let N_j^present = Σ_i M[i, j] · N_i   (daytime population of j)
    I_j^present = Σ_i M[i, j] · I_i   (daytime infectious in j)

The per-capita FoI felt by a resident of i is then:

    λ_i = β · Σ_j  M[i, j] · I_j^present / N_j^present

This reduces to the well-mixed SIR FoI when M = diag(1)
(nobody commutes) and to the classic Tizzoni 2014 form otherwise.

The implementation is pure numpy — no scipy dependency at FoI-level.
"""
from __future__ import annotations

import numpy as np


__all__ = ["compute_foi", "effective_daytime_population"]


def effective_daytime_population(
    mobility: np.ndarray,
    populations: np.ndarray,
) -> np.ndarray:
    """Daytime population per district: N_j^present = Σ_i M[i, j] · N_i.

    Parameters
    ----------
    mobility : (G, G) row-stochastic matrix
    populations : (G,)
    """
    return mobility.T @ np.asarray(populations, dtype=float)


def compute_foi(
    I: np.ndarray,
    S: np.ndarray,
    populations: np.ndarray,
    mobility: np.ndarray,
    beta: float,
    *,
    daytime_pop: np.ndarray | None = None,
    floor: float = 1.0,
) -> np.ndarray:
    """Compute the per-capita force of infection λ_i for each district.

    Parameters
    ----------
    I : (G,) infectious counts
    S : (G,) susceptible counts — present only for signature symmetry with
        future extensions (age-stratified, heterogeneous contact) and
        unused in the standard FoI.
    populations : (G,) resident populations (used only if ``daytime_pop`` omitted)
    mobility : (G, G) row-stochastic mobility matrix
    beta : per-contact transmission coefficient (= R0 · γ for homogeneous mixing)
    daytime_pop : optional pre-computed daytime pop vector (save a matmul)
    floor : denominator floor to prevent divide-by-zero when a district
        empties out — 1.0 person is a safe choice since populations are
        measured in whole people.

    Returns
    -------
    lam : (G,) force-of-infection vector
    """
    I = np.asarray(I, dtype=float)
    if daytime_pop is None:
        daytime_pop = effective_daytime_population(mobility, populations)

    # Present infectious in each district j
    I_present = mobility.T @ I
    # Per-capita prevalence at j (clipped so empty districts don't blow up)
    prev_j = I_present / np.maximum(daytime_pop, floor)
    # FoI at home: average over destinations weighted by the residents'
    # own mobility profile.
    lam = beta * (mobility @ prev_j)
    # Numerical guard: FoI cannot be negative.
    return np.maximum(lam, 0.0)
