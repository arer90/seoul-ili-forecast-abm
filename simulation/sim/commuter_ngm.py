"""simulation.sim.commuter_ngm
================================
Commuter-coupled DISTRICT-RESOLVED transmission analytics — the spatial title
clause ("Commuter-Coupled District Transmission Modeling for Seoul's 25 Districts").

The metapopulation FoI (:mod:`simulation.sim.foi`) couples the 25 districts through
the row-stochastic commuter matrix ``M`` (KOSIS OD data): a resident of district *i*
feels ``λ_i = β · Σ_j M[i,j] · I_j^present / N_j^present`` where the daytime pools are
``I_j^present = Σ_k M[k,j] I_k`` and ``N_j^present = Σ_k M[k,j] N_k``. This module
turns that coupling into district-resolved, leak-free quantities that an aggregate
(single-Seoul) model structurally cannot produce:

  * **Next-Generation Matrix (NGM)** ``K`` and the system reproduction number
    ``R_eff = ρ(K)`` (spectral radius), plus per-district *target* (row) and *source*
    (column) reproduction loads — which districts import vs seed transmission.
  * **Import fraction** per district: the share of a district's infection pressure that
    comes from OTHER districts' residents via commuting (0 = self-contained, →1 =
    commuter-driven). A pure function of ``M`` and ``N``.
  * **Commuter-weighted Moran's I**: spatial autocorrelation of a district surface
    using the commuter flows as the spatial weight — a convergent-validity check that
    the model's district burden is coherent with the real commuting network.

Leak-free: everything is a function of the commuter matrix ``M`` (pre-forecast KOSIS
OD data) and district populations / model outputs — no forward ILI is read.
"""
from __future__ import annotations

import numpy as np

from simulation.sim.foi import effective_daytime_population

__all__ = ["commuter_ngm", "import_fraction", "commuter_weighted_moran"]


def commuter_ngm(mobility: np.ndarray, populations: np.ndarray, *, beta: float,
                 gamma: float, floor: float = 1.0) -> dict:
    """Next-generation matrix of the commuter-coupled metapopulation at the
    disease-free equilibrium, and its district-resolved reproduction structure.

    ``K[i,k]`` = expected secondary infections in district *i* caused by one
    infectious resident of district *k* over their infectious period ``1/γ``:

        K = (β/γ) · diag(N) · M · diag(1/N^present) · Mᵀ

    (derived by linearising ``λ_i S_i`` at ``S_i = N_i``). The system reproduction
    number is the spectral radius ``R_eff = ρ(K)``; with ``M = I`` (no commuting) this
    reduces to the well-mixed ``β/γ``.

    Args:
        mobility: ``(G, G)`` row-stochastic commuter matrix ``M`` (``M[i,j]`` = share
            of district-*i* residents present in *j* by day).
        populations: ``(G,)`` resident (night) populations ``N``; must be > 0.
        beta: transmission coefficient (``= R0·γ`` under homogeneous mixing).
        gamma: recovery rate (infectious period ``1/γ``).
        floor: daytime-population denominator floor (empty-district guard).

    Returns:
        ``{"K": (G,G), "r_eff": float, "district_in": (G,), "district_out": (G,),
        "dominant_eigvec": (G,)}`` — ``district_in`` = row sums (infections a district
        receives per infectious), ``district_out`` = column sums (infections one of its
        residents generates city-wide), ``dominant_eigvec`` = the normalised right
        eigenvector for ``R_eff`` (the district reproduction profile, entries ≥ 0).

    Raises:
        ValueError: if shapes mismatch or a population is ≤ 0.

    Performance: O(G³) (eigydecomposition of a 25×25 matrix — microseconds).
    Side effects: none (pure).
    """
    M = np.asarray(mobility, dtype=float)
    N = np.asarray(populations, dtype=float)
    g = N.size
    if M.shape != (g, g):
        raise ValueError(f"mobility {M.shape} incompatible with populations {N.shape}")
    if np.any(N <= 0):
        raise ValueError("populations must be strictly positive")
    n_present = np.maximum(effective_daytime_population(M, N), floor)
    # K = (β/γ) diag(N) M diag(1/n_present) Mᵀ. n_present ≥ floor > 0 makes every term
    # finite; errstate silences a spurious BLAS matmul warning on some platforms.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        K = (beta / gamma) * (N[:, None] * (M * (1.0 / n_present)[None, :]) @ M.T)
    if not np.all(np.isfinite(K)):
        raise ValueError("non-finite NGM — check mobility/populations")
    eigvals, eigvecs = np.linalg.eig(K)
    k = int(np.argmax(eigvals.real))
    r_eff = float(eigvals.real[k])
    vec = np.abs(eigvecs[:, k].real)
    vec = vec / vec.sum() if vec.sum() > 0 else vec
    return {
        "K": K,
        "r_eff": r_eff,
        "district_in": K.sum(axis=1),      # infections district i receives per infectious
        "district_out": K.sum(axis=0),     # infections one resident of k seeds city-wide
        "dominant_eigvec": vec,
    }


def import_fraction(mobility: np.ndarray, populations: np.ndarray,
                    infectious: np.ndarray | None = None, *, floor: float = 1.0) -> np.ndarray:
    """Per-district share of infection pressure coming from OTHER districts' residents.

    Of the FoI ``λ_i = β Σ_j M[i,j] I_j^present / N_j^present`` felt by district *i*,
    the part contributed by *i*'s own residents is ``β Σ_j M[i,j]·M[i,j] I_i /
    N_j^present``; the remainder is imported through commuting. The import fraction is
    ``1 − own/total`` ∈ [0, 1] (0 = self-contained, →1 = commuter-driven).

    Args:
        mobility: ``(G, G)`` row-stochastic commuter matrix.
        populations: ``(G,)`` resident populations.
        infectious: optional ``(G,)`` infectious counts; if omitted, the *structural*
            import fraction is returned using uniform prevalence (``I ∝ N``), a pure
            function of ``M`` and ``N`` that measures each district's commuting exposure.
        floor: daytime-population denominator floor.

    Returns:
        ``(G,)`` import fractions in ``[0, 1]``.

    Side effects: none (pure). Caller responsibility: ``M`` row-stochastic, ``N`` > 0.
    """
    M = np.asarray(mobility, dtype=float)
    N = np.asarray(populations, dtype=float)
    I = N.copy() if infectious is None else np.asarray(infectious, dtype=float)
    n_present = np.maximum(effective_daytime_population(M, N), floor)
    i_present = M.T @ I                                   # Σ_k M[k,j] I_k
    prev_j = i_present / n_present                        # per-capita prevalence at j
    total = M @ prev_j                                    # λ_i / β
    # own_i = Σ_j M[i,j]·(M[i,j] I_i)/n_present_j = I_i · Σ_j M[i,j]^2 / n_present_j
    own = (M * M).dot(1.0 / n_present) * I
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = 1.0 - np.where(total > 0, own / total, 0.0)
    return np.clip(frac, 0.0, 1.0)


def commuter_weighted_moran(values: np.ndarray, mobility: np.ndarray, *,
                            n_perm: int = 999, seed: int = 0) -> dict:
    """Moran's I spatial autocorrelation of a district surface, weighted by commuter flow.

    Tests whether districts connected by commuting carry similar values (a
    convergent-validity check on a model-derived district burden surface). The spatial
    weight is the symmetrised off-diagonal commuter matrix ``W = (M + Mᵀ)/2`` with a
    zero diagonal (a district is not its own neighbour).

    Args:
        values: ``(G,)`` district surface (e.g. attack rate, import pressure).
        mobility: ``(G, G)`` commuter matrix.
        n_perm: label permutations for the significance test.
        seed: RNG seed (reproducibility; no ``Math.random``).

    Returns:
        ``{"moran_i": float, "expected_i": float, "p_value": float, "n": int}`` —
        ``expected_i = −1/(G−1)`` under spatial randomness; ``p_value`` is the share of
        permutations with Moran's I ≥ the observed (one-sided, positive autocorrelation).

    Side effects: none (pure).
    """
    x = np.asarray(values, dtype=float)
    W = np.asarray(mobility, dtype=float).copy()
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    g = x.size
    s0 = W.sum()

    def _moran(v: np.ndarray) -> float:
        z = v - v.mean()
        denom = float((z * z).sum())
        if denom <= 0 or s0 <= 0:
            return 0.0
        return float((g / s0) * (z @ W @ z) / denom)

    obs = _moran(x)
    rng = np.random.default_rng(seed)
    perm = np.array([_moran(rng.permutation(x)) for _ in range(int(n_perm))])
    p = float((1 + np.sum(perm >= obs)) / (n_perm + 1))
    return {"moran_i": obs, "expected_i": -1.0 / (g - 1), "p_value": p, "n": g}
