"""Age-structured metapopulation SEIR-V-D with a WAIFW contact matrix (#4, 2026-06-10).

Closes the honest gap flagged by the age-FOI TDD audit: the existing scalar metapop
(``MetapopSEIRVD``) and the agent kernel apply age only as a per-agent row-MEAN contact
WEIGHT — there is NO age_i × age_j mixing (an infectious child does not preferentially
infect other children). This module adds the full **who-acquires-infection-from-whom
(WAIFW)** force of infection over a (gu × age × compartment) state, coupling space via the
commuter mobility matrix M and age via the contact matrix C:

    λ_{i,a}(t) = β(t) · Σ_b  Ĉ[a,b] · ( Σ_j M[i,j] · I_{j,b} / N_{j,b} )

where Ĉ = C / ρ(C) is the spectral-radius-normalized contact matrix so that the overall
basic reproduction number stays ≈ β/γ regardless of the age structure (age redistributes
*who* is infected, not the total R0). Susceptibles S_{i,a} are depleted at λ_{i,a}.

OPT-IN / SEPARATE: this does NOT modify the frozen age-agnostic MetapopSEIRVD path (the
thesis results depend on it). It is a new deep module. Integration uses the same
mass-conserving exp-Euler scheme as ``stepper.expeuler_step`` (unconditionally positive,
S+E+I+R+V+D conserved to machine precision per (gu, age) cell), generalized to the age axis.

Reduces exactly to the scalar commuter-coupled metapop when A=1 (Ĉ=[[1]]).

The 7 age bands are decades (``_age_band_from_label``: 0-9, 10-19, …, 50-59, 60+). Validation
against real age-stratified ILI (``sentinel_influenza``) lives in ``test_age_metapop.py``: with
the correct year-level decade crosswalk the model's age attack-rate reproduces the real
school-high (10-19) / elderly-low (60+) gradient at Spearman ρ ≈ 0.68 — it captures the peak and
trough but real working-age ILI is flatter than the contact matrix alone predicts (honest scope).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulation.sim.parameters import IDX_S, IDX_E, IDX_I, IDX_R, IDX_V, IDX_D, N_COMPARTMENTS


def normalize_contact(C: np.ndarray) -> np.ndarray:
    """Spectral-radius-normalize a contact matrix → dominant eigenvalue 1.

    Args:
        C: (A, A) non-negative contact matrix (symmetric or not). For A=1 returns [[1]].

    Returns:
        (A, A) Ĉ = C / ρ(C). Keeps the *relative* age-mixing structure while fixing the
        overall transmission scale so β = R0·γ is interpretable across age structures.

    Raises:
        ValueError: C not square or ρ(C) ≤ 0.
    """
    C = np.asarray(C, dtype=np.float64)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"contact matrix must be square, got {C.shape}")
    rho = float(np.max(np.abs(np.linalg.eigvals(C))))
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError(f"contact matrix spectral radius must be > 0, got {rho}")
    return C / rho


def age_foi(I_age: np.ndarray, N_age: np.ndarray, beta: float,
            contact_norm: np.ndarray, mobility: np.ndarray) -> np.ndarray:
    """WAIFW + commuter-coupled force of infection per (gu, age). PURE.

    Args:
        I_age: (G, A) infectious counts.
        N_age: (G, A) population per gu×age (>0).
        beta: transmission scale (= R0·γ).
        contact_norm: (A, A) spectral-normalized contact matrix Ĉ. Ĉ[a,b] = contact of
            a susceptible age-a with infectious age-b.
        mobility: (G, G) row-stochastic commuter matrix M (M[i,j] = i's contact share with j).

    Returns:
        (G, A) λ_{i,a} = β · Σ_b Ĉ[a,b] · (M @ (I_{·,b}/N_{·,b})). Non-negative.

    Performance: O(G²·A + G·A²). Side effects: none.
    """
    prev = I_age / np.maximum(N_age, 1.0)        # (G,A) infectious prevalence by gu×age
    coupled = mobility @ prev                     # (G,A) spatially-mixed prevalence per age
    # age mixing: λ[:,a] = β Σ_b Ĉ[a,b] coupled[:,b]  →  (G,A) = coupled @ Ĉ.T
    return beta * (coupled @ contact_norm.T)


@dataclass
class AgeDiseaseParams:
    """SEIR-V-D rates (per day). delta/gamma may be age-vectors(A,) for age-specific severity."""
    beta: float = 0.45
    sigma: float = 1.0 / 1.5
    gamma: float = 1.0 / 3.0
    delta: float = 0.0          # I→D
    nu: float = 0.0             # S→V vaccination
    omega: float = 0.0          # R→S waning
    v_waning: float = 0.0       # V→S waning


def age_expeuler_step(state: np.ndarray, dt: float, *, params_kwargs: dict) -> np.ndarray:
    """Mass-conserving exp-Euler step for the (G, A, 6) age-metapop. PURE (returns new array).

    Each compartment loses exactly 1-exp(-total_rate·dt) of its mass through competing
    exponential hazards, routed to destinations by rate proportion. Unconditionally
    positive; S+E+I+R+V+D conserved per (gu,age) to machine precision for any dt.

    params_kwargs keys: beta, sigma, gamma, delta, nu, omega, v_waning(scalars or (A,) for
    gamma/delta), contact_norm (A,A), mobility (G,G), populations_age (G,A).
    """
    p = params_kwargs
    S = state[:, :, IDX_S]; E = state[:, :, IDX_E]; I = state[:, :, IDX_I]
    R = state[:, :, IDX_R]; V = state[:, :, IDX_V]; D = state[:, :, IDX_D]
    A = state.shape[1]

    def _vec(x):  # scalar → (A,) broadcast
        a = np.asarray(x, dtype=np.float64)
        return a if a.ndim == 1 else np.full(A, float(a))

    sigma = _vec(p["sigma"]); gamma = _vec(p["gamma"]); delta = _vec(p.get("delta", 0.0))
    omega = float(p.get("omega", 0.0)); vwan = float(p.get("v_waning", 0.0))
    nu = float(p.get("nu", 0.0))

    lam = age_foi(I, p["populations_age"], float(p["beta"]),
                  p["contact_norm"], p["mobility"])          # (G,A)

    out = np.array(state, dtype=np.float64, copy=True)

    # S → {E (λ), V (ν)}  competing
    s_rate = lam + nu
    s_out = S * (1.0 - np.exp(-s_rate * dt))
    s_safe = np.maximum(s_rate, 1e-300)
    to_E = s_out * (lam / s_safe)
    to_V = s_out * (nu / s_safe)
    # E → I (σ)
    e_out = E * (1.0 - np.exp(-sigma[None, :] * dt))
    # I → {R (γ), D (δ)} competing
    i_rate = gamma[None, :] + delta[None, :]
    i_out = I * (1.0 - np.exp(-i_rate * dt))
    i_safe = np.maximum(i_rate, 1e-300)
    to_R = i_out * (gamma[None, :] / i_safe)
    to_D = i_out * (delta[None, :] / i_safe)
    # R → S (ω), V → S (v_waning)
    r_out = R * (1.0 - np.exp(-omega * dt)) if omega > 0 else np.zeros_like(R)
    v_out = V * (1.0 - np.exp(-vwan * dt)) if vwan > 0 else np.zeros_like(V)

    out[:, :, IDX_S] = S - s_out + r_out + v_out
    out[:, :, IDX_E] = E + to_E - e_out
    out[:, :, IDX_I] = I + e_out - i_out
    out[:, :, IDX_R] = R + to_R - r_out
    out[:, :, IDX_V] = V + to_V - v_out
    out[:, :, IDX_D] = D + to_D
    return out


@dataclass
class AgeMetapopResult:
    """Trajectories. S/E/I/R/V/D each (T, G, A). incidence (T, G, A) = daily new infections."""
    S: np.ndarray; E: np.ndarray; I: np.ndarray; R: np.ndarray; V: np.ndarray; D: np.ndarray
    incidence: np.ndarray

    def city_I_by_age(self) -> np.ndarray:
        """(T, A) — Seoul-total infectious by age band."""
        return self.I.sum(axis=1)

    def attack_rate_by_age(self, N_age: np.ndarray) -> np.ndarray:
        """(A,) cumulative attack rate per age = total incidence / age population."""
        cum = self.incidence.sum(axis=(0, 1))      # (A,)
        return cum / np.maximum(N_age.sum(axis=0), 1.0)


def run_age_metapop(populations_age: np.ndarray, mobility: np.ndarray, contact: np.ndarray,
                    disease: AgeDiseaseParams, *, initial_infected_age: np.ndarray,
                    days: int = 180, dt: float = 0.5, sub_steps: int | None = None) -> AgeMetapopResult:
    """Run the age-structured commuter-coupled SEIR-V-D.

    Args:
        populations_age: (G, A) population per gu×age (>0).
        mobility: (G, G) row-stochastic commuter matrix.
        contact: (A, A) RAW contact matrix (e.g. CONTACT_MATRIX_7x7); spectral-normalized inside.
        disease: AgeDiseaseParams.
        initial_infected_age: (G, A) seed infectious counts (subtracted from S).
        days: number of daily output rows. dt: integrator step (days). sub_steps: steps/day.

    Returns:
        AgeMetapopResult with (T=days, G, A) trajectories + daily incidence (new E inflow).

    Performance: O(days·(1/dt)·(G²A + GA²)). Side effects: none.
    Caller responsibility: populations_age > 0, mobility rows sum to ~1.
    """
    G, A = populations_age.shape
    Cn = normalize_contact(contact)
    if sub_steps is None:
        sub_steps = max(1, int(round(1.0 / dt)))

    state = np.zeros((G, A, N_COMPARTMENTS), dtype=np.float64)
    seed = np.asarray(initial_infected_age, dtype=np.float64)
    state[:, :, IDX_I] = seed
    state[:, :, IDX_S] = np.maximum(populations_age - seed, 0.0)

    kwargs = {"beta": disease.beta, "sigma": disease.sigma, "gamma": disease.gamma,
              "delta": disease.delta, "nu": disease.nu, "omega": disease.omega,
              "v_waning": disease.v_waning, "contact_norm": Cn, "mobility": mobility,
              "populations_age": populations_age}

    T = days
    S = np.empty((T, G, A)); E = np.empty((T, G, A)); I = np.empty((T, G, A))
    R = np.empty((T, G, A)); V = np.empty((T, G, A)); Dd = np.empty((T, G, A))
    inc = np.zeros((T, G, A))
    for d in range(T):
        day_inc = np.zeros((G, A))
        for _ in range(sub_steps):
            day_inc += disease.sigma * state[:, :, IDX_E] * dt  # E→I flux ≈ realized incidence
            state = age_expeuler_step(state, dt, params_kwargs=kwargs)
        S[d] = state[:, :, IDX_S]; E[d] = state[:, :, IDX_E]; I[d] = state[:, :, IDX_I]
        R[d] = state[:, :, IDX_R]; V[d] = state[:, :, IDX_V]; Dd[d] = state[:, :, IDX_D]
        inc[d] = day_inc
    return AgeMetapopResult(S=S, E=E, I=I, R=R, V=V, D=Dd, incidence=inc)
