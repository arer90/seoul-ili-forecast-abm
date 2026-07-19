"""
simulation.sim.stepper
======================
Fixed-step RK4 integrator for the metapop SEIR-V-D ODE.

Why RK4 (not scipy.integrate)?
  - Deterministic step size → exactly reproducible trajectories.
  - No scipy dependency for the simulator core. scipy is a heavy import
    and pulls BLAS init; the sim import-chain stays lean.
  - Interventions apply at day boundaries, so a fixed step aligned with
    a daily sub-grid is the natural choice anyway.

The right-hand side is packaged in ``seirvd_derivative`` below and takes
the fully-resolved parameter view (post-intervention) + current state.

Performance
-----------
Two code paths:
  1. Pure-numpy ``rk4_step`` (kwarg-based, flexible).
  2. Numba-JIT ``rk4_step_jit`` (positional, ~5-20× faster on 25-gu metapop).

``rk4_step_jit`` is automatically selected by higher-level callers when
numba is available (it is a runtime dependency declared in pyproject.toml,
but the numpy fallback is preserved for debugging or environments with
broken JIT).

A Rust-native path (simulation/rust/ → seir_core) is also supported and
wins for very large grids (>1000 gu) where the 25-gu Seoul case already
saturates on the Python-JIT path.
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np

from .foi import compute_foi, effective_daytime_population


__all__ = [
    "seirvd_derivative", "rk4_step", "rk4_step_jit", "expeuler_step",
    "HAS_NUMBA", "HAS_C_BACKEND", "HAS_RUST_BACKEND",
    "rk4_step_c", "rk4_step_rs",
]


# ───────────────────────────────────────────────────────────────
# Backend loader chain:  Rust (PyO3) > C (ctypes) > Numba JIT > numpy
# ───────────────────────────────────────────────────────────────
# 1) Rust via PyO3 — built by `cd simulation/rust && maturin develop --release`
try:
    import seir_core as _rust_core
    rk4_step_rs = _rust_core.rk4_step_rs
    HAS_RUST_BACKEND = True
except ImportError:
    rk4_step_rs = None
    HAS_RUST_BACKEND = False


# 2) Pure C via ctypes — built by `bash simulation/c/build.sh`
def _load_c_backend():
    """Locate and load simulation/c/seir_core.{dylib,so,dll} via ctypes."""
    ext = {"darwin": "dylib", "linux": "so", "win32": "dll"}.get(
        sys.platform, "so"
    )
    candidate = Path(__file__).resolve().parent.parent / "c" / f"seir_core.{ext}"
    if not candidate.exists():
        return None, None

    try:
        lib = ctypes.CDLL(str(candidate))
    except OSError:
        return None, None

    # rk4_step_c(state_in*, state_out*, G, dt,
    #            beta, sigma, gamma, omega, VE, V_waning, ifr,
    #            vax_rate*, populations*, mobility*, daytime_pop*)
    c_rk4 = lib.rk4_step_c
    DOUBLE_P = ctypes.POINTER(ctypes.c_double)
    c_rk4.argtypes = [
        DOUBLE_P, DOUBLE_P, ctypes.c_int, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        DOUBLE_P, DOUBLE_P, DOUBLE_P, DOUBLE_P,
    ]
    c_rk4.restype = None
    return lib, c_rk4


_c_lib, _c_rk4 = _load_c_backend()
HAS_C_BACKEND = _c_rk4 is not None


def rk4_step_c(
    state: np.ndarray,
    dt: float,
    beta: float,
    sigma: float,
    gamma: float,
    omega: float,
    VE: float,
    V_waning: float,
    ifr: float,
    vax_rate: np.ndarray,
    populations: np.ndarray,
    mobility: np.ndarray,
    daytime_pop: np.ndarray,
) -> np.ndarray:
    """Thin numpy→ctypes wrapper calling the pure-C `rk4_step_c` kernel.

    All arrays coerced to contiguous float64 before the call. Returns a
    fresh (G, 6) array. Raises RuntimeError if the C backend failed to load.
    """
    if _c_rk4 is None:
        raise RuntimeError(
            "C backend not available. Build it via: bash simulation/c/build.sh"
        )
    s = np.ascontiguousarray(state, dtype=np.float64)
    out = np.empty_like(s)
    vax = np.ascontiguousarray(vax_rate, dtype=np.float64)
    pop = np.ascontiguousarray(populations, dtype=np.float64)
    mob = np.ascontiguousarray(mobility, dtype=np.float64)
    day = np.ascontiguousarray(daytime_pop, dtype=np.float64)
    g = s.shape[0]

    DOUBLE_P = ctypes.POINTER(ctypes.c_double)
    _c_rk4(
        s.ctypes.data_as(DOUBLE_P),
        out.ctypes.data_as(DOUBLE_P),
        ctypes.c_int(g),
        ctypes.c_double(dt),
        ctypes.c_double(beta),
        ctypes.c_double(sigma),
        ctypes.c_double(gamma),
        ctypes.c_double(omega),
        ctypes.c_double(VE),
        ctypes.c_double(V_waning),
        ctypes.c_double(ifr),
        vax.ctypes.data_as(DOUBLE_P),
        pop.ctypes.data_as(DOUBLE_P),
        mob.ctypes.data_as(DOUBLE_P),
        day.ctypes.data_as(DOUBLE_P),
    )
    return out


# 3) Numba JIT — optional, falls back to pure numpy on import fail.
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:  # pragma: no cover — fallback exercised only when numba missing
    HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        """No-op decorator when numba is unavailable."""
        def _wrap(fn):
            return fn
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return _wrap


@njit(cache=True, fastmath=False)
def _seirvd_rhs_jit(
    state: np.ndarray,
    beta: float,
    sigma: float,
    gamma: float,
    omega: float,
    VE: float,
    V_waning: float,
    ifr: float,
    vax_rate: np.ndarray,
    populations: np.ndarray,
    mobility: np.ndarray,
    daytime_pop: np.ndarray,
) -> np.ndarray:
    """JIT right-hand side. state shape (G, 6) → derivative shape (G, 6)."""
    G = state.shape[0]
    # Slice compartments
    S = state[:, 0]
    E = state[:, 1]
    I = state[:, 2]
    R = state[:, 3]
    V = state[:, 4]

    # FoI: lam = beta * (mobility @ ((mobility.T @ I) / max(daytime_pop, 1)))
    # Inline to avoid per-call numpy dispatch.
    I_present = np.zeros(G)
    for j in range(G):
        s = 0.0
        for i in range(G):
            s += mobility[i, j] * I[i]
        I_present[j] = s

    prev_j = np.empty(G)
    for j in range(G):
        d = daytime_pop[j]
        if d < 1.0:
            d = 1.0
        prev_j[j] = I_present[j] / d

    lam = np.empty(G)
    for i in range(G):
        s = 0.0
        for j in range(G):
            s += mobility[i, j] * prev_j[j]
        v = beta * s
        lam[i] = v if v > 0.0 else 0.0

    lam_v = (1.0 - VE) * lam

    out = np.empty_like(state)
    for i in range(G):
        si, ei, ii, ri, vi = S[i], E[i], I[i], R[i], V[i]
        out[i, 0] = -lam[i] * si - vax_rate[i] * si + omega * ri + V_waning * vi       # dS
        out[i, 1] = lam[i] * si + lam_v[i] * vi - sigma * ei                            # dE
        out[i, 2] = sigma * ei - gamma * ii                                             # dI
        out[i, 3] = gamma * (1.0 - ifr) * ii - omega * ri                               # dR
        out[i, 4] = vax_rate[i] * si - lam_v[i] * vi - V_waning * vi                    # dV
        out[i, 5] = gamma * ifr * ii                                                    # dD
    return out


@njit(cache=True, fastmath=False)
def _box_clamp(s: np.ndarray, populations: np.ndarray) -> np.ndarray:
    """Bound each compartment to the physical box [0, N_i] in place.

    N_i (district population) is the mass upper bound for any single compartment.
    Operates on the caller's array (the RK4 stages pass fresh ``state + c*dt*k``
    temporaries, never the live state)."""
    G, C = s.shape
    for i in range(G):
        ni = populations[i]
        for c in range(C):
            v = s[i, c]
            if v < 0.0:
                s[i, c] = 0.0
            elif v > ni:
                s[i, c] = ni
    return s


@njit(cache=True, fastmath=False)
def rk4_step_jit(
    state: np.ndarray,
    dt: float,
    beta: float,
    sigma: float,
    gamma: float,
    omega: float,
    VE: float,
    V_waning: float,
    ifr: float,
    vax_rate: np.ndarray,
    populations: np.ndarray,
    mobility: np.ndarray,
    daytime_pop: np.ndarray,
) -> np.ndarray:
    """Positivity-PRESERVING Numba-JIT RK4 step. Positional args only.

    Each intermediate stage state is bounded to the physical box [0, N_i] BEFORE
    the next RHS evaluation, not only the final ``new``. The classic scheme
    (clamp only ``new`` ≥ 0) lets a stage leave the physical domain; combined with
    the behavioural coupling that rescales beta, a stray per-process JIT-compilation
    rounding can then be amplified into a cross-process blow-up (measured ~2/9
    fresh single-threaded processes hit 1e129–1e264 with bit-identical inputs).
    The stage clamp is a strict NO-OP for well-behaved trajectories (per-stage
    outflow ≤ ~6 %, so compartments already stay in [0, N]) — Stage-5 scenario
    sims and the behaviour-off invariant are unchanged — and only contains the
    pathological realisations."""
    k1 = _seirvd_rhs_jit(state, beta, sigma, gamma, omega, VE, V_waning, ifr,
                         vax_rate, populations, mobility, daytime_pop)
    k2 = _seirvd_rhs_jit(_box_clamp(state + 0.5 * dt * k1, populations),
                         beta, sigma, gamma, omega, VE, V_waning, ifr,
                         vax_rate, populations, mobility, daytime_pop)
    k3 = _seirvd_rhs_jit(_box_clamp(state + 0.5 * dt * k2, populations),
                         beta, sigma, gamma, omega, VE, V_waning, ifr,
                         vax_rate, populations, mobility, daytime_pop)
    k4 = _seirvd_rhs_jit(_box_clamp(state + dt * k3, populations),
                         beta, sigma, gamma, omega, VE, V_waning, ifr,
                         vax_rate, populations, mobility, daytime_pop)
    return _box_clamp(state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), populations)


# ───────────────────────────────────────────────────────────────
# Pure-numpy path — always available, used for debugging and as fallback.
# ───────────────────────────────────────────────────────────────
def seirvd_derivative(
    state: np.ndarray,
    *,
    beta: float,
    sigma: float,
    gamma: float,
    omega: float,
    VE: float,
    V_waning: float,
    ifr: float,
    vax_rate: np.ndarray,       # (G,)
    populations: np.ndarray,    # (G,)
    mobility: np.ndarray,       # (G, G)
    daytime_pop: np.ndarray | None = None,
) -> np.ndarray:
    """Compute d(state)/dt for the metapop SEIR-V-D system.

    ``state`` is shape (G, 6) with compartment order (S, E, I, R, V, D).
    The returned derivative has the same shape.

    Dynamics::

        dS/dt = -λ·S - ν·S  + ω·R + V_waning·V
        dE/dt = +λ·S + (1-VE)·λ·V - σ·E
        dI/dt = +σ·E - γ·I
        dR/dt = +γ·(1 - ifr)·I - ω·R
        dV/dt = +ν·S - (1-VE)·λ·V - V_waning·V
        dD/dt = +γ·ifr·I

    The combination ν·S + V_waning·V conserves mass across S↔V (minus the
    I/E leakage through vaccinated breakthroughs).
    """
    S = state[:, 0]
    E = state[:, 1]
    I = state[:, 2]
    R = state[:, 3]
    V = state[:, 4]
    # D unused on RHS

    lam = compute_foi(I, S, populations, mobility, beta, daytime_pop=daytime_pop)

    # Breakthrough FoI felt by vaccinated individuals (leaky VE)
    lam_v = (1.0 - VE) * lam

    dS = -lam * S - vax_rate * S + omega * R + V_waning * V
    dE = lam * S + lam_v * V - sigma * E
    dI = sigma * E - gamma * I
    dR = gamma * (1.0 - ifr) * I - omega * R
    dV = vax_rate * S - lam_v * V - V_waning * V
    dD = gamma * ifr * I

    out = np.empty_like(state)
    out[:, 0] = dS
    out[:, 1] = dE
    out[:, 2] = dI
    out[:, 3] = dR
    out[:, 4] = dV
    out[:, 5] = dD
    return out


def expeuler_step(
    state: np.ndarray,
    dt: float,
    *,
    params_kwargs: dict,
) -> np.ndarray:
    """Flux-conservative exponential-Euler step (B-P1/M7).

    Unconditionally **positive** AND exactly **mass-conserving**, unlike the
    explicit ``rk4_step`` which diverges (~1/3 of behavioural runs) and is
    band-aided with a box-clamp + per-substep ``nan_to_num``. Each compartment
    loses exactly the fraction ``1 - exp(-rate·dt)`` of its mass through competing
    exponential hazards, and that mass is routed to the destination compartment(s)
    — the deterministic mean-field of the tau-leap agent kernel (the project's
    already-trusted stable Engine A). Consequences: ``S+E+I+R+V+D`` is conserved
    to machine precision, no compartment can go negative for ANY ``dt``, and there
    is no stability limit. Same ``params_kwargs`` contract as :func:`rk4_step`.

    Returns a new state; the input is not mutated.

    Performance: O(G²) for the commuter FoI matmul + O(G) transitions; pure numpy.
    """
    p = params_kwargs
    beta = float(p["beta"]); sigma = float(p["sigma"]); gamma = float(p["gamma"])
    omega = float(p["omega"]); VE = float(p["VE"]); V_waning = float(p["V_waning"])
    ifr = float(p["ifr"])
    vax = np.asarray(p["vax_rate"], dtype=np.float64).ravel()
    M = np.asarray(p["mobility"], dtype=np.float64)
    pops = np.asarray(p["populations"], dtype=np.float64)
    daytime = p.get("daytime_pop")
    if daytime is None:
        daytime = effective_daytime_population(M, pops)
    daytime = np.asarray(daytime, dtype=np.float64)

    s = np.asarray(state, dtype=np.float64)
    S, E, I, R, V = s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4]

    # Commuter FoI, identical to _seirvd_rhs_jit:
    #   lam_i = beta * sum_j M[i,j] * (sum_k M[k,j] I_k) / max(daytime_j, 1)
    I_present = M.T @ I
    prev_j = I_present / np.maximum(daytime, 1.0)
    lam = np.maximum(beta * (M @ prev_j), 0.0)
    lam_v = (1.0 - VE) * lam

    def _frac(rate):
        return -np.expm1(-np.asarray(rate, dtype=np.float64) * dt)  # 1 - e^{-rate·dt}

    def _split(out, num, den):
        w = np.divide(num, den, out=np.zeros_like(den), where=den > 0)
        a = out * w
        return a, out - a

    # S → {E via lam, V via vax}
    rS = lam + vax
    outS = S * _frac(rS)
    S_to_E, S_to_V = _split(outS, lam, rS)
    # E → I (sigma)
    E_to_I = E * _frac(sigma)
    # I → {R via gamma(1-ifr), D via gamma·ifr}; total rate gamma
    outI = I * _frac(gamma)
    I_to_R = outI * (1.0 - ifr)
    I_to_D = outI - I_to_R
    # R → S (omega)
    R_to_S = R * _frac(omega)
    # V → {E via lam_v, S via V_waning}
    rV = lam_v + V_waning
    outV = V * _frac(rV)
    V_to_E, V_to_S = _split(outV, lam_v, rV)

    new = np.empty_like(s)
    new[:, 0] = S - S_to_E - S_to_V + R_to_S + V_to_S
    new[:, 1] = E - E_to_I + S_to_E + V_to_E
    new[:, 2] = I - outI + E_to_I
    new[:, 3] = R - R_to_S + I_to_R
    new[:, 4] = V - outV + S_to_V
    new[:, 5] = s[:, 5] + I_to_D
    return new


def rk4_step(
    state: np.ndarray,
    dt: float,
    *,
    params_kwargs: dict,
) -> np.ndarray:
    """Classic RK4 step. ``params_kwargs`` is the dict of keyword arguments
    passed to ``seirvd_derivative`` (already resolved by the intervention
    layer for the current day).

    Returns a new state; the input is not mutated.

    Uses the Numba-JIT path when available (``HAS_NUMBA == True``) and all
    array kwargs are contiguous float64. Falls back to pure numpy otherwise.
    """
    if HAS_NUMBA:
        # Resolve daytime_pop once.
        daytime_pop = params_kwargs.get("daytime_pop")
        if daytime_pop is None:
            daytime_pop = effective_daytime_population(
                params_kwargs["mobility"], params_kwargs["populations"]
            )
        # Ensure float64 for numba signature.
        return rk4_step_jit(
            np.ascontiguousarray(state, dtype=np.float64),
            float(dt),
            float(params_kwargs["beta"]),
            float(params_kwargs["sigma"]),
            float(params_kwargs["gamma"]),
            float(params_kwargs["omega"]),
            float(params_kwargs["VE"]),
            float(params_kwargs["V_waning"]),
            float(params_kwargs["ifr"]),
            np.ascontiguousarray(params_kwargs["vax_rate"], dtype=np.float64),
            np.ascontiguousarray(params_kwargs["populations"], dtype=np.float64),
            np.ascontiguousarray(params_kwargs["mobility"], dtype=np.float64),
            np.ascontiguousarray(daytime_pop, dtype=np.float64),
        )

    # Pure-numpy fallback — same positivity-preserving box clamp as the JIT path
    # (clamp each intermediate stage to [0, N] before the next RHS), so both
    # backends produce the identical bounded trajectory.
    pops = np.asarray(params_kwargs["populations"], dtype=float)[:, None]
    _box = lambda s: np.clip(s, 0.0, pops)
    k1 = seirvd_derivative(state, **params_kwargs)
    k2 = seirvd_derivative(_box(state + 0.5 * dt * k1), **params_kwargs)
    k3 = seirvd_derivative(_box(state + 0.5 * dt * k2), **params_kwargs)
    k4 = seirvd_derivative(_box(state + dt * k3), **params_kwargs)
    new = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return _box(new)
