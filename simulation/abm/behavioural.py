"""
simulation.abm.behavioural
==========================
Four-parameter behavioural ABM coupled to the metapop SEIR-V-D kernel.

Mathematical contract (per district i, mean-field aggregate over the
~400-agent household layer):

    dR_i/dt = alpha  * (I_i / N_i) - lambda_R * R_i        (risk perception)
    dF_i/dt = delta  * 1[compliant_i]      - rho    * F_i  (fatigue)
    compliant_i = 1[R_i - kappa * F_i > theta]             (utility threshold)
    c_i        = c0 * (1 - strength * compliant_i)         (contact reduction)
    beta_i(t)  = beta_0 * (c_i / c0)^2                     (quadratic coupling)

The four ESTIMATED parameters are (alpha, kappa, tau, theta) where
``tau = 1 / rho`` is the fatigue time constant. The three frozen
constants (lambda_R, delta, rho) take literature-anchored values from
Rahmandad, Lim, Sterman 2021 [86]:

    lambda_R = 1/30  d^-1   (30-day risk-perception half-life)
    delta    = 0.05  d^-1   (fatigue accrual per compliant day)
    rho      = 1/tau        (tau is the estimated parameter)

Invariance guarantee
--------------------
Setting ``alpha = 0, kappa = 0, tau -> inf`` (equivalently rho = 0)
reduces R_i, F_i, and compliant_i identically to zero for all t, so
beta_i(t) = beta_0 for all t. The coupled trajectory then reproduces
the behaviour-off kernel-only baseline to RK4 round-off precision
(tested in ``run_invariant_test``).
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from simulation.sim.metapop_seirvd import SimResult
from simulation.sim.parameters import (
    IDX_E,
    IDX_I,
    IDX_R,
    IDX_S,
    IDX_V,
    MetapopParams,
    N_COMPARTMENTS,
)
from simulation.sim.stepper import expeuler_step, rk4_step
from simulation.sim.foi import effective_daytime_population

log = logging.getLogger(__name__)

__all__ = [
    "BehaviouralParams",
    "ABMResult",
    "run_coupled_abm",
    "run_invariant_test",
    "run_rebound_scenario",
]


# Numba JIT path for the R/F/C coupled ODE update. Falls back to numpy.
try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        def _wrap(fn):
            return fn
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return _wrap


@njit(cache=True, fastmath=False)
def _behavioural_step_jit(
    I_prev: np.ndarray,
    N_prev: np.ndarray,
    R_prev: np.ndarray,
    F_prev: np.ndarray,
    C_prev: np.ndarray,
    alpha: float,
    lambda_R: float,
    kappa: float,
    theta: float,
    delta: float,
    rho: float,
    strength: float,
    beta_0: float,
    steepness: float,
) -> tuple:
    """One-day Euler step of the R/F/C coupled behavioural ODE.

    Returns (R_next, F_next, C_next, beta_eff_district, beta_effective_mean).
    All arrays are (G,) float64.
    """
    G = I_prev.shape[0]
    R_next = np.empty(G, dtype=np.float64)
    F_next = np.empty(G, dtype=np.float64)
    C_next = np.empty(G, dtype=np.float64)
    beta_eff_district = np.empty(G, dtype=np.float64)
    scale_mean = 0.0

    for i in range(G):
        nn = N_prev[i] if N_prev[i] > 1.0 else 1.0
        prev_ratio = I_prev[i] / nn

        dR = alpha * prev_ratio - lambda_R * R_prev[i]
        r_n = R_prev[i] + dR

        # Compliance uses the UPDATED R; fatigue uses the PREVIOUS compliance.
        # steepness=+inf ⇒ hard Heaviside (back-compat); finite ⇒ smooth logistic
        # (removes the FP knife-edge that drove cross-process non-determinism).
        margin = r_n - kappa * F_prev[i] - theta
        if steepness > 1e15:
            compliant = 1.0 if margin > 0.0 else 0.0
        else:
            compliant = 1.0 / (1.0 + np.exp(-steepness * margin))

        dF = delta * C_prev[i] - rho * F_prev[i]
        f_n = F_prev[i] + dF

        if r_n < 0.0:
            r_n = 0.0
        if f_n < 0.0:
            f_n = 0.0

        R_next[i] = r_n
        F_next[i] = f_n
        C_next[i] = compliant

        s = 1.0 - strength * compliant
        scale = s * s  # (1 - strength * compliant) ** 2
        beta_eff_district[i] = beta_0 * scale
        scale_mean += scale

    beta_effective_mean = beta_0 * (scale_mean / G)
    return R_next, F_next, C_next, beta_eff_district, beta_effective_mean


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BehaviouralParams:
    """Behavioural-layer parameters. MVP estimates (alpha, kappa, tau, theta).

    Defaults correspond to a mild-response, post-COVID-fatigue regime that
    produces a measurable deviation from the behaviour-off baseline without
    driving the epidemic to extinction. See ``run_rebound_scenario`` for the
    2022-24 rebound configuration.
    """
    # Estimated
    alpha: float = 1.0       # risk sensitivity: 0 = no behavioural coupling
    kappa: float = 0.5       # fatigue weight in the compliance decision
    tau: float = 30.0        # fatigue time constant (days); rho = 1/tau
    theta: float = 0.20      # compliance-activation threshold
    # Frozen (literature-anchored)
    lambda_R: float = 1.0 / 30.0   # 30-day risk-perception decay
    delta: float = 0.05            # fatigue accrual rate per compliant day
    strength: float = 0.6          # peak contact reduction when compliant
    # Numerical: compliance-decision sharpness. +inf = the hard Heaviside
    # threshold 1[R-κF>θ] (DEFAULT — back-compatible, exact behaviour-off
    # invariant). A FINITE value replaces it with the smooth logistic
    # σ(steepness·(R-κF-θ)), which removes the floating-point knife-edge at
    # near-critical crossings (the cross-process non-determinism source) and
    # gives a physically gradual response. steepness≈30 ⇒ a ~0.03-wide
    # transition in risk-margin units; the behaviour-off residual σ(-steepness·θ)
    # is then ~0.2% (no longer bit-zero — see run_invariant_test tolerance).
    compliance_steepness: float = float("inf")

    @property
    def rho(self) -> float:
        """Fatigue decay rate rho = 1 / tau."""
        return 0.0 if not np.isfinite(self.tau) or self.tau <= 0 else 1.0 / self.tau

    def is_behaviour_off(self) -> bool:
        """True iff the parameter set would produce no behavioural effect."""
        return self.alpha == 0.0 and self.kappa == 0.0 and self.rho == 0.0

    def validate(self) -> None:
        """Raise ``ValueError`` on out-of-range behavioural parameters.

        Without this an invalid ``tau <= 0`` silently collapses to ``rho = 0``
        (no fatigue) via the :pyattr:`rho` property, masking a caller error;
        ``tau = +inf`` is the explicit behaviour-off limit and is allowed.
        """
        if not (np.isfinite(self.alpha) and self.alpha >= 0.0):
            raise ValueError(f"alpha must be finite and >= 0; got {self.alpha}")
        if not (np.isfinite(self.kappa) and self.kappa >= 0.0):
            raise ValueError(f"kappa must be finite and >= 0; got {self.kappa}")
        if not (np.isfinite(self.theta) and self.theta >= 0.0):
            raise ValueError(f"theta must be finite and >= 0; got {self.theta}")
        if not (0.0 <= self.strength <= 1.0):
            raise ValueError(f"strength must be in [0, 1]; got {self.strength}")
        if not (self.tau > 0):  # +inf passes (behaviour-off); <=0 and NaN fail
            raise ValueError(
                f"tau must be positive (or +inf for behaviour-off); got {self.tau}"
            )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class ABMResult:
    """SEIR-V-D trajectory plus the ABM behavioural state time series.

    Attributes
    ----------
    seir : SimResult
        Standard SEIR-V-D output (state, incidence, city totals).
    risk : (T+1, G) ndarray, R_i(t) across days and districts.
    fatigue : (T+1, G) ndarray, F_i(t).
    compliance : (T+1, G) ndarray, compliant_i(t) in {0, 1}.
    beta_eff : (T+1, G) ndarray, effective per-district beta_i(t).
    behaviour : BehaviouralParams used for this run (copy).
    behaviour_off : bool, True if the run was the behaviour-off baseline.
    """
    seir: SimResult
    risk: np.ndarray
    fatigue: np.ndarray
    compliance: np.ndarray
    beta_eff: np.ndarray
    behaviour: BehaviouralParams
    behaviour_off: bool = False

    @property
    def days(self) -> np.ndarray:
        return self.seir.days

    @property
    def district_names(self) -> list[str]:
        return self.seir.district_names

    def city_I(self) -> np.ndarray:
        """(T+1,) city-wide infectious totals."""
        return self.seir.city_total("I")

    def mean_compliance(self) -> np.ndarray:
        """(T+1,) city-wide mean compliance fraction."""
        return self.compliance.mean(axis=1)

    def mean_beta_scale(self) -> np.ndarray:
        """(T+1,) city-wide mean of beta_eff / beta_0."""
        beta_0 = self.behaviour_scale_anchor()
        return self.beta_eff.mean(axis=1) / beta_0

    def behaviour_scale_anchor(self) -> float:
        """Return beta_0 from the SimResult params (for scale normalisation)."""
        return float(self.seir.params.disease.R0 * self.seir.params.disease.gamma)


# ---------------------------------------------------------------------------
# Coupled runner
# ---------------------------------------------------------------------------
def run_coupled_abm(
    metapop_params: MetapopParams,
    behaviour: BehaviouralParams,
    *,
    verbose: bool = False,
) -> ABMResult:
    """Run the behavioural ABM tightly coupled to the SEIR-V-D kernel.

    The run advances one day at a time. At the start of each day:

    1. Read current I_i and N_i from the SEIR state.
    2. Integrate the per-district behavioural ODE (Euler, 1 step per day).
    3. Compute compliance_i and the behavioural scale s_i = (1 - strength *
       compliance_i)^2.
    4. Feed scalar beta_0 into rk4_step; the per-district beta is applied
       by scaling the force-of-infection after rk4_step via a direct
       re-integration over the sub-day grid. For MVP we use the city-wide
       mean scale s_bar(t) as the scalar beta multiplier to keep the kernel
       API untouched; this is a defensible simplification because the
       commuter coupling already homogenises the force-of-infection across
       districts within a day. District-specific beta is preserved in the
       behavioural-state tensor for §4.16 figures.

    Parameters
    ----------
    metapop_params : MetapopParams
        Standard kernel inputs (populations, mobility, initial infected, ...).
    behaviour : BehaviouralParams
        Behavioural-layer parameter set.
    verbose : bool
        Log one progress line per ~50 days.

    Returns
    -------
    ABMResult
    """
    metapop_params.validate()
    behaviour.validate()
    G = int(metapop_params.populations.size)
    pops = np.asarray(metapop_params.populations, dtype=float)
    M = np.asarray(metapop_params.mobility, dtype=float)
    names = list(metapop_params.district_names) if metapop_params.district_names else [
        f"gu_{i}" for i in range(G)
    ]
    daytime = effective_daytime_population(M, pops)

    disease = metapop_params.disease
    days = int(metapop_params.days)
    dt = float(metapop_params.dt)
    if dt <= 0 or days <= 0:
        raise ValueError(f"dt and days must be positive; got dt={dt}, days={days}")
    sub_steps = int(round(1.0 / dt))
    if abs(sub_steps * dt - 1.0) > 1e-9:
        raise ValueError(f"dt={dt} must divide a day evenly")

    # -----------------------------------------------------------------
    # Initial state
    # -----------------------------------------------------------------
    state = _build_initial_state(metapop_params, G)
    out_state = np.zeros((days + 1, G, N_COMPARTMENTS), dtype=float)
    out_state[0] = state
    incidence = np.zeros((days + 1, G), dtype=float)
    incidence[0] = state[:, IDX_E] * disease.sigma

    # Behavioural state, initial
    R_t = np.zeros((days + 1, G), dtype=float)
    F_t = np.zeros((days + 1, G), dtype=float)
    C_t = np.zeros((days + 1, G), dtype=float)  # compliance indicator
    B_t = np.zeros((days + 1, G), dtype=float)  # effective beta per district
    beta_0 = disease.beta
    B_t[0] = beta_0  # before any behavioural response

    vax_rate = _as_G_vector(metapop_params.vaccination_rate, G)

    # -----------------------------------------------------------------
    # Main day loop
    # -----------------------------------------------------------------
    # Integrator: stable exp-Euler (B-P1) is now the DEFAULT — mass-conserving and
    # unconditionally positive, so it cannot diverge for any dt and needs no band-aid.
    # Opt OUT to legacy RK4 with MPH_STABLE_INTEGRATOR=0 (reproduces frozen pre-2026-06-10
    # thesis runs). The per-substep nan_to_num clamp below is applied ONLY for RK4.
    _step_fn = rk4_step if os.environ.get("MPH_STABLE_INTEGRATOR") == "0" else expeuler_step
    _needs_clamp = _step_fn is rk4_step
    t_start = time.time()
    for d in range(days):
        # 1) Behavioural ODE Euler step — Numba path fuses the 8 numpy ops
        #    into a single C-loop (3-8× measured).
        I_prev = np.ascontiguousarray(state[:, IDX_I], dtype=np.float64)
        N_prev = np.maximum(state.sum(axis=1), 1.0).astype(np.float64)

        if _HAS_NUMBA:
            R_next, F_next, C_next, beta_eff_district, _scalar_mean = _behavioural_step_jit(
                I_prev,
                N_prev,
                R_t[d].astype(np.float64),
                F_t[d].astype(np.float64),
                C_t[d].astype(np.float64),
                float(behaviour.alpha),
                float(behaviour.lambda_R),
                float(behaviour.kappa),
                float(behaviour.theta),
                float(behaviour.delta),
                float(behaviour.rho),
                float(behaviour.strength),
                float(beta_0),
                float(behaviour.compliance_steepness),
            )
            R_t[d + 1] = R_next
            F_t[d + 1] = F_next
            C_t[d + 1] = C_next
            B_t[d + 1] = beta_eff_district
        else:
            prev_ratio = I_prev / N_prev
            dR = behaviour.alpha * prev_ratio - behaviour.lambda_R * R_t[d]
            R_next = R_t[d] + dR
            margin = R_next - behaviour.kappa * F_t[d] - behaviour.theta
            if behaviour.compliance_steepness > 1e15:
                C_next = (margin > 0.0).astype(float)  # hard Heaviside (back-compat)
            else:
                C_next = 1.0 / (1.0 + np.exp(-behaviour.compliance_steepness * margin))
            dF = behaviour.delta * C_t[d] - behaviour.rho * F_t[d]
            F_next = F_t[d] + dF
            R_next = np.maximum(R_next, 0.0)
            F_next = np.maximum(F_next, 0.0)
            R_t[d + 1] = R_next
            F_t[d + 1] = F_next
            C_t[d + 1] = C_next
            scale_district = (1.0 - behaviour.strength * C_next) ** 2
            beta_eff_district = beta_0 * scale_district
            B_t[d + 1] = beta_eff_district

        # 3) City-wide mean scale for the scalar-beta kernel sub-step.
        #    beta_eff_district is set in BOTH paths (Numba return at line ~312;
        #    non-Numba at line ~345 as beta_0*scale_district), and
        #    (beta_0*scale_district).mean() == beta_eff_district.mean() since beta_0
        #    is scalar — so this avoids the UnboundLocalError on `scale_district`
        #    that the Numba path triggered (scale_district only existed off-Numba).
        beta_effective_day = float(beta_eff_district.mean())

        # 4) Advance SEIR with the effective scalar beta
        kwargs = {
            "beta": beta_effective_day,
            "sigma": disease.sigma,
            "gamma": disease.gamma,
            "omega": disease.omega,
            "VE": disease.VE,
            "V_waning": disease.V_waning,
            "ifr": disease.ifr,
            "vax_rate": vax_rate,
            "populations": pops,
            "mobility": M,
            "daytime_pop": daytime,
        }
        daily_inc = np.zeros(G, dtype=float)
        for _ in range(sub_steps):
            daily_inc += disease.sigma * state[:, IDX_E] * dt
            state = _step_fn(state, dt, params_kwargs=kwargs)
            # RK4-only band-aid: kill within-day overflow (NaN/±inf) every sub-step.
            # The stable exp-Euler default cannot diverge, so this clamp is skipped
            # there (no more silent diverge→0 masking). The physical [0,N] cap stays
            # ONCE per day (below) for both paths. Mirrored in run_agent_abm.
            if _needs_clamp:
                np.nan_to_num(state, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(state, 0.0, pops[:, None], out=state)

        out_state[d + 1] = state
        incidence[d + 1] = daily_inc

        if verbose and (d + 1) % 50 == 0:
            I_city = float(state[:, IDX_I].sum())
            comp = float(C_next.mean())
            log.info(
                "day %d  I_city=%.1f  comp_frac=%.3f  beta_mean=%.4f",
                d + 1, I_city, comp, beta_effective_day,
            )

    elapsed = time.time() - t_start
    if verbose:
        log.info("coupled ABM run: %d days in %.2fs", days, elapsed)

    seir_result = SimResult(
        state=out_state,
        days=np.arange(days + 1),
        district_names=names,
        incidence=incidence,
        params=metapop_params,
        interventions=[],
        epi_validity={"_skipped": "ABM run", "elapsed_sec": elapsed},
    )

    return ABMResult(
        seir=seir_result,
        risk=R_t,
        fatigue=F_t,
        compliance=C_t,
        beta_eff=B_t,
        behaviour=behaviour,
        behaviour_off=behaviour.is_behaviour_off(),
    )


# ---------------------------------------------------------------------------
# Invariant test (alpha = 0 -> byte-identity to kernel-only)
# ---------------------------------------------------------------------------
def run_invariant_test(
    metapop_params: MetapopParams,
    *,
    tolerance: float = 1e-6,
) -> dict:
    """Run the alpha = 0 invariant test and return a structured report.

    The test runs the coupled ABM with a behaviour-off parameter set
    (alpha = 0, kappa = 0, tau = inf) and compares the SEIR state to a
    reference kernel-only run. The RMSE across (days+1, 25 gu, 6
    compartments) must be below ``tolerance`` for the test to pass.

    Returns
    -------
    dict
        {
            "passed": bool,
            "rmse": float,
            "max_abs_err": float,
            "tolerance": float,
            "abm_mean_compliance": float,  # must be 0.0
            "kernel_city_I_peak": float,
            "abm_city_I_peak": float,
        }
    """
    from simulation.sim.metapop_seirvd import MetapopSEIRVD

    # G-266 (2026-06-13 ABM 정밀 audit): kernel 기본 적분기=RK4(metapop_seirvd.py:257) vs ABM 기본=exp-Euler
    #   (behavioural.py:344, 2026-06-10~)로 **비대칭** → env 미설정(운영 기본)이면 두 엔진이 서로 다른 적분기로
    #   돌아 invariant 가 행동결합이 아니라 *적분기 차이*를 포착(rmse≈5508, peak 16.5%차). 행동 버그 아님.
    #   양측을 동일 적분기(exp-Euler=ABM 운영 기본)로 고정해 apples-to-apples 비교 → byte-exact.
    _prev_si = os.environ.get("MPH_STABLE_INTEGRATOR")
    os.environ["MPH_STABLE_INTEGRATOR"] = "1"      # kernel·ABM 양측 exp-Euler 강제
    try:
        # Kernel-only reference
        kernel_result = MetapopSEIRVD(metapop_params).run(run_validator=False)
        # Behaviour-off ABM
        off = BehaviouralParams(alpha=0.0, kappa=0.0, tau=float("inf"))
        abm = run_coupled_abm(metapop_params, off)
    finally:
        if _prev_si is None:
            os.environ.pop("MPH_STABLE_INTEGRATOR", None)
        else:
            os.environ["MPH_STABLE_INTEGRATOR"] = _prev_si

    diff = abm.seir.state - kernel_result.state
    rmse = float(np.sqrt((diff ** 2).mean()))
    max_err = float(np.abs(diff).max())
    passed = rmse <= tolerance and max_err <= tolerance * 1e3

    return {
        "passed": bool(passed),
        "rmse": rmse,
        "max_abs_err": max_err,
        "tolerance": float(tolerance),
        "abm_mean_compliance": float(abm.compliance.mean()),
        "kernel_city_I_peak": float(kernel_result.city_total("I").max()),
        "abm_city_I_peak": float(abm.city_I().max()),
    }


# ---------------------------------------------------------------------------
# S-rebound scenario (headline §4.16)
# ---------------------------------------------------------------------------
def run_rebound_scenario(
    metapop_params: MetapopParams,
    *,
    behav_off: Optional[BehaviouralParams] = None,
    behav_on: Optional[BehaviouralParams] = None,
) -> dict:
    """Run the 2022-24 rebound natural-experiment scenario.

    The scenario compares:

    - behaviour-off baseline: the Rust kernel with no ABM (alpha = 0)
    - behaviour-on (post-COVID fatigue): alpha > 0 with long tau, so
      accumulated fatigue dampens compliance during the rebound window
      and the epidemic returns to a pre-COVID-like trajectory.

    Reports the headline metric: fraction of rebound peak amplitude
    attributable to endogenous behavioural relaxation vs the static
    baseline. Positive values indicate behaviour-on under-predicts the
    observed rebound relative to behaviour-off (i.e. fatigue reduces
    compliance and lets the epidemic rebound), negative indicates the
    opposite.

    Returns
    -------
    dict
        {
            "behaviour_off": ABMResult-like summary,
            "behaviour_on":  ABMResult-like summary,
            "peak_off": float,
            "peak_on":  float,
            "peak_shift_pct": float,
            "day_of_peak_off": int,
            "day_of_peak_on":  int,
            "mean_compliance_on": float,
            "city_I_off": list[float],  # city-wide I trajectory
            "city_I_on":  list[float],
            "behaviour_on_params": dict,
        }
    """
    if behav_off is None:
        behav_off = BehaviouralParams(alpha=0.0, kappa=0.0, tau=float("inf"))
    if behav_on is None:
        # Post-COVID fatigue regime: alpha=1.5 (risk-averse), tau=60 (slow decay),
        # theta=0.15 (easy activation), kappa=0.8 (high fatigue weight).
        behav_on = BehaviouralParams(
            alpha=1.5, kappa=0.8, tau=60.0, theta=0.15,
        )

    off = run_coupled_abm(metapop_params, behav_off)
    on = run_coupled_abm(metapop_params, behav_on)

    I_off = off.city_I()
    I_on = on.city_I()
    peak_off = float(I_off.max())
    peak_on = float(I_on.max())
    day_off = int(np.argmax(I_off))
    day_on = int(np.argmax(I_on))
    if peak_off > 0:
        shift_pct = 100.0 * (peak_on - peak_off) / peak_off
    else:
        shift_pct = 0.0

    return {
        "peak_off": peak_off,
        "peak_on": peak_on,
        "peak_shift_pct": shift_pct,
        "day_of_peak_off": day_off,
        "day_of_peak_on": day_on,
        "mean_compliance_on": float(on.compliance.mean()),
        "mean_compliance_off": float(off.compliance.mean()),
        "city_I_off": I_off.tolist(),
        "city_I_on": I_on.tolist(),
        "behaviour_on_params": {
            "alpha": behav_on.alpha, "kappa": behav_on.kappa,
            "tau": behav_on.tau, "theta": behav_on.theta,
        },
        "district_names": off.district_names,
        "days": days_trajectory(off),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_initial_state(params: MetapopParams, G: int) -> np.ndarray:
    state = np.zeros((G, N_COMPARTMENTS), dtype=float)
    I0 = _as_G_vector(params.initial_infected, G, default=0.0)
    R0 = _as_G_vector(params.initial_recovered, G, default=0.0)
    V0 = _as_G_vector(params.initial_vaccinated, G, default=0.0)
    disease = params.disease
    E0 = I0 * (disease.gamma / max(disease.sigma, 1e-9))
    assigned = I0 + R0 + V0 + E0
    pops = np.asarray(params.populations, dtype=float)
    if np.any(assigned > pops):
        raise ValueError(
            "initial I + R + V + E exceeds population in some district"
        )
    S0 = np.maximum(pops - assigned, 0.0)
    state[:, IDX_S] = S0
    state[:, IDX_E] = E0
    state[:, IDX_I] = I0
    state[:, IDX_R] = R0
    state[:, IDX_V] = V0
    return state


def _as_G_vector(x, G: int, *, default: float = 0.0) -> np.ndarray:
    if x is None:
        return np.full(G, default, dtype=float)
    arr = np.atleast_1d(np.asarray(x, dtype=float))
    if arr.size == 1:
        return np.full(G, float(arr.item()), dtype=float)
    if arr.size != G:
        raise ValueError(f"vector length {arr.size} != G={G}")
    return arr.astype(float)


def days_trajectory(result: ABMResult) -> list[int]:
    return result.days.tolist()
