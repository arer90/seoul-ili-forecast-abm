"""
simulation.abm.agent_based
==========================
TRUE agent-based behavioural layer — the ~10 000-individual-household-agents-
per-district model that the thesis (§3.4a, remaining-gap G1) records but the
mean-field :func:`simulation.abm.behavioural.run_coupled_abm` only *aggregates*.

Each district holds ``n_agents`` individual household agents. An agent carries
its **own** fatigue state ``F_i`` and its **own** compliance threshold
``theta_i`` (household heterogeneity). The shared district risk signal
``R_d(t)`` is the perceived prevalence ``I_d / N_d``. Every day each agent makes
an **individual** compliance decision

    comply_i = 1[ R_d - kappa * F_i  >  theta_i ]            (Bernoulli-free, threshold)

and the district transmission scale is driven by the **realised compliance
fraction** over the agents (not a single 0/1):

    s_bar_d = mean_i(comply_i)
    beta_d  = beta_0 * (1 - strength * s_bar_d) ** 2

Mean-field recovery (validated in ``simulation/tests/test_abm_agent_based.py``):
as ``theta_sd -> 0`` and ``n_agents -> inf`` every agent becomes identical and
``s_bar_d -> 1[R_d - kappa*F > theta]`` — exactly the mean-field
``run_coupled_abm``. The agent model therefore *generalises* the mean-field
(adds household heterogeneity + finite-N Monte-Carlo texture) and the
``Multi-Agent`` title becomes literal: ``n_agents * G`` decision-makers.

Vectorised: the per-agent state is a ``(G, n_agents)`` array, so a 10 000-agent,
25-district, 180-day run is ~45 M threshold comparisons — a few seconds.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import numpy as np

from simulation.abm.behavioural import BehaviouralParams, _build_initial_state
from simulation.sim.parameters import IDX_E, IDX_I, MetapopParams, N_COMPARTMENTS
from simulation.sim.stepper import expeuler_step, rk4_step
from simulation.sim.foi import effective_daytime_population
from simulation.sim.metapop_seirvd import SimResult

log = logging.getLogger(__name__)

__all__ = ["AgentABMResult", "run_agent_abm"]


@dataclass
class AgentABMResult:
    """SEIR-V-D trajectory from the agent-based behavioural layer + diagnostics."""
    seir: SimResult
    compliance_fraction: np.ndarray   # (T+1, G) realised compliance fraction
    beta_eff: np.ndarray              # (T+1, G) effective per-district beta
    behaviour: BehaviouralParams
    n_agents: int                     # agents per district (max, when heterogeneous)
    theta_sd: float                   # household threshold heterogeneity (rel.)
    total_agents: int                 # actual sum of per-district agent counts
    n_agents_per_district: np.ndarray | None = None  # (G,) when density-allocated

    def city_I(self) -> np.ndarray:
        return self.seir.city_total("I")

    def mean_compliance(self) -> np.ndarray:
        return self.compliance_fraction.mean(axis=1)


def run_agent_abm(
    metapop_params: MetapopParams,
    behaviour: BehaviouralParams,
    *,
    n_agents: int = 10_000,
    n_agents_per_district: np.ndarray | None = None,
    theta_sd: float = 0.25,
    seed: int = 42,
    verbose: bool = False,
) -> AgentABMResult:
    """Run the individual-agent behavioural ABM coupled to the SEIR-V-D kernel.

    Args:
        metapop_params: kernel inputs (populations, mobility, disease, days, dt).
        behaviour: behavioural parameters (alpha, kappa, tau, theta, strength).
        n_agents: individual household agents **per district** (total = n_agents*G).
            Used as the UNIFORM count when ``n_agents_per_district`` is None.
        n_agents_per_district: optional (G,) integer array of **per-district**
            agent counts (e.g. density-proportional allocation, G-388). When
            given it OVERRIDES the uniform ``n_agents``: district g holds exactly
            ``n_agents_per_district[g]`` decision-makers and the realised
            compliance fraction is a per-district masked mean over only that
            district's active agents (padded columns are excluded — no dilution).
            ``total_agents = sum(n_agents_per_district)``.
        theta_sd: relative SD of the per-agent compliance threshold
            (household heterogeneity); ``theta_sd=0`` + large ``n_agents``
            reproduces the mean-field ``run_coupled_abm``.
        seed: RNG seed for the agent threshold draw (reproducible).
        verbose: log one progress line per ~50 days.

    Returns:
        AgentABMResult (``n_agents_per_district`` populated when heterogeneous).

    Performance: O(days * G * n_max) time, ~G*n_max*8 bytes peak (n_max = max
        per-district count when heterogeneous, else n_agents).
    Side effects: none (pure compute).
    Caller responsibility: n_agents >= 1 (or every n_agents_per_district >= 1);
        behaviour.validate() is enforced.
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
    sub_steps = int(round(1.0 / dt))
    beta_0 = disease.beta
    kappa, theta, strength = behaviour.kappa, behaviour.theta, behaviour.strength
    steepness = float(behaviour.compliance_steepness)  # +inf = hard threshold
    alpha, lambda_R, rho, delta = (
        behaviour.alpha, behaviour.lambda_R, behaviour.rho, behaviour.delta,
    )

    # ── Per-district agent budget ─────────────────────────────────────────
    # Uniform path (n_agents_per_district is None) keeps the legacy (G, n_agents)
    # arrays byte-identical. Density-allocated path pads to n_max and carries a
    # boolean ``active`` mask so inactive (padded) columns never enter the mean.
    if n_agents_per_district is None:
        if n_agents < 1:
            raise ValueError(f"n_agents must be >= 1; got {n_agents}")
        counts = np.full(G, int(n_agents), dtype=np.int64)
    else:
        counts = np.asarray(n_agents_per_district, dtype=np.int64).ravel()
        if counts.size != G:
            raise ValueError(
                f"n_agents_per_district must have length G={G}; got {counts.size}"
            )
        if np.any(counts < 1):
            raise ValueError(
                f"every n_agents_per_district must be >= 1; got min {int(counts.min())}"
            )
    n_max = int(counts.max())
    # active[g, j] = True iff agent column j is one of district g's real agents.
    active = np.arange(n_max)[None, :] < counts[:, None]   # (G, n_max) bool
    n_active = counts.astype(np.float64)[:, None]           # (G, 1) for masked mean

    # Per-agent heterogeneous compliance threshold theta_i (drawn once).
    rng = np.random.default_rng(seed)
    if theta_sd > 0:
        theta_i = theta * (1.0 + theta_sd * rng.standard_normal((G, n_max)))
        np.clip(theta_i, 0.0, None, out=theta_i)
    else:
        theta_i = np.full((G, n_max), float(theta))

    # State
    state = _build_initial_state(metapop_params, G)
    out_state = np.zeros((days + 1, G, N_COMPARTMENTS), dtype=float)
    out_state[0] = state
    incidence = np.zeros((days + 1, G), dtype=float)
    incidence[0] = state[:, IDX_E] * disease.sigma

    R_d = np.zeros(G, dtype=float)                 # district risk signal (shared)
    F_i = np.zeros((G, n_max), dtype=float)        # per-agent fatigue
    C_i = np.zeros((G, n_max), dtype=float)        # per-agent compliance {0,1}
    comp_frac = np.zeros((days + 1, G), dtype=float)
    beta_eff = np.zeros((days + 1, G), dtype=float)
    beta_eff[0] = beta_0

    vax_rate = metapop_params.vaccination_rate
    # Integrator: stable exp-Euler (B-P1) is DEFAULT (mass-conserving, can't diverge).
    # Opt OUT to legacy RK4 with MPH_STABLE_INTEGRATOR=0. nan_to_num clamp = RK4-only.
    _step_fn = rk4_step if os.environ.get("MPH_STABLE_INTEGRATOR") == "0" else expeuler_step
    _needs_clamp = _step_fn is rk4_step
    t0 = time.time()
    for d in range(days):
        I_prev = state[:, IDX_I]
        N_prev = np.maximum(state.sum(axis=1), 1.0)
        # Staggered update matching the mean-field: compliance reads the
        # PREVIOUS fatigue; fatigue accrues from the PREVIOUS compliance.
        # 1) district risk perception R_d (shared signal, mean-field ODE)
        R_d = R_d + (alpha * (I_prev / N_prev) - lambda_R * R_d)
        np.clip(R_d, 0.0, None, out=R_d)
        # 2) per-agent compliance decision (individual threshold, previous F_i).
        #    steepness=+inf ⇒ hard threshold; finite ⇒ smooth per-agent logistic
        #    (each agent's compliance is then a probability in (0,1), and the
        #    realised fraction stays a continuous mean — no FP knife-edge).
        margin = R_d[:, None] - kappa * F_i - theta_i
        if steepness > 1e15:
            C_new = (margin > 0.0).astype(np.float64)
        else:
            C_new = 1.0 / (1.0 + np.exp(-steepness * margin))
        # 3) per-agent fatigue F_i accrues from the PREVIOUS compliance C_i
        F_i = F_i + (delta * C_i - rho * F_i)
        np.clip(F_i, 0.0, None, out=F_i)
        C_i = C_new
        # 4) realised district compliance fraction -> transmission scale.
        #    Masked mean over each district's ACTIVE agents only: padded columns
        #    (heterogeneous-budget path) are zeroed so they never dilute s_bar.
        #    Uniform path: active is all-True ⇒ identical to C_i.mean(axis=1).
        s_bar = (C_i * active).sum(axis=1) / n_active[:, 0]
        comp_frac[d + 1] = s_bar
        beta_district = beta_0 * (1.0 - strength * s_bar) ** 2
        beta_eff[d + 1] = beta_district
        beta_effective_day = float(beta_district.mean())

        # 5) advance SEIR with the effective scalar beta
        kwargs = {
            "beta": beta_effective_day, "sigma": disease.sigma,
            "gamma": disease.gamma, "omega": disease.omega, "VE": disease.VE,
            "V_waning": disease.V_waning, "ifr": disease.ifr,
            "vax_rate": vax_rate, "populations": pops, "mobility": M,
            "daytime_pop": daytime,
        }
        daily_inc = np.zeros(G, dtype=float)
        for _ in range(sub_steps):
            daily_inc += disease.sigma * state[:, IDX_E] * dt
            state = _step_fn(state, dt, params_kwargs=kwargs)
            # RK4-only band-aid: kill within-day overflow (NaN/±inf) every sub-step.
            # The stable exp-Euler default cannot diverge, so it is skipped there
            # (no silent diverge→0 masking). [0,N] cap stays once per day for both.
            if _needs_clamp:
                np.nan_to_num(state, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(state, 0.0, pops[:, None], out=state)
        out_state[d + 1] = state
        incidence[d + 1] = daily_inc
        if verbose and (d + 1) % 50 == 0:
            log.info("day %d  I_city=%.1f  comp_frac=%.3f  beta_mean=%.4f",
                     d + 1, float(state[:, IDX_I].sum()), float(s_bar.mean()),
                     beta_effective_day)

    elapsed = time.time() - t0
    total_agents = int(counts.sum())
    if verbose:
        log.info("agent ABM: %d days x %d gu (%d total agents, %s) in %.2fs",
                 days, G, total_agents,
                 "uniform" if n_agents_per_district is None else "density-allocated",
                 elapsed)
    seir_result = SimResult(
        state=out_state, days=np.arange(days + 1), district_names=names,
        incidence=incidence, params=metapop_params, interventions=[],
        epi_validity={"_skipped": "agent ABM run", "elapsed_sec": elapsed},
    )
    return AgentABMResult(
        seir=seir_result, compliance_fraction=comp_frac, beta_eff=beta_eff,
        behaviour=behaviour, n_agents=n_max, theta_sd=float(theta_sd),
        total_agents=total_agents,
        n_agents_per_district=(None if n_agents_per_district is None
                               else counts.copy()),
    )
