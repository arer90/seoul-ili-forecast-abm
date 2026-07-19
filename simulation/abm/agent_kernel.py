"""Pure-NumPy discrete-time agent kernel for SEIR-V-D simulations.

This module is the Python portability path for the planned Rust agent kernel:
agents are stored in a Structure-of-Arrays layout, transitions are daily
``1-exp(-rate)`` tau-leap draws, and random streams are keyed by day and gu.
It deliberately does not call the legacy RK4 metapopulation stepper.
"""
from __future__ import annotations

import logging

import numpy as np

from simulation.abm.contact_structure import (
    CONTACT_MATRIX_7x7,
    OCCUPATION_EXPOSURE,
)

try:
    from seir_core import run_agent_world_rs
    RUST_BACKEND_AVAILABLE = True
except ImportError:
    run_agent_world_rs = None
    RUST_BACKEND_AVAILABLE = False

__all__ = ["run_agent_world", "RUST_BACKEND_AVAILABLE"]

log = logging.getLogger(__name__)


STATE_S = 0
STATE_E = 1
STATE_I = 2
STATE_R = 3
STATE_V = 4
STATE_D = 5

_COMPARTMENT_NAMES = ("S", "E", "I", "R", "V", "D")
_N_COMPARTMENTS = 6
_N_GU = 25
_N_AGE = 7
_DT = 1.0
_INITIAL_INFECTED_FRAC = 0.01
_RISK_DECAY = 1.0 / 30.0
_FATIGUE_ACCRUAL = 0.05
_COMPLIANCE_STRENGTH = 0.6
_COMPLIANCE_STEEPNESS = 30.0
_INIT_STREAM = 917_531
# Fixed assumption: high-severity agents use 3x baseline death-rate pressure.
HIGH_SEVERITY_DELTA_SCALE = 3.0
_LOW_SEVERITY_DELTA_SCALE = 0.5
_DEFAULT_OCCUPATION_CODE_CATEGORIES = np.array(
    ["office", "office", "service", "other", "essential", "essential"],
    dtype=object,
)


def run_agent_world(
    N,
    T_days,
    beta,
    sigma,
    gamma,
    delta,
    nu,
    mixing_matrix=None,
    age_dist=None,
    population=None,
    contact_matrix=None,
    global_seed=42,
    theta_mean=0.5,
    theta_sd=0.15,
    alpha_mean=0.3,
    kappa_mean=0.5,
    tau_mean=7.0,
    beta_amp=0.0,
    beta_phase=0.0,
    import_rate=0.0,
    initial_vaccinated=None,
    waning=0.0,
    initial_state=None,
    param_dtype=np.float32,   # TDD 검증(test_param_dtype_reproducibility): float32≡float64
    transmission_mode="meanfield",  # A='meanfield'(byte-identical); B='network'; 'hybrid'=A+B
    beta_by_layer=None,       # variant B: layer→per-edge hazard; None=derive from beta
    network_kwargs=None,      # variant B: build_multilayer_network kwargs (hh_size…)
    network_seed=None,        # variant B: contact-graph seed (default = global_seed)
    hybrid_weight=0.5,        # hybrid: FoI = w·mean-field + (1-w)·network
) -> dict:                    # 비트동일(전 compartment) + 메모리 절감. float64 명시도 가능.
    """Run a daily binomial tau-leap SEIR-V-D agent world.

    Args:
        N: Number of agents. Must be >= 1; supports arbitrary values, including
            non-multiples of 25.
        T_days: Number of daily aggregate states to return. Day 0 is the
            initialized world, so exactly ``T_days`` rows are returned.
        beta: Transmission hazard multiplier per infectious contact day.
        sigma: E->I daily rate.
        gamma: I->R daily rate.
        delta: I->D daily rate.
        nu: S->V daily vaccination rate, either scalar or length-25 gu vector.
        mixing_matrix: Optional 25x25 commuter matrix. Rows are normalized if
            they are positive but not already row-stochastic.
        age_dist: Optional 7-vector or 25x7 matrix of age-band probabilities.
        population: Optional synthetic-population SoA dict with ``home_gu``,
            ``age_band``, ``occupation``, ``severity``, and ``work_gu`` arrays.
            When supplied, NumPy is authoritative and the Rust backend is
            bypassed. The kernel runs a half-day work phase and half-day home
            phase: susceptible infection rate is ``beta * lambda_gu`` times an
            occupation exposure multiplier and an age contact-row factor from
            ``CONTACT_MATRIX_7x7`` normalized to mean 1 across agents.
            ``severity=1`` or ``"high"`` applies
            ``delta * HIGH_SEVERITY_DELTA_SCALE``; low severity applies
            ``delta * 0.5``. Death pressure uses home-gu infectious fractions
            weighted by each infectious agent's age and occupation factors.
        global_seed: Integer seed. Daily transition streams use
            ``SeedSequence(global_seed).spawn(day*25 + gu)`` child ordering.
        theta_mean: Mean compliance threshold.
        theta_sd: Relative SD for heterogeneous per-agent thresholds.
        alpha_mean: Risk-sensitivity value assigned to each agent.
        kappa_mean: Fatigue weight assigned to each agent.
        tau_mean: Fatigue time constant assigned to each agent.
        beta_amp: Seasonal forcing amplitude epsilon in
            ``beta(t) = beta * max(0, 1 + epsilon*cos(2*pi*(t-phase)/365.25))``.
            0.0 (default) disables forcing and preserves constant-beta exactly.
        beta_phase: Seasonal forcing peak day phi (sim-day of max transmission).
            Used only when ``beta_amp != 0``.
        import_rate: Flat external infection hazard per susceptible per day.
            A tiny value (e.g. 1e-4) prevents off-season stochastic extinction
            and seeds the epidemic; 0.0 (default) disables importation.
        initial_vaccinated: Optional length-N boolean mask of agents that start
            immune (STATE_V) at t=0, modelling a pre-season vaccination campaign.
            The initial infectious seed is drawn only from the remaining
            susceptibles. None (default) starts everyone susceptible. Enables
            targeted-vs-uniform vaccination counterfactuals.
        waning: Daily R->S immunity-loss hazard. 0.0 (default) = lifelong
            immunity (R is terminal); >0 replenishes susceptibles so the model
            can produce repeated annual epidemics across multiple seasons.
            Requires a population (rich-pop path).
        initial_state: Optional length-N int8 per-agent compartment state
            (STATE_S..STATE_D) to RESUME from instead of fresh-seeding the
            epidemic. None (default) = fresh seed. Used by
            ``run_adaptive_agent_world`` for in-run dynamic-resolution epoch
            chaining (forces the NumPy path; behavioural memory re-inits/epoch).

    Returns:
        Dict with aggregate count arrays ``S,E,I,R,V,D`` each shaped
        ``(T_days,)`` and an ``agents`` dict containing final SoA arrays.
        ``attr_dynamics_active`` is ``True`` only when ``population`` drives
        age, occupation, severity, and scheduled movement dynamics.

    Raises:
        ValueError: On invalid sizes, rates, probabilities, or matrix shapes.

    Performance: O(T_days * N) time and O(N) memory.
    Side effects: none.
    Caller responsibility: disease rates are per-day hazards in the same units.
    """
    # Realism-ablation transmission mode. Variant A ('meanfield') is the current
    # byte-identical baseline (forward_r2=0.722); agent-to-agent 'network'
    # (variant B) is under construction and not yet a valid value.
    if transmission_mode not in ("meanfield", "network", "hybrid"):
        raise ValueError(
            f"transmission_mode={transmission_mode!r} unknown — use 'meanfield' "
            "(variant A), 'network' (variant B), or 'hybrid' (A+B fusion).")
    if transmission_mode in ("network", "hybrid") and population is None:
        raise ValueError(
            f"transmission_mode={transmission_mode!r} requires a population dict "
            "(agent-level contact layers); population=None only supports 'meanfield'.")
    if (
        population is None
        and RUST_BACKEND_AVAILABLE
        and run_agent_world_rs is not None
        and beta_amp == 0.0
        and import_rate == 0.0
        and initial_vaccinated is None
        and waning == 0.0
        and initial_state is None
    ):
        out = run_agent_world_rs(
            N,
            T_days,
            beta,
            sigma,
            gamma,
            delta,
            nu,
            mixing_matrix=mixing_matrix,
            age_dist=age_dist,
            global_seed=global_seed,
            theta_mean=theta_mean,
            theta_sd=theta_sd,
            alpha_mean=alpha_mean,
            kappa_mean=kappa_mean,
            tau_mean=tau_mean,
        )
        out["attr_dynamics_active"] = False
        return out
    if population is not None and RUST_BACKEND_AVAILABLE and run_agent_world_rs is not None:
        log.warning("Rust path ignores population; using numpy")

    N = _validate_positive_int("N", N)
    T_days = _validate_positive_int("T_days", T_days)
    beta = _validate_rate("beta", beta)
    sigma = _validate_rate("sigma", sigma)
    gamma = _validate_rate("gamma", gamma)
    delta = _validate_rate("delta", delta)
    nu_by_gu = _as_gu_rate("nu", nu)
    theta_mean = _validate_nonnegative_finite("theta_mean", theta_mean)
    theta_sd = _validate_nonnegative_finite("theta_sd", theta_sd)
    alpha_mean = _validate_nonnegative_finite("alpha_mean", alpha_mean)
    kappa_mean = _validate_nonnegative_finite("kappa_mean", kappa_mean)
    tau_mean = _validate_tau(tau_mean)
    beta_amp = _validate_nonnegative_finite("beta_amp", beta_amp)
    beta_phase = _validate_nonnegative_finite("beta_phase", beta_phase)
    import_rate = _validate_nonnegative_finite("import_rate", import_rate)
    waning = _validate_nonnegative_finite("waning", waning)
    if waning > 0.0 and population is None:
        raise ValueError(
            "waning (R->S) requires a population; the no-population path does not model it"
        )
    mixing = _normalise_mixing_matrix(mixing_matrix)
    age_prob = _normalise_age_dist(age_dist)
    seed = int(global_seed)
    pop_arrays = (_prepare_population_arrays(population, N, contact_matrix=contact_matrix)
                  if population is not None else None)

    init_rng = np.random.default_rng(np.random.SeedSequence([seed, _INIT_STREAM]))
    if pop_arrays is None:
        home_gu, gu_slices = _build_home_gu(N)
        work_gu = home_gu.copy()
        age_band = _draw_age_bands(init_rng, gu_slices, age_prob, N)
        occupation_multiplier = None
        age_contact_factor = None
        severity_delta_scale = None
    else:
        home_gu = pop_arrays["home_gu"]
        work_gu = pop_arrays["work_gu"]
        age_band = pop_arrays["age_band"]
        gu_slices = None
        occupation_multiplier = pop_arrays["occupation_multiplier"]
        age_contact_factor = pop_arrays["age_contact_factor"]
        severity_delta_scale = pop_arrays["severity_delta_scale"]

    if initial_state is not None:
        # In-run resume (run_adaptive_agent_world): use the carried per-agent
        # compartment state instead of a fresh infection seed. Behavioural memory
        # (fatigue/risk) is re-initialised per epoch — a documented approximation.
        state = np.asarray(initial_state, dtype=np.int8).copy()
        if state.shape != (N,):
            raise ValueError(
                f"initial_state must have shape ({N},), got {state.shape}"
            )
    else:
        state = np.full(N, STATE_S, dtype=np.int8)
        if initial_vaccinated is not None:
            vmask = np.asarray(initial_vaccinated, dtype=bool)
            if vmask.shape != (N,):
                raise ValueError(
                    f"initial_vaccinated must have shape ({N},), got {vmask.shape}"
                )
            state[vmask] = STATE_V
        susceptible_idx = np.flatnonzero(state == STATE_S)
        initial_i = max(1, int(round(N * _INITIAL_INFECTED_FRAC)))
        initial_i = min(initial_i, susceptible_idx.size)
        infected_idx = init_rng.choice(susceptible_idx, size=initial_i, replace=False)
        state[infected_idx] = STATE_I

    # per-agent behavioral parameter storage dtype (외부평가 메모리 권고 — float32 시
    # 행렬당 절반 절감). 기본 float64(재현성 보존). 안전성은 TDD로 검증:
    # test_param_dtype_reproducibility.
    if theta_sd == 0.0:
        theta = np.full(N, theta_mean, dtype=param_dtype)
    else:
        theta = theta_mean * (1.0 + theta_sd * init_rng.standard_normal(N))
        theta = np.clip(theta, 0.0, None).astype(param_dtype, copy=False)
    alpha = np.full(N, alpha_mean, dtype=param_dtype)
    kappa = np.full(N, kappa_mean, dtype=param_dtype)
    tau = np.full(N, tau_mean, dtype=param_dtype)
    rho = np.zeros(N, dtype=np.float64)   # 1/tau decay rate — float64(누적 정밀도)
    finite_tau = np.isfinite(tau)
    rho[finite_tau] = 1.0 / tau[finite_tau]
    fatigue = np.zeros(N, dtype=np.float64)
    compliance = np.zeros(N, dtype=np.float64)
    risk_by_gu = np.zeros(_N_GU, dtype=np.float64)
    if pop_arrays is None:
        alpha_by_gu = _mean_by_gu(alpha, gu_slices)
    else:
        alpha_by_gu = _mean_by_group(alpha, home_gu)

    out = {
        name: np.zeros(T_days, dtype=np.int64)
        for name in _COMPARTMENT_NAMES
    }
    out["attr_dynamics_active"] = population is not None
    _record_counts(state, out, 0)

    # Variant B: build the explicit contact layers once; derive a default per-edge
    # hazard from the kernel beta so citywide FoI is comparable to the mean-field
    # baseline (leak-free per-layer calibration is the validation harness's job).
    _net_layers = _net_beta = _edge_log = None
    if transmission_mode in ("network", "hybrid"):
        from simulation.abm.contact_network import (
            build_multilayer_network, degree_summary)
        _nk = {k: v for k, v in (network_kwargs or {}).items() if k != "provenance"}
        _net_layers = build_multilayer_network(
            population,
            seed=(int(network_seed) if network_seed is not None else int(global_seed)),
            **_nk)
        if beta_by_layer is not None:
            _net_beta = {k: float(v) for k, v in beta_by_layer.items()}
        else:
            deg = degree_summary(_net_layers)
            deg_total = float(deg.get(
                "_total", sum(v for k, v in deg.items() if k != "_total")))
            _net_beta = {name: beta / max(deg_total, 1.0) for name in _net_layers}
        _edge_log = []

    children = np.random.SeedSequence(seed).spawn(max(1, T_days * _N_GU))
    _two_pi_year = 2.0 * np.pi / 365.25
    for out_day in range(1, T_days):
        day = out_day - 1
        if beta_amp != 0.0:
            beta_t = beta * max(
                0.0, 1.0 + beta_amp * np.cos(_two_pi_year * (day - beta_phase))
            )
        else:
            beta_t = beta
        if pop_arrays is not None:
            _step_population_day(
                state=state,
                home_gu=home_gu,
                work_gu=work_gu,
                occupation_multiplier=occupation_multiplier,
                age_contact_factor=age_contact_factor,
                severity_delta_scale=severity_delta_scale,
                beta=beta_t,
                import_rate=import_rate,
                waning=waning,
                sigma=sigma,
                gamma=gamma,
                delta=delta,
                nu_by_gu=nu_by_gu,
                theta=theta,
                alpha=alpha,
                kappa=kappa,
                rho=rho,
                fatigue=fatigue,
                compliance=compliance,
                risk_by_gu=risk_by_gu,
                alpha_by_gu=alpha_by_gu,
                day_seed=children[day * _N_GU],
                transmission_mode=transmission_mode,
                layers=_net_layers,
                beta_by_layer=_net_beta,
                edge_log=_edge_log,
                out_day=out_day,
                hybrid_weight=hybrid_weight,
            )
            _record_counts(state, out, out_day)
            continue

        rng_by_gu = [
            np.random.default_rng(children[day * _N_GU + gu])
            for gu in range(_N_GU)
        ]

        if mixing is not None:
            for gu, sl in enumerate(gu_slices):
                n_gu_agents = sl.stop - sl.start
                if n_gu_agents:
                    work_gu[sl] = rng_by_gu[gu].choice(
                        _N_GU, size=n_gu_agents, p=mixing[gu]
                    ).astype(np.int8, copy=False)

        alive = state != STATE_D
        infected = state == STATE_I
        if mixing is None:
            alive_total = int(alive.sum())
            global_prevalence = (
                float(infected.sum()) / float(alive_total)
                if alive_total > 0 else 0.0
            )
            prevalence_by_work = None
        else:
            present_n = np.bincount(
                work_gu[alive].astype(np.int64), minlength=_N_GU
            ).astype(np.float64)
            present_i = np.bincount(
                work_gu[infected].astype(np.int64), minlength=_N_GU
            ).astype(np.float64)
            prevalence_by_work = np.divide(
                present_i,
                np.maximum(present_n, 1.0),
                out=np.zeros(_N_GU, dtype=np.float64),
                where=present_n > 0.0,
            )
            global_prevalence = 0.0

        home_n = np.bincount(home_gu[alive].astype(np.int64), minlength=_N_GU)
        home_i = np.bincount(home_gu[infected].astype(np.int64), minlength=_N_GU)
        home_prev = np.divide(
            home_i.astype(np.float64),
            np.maximum(home_n.astype(np.float64), 1.0),
            out=np.zeros(_N_GU, dtype=np.float64),
            where=home_n > 0,
        )
        risk_by_gu += alpha_by_gu * home_prev - _RISK_DECAY * risk_by_gu
        np.clip(risk_by_gu, 0.0, None, out=risk_by_gu)

        for gu, sl in enumerate(gu_slices):
            if sl.start == sl.stop:
                continue
            current = state[sl].copy()
            next_state = current.copy()

            margin = risk_by_gu[gu] - kappa[sl] * fatigue[sl] - theta[sl]
            new_compliance = _logistic(_COMPLIANCE_STEEPNESS * margin)
            contact_multiplier = 1.0 - _COMPLIANCE_STRENGTH * new_compliance
            if prevalence_by_work is None:
                lam = beta_t * contact_multiplier * global_prevalence + import_rate
            else:
                lam = beta_t * contact_multiplier * prevalence_by_work[
                    work_gu[sl].astype(np.int64)
                ] + import_rate

            rng = rng_by_gu[gu]
            s_pos = np.flatnonzero(current == STATE_S)
            if s_pos.size:
                inf_rate = lam[s_pos]
                vax_rate = nu_by_gu[gu]
                total_rate = inf_rate + vax_rate
                p_out = _hazard(total_rate)
                p_inf = np.divide(
                    p_out * inf_rate,
                    total_rate,
                    out=np.zeros_like(total_rate),
                    where=total_rate > 0.0,
                )
                p_vax = p_out - p_inf
                u = rng.random(s_pos.size)
                next_state[s_pos[u < p_inf]] = STATE_E
                vax_mask = (u >= p_inf) & (u < (p_inf + p_vax))
                next_state[s_pos[vax_mask]] = STATE_V

            e_pos = np.flatnonzero(current == STATE_E)
            if e_pos.size:
                u = rng.random(e_pos.size)
                next_state[e_pos[u < _hazard(sigma)]] = STATE_I

            i_pos = np.flatnonzero(current == STATE_I)
            if i_pos.size:
                total_rate = gamma + delta
                if total_rate > 0.0:
                    p_out = float(_hazard(total_rate))
                    p_rec = p_out * gamma / total_rate
                    p_die = p_out - p_rec
                    u = rng.random(i_pos.size)
                    next_state[i_pos[u < p_rec]] = STATE_R
                    death_mask = (u >= p_rec) & (u < (p_rec + p_die))
                    next_state[i_pos[death_mask]] = STATE_D

            state[sl] = next_state
            fatigue[sl] += _FATIGUE_ACCRUAL * compliance[sl] - rho[sl] * fatigue[sl]
            np.clip(fatigue[sl], 0.0, None, out=fatigue[sl])
            compliance[sl] = new_compliance

        _record_counts(state, out, out_day)

    out["agents"] = {
        "state": state.copy(),
        "age_band": age_band.copy(),
        "home_gu": home_gu.copy(),
        "work_gu": work_gu.copy(),
        "alpha": alpha.copy(),
        "kappa": kappa.copy(),
        "tau": tau.copy(),
        "theta": theta.copy(),
        "fatigue": fatigue.copy(),
        "compliance": compliance.copy(),
    }
    if transmission_mode in ("network", "hybrid"):
        # Variant B/hybrid: the who-infected-whom transmission tree (empty arrays if
        # no network transmission occurred). infector=-1 means imported/mean-field.
        if _edge_log:
            out["transmission_tree"] = {
                "day": np.concatenate([np.full(e[1].size, e[0], np.int64) for e in _edge_log]),
                "infectee": np.concatenate([e[1] for e in _edge_log]),
                "infector": np.concatenate([e[2] for e in _edge_log]),
                "layer": np.concatenate([e[3] for e in _edge_log]),
                "layer_names": sorted(_net_layers),
                "beta_by_layer": dict(_net_beta),
            }
        else:
            _empty = np.array([], dtype=np.int64)
            out["transmission_tree"] = {
                "day": _empty, "infectee": _empty, "infector": _empty,
                "layer": _empty, "layer_names": sorted(_net_layers or {}),
                "beta_by_layer": dict(_net_beta or {}),
            }
    return out


def _step_population_day(
    *,
    state: np.ndarray,
    home_gu: np.ndarray,
    work_gu: np.ndarray,
    occupation_multiplier: np.ndarray,
    age_contact_factor: np.ndarray,
    severity_delta_scale: np.ndarray,
    beta: float,
    import_rate: float = 0.0,
    waning: float = 0.0,
    sigma: float,
    gamma: float,
    delta: float,
    nu_by_gu: np.ndarray,
    theta: np.ndarray,
    alpha: np.ndarray,
    kappa: np.ndarray,
    rho: np.ndarray,
    fatigue: np.ndarray,
    compliance: np.ndarray,
    risk_by_gu: np.ndarray,
    alpha_by_gu: np.ndarray,
    day_seed: np.random.SeedSequence,
    transmission_mode: str = "meanfield",
    layers=None,
    beta_by_layer=None,
    edge_log=None,
    out_day: int = 0,
    hybrid_weight: float = 0.5,
) -> None:
    """Advance one attribute-driven NumPy day in place.

    ``transmission_mode='meanfield'`` (variant A) is byte-identical to the original
    kernel. ``'network'`` (variant B) replaces the district mean-field force of
    infection with edge-based :func:`~simulation.abm.contact_network.network_foi`
    over ``layers`` (built by the caller) and records who-infected-whom into
    ``edge_log`` via :func:`~simulation.abm.contact_network.sample_infector`.
    """
    rng = np.random.default_rng(day_seed)
    alive = state != STATE_D
    infected = state == STATE_I

    home_prev, home_alive = _prevalence_by_group(home_gu, alive, infected)
    work_prev, _ = _prevalence_by_group(work_gu, alive, infected)

    death_weights = occupation_multiplier * age_contact_factor
    weighted_home_i = np.bincount(
        home_gu[infected].astype(np.int64),
        weights=death_weights[infected],
        minlength=_N_GU,
    ).astype(np.float64)
    home_i_fraction = np.divide(
        weighted_home_i,
        np.maximum(home_alive, 1.0),
        out=np.zeros(_N_GU, dtype=np.float64),
        where=home_alive > 0.0,
    )

    risk_by_gu += alpha_by_gu * home_prev - _RISK_DECAY * risk_by_gu
    np.clip(risk_by_gu, 0.0, None, out=risk_by_gu)

    margin = risk_by_gu[home_gu.astype(np.int64)] - kappa * fatigue - theta
    new_compliance = _logistic(_COMPLIANCE_STEEPNESS * margin)
    contact_multiplier = 1.0 - _COMPLIANCE_STRENGTH * new_compliance
    if transmission_mode in ("network", "hybrid"):
        # Variant B: edge-based FoI over the explicit contact layers (occupation /
        # age / household structure is encoded in the network, not in a scalar),
        # still modulated by each agent's behavioural contact reduction.
        from simulation.abm.contact_network import network_foi  # lazy: circular import
        foi_net = network_foi(state, layers, beta_by_layer) * contact_multiplier
    if transmission_mode in ("meanfield", "hybrid"):
        susceptibility = occupation_multiplier * age_contact_factor * contact_multiplier
        phase_prev = (
            0.5 * work_prev[work_gu.astype(np.int64)]
            + 0.5 * home_prev[home_gu.astype(np.int64)]
        )
        foi_mf = beta * susceptibility * phase_prev
    if transmission_mode == "hybrid":
        # A+B FUSION: per-agent FoI = w·mean-field (smooth aggregate accuracy) +
        # (1-w)·network (agent-to-agent structure). ONE simulation that is both —
        # mean-field smoothness AND the who-infected-whom tree.
        w = float(hybrid_weight)
        infection_rate = w * foi_mf + (1.0 - w) * foi_net + import_rate
    elif transmission_mode == "network":
        infection_rate = foi_net + import_rate
    else:
        infection_rate = foi_mf + import_rate

    next_state = state.copy()

    s_pos = np.flatnonzero(state == STATE_S)
    if s_pos.size:
        inf_rate = infection_rate[s_pos]
        vax_rate = nu_by_gu[home_gu[s_pos].astype(np.int64)]
        total_rate = inf_rate + vax_rate
        p_out = _hazard(total_rate)
        p_inf = np.divide(
            p_out * inf_rate,
            total_rate,
            out=np.zeros_like(total_rate),
            where=total_rate > 0.0,
        )
        p_vax = p_out - p_inf
        u = rng.random(s_pos.size)
        inf_mask = u < p_inf
        next_state[s_pos[inf_mask]] = STATE_E
        vax_mask = (u >= p_inf) & (u < (p_inf + p_vax))
        next_state[s_pos[vax_mask]] = STATE_V
        if transmission_mode in ("network", "hybrid") and edge_log is not None:
            newly = s_pos[inf_mask]
            if newly.size:
                from simulation.abm.contact_network import sample_infector
                infectors, layer_ids = sample_infector(
                    newly, state, layers, beta_by_layer, rng)
                edge_log.append((int(out_day), newly.copy(), infectors, layer_ids))

    e_pos = np.flatnonzero(state == STATE_E)
    if e_pos.size:
        u = rng.random(e_pos.size)
        next_state[e_pos[u < _hazard(sigma)]] = STATE_I

    i_pos = np.flatnonzero(state == STATE_I)
    if i_pos.size:
        death_rate = (
            delta
            * severity_delta_scale[i_pos]
            * home_i_fraction[home_gu[i_pos].astype(np.int64)]
        )
        total_rate = gamma + death_rate
        p_out = _hazard(total_rate)
        p_rec = np.divide(
            p_out * gamma,
            total_rate,
            out=np.zeros_like(total_rate),
            where=total_rate > 0.0,
        )
        p_die = p_out - p_rec
        u = rng.random(i_pos.size)
        next_state[i_pos[u < p_rec]] = STATE_R
        death_mask = (u >= p_rec) & (u < (p_rec + p_die))
        next_state[i_pos[death_mask]] = STATE_D

    if waning > 0.0:
        r_pos = np.flatnonzero(state == STATE_R)
        if r_pos.size:
            u = rng.random(r_pos.size)
            next_state[r_pos[u < _hazard(waning)]] = STATE_S

    state[:] = next_state
    fatigue += _FATIGUE_ACCRUAL * compliance - rho * fatigue
    np.clip(fatigue, 0.0, None, out=fatigue)
    compliance[:] = new_compliance


def _prepare_population_arrays(population: dict, N: int, *,
                               contact_matrix=None) -> dict[str, np.ndarray]:
    if not isinstance(population, dict):
        raise ValueError("population must be a dict of one-dimensional arrays")
    home_gu = _population_int_codes(population, "home_gu", N, 0, _N_GU - 1, np.int8)
    work_gu = _population_int_codes(population, "work_gu", N, 0, _N_GU - 1, np.int8)
    age_band = _population_int_codes(population, "age_band", N, 0, _N_AGE - 1, np.int8)
    occupation = _required_population_array(population, "occupation", N)
    severity = _required_population_array(population, "severity", N)
    return {
        "home_gu": home_gu,
        "work_gu": work_gu,
        "age_band": age_band,
        "occupation_multiplier": _occupation_multiplier_by_agent(occupation),
        "age_contact_factor": _age_contact_factor_by_agent(age_band, contact_matrix),
        "severity_delta_scale": _severity_delta_scale_by_agent(severity),
    }


def _required_population_array(population: dict, key: str, N: int) -> np.ndarray:
    if key not in population:
        raise ValueError(f"population missing required key {key!r}")
    arr = np.asarray(population[key])
    if arr.shape != (N,):
        raise ValueError(f"population[{key!r}] must have shape ({N},); got {arr.shape}")
    return arr


def _population_int_codes(
    population: dict,
    key: str,
    N: int,
    low: int,
    high: int,
    dtype,
) -> np.ndarray:
    raw = _required_population_array(population, key, N)
    try:
        arr = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"population[{key!r}] must contain integer codes") from exc
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"population[{key!r}] must contain finite integer codes")
    codes = arr.astype(np.int64)
    if not np.all(arr == codes):
        raise ValueError(f"population[{key!r}] must contain integer codes")
    if np.any(codes < low) or np.any(codes > high):
        raise ValueError(
            f"population[{key!r}] must be in [{low}, {high}]; got "
            f"[{int(codes.min())}, {int(codes.max())}]"
        )
    return codes.astype(dtype, copy=True)


def _age_contact_factor_by_agent(age_band: np.ndarray, contact_matrix=None) -> np.ndarray:
    """Per-agent age contact factor (row-mean of the 7×7 age matrix, normalized to
    mean 1). ``contact_matrix`` overrides the default POLYMOD assumption — pass an
    alternative (e.g. homogeneous mixing) for structural-robustness analysis."""
    matrix = CONTACT_MATRIX_7x7 if contact_matrix is None else np.asarray(contact_matrix, dtype=np.float64)
    if matrix.shape != (7, 7):   # gray-box contract: 7 age bands (G-/D-5) — a wrong
        raise ValueError(        # shape would silently mis-map row_mean[age_band] below
            f"contact_matrix must be 7×7 (7 age bands); got {matrix.shape}")
    row_mean = matrix.mean(axis=1).astype(np.float64)
    factors = row_mean[age_band.astype(np.int64)].astype(np.float64, copy=True)
    mean_factor = float(factors.mean())
    if mean_factor <= 0.0 or not np.isfinite(mean_factor):
        raise ValueError("population age contact factors must have positive finite mean")
    factors /= mean_factor
    return factors


def _severity_delta_scale_by_agent(severity: np.ndarray) -> np.ndarray:
    arr = np.asarray(severity)
    if np.issubdtype(arr.dtype, np.number):
        values = np.asarray(arr, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("population['severity'] must contain finite values")
        codes = values.astype(np.int64)
        if not np.all(values == codes) or not set(np.unique(codes)).issubset({0, 1}):
            raise ValueError("population['severity'] must contain 0/1 or low/high")
        return np.where(codes == 1, HIGH_SEVERITY_DELTA_SCALE, _LOW_SEVERITY_DELTA_SCALE).astype(
            np.float64
        )

    labels = np.char.lower(arr.astype(str))
    high = np.isin(labels, ["1", "high"])
    low = np.isin(labels, ["0", "low"])
    if not np.all(high | low):
        raise ValueError("population['severity'] must contain 0/1 or low/high")
    return np.where(high, HIGH_SEVERITY_DELTA_SCALE, _LOW_SEVERITY_DELTA_SCALE).astype(
        np.float64
    )


def _occupation_multiplier_by_agent(occupation: np.ndarray) -> np.ndarray:
    arr = np.asarray(occupation)
    out = np.ones(arr.shape[0], dtype=np.float64)
    if np.issubdtype(arr.dtype, np.number):
        values = np.asarray(arr, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("population['occupation'] must contain finite codes")
        codes = values.astype(np.int64)
        if not np.all(values == codes) or np.any(codes < 0):
            raise ValueError("population['occupation'] must contain nonnegative integer codes")
        categories = _occupation_code_categories()
        valid = codes < categories.shape[0]
        for code in np.unique(codes[valid]):
            category = str(categories[int(code)])
            out[codes == code] = OCCUPATION_EXPOSURE.get(category, 1.0)
        return out

    labels = arr.astype(str)
    for label in np.unique(labels):
        category = _occupation_category_for_label(label)
        out[labels == label] = OCCUPATION_EXPOSURE.get(category, 1.0)
    return out


def _occupation_code_categories() -> np.ndarray:
    categories = _DEFAULT_OCCUPATION_CODE_CATEGORIES.copy()
    try:
        from simulation.abm.synthetic_population import INDUSTRY_NAMES
    except Exception:
        return categories

    if len(INDUSTRY_NAMES) > categories.shape[0]:
        padded = np.full(len(INDUSTRY_NAMES), "other", dtype=object)
        padded[: categories.shape[0]] = categories
        categories = padded
    for idx, label in enumerate(INDUSTRY_NAMES[: categories.shape[0]]):
        categories[idx] = _occupation_category_for_label(label) or str(categories[idx])
    return categories


def _occupation_category_for_label(label: object) -> str | None:
    text = str(label).strip().lower()
    if text in OCCUPATION_EXPOSURE:
        return text
    if not text:
        return None
    if any(token in text for token in ("service", "서비스", "판매")):
        return "service"
    if any(token in text for token in ("essential", "기능", "기계", "조립", "단순노무")):
        return "essential"
    if any(token in text for token in ("office", "사무", "관리자", "전문가")):
        return "office"
    if any(token in text for token in ("school", "student", "교육", "학교", "학생")):
        return "school"
    if any(token in text for token in ("unemployed", "무직", "실업")):
        return "unemployed"
    if any(token in text for token in ("other", "농림", "어업", "unavailable")):
        return "other"
    return None


def _prevalence_by_group(
    groups: np.ndarray,
    alive: np.ndarray,
    infected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    alive_n = np.bincount(
        groups[alive].astype(np.int64), minlength=_N_GU
    ).astype(np.float64)
    infected_n = np.bincount(
        groups[infected].astype(np.int64), minlength=_N_GU
    ).astype(np.float64)
    prevalence = np.divide(
        infected_n,
        np.maximum(alive_n, 1.0),
        out=np.zeros(_N_GU, dtype=np.float64),
        where=alive_n > 0.0,
    )
    return prevalence, alive_n


def _validate_positive_int(name: str, value) -> int:
    try:
        ivalue = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer; got {value!r}") from exc
    if ivalue < 1:
        raise ValueError(f"{name} must be >= 1; got {ivalue}")
    return ivalue


def _validate_rate(name: str, value) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and >= 0; got {value!r}")
    return value


def _validate_nonnegative_finite(name: str, value) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and >= 0; got {value!r}")
    return value


def _validate_tau(value) -> float:
    value = float(value)
    if not (value > 0.0):
        raise ValueError(f"tau_mean must be positive or +inf; got {value!r}")
    return value


def _as_gu_rate(name: str, value) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        rate = _validate_rate(name, float(arr))
        return np.full(_N_GU, rate, dtype=np.float64)
    if arr.shape != (_N_GU,):
        raise ValueError(f"{name} must be scalar or shape ({_N_GU},); got {arr.shape}")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError(f"{name} must contain finite nonnegative rates")
    return arr.astype(np.float64, copy=True)


def _normalise_mixing_matrix(mixing_matrix) -> np.ndarray | None:
    if mixing_matrix is None:
        return None
    M = np.asarray(mixing_matrix, dtype=np.float64)
    if M.shape != (_N_GU, _N_GU):
        raise ValueError(
            f"mixing_matrix must have shape ({_N_GU}, {_N_GU}); got {M.shape}"
        )
    if not np.all(np.isfinite(M)) or np.any(M < 0.0):
        raise ValueError("mixing_matrix must contain finite nonnegative values")
    row_sum = M.sum(axis=1)
    if np.any(row_sum <= 0.0):
        raise ValueError("mixing_matrix rows must have positive sums")
    return M / row_sum[:, None]


def _normalise_age_dist(age_dist) -> np.ndarray:
    if age_dist is None:
        probs = np.asarray([0.06, 0.09, 0.12, 0.16, 0.18, 0.20, 0.19])
        return np.tile(probs / probs.sum(), (_N_GU, 1))
    arr = np.asarray(age_dist, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape != (_N_AGE,):
            raise ValueError(f"age_dist must have length {_N_AGE}; got {arr.shape}")
        probs = _normalise_probability_row(arr, "age_dist")
        return np.tile(probs, (_N_GU, 1))
    if arr.shape != (_N_GU, _N_AGE):
        raise ValueError(
            f"age_dist must be shape ({_N_AGE},) or ({_N_GU}, {_N_AGE}); got {arr.shape}"
        )
    out = np.empty_like(arr, dtype=np.float64)
    for gu in range(_N_GU):
        out[gu] = _normalise_probability_row(arr[gu], f"age_dist[{gu}]")
    return out


def _normalise_probability_row(row: np.ndarray, name: str) -> np.ndarray:
    if not np.all(np.isfinite(row)) or np.any(row < 0.0):
        raise ValueError(f"{name} must contain finite nonnegative probabilities")
    total = float(row.sum())
    if total <= 0.0:
        raise ValueError(f"{name} must have positive mass")
    return row / total


def _build_home_gu(N: int) -> tuple[np.ndarray, list[slice]]:
    counts = np.full(_N_GU, N // _N_GU, dtype=np.int64)
    counts[: N % _N_GU] += 1
    home_gu = np.repeat(np.arange(_N_GU, dtype=np.int8), counts).astype(np.int8)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    gu_slices = [slice(int(offsets[i]), int(offsets[i + 1])) for i in range(_N_GU)]
    return home_gu, gu_slices


def _draw_age_bands(
    rng: np.random.Generator,
    gu_slices: list[slice],
    age_prob: np.ndarray,
    N: int,
) -> np.ndarray:
    age_band = np.empty(N, dtype=np.int8)
    for gu, sl in enumerate(gu_slices):
        n = sl.stop - sl.start
        if n:
            age_band[sl] = rng.choice(_N_AGE, size=n, p=age_prob[gu]).astype(np.int8)
    return age_band


def _mean_by_gu(values: np.ndarray, gu_slices: list[slice]) -> np.ndarray:
    out = np.zeros(_N_GU, dtype=np.float64)
    for gu, sl in enumerate(gu_slices):
        out[gu] = float(values[sl].mean()) if sl.stop > sl.start else 0.0
    return out


def _mean_by_group(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    counts = np.bincount(groups.astype(np.int64), minlength=_N_GU).astype(np.float64)
    totals = np.bincount(
        groups.astype(np.int64), weights=values.astype(np.float64), minlength=_N_GU
    ).astype(np.float64)
    return np.divide(
        totals,
        np.maximum(counts, 1.0),
        out=np.zeros(_N_GU, dtype=np.float64),
        where=counts > 0.0,
    )


def _hazard(rate):
    return 1.0 - np.exp(-np.asarray(rate, dtype=np.float64) * _DT)


def _logistic(x: np.ndarray) -> np.ndarray:
    z = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def _record_counts(state: np.ndarray, out: dict[str, np.ndarray], day: int) -> None:
    counts = np.bincount(state.astype(np.int64), minlength=_N_COMPARTMENTS)
    for idx, name in enumerate(_COMPARTMENT_NAMES):
        out[name][day] = int(counts[idx])
