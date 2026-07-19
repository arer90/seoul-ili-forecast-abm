"""simulation.abm.adaptive_world — In-run dynamic-agent SEIR-V-D.

``run_adaptive_agent_world`` wires the in-run controllers (``adaptive_allocation``)
into the production ``run_agent_world`` kernel: the simulation runs in **epochs**, and
BETWEEN epochs the agent population is **resampled** to a resolution ``M`` that scales
with current prevalence (peak → high-res, off-season → low-res), conserving the
represented population ``P`` and remaining unbiased.

This is adaptive mesh refinement for an ABM — spend agents where resolution matters
(prevalence) without changing the epidemic in expectation. (사용자 제안 2026-06-05:
"agent는 가변적(동적)으로 변화에 따라서 최적화" — between-run 수렴 + **in-run** 적응.)

Gray-box contract
-----------------
- Reuses ``run_agent_world`` (no SEIR duplication) via its ``population`` path →
  arbitrary agent order survives resampling (no gu-contiguity assumption) + its
  new ``initial_state`` resume param chains epoch state.
- **Conserves P**: every reported compartment count is weighted by ``P / M_epoch``;
  since the kernel keeps all M agents in some compartment (incl. D), each day's
  S+E+I+R+V+D == P (within float rounding).
- **Unbiased**: ``resample_weighted`` is particle-filter systematic
  (E[copies_i] ∝ w_i); with uniform within-epoch weights this is unbiased
  sub/over-sampling → the weighted epidemic curve matches a fixed-N run within MC.
- Behavioural memory (fatigue/risk) re-inits per epoch — a documented approximation
  (~30-day half-life vs epoch_len ~7-14 d).

Performance: O(total_days * mean(M)) time, O(max(M)) memory. Side effects: none.
Caller responsibility: disease rates are per-day hazards; peak_prevalence > 0.
"""
from __future__ import annotations

import math

import numpy as np

from simulation.abm.adaptive_allocation import AdaptiveAllocator, resample_weighted
from simulation.abm.agent_kernel import STATE_D, STATE_I, _N_AGE, _N_GU, run_agent_world

_COMPARTMENTS = ("S", "E", "I", "R", "V", "D")
_N_OCCUPATION = 6   # _DEFAULT_OCCUPATION_CODE_CATEGORIES length


def _build_demo_population(n: int, rng: np.random.Generator) -> dict:
    """Synthetic SoA population for the kernel population path (arbitrary order OK).

    Returns dict with int-code arrays: home_gu/work_gu (0..24), age_band (0..6),
    occupation (0..5), severity (0/1). 20% commute to a random work gu;
    ~10% high severity.
    """
    home_gu = rng.integers(0, _N_GU, size=n).astype(np.int64)
    work_gu = home_gu.copy()
    commute = rng.random(n) < 0.2
    n_commute = int(commute.sum())
    if n_commute:
        work_gu[commute] = rng.integers(0, _N_GU, size=n_commute)
    return {
        "home_gu": home_gu,
        "work_gu": work_gu,
        "age_band": rng.integers(0, _N_AGE, size=n).astype(np.int64),
        "occupation": rng.integers(0, _N_OCCUPATION, size=n).astype(np.int64),
        "severity": (rng.random(n) < 0.1).astype(np.int64),
    }


def run_adaptive_agent_world(
    n0: int,
    total_days: int,
    *,
    allocator: AdaptiveAllocator,
    peak_prevalence: float,
    beta: float,
    sigma: float,
    gamma: float,
    delta: float = 0.0,
    nu: float = 0.0,
    population_size: int | None = None,
    epoch_len: int = 7,
    global_seed: int = 42,
    theta_mean: float = 0.5,
    theta_sd: float = 0.15,
    alpha_mean: float = 0.3,
    kappa_mean: float = 0.5,
    tau_mean: float = 7.0,
    beta_amp: float = 0.0,
    beta_phase: float = 0.0,
    import_rate: float = 0.0,
) -> dict:
    """Run an in-run dynamic-resolution agent SEIR-V-D world.

    Args:
        n0: initial agent resolution (epoch 0). >= 1.
        total_days: target number of simulated days (>= 1).
        allocator: AdaptiveAllocator — maps prevalence → next-epoch effective N.
        peak_prevalence: reference peak prevalence in (0,1] for the allocator scale.
        beta/sigma/gamma/delta/nu: per-day SEIR-V-D hazards.
        population_size: represented population P (default = n0). Weighted counts
            are scaled to this; conserved every day.
        epoch_len: days between resampling events (resolution refresh cadence).
        global_seed: base RNG seed (per-epoch streams derived deterministically).
        theta_*/alpha_*/kappa_*/tau_*: behavioural coupling params (per agent).
        beta_amp/beta_phase: seasonal forcing. import_rate: external seeding hazard.

    Returns:
        dict: ``S,E,I,R,V,D`` each float ndarray (length L ≈ total_days) of
        **population-weighted** daily counts (Σ over compartments == P each day);
        ``n_trajectory`` int ndarray (per-day agent resolution M);
        ``P`` (float), ``n_epochs`` (int).

    Raises:
        ValueError: invalid allocator / peak_prevalence / sizes.
    """
    if not isinstance(allocator, AdaptiveAllocator):
        raise ValueError("allocator must be an AdaptiveAllocator")
    if not (peak_prevalence > 0.0) or not math.isfinite(peak_prevalence):
        raise ValueError(f"peak_prevalence must be > 0; got {peak_prevalence}")
    n0 = int(n0)
    total_days = int(total_days)
    epoch_len = int(epoch_len)
    if n0 < 1 or total_days < 1 or epoch_len < 1:
        raise ValueError("n0 / total_days / epoch_len must each be >= 1")

    P = float(population_size if population_size is not None else n0)
    rng = np.random.default_rng(int(global_seed))
    pop = _build_demo_population(n0, rng)
    state = None                      # epoch 0 = fresh seed
    M = n0
    series: dict[str, list[float]] = {k: [] for k in _COMPARTMENTS}
    n_traj: list[int] = []

    n_epochs = max(1, math.ceil(total_days / epoch_len))
    for epoch in range(n_epochs):
        res = run_agent_world(
            N=M,
            T_days=epoch_len + 1,        # 1 init row + epoch_len stepped rows
            beta=beta, sigma=sigma, gamma=gamma, delta=delta, nu=nu,
            population=pop,
            initial_state=state,
            global_seed=int(global_seed) + epoch + 1,
            theta_mean=theta_mean, theta_sd=theta_sd, alpha_mean=alpha_mean,
            kappa_mean=kappa_mean, tau_mean=tau_mean,
            beta_amp=beta_amp, beta_phase=beta_phase, import_rate=import_rate,
        )
        w = P / float(M)
        # epoch 0 keeps the t=0 init row; resumed epochs drop row 0 (== prev final)
        start = 0 if epoch == 0 else 1
        n_rows = 0
        for name in _COMPARTMENTS:
            rows = np.asarray(res[name], dtype=np.float64)[start:]
            series[name].extend((rows * w).tolist())
            n_rows = len(rows)
        n_traj.extend([M] * n_rows)

        # ── resample resolution for the next epoch ∝ realized prevalence ──
        final_state = np.asarray(res["agents"]["state"])
        alive = int((final_state != STATE_D).sum())
        prev = float((final_state == STATE_I).sum()) / float(max(alive, 1))
        M_new = int(allocator.target_n(prev, peak_prevalence))
        idx, _ = resample_weighted(np.ones(M, dtype=np.float64), M_new, rng)
        state = final_state[idx].copy()
        pop = {key: np.asarray(val)[idx].copy() for key, val in pop.items()}
        M = M_new

    # trim to total_days (epoch 0's extra init row can overshoot by <= 1 epoch)
    L = min(len(series["S"]), total_days)
    out: dict = {k: np.asarray(series[k][:L], dtype=np.float64) for k in _COMPARTMENTS}
    out["n_trajectory"] = np.asarray(n_traj[:L], dtype=np.int64)
    out["P"] = P
    out["n_epochs"] = n_epochs
    return out
