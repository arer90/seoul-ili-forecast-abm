"""Tau-leap ≡ exp-Euler mean-field equivalence regression lock (DIMENSION #3).

박제 (what this locks):
  averaging the stochastic binomial tau-leap agent kernel
  (``simulation.abm.agent_kernel.run_agent_world``, the 1-exp(-rate) draw world)
  over many seeds CONVERGES to the deterministic exp-Euler ODE trajectory
  (``simulation.sim.stepper.expeuler_step``) of the SAME parameters, in the
  homogeneous well-mixed limit with behavioural heterogeneity neutralised and
  no importation/forcing. This is the empirical justification for the project's
  claim (stepper.py:359-369) that exp-Euler is "the deterministic mean-field of
  the tau-leap agent kernel" — the reason it was made the DEFAULT ABM integrator
  on 2026-06-10.

Why the equivalence is structurally EXACT (not just asymptotic):
  Both kernels use the IDENTICAL discrete-time hazard ``1 - exp(-rate·dt)`` with
  ``dt = 1.0`` day. agent_kernel._hazard == stepper.expeuler_step._frac. So the
  tau-leap is literally a binomial draw whose per-step mean equals the exp-Euler
  flux. The Monte-Carlo error therefore shrinks ~1/sqrt(N·seeds); it is NOT a
  continuous-time vs discrete-time approximation gap.

Setup that pins the mean-field:
  - mixing=None  → agent kernel uses GLOBAL prevalence FoI  lam = beta·I_alive/N_alive,
    which equals the metapop FoI at G=1, mobility=[[1]], daytime=N.
  - theta_mean=100, theta_sd=0, alpha_mean=0, kappa_mean=0 → compliance ≈ 0,
    contact_multiplier ≡ 1 → NO behavioural damping (pure SEIR-V-D).
  - import_rate=0, beta_amp=0, nu=0 (or set), waning=0 → no forcing/importation.
  - exp-Euler run at G=1, dt=1.0, same beta/sigma/gamma/delta(=ifr·... see below).

Note on death: the agent kernel routes I→{R via gamma, D via delta} with total
  outflow rate gamma+delta. The exp-Euler maps that as total rate gamma_tot=gamma+delta
  with ifr = delta/(gamma+delta) (gamma·(1-ifr)=gamma_orig recovery, gamma·ifr=delta
  deaths). The helper ``_expeuler_seir`` below builds exactly that mapping.

Synthetic params only (no DB). Short runtimes (toy N, ~70 days, tens of seeds).
Run:  .venv/bin/python simulation/tests/test_tauleap_expeuler_equivalence.py
"""
from __future__ import annotations

import sys

import numpy as np

from simulation.abm.agent_kernel import run_agent_world, _INITIAL_INFECTED_FRAC
from simulation.sim.stepper import expeuler_step


# ── Behaviour-neutralising knobs (compliance stays ≈ 0 → contact_multiplier ≡ 1) ──
_NEUTRAL = dict(theta_mean=100.0, theta_sd=0.0, alpha_mean=0.0, kappa_mean=0.0)


def _expeuler_seir(N, T, beta, sigma, gamma, delta, nu=0.0, I0=None):
    """Deterministic exp-Euler SEIR-V-D trajectory, G=1 well-mixed, dt=1 day.

    Returns dict of length-T arrays {S,E,I,R,V,D}, matching the agent kernel's
    compartment bookkeeping. I→{R,D} is modelled as a single total outflow rate
    gamma+delta split by ifr=delta/(gamma+delta), which reproduces the agent
    kernel's competing-hazard routing exactly.
    """
    pops = np.array([float(N)])
    M = np.array([[1.0]])
    gamma_tot = gamma + delta
    ifr = (delta / gamma_tot) if gamma_tot > 0 else 0.0
    params = dict(
        beta=beta, sigma=sigma, gamma=gamma_tot, omega=0.0, VE=0.0,
        V_waning=0.0, ifr=ifr, vax_rate=np.array([float(nu)]),
        populations=pops, mobility=M, daytime_pop=np.array([float(N)]),
    )
    if I0 is None:
        I0 = max(1, int(round(N * _INITIAL_INFECTED_FRAC)))
    state = np.array([[float(N - I0), 0.0, float(I0), 0.0, 0.0, 0.0]])
    out = {k: np.zeros(T) for k in ("S", "E", "I", "R", "V", "D")}
    for c, k in enumerate(("S", "E", "I", "R", "V", "D")):
        out[k][0] = state[0, c]
    for t in range(1, T):
        state = expeuler_step(state, 1.0, params_kwargs=params)
        for c, k in enumerate(("S", "E", "I", "R", "V", "D")):
            out[k][t] = state[0, c]
    return out


def _mc_tauleap(N, T, beta, sigma, gamma, delta, nu=0.0, seeds=40, seed0=2000):
    """Monte-Carlo mean of the tau-leap agent kernel over `seeds` realisations.

    Returns dict of length-T mean arrays {S,E,I,R,V,D}.
    """
    acc = {k: np.zeros(T) for k in ("S", "E", "I", "R", "V", "D")}
    for s in range(seeds):
        o = run_agent_world(
            N=N, T_days=T, beta=beta, sigma=sigma, gamma=gamma, delta=delta,
            nu=nu, global_seed=seed0 + s, **_NEUTRAL,
        )
        for k in acc:
            acc[k] += o[k].astype(np.float64)
    return {k: v / seeds for k, v in acc.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────
def test_initial_seed_matches():
    """Both kernels seed exactly round(N·0.01) infectious at t=0 (alignment)."""
    N = 30000
    o = run_agent_world(N=N, T_days=5, beta=0.5, sigma=0.5, gamma=0.3, delta=0.0,
                        nu=0.0, global_seed=1, **_NEUTRAL)
    ref = _expeuler_seir(N, 5, 0.5, 0.5, 0.3, 0.0)
    expect_i0 = max(1, int(round(N * _INITIAL_INFECTED_FRAC)))
    assert int(o["I"][0]) == expect_i0, f"tau-leap I0={o['I'][0]} != {expect_i0}"
    assert int(ref["I"][0]) == expect_i0, f"exp-Euler I0={ref['I'][0]} != {expect_i0}"


def test_behaviour_is_neutralised():
    """Neutral knobs really disable damping: the kernel's compliance machinery
    leaves contact_multiplier ≡ 1, so the run is a pure mean-field SEIR (sanity
    that the equivalence test isn't comparing against a behaviour-damped run)."""
    from simulation.abm.agent_kernel import (
        _logistic, _COMPLIANCE_STEEPNESS, _COMPLIANCE_STRENGTH,
    )
    # risk=0, fatigue=0 at t=0 → margin = -theta_mean (very negative) → compliance≈0
    margin = 0.0 - 0.0 - _NEUTRAL["theta_mean"]
    compliance = float(_logistic(_COMPLIANCE_STEEPNESS * margin))
    contact_mult = 1.0 - _COMPLIANCE_STRENGTH * compliance
    assert compliance < 1e-12, f"compliance not ~0 under neutral knobs: {compliance}"
    assert abs(contact_mult - 1.0) < 1e-12, f"contact_multiplier not ~1: {contact_mult}"


def test_peak_height_and_timing_no_death():
    """MEAN(tau-leap) peak height ≈ exp-Euler peak height; peak DAY identical.
    N=50k, 40 seeds → MC noise ~1/sqrt(N·seeds) ≈ 0.07% of N. Lock <2% height,
    ≤1 day timing (generous vs measured 0.17% / 0 days)."""
    N, T, seeds = 50000, 70, 40
    beta, sigma, gamma, delta = 0.55, 0.5, 0.3, 0.0
    ref = _expeuler_seir(N, T, beta, sigma, gamma, delta)
    mc = _mc_tauleap(N, T, beta, sigma, gamma, delta, seeds=seeds)

    ref_peak, mc_peak = ref["I"].max(), mc["I"].max()
    rel_h = abs(mc_peak - ref_peak) / ref_peak
    dt_timing = abs(int(np.argmax(mc["I"])) - int(np.argmax(ref["I"])))
    assert rel_h < 0.02, f"peak height rel diff {rel_h:.4f} >= 0.02 (ref={ref_peak:.0f}, mc={mc_peak:.0f})"
    assert dt_timing <= 1, f"peak timing diff {dt_timing} days > 1"


def test_full_trajectory_rmse_no_death():
    """Whole I(t) trajectory: normalised RMSE small + final R within tolerance
    (locks attack-rate / final-size agreement, not just the peak)."""
    N, T, seeds = 50000, 70, 40
    beta, sigma, gamma, delta = 0.55, 0.5, 0.3, 0.0
    ref = _expeuler_seir(N, T, beta, sigma, gamma, delta)
    mc = _mc_tauleap(N, T, beta, sigma, gamma, delta, seeds=seeds)

    rmse = np.sqrt(np.mean((mc["I"] - ref["I"]) ** 2)) / ref["I"].max()
    assert rmse < 0.02, f"I(t) normalised RMSE {rmse:.4f} >= 0.02"

    # Final size (cumulative recovered) agreement
    rel_R = abs(mc["R"][-1] - ref["R"][-1]) / max(ref["R"][-1], 1.0)
    assert rel_R < 0.02, f"final R rel diff {rel_R:.4f} >= 0.02 (ref={ref['R'][-1]:.0f}, mc={mc['R'][-1]:.0f})"


def test_competing_hazard_with_death():
    """With delta>0 the I→{R,D} competing-hazard split (total rate gamma+delta,
    ifr=delta/(gamma+delta)) must still match exp-Euler. Locks deaths D(t)."""
    N, T, seeds = 50000, 70, 40
    beta, sigma, gamma, delta = 0.55, 0.5, 0.28, 0.02
    ref = _expeuler_seir(N, T, beta, sigma, gamma, delta)
    mc = _mc_tauleap(N, T, beta, sigma, gamma, delta, seeds=seeds)

    rel_peak = abs(mc["I"].max() - ref["I"].max()) / ref["I"].max()
    assert rel_peak < 0.03, f"(death) peak height rel diff {rel_peak:.4f} >= 0.03"
    # Cumulative deaths agreement (small counts → looser tol)
    rel_D = abs(mc["D"][-1] - ref["D"][-1]) / max(ref["D"][-1], 1.0)
    assert rel_D < 0.06, f"final D rel diff {rel_D:.4f} >= 0.06 (ref={ref['D'][-1]:.0f}, mc={mc['D'][-1]:.0f})"


def test_monte_carlo_error_shrinks_with_more_seeds():
    """The disagreement is Monte-Carlo noise, not bias: averaging MORE seeds
    reduces the peak-height error. Lock that 80-seed error <= 40-seed error (with
    a small slack), which is the ~1/sqrt(seeds) convergence signature that proves
    exp-Euler is the true mean-field limit (a biased kernel would NOT converge)."""
    N, T = 40000, 70
    beta, sigma, gamma, delta = 0.55, 0.5, 0.3, 0.0
    ref = _expeuler_seir(N, T, beta, sigma, gamma, delta)

    mc_few = _mc_tauleap(N, T, beta, sigma, gamma, delta, seeds=10, seed0=5000)
    mc_many = _mc_tauleap(N, T, beta, sigma, gamma, delta, seeds=80, seed0=5000)
    err_few = abs(mc_few["I"].max() - ref["I"].max()) / ref["I"].max()
    err_many = abs(mc_many["I"].max() - ref["I"].max()) / ref["I"].max()
    # 80 seeds should not be worse than 10 seeds (allow tiny slack for finite-sample luck)
    assert err_many <= err_few + 0.003, (
        f"more seeds did not reduce error: 10-seed={err_few:.4f}, 80-seed={err_many:.4f} "
        f"(would contradict mean-field convergence)"
    )
    # And the many-seed error must be genuinely small in absolute terms.
    assert err_many < 0.015, f"80-seed peak error {err_many:.4f} >= 0.015"


if __name__ == "__main__":
    funcs = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    p = f = 0
    for fn in funcs:
        try:
            fn()
            print(f"  ✓ PASS  {fn.__name__}")
            p += 1
        except Exception as e:
            print(f"  ✗ FAIL  {fn.__name__}: {e}")
            f += 1
    print(f"\n  {p} PASS / {f} FAIL")
    sys.exit(1 if f else 0)
