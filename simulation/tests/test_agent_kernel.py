"""Tests for the pure-NumPy SEIR-V-D agent kernel."""
from __future__ import annotations

import numpy as np

from simulation.abm.agent_kernel import run_agent_world


def _mean_field_oracle(initial, T_days, beta, sigma, gamma, delta, nu):
    traj = {name: np.zeros(T_days, dtype=np.float64) for name in "SEIRVD"}
    for name in traj:
        traj[name][0] = float(initial[name])

    p_ei = 1.0 - np.exp(-sigma)
    total_i_rate = gamma + delta
    if total_i_rate > 0.0:
        p_i_out = 1.0 - np.exp(-total_i_rate)
        p_ir = p_i_out * gamma / total_i_rate
        p_id = p_i_out - p_ir
    else:
        p_ir = p_id = 0.0

    for t in range(1, T_days):
        S = traj["S"][t - 1]
        E = traj["E"][t - 1]
        I = traj["I"][t - 1]
        R = traj["R"][t - 1]
        V = traj["V"][t - 1]
        D = traj["D"][t - 1]
        alive = max(S + E + I + R + V, 1.0)
        lam = beta * I / alive
        total_s_rate = lam + nu
        if total_s_rate > 0.0:
            p_s_out = 1.0 - np.exp(-total_s_rate)
            p_se = p_s_out * lam / total_s_rate
            p_sv = p_s_out - p_se
        else:
            p_se = p_sv = 0.0

        s_to_e = S * p_se
        s_to_v = S * p_sv
        e_to_i = E * p_ei
        i_to_r = I * p_ir
        i_to_d = I * p_id

        traj["S"][t] = S - s_to_e - s_to_v
        traj["E"][t] = E + s_to_e - e_to_i
        traj["I"][t] = I + e_to_i - i_to_r - i_to_d
        traj["R"][t] = R + i_to_r
        traj["V"][t] = V + s_to_v
        traj["D"][t] = D + i_to_d
    return traj


def test_mass_conservation():
    r = run_agent_world(
        500, 30, beta=0.45, sigma=0.35, gamma=0.18, delta=0.003, nu=0.001
    )
    total = r["S"] + r["E"] + r["I"] + r["R"] + r["V"] + r["D"]
    assert np.array_equal(total, np.full(30, 500))


def test_mean_field_oracle():
    N = 5000
    T_days = 90
    params = dict(beta=0.55, sigma=0.35, gamma=0.18, delta=0.001, nu=0.0)
    r = run_agent_world(
        N,
        T_days,
        global_seed=7,
        theta_mean=10.0,
        theta_sd=1e-6,
        alpha_mean=0.0,
        kappa_mean=0.0,
        tau_mean=1e9,
        **params,
    )
    initial = {name: r[name][0] for name in "SEIRVD"}
    mf = _mean_field_oracle(initial, T_days, **params)

    sim_epi = np.vstack([r[name] for name in "EIR"]).astype(np.float64)
    mf_epi = np.vstack([mf[name] for name in "EIR"])
    rel_error = np.sum(np.abs(sim_epi - mf_epi)) / np.sum(np.maximum(mf_epi, 1.0))
    assert rel_error < 0.10

    sim_peak_day = int(np.argmax(r["I"]))
    mf_peak_day = int(np.argmax(mf["I"]))
    assert abs(sim_peak_day - mf_peak_day) <= 3

    sim_peak = float(np.max(r["I"]))
    mf_peak = float(np.max(mf["I"]))
    assert abs(sim_peak - mf_peak) / mf_peak < 0.15


def test_determinism():
    kwargs = dict(
        N=1000,
        T_days=35,
        beta=0.50,
        sigma=0.30,
        gamma=0.16,
        delta=0.002,
        nu=0.001,
    )
    a = run_agent_world(global_seed=123, **kwargs)
    b = run_agent_world(global_seed=123, **kwargs)
    c = run_agent_world(global_seed=124, **kwargs)

    for name in "SEIRVD":
        assert np.array_equal(a[name], b[name])
    assert np.array_equal(a["agents"]["state"], b["agents"]["state"])

    differs = any(not np.array_equal(a[name], c[name]) for name in "SEIRVD")
    assert differs


def test_no_nan_inf():
    r = run_agent_world(
        1000,
        40,
        beta=2.0,
        sigma=0.8,
        gamma=0.05,
        delta=0.02,
        nu=0.01,
        global_seed=5,
        theta_mean=0.1,
        theta_sd=0.3,
        alpha_mean=1.0,
        kappa_mean=0.5,
        tau_mean=3.0,
    )
    for name in "SEIRVD":
        assert np.all(np.isfinite(r[name]))
    for value in r["agents"].values():
        assert np.all(np.isfinite(np.asarray(value, dtype=np.float64)))


def test_n_flexibility():
    for N in [100, 1000, 37500]:
        r = run_agent_world(
            N, 10, beta=0.35, sigma=0.30, gamma=0.12, delta=0.001, nu=0.001
        )
        assert r["S"].shape == (10,)
        total = r["S"] + r["E"] + r["I"] + r["R"] + r["V"] + r["D"]
        assert np.array_equal(total, np.full(10, N))


def test_forcing_backcompat_amp_zero():
    """beta_amp=0 and import_rate=0 (defaults) reproduce the unforced kernel
    exactly, including the Rust fast path (which is only taken when forcing is
    off). beta_phase is ignored when amplitude is zero."""
    kw = dict(
        N=1000, T_days=40, beta=0.5, sigma=0.3, gamma=0.16,
        delta=0.002, nu=0.001, global_seed=42,
    )
    a = run_agent_world(**kw)
    b = run_agent_world(beta_amp=0.0, beta_phase=120.0, import_rate=0.0, **kw)
    for name in "SEIRVD":
        assert np.array_equal(a[name], b[name])


def _minipop(N):
    return {
        "home_gu": (np.arange(N) % 25).astype(np.int8),
        "work_gu": (np.arange(N) % 25).astype(np.int8),
        "age_band": np.full(N, 3, dtype=np.int8),
        "sex": np.zeros(N, dtype=np.int8),
        "occupation": np.full(N, "office", dtype=object),
        "severity": np.zeros(N, dtype=np.int8),
    }


def test_waning_requires_population():
    """waning (R->S) is only modelled on the rich-population path; the
    no-population path rejects it instead of silently ignoring it."""
    import pytest

    with pytest.raises(ValueError, match="waning"):
        run_agent_world(
            1000, 30, beta=0.5, sigma=0.3, gamma=0.18, delta=0.002, nu=0.0,
            waning=0.01,
        )


def test_waning_replenishes_susceptibles():
    """waning>0 feeds R back to S so susceptibles replenish (enabling repeated
    waves); waning=0 keeps R terminal. Mass is conserved either way."""
    N = 4000
    common = dict(
        N=N, T_days=140, beta=0.6, sigma=0.45, gamma=0.30, delta=0.001, nu=0.0,
        population=_minipop(N), global_seed=0,
        theta_mean=10.0, alpha_mean=0.0, kappa_mean=0.0, tau_mean=1e9,
    )
    no_wane = run_agent_world(**common, waning=0.0)
    wane = run_agent_world(**common, waning=0.03)
    # with waning, the recovered pool is drained back into S over time
    assert wane["R"][-1] < no_wane["R"][-1]
    assert wane["S"][-1] > no_wane["S"][-1]
    for r in (no_wane, wane):
        total = r["S"] + r["E"] + r["I"] + r["R"] + r["V"] + r["D"]
        assert np.array_equal(total, np.full(140, N))


def test_seasonal_forcing_shifts_peak():
    """A later forcing phase delays the epidemic peak; importation keeps the
    forced epidemic from going extinct in the low-transmission window. This is
    the mechanism that lets the ABM match the real ~20-week ILI season instead
    of the constant-beta 3-week burnout."""
    base = dict(
        N=8000, T_days=220, beta=0.30, sigma=0.45, gamma=0.18,
        delta=0.002, nu=0.0002, global_seed=0,
        theta_mean=10.0, theta_sd=1e-6, alpha_mean=0.0,
        kappa_mean=0.0, tau_mean=1e9, import_rate=3e-4,
    )
    early = run_agent_world(beta_amp=0.6, beta_phase=40, **base)
    late = run_agent_world(beta_amp=0.6, beta_phase=130, **base)
    early_peak = int(np.argmax(early["I"]))
    late_peak = int(np.argmax(late["I"]))
    assert late_peak > early_peak + 10
    assert early["I"].max() > 50 and late["I"].max() > 50
    for r in (early, late):
        total = r["S"] + r["E"] + r["I"] + r["R"] + r["V"] + r["D"]
        assert np.array_equal(total, np.full(base["T_days"], base["N"]))
