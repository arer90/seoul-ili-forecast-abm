"""B-P1 (M7): flux-conservative exponential-Euler integrator.

Unconditionally positive + exactly mass-conserving — the stable replacement for
the explicit RK4 that diverges (~1/3 of behavioural runs) and needs box-clamp +
nan_to_num. Opt-in (MPH_STABLE_INTEGRATOR=1); RK4 stays default until validated
on the §4.16 grid (a retrain-time check).
"""
import numpy as np

from simulation.sim.stepper import expeuler_step, rk4_step


def _params(beta=0.5, G=2, **over):
    M = np.full((G, G), 0.1 / (G - 1)) if G > 1 else np.array([[1.0]])
    np.fill_diagonal(M, 0.9)
    p = {"beta": beta, "sigma": 0.2, "gamma": 0.1, "omega": 1.0 / 180,
         "VE": 0.5, "V_waning": 1.0 / 180, "ifr": 0.001,
         "vax_rate": np.full(G, 0.001), "populations": np.full(G, 1000.0),
         "mobility": M, "daytime_pop": None}
    p.update(over)
    return p


def _state(G=2):
    s = np.zeros((G, 6))
    s[:, 0] = 990.0  # S
    s[:, 2] = 10.0   # I
    return s


def test_mass_conserved_to_machine_precision():
    s, p = _state(), _params()
    n0 = s.sum()
    for _ in range(80):
        s = expeuler_step(s, 0.25, params_kwargs=p)
    assert abs(s.sum() - n0) < 1e-8


def test_unconditionally_positive_under_extreme_stiffness():
    # huge beta + large dt — an explicit RK4 stage overshoots negative here.
    s, p = _state(), _params(beta=50.0)
    for _ in range(30):
        s = expeuler_step(s, 1.0, params_kwargs=p)
        assert np.all(s >= -1e-9), f"compartment went negative: {s.min()}"
    assert np.all(np.isfinite(s))


def test_pure_decay_matches_analytic():
    # beta=0, no vaccination/waning → I decays exactly as I·exp(-gamma·dt)
    p = _params(beta=0.0, vax_rate=np.zeros(2), omega=0.0, V_waning=0.0)
    s = _state()
    i0 = s[:, 2].copy()
    s2 = expeuler_step(s, 1.0, params_kwargs=p)
    assert np.allclose(s2[:, 2], i0 * np.exp(-p["gamma"] * 1.0), atol=1e-9)


def test_agrees_with_rk4_on_a_stable_case():
    s, p = _state(), _params(beta=0.3)
    a = expeuler_step(s, 0.1, params_kwargs=p)
    b = rk4_step(s, 0.1, params_kwargs=p)
    assert np.allclose(a[:, :5], b[:, :5], rtol=0.1, atol=2.0)
