"""Smoke tests for the agent-world ensemble + global sensitivity (sensitivity.py).

Guards the two external-review gap fixes (stochastic ensemble, LHS+PRCC global
SA). Fast (small N / sample counts).
"""
import numpy as np

from simulation.abm.sensitivity import (
    SA_RANGES,
    global_sensitivity,
    lhs,
    prcc,
    run_ensemble,
)


def test_ensemble_structure_and_ci():
    ens = run_ensemble(n_seeds=20)
    assert ens["n_seeds"] == 20
    for name, m in ens["metrics"].items():
        assert m["ci2.5"] <= m["mean"] <= m["ci97.5"], name
        assert len(m["values"]) == 20
    assert 0.0 <= ens["metrics"]["attack_rate"]["mean"] <= 1.0


def test_ensemble_reproducible():
    a = run_ensemble(n_seeds=15, seed0=0)["metrics"]["attack_rate"]["mean"]
    b = run_ensemble(n_seeds=15, seed0=0)["metrics"]["attack_rate"]["mean"]
    assert a == b  # same seeds → identical (deterministic given seed)


def test_variance_stabilization_present():
    vs = run_ensemble(n_seeds=30)["variance_stabilization"]
    assert len(vs["n"]) == len(vs["running_cv_attack"]) >= 2
    assert all(c >= 0 for c in vs["running_cv_attack"])


def test_lhs_shape_and_bounds():
    X, names = lhs(SA_RANGES, n=50, seed=1)
    assert X.shape == (50, len(SA_RANGES))
    for j, nm in enumerate(names):
        lo, hi = SA_RANGES[nm]
        assert X[:, j].min() >= lo - 1e-9 and X[:, j].max() <= hi + 1e-9


def test_prcc_signs_make_sense():
    # higher beta → larger output; higher gamma → smaller. PRCC must reflect it.
    rng = np.random.default_rng(0)
    n = 200
    beta = rng.uniform(0.5, 1.3, n)
    gamma = rng.uniform(0.15, 0.4, n)
    y = 3.0 * beta - 2.0 * gamma + rng.normal(0, 0.05, n)
    pr, pv = prcc(np.column_stack([beta, gamma]), y)
    assert pr[0] > 0.5 and pr[1] < -0.5
    assert pv[0] < 0.05 and pv[1] < 0.05


def test_global_sensitivity_beta_dominates_attack():
    sa = global_sensitivity(n_samples=120)
    pr = sa["prcc"]["attack_rate"]
    assert pr["beta"]["prcc"] > 0.3            # transmission drives attack up
    assert pr["gamma"]["prcc"] < 0             # recovery drives it down
    assert set(pr) == set(SA_RANGES)
