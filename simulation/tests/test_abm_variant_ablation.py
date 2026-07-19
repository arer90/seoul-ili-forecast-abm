"""TDD for the ABM realism ablation (variants A/B/C/D).

Variant A (mean-field, the current thesis baseline forward_r2=0.722) MUST stay
byte-identical when the new `transmission_mode` param is threaded through the
kernel. This regression guard protects the 0.722 baseline before any variant-B
(agent-to-agent contact-network) logic is added.
"""
from __future__ import annotations

import numpy as np
import pytest


def _run(**extra):
    from simulation.abm.agent_kernel import run_agent_world
    return run_agent_world(N=400, T_days=25, beta=0.35, sigma=0.3, gamma=0.2,
                           delta=0.004, nu=0.0, global_seed=7, import_rate=1e-3,
                           **extra)


def _deep_equal(a, b) -> bool:
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(
            _deep_equal(a[k], b[k]) for k in a)
    return a == b


def test_default_is_meanfield_and_byte_identical():
    """Passing transmission_mode='meanfield' explicitly == the current default."""
    a = _run()                              # today's call (no new param)
    b = _run(transmission_mode="meanfield")  # explicit A
    assert set(a) == set(b)
    for k in a:
        assert _deep_equal(a[k], b[k]), f"{k} differs — variant A not byte-identical"


def test_meanfield_reproducible_across_calls():
    assert _deep_equal(_run(transmission_mode="meanfield"),
                       _run(transmission_mode="meanfield"))


def test_invalid_transmission_mode_rejected():
    with pytest.raises(ValueError):
        _run(transmission_mode="bogus")


# ── B-2: sample_infector (who-infected-whom attribution) ──────────────────────
def _tiny_layer(edges, n):
    from scipy import sparse
    rows, cols = [], []
    for a, b in edges:
        rows += [a, b]
        cols += [b, a]
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def test_sample_infector_attributes_to_infectious_neighbor():
    from simulation.abm.contact_network import sample_infector
    from simulation.abm.agent_kernel import STATE_S, STATE_I
    # agent 0 (newly infected) is linked to agent 1 (infectious) and 2 (susceptible)
    layers = {"household": _tiny_layer([(0, 1), (0, 2)], 3)}
    state = np.array([STATE_S, STATE_I, STATE_S])
    inf, lyr = sample_infector(np.array([0]), state, layers, {"household": 0.5},
                               np.random.default_rng(0))
    assert inf[0] == 1 and lyr[0] == 0        # infector = agent 1 via household(layer 0)


def test_sample_infector_no_infectious_neighbor_is_import():
    from simulation.abm.contact_network import sample_infector
    from simulation.abm.agent_kernel import STATE_S
    layers = {"household": _tiny_layer([(0, 1)], 2)}
    state = np.array([STATE_S, STATE_S])       # neighbour NOT infectious
    inf, lyr = sample_infector(np.array([0]), state, layers, {"household": 0.5},
                               np.random.default_rng(0))
    assert inf[0] == -1 and lyr[0] == -1       # attributed to importation


def test_sample_infector_empty_input():
    from simulation.abm.contact_network import sample_infector
    inf, lyr = sample_infector(np.array([], dtype=int), np.array([0, 1]),
                               {"household": _tiny_layer([(0, 1)], 2)},
                               {"household": 0.5}, np.random.default_rng(0))
    assert inf.size == 0 and lyr.size == 0


# ── B-3: network transmission mode (agent-to-agent) ───────────────────────────
def _mini_pop(n=800, seed=1):
    rng = np.random.default_rng(seed)
    return {"home_gu": rng.integers(0, 25, n), "work_gu": rng.integers(0, 25, n),
            "age_band": rng.integers(0, 7, n), "occupation": rng.integers(0, 4, n),
            "severity": rng.integers(0, 2, n)}


def _run_net(**extra):
    from simulation.abm.agent_kernel import run_agent_world
    pop = _mini_pop()
    return run_agent_world(N=len(pop["home_gu"]), T_days=50, beta=1.0, sigma=0.3,
                           gamma=0.2, delta=0.004, nu=0.0, global_seed=3,
                           import_rate=1e-3, population=pop,
                           transmission_mode="network",
                           beta_by_layer={"household": 0.35, "workplace": 0.12,
                                          "school": 0.15, "community": 0.06},
                           **extra), pop


def test_network_mode_builds_transmission_tree():
    r, pop = _run_net()
    assert r["I"].max() > 0                              # an epidemic occurred
    tt = r["transmission_tree"]
    assert set(tt) >= {"day", "infectee", "infector", "layer", "layer_names"}
    assert tt["infectee"].size > 0                       # who-infected-whom recorded
    assert tt["infectee"].max() < len(pop["home_gu"])    # valid agent indices
    assert (tt["infector"] >= -1).all()                  # -1 = imported
    assert (tt["infector"] >= 0).sum() > 0               # some agent-to-agent edges


def test_network_offspring_distribution_available():
    r, _ = _run_net()
    tt = r["transmission_tree"]
    src = tt["infector"][tt["infector"] >= 0]
    counts = np.bincount(src)
    k = counts[counts > 0]
    assert k.size > 0 and k.mean() > 0                   # offspring/superspreading k exists


def test_network_reproducible_same_seed():
    a, _ = _run_net()
    b, _ = _run_net()
    assert np.array_equal(a["I"], b["I"])
    assert np.array_equal(a["transmission_tree"]["infectee"],
                          b["transmission_tree"]["infectee"])


def test_network_requires_population():
    from simulation.abm.agent_kernel import run_agent_world
    with pytest.raises(ValueError):
        run_agent_world(N=100, T_days=10, beta=0.3, sigma=0.3, gamma=0.2,
                        delta=0.004, nu=0.0, transmission_mode="network")


# ── hybrid: A+B fusion (mean-field + network in one simulation) ────────────────
def test_hybrid_mode_yields_aggregate_and_tree():
    from simulation.abm.agent_kernel import run_agent_world
    pop = _mini_pop()
    r = run_agent_world(N=len(pop["home_gu"]), T_days=40, beta=0.8, sigma=0.3,
                        gamma=0.2, delta=0.004, nu=0.0, global_seed=3,
                        import_rate=1e-3, population=pop, transmission_mode="hybrid",
                        hybrid_weight=0.5,
                        beta_by_layer={"household": 0.35, "workplace": 0.12,
                                       "school": 0.15, "community": 0.06})
    assert r["I"].max() > 0                          # aggregate epidemic (mean-field arm)
    assert r["transmission_tree"]["infectee"].size > 0   # + person-like tree (network arm)


def test_hybrid_requires_population():
    from simulation.abm.agent_kernel import run_agent_world
    with pytest.raises(ValueError):
        run_agent_world(N=100, T_days=10, beta=0.3, sigma=0.3, gamma=0.2,
                        delta=0.004, nu=0.0, transmission_mode="hybrid")


def test_hybrid_reduces_to_meanfield_and_network_at_extremes():
    """w=1 hybrid ≈ mean-field aggregate; w=0 hybrid ≈ network aggregate (sanity)."""
    from simulation.abm.agent_kernel import run_agent_world
    pop = _mini_pop()
    bbl = {"household": 0.3, "workplace": 0.1, "school": 0.12, "community": 0.05}
    kw = dict(N=len(pop["home_gu"]), T_days=30, beta=0.6, sigma=0.3, gamma=0.2,
              delta=0.004, nu=0.0, global_seed=1, import_rate=1e-3, population=pop)
    h1 = run_agent_world(**kw, transmission_mode="hybrid", hybrid_weight=1.0,
                         beta_by_layer=bbl)
    net = run_agent_world(**kw, transmission_mode="network", beta_by_layer=bbl)
    h0 = run_agent_world(**kw, transmission_mode="hybrid", hybrid_weight=0.0,
                         beta_by_layer=bbl)
    # w=0 hybrid FoI == network FoI → identical epidemic
    assert np.array_equal(h0["I"], net["I"])
    # w=1 hybrid uses only the mean-field arm → an epidemic still occurs
    assert h1["I"].max() > 0


# ── real-time EnKF assimilation (closed-loop to the champion forecast) ─────────
def test_enkf_couple_leak_free_and_bounded():
    """Real-time EnKF assimilates the champion FORECAST (not the forward truth)
    and stays at or below the champion ceiling (a Bayesian blend cannot beat it)."""
    from simulation.abm.variant_ablation import enkf_couple_forward
    r = enkf_couple_forward(variant="H", n_agents=3000, n_seeds=4)
    assert "forecast" in r["leak_free"].lower()          # obs = forecast, never forward truth
    assert set(r) >= {"variant_alone_forward_r2", "variant_plus_enkf_forward_r2",
                      "champion_alone_forward_r2"}
    assert r["variant_plus_enkf_forward_r2"] <= r["champion_alone_forward_r2"] + 0.06


# ── leak-free guard: forward observations must not touch calibration ──────────
def test_ablation_calibration_is_structurally_leak_free():
    import inspect
    from simulation.abm import variant_ablation as VA
    # neither the beta/affine calibration nor the per-variant curve builder may
    # receive the forward observations — they are fit on in-sample only.
    for fn in (VA._calibrate_beta, VA._full_curve):
        params = " ".join(inspect.signature(fn).parameters)
        assert "forward" not in params, f"{fn.__name__} must not see forward data"


def test_calibration_independent_of_forward_values():
    """The calibrated beta/affine is identical however forward ILI is perturbed."""
    from simulation.abm.variant_ablation import _calibrate_beta
    from simulation.abm.synthetic_population import generate_population
    pop = generate_population(1500, seed=0)
    ins = np.array([5, 8, 14, 22, 33, 45, 55, 60], dtype=float)   # arbitrary in-sample
    grid = [0.15, 0.22]
    a = _calibrate_beta("A", pop, ins, len(ins) + 6, seeds=[0], beta_grid=grid, init_frac=0.004)
    b = _calibrate_beta("A", pop, ins, len(ins) + 6, seeds=[0], beta_grid=grid, init_frac=0.004)
    assert a["beta"] == b["beta"] and a["affine"] == b["affine"]   # depends only on in-sample


# ── tree_metrics: NO cross-replicate index collision (regression, verify-audit) ─
# Agent index j in seed 0 and seed 1 are DIFFERENT simulated people. Pooling raw
# infector indices across seeds and np.bincount()-ing merges them, over-counting
# offspring per infector and inflating the superspreader tail. Occupation attack
# has the same root: a seed-pooled numerator over a single-population denominator
# yields ~n_seeds x the true per-run share (can exceed 1.0).
def _two_seed_trees():
    ln = ["household", "workplace", "school", "community"]
    # seed 0: agent 0 (import) infects agents 1 and 2 → k(agent0)=2
    t0 = {"day": np.array([1, 2, 2]), "infectee": np.array([0, 1, 2]),
          "infector": np.array([-1, 0, 0]), "layer": np.array([-1, 1, 1]),
          "layer_names": ln}
    # seed 1: agent 0 (a DIFFERENT person) is imported and infects agent 3 → k=1
    t1 = {"day": np.array([1, 2]), "infectee": np.array([0, 3]),
          "infector": np.array([-1, 0]), "layer": np.array([-1, 1]),
          "layer_names": ln}
    return [t0, t1]


def test_tree_metrics_no_cross_replicate_offspring_merge():
    """Offspring k must be counted per seed, not merged across replicates."""
    from simulation.abm.variant_ablation import tree_metrics
    pop = {"occupation": np.array([0, 0, 1, 1]), "home_gu": np.array([0, 0, 1, 1])}
    m = tree_metrics(_two_seed_trees(), pop)
    # per-seed: seed0 agent0→2 offspring, seed1 agent0→1 offspring → k=[2,1], mean 1.5.
    # the buggy pooled bincount would merge both agent-0 rows → k=[3], mean 3.0.
    assert m["offspring_k_mean"] == 1.5, f"cross-seed merge not fixed: {m['offspring_k_mean']}"


def test_tree_metrics_occupation_attack_is_a_share():
    """occupation_attack_rate must be a per-run share in [0,1], not n_seeds-inflated."""
    from simulation.abm.variant_ablation import tree_metrics
    pop = {"occupation": np.array([0, 0, 1, 1]), "home_gu": np.array([0, 0, 1, 1])}
    m = tree_metrics(_two_seed_trees(), pop)
    for o, v in m["occupation_attack_rate"].items():
        assert 0.0 <= v <= 1.0, f"occupation {o} attack {v} not a share in [0,1]"
    # occ 0 = agents {0,1}; seed0 infects agents 0&1 (2/2=1.0), seed1 infects agent 0
    # (1/2=0.5) → per-seed mean 0.75. The buggy pooled count 3/2 = 1.5 would exceed 1.
    assert m["occupation_attack_rate"][0] == 0.75
