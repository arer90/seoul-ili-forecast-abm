"""In-run dynamic-agent SEIR-V-D TDD — resolution ∝ prevalence, population-conserving, unbiased.

사용자 제안(2026-06-05) "agent는 가변적(동적)으로 변화에 따라서 최적화" — between-run 수렴 선택은
adaptive_agent_count, **in-run** 적응 resize 가 run_adaptive_agent_world. macOS: run PER-FILE.
"""
import numpy as np
import pytest

from simulation.abm.adaptive_allocation import AdaptiveAllocator
from simulation.abm.adaptive_world import _build_demo_population, run_adaptive_agent_world
from simulation.abm.agent_kernel import STATE_I, run_agent_world

# per-day SEIR-V-D hazards (R0≈beta/gamma≈1.9)
_EPI = dict(beta=0.55, sigma=0.5, gamma=0.2857, delta=0.001, import_rate=1e-4)


def _alloc(base=2000, mx=8000, fl=500):
    return AdaptiveAllocator(base_n=base, max_n=mx, floor_n=fl, sensitivity=1.0)


def test_population_conserved_every_day():
    """Σ(S..D) == P every day, regardless of the changing resolution M."""
    out = run_adaptive_agent_world(2000, 100, allocator=_alloc(), peak_prevalence=0.08,
                                   population_size=2000, epoch_len=10, **_EPI)
    total = sum(out[k] for k in "SEIRVD")
    assert np.allclose(total, out["P"], rtol=1e-6, atol=1e-3), \
        f"conservation broken: max|Σ-P|={float(np.max(np.abs(total - out['P']))):.3g}"
    assert len(out["n_trajectory"]) == len(out["I"])


def test_resolution_is_dynamic_and_bounded():
    """Effective N tracks prevalence (peak → high-res) within allocator bounds."""
    out = run_adaptive_agent_world(2000, 130, allocator=_alloc(), peak_prevalence=0.10,
                                   population_size=2000, epoch_len=10, **_EPI)
    n = out["n_trajectory"]
    assert n.max() > n.min(), "resolution did not adapt at all"
    assert n.min() >= 500 and n.max() <= 8000, "resolution escaped allocator [floor,max]"
    peak_day = int(np.argmax(out["I"]))
    assert n[peak_day] >= n[-1], "resolution not higher at epidemic peak than off-season tail"


def test_unbiased_vs_fixed_n_behaviour_off():
    """Resampling is unbiased: weighted attack rate ≈ fixed-N (behaviour-off isolates
    the resampler from the per-epoch fatigue re-init approximation)."""
    P = 4000
    off = dict(alpha_mean=0.0, kappa_mean=0.0)  # no behavioural damping → clean comparison
    out = run_adaptive_agent_world(
        P, 150, allocator=_alloc(base=P, mx=P * 2, fl=800), peak_prevalence=0.12,
        population_size=P, epoch_len=15, global_seed=7, **_EPI, **off)
    ar_adapt = float((out["R"][-1] + out["D"][-1]) / out["P"])
    # fixed-N reference: SAME population path (heterogeneity), no resampling
    ref_pop = _build_demo_population(P, np.random.default_rng(99))
    ref = run_agent_world(N=P, T_days=150, nu=0.0, population=ref_pop, global_seed=7, **_EPI, **off)
    ar_fixed = float((ref["R"][-1] + ref["D"][-1]) / P)
    assert ar_adapt > 0.1, f"no epidemic (adaptive AR={ar_adapt:.3f})"
    assert abs(ar_adapt - ar_fixed) < 0.12, \
        f"attack rate biased by resampling: adaptive={ar_adapt:.3f} vs fixed={ar_fixed:.3f}"


def test_kernel_initial_state_resume():
    """run_agent_world(initial_state=...) resumes from a given compartment state
    instead of the fresh 1% seed — the chaining primitive for in-run resize."""
    pop = _build_demo_population(800, np.random.default_rng(1))
    init = np.full(800, STATE_I, dtype=np.int8)          # everyone infectious at t=0
    res = run_agent_world(N=800, T_days=4, beta=0.5, sigma=0.5, gamma=0.3, delta=0.01,
                          nu=0.0, population=pop, initial_state=init, global_seed=3)
    assert res["I"][0] == 800, "initial_state ignored (fresh seed used)"
    assert res["R"][-1] + res["D"][-1] > 0, "I should transition out to R/D"


def test_demo_population_format():
    pop = _build_demo_population(500, np.random.default_rng(0))
    assert set(pop) == {"home_gu", "work_gu", "age_band", "occupation", "severity"}
    assert all(v.shape == (500,) for v in pop.values())
    assert set(np.unique(pop["severity"])).issubset({0, 1})
    assert 0 <= pop["occupation"].min() and pop["occupation"].max() < 6
    assert 0 <= pop["home_gu"].min() and pop["home_gu"].max() < 25


def test_validation():
    with pytest.raises(ValueError):
        run_adaptive_agent_world(2000, 50, allocator="nope", peak_prevalence=0.05, **_EPI)
    with pytest.raises(ValueError):
        run_adaptive_agent_world(2000, 50, allocator=_alloc(), peak_prevalence=0.0, **_EPI)
    with pytest.raises(ValueError):
        run_adaptive_agent_world(0, 50, allocator=_alloc(), peak_prevalence=0.05, **_EPI)
