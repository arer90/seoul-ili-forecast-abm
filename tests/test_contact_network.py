"""Smoke tests for the explicit multi-layer contact network (contact_network.py).

불변식 검증:
  1. shape — 각 layer 가 N×N CSR.
  2. 무방향 대칭 — (A != A.T).nnz == 0, 대각 0(자기루프 없음), 값 binary.
  3. household degree ≈ hh_size-1.
  4. 결정성 — 같은 seed → 동일 망, 다른 seed → 다른 망.
  5. FoI ≠ mean-field — edge 기반 이질성 실재(분산 > 0).
  6. edge case — 빈 학교(학생 없음)/단일 에이전트.
  7. leak-free — 사망자는 FoI 0, 감염자 없으면 FoI 전부 0.
  8. 망기반 개입 — 노드 제거(감염 이웃 격리) → FoI 감소.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from simulation.abm.contact_network import (
    build_multilayer_network,
    degree_summary,
    network_foi,
)
from simulation.abm.agent_kernel import STATE_D, STATE_I, STATE_S
from simulation.abm.synthetic_population import generate_population

_LAYERS = ("household", "workplace", "school", "community")


def _toy_population(n: int = 400, seed: int = 1) -> dict[str, np.ndarray]:
    """DB 기반 합성 인구(빠른 소표본)."""
    return generate_population(n, seed=seed)


def test_shapes_and_layer_keys():
    """1. 네 layer 모두 N×N CSR 로 반환."""
    pop = _toy_population(300, seed=2)
    n = pop["home_gu"].shape[0]
    layers = build_multilayer_network(pop, seed=42)
    assert set(layers) == set(_LAYERS)
    for name in _LAYERS:
        mat = layers[name]
        assert sparse.issparse(mat)
        assert mat.shape == (n, n), name


def test_undirected_symmetric_binary_no_selfloops():
    """2. 각 layer 무방향 대칭 + 대각 0 + 값 binary."""
    pop = _toy_population(350, seed=3)
    layers = build_multilayer_network(pop, seed=7)
    for name, mat in layers.items():
        assert (mat != mat.T).nnz == 0, f"{name} not symmetric"
        assert mat.diagonal().sum() == 0.0, f"{name} has self-loops"
        if mat.nnz:
            assert set(np.unique(mat.data)).issubset({1.0}), f"{name} not binary"


def test_household_degree_matches_group_size():
    """3. household 평균 degree 가 hh_size-1 근처(2-4 → 평균 ~2)."""
    pop = _toy_population(600, seed=4)
    layers = build_multilayer_network(pop, seed=11, hh_size=(2, 4))
    deg = degree_summary(layers)
    # 그룹 크기 2-4 균일 → degree 1..3, 평균 약 2. 흡수로 약간 상향 가능.
    assert 1.0 <= deg["household"] <= 3.5, deg["household"]


def test_determinism_same_seed_identical_diff_seed_differs():
    """4. 같은 seed → byte-identical, 다른 seed → 달라짐."""
    pop = _toy_population(300, seed=5)
    a = build_multilayer_network(pop, seed=99)
    b = build_multilayer_network(pop, seed=99)
    c = build_multilayer_network(pop, seed=100)
    for name in _LAYERS:
        assert (a[name] != b[name]).nnz == 0, f"{name} not deterministic"
    # 적어도 한 layer 는 seed 가 다르면 달라야 한다.
    assert any((a[name] != c[name]).nnz > 0 for name in _LAYERS)


def test_foi_is_not_meanfield_heterogeneous():
    """5. network FoI 가 평균장과 다름 — agent 간 이질성(분산 > 0) 실재."""
    pop = _toy_population(500, seed=6)
    layers = build_multilayer_network(pop, seed=13)
    rng = np.random.default_rng(0)
    state = np.full(pop["home_gu"].shape[0], STATE_S, dtype=np.int8)
    # 10% 무작위 감염.
    inf = rng.choice(state.shape[0], size=state.shape[0] // 10, replace=False)
    state[inf] = STATE_I
    beta = {"household": 0.3, "workplace": 0.1, "school": 0.2, "community": 0.05}
    foi = network_foi(state, layers, beta)
    assert foi.shape == state.shape
    assert np.all(foi >= 0.0)
    # 평균장이면 동일 구 내 모두 같은 FoI → 분산 0. edge 기반이면 분산 > 0.
    assert foi.var() > 0.0
    # mean-field 비교: 구 평균 prevalence 균질 노출과 명백히 다른 분포.
    assert foi.max() > foi.mean()  # 일부 agent 가 다수 감염 이웃 보유


def test_empty_school_and_single_agent_edge_cases():
    """6. edge case — 학생 없는 인구는 school 빈 layer, 단일 agent OK."""
    # 학생(age_band==1) 전혀 없는 합성 인구.
    pop = {
        "home_gu": np.array([0, 0, 1, 1, 2], dtype=np.int64),
        "work_gu": np.array([0, 1, 1, 2, 2], dtype=np.int64),
        "age_band": np.array([3, 4, 5, 3, 6], dtype=np.int64),  # 학령 없음
        "occupation": np.array([0, 1, 0, 2, 1], dtype=np.int64),
    }
    layers = build_multilayer_network(pop, seed=1)
    assert layers["school"].nnz == 0
    # 단일 agent → 모든 layer 빈(1x1).
    solo = {k: v[:1] for k, v in pop.items()}
    lay1 = build_multilayer_network(solo, seed=1)
    for name in _LAYERS:
        assert lay1[name].shape == (1, 1)
        assert lay1[name].nnz == 0


def test_leak_free_dead_and_no_infection():
    """7. leak-free — 사망자 FoI=0, 감염자 없으면 전부 0."""
    pop = _toy_population(300, seed=8)
    layers = build_multilayer_network(pop, seed=21)
    n = pop["home_gu"].shape[0]
    beta = {k: 0.2 for k in _LAYERS}
    # 감염자 0 → FoI 전부 0.
    state0 = np.full(n, STATE_S, dtype=np.int8)
    assert np.allclose(network_foi(state0, layers, beta), 0.0)
    # 일부 감염 + 일부 사망. 사망자 위치 FoI 는 0이어야 함.
    rng = np.random.default_rng(3)
    state = np.full(n, STATE_S, dtype=np.int8)
    state[rng.choice(n, size=30, replace=False)] = STATE_I
    dead = rng.choice(np.flatnonzero(state == STATE_S), size=20, replace=False)
    state[dead] = STATE_D
    foi = network_foi(state, layers, beta)
    assert np.allclose(foi[dead], 0.0)


def test_network_intervention_node_removal_reduces_foi():
    """8. 망기반 개입 — 감염 이웃 격리(노드 제거)하면 이웃 FoI 감소."""
    pop = _toy_population(500, seed=9)
    layers = build_multilayer_network(pop, seed=33)
    n = pop["home_gu"].shape[0]
    beta = {k: 0.25 for k in _LAYERS}
    rng = np.random.default_rng(4)
    state = np.full(n, STATE_S, dtype=np.int8)
    inf = rng.choice(n, size=50, replace=False)
    state[inf] = STATE_I
    foi_before = network_foi(state, layers, beta)
    total_before = foi_before.sum()
    # 개입: 감염자 중 절반을 격리(R 처럼 비감염원화 = STATE_S 로 회수가 아니라
    # 망에서 제거 효과 = 사망/격리로 모델). 여기선 STATE_D 로 두어 감염원 차단.
    quarantine = inf[: len(inf) // 2]
    state[quarantine] = STATE_D
    foi_after = network_foi(state, layers, beta)
    assert foi_after.sum() < total_before  # 전체 FoI 감소
    # 격리되지 않은 감염자만 남았으므로 감염원 수 감소 → 모든 위치 FoI <= 이전.
    assert np.all(foi_after <= foi_before + 1e-9)


def test_degree_summary_keys_and_total():
    """보조: degree_summary 가 모든 layer + _total 키 반환."""
    pop = _toy_population(300, seed=10)
    layers = build_multilayer_network(pop, seed=55)
    deg = degree_summary(layers)
    for name in _LAYERS:
        assert name in deg and deg[name] >= 0.0
    assert "_total" in deg


def test_invalid_inputs_raise():
    """보조: 필수 키 누락/잘못된 크기범위 → ValueError."""
    pop = _toy_population(100, seed=11)
    bad = {k: v for k, v in pop.items() if k != "occupation"}
    with pytest.raises(ValueError):
        build_multilayer_network(bad, seed=1)
    with pytest.raises(ValueError):
        build_multilayer_network(pop, seed=1, hh_size=(4, 2))
    with pytest.raises(ValueError):
        network_foi(np.zeros((2, 2)), build_multilayer_network(pop, seed=1), {})
