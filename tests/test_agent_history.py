"""Smoke tests for simulation.abm.agent_history (per-agent SEIR 궤적 추적).

TDD (ENGINEERING_PRINCIPLES.md D-3): 보존·shape·edge·leak-free(결정성) + 개인/인구집단 추출 검증.
macOS는 파일 단위 실행: `.venv/bin/python -m pytest tests/test_agent_history.py -x -q`.
"""
from __future__ import annotations

import numpy as np
import pytest

from simulation.abm.agent_history import (
    STATE_LABELS,
    extract_agent_trajectory,
    population_summary,
    simulate_with_history,
)


# 작은 N + 약한 outbreak 구성 1회 — 여러 테스트가 공유(빠르게).
def _run(N=300, T_days=25, seed=11, beta=1.0, import_rate=1e-3):
    return simulate_with_history(
        N=N,
        T_days=T_days,
        seed=seed,
        beta=beta,
        sigma=0.5,
        gamma=0.25,
        delta=0.0,
        nu=0.0,
        import_rate=import_rate,
    )


def test_shapes_and_dtypes():
    """history/위치 행렬 shape (T_days, N) + int8, aggregate (T_days, 4/6)."""
    res = _run(N=300, T_days=25)
    assert res["history_state"].shape == (25, 300)
    assert res["history_state"].dtype == np.int8
    assert res["history_location"].shape == (25, 300)
    assert res["history_location"].dtype == np.int8
    assert res["history_location_night"].shape == (25, 300)
    assert res["aggregate"].shape == (25, 4)
    assert res["aggregate_full"].shape == (25, 6)
    # attrs per-agent 배열 길이 N
    for key in ("home_gu", "work_gu", "age_band", "sex", "severity", "occupation"):
        assert res["attrs"][key].shape == (300,), key
    assert len(res["attrs"]["gu_names"]) == 25


def test_conservation_SEIR_equals_N():
    """매일 S+E+I+R+V+D == N (delta=0 이면 D=0 이라 S+E+I+R==N). 불변식."""
    N, T = 400, 20
    res = _run(N=N, T_days=T)
    daily_sum = res["aggregate_full"].sum(axis=1)
    assert np.all(daily_sum == N), f"보존 위반: {set(daily_sum.tolist())}"
    # aggregate(S,E,I,R) 도 delta=0 이므로 합 == N
    assert np.all(res["aggregate"].sum(axis=1) == N)


def test_history_state_value_range():
    """history_state ∈ {0,1,2,3,4,5}; delta=0/nu=0 이면 {0,1,2,3} 안에 머문다."""
    res = _run()
    vals = set(np.unique(res["history_state"]).tolist())
    assert vals.issubset({0, 1, 2, 3, 4, 5})
    # 이 구성(nu=0, delta=0)에서는 V/D 미발생
    assert vals.issubset({0, 1, 2, 3})


def test_history_matches_aggregate():
    """history_state 로 다시 센 집계 == 반환 aggregate_full (단일 신뢰원 일치)."""
    res = _run()
    T = res["history_state"].shape[0]
    recount = np.zeros((T, 6), dtype=np.int64)
    for d in range(T):
        recount[d] = np.bincount(res["history_state"][d].astype(np.int64), minlength=6)[:6]
    assert np.array_equal(recount, res["aggregate_full"])


def test_determinism_leak_free():
    """동일 seed → 비트 동일 history (재현성 G-원칙 5). 두 번 호출 결과 일치."""
    a = _run(seed=2024)
    b = _run(seed=2024)
    assert np.array_equal(a["history_state"], b["history_state"])
    assert np.array_equal(a["history_location"], b["history_location"])
    assert np.array_equal(a["attrs"]["home_gu"], b["attrs"]["home_gu"])
    # 다른 seed → 달라야 함 (난수 실제 작동 확인)
    c = _run(seed=999)
    assert not np.array_equal(a["history_state"], c["history_state"])


def test_outbreak_actually_happens():
    """실제 동역학: 감염이 퍼져 누군가 I→R 전이 발생(placeholder 아님)."""
    res = _run(N=500, T_days=40, beta=1.2, import_rate=1e-3)
    # 최종 R > 초기 R (감염 확산 후 회복 발생)
    r_idx = STATE_LABELS.index("R")
    i_idx = STATE_LABELS.index("I")
    assert res["aggregate_full"][-1, r_idx] > res["aggregate_full"][0, r_idx]
    assert res["aggregate_full"][:, i_idx].max() > res["aggregate_full"][0, i_idx]


def test_day0_is_initial_seed():
    """day 0 = 초기 시드 상태: 일부 I, 나머지 S, E/R 없음(fresh seed)."""
    res = _run(N=300, T_days=10)
    day0 = res["history_state"][0]
    assert (day0 == 2).sum() >= 1               # 최소 1명 감염 시드
    assert set(np.unique(day0).tolist()).issubset({0, 2})  # S 또는 I 만


def test_extract_agent_trajectory():
    """개인 추출: 속성 + 상태 시퀀스 + 전이 + 위치 궤적."""
    res = _run(N=300, T_days=25)
    traj = extract_agent_trajectory(res, agent_id=0)
    assert traj["agent_id"] == 0
    a = traj["attrs"]
    assert 0 <= a["home_gu"] < 25 and 0 <= a["work_gu"] < 25
    assert a["home_gu_name"] in res["attrs"]["gu_names"]
    assert a["age_band_label"] in ("0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60+")
    assert a["sex_label"] in ("male", "female")
    assert a["severity_label"] in ("low", "high")
    assert a["is_commuter"] == (a["work_gu"] != a["home_gu"])
    # 상태 시퀀스 길이 == T_days, 라벨 일치
    assert traj["states"].shape == (25,)
    assert len(traj["state_labels"]) == 25
    # transitions[0] 은 day 0
    assert traj["transitions"][0][0] == 0
    # 단조 day 증가
    days = [t[0] for t in traj["transitions"]]
    assert days == sorted(days)
    # 위치 궤적 길이
    assert traj["location_day"].shape == (25,) and traj["location_night"].shape == (25,)


def test_extract_infected_agent_has_transition():
    """감염된 agent는 S→E/I 전이가 transitions 에 기록되고 infected_day 설정."""
    res = _run(N=500, T_days=40, beta=1.2, import_rate=1e-3)
    # 감염된(E/I 거쳐간) agent 하나 찾기
    ever_infected = np.flatnonzero(
        ((res["history_state"] == 1) | (res["history_state"] == 2)).any(axis=0)
    )
    assert ever_infected.size > 0
    traj = extract_agent_trajectory(res, agent_id=int(ever_infected[0]))
    assert traj["infected_day"] is not None
    # 적어도 한 번 상태 변화(전이 >= 2 entries)
    assert len(traj["transitions"]) >= 2


def test_extract_out_of_range_raises():
    """agent_id 범위 밖 → IndexError (fail-fast, silent 금지)."""
    res = _run(N=100, T_days=10)
    with pytest.raises(IndexError):
        extract_agent_trajectory(res, agent_id=100)
    with pytest.raises(IndexError):
        extract_agent_trajectory(res, agent_id=-1)


def test_population_summary():
    """인구집단 요약: 분포 합 == N, peak/attack_rate 일관."""
    N = 400
    res = _run(N=N, T_days=30, beta=1.2, import_rate=1e-3)
    summ = population_summary(res)
    assert summ["n_agents"] == N
    assert sum(summ["age_distribution"].values()) == N
    assert sum(summ["sex_distribution"].values()) == N
    assert sum(summ["severity_distribution"].values()) == N
    assert sum(summ["home_gu_distribution"].values()) == N
    assert 0 <= summ["commuter_count"] <= N
    # peak 통계
    assert set(summ["aggregate_peak"].keys()) == set(STATE_LABELS)
    for name, info in summ["aggregate_peak"].items():
        assert 0 <= info["peak_day"] < res["history_state"].shape[0]
    # attack_rate ∈ [0,1], outbreak 가 있었으니 > 0
    assert 0.0 <= summ["attack_rate"] <= 1.0
    assert summ["attack_rate"] > 0.0
    assert 0.0 < summ["peak_prevalence"] <= 1.0


def test_edge_single_agent_and_single_day():
    """edge: N=1 / T_days=1 도 깨지지 않음 + 잘못된 입력은 ValueError."""
    res1 = simulate_with_history(N=1, T_days=1, seed=1, beta=0.5)
    assert res1["history_state"].shape == (1, 1)
    assert res1["aggregate_full"].sum() == 1
    res2 = simulate_with_history(N=5, T_days=1, seed=1, beta=0.5)
    assert res2["history_state"].shape == (1, 5)
    with pytest.raises(ValueError):
        simulate_with_history(N=0, T_days=5)
    with pytest.raises(ValueError):
        simulate_with_history(N=10, T_days=0)


def test_non_multiple_of_25_agents():
    """N 이 25 배수가 아니어도 동작 (kernel 계약)."""
    res = simulate_with_history(N=37, T_days=8, seed=3, beta=0.6)
    assert res["history_state"].shape == (8, 37)
    assert np.all(res["aggregate_full"].sum(axis=1) == 37)


def test_location_day_is_workgu_night_is_homegu():
    """위치 의미 검증: 낮 위치 == work_gu, 밤 위치 == home_gu (통근 모델)."""
    res = _run(N=200, T_days=12)
    work_gu = res["attrs"]["work_gu"]
    home_gu = res["attrs"]["home_gu"]
    # 모든 날에 낮=work, 밤=home (스케줄 고정)
    assert np.array_equal(res["history_location"][5], work_gu)
    assert np.array_equal(res["history_location_night"][5], home_gu)
