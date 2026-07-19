"""multi_strain.run_multistrain / strain_competition_summary smoke tests.

불변식 검증:
  - 전 strain 합 보존(S + sum(E) + sum(I) + R == N, 매 step)
  - shape 정합(시계열·agent SoA)
  - edge(전부 면역 cross_immunity=1, beta=0, 1-strain 경계)
  - leak-free 결정성(같은 seed → bit-identical, 다른 seed → 다름)
  - 교차면역 ↑ → 2차 strain attack rate ↓ (실측)
  - beta 큰 strain 우점(경쟁)
  - 2-strain · 3-strain case
서울 25구 ILI(WHO FluNet KR subtype 비율) grounding 케이스 포함.
"""
from __future__ import annotations

import numpy as np
import pytest

from simulation.abm.multi_strain import (
    run_multistrain,
    strain_competition_summary,
)


# WHO FluNet 한국(iso3=KOR) 실측 subtype 비율(A/H1N1 : A/H3N2 : B ≈ 27:40:33).
_KR_STRAINS = ["A/H1N1pdm09", "A/H3N2", "B"]


def _total_per_day(res: dict) -> np.ndarray:
    return (
        res["S"]
        + res["E"].sum(axis=1)
        + res["I"].sum(axis=1)
        + res["R"]
    )


def test_conservation_2strain():
    """전 strain 합 보존: 매 step S+E+I+R == N (2-strain)."""
    N = 600
    res = run_multistrain(
        N, 40, seed=1,
        betas=[0.4, 0.3],
        cross_immunity=[[1.0, 0.3], [0.3, 1.0]],
        sigma=0.5, gamma=0.25,
        initial_infected=[5, 5],
    )
    totals = _total_per_day(res)
    assert np.all(totals == N), "compartment 합이 N 으로 보존되지 않음"
    # 음수 카운트 없음
    assert res["S"].min() >= 0 and res["R"].min() >= 0
    assert res["E"].min() >= 0 and res["I"].min() >= 0


def test_conservation_3strain_kr():
    """3-strain(KR 비율) 보존 + shape 정합."""
    N = 900
    init = [round(0.27 * 30), round(0.40 * 30), round(0.33 * 30)]
    res = run_multistrain(
        N, 50, seed=7,
        betas=[0.35, 0.45, 0.30],
        cross_immunity=[
            [1.0, 0.4, 0.1],
            [0.4, 1.0, 0.1],
            [0.1, 0.1, 1.0],
        ],
        sigma=0.5, gamma=0.25,
        initial_infected=init,
        strain_names=_KR_STRAINS,
    )
    assert np.all(_total_per_day(res) == N)
    assert res["E"].shape == (50, 3)
    assert res["I"].shape == (50, 3)
    assert res["incidence"].shape == (50, 3)
    assert res["cumulative_incidence"].shape == (50, 3)
    assert res["strain_names"] == _KR_STRAINS
    assert res["agents"]["recovered_hist"].shape == (N, 3)


def test_shapes_and_agent_soa():
    """반환 시계열·agent SoA shape/타입 계약."""
    N, T = 300, 25
    res = run_multistrain(
        N, T, seed=3,
        betas=[0.4, 0.35],
        cross_immunity=np.eye(2),
        sigma=0.5, gamma=0.25,
    )
    assert res["S"].shape == (T,)
    assert res["R"].shape == (T,)
    assert res["E"].shape == (T, 2)
    ag = res["agents"]
    assert ag["phase"].shape == (N,)
    assert ag["strain_of"].shape == (N,)
    assert ag["home_gu"].shape == (N,)
    # incidence day0 == 초기 시드 수
    assert res["incidence"][0].sum() == res["cumulative_incidence"][0].sum()


def test_determinism_same_seed_bit_identical():
    """같은 seed → bit-identical (leak-free 결정성)."""
    kw = dict(
        betas=[0.4, 0.3, 0.25],
        cross_immunity=np.eye(3) * 0.0 + np.diag([1, 1, 1]),
        sigma=0.5, gamma=0.25, initial_infected=[6, 6, 6],
    )
    a = run_multistrain(500, 35, seed=99, **kw)
    b = run_multistrain(500, 35, seed=99, **kw)
    assert np.array_equal(a["I"], b["I"])
    assert np.array_equal(a["incidence"], b["incidence"])
    assert np.array_equal(a["agents"]["phase"], b["agents"]["phase"])


def test_determinism_different_seed_differs():
    """다른 seed → 동역학 달라짐(결정성이 상수가 아님)."""
    kw = dict(
        betas=[0.45, 0.3],
        cross_immunity=[[1.0, 0.2], [0.2, 1.0]],
        sigma=0.5, gamma=0.25, initial_infected=[5, 5],
    )
    a = run_multistrain(500, 40, seed=1, **kw)
    b = run_multistrain(500, 40, seed=2, **kw)
    assert not np.array_equal(a["incidence"], b["incidence"])


def test_cross_immunity_reduces_secondary_attack_rate():
    """교차면역 ↑ → 2차 strain attack rate ↓ (실측).

    strain0(강·먼저 우점)이 인구를 휩쓴 뒤, strain1 에 대한 교차보호가
    클수록 strain1 누적 감염(attack rate)이 작아야 한다.
    """
    N = 2000
    # 두 strain 이 강하게 동시 순환(strain0 가 약간 선행). 회복자가 다시
    # SUSCEPTIBLE 로 돌아가 strain1 에 노출되므로, strain0→strain1 교차보호가
    # 클수록 strain1 누적 감염이 작아져야 한다.
    base = dict(
        betas=[0.7, 0.6],
        sigma=0.7, gamma=0.2,
        initial_infected=[25, 8],
        strain_names=["dominant", "secondary"],
    )
    # strain0 회복자 → strain1 보호 낮음(0.0) vs 높음(0.9)
    low = run_multistrain(
        N, 250, seed=11,
        cross_immunity=[[1.0, 0.0], [0.0, 1.0]],
        **base,
    )
    high = run_multistrain(
        N, 250, seed=11,
        cross_immunity=[[1.0, 0.0], [0.9, 1.0]],
        **base,
    )
    ar_low = strain_competition_summary(low)["attack_rate"]["secondary"]
    ar_high = strain_competition_summary(high)["attack_rate"]["secondary"]
    assert ar_high < ar_low, (
        f"교차면역↑ 인데 2차 attack rate 안 줄어듦: high={ar_high:.3f} "
        f">= low={ar_low:.3f}"
    )


def test_larger_beta_strain_dominates():
    """beta 큰 strain 이 경쟁에서 우점(같은 시드·대칭 초기조건)."""
    N = 1500
    res = run_multistrain(
        N, 100, seed=5,
        betas=[0.65, 0.30],            # strain0 >> strain1
        cross_immunity=[[1.0, 0.5], [0.5, 1.0]],  # 양방향 동일 교차면역
        sigma=0.6, gamma=0.2,
        initial_infected=[10, 10],     # 동일 시드 수
        strain_names=["fast", "slow"],
    )
    summ = strain_competition_summary(res)
    assert summ["dominant_strain"] == "fast", (
        f"beta 큰 strain 이 우점이어야 하는데 dominant={summ['dominant_strain']}, "
        f"cum={summ['final_cumulative_incidence']}"
    )
    assert (
        summ["final_cumulative_incidence"]["fast"]
        > summ["final_cumulative_incidence"]["slow"]
    )


def test_edge_full_homologous_immunity_no_reinfection():
    """edge: 대각 교차면역=1 + waning=0 → 동종 재감염 없음(누적 ≤ N).

    단일 strain, 면역소실 없음 → 회복이력이 strain0 보호=1 이라 재감염 0,
    누적 incidence 가 N 을 넘지 않아야 한다(종신면역).
    """
    N = 400
    res = run_multistrain(
        N, 120, seed=2,
        betas=[0.6],
        cross_immunity=[[1.0]],
        sigma=0.6, gamma=0.2,
        initial_infected=[10],
        waning=0.0,   # 종신면역 → 동종 재감염 차단
    )
    assert np.all(_total_per_day(res) == N)
    # 누적 신규감염이 N 을 못 넘음(동종 재감염 차단 증거)
    assert res["cumulative_incidence"][-1, 0] <= N


def test_waning_enables_reinfection():
    """waning>0 → strain별 면역소실로 누적 incidence 가 N 을 초과 가능(다년 재유행).

    단일 strain·면역소실 켜짐 → 회복자가 재감염되어 누적 신규감염이 N 을
    넘을 수 있어야 한다(보존은 유지).
    """
    N = 400
    res = run_multistrain(
        N, 400, seed=2,
        betas=[0.8],
        cross_immunity=[[1.0]],
        sigma=0.7, gamma=0.25,
        initial_infected=[10],
        waning=0.05,        # strain0 면역소실 → 재감염 가능
        import_rate=1e-3,   # off-season 소멸 방지(재점화)
    )
    assert np.all(_total_per_day(res) == N)
    assert res["cumulative_incidence"][-1, 0] > N, "waning 인데 재감염이 안 일어남"


def test_edge_zero_beta_no_spread():
    """edge: beta=0 → 초기 시드 외 신규감염 없음."""
    N = 300
    res = run_multistrain(
        N, 30, seed=4,
        betas=[0.0, 0.0],
        cross_immunity=np.eye(2),
        sigma=0.6, gamma=0.2,
        initial_infected=[5, 5],
        import_rate=0.0,
    )
    # day0 시드 10 명 외 신규 0
    assert int(res["incidence"][1:].sum()) == 0
    assert np.all(_total_per_day(res) == N)


def test_edge_single_strain_runs():
    """edge: 1-strain (경계) 도 정상 동작 + summary."""
    res = run_multistrain(
        200, 20, seed=8,
        betas=[0.5],
        cross_immunity=[[1.0]],
        sigma=0.5, gamma=0.25,
    )
    summ = strain_competition_summary(res)
    assert set(summ["attack_rate"]) == {"strain_0"}
    assert summ["dominant_strain"] in ("strain_0", None)


def test_validation_errors():
    """fail-fast: shape/범위 위반은 ValueError."""
    with pytest.raises(ValueError):
        run_multistrain(100, 10, seed=1, betas=[0.4, 0.3],
                        cross_immunity=np.eye(3),  # 3x3 vs 2 strain
                        sigma=0.5, gamma=0.25)
    with pytest.raises(ValueError):
        run_multistrain(100, 10, seed=1, betas=[0.4],
                        cross_immunity=[[1.5]],    # >1 보호확률
                        sigma=0.5, gamma=0.25)
    with pytest.raises(ValueError):
        run_multistrain(10, 10, seed=1, betas=[0.4],
                        cross_immunity=[[1.0]],
                        sigma=0.5, gamma=0.25,
                        initial_infected=[20])     # > N


def test_population_path_spatial():
    """population(25구 home_gu) 경로도 보존 + 동작."""
    from simulation.abm.synthetic_population import generate_population
    pop = generate_population(800, seed=3)
    res = run_multistrain(
        800, 40, seed=3,
        betas=[0.4, 0.35],
        cross_immunity=[[1.0, 0.3], [0.3, 1.0]],
        sigma=0.5, gamma=0.25,
        initial_infected=[8, 8],
        population=pop,
        strain_names=["A", "B"],
    )
    assert np.all(_total_per_day(res) == 800)
    summ = strain_competition_summary(res)
    assert 0.0 <= summ["attack_rate"]["A"] <= 1.0
    assert 0.0 <= summ["attack_rate"]["B"] <= 1.0
