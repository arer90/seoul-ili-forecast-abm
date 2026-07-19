"""waning_immunity.run_waning_seirs smoke 테스트(SEIRS-V-D 면역소실+재감염).

불변식 검증:
  - 보존: S+E+I+R+V+D == N (매 step)
  - shape 정합(시계열 (T,) + agent SoA (N,))
  - leak-free 결정성(같은 seed → bit-identical, 다른 seed → 다름)
  - edge: omega_r=omega_v=0 → 재감염 0 (SEIR 환원)
  - edge: beta=0 → 초기 시드 외 신규감염 0
  - waning↑ → 2차파/reinfection↑ (실측)
  - V→S 접종면역 소실(omega_v) 단독으로도 재감염 발생
  - fail-fast 검증(범위/분율 위반 ValueError)
서울 25구 ILI(337주, 2022-24 rebound) grounding 케이스 포함.
"""
from __future__ import annotations

import numpy as np
import pytest

from simulation.abm.waning_immunity import run_waning_seirs


def _total_per_day(res: dict) -> np.ndarray:
    return res["S"] + res["E"] + res["I"] + res["R"] + res["V"] + res["D"]


def test_conservation_holds_every_step():
    """보존 불변식: 매 step S+E+I+R+V+D == N (사망 포함)."""
    N = 800
    res = run_waning_seirs(
        N, 120, seed=1,
        beta=0.5, sigma=0.5, gamma=0.25,
        omega_r=0.02, omega_v=0.01, nu=0.005, delta=0.001,
        import_rate=1e-3,
    )
    totals = _total_per_day(res)
    assert np.all(totals == N), "compartment 합이 N 으로 보존되지 않음"
    for k in ("S", "E", "I", "R", "V", "D"):
        assert res[k].min() >= 0, f"{k} 에 음수 카운트"


def test_shapes_and_agent_soa():
    """반환 시계열·agent SoA shape/타입 계약."""
    N, T = 300, 40
    res = run_waning_seirs(N, T, seed=3, beta=0.4, sigma=0.5, gamma=0.25, omega_r=0.02)
    for k in ("S", "E", "I", "R", "V", "D", "incidence", "reinfections"):
        assert res[k].shape == (T,), f"{k} shape != ({T},)"
    ag = res["agents"]
    assert ag["state"].shape == (N,)
    assert ag["ever_immune"].shape == (N,)
    assert ag["infection_count"].shape == (N,)
    assert ag["ever_immune"].dtype == bool
    # day0 incidence == 초기 시드 수, day0 reinfection == 0
    assert res["incidence"][0] == int(round(N * 0.01))
    assert res["reinfections"][0] == 0
    assert res["cumulative_reinfections"] == int(res["reinfections"].sum())


def test_determinism_same_seed_bit_identical():
    """같은 seed → bit-identical (leak-free 결정성)."""
    kw = dict(beta=0.5, sigma=0.5, gamma=0.25, omega_r=0.03, omega_v=0.02,
              nu=0.005, import_rate=1e-3)
    a = run_waning_seirs(600, 90, seed=99, **kw)
    b = run_waning_seirs(600, 90, seed=99, **kw)
    assert np.array_equal(a["I"], b["I"])
    assert np.array_equal(a["incidence"], b["incidence"])
    assert np.array_equal(a["reinfections"], b["reinfections"])
    assert np.array_equal(a["agents"]["infection_count"], b["agents"]["infection_count"])


def test_determinism_different_seed_differs():
    """다른 seed → 동역학 달라짐(결정성이 상수가 아님)."""
    kw = dict(beta=0.5, sigma=0.5, gamma=0.25, omega_r=0.03, import_rate=1e-3)
    a = run_waning_seirs(600, 90, seed=1, **kw)
    b = run_waning_seirs(600, 90, seed=2, **kw)
    assert not np.array_equal(a["incidence"], b["incidence"])


def test_no_waning_means_no_reinfection_seir_reduction():
    """edge: omega_r=omega_v=0 → 재감염 0 (SEIR 환원).

    회복·접종 면역이 종신이면 누구도 R/V 를 거친 뒤 재감염될 수 없다.
    누적 신규감염도 N 을 초과할 수 없다(각 에이전트 최대 1회 감염).
    """
    N = 500
    res = run_waning_seirs(
        N, 200, seed=7,
        beta=0.8, sigma=0.6, gamma=0.2,
        omega_r=0.0, omega_v=0.0, nu=0.01,
        import_rate=1e-3,   # 유입 있어도 종신면역이면 재감염 0
    )
    assert np.all(_total_per_day(res) == N)
    assert res["reinfections"].sum() == 0, "종신면역인데 재감염 발생"
    assert res["cumulative_reinfections"] == 0
    # 누적 신규감염 <= N (각자 최대 1회)
    assert int(res["incidence"].sum()) <= N
    # agent별 감염 횟수도 1 이하
    assert int(res["agents"]["infection_count"].max()) <= 1


def test_edge_zero_beta_no_spread():
    """edge: beta=0 + import=0 → 초기 시드 외 신규감염 0."""
    N = 300
    res = run_waning_seirs(
        N, 50, seed=4,
        beta=0.0, sigma=0.6, gamma=0.2,
        omega_r=0.05, import_rate=0.0,
    )
    assert int(res["incidence"][1:].sum()) == 0, "beta=0 인데 신규감염 발생"
    assert res["reinfections"].sum() == 0
    assert np.all(_total_per_day(res) == N)


def test_waning_increases_reinfection_and_second_wave():
    """waning↑ → 2차파/reinfection↑ (실측).

    동일 seed·동일 조건에서 omega_r 만 0 → 0.05 로 키우면 재감염 누적이
    엄격히 증가해야 하고(종신면역=0 대비), 누적 신규감염이 N 을 초과해야
    한다(다년 재유행 증거).
    """
    N = 1000
    base = dict(beta=0.7, sigma=0.6, gamma=0.2, import_rate=2e-3)
    lifelong = run_waning_seirs(N, 400, seed=11, omega_r=0.0, **base)
    waning = run_waning_seirs(N, 400, seed=11, omega_r=0.05, **base)
    assert lifelong["cumulative_reinfections"] == 0
    assert waning["cumulative_reinfections"] > 0, "waning 인데 재감염 없음"
    # 다년 재유행: 누적 신규감염이 인구 N 초과
    assert int(waning["incidence"].sum()) > N, "waning 인데 누적감염이 N 이하(재유행 없음)"
    # 일부 에이전트는 2회 이상 감염
    assert int(waning["agents"]["infection_count"].max()) >= 2


def test_vaccine_waning_alone_causes_reinfection():
    """V→S 접종면역 소실(omega_v) 단독으로도 재감염 발생.

    회복면역은 종신(omega_r=0)이지만 접종면역만 소실(omega_v>0)되면,
    접종(V)을 경험한 뒤 감수성으로 돌아간 에이전트가 감염되어 재감염으로
    집계되어야 한다.
    """
    N = 800
    res = run_waning_seirs(
        N, 300, seed=21,
        beta=0.6, sigma=0.6, gamma=0.2,
        omega_r=0.0, omega_v=0.06, nu=0.0,
        initial_vaccinated_frac=0.5,  # 절반을 접종 V 로 시작
        import_rate=2e-3,
    )
    assert np.all(_total_per_day(res) == N)
    assert res["cumulative_reinfections"] > 0, "접종면역 소실인데 재감염 없음"


def test_seoul_2022_24_rebound_grounding():
    """서울 ILI 2022-24 rebound grounding: 면역소실로 2차 파 형성.

    팬데믹 후 누적 면역(높은 초기 R) 상태에서 회복·접종 면역이 모두
    소실되면, 감수성 보충으로 한 차례 더 유행 정점(2차 파)이 형성된다.
    구체 수치가 아닌 '2차 파가 존재한다'는 기전적 사실만 검증(정직).
    """
    N = 1500
    res = run_waning_seirs(
        N, 730, seed=33,                  # ~2년(주간 ILI rebound horizon)
        beta=0.6, sigma=0.5, gamma=0.2,
        omega_r=0.01, omega_v=0.008, nu=0.003, delta=0.0005,
        import_rate=1e-3,
        initial_infected_frac=0.02,
        initial_vaccinated_frac=0.3,
    )
    assert np.all(_total_per_day(res) == N)
    inc = res["incidence"].astype(np.float64)
    # 주간 평활(7일) 후 국소 정점이 2개 이상이면 2차 파 존재.
    kernel = np.ones(7) / 7.0
    smooth = np.convolve(inc, kernel, mode="same")
    # 충분히 큰 정점만(노이즈 배제): 전체 최대의 25% 이상
    thresh = 0.25 * smooth.max()
    peaks = 0
    for t in range(1, len(smooth) - 1):
        if smooth[t] >= thresh and smooth[t] > smooth[t - 1] and smooth[t] >= smooth[t + 1]:
            peaks += 1
    assert peaks >= 2, f"면역소실 rebound 인데 2차 파(다중 정점) 없음: peaks={peaks}"
    assert res["cumulative_reinfections"] > 0


def test_validation_errors():
    """fail-fast: 범위/분율 위반은 ValueError."""
    with pytest.raises(ValueError):
        run_waning_seirs(0, 10, seed=1)                       # N < 1
    with pytest.raises(ValueError):
        run_waning_seirs(100, 0, seed=1)                      # T_days < 1
    with pytest.raises(ValueError):
        run_waning_seirs(100, 10, seed=1, beta=-0.1)          # 음수 rate
    with pytest.raises(ValueError):
        run_waning_seirs(100, 10, seed=1, omega_r=-0.01)      # 음수 waning
    with pytest.raises(ValueError):
        run_waning_seirs(100, 10, seed=1, initial_infected_frac=1.5)   # 분율 > 1
    with pytest.raises(ValueError):
        run_waning_seirs(                                     # 시드+접종 분율 합 > 1
            100, 10, seed=1,
            initial_infected_frac=0.7, initial_vaccinated_frac=0.5,
        )
