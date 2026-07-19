"""SEIRS-V-D 면역소실(waning) + 재감염(reinfection) 에이전트 커널.

2022-24 ILI rebound 기전을 모델링하기 위한 모듈이다. 기존 ``agent_kernel``
의 SEIR-V-D 가 R(회복)·V(접종) 을 **종신 면역**(terminal) 으로 취급하는 데
비해, 여기서는 회복·접종 면역이 시간에 따라 **소실**되어 다시 감수성(S)으로
돌아가는 SEIRS-V-D 동역학을 구현한다:

    S → E → I → R  (회복)        R --omega_r--> S   (회복면역 소실)
    S ----nu----> V (접종)        V --omega_v--> S   (접종면역 소실)
    I --delta--> D  (사망, 종착)

면역소실로 인해 재감수성이 보충되므로 단일 시즌 SEIR 과 달리 **다년 재유행
(2차/3차 파)** 과 **재감염(reinfection)** 이 발생할 수 있다. 이 모듈은 그
재감염 횟수를 명시적으로 집계한다.

이 모듈은 ``agent_kernel`` 의 compartment 상수와 일별 ``1-exp(-rate)``
tau-leap 변환 헬퍼를 **import 만** 하여 재사용한다(원본 무수정). 행동 layer
(risk/fatigue compliance) 는 포함하지 않는 순수 SEIRS-V-D 기전 엔진이다 —
모델 비종속(특정 forecaster/모델 하드코딩 없음).
"""
from __future__ import annotations

import numpy as np

# 기존 커널의 SSOT compartment 코드 + tau-leap 헬퍼를 import 만(원본 무수정).
from simulation.abm.agent_kernel import (
    STATE_S,
    STATE_E,
    STATE_I,
    STATE_R,
    STATE_V,
    STATE_D,
    _hazard,
)

__all__ = ["run_waning_seirs"]

_COMPARTMENT_NAMES = ("S", "E", "I", "R", "V", "D")
_N_COMPARTMENTS = 6
_DT = 1.0


def run_waning_seirs(
    N: int,
    T_days: int,
    *,
    seed: int = 42,
    beta: float = 0.5,
    sigma: float = 0.5,
    gamma: float = 0.25,
    omega_r: float = 0.0,
    omega_v: float = 0.0,
    nu: float = 0.0,
    delta: float = 0.0,
    import_rate: float = 0.0,
    initial_infected_frac: float = 0.01,
    initial_vaccinated_frac: float = 0.0,
) -> dict:
    """일별 이항 tau-leap SEIRS-V-D(면역소실+재감염) 월드를 실행한다.

    전 인구를 단일 well-mixed pool 로 취급하고, 매일 각 compartment 의
    전이를 ``p = 1 - exp(-rate)`` 확률의 이항 추출(개별 에이전트 단위)로
    진행한다. 감염력(force of infection)은 ``lambda = beta * I/alive
    + import_rate`` 로, 살아있는 인구 대비 감염 유병률에 비례한다.

    면역소실 항(omega_r, omega_v)이 R·V 를 S 로 되돌려 재감수성을 보충하므로
    단일 시즌 SEIR 과 달리 2차/3차 파와 재감염이 발생할 수 있다. 한 번이라도
    회복(R)·접종(V)을 거친 뒤 다시 감염(E 진입)되는 에이전트를 재감염으로
    집계한다.

    Args:
        N: 에이전트 수. 1 이상의 정수.
        T_days: 반환할 일별 집계 상태 수. day 0 은 초기화 월드이므로 정확히
            ``T_days`` 행이 반환된다.
        seed: 결정성 시드(정수). 모든 난수는
            ``np.random.default_rng(seed)`` 단일 스트림에서 소비되어, 같은
            seed → bit-identical, 다른 seed → 다른 동역학을 보장한다.
        beta: 감염성 접촉당 전파 위험률(일 단위 hazard). >= 0.
        sigma: E→I 일별 전이율(잠복기 역수). >= 0.
        gamma: I→R 일별 회복률. >= 0.
        omega_r: R→S 일별 회복면역 소실률. 0.0(기본)=종신 회복면역.
            >0 이면 회복자가 감수성으로 돌아가 재유행/재감염을 유발한다.
        omega_v: V→S 일별 접종면역 소실률. 0.0(기본)=종신 접종면역.
            >0 이면 접종자가 감수성으로 돌아간다.
        nu: S→V 일별 접종률. >= 0. 접종 캠페인을 모델링.
        delta: I→D 일별 사망률. 0.0(기본)=사망 없음. >= 0.
        import_rate: 감수성당 외부 유입 감염 위험률(일 단위, flat). 작은 값
            (예: 1e-3)은 off-season 확률적 소멸을 막아 다년 파를 재점화한다.
            0.0(기본)=유입 없음.
        initial_infected_frac: t=0 초기 감염 분율(나머지 감수성에서 추출).
            [0, 1]. 최소 1명(분율>0 이고 가용 감수성>0 일 때)을 보장.
        initial_vaccinated_frac: t=0 초기 접종(V) 분율. [0, 1]. 초기 감염
            시드는 남은 감수성에서만 뽑는다.

    Returns:
        dict — 다음 키를 포함:
            ``S,E,I,R,V,D``: 각 ``(T_days,)`` int64 일별 compartment 카운트.
            ``incidence``: ``(T_days,)`` int64 — 당일 신규 감염(S→E) 수.
                day 0 은 초기 감염 시드 수.
            ``reinfections``: ``(T_days,)`` int64 — 당일 신규 감염 중 과거에
                R 또는 V 를 경험했던 에이전트(재감염) 수. day 0 = 0.
            ``cumulative_reinfections``: int — 전 기간 누적 재감염 총수.
            ``agents``: dict — 최종 SoA 배열(``state`` (N,) int8,
                ``ever_immune`` (N,) bool — R/V 경험 여부,
                ``infection_count`` (N,) int32 — 에이전트별 누적 감염 횟수).

    Raises:
        ValueError: N/T_days 가 양수 아님, 음수·비유한 rate, 분율 범위 위반,
            또는 ``initial_infected_frac + initial_vaccinated_frac`` 가 1 초과.

    Performance: O(T_days * N) time, O(N) memory.
    Side effects: 없음(disk/DB/global 미접촉, 순수 함수).
    Caller responsibility: 모든 rate 는 동일한 일(day) 단위 hazard 로 전달.
        omega_r=omega_v=0 이면 재감염이 0(SEIR 환원)이 되는 것이 불변식이다.
    """
    N = _validate_positive_int("N", N)
    T_days = _validate_positive_int("T_days", T_days)
    beta = _validate_rate("beta", beta)
    sigma = _validate_rate("sigma", sigma)
    gamma = _validate_rate("gamma", gamma)
    omega_r = _validate_rate("omega_r", omega_r)
    omega_v = _validate_rate("omega_v", omega_v)
    nu = _validate_rate("nu", nu)
    delta = _validate_rate("delta", delta)
    import_rate = _validate_rate("import_rate", import_rate)
    initial_infected_frac = _validate_frac("initial_infected_frac", initial_infected_frac)
    initial_vaccinated_frac = _validate_frac(
        "initial_vaccinated_frac", initial_vaccinated_frac
    )
    if initial_infected_frac + initial_vaccinated_frac > 1.0 + 1e-9:
        raise ValueError(
            "initial_infected_frac + initial_vaccinated_frac must be <= 1; got "
            f"{initial_infected_frac + initial_vaccinated_frac}"
        )

    rng = np.random.default_rng(int(seed))

    state = np.full(N, STATE_S, dtype=np.int8)
    # 한 번이라도 R 또는 V 를 경험했는지(재감염 판정용). 초기 접종자는 즉시 면역경험.
    ever_immune = np.zeros(N, dtype=bool)
    infection_count = np.zeros(N, dtype=np.int32)

    # 초기 접종(V): 감수성 풀에서 추출.
    n_vax0 = int(round(N * initial_vaccinated_frac))
    n_vax0 = min(n_vax0, N)
    if n_vax0 > 0:
        vax_idx = rng.choice(N, size=n_vax0, replace=False)
        state[vax_idx] = STATE_V
        ever_immune[vax_idx] = True

    # 초기 감염(I): 남은 감수성에서만 추출.
    susceptible_idx = np.flatnonzero(state == STATE_S)
    if initial_infected_frac > 0.0 and susceptible_idx.size > 0:
        n_inf0 = max(1, int(round(N * initial_infected_frac)))
        n_inf0 = min(n_inf0, susceptible_idx.size)
        inf_idx = rng.choice(susceptible_idx, size=n_inf0, replace=False)
        state[inf_idx] = STATE_I
        infection_count[inf_idx] = 1
    else:
        n_inf0 = 0

    out = {name: np.zeros(T_days, dtype=np.int64) for name in _COMPARTMENT_NAMES}
    incidence = np.zeros(T_days, dtype=np.int64)
    reinfections = np.zeros(T_days, dtype=np.int64)
    _record_counts(state, out, 0)
    incidence[0] = n_inf0  # day0 신규감염 = 초기 시드; 재감염은 정의상 0.

    p_sigma = float(_hazard(sigma))
    p_gamma_die = _hazard(gamma + delta)  # I 의 총 유출(회복+사망)
    p_omega_r = float(_hazard(omega_r))
    p_omega_v = float(_hazard(omega_v))
    p_nu = float(_hazard(nu))

    for out_day in range(1, T_days):
        alive = state != STATE_D
        alive_total = int(alive.sum())
        n_infected = int((state == STATE_I).sum())
        prevalence = (
            float(n_infected) / float(alive_total) if alive_total > 0 else 0.0
        )
        lam = beta * prevalence + import_rate

        next_state = state.copy()
        day_incidence = 0
        day_reinf = 0

        # S → E (감염) 또는 S → V (접종) 경쟁 위험.
        s_pos = np.flatnonzero(state == STATE_S)
        if s_pos.size:
            total_rate = lam + nu
            p_out = float(_hazard(total_rate))
            if total_rate > 0.0:
                p_inf = p_out * lam / total_rate
                p_vax = p_out - p_inf
            else:
                p_inf = 0.0
                p_vax = 0.0
            u = rng.random(s_pos.size)
            infected_sel = s_pos[u < p_inf]
            vax_sel = s_pos[(u >= p_inf) & (u < (p_inf + p_vax))]
            if infected_sel.size:
                next_state[infected_sel] = STATE_E
                infection_count[infected_sel] += 1
                day_incidence = int(infected_sel.size)
                # 재감염 = 과거 R/V 를 경험한 에이전트의 신규 감염.
                day_reinf = int(ever_immune[infected_sel].sum())
            if vax_sel.size:
                next_state[vax_sel] = STATE_V
                ever_immune[vax_sel] = True

        # E → I (잠복 종료).
        e_pos = np.flatnonzero(state == STATE_E)
        if e_pos.size and p_sigma > 0.0:
            u = rng.random(e_pos.size)
            next_state[e_pos[u < p_sigma]] = STATE_I

        # I → R (회복) / I → D (사망) 경쟁 위험.
        i_pos = np.flatnonzero(state == STATE_I)
        if i_pos.size:
            total_rate = gamma + delta
            if total_rate > 0.0:
                p_out = float(p_gamma_die)
                p_rec = p_out * gamma / total_rate
                p_die = p_out - p_rec
                u = rng.random(i_pos.size)
                rec_sel = i_pos[u < p_rec]
                die_sel = i_pos[(u >= p_rec) & (u < (p_rec + p_die))]
                if rec_sel.size:
                    next_state[rec_sel] = STATE_R
                    ever_immune[rec_sel] = True
                if die_sel.size:
                    next_state[die_sel] = STATE_D

        # R → S (회복면역 소실).
        if p_omega_r > 0.0:
            r_pos = np.flatnonzero(state == STATE_R)
            if r_pos.size:
                u = rng.random(r_pos.size)
                next_state[r_pos[u < p_omega_r]] = STATE_S

        # V → S (접종면역 소실).
        if p_omega_v > 0.0:
            v_pos = np.flatnonzero(state == STATE_V)
            if v_pos.size:
                u = rng.random(v_pos.size)
                next_state[v_pos[u < p_omega_v]] = STATE_S

        state = next_state
        _record_counts(state, out, out_day)
        incidence[out_day] = day_incidence
        reinfections[out_day] = day_reinf

    out["incidence"] = incidence
    out["reinfections"] = reinfections
    out["cumulative_reinfections"] = int(reinfections.sum())
    out["agents"] = {
        "state": state.copy(),
        "ever_immune": ever_immune.copy(),
        "infection_count": infection_count.copy(),
    }
    return out


def _record_counts(state: np.ndarray, out: dict[str, np.ndarray], day: int) -> None:
    counts = np.bincount(state.astype(np.int64), minlength=_N_COMPARTMENTS)
    for idx, name in enumerate(_COMPARTMENT_NAMES):
        out[name][day] = int(counts[idx])


def _validate_positive_int(name: str, value) -> int:
    try:
        ivalue = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer; got {value!r}") from exc
    if ivalue < 1:
        raise ValueError(f"{name} must be >= 1; got {ivalue}")
    return ivalue


def _validate_rate(name: str, value) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and >= 0; got {value!r}")
    return value


def _validate_frac(name: str, value) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1]; got {value!r}")
    return value
