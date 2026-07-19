"""개인 단위(per-agent) SEIR 궤적 추적 — 시각화 프로덕션화 레이어.

집계 곡선만 반환하는 ``run_agent_world`` 위에, **각 agent의 일별 상태와 위치**를
기록하는 얇은 인터페이스를 얹는다. 서울 25구 ILI 합성 인구(``generate_population``)로
생성한 실제 agent(home_gu·age_band·sex·severity·work_gu·occupation)를 그대로 사용하므로
개인 궤적과 인구집단 분포를 동시에 추출할 수 있다.

설계 규율
---------
- **base 코드 미수정** (ENGINEERING_PRINCIPLES.md K-3 / D-4): ``agent_kernel.run_agent_world`` 와
  ``synthetic_population.generate_population`` 은 **import만** 한다. SEIR 동역학을
  재구현하지 않는다 — 검증된 kernel을 그대로 호출해 정직성(가짜 동역학 0)을 보장.
- **일별 상태 history 획득 방식**: kernel은 중간일 per-agent 상태를 노출하지 않으므로
  (집계 곡선 + 최종 SoA만 반환), ``run_adaptive_agent_world`` 가 쓰는 epoch-chaining
  패턴을 그대로 적용한다 — 하루씩(``T_days=2`` = init 1행 + step 1행) 호출하며
  ``initial_state`` 로 per-agent 상태를 체이닝하고, ``agents['state']`` 를 매일 기록.
  체이닝은 결정적(deterministic)이고 S+E+I+R(+V+D)=N 보존을 만족한다(TDD 검증).
- **행동 메모리 근사**: 하루 단위 체이닝은 fatigue/risk 행동 메모리를 매일 재초기화한다
  — ``run_adaptive_agent_world`` 가 epoch 경계에서 수용한 것과 동일한, 문서화된 근사다
  (행동 결합이 꺼져 있거나(기본) tau≫1일이면 무시 가능). 순수 역학(beta/sigma/gamma)은
  영향을 받지 않는다.

deep module (D-4): 공개 함수 3개(작은 인터페이스) + 내부 체이닝/집계 로직(풍부한 구현).
"""
from __future__ import annotations

import numpy as np

from simulation.abm.agent_kernel import (
    STATE_S,
    STATE_E,
    STATE_I,
    STATE_R,
    STATE_V,
    STATE_D,
    run_agent_world,
)
from simulation.abm.synthetic_population import (
    GU_NAMES,
    generate_population,
)

__all__ = [
    "simulate_with_history",
    "extract_agent_trajectory",
    "population_summary",
]

# 상태 코드 ↔ 라벨 (kernel STATE_* SSOT 기준). history_state ∈ {0,1,2,3,4,5}
STATE_LABELS: tuple[str, ...] = ("S", "E", "I", "R", "V", "D")
AGE_BAND_LABELS: tuple[str, ...] = (
    "0-9",
    "10-19",
    "20-29",
    "30-39",
    "40-49",
    "50-59",
    "60+",
)
SEX_LABELS: tuple[str, ...] = ("male", "female")
SEVERITY_LABELS: tuple[str, ...] = ("low", "high")


def simulate_with_history(
    N: int,
    T_days: int,
    *,
    seed: int = 42,
    beta: float = 0.4,
    sigma: float = 0.3,
    gamma: float = 0.2,
    delta: float = 0.0,
    nu: float = 0.0,
    import_rate: float = 0.0,
    beta_amp: float = 0.0,
    beta_phase: float = 0.0,
    theta_mean: float = 0.5,
    theta_sd: float = 0.15,
    alpha_mean: float = 0.3,
    kappa_mean: float = 0.5,
    tau_mean: float = 7.0,
    year: int | None = None,
) -> dict:
    """서울 25구 합성 인구로 per-agent SEIR 시뮬레이션 + 개인별 상태·위치 history 기록.

    ``generate_population`` 으로 실제 attribute(home_gu·age_band·sex·severity·
    occupation·work_gu)를 가진 agent를 생성한 뒤, ``run_agent_world`` 의 population
    경로를 하루씩 체이닝해 **각 agent의 매일 SEIR 상태**와 **매일 위치**(낮=work_gu,
    밤=home_gu)를 (T_days, N) 행렬로 누적한다.

    Args:
        N: agent 수. >= 1. 25의 배수가 아니어도 된다.
        T_days: 반환할 일별 행 수. day 0 = 초기 세팅 상태, 이후 ``T_days-1`` 일 진행.
        seed: 결정성 시드. 인구 생성(``generate_population``)과 kernel 체이닝 모두에
            사용. 동일 seed → 비트 동일 재현.
        beta: 감염 hazard 배수(감염 접촉일당). >= 0.
        sigma: E->I 일별 rate. >= 0.
        gamma: I->R 일별 rate. >= 0.
        delta: I->D 일별 death hazard(severity로 변조). 0.0(기본) = 사망 없음.
        nu: S->V 일별 백신 rate(스칼라 또는 길이-25 구별 벡터). >= 0.
        import_rate: 감수성자당 외부 유입 hazard. 1e-4 등 작은 값이 off-season
            확률적 소멸을 막는다. 0.0(기본) = 유입 없음.
        beta_amp: 계절 강제 진폭 epsilon. 0.0(기본) = 상수-beta.
        beta_phase: 계절 peak 일(beta_amp != 0 일 때만 사용).
        theta_mean: 평균 순응 임계값. theta_sd: per-agent 임계값 상대 SD.
        alpha_mean: 위험 민감도. kappa_mean: 피로 가중. tau_mean: 피로 시정수(일).
        year: 합성 인구 참조 연도(None = 최신). 연도별 DB 입력 period 선택에 사용.

    Returns:
        dict:
          - ``history_state``: (T_days, N) int8. 각 agent의 일별 상태 코드
            (STATE_S..STATE_D ∈ {0,1,2,3,4,5}). 행 t = day t 종료 시점 상태.
          - ``history_location``: (T_days, N) int8. 각 agent의 일별 **낮 위치**
            구 인덱스(0..24). 낮 위치 = work_gu(통근), 밤 위치 = home_gu(고정).
            낮 위치만 시간변화 — 통근자는 낮에 work_gu, 그 외엔 home_gu와 동일.
          - ``history_location_night``: (T_days, N) int8. 일별 밤 위치(= home_gu,
            상수). 낮/밤 위치 둘 다 시각화에 필요해 함께 제공.
          - ``attrs``: dict — agent 정적 속성(home_gu, work_gu, age_band, sex,
            severity, occupation), 각 (N,) 배열. ``gu_names`` 도 포함(인덱스→구 이름).
          - ``aggregate``: (T_days, 4) int64. 열 = [S, E, I, R] 일별 합계
            (시각화 SEIR 곡선용; V/D는 ``aggregate_full`` 참조).
          - ``aggregate_full``: (T_days, 6) int64. 열 = [S, E, I, R, V, D].
          - ``params``: 재현에 쓰인 시뮬 파라미터 dict.

    Raises:
        ValueError: N < 1 또는 T_days < 1, 혹은 kernel rate 검증 실패 시.

    Performance: O(T_days * N) time, O(T_days * N) memory(history 행렬 2장).
        하루씩 kernel을 호출하므로 단일 풀-런 대비 호출 오버헤드가 있다(소~중규모
        N, T_days 시각화용으로 적정; 대규모 배치 곡선만 필요하면 run_agent_world 직접 사용).
    Side effects: ``generate_population`` 이 epi_real_seoul.db 를 read-only 로 연다
        (DB write 없음). 파일시스템 write 없음.
    Caller responsibility: 모든 rate는 동일 단위의 per-day hazard. N, T_days >= 1.
    """
    N = int(N)
    T_days = int(T_days)
    if N < 1:
        raise ValueError(f"N must be >= 1; got {N}")
    if T_days < 1:
        raise ValueError(f"T_days must be >= 1; got {T_days}")

    # 1) 실제 서울 25구 합성 인구 생성 (base 코드 재사용, 미수정)
    population = generate_population(N, seed=int(seed), year=year)
    home_gu = np.asarray(population["home_gu"], dtype=np.int8)
    work_gu = np.asarray(population["work_gu"], dtype=np.int8)

    kernel_kwargs = dict(
        beta=beta,
        sigma=sigma,
        gamma=gamma,
        delta=delta,
        nu=nu,
        population=population,
        import_rate=import_rate,
        beta_amp=beta_amp,
        beta_phase=beta_phase,
        theta_mean=theta_mean,
        theta_sd=theta_sd,
        alpha_mean=alpha_mean,
        kappa_mean=kappa_mean,
        tau_mean=tau_mean,
    )

    history_state = np.empty((T_days, N), dtype=np.int8)

    # 2) day 0 = 초기 세팅 상태 (T_days=1 = init 행만, 진행 없음)
    init = run_agent_world(N=N, T_days=1, global_seed=int(seed), **kernel_kwargs)
    state = np.asarray(init["agents"]["state"], dtype=np.int8).copy()
    history_state[0] = state

    # 3) 하루씩 체이닝: T_days=2 (init 1행 + step 1행), initial_state 로 상태 전달.
    #    run_adaptive_agent_world 와 동일한 epoch-chaining 패턴 (epoch_len=1).
    #    per-epoch global_seed 오프셋으로 일별 독립 난수 스트림 확보 + 결정성 유지.
    for day in range(1, T_days):
        res = run_agent_world(
            N=N,
            T_days=2,
            global_seed=int(seed) + day,
            initial_state=state,
            **kernel_kwargs,
        )
        state = np.asarray(res["agents"]["state"], dtype=np.int8).copy()
        history_state[day] = state

    # 4) per-agent 위치 history.
    #    kernel은 낮(work) / 밤(home) 반일 스케줄을 쓴다. work_gu/home_gu 는 시뮬
    #    동안 고정(인구 경로) → 낮 위치 = work_gu(상수), 밤 위치 = home_gu(상수).
    #    위치를 (T_days, N) 으로 broadcast 해 상태 history와 정렬(프레임별 추출 편의).
    history_location = np.broadcast_to(work_gu, (T_days, N)).astype(np.int8, copy=True)
    history_location_night = np.broadcast_to(home_gu, (T_days, N)).astype(
        np.int8, copy=True
    )

    # 5) 집계 곡선 (history_state 로부터 직접 — kernel 집계와 일치, 별도 신뢰원 불필요)
    aggregate_full = _aggregate_from_history(history_state)

    attrs = {
        "home_gu": home_gu.copy(),
        "work_gu": work_gu.copy(),
        "age_band": np.asarray(population["age_band"], dtype=np.int8).copy(),
        "sex": np.asarray(population["sex"], dtype=np.int8).copy(),
        "severity": np.asarray(population["severity"], dtype=np.int8).copy(),
        "occupation": np.asarray(population["occupation"]).copy(),
        "gu_names": list(GU_NAMES),
    }

    return {
        "history_state": history_state,
        "history_location": history_location,
        "history_location_night": history_location_night,
        "attrs": attrs,
        "aggregate": aggregate_full[:, :4].copy(),  # S,E,I,R
        "aggregate_full": aggregate_full,            # S,E,I,R,V,D
        "params": {
            "N": N,
            "T_days": T_days,
            "seed": int(seed),
            "beta": float(beta),
            "sigma": float(sigma),
            "gamma": float(gamma),
            "delta": float(delta),
            "nu": nu,
            "import_rate": float(import_rate),
            "beta_amp": float(beta_amp),
            "beta_phase": float(beta_phase),
            "year": year,
        },
    }


def extract_agent_trajectory(result: dict, agent_id: int) -> dict:
    """한 agent의 정적 속성 + 상태 전이 시퀀스 + 위치 궤적을 추출.

    Args:
        result: ``simulate_with_history`` 반환 dict.
        agent_id: 추출할 agent 인덱스. 0 <= agent_id < N.

    Returns:
        dict:
          - ``agent_id``: int.
          - ``attrs``: 이 agent의 정적 속성 dict — ``home_gu``/``work_gu``(int 인덱스),
            ``home_gu_name``/``work_gu_name``(구 이름), ``age_band``/``age_band_label``,
            ``sex``/``sex_label``, ``severity``/``severity_label``, ``occupation``,
            ``is_commuter``(work_gu != home_gu).
          - ``states``: (T_days,) int8 — 일별 상태 코드.
          - ``state_labels``: list[str] — 일별 상태 라벨("S"/"E"/...).
          - ``transitions``: list[(day, state_code, state_label)] — 상태가 **바뀐**
            시점만(day 0 초기 상태 포함). 시각화 타임라인용.
          - ``location_day``: (T_days,) int8 — 일별 낮 위치 구 인덱스.
          - ``location_night``: (T_days,) int8 — 일별 밤 위치 구 인덱스.
          - ``infected_day``: int | None — 처음 E 또는 I 가 된 day(미감염이면 None).

    Raises:
        KeyError: result 에 필요한 키가 없을 때.
        IndexError: agent_id 가 [0, N) 범위 밖일 때.

    Performance: O(T_days) time/memory.
    Side effects: 없음.
    Caller responsibility: result 는 ``simulate_with_history`` 산출물이어야 한다.
    """
    history_state = result["history_state"]
    T_days, N = history_state.shape
    agent_id = int(agent_id)
    if not (0 <= agent_id < N):
        raise IndexError(f"agent_id must be in [0, {N}); got {agent_id}")

    states = np.asarray(history_state[:, agent_id], dtype=np.int8).copy()
    loc_day = np.asarray(result["history_location"][:, agent_id], dtype=np.int8).copy()
    loc_night = np.asarray(
        result["history_location_night"][:, agent_id], dtype=np.int8
    ).copy()

    attrs_all = result["attrs"]
    gu_names = attrs_all["gu_names"]
    home_idx = int(attrs_all["home_gu"][agent_id])
    work_idx = int(attrs_all["work_gu"][agent_id])
    age_idx = int(attrs_all["age_band"][agent_id])
    sex_idx = int(attrs_all["sex"][agent_id])
    sev_idx = int(attrs_all["severity"][agent_id])

    attrs = {
        "home_gu": home_idx,
        "home_gu_name": _safe_label(gu_names, home_idx),
        "work_gu": work_idx,
        "work_gu_name": _safe_label(gu_names, work_idx),
        "age_band": age_idx,
        "age_band_label": _safe_label(AGE_BAND_LABELS, age_idx),
        "sex": sex_idx,
        "sex_label": _safe_label(SEX_LABELS, sex_idx),
        "severity": sev_idx,
        "severity_label": _safe_label(SEVERITY_LABELS, sev_idx),
        "occupation": int(attrs_all["occupation"][agent_id]),
        "is_commuter": bool(work_idx != home_idx),
    }

    # 상태 전이: 값이 바뀐 시점만 (day 0 시드 상태 포함)
    transitions: list[tuple[int, int, str]] = []
    prev = None
    for day in range(T_days):
        code = int(states[day])
        if code != prev:
            transitions.append((day, code, _safe_label(STATE_LABELS, code)))
            prev = code

    infected = np.flatnonzero((states == STATE_E) | (states == STATE_I))
    infected_day = int(infected[0]) if infected.size else None

    return {
        "agent_id": agent_id,
        "attrs": attrs,
        "states": states,
        "state_labels": [_safe_label(STATE_LABELS, int(s)) for s in states],
        "transitions": transitions,
        "location_day": loc_day,
        "location_night": loc_night,
        "infected_day": infected_day,
    }


def population_summary(result: dict) -> dict:
    """인구집단 분포(연령/성별/기저질환/구) + 집계 SEIR peak 통계.

    Args:
        result: ``simulate_with_history`` 반환 dict.

    Returns:
        dict:
          - ``n_agents``: int.
          - ``age_distribution``: {age_band_label: count} (7개 밴드).
          - ``sex_distribution``: {sex_label: count}.
          - ``severity_distribution``: {severity_label: count} — 기저질환(고위험) 분포.
          - ``home_gu_distribution``: {gu_name: count} — 거주 구 분포.
          - ``commuter_count``: int — work_gu != home_gu 인 agent 수.
          - ``aggregate_peak``: {compartment: {peak, peak_day}} — S,E,I,R,V,D 각
            최대값과 도달 day(시각화 곡선 annotation 용).
          - ``attack_rate``: float — 최종 (R+D) / N (시뮬 종료 시점 누적 감염 비율).
          - ``peak_prevalence``: float — I 최대값 / N.

    Raises:
        KeyError: result 에 필요한 키가 없을 때.

    Performance: O(T_days + N) time.
    Side effects: 없음.
    Caller responsibility: result 는 ``simulate_with_history`` 산출물이어야 한다.
    """
    attrs = result["attrs"]
    age_band = np.asarray(attrs["age_band"])
    sex = np.asarray(attrs["sex"])
    severity = np.asarray(attrs["severity"])
    home_gu = np.asarray(attrs["home_gu"])
    work_gu = np.asarray(attrs["work_gu"])
    gu_names = attrs["gu_names"]
    n_agents = int(age_band.shape[0])

    age_distribution = {
        _safe_label(AGE_BAND_LABELS, b): int((age_band == b).sum())
        for b in range(len(AGE_BAND_LABELS))
    }
    sex_distribution = {
        _safe_label(SEX_LABELS, s): int((sex == s).sum())
        for s in range(len(SEX_LABELS))
    }
    severity_distribution = {
        _safe_label(SEVERITY_LABELS, v): int((severity == v).sum())
        for v in range(len(SEVERITY_LABELS))
    }
    home_counts = np.bincount(home_gu.astype(np.int64), minlength=len(gu_names))
    home_gu_distribution = {
        _safe_label(gu_names, g): int(home_counts[g]) for g in range(len(gu_names))
    }
    commuter_count = int((work_gu != home_gu).sum())

    aggregate_full = np.asarray(result["aggregate_full"])
    aggregate_peak: dict[str, dict] = {}
    for idx, name in enumerate(STATE_LABELS):
        col = aggregate_full[:, idx]
        peak_day = int(np.argmax(col))
        aggregate_peak[name] = {"peak": int(col[peak_day]), "peak_day": peak_day}

    # attack_rate: 마지막 행의 (R + D) / N
    last = aggregate_full[-1]
    r_idx = STATE_LABELS.index("R")
    d_idx = STATE_LABELS.index("D")
    attack_rate = float(last[r_idx] + last[d_idx]) / float(n_agents)
    i_idx = STATE_LABELS.index("I")
    peak_prevalence = float(aggregate_full[:, i_idx].max()) / float(n_agents)

    return {
        "n_agents": n_agents,
        "age_distribution": age_distribution,
        "sex_distribution": sex_distribution,
        "severity_distribution": severity_distribution,
        "home_gu_distribution": home_gu_distribution,
        "commuter_count": commuter_count,
        "aggregate_peak": aggregate_peak,
        "attack_rate": attack_rate,
        "peak_prevalence": peak_prevalence,
    }


def _aggregate_from_history(history_state: np.ndarray) -> np.ndarray:
    """(T_days, N) 상태 history → (T_days, 6) 일별 [S,E,I,R,V,D] 합계."""
    T_days = history_state.shape[0]
    out = np.zeros((T_days, len(STATE_LABELS)), dtype=np.int64)
    for day in range(T_days):
        counts = np.bincount(
            history_state[day].astype(np.int64), minlength=len(STATE_LABELS)
        )
        out[day] = counts[: len(STATE_LABELS)]
    return out


def _safe_label(labels, idx: int) -> str:
    """인덱스→라벨, 범위 밖이면 정수 문자열(silent 손실 없이 안전 fallback)."""
    if 0 <= idx < len(labels):
        return str(labels[idx])
    return str(int(idx))
