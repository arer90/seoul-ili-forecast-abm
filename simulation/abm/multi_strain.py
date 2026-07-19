"""다중 strain 경쟁 + 교차면역 agent-based SEIR 커널.

서울 25구 합성 인구(``synthetic_population.generate_population``) 위에서 여러
인플루엔자 strain(예: A/H1N1pdm09 · A/H3N2 · B)이 **동시에** 순환하며 서로
경쟁하는 daily binomial tau-leap agent 시뮬레이션을 제공한다. base 단일-strain
커널(``agent_kernel.run_agent_world``)의 설계 패턴(SoA 상태배열, ``1-exp(-rate)``
hazard, ``np.random.default_rng`` / ``SeedSequence`` 결정성)을 그대로 따르되,
strain-specific 감염 상태와 회복 후 **교차면역 행렬**을 추가로 모델링한다.

질병자연사(epidemiology) 가정
------------------------------
- agent 의 **phase** 는 매 step 에서 정확히 하나다: ``SUSCEPTIBLE``(현재 미감염) ·
  ``EXPOSED``(strain k 잠복) · ``INFECTIOUS``(strain k 감염력). 여기에 더해 strain별
  면역이력 boolean 벡터(``recovered_hist``)를 따로 둔다.
  **동시감염(co-infection)은 모델링하지 않는다** — 인플루엔자 동시감염은 드물고
  소표본 ABM 에서 노이즈만 키우므로(K-2 simplicity), agent 는 한 번에 한 strain.
- ★핵심: 회복은 **global 흡수상태 R 이 아니다**. strain k 에서 회복하면
  ``recovered_hist[k]=True`` 로 기록한 뒤 다시 ``SUSCEPTIBLE`` phase 로 돌아가
  *다른 strain* 에는 (교차면역이 막지 않는 한) 여전히 감염될 수 있다. 이래야
  "strain i 회복자가 strain j 에 보호" 라는 교차면역이 실제로 작동한다(global R
  흡수면 회복자가 어떤 strain 도 못 걸려 교차면역이 死구문이 됨).
- strain j 노출 시 잔여 susceptibility 는
  ``1 - max_i( recovered_hist[i] * cross_immunity[i][j] )`` 배(가장 강한 교차보호
  적용). ``cross_immunity[k][k]=1`` 이면 homologous 재감염 차단.
- 집계상 ``R`` = SUSCEPTIBLE phase 중 **모든 strain 에 (교차)면역이라 더 이상 어떤
  strain 도 못 걸리는** agent. ``S`` = 아직 걸릴 수 있는 strain 이 하나라도 남은
  SUSCEPTIBLE agent. ⇒ ``S + ΣE + ΣI + R == N`` 보존.
- strain k 에 대한 신규감염 hazard = ``beta_k * (해당 strain prevalence) *
  (잔여 susceptibility)``. beta 가 큰 strain 일수록 force-of-infection 이 커서
  경쟁에서 우점(dominant)한다.

서울 25구 ILI 데이터 grounding
------------------------------
기본 strain 라벨/시드 비율은 WHO FluNet 한국(iso3=KOR) 실측 subtype 분포에서
관측된 A/H1N1 : A/H3N2 : B ≈ 0.27 : 0.40 : 0.33 을 참고할 수 있도록, 호출자가
strain별 초기 감염 수를 ``initial_infected`` 로 직접 지정하게 한다(placeholder
하드코딩 금지 — 실제 비율은 호출자/테스트가 주입).
"""
from __future__ import annotations

import logging

import numpy as np

# base 커널/인구생성기는 import 만(재사용) — 절대 편집하지 않는다.
from simulation.abm.agent_kernel import _hazard

__all__ = ["run_multistrain", "strain_competition_summary"]

log = logging.getLogger(__name__)

# 다중-strain phase 코드. base STATE_S(=0) 와 호환되도록 SUSCEPTIBLE=0 유지.
# E / I 는 strain index 와 함께 (phase, strain) 튜플로 추적하지 않고, 메모리/속도를
# 위해 두 개의 정수배열(phase, strain_of_agent)로 SoA 분해한다. 면역이력은 별도의
# recovered_hist(N, S) bool. global 흡수 R 은 없다(위 docstring 참조).
_PHASE_S = 0  # SUSCEPTIBLE — 현재 미감염(strain_of = -1). 면역이력은 recovered_hist.
_PHASE_E = 1  # EXPOSED to strain_of
_PHASE_I = 2  # INFECTIOUS with strain_of

_NO_STRAIN = -1
_INIT_STREAM = 917_531  # base 커널과 동일 관례(초기화 스트림 분리)
_DT = 1.0


def run_multistrain(
    N,
    T_days,
    *,
    seed=42,
    betas,
    cross_immunity,
    sigma,
    gamma,
    initial_infected=None,
    population=None,
    waning=0.0,
    import_rate=0.0,
    strain_names=None,
) -> dict:
    """다중 strain 경쟁 + 교차면역 daily tau-leap SEIR 를 실행한다.

    Args:
        N: agent 수. ``population`` 미지정 시 25구 균등배치로 합성. ``>= 1``.
        T_days: 반환할 일별 집계 행 수. day 0 = 초기화 직후 상태. 정확히
            ``T_days`` 행이 반환된다. ``>= 1``.
        seed: ``np.random.default_rng`` / ``SeedSequence`` 결정성 정수 시드.
        betas: strain별 전파 hazard. 길이 ``S`` 의 1차원 array-like(각 원소는
            유한·``>= 0``). ``S`` = strain 수(2 또는 3 이상 임의). beta 가 큰
            strain 이 경쟁에서 우점.
        cross_immunity: ``(S, S)`` 행렬. ``cross_immunity[i][j]`` = strain i
            회복자가 strain j 노출에 갖는 보호확률(0=무보호, 1=완전보호).
            각 원소 ``[0, 1]``. 비대칭 허용(i→j ≠ j→i). 대각이 1 이면 동종
            재감염 차단. 행렬 ↑ 일수록 2차 strain attack rate ↓.
        sigma: 전 strain 공통 E->I 일별 rate(``>= 0``, 유한). 잠복기 ≈ 1/sigma.
        gamma: 전 strain 공통 I->R 일별 rate(``>= 0``, 유한). 감염기 ≈ 1/gamma.
        initial_infected: 길이 ``S`` 의 strain별 초기 감염 agent 수(정수,
            ``>= 0``). None(기본) 이면 각 strain 에 ``max(1, round(0.005*N))``
            명. 합이 N 을 초과하면 ValueError. 서울 KR FluNet subtype 비율
            (예: A/H1N1:A/H3N2:B ≈ 27:40:33)을 여기에 주입해 grounding 한다.
        population: ``generate_population`` SoA dict(``home_gu`` 등) 선택.
            제공 시 구(gu)별 prevalence 로 force-of-infection 을 계산(공간구조
            반영). None 이면 25구 균등배치 + 구별 mixing(home_gu 기반).
        waning: 일별 **strain별 면역소실** hazard(``>= 0``). >0 이면 회복이력
            ``recovered_hist[k]`` 가 확률적으로 풀려 strain k 재감염이 가능해진다
            (다년 재유행 모델링). phase 는 이미 SUSCEPTIBLE 이라 phase 변화 없음.
            0.0(기본)=종신면역.
        import_rate: strain별 외부유입 hazard(susceptible 당 일별). 단일 스칼라
            이면 전 strain 동일, 길이 ``S`` array-like 이면 strain별. ``>= 0``.
            0.0(기본)=유입없음. 소량(예: 1e-4)은 stochastic 소멸 방지.
        strain_names: 길이 ``S`` 의 strain 라벨 리스트. None 이면
            ``["strain_0", ...]``. 반환 dict 의 시계열 라벨에 사용.

    Returns:
        dict — 다음 키를 가진다:
            ``"S"``: ``(T_days,)`` int64. 아직 걸릴 수 있는 strain 이 남은
                SUSCEPTIBLE agent 수.
            ``"R"``: ``(T_days,)`` int64. 모든 strain 에 (교차)면역이라 더는 못
                걸리는 agent 수(global 흡수 R 이 아니라 면역상태 집계).
            ``"E"``, ``"I"``: 각 ``(T_days, S)`` int64. strain별 잠복/감염 수.
            ``"incidence"``: ``(T_days, S)`` int64. 일별 신규 strain-k 감염 수
                (S->E_k 전이; day 0 = 초기시드 수).
            ``"cumulative_incidence"``: ``(T_days, S)`` int64. 누적 신규감염.
            ``"strain_names"``: 길이 S 의 라벨 리스트.
            ``"agents"``: dict — ``"phase"``(N,) int8, ``"strain_of"``(N,) int8,
                ``"recovered_hist"``(N, S) bool(strain별 과거감염 이력),
                ``"home_gu"``(N,) int8.
            ``"N"``: int, ``"betas"``: (S,) float64.

    Raises:
        ValueError: 크기/rate/확률/행렬 shape 위반, 또는 ``sum(initial_infected)
            > N`` 시.

    Performance: O(T_days * N * S) time, O(N * S) memory.
    Side effects: ``population`` 미지정 시에도 DB 접근 없음(균등배치 합성).
        none.
    Caller responsibility: ``betas``/rate 는 모두 동일 일별 hazard 단위.
        ``cross_immunity`` 대각 원소 의미(동종 재감염 보호)를 호출자가 인지.
    """
    N = _validate_positive_int("N", N)
    T_days = _validate_positive_int("T_days", T_days)
    sigma = _validate_rate("sigma", sigma)
    gamma = _validate_rate("gamma", gamma)
    waning = _validate_rate("waning", waning)

    betas = _validate_betas(betas)
    n_strains = betas.shape[0]
    cross = _validate_cross_immunity(cross_immunity, n_strains)
    import_by_strain = _validate_import_rate(import_rate, n_strains)
    strain_names = _resolve_strain_names(strain_names, n_strains)
    seed = int(seed)

    init_seq = np.random.SeedSequence([seed, _INIT_STREAM])
    init_rng = np.random.default_rng(init_seq)

    home_gu, gu_count = _resolve_home_gu(population, N)

    initial = _resolve_initial_infected(initial_infected, n_strains, N)

    # 상태 SoA: phase(0..3) + strain_of(감염 중인 strain, S/R 이면 -1) +
    # recovered_hist(N, S) bool 면역이력.
    phase = np.full(N, _PHASE_S, dtype=np.int8)
    strain_of = np.full(N, _NO_STRAIN, dtype=np.int8)
    recovered_hist = np.zeros((N, n_strains), dtype=bool)

    # 초기 strain별 감염 agent 를 겹치지 않게 배정(co-infection 없음).
    free_idx = init_rng.permutation(N)
    cursor = 0
    incidence = np.zeros((T_days, n_strains), dtype=np.int64)
    for k in range(n_strains):
        n_k = int(initial[k])
        if n_k == 0:
            continue
        chosen = free_idx[cursor:cursor + n_k]
        cursor += n_k
        phase[chosen] = _PHASE_I
        strain_of[chosen] = k
        incidence[0, k] = n_k

    out_S = np.zeros(T_days, dtype=np.int64)
    out_R = np.zeros(T_days, dtype=np.int64)
    out_E = np.zeros((T_days, n_strains), dtype=np.int64)
    out_I = np.zeros((T_days, n_strains), dtype=np.int64)
    _record(phase, strain_of, recovered_hist, cross, n_strains,
            out_S, out_E, out_I, out_R, 0)

    # base 커널과 동일하게 day×gu 단위 child 스트림 분리(결정성·구별 독립).
    children = np.random.SeedSequence(seed).spawn(max(1, T_days * gu_count))

    for out_day in range(1, T_days):
        day = out_day - 1
        _step_day(
            phase=phase,
            strain_of=strain_of,
            recovered_hist=recovered_hist,
            home_gu=home_gu,
            gu_count=gu_count,
            betas=betas,
            cross=cross,
            sigma=sigma,
            gamma=gamma,
            waning=waning,
            import_by_strain=import_by_strain,
            n_strains=n_strains,
            incidence_row=incidence[out_day],
            day_children=children[day * gu_count:(day + 1) * gu_count],
        )
        _record(phase, strain_of, recovered_hist, cross, n_strains,
                out_S, out_E, out_I, out_R, out_day)

    return {
        "S": out_S,
        "E": out_E,
        "I": out_I,
        "R": out_R,
        "incidence": incidence,
        "cumulative_incidence": np.cumsum(incidence, axis=0),
        "strain_names": list(strain_names),
        "agents": {
            "phase": phase.copy(),
            "strain_of": strain_of.copy(),
            "recovered_hist": recovered_hist.copy(),
            "home_gu": home_gu.copy(),
        },
        "N": N,
        "betas": betas.copy(),
    }


def strain_competition_summary(result: dict) -> dict:
    """다중-strain 결과에서 우점 strain·strain별 attack rate 를 요약한다.

    Args:
        result: ``run_multistrain`` 반환 dict. ``cumulative_incidence``,
            ``strain_names``, ``N``, ``I`` 키를 사용한다.

    Returns:
        dict — 다음 키:
            ``"attack_rate"``: dict[str, float]. strain별 누적 attack rate
                (= 누적 신규감염 / N, ``[0, 1]``). 초기시드 포함.
            ``"final_cumulative_incidence"``: dict[str, int]. strain별 최종
                누적 감염 수.
            ``"dominant_strain"``: str. 최종 누적감염이 최대인 strain 라벨
                (동률이면 index 작은 쪽). 전 strain 0 이면 None.
            ``"peak_infectious"``: dict[str, int]. strain별 동시감염(I) 최댓값.
            ``"peak_day"``: dict[str, int]. strain별 I 최댓값 발생 day index.

    Raises:
        ValueError: ``result`` 가 필수 키를 결여하거나 shape 가 불일치할 때.

    Performance: O(T_days * S). Side effects: none.
    Caller responsibility: ``run_multistrain`` 의 반환 dict 를 그대로 전달.
    """
    for key in ("cumulative_incidence", "strain_names", "N", "I"):
        if key not in result:
            raise ValueError(f"result missing required key {key!r}")
    cum = np.asarray(result["cumulative_incidence"], dtype=np.int64)
    I = np.asarray(result["I"], dtype=np.int64)
    names = list(result["strain_names"])
    N = int(result["N"])
    if cum.ndim != 2 or cum.shape[1] != len(names):
        raise ValueError(
            f"cumulative_incidence shape {cum.shape} inconsistent with "
            f"{len(names)} strain_names"
        )
    if N <= 0:
        raise ValueError(f"N must be positive; got {N}")

    final_cum = cum[-1]
    attack_rate = {names[k]: float(final_cum[k]) / float(N) for k in range(len(names))}
    final_dict = {names[k]: int(final_cum[k]) for k in range(len(names))}

    if int(final_cum.sum()) == 0:
        dominant = None
    else:
        dominant = names[int(np.argmax(final_cum))]

    peak_inf = {names[k]: int(I[:, k].max()) for k in range(len(names))}
    peak_day = {names[k]: int(np.argmax(I[:, k])) for k in range(len(names))}

    return {
        "attack_rate": attack_rate,
        "final_cumulative_incidence": final_dict,
        "dominant_strain": dominant,
        "peak_infectious": peak_inf,
        "peak_day": peak_day,
    }


# --------------------------------------------------------------------------
# 내부 구현(deep module — 큰 implementation, 작은 public interface)
# --------------------------------------------------------------------------


def _step_day(
    *,
    phase: np.ndarray,
    strain_of: np.ndarray,
    recovered_hist: np.ndarray,
    home_gu: np.ndarray,
    gu_count: int,
    betas: np.ndarray,
    cross: np.ndarray,
    sigma: float,
    gamma: float,
    waning: float,
    import_by_strain: np.ndarray,
    n_strains: int,
    incidence_row: np.ndarray,
    day_children,
) -> None:
    """하루치 다중-strain 전이를 제자리(in place)에서 진행한다.

    구별(gu) prevalence 로 strain별 force-of-infection 을 계산하고, susceptible
    의 strain 선택은 강도 비례(beta_k × prevalence_k)로 — beta 큰 strain 이 우점.
    교차면역은 ``recovered_hist`` 와 ``cross`` 로 strain별 susceptibility 를 감쇠.
    """
    gu_idx = home_gu.astype(np.int64)
    is_S = phase == _PHASE_S
    is_E = phase == _PHASE_E
    is_I = phase == _PHASE_I

    # 구별 생존자 수(D 상태 없음 → 전원 alive). prevalence 분모 = 구별 인구.
    gu_pop = np.bincount(gu_idx, minlength=gu_count).astype(np.float64)

    # 구 × strain prevalence: strain k 로 감염력 있는 agent 비율.
    # prevalence_gk[gu, k]
    prevalence_gk = np.zeros((gu_count, n_strains), dtype=np.float64)
    if is_I.any():
        inf_gu = gu_idx[is_I]
        inf_strain = strain_of[is_I].astype(np.int64)
        counts = np.zeros((gu_count, n_strains), dtype=np.float64)
        np.add.at(counts, (inf_gu, inf_strain), 1.0)
        np.divide(
            counts,
            np.maximum(gu_pop, 1.0)[:, None],
            out=prevalence_gk,
            where=gu_pop[:, None] > 0.0,
        )

    next_phase = phase.copy()
    next_strain = strain_of.copy()

    # --- S -> E_k : strain별 hazard + 교차면역 + 강도비례 strain 선택 ---
    s_pos = np.flatnonzero(is_S)
    if s_pos.size:
        rng_s = np.random.default_rng(day_children[0])
        # strain별 base hazard (구 prevalence 기반) + 외부유입
        # lam_sk[i, k] : agent i 의 strain k 노출 hazard(교차면역 적용 전)
        lam_sk = (
            betas[None, :] * prevalence_gk[gu_idx[s_pos]]  # (n_s, S)
            + import_by_strain[None, :]
        )
        # 교차면역: agent 의 면역이력에 대해 strain j 잔여 susceptibility
        #   resid[i, j] = 1 - max_i_recovered( cross[i, j] )
        # recovered_hist[s_pos] (n_s, S) bool; cross (S, S)
        hist = recovered_hist[s_pos]  # (n_s, S)
        # protection[i, j] = max over recovered strain r of cross[r, j]
        # (가장 강한 교차보호 적용 — 합/평균이 아닌 max).
        protection = _max_protection(hist, cross)
        resid = np.clip(1.0 - protection, 0.0, 1.0)  # (n_s, S)
        lam_sk = lam_sk * resid

        total_lam = lam_sk.sum(axis=1)  # (n_s,)
        p_any = _hazard(total_lam)  # 1-exp(-sum lam)
        u = rng_s.random(s_pos.size)
        infected_mask = u < p_any
        if infected_mask.any():
            inf_local = np.flatnonzero(infected_mask)
            # strain 선택: hazard 비례(beta 큰 strain 우점). 결정성 위해 같은 rng.
            probs = lam_sk[inf_local]
            row_sum = probs.sum(axis=1, keepdims=True)
            probs = np.divide(
                probs, row_sum, out=np.zeros_like(probs), where=row_sum > 0.0
            )
            u2 = rng_s.random(inf_local.size)
            cdf = np.cumsum(probs, axis=1)
            chosen_strain = (u2[:, None] < cdf).argmax(axis=1).astype(np.int8)
            # row_sum==0 (전부 면역) 인 경우 감염 취소
            valid = row_sum[:, 0] > 0.0
            target = s_pos[inf_local[valid]]
            chosen_valid = chosen_strain[valid]
            next_phase[target] = _PHASE_E
            next_strain[target] = chosen_valid
            np.add.at(incidence_row, chosen_valid.astype(np.int64), 1)

    # --- E_k -> I_k : 공통 sigma ---
    e_pos = np.flatnonzero(is_E)
    if e_pos.size:
        rng_e = np.random.default_rng(day_children[min(1, gu_count - 1)])
        u = rng_e.random(e_pos.size)
        promote = e_pos[u < _hazard(sigma)]
        next_phase[promote] = _PHASE_I  # strain_of 유지

    # --- I_k -> SUSCEPTIBLE : 공통 gamma. 회복 시 strain k 면역이력 기록 ---
    # ★ global R 흡수 아님: 회복자는 SUSCEPTIBLE 로 돌아가 *다른* strain 에는 여전히
    #    걸릴 수 있다(교차면역이 막지 않는 한). 이게 교차면역을 실제 작동시킨다.
    i_pos = np.flatnonzero(is_I)
    if i_pos.size:
        rng_i = np.random.default_rng(day_children[min(2, gu_count - 1)])
        u = rng_i.random(i_pos.size)
        recover = i_pos[u < _hazard(gamma)]
        if recover.size:
            recovered_hist[recover, strain_of[recover].astype(np.int64)] = True
            next_phase[recover] = _PHASE_S
            next_strain[recover] = _NO_STRAIN

    # --- 면역소실(waning): strain별 recovered_hist 가 확률적으로 False 로 ---
    # global R->S 가 아니라 strain-specific 면역소실(다년 재유행 모델링). 이미
    # SUSCEPTIBLE phase 라 phase 변화는 없고 strain k 재감염 가능성만 복원.
    if waning > 0.0:
        rec_idx = np.flatnonzero(recovered_hist.any(axis=1))
        if rec_idx.size:
            rng_w = np.random.default_rng(day_children[min(3, gu_count - 1)])
            p_wane = _hazard(waning)
            u = rng_w.random((rec_idx.size, n_strains))
            lose = (u < p_wane) & recovered_hist[rec_idx]
            recovered_hist[rec_idx] = recovered_hist[rec_idx] & ~lose

    phase[:] = next_phase
    strain_of[:] = next_strain


def _max_protection(hist: np.ndarray, cross: np.ndarray) -> np.ndarray:
    """면역이력 hist(n, S) bool 과 cross(S, S) 에서 strain별 최대 교차보호 계산.

    Returns:
        ``(n, S)`` float64. ``out[i, j] = max_r( hist[i, r] ? cross[r, j] : 0 )``.
        회복이력이 없는 agent 행은 전부 0.
    """
    n = hist.shape[0]
    n_strains = cross.shape[0]
    out = np.zeros((n, n_strains), dtype=np.float64)
    # strain 수가 적으므로(보통 2-3) strain별 루프가 가장 단순·정확(K-2).
    # 주의: out[rec_r] 는 fancy-index 사본이므로 in-place max 가 원본에 반영되지
    # 않는다 → 명시적으로 사본을 max 한 뒤 되써넣는다.
    for r in range(n_strains):
        rec_r = hist[:, r]  # (n,) bool — strain r 회복자
        if rec_r.any():
            out[rec_r] = np.maximum(out[rec_r], cross[r][None, :])
    return out


def _record(
    phase: np.ndarray,
    strain_of: np.ndarray,
    recovered_hist: np.ndarray,
    cross: np.ndarray,
    n_strains: int,
    out_S: np.ndarray,
    out_E: np.ndarray,
    out_I: np.ndarray,
    out_R: np.ndarray,
    day: int,
) -> None:
    """현재 상태를 집계 시계열에 기록(보존 불변식의 기반).

    global 흡수 R 이 없으므로 ``R`` = SUSCEPTIBLE phase 중 **모든 strain 에 완전
    면역(잔여 susceptibility 0)** 이라 더 못 걸리는 agent, ``S`` = 걸릴 수 있는
    strain 이 하나라도 남은 SUSCEPTIBLE agent. ⇒ S + ΣE + ΣI + R == N 보존.
    """
    e_mask = phase == _PHASE_E
    i_mask = phase == _PHASE_I
    s_mask = phase == _PHASE_S
    if e_mask.any():
        out_E[day] = np.bincount(
            strain_of[e_mask].astype(np.int64), minlength=n_strains
        )[:n_strains]
    if i_mask.any():
        out_I[day] = np.bincount(
            strain_of[i_mask].astype(np.int64), minlength=n_strains
        )[:n_strains]

    # SUSCEPTIBLE phase 를 S(아직 걸릴 수 있음) vs R(전 strain 면역)로 분해.
    s_idx = np.flatnonzero(s_mask)
    if s_idx.size:
        protection = _max_protection(recovered_hist[s_idx], cross)  # (n_s, S)
        # 잔여 susceptibility 0(=protection>=1) 이 모든 strain 에 대해 성립하면 R.
        fully_immune = np.all(protection >= 1.0, axis=1)
        n_r = int(fully_immune.sum())
        out_R[day] = n_r
        out_S[day] = int(s_idx.size) - n_r
    else:
        out_R[day] = 0
        out_S[day] = 0


# --------------------------------------------------------------------------
# 검증 helper (fail-fast — base 커널 관례 준수)
# --------------------------------------------------------------------------


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


def _validate_betas(betas) -> np.ndarray:
    arr = np.asarray(betas, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] < 1:
        raise ValueError(f"betas must be a 1D array with >= 1 strain; got shape {arr.shape}")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError("betas must contain finite nonnegative values")
    return arr


def _validate_cross_immunity(cross_immunity, n_strains: int) -> np.ndarray:
    arr = np.asarray(cross_immunity, dtype=np.float64)
    if arr.shape != (n_strains, n_strains):
        raise ValueError(
            f"cross_immunity must be ({n_strains}, {n_strains}); got {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("cross_immunity must contain finite values")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError("cross_immunity entries must be in [0, 1]")
    return arr.copy()


def _validate_import_rate(import_rate, n_strains: int) -> np.ndarray:
    arr = np.asarray(import_rate, dtype=np.float64)
    if arr.ndim == 0:
        rate = _validate_rate("import_rate", float(arr))
        return np.full(n_strains, rate, dtype=np.float64)
    if arr.shape != (n_strains,):
        raise ValueError(
            f"import_rate must be scalar or shape ({n_strains},); got {arr.shape}"
        )
    if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError("import_rate must contain finite nonnegative rates")
    return arr.astype(np.float64, copy=True)


def _resolve_strain_names(strain_names, n_strains: int) -> list[str]:
    if strain_names is None:
        return [f"strain_{k}" for k in range(n_strains)]
    names = [str(s) for s in strain_names]
    if len(names) != n_strains:
        raise ValueError(
            f"strain_names length {len(names)} != betas length {n_strains}"
        )
    return names


def _resolve_initial_infected(initial_infected, n_strains: int, N: int) -> np.ndarray:
    if initial_infected is None:
        per = max(1, int(round(0.005 * N)))
        arr = np.full(n_strains, per, dtype=np.int64)
    else:
        arr = np.asarray(initial_infected)
        if arr.shape != (n_strains,):
            raise ValueError(
                f"initial_infected must have shape ({n_strains},); got {arr.shape}"
            )
        try:
            arr = arr.astype(np.int64)
        except (TypeError, ValueError) as exc:
            raise ValueError("initial_infected must contain integers") from exc
        if np.any(arr < 0):
            raise ValueError("initial_infected must be nonnegative")
    if int(arr.sum()) > N:
        raise ValueError(
            f"sum(initial_infected)={int(arr.sum())} exceeds N={N}"
        )
    return arr


def _resolve_home_gu(population, N: int) -> tuple[np.ndarray, int]:
    """``home_gu`` 배열과 구(gu) 수를 결정한다.

    ``population`` 제공 시 그 ``home_gu`` 를 사용(공간구조 반영), 미지정 시
    25구 균등 round-robin 배치(base 커널 ``_build_home_gu`` 와 동일 분포).
    """
    if population is not None:
        if not isinstance(population, dict) or "home_gu" not in population:
            raise ValueError("population must be a dict containing 'home_gu'")
        home = np.asarray(population["home_gu"])
        if home.shape != (N,):
            raise ValueError(
                f"population['home_gu'] must have shape ({N},); got {home.shape}"
            )
        home = home.astype(np.int64)
        if np.any(home < 0):
            raise ValueError("population['home_gu'] must contain nonnegative codes")
        gu_count = int(home.max()) + 1
        return home.astype(np.int16), gu_count
    # 균등 25구 배치(base 커널과 동일 관례).
    gu_count = 25
    counts = np.full(gu_count, N // gu_count, dtype=np.int64)
    counts[: N % gu_count] += 1
    home = np.repeat(np.arange(gu_count, dtype=np.int16), counts)
    return home, gu_count
