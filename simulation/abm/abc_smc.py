"""
simulation.abm.abc_smc
======================
ABC-SMC (Sequential Monte Carlo Approximate Bayesian Computation) 보정.

기존 ``simulation.abm.calibrate``(grid floor)·``abc_posterior_calibration``
(단일-round ABC rejection)·``sbi_calibration``(NPE) 을 잇는 **tolerance-annealing**
사후추정이다. 단일 round rejection 은 ε 를 한 번만 적용해 prior 가 넓으면 accept
효율이 처참하지만, ABC-SMC 는 ε_1 > ε_2 > … > ε_T 의 점감 스케줄을 따라
**이전 round 의 채택 입자(particle)를 perturbation kernel 로 교란**해 다음 round 의
proposal 로 재사용한다 — round 가 진행될수록 사후질량이 집중되는 곳에서만 샘플링하므로
약식별(weak identifiability) 상황에서도 grid 인공물 없이 신뢰구간이 축소된다
(Sisson, Fan & Tanaka 2007; Toni et al. 2009; Beaumont et al. 2009 의 adaptive
weight 보정).

알고리즘 (Toni et al. 2009, importance-weighted SMC-ABC)
-------------------------------------------------------
round 1 (ε_1):
    prior 에서 θ 추출 → simulator(θ) 와 observed 의 거리 < ε_1 이면 채택.
    채택 입자의 weight = 1 (prior 자체에서 뽑았으므로 균등).
round t ≥ 2 (ε_t):
    이전 round 입자를 weight 비례로 resample → Gaussian perturbation kernel K_t
    (대역폭 = 이전 입자 가중분산의 2배, Beaumont 2009 의 optimal-local rule)로 교란
    → 거리 < ε_t 이면 채택.
    importance weight  w_i ∝ π(θ_i) / Σ_j W_j K_t(θ_i | θ_j)
    (prior π 대비 proposal 밀도의 보정 — 이 보정이 빠지면 사후가 편향됨).

거리(distance)
-------------
simulator-비종속: ``observed`` 와 ``simulator(θ, rng)`` 출력(둘 다 1-D)의
정규화 유클리드 거리  ‖x_sim − x_obs‖₂ / √d  (d = 요약통계 차원).
NaN/비유한 시뮬레이션은 거리=inf 로 자동 기각(blow-up 가드).

불변식 (TDD ``tests/test_abc_smc.py`` 가 강제)
------------------------------------------------
- **수렴**: 알려진 θ* 의 합성 simulator 에서 posterior_mean → θ*.
- **수축**: tolerance 가 감소하면 posterior_std 가 감소(round 간 단조 비증가 경향).
- **정규화**: 반환 weights 의 합 = 1.
- **결정성**: 같은 seed → byte-동일 결과.
- **leak-free**: observed 만 거리에 들어가고 미래 정보 누출 없음(simulator 가
  observed 를 보지 않음 — 호출자 책임).

데이터
------
서울 25 구 ILI 337 주(behavioural ABM 의 (α, κ, τ, θ) 보정)를 겨냥하지만 코드는
완전 모델-비종속이다(특정 모델/파라미터 하드코딩 없음). 호출자가 simulator·priors 만
주입한다.
"""
from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np

__all__ = ["abc_smc"]


# --------------------------------------------------------------------------- #
# 내부 helper — deep module 의 캡슐화된 구현 (호출자는 abc_smc 만 안다)
# --------------------------------------------------------------------------- #
def _normalized_distance(x_sim: np.ndarray, x_obs: np.ndarray) -> float:
    """요약통계 간 차원-정규화 유클리드 거리.

    Args:
        x_sim: simulator 출력 (1-D, 길이 d). NaN/inf 포함 가능.
        x_obs: 관측 요약통계 (1-D, 길이 d).

    Returns:
        ‖x_sim − x_obs‖₂ / √d. x_sim 이 비유한이면 +inf (blow-up 기각).
    """
    if not np.all(np.isfinite(x_sim)):
        return float("inf")
    diff = x_sim - x_obs
    return float(np.sqrt(np.dot(diff, diff) / x_obs.size))


def _gaussian_kernel_logpdf(theta: np.ndarray, centers: np.ndarray,
                            cov: np.ndarray) -> np.ndarray:
    """각 center 에 놓인 동일-공분산 Gaussian 의 θ 에서의 밀도(정규화 상수 생략).

    importance weight 의 분모 Σ_j W_j K_t(θ | θ_j) 계산용. 정규화 상수는
    모든 항에 공통이라 weight 정규화 단계에서 상쇄되므로 생략한다.

    Args:
        theta: 평가점 (1-D, 길이 D).
        centers: 이전 round 입자 (n_prev, D).
        cov: perturbation kernel 공분산 (D, D), 대각.

    Returns:
        (n_prev,) — 각 center Gaussian 의 theta 에서의 (비정규화) 밀도.
    """
    inv_var = 1.0 / np.diag(cov)               # 대각 kernel → 항별 분리
    d = theta[None, :] - centers               # (n_prev, D)
    quad = np.sum(d * d * inv_var[None, :], axis=1)
    return np.exp(-0.5 * quad)


def _sample_one(
    simulator: Callable[..., np.ndarray],
    observed: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    eps: float,
    *,
    round_idx: int,
    prev_particles: np.ndarray | None,
    prev_weights: np.ndarray | None,
    kernel_cov: np.ndarray | None,
    rng: np.random.Generator,
    max_tries: int,
) -> tuple[np.ndarray, float] | None:
    """하나의 채택 입자를 생성(거리 < eps 까지 반복).

    round 1 은 prior 에서, round ≥ 2 는 이전 입자를 weight-resample 후
    Gaussian kernel 로 교란해 proposal 을 만든다. prior support 밖 제안은 기각.

    Returns:
        (theta (D,), raw_weight float) 또는 max_tries 초과 시 None.
        round 1 raw_weight = 1.0; round ≥ 2 raw_weight = π(θ)/Σ_j W_j K(θ|θ_j),
        여기서 prior π 는 box-uniform 이라 support 안에서 상수 → 분모만 남는다.
    """
    for _ in range(max_tries):
        if round_idx == 0 or prev_particles is None:
            theta = rng.uniform(lows, highs)
            raw_w = 1.0
        else:
            j = rng.choice(len(prev_particles), p=prev_weights)
            theta = rng.normal(prev_particles[j], np.sqrt(np.diag(kernel_cov)))
            if np.any(theta < lows) or np.any(theta > highs):
                continue                        # prior support 밖 → 재시도
            dens = _gaussian_kernel_logpdf(theta, prev_particles, kernel_cov)
            denom = float(np.dot(prev_weights, dens))
            if denom <= 0.0 or not np.isfinite(denom):
                continue
            raw_w = 1.0 / denom                 # uniform prior → π(θ)=const, 상쇄
        x_sim = np.asarray(simulator(theta, rng), dtype=np.float64).ravel()
        if x_sim.size != observed.size:
            raise ValueError(
                f"simulator 출력 길이 {x_sim.size} != observed 길이 {observed.size}"
            )
        if _normalized_distance(x_sim, observed) < eps:
            return theta, raw_w
    return None


def abc_smc(
    simulator: Callable[..., np.ndarray],
    observed: Sequence[float] | np.ndarray,
    priors: Mapping[str, tuple[float, float]],
    *,
    n_particles: int = 200,
    tolerance_schedule: Sequence[float] = (2.0, 1.0, 0.5, 0.25),
    seed: int = 42,
    max_tries_per_particle: int = 2000,
    kernel_scale: float = 2.0,
) -> dict:
    """tolerance-annealing SMC-ABC 사후추정(모델-비종속).

    ε_1 > ε_2 > … > ε_T 스케줄을 따라 round 마다 ``n_particles`` 개의 채택 입자를
    생성한다. round 1 은 prior 에서, 이후 round 는 이전 채택 입자를
    importance-weighted resample + Gaussian perturbation kernel(대역폭 =
    ``kernel_scale`` × 가중분산, Beaumont et al. 2009 의 local-optimal rule)로
    교란해 proposal 을 형성하고, importance weight 로 prior 대비 proposal 편향을
    보정한다(Toni et al. 2009).

    Args:
        simulator: ``callable(theta_1d: np.ndarray, rng: np.random.Generator)
            -> np.ndarray`` — 파라미터 벡터(priors 키 순서)와 RNG 를 받아 1-D
            요약통계를 반환. rng 를 받으므로 stochastic simulator 도 결정적으로
            재현된다. 출력 길이는 ``len(observed)`` 와 같아야 한다.
        observed: 관측 요약통계 (1-D, 길이 d). 서울 25 구 ILI 시계열 요약 등.
        priors: ``{param_name: (low, high)}`` box-uniform prior. dict 삽입
            순서가 theta 벡터의 축 순서를 정의한다(Python 3.7+ 보장).
        n_particles: round 당 채택 입자 수(사후 표본 크기). ≥ 2.
        tolerance_schedule: 거리 임계값 ε 의 점감 스케줄(엄격 양수, 비증가 권장).
            길이 = round 수 T.
        seed: RNG seed(``np.random.default_rng``). 결정성 보장.
        max_tries_per_particle: round 당 입자별 거리-충족 재시도 상한.
            초과 시 그 round 는 부분 채택(채택분만 사용)으로 진행.
        kernel_scale: perturbation kernel 분산 = kernel_scale × 가중 표본분산.
            2.0 = Beaumont 2009 의 점근-최적 두 배 분산.

    Returns:
        ``{
            "particles": np.ndarray (n_final, D),   # 마지막 round 채택 입자
            "weights":   np.ndarray (n_final,),     # 정규화 importance weight (합=1)
            "posterior_mean": np.ndarray (D,),       # weight 가중 평균
            "posterior_std":  np.ndarray (D,),       # weight 가중 표준편차
            "n_rounds": int,                          # 실행된 round 수
            "param_names": list[str],                 # theta 축 순서
            "accept_counts": list[int],               # round 별 채택 입자 수
            "tolerance_schedule": list[float],        # 실제 사용 ε 스케줄
            "posterior_std_per_round": list[list[float]],  # round 별 가중 std (수축 진단)
        }``
        n_final = 마지막 round 채택 입자 수(보통 n_particles, 예산 소진 시 더 작을 수 있음).

    Raises:
        ValueError: priors 가 비었거나(low ≥ high), n_particles < 2,
            tolerance_schedule 이 비었거나 비양수 ε 포함, observed 가 1-D 가 아닐 때,
            simulator 출력 길이가 observed 와 불일치할 때.
        RuntimeError: round 1 에서 단 하나의 입자도 채택 못 했을 때(ε 너무 작음).

    Performance:
        O(T · n_particles · avg_tries) simulator 호출 — avg_tries 는 ε 가 작을수록
        급증한다. round 당 weight 보정은 O(n_particles²) (kernel 밀도). 메모리는
        O(n_particles · D) 로 작다. 단일 스레드, NumPy only.

    Side effects:
        없음(디스크/DB/전역 미접촉). RNG 는 함수-로컬.

    Caller responsibility:
        - simulator 가 observed/미래 정보를 직접 보지 않아야 leak-free(누출 없음).
        - tolerance_schedule 은 simulator 거리 스케일에 맞춰 제공(거리 =
          차원-정규화 유클리드).
        - priors 키 순서 = simulator 가 기대하는 theta 축 순서.
    """
    # ---- 입력 검증 (fail-fast, D-5 gray-box 계약) ----
    if not priors:
        raise ValueError("priors 가 비었습니다 — 최소 1개 파라미터 필요")
    if n_particles < 2:
        raise ValueError(f"n_particles 는 ≥ 2 여야 합니다 (받음 {n_particles})")
    if len(tolerance_schedule) == 0:
        raise ValueError("tolerance_schedule 이 비었습니다")
    if any((not np.isfinite(e)) or e <= 0 for e in tolerance_schedule):
        raise ValueError(f"tolerance_schedule 은 유한 양수만 허용: {tolerance_schedule}")

    param_names = list(priors.keys())
    lows = np.array([priors[k][0] for k in param_names], dtype=np.float64)
    highs = np.array([priors[k][1] for k in param_names], dtype=np.float64)
    if np.any(lows >= highs):
        bad = [k for k in param_names if priors[k][0] >= priors[k][1]]
        raise ValueError(f"prior low ≥ high 인 파라미터: {bad}")

    observed = np.asarray(observed, dtype=np.float64).ravel()
    if observed.ndim != 1 or observed.size == 0:
        raise ValueError("observed 는 비지 않은 1-D 여야 합니다")

    D = len(param_names)
    rng = np.random.default_rng(seed)

    prev_particles: np.ndarray | None = None
    prev_weights: np.ndarray | None = None
    kernel_cov: np.ndarray | None = None

    accept_counts: list[int] = []
    std_per_round: list[list[float]] = []
    n_rounds = 0

    for t, eps in enumerate(tolerance_schedule):
        particles = np.empty((n_particles, D), dtype=np.float64)
        raw_weights = np.empty(n_particles, dtype=np.float64)
        filled = 0
        for _ in range(n_particles):
            res = _sample_one(
                simulator, observed, lows, highs, float(eps),
                round_idx=t,
                prev_particles=prev_particles,
                prev_weights=prev_weights,
                kernel_cov=kernel_cov,
                rng=rng,
                max_tries=max_tries_per_particle,
            )
            if res is None:
                continue                        # 예산 소진 — 부분 채택으로 진행
            theta, raw_w = res
            particles[filled] = theta
            raw_weights[filled] = raw_w
            filled += 1

        if filled == 0:
            if t == 0:
                raise RuntimeError(
                    f"round 1 에서 채택 0 — ε_1={eps} 가 너무 작거나 prior 가 "
                    "관측과 동떨어졌습니다."
                )
            # 이후 round 에서 채택 0 → 이전 round 결과로 조기 종료
            break

        particles = particles[:filled]
        raw_weights = raw_weights[:filled]
        weights = raw_weights / raw_weights.sum()   # 정규화 (합=1)

        # round 별 가중 표준편차(수축 진단)
        mean_t = np.average(particles, axis=0, weights=weights)
        var_t = np.average((particles - mean_t) ** 2, axis=0, weights=weights)
        std_per_round.append([float(s) for s in np.sqrt(var_t)])
        accept_counts.append(filled)
        n_rounds += 1

        # 다음 round 의 perturbation kernel 공분산 = scale × 가중 표본분산(대각)
        kernel_cov = np.diag(np.maximum(kernel_scale * var_t, 1e-12))
        prev_particles = particles
        prev_weights = weights

    # 최종 사후 요약 (마지막 완료 round)
    final_mean = np.average(prev_particles, axis=0, weights=prev_weights)
    final_var = np.average((prev_particles - final_mean) ** 2, axis=0,
                           weights=prev_weights)
    final_std = np.sqrt(final_var)

    return {
        "particles": prev_particles,
        "weights": prev_weights,
        "posterior_mean": final_mean,
        "posterior_std": final_std,
        "n_rounds": n_rounds,
        "param_names": param_names,
        "accept_counts": accept_counts,
        "tolerance_schedule": [float(e) for e in tolerance_schedule[:n_rounds]],
        "posterior_std_per_round": std_per_round,
    }
