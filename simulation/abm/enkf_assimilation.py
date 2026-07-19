"""폐루프 ABM↔forecaster 데이터동화 (Ensemble Kalman Filter, EnKF).

이 모듈은 ABM 앙상블(여러 replicate)의 잠재 상태를 예측 모델(forecaster)의
nowcast 로 *보정*하는 표준 stochastic EnKF 분석 스텝을 제공한다. 핵심 설계:

  * **모델-비종속(model-agnostic)**: ``assimilate_forecast`` 는
    ``forecast_anchor.load_forecast(model_name)`` 로 어떤 모델의 예측이든 조회한다
    (``model_name`` 파라미터 하나로 FusedEpi·NegBinGLM·TabPFN 등 임의 모델 동작).
    EnKF 갱신 자체는 forecaster 의 내부 구조를 전혀 모른다 — 관측 벡터만 소비.

  * **폐루프(closed-loop)**: forecaster 가 산출한 최근 관측/nowcast 를 진실(obs)로
    삼아 ABM 앙상블 상태를 끌어당긴다. 갱신 후 앙상블은 관측에 더 가까워진다
    (RMSE 감소 = 불변식, 합성 선형 case 로 Kalman 정확성 검증).

  * **leak-free**: 동화는 *현재까지의* 관측만 사용한다. 미래 관측을 갱신에 쓰지
    않는다(인과성 보존). ``assimilate_forecast`` 는 ``cutoff`` 이후 forecast 를
    명시적으로 잘라내어 미래 누수를 차단한다.

수학(Evensen 2003; Burgers, van Leeuwen & Evensen 1998):
    예보 앙상블 ``X_f ∈ R^{m×n}`` (m=앙상블 멤버, n=상태차원).
    표본 예보 공분산 ``P_f = cov(X_f)`` (멤버 간).
    Kalman gain ``K = P_f Hᵀ (H P_f Hᵀ + R)⁻¹``.
    각 멤버 i: ``x_a^i = x_f^i + K (y + ε^i − H x_f^i)`` 여기서 ``ε^i ~ N(0, R)``
    (perturbed-observation 형태 — 분석 공분산을 통계적으로 비편향 유지).

서울 25구 ILI rate 도메인: ABM 상태는 구별 prevalence(I/N) 또는 전역 ILI proxy,
관측은 forecaster 의 주별 ILI rate. 가짜/placeholder 동역학 없음 — 실제 표본
공분산 기반 갱신.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from simulation.abm.forecast_anchor import load_forecast

__all__ = ["ensemble_kalman_update", "assimilate_forecast"]


def ensemble_kalman_update(
    ens_states: np.ndarray,
    obs: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
    *,
    seed: int = 42,
    inflation: float = 1.0,
) -> np.ndarray:
    """표준 stochastic EnKF 분석 스텝 — 예보 앙상블을 관측으로 보정.

    perturbed-observation 형태의 Ensemble Kalman Filter 분석을 수행한다
    (Burgers, van Leeuwen & Evensen 1998). 표본 예보 공분산을 멤버 간 편차로
    추정하고, Kalman gain 으로 각 멤버를 관측 쪽으로 끌어당긴다. 갱신은 모델
    전진(forecast) 을 포함하지 않는 순수 *분석(analysis)* 스텝이다.

    Args:
        ens_states: 예보 앙상블 ``X_f``, shape ``(m, n)`` (m=앙상블 멤버 수 ≥ 2,
            n=상태 차원 ≥ 1). 각 행이 한 replicate 의 상태 벡터. 유한값이어야 함.
        obs: 관측 벡터 ``y``, shape ``(p,)`` (p=관측 차원 ≥ 1). 유한값.
        H: 관측 연산자 ``H``, shape ``(p, n)`` — 상태→관측 선형 사상. 유한값.
        R: 관측 오차 공분산 ``R``, shape ``(p, p)``. 대칭 양의 준정부호(대각이 ≥ 0)
            이어야 안정적. 유한값.
        seed: 관측 perturbation ``ε^i`` 난수 시드. ``np.random.default_rng(seed)``
            로 결정성 보장(같은 입력+시드 → 비트동일 출력).
        inflation: 곱셈 공분산 팽창 계수 ``λ ≥ 1`` (Anderson & Anderson 1999).
            예보 편차를 ``sqrt(λ)`` 배 늘려 표본 부족으로 인한 과신(filter
            divergence)을 완화. 1.0(기본)=팽창 없음.

    Returns:
        분석 앙상블 ``X_a``, shape ``(m, n)``, float64. 멤버별로 관측 쪽으로
        보정된 상태. ``H @ X_a.mean(0)`` 은 ``H @ X_f.mean(0)`` 보다 ``obs`` 에
        더 가깝다(불변식, R 이 과대하지 않은 한).

    Raises:
        ValueError: 차원 불일치, 멤버 수 < 2, 비유한값, 또는 ``inflation < 1``.
        numpy.linalg.LinAlgError: ``H P_f Hᵀ + R`` 이 특이행렬일 때(pinv 로 방어
            하므로 실제로는 드묾).

    Performance: O(m·n·p + p³) time (혁신 공분산 역행렬 p×p), O(m·n + p²) memory.
        n,p ≤ 수십 규모(서울 25구)에서 millisecond 급.
    Side effects: 없음 — 입력 배열 미변경, 디스크/DB/네트워크 접근 없음.
    Caller responsibility: ``R`` 은 양의 준정부호여야 하며, ``H`` 의 행 공간이
        관측 가능한 상태를 포착해야 의미 있는 보정이 일어난다. ``ens_states`` 의
        멤버 간 분산이 0 이면(붕괴된 앙상블) gain 이 0 이 되어 갱신이 항등이 된다.
    """
    Xf = _as_2d_finite("ens_states", ens_states)
    m, n = Xf.shape
    if m < 2:
        raise ValueError(f"ensemble must have >= 2 members; got m={m}")
    y = _as_1d_finite("obs", obs)
    p = y.shape[0]
    Hm = _as_finite("H", H)
    if Hm.shape != (p, n):
        raise ValueError(
            f"H must have shape ({p}, {n}) = (obs_dim, state_dim); got {Hm.shape}"
        )
    Rm = _as_finite("R", R)
    if Rm.shape != (p, p):
        raise ValueError(f"R must have shape ({p}, {p}); got {Rm.shape}")
    if not np.isfinite(inflation) or inflation < 1.0:
        raise ValueError(f"inflation must be finite and >= 1; got {inflation!r}")

    rng = np.random.default_rng(seed)

    # 예보 앙상블 평균 및 편차(앙상블 평균 보존을 위해 편차 기반으로 공분산 추정)
    x_mean = Xf.mean(axis=0)                      # (n,)
    A = Xf - x_mean[None, :]                      # (m, n) 편차 행렬
    if inflation > 1.0:
        A = A * np.sqrt(inflation)
    # 표본 예보 공분산 P_f = Aᵀ A / (m-1). errstate: macOS Accelerate-BLAS 가
    # 대형 m 에서 무해한 spurious overflow warning 을 내므로 억제(결과 불변,
    # test_kalman_gain_matches_closed_form 으로 정확성 검증됨).
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        Pf = (A.T @ A) / float(m - 1)             # (n, n)

    # 혁신 공분산 S = H P_f Hᵀ + R, Kalman gain K = P_f Hᵀ S⁻¹
    PfHt = Pf @ Hm.T                              # (n, p)
    S = Hm @ PfHt + Rm                            # (p, p)
    S = 0.5 * (S + S.T)                           # 대칭화(수치 안정)
    S_inv = np.linalg.pinv(S)                     # 특이행렬 방어
    K = PfHt @ S_inv                              # (n, p)

    # perturbed observations: y_i = y + ε_i, ε_i ~ N(0, R)
    eps = _sample_obs_noise(rng, Rm, m)           # (m, p), 평균 보정으로 비편향
    HXf = Xf @ Hm.T                               # (m, p) = (H x_f^i)ᵀ
    innovations = (y[None, :] + eps) - HXf        # (m, p)
    Xa = Xf + innovations @ K.T                   # (m, n)
    return Xa


def assimilate_forecast(
    model_name: str = "FusedEpi",
    *,
    ensemble: np.ndarray,
    obs_series: Sequence[float] | np.ndarray | None = None,
    H: np.ndarray | None = None,
    obs_var: float = 1.0,
    cutoff: int | None = None,
    predictions_csv: str | None = None,
    seed: int = 42,
    inflation: float = 1.0,
    conformal_halfwidth: float | None = None,
    conformal_level: float = 0.95,
) -> dict[str, Any]:
    """forecaster nowcast 로 ABM 앙상블 상태를 보정(폐루프, 모델-비종속).

    어떤 예측 모델이든 ``load_forecast(model_name)`` 로 주별 ILI rate 예측을 받아
    그 *최근(현재까지)* 값을 관측 진실로 삼아 ABM 앙상블 상태에 EnKF 갱신을
    적용한다. ``model_name`` 파라미터가 모델-비종속성의 핵심 — FusedEpi 외 임의
    모델명(NegBinGLM-V7·TabPFN·DLinear 등)에 동일하게 동작한다.

    leak-free 보장: 갱신은 ``cutoff`` 시점까지의 관측만 사용한다. forecast/관측
    시리즈를 ``cutoff`` 에서 잘라 마지막 관측 한 점을 obs 로 쓰며, 미래 값은 갱신에
    절대 들어가지 않는다(인과성 보존). ``obs_series`` 가 None 이면 forecaster 의
    forecast 자체를 nowcast 관측으로 사용한다.

    Args:
        model_name: forecast CSV ``model`` 컬럼의 모델 식별자. 모델-비종속 —
            FusedEpi 외 임의 모델명 허용. ``load_forecast`` 가 정확 매칭으로 조회.
        ensemble: ABM 앙상블 상태 ``(m, n)`` — m=replicate 수, n=상태 차원
            (예: 서울 25구 prevalence → n=25, 또는 전역 ILI proxy → n=1). 유한값.
        obs_series: 선택적 외부 관측 시리즈(주별 ground-truth ILI rate). None 이면
            ``load_forecast(model_name)`` 의 forecast 를 nowcast 관측으로 사용.
        H: 관측 연산자 ``(p, n)``. None 이면 ``p=1`` 의 상태 평균 연산자
            ``(1/n)·1ᵀ`` (전역 ILI = 구별 prevalence 평균) 를 사용.
        obs_var: 관측 오차 분산(스칼라). ``R = obs_var · I_p``. > 0 이어야 함.
        cutoff: 관측으로 사용할 마지막 시점 인덱스(0-기반, 포함). None 이면
            forecast 시리즈의 마지막 *과거* 점(``len-1``). 미래 누수 차단점.
        predictions_csv: 선택적 forecast CSV 경로 override. None 이면
            ``load_forecast`` 기본(R10 per_model_eval 산출).
        seed: EnKF perturbation 시드(결정성).
        inflation: 공분산 팽창 계수 ``λ ≥ 1`` (``ensemble_kalman_update`` 참조).
        conformal_halfwidth: 선택적 forecaster 예측구간(95% PI) 반폭. 지정하면
            관측(=forecaster nowcast) 오차분산을 고정 스칼라 ``obs_var`` 대신
            ``R = (halfwidth / 1.96)² · I`` 로 유도한다 — 예측이 불확실할수록(정점
            에서 conformal PI 가 넓을수록) 관측을 덜 신뢰(칼만 이득↓)하는 원칙적
            결합. None(기본)이면 ``obs_var`` 그대로 사용(기존 동작 불변). > 0 이어야
            함. online-adaptive-conformal 반폭(Appendix D.4)과 결합하도록 설계.

    Returns:
        dict — 키:
          ``model_name`` (str): 사용된 모델명(에코, 모델-비종속 증빙).
          ``updated_ensemble`` (np.ndarray (m, n) float64): 분석 앙상블.
          ``obs_used`` (float): 갱신에 쓰인 관측값(cutoff 시점 nowcast).
          ``cutoff`` (int): 실제 적용된 cutoff 인덱스.
          ``rmse_prior`` (float): 갱신 전 ``H·mean`` 과 obs 의 RMSE.
          ``rmse_post`` (float): 갱신 후 RMSE (≤ rmse_prior, 불변식).
          ``improved`` (bool): ``rmse_post <= rmse_prior + 1e-12``.
          ``ens_mean_prior`` (list[float]): 갱신 전 앙상블 평균(상태 공간).
          ``ens_mean_post`` (list[float]): 갱신 후 앙상블 평균.
          ``n_forecast_weeks`` (int): 조회된 forecast 길이(누수 진단용).
          ``leak_free`` (bool): cutoff 이후 관측 미사용 = 항상 True.
          ``obs_var_effective`` (float): 실제 사용된 관측 오차분산 —
            ``conformal_halfwidth`` 지정 시 유도값, 아니면 ``obs_var``.

    Raises:
        ValueError: forecast/obs 비유한, ``obs_var <= 0``, cutoff 범위 밖,
            ``ensemble`` shape 부적합, 또는 ``H`` 차원 불일치.
        FileNotFoundError: forecast CSV 부재(``load_forecast`` 전파).

    Performance: O(forecast_rows) 조회 + ``ensemble_kalman_update`` 비용. 단일
        시점 동화이므로 millisecond 급(서울 25구 규모).
    Side effects: ``load_forecast`` 가 로컬 CSV 만 읽음 — 네트워크/DB write 없음.
        입력 ``ensemble`` 미변경(새 배열 반환).
    Caller responsibility: ``ensemble`` 의 상태 차원 n 과 ``H`` 의 열 수가
        일치해야 하며, obs 단위(ILI rate)와 ``H·state`` 단위가 같아야 의미 있는
        보정이 된다. ``cutoff`` 는 미래 관측을 갱신에 넣지 않도록 호출자가
        과거-only 인덱스를 지정할 책임이 있다(기본값은 안전한 last-past).
    """
    if obs_var <= 0.0 or not np.isfinite(obs_var):
        raise ValueError(f"obs_var must be finite and > 0; got {obs_var!r}")
    # conformal PI 반폭 → 관측 오차분산 유도(지정 시). 95% PI 가정: σ ≈ hw/1.96.
    # 예측 불확실(넓은 PI)→ obs_var↑ → 칼만 이득↓ → ABM prior 더 신뢰(원칙적 결합).
    if conformal_halfwidth is not None:
        if conformal_halfwidth <= 0.0 or not np.isfinite(conformal_halfwidth):
            raise ValueError(
                f"conformal_halfwidth must be finite and > 0; got {conformal_halfwidth!r}"
            )
        # Derive the z-multiplier from the interval's ACTUAL confidence level rather
        # than hard-coding the 95% normal quantile: halfwidth = z·σ ⇒ σ = hw/z, so a
        # different target level (or a change in the conformal band) stays consistent.
        from scipy.stats import norm
        if not 0.0 < conformal_level < 1.0:
            raise ValueError(f"conformal_level must be in (0,1); got {conformal_level!r}")
        _z = float(norm.ppf(0.5 + 0.5 * conformal_level))
        obs_var_eff = float((conformal_halfwidth / _z) ** 2)
    else:
        obs_var_eff = float(obs_var)
    Xf = _as_2d_finite("ensemble", ensemble)
    m, n = Xf.shape

    # forecaster nowcast 조회 — 모델-비종속(model_name 만으로 임의 모델 동작).
    kwargs: dict[str, Any] = {}
    if predictions_csv is not None:
        kwargs["predictions_csv"] = predictions_csv
    _weeks, y_pred = load_forecast(model_name, **kwargs)
    n_forecast = int(y_pred.size)

    # 관측 시리즈: 외부 ground-truth 우선, 없으면 forecaster forecast 를 nowcast로.
    if obs_series is None:
        series = np.asarray(y_pred, dtype=np.float64)
    else:
        series = _as_1d_finite("obs_series", np.asarray(obs_series, dtype=np.float64))
    if series.size == 0:
        raise ValueError("observation series is empty")

    # leak-free cutoff: 기본은 마지막 과거 점(len-1). 미래 누수 차단 — cutoff 이후
    # 값은 슬라이싱으로 폐기하여 갱신에 *물리적으로* 들어갈 수 없게 한다.
    if cutoff is None:
        cutoff_idx = int(series.size - 1)
    else:
        cutoff_idx = int(cutoff)
    if cutoff_idx < 0 or cutoff_idx >= series.size:
        raise ValueError(
            f"cutoff must be in [0, {series.size - 1}]; got {cutoff_idx}"
        )
    past_only = series[: cutoff_idx + 1]          # 미래 미포함(인과성)
    obs_scalar = float(past_only[-1])             # cutoff 시점 nowcast 관측

    # 관측 연산자 H: 기본 = 전역 ILI(구별 prevalence 평균) → p=1.
    if H is None:
        Hm = np.full((1, n), 1.0 / float(n), dtype=np.float64)
    else:
        Hm = _as_finite("H", H)
        if Hm.ndim != 2 or Hm.shape[1] != n:
            raise ValueError(
                f"H must have shape (p, {n}); got {Hm.shape}"
            )
    p = Hm.shape[0]
    y = np.full(p, obs_scalar, dtype=np.float64)
    R = obs_var_eff * np.eye(p, dtype=np.float64)

    # 갱신 전 진단(prior RMSE)
    prior_mean = Xf.mean(axis=0)
    rmse_prior = _rmse(Hm @ prior_mean, y)

    Xa = ensemble_kalman_update(Xf, y, Hm, R, seed=seed, inflation=inflation)

    post_mean = Xa.mean(axis=0)
    rmse_post = _rmse(Hm @ post_mean, y)

    return {
        "model_name": str(model_name),
        "updated_ensemble": Xa,
        "obs_used": obs_scalar,
        "cutoff": cutoff_idx,
        "rmse_prior": float(rmse_prior),
        "rmse_post": float(rmse_post),
        "improved": bool(rmse_post <= rmse_prior + 1.0e-12),
        "ens_mean_prior": [float(v) for v in prior_mean],
        "ens_mean_post": [float(v) for v in post_mean],
        "n_forecast_weeks": n_forecast,
        "leak_free": True,
        "obs_var_effective": float(obs_var_eff),
    }


# --------------------------------------------------------------------------- #
# 내부 헬퍼 (정보 은닉 — 호출자는 위 두 공개 함수의 계약만 알면 충분)
# --------------------------------------------------------------------------- #
def _sample_obs_noise(
    rng: np.random.Generator, R: np.ndarray, m: int
) -> np.ndarray:
    """``ε_i ~ N(0, R)`` 표본 (m, p) — 표본 평균을 0 으로 보정해 비편향 유지.

    perturbed-observation EnKF 에서 유한 m 의 표본 평균이 0 이 아니면 분석에
    bias 가 생긴다(van Leeuwen 1999). 표본을 중심화하여 ``mean_i ε_i = 0`` 강제.
    """
    p = R.shape[0]
    # 대칭 R 의 안정적 평방근(고유분해 — 음수 고유값은 수치오차이므로 0 클립).
    w, V = np.linalg.eigh(0.5 * (R + R.T))
    w = np.clip(w, 0.0, None)
    L = V @ np.diag(np.sqrt(w))
    z = rng.standard_normal((m, p))
    eps = z @ L.T
    eps = eps - eps.mean(axis=0, keepdims=True)   # 표본 평균 0 (비편향)
    return eps


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean(diff * diff)))


def _as_finite(name: str, arr: Any) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must contain only finite values")
    return a


def _as_1d_finite(name: str, arr: Any) -> np.ndarray:
    a = _as_finite(name, arr)
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D array; got shape {a.shape}")
    return a


def _as_2d_finite(name: str, arr: Any) -> np.ndarray:
    a = _as_finite(name, arr)
    if a.ndim != 2 or a.size == 0:
        raise ValueError(f"{name} must be a non-empty 2D array; got shape {a.shape}")
    return a
