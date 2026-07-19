"""simulation.abm.observation_model

잠재 감염상태(SEIR-V-D) → 관측치(ILI 신고 · 바이러스 양성률) 사상.

WHO: ILI 는 임상상 타 호흡기 바이러스와 구분되지 않아 실험실 확인이 필요하다. 따라서
기계론적 ABM 의 잠재 일일 신규감염 I_t^new 을 **관측 가능한** ILI rate / 양성률로 잇는
관측모형이 있어야 "전염 시뮬레이션"과 "감시자료 평가"가 연결된다 (논문 §관측모형).

관측 방정식
-----------
증상화:        I_t^sym = symptomatic_frac · I_t^new
ILI 신고:      Y_t^ILI ~ NegBin(mean = ρ · I_t^sym, dispersion = φ)
                 (음이항: 과분산 — Var = μ + μ²/φ; φ→∞ 면 Poisson)
바이러스 양성률: Pos_t ~ Binom(n_t, π_t),  π_t = I_t^sym / (I_t^sym + B_t)
                 (B_t = 비-인플루엔자 호흡기 배경; FluNet/검사 양성률 연결)

ρ = 의료이용·보고율 (care-seeking × reporting), 연령가중 가능.

Gray-box 계약
-------------
- 결정적 평균: ``ili_mean(infections_sym, rho)`` = ρ·I^sym (NaN/음수 입력 → ValueError).
- 확률 샘플: ``sample_ili(...)`` 는 ``rng`` 필수 (재현성). Gamma-Poisson 혼합 = NegBin.
- 우도: ``negbin_loglik(y_obs, mu, phi)`` — 보정/평가용 (scipy 불요, math.lgamma).
- 부작용 없음(순수 함수). Performance: O(T).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObservationParams:
    """관측모형 파라미터 (frozen — 재현성).

    Attributes:
        symptomatic_frac: 감염→증상화 비율 (인플루엔자 ~0.5–0.7). 범위 (0, 1].
        care_seeking: 증상자 중 ILI 진료/보고 비율 ρ 의 핵심항. 범위 (0, 1].
        reporting_rate: 표본감시 포착률(추가 ρ 항). 범위 (0, 1].
        nb_dispersion: 음이항 φ (>0). 작을수록 과분산. 큰 값 → Poisson 근사.
        background_rate: 양성률용 비-인플루엔자 호흡기 배경 B (≥0, 일일 등가 규모).
    """
    symptomatic_frac: float = 0.667
    care_seeking: float = 0.40
    reporting_rate: float = 1.0
    nb_dispersion: float = 10.0
    background_rate: float = 50.0

    @property
    def rho(self) -> float:
        """보고 계수 ρ = care_seeking × reporting_rate."""
        return float(self.care_seeking) * float(self.reporting_rate)

    def validate(self) -> None:
        if not (0.0 < self.symptomatic_frac <= 1.0):
            raise ValueError(f"symptomatic_frac ∈ (0,1] 위반: {self.symptomatic_frac}")
        if not (0.0 < self.care_seeking <= 1.0):
            raise ValueError(f"care_seeking ∈ (0,1] 위반: {self.care_seeking}")
        if not (0.0 < self.reporting_rate <= 1.0):
            raise ValueError(f"reporting_rate ∈ (0,1] 위반: {self.reporting_rate}")
        if not (self.nb_dispersion > 0.0 and math.isfinite(self.nb_dispersion)):
            raise ValueError(f"nb_dispersion > 0 위반: {self.nb_dispersion}")
        if not (self.background_rate >= 0.0 and math.isfinite(self.background_rate)):
            raise ValueError(f"background_rate ≥ 0 위반: {self.background_rate}")


def _as_nonneg_array(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any(a < 0):
        raise ValueError(f"{name} contains negatives")
    return a


def symptomatic_incidence(new_infections, params: ObservationParams) -> np.ndarray:
    """일일 신규감염 → 증상 발생 I^sym = symptomatic_frac · I^new."""
    params.validate()
    inf = _as_nonneg_array(new_infections, "new_infections")
    return params.symptomatic_frac * inf


def ili_mean(infections_sym, params: ObservationParams) -> np.ndarray:
    """결정적 ILI 평균 μ_t = ρ · I_t^sym (관측 기대값)."""
    params.validate()
    sym = _as_nonneg_array(infections_sym, "infections_sym")
    return params.rho * sym


def sample_ili(infections_sym, params: ObservationParams, rng: np.random.Generator) -> np.ndarray:
    """ILI 신고 Y ~ NegBin(mean=ρ·I^sym, dispersion=φ) 샘플 (재현성: rng 필수).

    NegBin = Gamma-Poisson 혼합: λ ~ Gamma(shape=φ, scale=μ/φ), Y ~ Poisson(λ).
    μ=0 인 날은 0 (degenerate-safe).
    """
    if rng is None:
        raise ValueError("rng (np.random.Generator) 필수 — 재현성")
    mu = ili_mean(infections_sym, params)
    phi = float(params.nb_dispersion)
    out = np.zeros_like(mu)
    pos = mu > 0
    if np.any(pos):
        lam = rng.gamma(shape=phi, scale=mu[pos] / phi)
        out[pos] = rng.poisson(lam).astype(np.float64)
    return out


def sample_positivity(infections_sym, n_tests, params: ObservationParams,
                      rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """바이러스 양성 Pos ~ Binom(n_t, π_t), π = I^sym/(I^sym+B). 반환 (pos, n_tests)."""
    if rng is None:
        raise ValueError("rng 필수")
    params.validate()
    sym = _as_nonneg_array(infections_sym, "infections_sym")
    n = np.asarray(n_tests, dtype=np.int64)
    if n.shape != sym.shape:
        raise ValueError(f"n_tests shape {n.shape} != infections_sym {sym.shape}")
    pi = sym / (sym + params.background_rate + 1e-12)
    pos = rng.binomial(np.maximum(n, 0), np.clip(pi, 0.0, 1.0)).astype(np.float64)
    return pos, n.astype(np.float64)


def negbin_loglik(y_obs, mu, phi: float) -> float:
    """관측 ILI y 의 NegBin(mean=μ, dispersion=φ) 총 로그우도 (보정/평가용).

    ll = Σ [ lgamma(y+φ) − lgamma(φ) − lgamma(y+1)
             + φ·log(φ/(φ+μ)) + y·log(μ/(φ+μ)) ]
    μ=0 & y=0 → 기여 0. μ=0 & y>0 → −inf (불가능 관측).
    """
    y = _as_nonneg_array(y_obs, "y_obs")
    m = _as_nonneg_array(mu, "mu")
    if y.shape != m.shape:
        raise ValueError(f"y_obs {y.shape} != mu {m.shape}")
    if not (phi > 0 and math.isfinite(phi)):
        raise ValueError(f"phi > 0 위반: {phi}")
    ll = 0.0
    for yi, mi in zip(y, m):
        if mi <= 0:
            if yi > 0:
                return float("-inf")
            continue
        ll += (math.lgamma(yi + phi) - math.lgamma(phi) - math.lgamma(yi + 1.0)
               + phi * math.log(phi / (phi + mi))
               + yi * math.log(mi / (phi + mi)))
    return float(ll)


def fit_report_rate(y_obs, infections_sym, params: ObservationParams) -> float:
    """관측 ILI 와 잠재 I^sym 으로 ρ 의 최우도(모먼트) 추정.

    NegBin 평균 μ=ρ·I^sym 에서 ρ̂ = Σy / Σ(symptomatic_frac·I^sym) (가중평균).
    care_seeking 보정에 사용 (관측모형 calibration).
    """
    y = _as_nonneg_array(y_obs, "y_obs")
    sym = _as_nonneg_array(infections_sym, "infections_sym")
    denom = float(np.sum(params.symptomatic_frac * sym))
    if denom <= 0:
        return float("nan")
    return float(np.sum(y) / denom)
