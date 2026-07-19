"""
simulation/models/seir_forced.py
====================================
SEIR V2 — deterministic SEIR with time-varying β(t) + NPI covariate.

V1 (BayesianSEIRForecaster / MetapopSEIRForecaster) 은 상수 β 로 MCMC fit 하여
peak undershoot 심각 (peak_ratio 0.01~0.09). V2 는:

  β(t) = β₀ · (1 + ε·cos(2π(t - φ)/52)) · (1 - κ·NPI(t))

  - ε, φ: 계절성 진폭 / 위상 (학교/기후 forcing)
  - κ·NPI(t): COVID NPI 감소 효과. NPI(t)=1 if 2020-03 ≤ t ≤ 2022-12 else 0.
  - ODE 는 SEIR dS/dt = -β(t)·S·I/N, dE/dt = β(t)·S·I/N - σE, dI/dt = σE - γI, dR/dt = γI.

Fit: scipy.optimize.minimize(SSE in ILI rate space) — 7 parameters (β₀, ε, φ, σ, γ, κ, I₀).

역학적 해석:
  - V1 peak_ratio ~ 0.05 (structural underfit), V2 목표 peak_ratio ∈ [0.6, 1.2].
  - COVID 구간 NPI 효과 분리 → post-COVID rebound 구간 fit 개선 기대.
  - MCMC 생략 (V1 과 contrast), point-estimate fit → faster convergence + 해석 용이.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from simulation.models.base import BaseForecaster, ModelMeta, REGISTRY

log = logging.getLogger(__name__)

# COVID NPI 구간: 2020-03-01 ~ 2022-12-31 (ISO 주 기반 — R3)
# simulation-advisor 권고: week index 하드코딩 금지. 데이터 시작 주(`data_start_iso`)를
# 받아 week index 로 변환. 기본 fallback 은 서울 ILI 2018-01 시작 기준.
_NPI_START_ISO = "2020-03-02"  # ISO Monday of 2020-W10
_NPI_END_ISO = "2022-12-26"    # ISO Monday of 2022-W52
_DEFAULT_DATA_START_ISO = "2018-01-01"  # Seoul ILI 시작 주 (2018-W01)
# fallback index (data_start_iso 를 주입 못 받은 경우, 서울 ILI 2018-01 시작 가정)
_NPI_START_WEEK_IDX_FALLBACK = 114  # 2018-01-01 ~ 2020-03-02 ≈ 114주
_NPI_END_WEEK_IDX_FALLBACK = 260    # 2018-01-01 ~ 2022-12-26 ≈ 260주


def _seir_ode(t, y, beta_fn, sigma, gamma, N):
    S, E, I, R = y
    beta = beta_fn(t)
    dS = -beta * S * I / N
    dE = beta * S * I / N - sigma * E
    dI = sigma * E - gamma * I
    dR = gamma * I
    return [dS, dE, dI, dR]


@dataclass
class SEIRForcedParams:
    beta0: float = 0.45        # baseline transmission
    epsilon: float = 0.20      # seasonal amplitude
    phi: float = 2.0           # seasonal phase (weeks, 0..52)
    sigma: float = 0.5         # latency rate (2d)
    gamma: float = 0.2         # recovery rate (5d)
    kappa: float = 0.35        # NPI reduction factor
    I0_frac: float = 0.001     # initial I / N


class SEIRForcedForecaster(BaseForecaster):
    """Deterministic SEIR with time-varying β + NPI covariate."""

    meta = ModelMeta(
        name="SEIR-V2-Forced",
        category="physics",
        level=17,
        min_data=60,
        description="SEIR V2 (time-varying β + seasonal forcing + COVID NPI covariate, OLS fit).",
        requires_gpu=False,
        dependencies=[],
    )

    def __init__(
        self,
        population: float = 9_400_000.0,
        rate_scale: float = 1000.0,
        data_start_iso: str = _DEFAULT_DATA_START_ISO,
    ):
        super().__init__()
        self._population = population
        self._rate_scale = rate_scale  # ILI rate 단위 (‰)
        self._params: Optional[SEIRForcedParams] = None
        self._last_state: Optional[np.ndarray] = None  # [S, E, I, R] at fit end
        self._train_len = 0
        # NPI window: ISO-date 기반 week index 계산
        self._npi_start_idx, self._npi_end_idx = self._compute_npi_indices(
            data_start_iso
        )

    @staticmethod
    def _compute_npi_indices(data_start_iso: str) -> Tuple[int, int]:
        """데이터 시작 ISO 일 기준으로 NPI window 의 week index 를 계산.

        실패 시 fallback 에 기록된 (114, 260) 사용 (서울 ILI 2018-01 시작 기준).
        """
        try:
            from datetime import date
            s = date.fromisoformat(data_start_iso)
            ns = date.fromisoformat(_NPI_START_ISO)
            ne = date.fromisoformat(_NPI_END_ISO)
            start_idx = max(0, (ns - s).days // 7)
            end_idx = max(start_idx, (ne - s).days // 7)
            return (start_idx, end_idx)
        except Exception as _e:
            log.warning(
                f"  [SEIR-V2-Forced] NPI ISO parse 실패 ({_e}) → fallback "
                f"({_NPI_START_WEEK_IDX_FALLBACK}, {_NPI_END_WEEK_IDX_FALLBACK})"
            )
            return (_NPI_START_WEEK_IDX_FALLBACK, _NPI_END_WEEK_IDX_FALLBACK)

    def _npi_at(self, t_idx: float) -> float:
        """Return 1.0 if t_idx in COVID NPI window else 0.0."""
        return 1.0 if (self._npi_start_idx <= t_idx <= self._npi_end_idx) else 0.0

    def _beta_fn(self, params: SEIRForcedParams):
        def _b(t):
            seasonal = 1.0 + params.epsilon * np.cos(2.0 * np.pi * (t - params.phi) / 52.0)
            npi_adj = 1.0 - params.kappa * self._npi_at(t)
            return max(params.beta0 * seasonal * npi_adj, 1e-4)
        return _b

    def _simulate(self, params: SEIRForcedParams, t_start: float, t_end: float,
                  init_state: Optional[np.ndarray] = None) -> np.ndarray:
        N = self._population
        if init_state is None:
            I0 = max(params.I0_frac * N, 1.0)
            E0 = I0  # heuristic
            S0 = N - I0 - E0
            R0 = 0.0
            y0 = [S0, E0, I0, R0]
        else:
            y0 = init_state.tolist()
        t_eval = np.arange(t_start, t_end + 1.0)
        # R3: max_step=0.1 (sub-daily). σ≤1.0 (latency ~1d) 이면 1주 step 은
        # E→I 전이 놓침 → 0.1주 (≈17시간) 로 축소.
        sol = solve_ivp(
            _seir_ode, (t_start, t_end + 0.1), y0,
            args=(self._beta_fn(params), params.sigma, params.gamma, N),
            t_eval=t_eval, method="RK45", max_step=0.1, rtol=1e-4, atol=1e-3,
        )
        if not sol.success:
            return np.full(len(t_eval), np.nan)
        return sol.y  # (4, T)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> "SEIRForcedForecaster":
        self._train_len = len(y_train)
        y_scaled = np.clip(y_train, 0.0, None).astype(float)

        def _obj(theta: np.ndarray) -> float:
            p = SEIRForcedParams(
                beta0=float(theta[0]),
                epsilon=float(theta[1]),
                phi=float(theta[2]),
                sigma=float(theta[3]),
                gamma=float(theta[4]),
                kappa=float(theta[5]),
                I0_frac=float(theta[6]),
            )
            sim = self._simulate(p, 0.0, float(self._train_len - 1))
            if np.any(np.isnan(sim)):
                return 1e12
            I_t = sim[2]
            pred = I_t / self._population * self._rate_scale
            # R3: weighted SSE — 70% log (peak-sensitive) + 30% linear
            # (off-peak sensitivity). simulation-advisor: log-only 는 ε 를
            # upper bound 로 밀어 winter peak 만 맞추고 summer trough overshoot.
            sse_log = np.sum((np.log1p(pred) - np.log1p(y_scaled)) ** 2)
            sse_lin = np.sum((pred - y_scaled) ** 2)
            # linear SSE 는 스케일이 훨씬 크므로 y 의 분산으로 정규화
            y_var = max(float(np.var(y_scaled)), 1e-6)
            return float(0.7 * sse_log + 0.3 * sse_lin / y_var)

        # R3: 임상 표준 + epi-validity gate 기반 bound 축소
        # (clinical-advisor + simulation-advisor 권고)
        x0 = np.array([0.45, 0.20, 2.0, 0.5, 0.2, 0.35, 0.001])
        bounds = [
            (0.1, 1.5),     # beta0
            (0.0, 0.35),    # epsilon — Yang 2015 JID, 온대 influenza 20-30%
            (0.0, 52.0),    # phi
            (0.25, 1.0),    # sigma (1~4d latent) — Lessler 2009
            (0.10, 0.5),    # gamma (2~10d infectious) — Ip 2017 CID; 20d 비현실적
            (0.0, 0.6),     # kappa (0~60% NPI) — Flaxman 2020 Nature, KR 2단계 β×0.4-0.6
            (1e-5, 0.01),   # I0_frac
        ]
        try:
            res = minimize(_obj, x0, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 200, "ftol": 1e-6})
            self._params = SEIRForcedParams(
                beta0=res.x[0], epsilon=res.x[1], phi=res.x[2],
                sigma=res.x[3], gamma=res.x[4], kappa=res.x[5], I0_frac=res.x[6],
            )
            # 마지막 상태 저장 (predict 시 이어서)
            sim = self._simulate(self._params, 0.0, float(self._train_len - 1))
            if not np.any(np.isnan(sim)):
                self._last_state = sim[:, -1]
            log.info(
                f"  [SEIR-V2-Forced] fit OK: β₀={self._params.beta0:.3f}, "
                f"ε={self._params.epsilon:.3f}, φ={self._params.phi:.2f}, "
                f"σ={self._params.sigma:.3f}, γ={self._params.gamma:.3f}, "
                f"κ={self._params.kappa:.3f}, I₀={self._params.I0_frac:.5f}, "
                f"loss={float(res.fun):.3f}, iter={res.nit}"
            )
        except Exception as e:
            log.warning(f"  [SEIR-V2-Forced] fit 실패: {e} → 기본 파라미터")
            self._params = SEIRForcedParams()

        self._fitted = True
        return self

    def rt_effective_trajectory(
        self, t_start: float = 0.0, t_end: Optional[float] = None
    ) -> Optional[np.ndarray]:
        """Rt_eff(t) = β(t)/γ · (S(t)/N) 궤적.

 R3 — simulation-advisor: epi-validity gate 와 EpiEstim overlay 용.
 `self._last_state` 가 있으면 train 끝 상태를 초기값으로, 없으면 전형적
 초기값으로 재시뮬. 반환 shape: (T) 또는 fit 전이면 None.
 """
        if self._params is None:
            return None
        if t_end is None:
            t_end = float(self._train_len - 1) if self._train_len > 0 else 100.0
        sim = self._simulate(self._params, t_start, t_end)
        if np.any(np.isnan(sim)):
            return None
        S, E, I, R = sim
        beta = np.array([self._beta_fn(self._params)(t) for t in np.arange(t_start, t_end + 1)])
        rt = beta / max(self._params.gamma, 1e-6) * (S / self._population)
        return rt

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        if not self._fitted or self._params is None:
            raise RuntimeError("SEIR-V2-Forced: fit() 먼저 호출 필수")
        n_steps = len(X_test)
        t_start = float(self._train_len)
        t_end = float(self._train_len + n_steps - 1)
        sim = self._simulate(self._params, t_start, t_end, init_state=self._last_state)
        if np.any(np.isnan(sim)):
            log.warning("  [SEIR-V2-Forced] predict 수렴 실패 → NaN")
            return np.full(n_steps, float(np.mean(sim[2][~np.isnan(sim[2])]) if np.any(~np.isnan(sim[2])) else 0.0))
        I_t = sim[2][:n_steps]
        pred = I_t / self._population * self._rate_scale
        return np.clip(pred, 0.0, None).astype(np.float32)


# 2026-05-26 prune (Codex + user): SEIR-V2-Forced REMOVED.
# Catastrophic R²=−1.09 documented as Tier T2 "F11 honest negative".
# Class kept for paper §F11 reference; not registered.
# try:
#     REGISTRY.register(SEIRForcedForecaster)
#     log.info("[seir_forced] SEIRForcedForecaster 등록됨")
# except Exception as _e:
#     log.debug(f"[seir_forced] 등록 skip: {_e}")
log.info("[seir_forced] SEIR-V2-Forced 등록 SKIP (2026-05-26 prune)")
