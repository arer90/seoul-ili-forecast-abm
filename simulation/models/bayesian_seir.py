"""
simulation/models/bayesian_seir.py
==================================
베이지안 SEIR 모델: Metropolis-Hastings MCMC 기반 불확실성 정량화

설계 원칙:
 1. BayesianSEIRModel: scipy + numpy만 사용 (가벼운 의존성)
 - Metropolis-Hastings MCMC로 posterior 추정
 - 관찰 오차 모델: Poisson (ILI rate 기반)
 
 2. BayesianSEIRForecaster: BaseForecaster 상속
 - fit에서 BayesianSEIRModel 실행
 - predict에서 posterior 샘플로 앞서 forecast
 - 신용구간(credible interval) 제공
 
 3. SEIR ODE: scipy.integrate.solve_ivp 호환
 - ILI rate(‰) ↔ I compartment 변환
 - 정해진 parameter로 forward simulation

[수학 모델]
 ODE:
 dS/dt = -β·S·I/N
 dE/dt = β·S·I/N - σ·E
 dI/dt = σ·E - γ·I
 dR/dt = γ·(1-cfr)·I
 dD/dt = γ·cfr·I

 ILI rate (‰) = I / N * 1000

 Priors (influenza):
 β ~ LogNormal(μ=ln(0.5), σ=0.3) [WHO range: 0.2~0.8]
 σ ~ Gamma(k=25, θ=0.02) [mean=0.5, ~2일 잠복기]
 γ ~ Gamma(k=16, θ=0.0125) [mean=0.2, ~5일 감염기]
 I₀/N ~ Beta(2, 100) [낮은 초기 감염비율]

 Likelihood (Poisson):
 y_t | I_t ~ Poisson(λ_t)
 λ_t = I_t / N * 1000

 Posterior:
 p(θ|y) ∝ p(y|θ) · p(θ)

변경 이력:
 - (2026-04-10): 초기 구현
 - Metropolis-Hastings MCMC (adaptive step size)
 - Poisson observation model
 - 신용구간 제공
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import logsumexp
from scipy.stats import norm, gamma as gamma_dist, beta as beta_dist

from simulation.models.base import BaseForecaster, ModelMeta, REGISTRY

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# SEIR ODE 함수
# ═══════════════════════════════════════════════════════════════════════════

def _seir_ode(
    t: float,
    y: list,
    beta: float,
    sigma: float,
    gamma: float,
    N: float,
    cfr: float = 0.001,
) -> list:
    """
    SEIR ODE 미분 방정식.

    Parameters
    ----------
    t : float
        시각 (일)
    y : list
        [S, E, I, R, D] 구획 수
    beta : float
        전파율 (1/일)
    sigma : float
        1/잠복기 (1/일)
    gamma : float
        1/감염기간 (1/일)
    N : float
        총 인구
    cfr : float
        사망률 (기본: 0.001)

    Returns
    -------
    list
        [dS/dt, dE/dt, dI/dt, dR/dt, dD/dt]
    """
    S, E, I, R, D = y

    # S + E + I + R + D == N (사망자 포함)
    N_eff = S + E + I + R
    if N_eff <= 0:
        return [0, 0, 0, 0, 0]

    # Force of infection
    foi = beta * S * I / N

    # 미분
    dS = -foi
    dE = foi - sigma * E
    dI = sigma * E - gamma * I
    dR = gamma * (1 - cfr) * I
    dD = gamma * cfr * I

    return [dS, dE, dI, dR, dD]


# ═══════════════════════════════════════════════════════════════════════════
# 베이지안 SEIR 모델 (독립 실행형)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MCMCTrace:
    """MCMC 샘플 추적."""
    beta: np.ndarray          # (n_samples,)
    sigma: np.ndarray
    gamma: np.ndarray
    I0_frac: np.ndarray       # I0 / N
    log_posterior: np.ndarray # (n_samples,)
    acceptance_rate: float


class BayesianSEIRModel:
    """
    베이지안 SEIR 모델 (Metropolis-Hastings MCMC).

    특징:
      - 4개 핵심 parameter의 posterior 추정
      - Poisson observation model
      - Adaptive step size (Gelman & Rubin)
      - Prior: 의학적 문헌 기반
    """

    def __init__(
        self,
        population: float = 9_400_000.0,
        cfr: float = 0.001,
    ):
        """
        Parameters
        ----------
        population : float
            총 인구 (기본: 서울)
        cfr : float
            사망률
        """
        self.N = population
        self.cfr = cfr
        self.trace: Optional[MCMCTrace] = None
        self._fitted = False

    def _prior_log_prob(
        self,
        beta: float,
        sigma: float,
        gamma: float,
        I0_frac: float,
    ) -> float:
        """
        Prior의 로그 확률.

        Parameters
        ----------
        beta : float
            전파율 (1/일)
        sigma : float
            1/잠복기
        gamma : float
            1/감염기간
        I0_frac : float
            초기 감염 비율

        Returns
        -------
        float
            log p(θ)
        """
        # β ~ LogNormal(ln(0.5), 0.3)
        # E[β] = 0.5, SD = 0.3
        if beta <= 0:
            return -np.inf
        mu_beta = np.log(0.5)
        sigma_beta = 0.3
        lp_beta = norm.logpdf(np.log(beta), mu_beta, sigma_beta) - np.log(beta)

        # σ ~ Gamma(k=25, θ=0.02)
        # E[σ] = k·θ = 0.5, SD ≈ 0.1
        if sigma <= 0:
            return -np.inf
        lp_sigma = gamma_dist.logpdf(sigma, a=25, scale=0.02)

        # γ ~ Gamma(k=16, θ=0.0125)
        # E[γ] = 0.2, SD ≈ 0.05
        if gamma <= 0:
            return -np.inf
        lp_gamma = gamma_dist.logpdf(gamma, a=16, scale=0.0125)

        # I₀/N ~ Beta(2, 100)
        if I0_frac <= 0 or I0_frac >= 1:
            return -np.inf
        lp_I0 = beta_dist.logpdf(I0_frac, 2, 100)

        return lp_beta + lp_sigma + lp_gamma + lp_I0

    def _likelihood_log_prob(
        self,
        observed_ili: np.ndarray,
        beta: float,
        sigma: float,
        gamma: float,
        I0_frac: float,
        t_eval: np.ndarray,
    ) -> float:
        """
        Likelihood의 로그 확률 (Poisson).

        Parameters
        ----------
        observed_ili : np.ndarray
            관찰된 ILI rate (‰)
        beta : float
        sigma : float
        gamma : float
        I0_frac : float
        t_eval : np.ndarray
            평가 시점

        Returns
        -------
        float
            log p(y|θ)
        """
        try:
            # 초기 조건
            I0 = self.N * I0_frac
            E0 = I0 * 0.5  # E ~ 0.5*I as typical in early phase
            S0 = self.N - I0 - E0
            R0 = 0
            D0 = 0

            y0 = [S0, E0, I0, R0, D0]

            # ODE 풀이
            sol = solve_ivp(
                _seir_ode,
                [t_eval[0], t_eval[-1]],
                y0,
                t_eval=t_eval,
                args=(beta, sigma, gamma, self.N, self.cfr),
                method="RK45",
                max_step=1.0,
                rtol=1e-6,
                atol=1e-8,
                dense_output=True,
            )

            if not sol.success:
                return -np.inf

            # I(t) → ILI rate(t)
            I_traj = sol.y[2]
            ili_traj = I_traj / self.N * 1000  # ‰로 변환

            # Poisson likelihood
            # 관찰값과 모델값의 길이 맞추기
            min_len = min(len(observed_ili), len(ili_traj))
            obs = observed_ili[:min_len]
            pred = ili_traj[:min_len]

            # ILI rate를 Poisson의 mean으로 사용 (scale 필요 없음)
            # λ_t = pred, y_t = obs
            # log p(y|λ) = Σ y·log(λ) - λ - log(y!)
            pred = np.maximum(pred, 1e-8)  # 수치 안정성
            lp = np.sum(obs * np.log(pred) - pred)

            return lp

        except Exception:
            return -np.inf

    def _metropolis_hastings_step(
        self,
        current_theta: np.ndarray,  # [beta, sigma, gamma, I0_frac]
        current_lp: float,
        observed_ili: np.ndarray,
        step_sizes: np.ndarray,
        t_eval: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, float, bool]:
        """
 MH 반복 한 스텝.

 Parameters
 ----------
 current_theta : np.ndarray
 현재 parameter 값
 current_lp : float
 현재 log posterior
 observed_ili : np.ndarray
 step_sizes : np.ndarray
 proposal 분포의 표준편차
 t_eval : np.ndarray
 rng : np.random.Generator
 : per-instance RNG — np.random global state 오염 방지

 Returns
 -------
 tuple
 (new_theta, new_lp, accepted)
 """
        # Proposal: Random walk Metropolis
        proposal = current_theta + rng.normal(0, step_sizes)

        beta_p, sigma_p, gamma_p, I0_frac_p = proposal

        # Prior + Likelihood
        lp_prior = self._prior_log_prob(beta_p, sigma_p, gamma_p, I0_frac_p)
        if lp_prior == -np.inf:
            return current_theta, current_lp, False

        lp_lik = self._likelihood_log_prob(
            observed_ili, beta_p, sigma_p, gamma_p, I0_frac_p, t_eval
        )
        if lp_lik == -np.inf:
            return current_theta, current_lp, False

        new_lp = lp_prior + lp_lik

        # MH acceptance ratio
        log_alpha = new_lp - current_lp
        if np.log(rng.random()) < log_alpha:
            return proposal, new_lp, True
        else:
            return current_theta, current_lp, False

    def fit(
        self,
        observed_ili: np.ndarray,
        population: Optional[float] = None,
        n_samples: int = 2000,
        burn_in: int = 1000,  # BUG-C fix: adaptive step 수렴 시간 확보
        t_eval: Optional[np.ndarray] = None,
        seed: int = 42,
    ) -> MCMCTrace:
        """
        MCMC로 posterior 추정.

        Parameters
        ----------
        observed_ili : np.ndarray
            관찰된 ILI rate 시계열 (‰)
        population : float, optional
            인구 (설정되면 업데이트)
        n_samples : int
            MCMC 샘플 수
        burn_in : int
            burn-in 기간
        t_eval : np.ndarray, optional
            평가 시점 (기본: 0부터 len(observed_ili)-1)
        seed : int
            난수 시드

        Returns
        -------
        MCMCTrace
            posterior 샘플
        """
        # : per-instance RNG → global np.random state 오염 제거.
        # seed 재현성은 fit-level 로 충분 (MCMC 체인이 동일 시드에서 결정적).
        rng = np.random.default_rng(seed)
        self._fit_seed = seed  # predict 에서 기본값으로 재사용 가능

        if population is not None:
            self.N = population

        if t_eval is None:
            t_eval = np.arange(len(observed_ili), dtype=float)

        log.info(
            f"[BayesianSEIR] MCMC 시작: n_samples={n_samples}, "
            f"burn_in={burn_in}, population={self.N:,.0f}"
        )

        # 초기화 (prior에서 샘플) — 동일 rng 로 일관 (seed 재설정 불필요)
        beta_init = rng.lognormal(np.log(0.5), 0.3)
        sigma_init = rng.gamma(25, 0.02)
        gamma_init = rng.gamma(16, 0.0125)
        I0_frac_init = rng.beta(2, 100)

        current_theta = np.array([beta_init, sigma_init, gamma_init, I0_frac_init])

        # 초기 log posterior
        lp_prior = self._prior_log_prob(
            beta_init, sigma_init, gamma_init, I0_frac_init
        )
        lp_lik = self._likelihood_log_prob(
            observed_ili,
            beta_init,
            sigma_init,
            gamma_init,
            I0_frac_init,
            t_eval,
        )
        current_lp = lp_prior + lp_lik

        # Step size 초기화
        step_sizes = np.array([0.1, 0.01, 0.01, 0.01])

        # BUG-C fix: Robbins-Monro adaptive.
        #   이전엔 burn-in 500 + 100-step bucket 평균 acceptance 만 이용해
        #   ±5% 만 조정했다. 4D chain 에서 acceptance 0.004 였던 이전 run 처럼
        #   proposal 이 너무 크면 5 번의 ±5% 로는 거의 줄지 않는다.
        #   여기선 매 스텝 log-step 을 target_accept 기준으로 업데이트.
        target_acceptance = 0.3   # 4D 체인에서 Roberts(1997) 권장 근사값
        adapt_gamma0 = 0.4        # 초기 학습률 (log-step 업데이트)
        adapt_kappa = 0.75        # decay rate (Andrieu-Thoms 2008)
        window = 50               # 평균 내는 최근 스텝

        # MCMC 루프
        total_samples = n_samples + burn_in
        samples_beta = []
        samples_sigma = []
        samples_gamma = []
        samples_I0 = []
        samples_lp = []
        n_accepted = 0
        recent_accepts: list[int] = []

        for iteration in range(total_samples):
            # MH 스텝 — 동일 rng 를 매 step 에 전달 (체인 내부 결정성)
            new_theta, new_lp, accepted = self._metropolis_hastings_step(
                current_theta, current_lp, observed_ili, step_sizes, t_eval, rng
            )

            if accepted:
                current_theta = new_theta
                current_lp = new_lp
                n_accepted += 1
            recent_accepts.append(1 if accepted else 0)
            if len(recent_accepts) > window:
                recent_accepts.pop(0)

            # Burn-in 이후 저장
            if iteration >= burn_in:
                samples_beta.append(current_theta[0])
                samples_sigma.append(current_theta[1])
                samples_gamma.append(current_theta[2])
                samples_I0.append(current_theta[3])
                samples_lp.append(current_lp)

            # BUG-C fix: Robbins-Monro log-step update.
            #   매 스텝 마다 수행하되, decaying learning rate.
            #   burn-in 구간에서만 adapt, 이후엔 frozen.
            if iteration < burn_in and len(recent_accepts) >= 20:
                lr = adapt_gamma0 / max(1.0, (iteration + 1) ** adapt_kappa)
                recent_acc = float(np.mean(recent_accepts))
                # log-step 증감: accept↑ → step↑, accept↓ → step↓
                log_adj = lr * (recent_acc - target_acceptance)
                step_sizes *= float(np.exp(log_adj))
                # 안전 clip (step 이 0 이나 ∞ 로 가지 않도록)
                step_sizes = np.clip(step_sizes, 1e-5, 2.0)

            if (iteration + 1) % 500 == 0:
                log.debug(
                    f"[BayesianSEIR] Iteration {iteration+1}/{total_samples}, "
                    f"acceptance rate: {n_accepted/(iteration+1):.3f}"
                )

        # 최종 acceptance rate
        acceptance_rate = n_accepted / total_samples

        # Trace 구성
        self.trace = MCMCTrace(
            beta=np.array(samples_beta),
            sigma=np.array(samples_sigma),
            gamma=np.array(samples_gamma),
            I0_frac=np.array(samples_I0),
            log_posterior=np.array(samples_lp),
            acceptance_rate=acceptance_rate,
        )

        self._fitted = True
        log.info(
            f"[BayesianSEIR] MCMC 완료: "
            f"acceptance_rate={acceptance_rate:.3f}, "
            f"E[β]={self.trace.beta.mean():.4f}, "
            f"E[γ]={self.trace.gamma.mean():.4f}"
        )

        return self.trace

    def predict(
        self,
        steps: int,
        n_trajectories: int = 100,
        credible_level: float = 0.95,
        seed: Optional[int] = None,
    ) -> dict:
        """
        Posterior에서 샘플링하여 forward simulation.

        Parameters
        ----------
        steps : int
            예측 기간 (일)
        n_trajectories : int
            샘플 궤적 수
        credible_level : float
            신용구간 수준 (기본: 95%)

        Returns
        -------
        dict
            - trajectories : (n_trajectories, steps) array of ILI rate
            - median : 중위수 궤적
            - credible_lower : 신용구간 하한
            - credible_upper : 신용구간 상한
        """
        if not self._fitted or self.trace is None:
            raise RuntimeError("모델을 먼저 fit()해야 합니다.")

        # Posterior에서 샘플링 (무복원) — : per-call RNG.
        # seed=None 이면 fit 시 저장한 _fit_seed 재사용, 그것도 없으면 fresh.
        if seed is None:
            seed = getattr(self, "_fit_seed", None)
        rng_pred = np.random.default_rng(seed)
        n_posterior = len(self.trace.beta)
        indices = rng_pred.choice(n_posterior, size=n_trajectories, replace=True)

        trajectories = []
        t_eval = np.arange(steps, dtype=float)

        for idx in indices:
            beta = self.trace.beta[idx]
            sigma = self.trace.sigma[idx]
            gamma = self.trace.gamma[idx]
            I0_frac = self.trace.I0_frac[idx]

            try:
                # 초기 조건
                I0 = self.N * I0_frac
                E0 = I0 * 0.5
                S0 = self.N - I0 - E0
                R0 = 0
                D0 = 0

                y0 = [S0, E0, I0, R0, D0]

                # ODE 풀이
                sol = solve_ivp(
                    _seir_ode,
                    [t_eval[0], t_eval[-1]],
                    y0,
                    t_eval=t_eval,
                    args=(beta, sigma, gamma, self.N, self.cfr),
                    method="RK45",
                    max_step=1.0,
                    rtol=1e-6,
                    atol=1e-8,
                )

                if sol.success:
                    I_traj = sol.y[2]
                    ili_traj = I_traj / self.N * 1000
                    trajectories.append(ili_traj)
            except Exception:
                continue

        trajectories = np.array(trajectories)

        # 통계량 계산
        alpha = 1 - credible_level
        lower_pct = (alpha / 2) * 100
        upper_pct = (1 - alpha / 2) * 100

        result = {
            "trajectories": trajectories,
            "median": np.median(trajectories, axis=0),
            "mean": np.mean(trajectories, axis=0),
            "credible_lower": np.percentile(trajectories, lower_pct, axis=0),
            "credible_upper": np.percentile(trajectories, upper_pct, axis=0),
            "std": np.std(trajectories, axis=0),
        }

        return result


# ═══════════════════════════════════════════════════════════════════════════
# BayesianSEIRForecaster (BaseForecaster 상속)
# ═══════════════════════════════════════════════════════════════════════════

class BayesianSEIRForecaster(BaseForecaster):
    """
    베이지안 SEIR 기반 ILI 예측 모델 (BaseForecaster).

    Features:
      - ILI 시계열로부터 SEIR parameter posterior 추정
      - Uncertainty quantification (신용구간)
      - 예측 시에 median trajectory 반환
    """

    meta = ModelMeta(
        name="Bayesian-SEIR",
        category="physics",
        level=16,
        min_data=30,
        description="베이지안 SEIR (MCMC 불확실성 정량화)",
        requires_gpu=False,
        dependencies=[],
    )

    def __init__(self):
        super().__init__()
        self._bayesian_model: Optional[BayesianSEIRModel] = None
        self._posterior_intervals: Optional[tuple] = None
        self._population = 9_400_000.0

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        population: float = 9_400_000.0,
        n_samples: int = 2000,
        burn_in: int = 1000,  # BUG-C fix: adaptive step 수렴 시간 확보
        **kwargs,
    ) -> "BayesianSEIRForecaster":
        """
        SEIR parameter를 y_train (ILI rate)에 맞춤.

        Parameters
        ----------
        X_train : np.ndarray
            (n_samples, n_features) 특성 행렬
            (현재 미사용 -- y_train만 사용)
        y_train : np.ndarray
            (n_samples,) ILI rate 시계열 (‰)
        population : float
            인구 (기본: 9.4M)
        n_samples : int
            MCMC 샘플 수
        burn_in : int
            Burn-in 기간

        Returns
        -------
        self
        """
        self._population = population
        self._bayesian_model = BayesianSEIRModel(population=population)

        # y_train의 길이에 맞춰 t_eval 설정
        t_eval = np.arange(len(y_train), dtype=float)

        # MCMC 실행
        trace = self._bayesian_model.fit(
            y_train,
            population=population,
            n_samples=n_samples,
            burn_in=burn_in,
            t_eval=t_eval,
        )

        self._fitted = True
        log.info(f"[BayesianSEIRForecaster] fit 완료 (n={len(y_train)})")

        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        """
        향후 ILI rate 예측 (median trajectory).

        Parameters
        ----------
        X_test : np.ndarray
            (n_steps, n_features) 특성 행렬
            (길이만 사용됨)

        Returns
        -------
        np.ndarray
            (n_steps,) 예측된 ILI rate (‰)
        """
        if not self._fitted or self._bayesian_model is None:
            raise RuntimeError("모델을 먼저 fit()해야 합니다.")

        steps = len(X_test)

        # Forward simulation
        pred_dict = self._bayesian_model.predict(steps, n_trajectories=100)

        # 신용구간 저장
        self._posterior_intervals = (
            pred_dict["credible_lower"],
            pred_dict["credible_upper"],
        )

        # Median 반환
        return np.maximum(pred_dict["median"], 0)

    def get_prediction_intervals(self) -> tuple[np.ndarray, np.ndarray]:
        """
        신용구간 반환 (predict() 호출 후).

        Returns
        -------
        tuple
            (lower_bound, upper_bound) 모두 (n_steps,)
        """
        if self._posterior_intervals is None:
            raise RuntimeError(
                "predict()를 먼저 호출하여 신용구간을 계산해야 합니다."
            )
        return self._posterior_intervals


# ═══════════════════════════════════════════════════════════════════════════
# 모델 등록
# ═══════════════════════════════════════════════════════════════════════════

# (2026-04-19): forecasting REGISTRY 에서 제외.
#   근거: smoke_seir_salvage (n_tr=234, n_te=12)
#     default          → R² = -8.23
#     burn_in=3000/n=2000 → R² = -7.37
#   구조적 한계: (a) X_train 피처 완전 무시, (b) MCMC acceptance=0.006 으로 stuck,
#   (c) end-of-train 상태에서 ODE forward → 계절성 ILI 와 궤적 불일치.
#   시뮬레이션 클래스 (BayesianSEIRModel) 는 그대로 유지 — import 는 가능.
# REGISTRY.register(BayesianSEIRForecaster)

log.debug("[bayesian_seir] BayesianSEIRForecaster 는 forecasting REGISTRY 에서 격리됨 (시뮬용)")
