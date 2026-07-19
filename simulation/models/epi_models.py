"""
simulation/models/epi_models.py
================================
보건역학·의학통계 기반 예측 모델.

ML 회귀와 달리 역학적 메커니즘, 베이지안 불확실성, 감염병 분포 특성을 반영.
모든 모델은 BaseForecaster를 상속하며 fit/predict 인터페이스 준수.

모델 목록:
  1. GaussianProcessForecaster      — GP 회귀 (RBF + Periodic 커널)
  2. BayesianRidgeForecaster         — 베이지안 릿지 (자동 정규화 + 사후분포)
  3. NegBinGLMForecaster             — 음이항 GLM (과분산 감염병 데이터)
  4. GAMForecaster                   — 일반화 가법 모형 (비선형 공변량 효과)
  5. BayesianMCMCForecaster          — MCMC 사후분포 샘플링 (MH 알고리즘)
  6. PoissonAutoregForecaster        — 포아송 자기회귀 (Endemic-Epidemic 분해)

참고문헌:
  - Held & Paul (2012): Endemic-Epidemic model (surveillance package)
  - Rasmussen & Williams (2006): Gaussian Processes for Machine Learning
  - Hilbe (2011): Negative Binomial Regression
  - Wood (2017): Generalized Additive Models
  - Gelman et al. (2013): Bayesian Data Analysis
"""

from __future__ import annotations

import gc
import logging
import warnings
from typing import Optional

import numpy as np

from simulation.models.base import BaseForecaster, ModelMeta, REGISTRY
from simulation.config_global import GLOBAL, Z95  # SSOT (2026-05-28)

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


# ═══════════════════════════════════════════════════════════════
# 1. Gaussian Process Regression
# ═══════════════════════════════════════════════════════════════

class GaussianProcessForecaster(BaseForecaster):
    """가우시안 프로세스 회귀 — 베이지안 비모수 모델.

    커널: RBF(전반적 추세) + Periodic(계절성 52주) + WhiteKernel(노이즈)
    장점: 예측 불확실성(σ) 동시 추정, 소표본에서도 안정적
    역학적 의의: CDC FluSight 대회에서 GP 기반 모델 상위 입상 (Ray et al., 2017)
    """

    meta = ModelMeta(
        name="GP-RBF-Periodic",
        category="epi",
        level=8,
        min_data=60,
        description="Gaussian Process (RBF+Periodic kernel). 베이지안 불확실성 추정.",
        dependencies=["sklearn"],
    )

    def __init__(self):
        super().__init__()
        self._model = None
        self._scaler_X = None
        self._scaler_y = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> "GaussianProcessForecaster":
        import gc
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import (
            RBF, WhiteKernel, ExpSineSquared, ConstantKernel
        )
        from sklearn.metrics import r2_score
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        # : RBF kernel 은 high-D 에서 curse of dimensionality → PCA 로
        # hard cap 10-12 components 로 공격적으로 축소.
        n_train, p_orig = X_train.shape
        n_comp_cap = min(p_orig, 12, max(5, n_train // 20))
        self._pca = None
        if p_orig > n_comp_cap:
            self._pca = PCA(n_components=n_comp_cap, random_state=42)
            X_reduced = self._pca.fit_transform(X_train)
            log.info(f"  [GP-RBF-Periodic] PCA 차원 축소: {p_orig} → {n_comp_cap}")
        else:
            X_reduced = X_train

        # Sprint 1.5 R5 (2026-05-26): use shared setup_xy_scalers helper
        from simulation.models.base import setup_xy_scalers
        self._scaler_X, self._scaler_y, X_s, y_s = setup_xy_scalers(X_reduced, y_train)

        def _build_kernel(ls_rbf, ls_per, period, noise):
            return (
                ConstantKernel(1.0, (0.01, 100.0))
                * RBF(length_scale=ls_rbf, length_scale_bounds=(0.5, 100.0))
                + ConstantKernel(0.5, (0.01, 10.0))
                * ExpSineSquared(
                    length_scale=ls_per, periodicity=period,
                    length_scale_bounds=(0.1, 20.0),
                    periodicity_bounds=(0.3, 5.0),
                )
                + WhiteKernel(noise_level=noise, noise_level_bounds=(1e-3, 5.0))
            )

        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            use_optuna = True
        except ImportError:
            use_optuna = False

        tscv = TimeSeriesSplit(n_splits=3)

        best_params = {
            "ls_rbf": 5.0, "ls_per": 1.0, "period": 1.0,
            "noise": 0.5, "alpha": 1e-2,
        }
        best_score = -np.inf

        if use_optuna:
            # B-7: Optuna TPE — kernel hyperprior + alpha jointly.
            def objective(trial):
                params = {
                    "ls_rbf": trial.suggest_float("ls_rbf", 0.5, 100.0, log=True),
                    "ls_per": trial.suggest_float("ls_per", 0.1, 20.0, log=True),
                    "period": trial.suggest_float("period", 0.3, 5.0),
                    "noise": trial.suggest_float("noise", 1e-3, 5.0, log=True),
                    "alpha": trial.suggest_float("alpha", 1e-4, 1.0, log=True),
                }
                fold_scores = []
                for i, (train_idx, val_idx) in enumerate(tscv.split(X_s)):
                    X_tr, X_val = X_s[train_idx], X_s[val_idx]
                    y_tr, y_val = y_s[train_idx], y_s[val_idx]
                    try:
                        m = GaussianProcessRegressor(
                            kernel=_build_kernel(
                                params["ls_rbf"], params["ls_per"],
                                params["period"], params["noise"],
                            ),
                            n_restarts_optimizer=2,
                            alpha=params["alpha"],
                            normalize_y=False,
                            random_state=42,
                        )
                        m.fit(X_tr, y_tr)
                        pred = m.predict(X_val)
                        fold_scores.append(r2_score(y_val, pred))
                        del m
                    except optuna.TrialPruned:
                        # Cat 1 (Codex 발견, 2026-05-12): broad except 가 TrialPruned 흡수 X
                        raise
                    except Exception as _ex:
                        # G-159 (2026-05-02): silent sentinel -1.0 → -inf 명시 + warning.
                        # -1.0 은 정상 R² 값과 구분 안 됨 (R² 가 -1 까지 갈 수 있음).
                        # -inf + warning 으로 catastrophic 와 borderline 구분.
                        log.warning(f"  [GP-RBF-Periodic] CV fold fail: "
                                    f"{type(_ex).__name__}: {_ex}")
                        fold_scores.append(float("-inf"))
                    # Cat 1 (Codex/ANO pattern, 2026-05-12): fold-level pruning.
                    _mean_r2 = float(np.mean(fold_scores))
                    if np.isfinite(_mean_r2):
                        trial.report(_mean_r2, i)
                        _r2_cutoff = GLOBAL.filter.r2_catastrophic_cutoff
                        if i == 0 and _mean_r2 < _r2_cutoff:
                            raise optuna.TrialPruned(
                                f"GP-RBF-Periodic fold 0 catastrophic R²={_mean_r2:.3f} < {_r2_cutoff}")
                        if trial.should_prune():
                            raise optuna.TrialPruned(
                                f"GP-RBF-Periodic pruned at fold {i} (R²={_mean_r2:.3f})")
                gc.collect()
                return float(np.mean(fold_scores))

            from simulation.models._optuna_budget import get_trials as _get_trials
            from simulation.models._optuna_pruners import get_best_pruner_for as _get_pruner
            _n_trials = _get_trials("GP-RBF-Periodic", default=20)
            study = optuna.create_study(direction="maximize",
                                        sampler=optuna.samplers.TPESampler(seed=42),
                                        pruner=_get_pruner("GP-RBF-Periodic"))
            # G-161 (2026-05-02): trial cleanup callback + gc_after_trial 강제.
            from simulation.models._optuna_torch import make_trial_cleanup_callback
            study.optimize(objective, n_trials=_n_trials,
                           callbacks=[make_trial_cleanup_callback("GP-RBF-Periodic")],
                           gc_after_trial=True, show_progress_bar=False)
            best_params = study.best_params
            best_score = study.best_value
            del study
            gc.collect()

        try:
            self._model = GaussianProcessRegressor(
                kernel=_build_kernel(
                    best_params["ls_rbf"], best_params["ls_per"],
                    best_params["period"], best_params["noise"],
                ),
                n_restarts_optimizer=5,
                alpha=best_params["alpha"],
                normalize_y=False,
                random_state=42,
            )
            self._model.fit(X_s, y_s)
        except Exception as e:
            log.warning(f"  [GP-RBF-Periodic] 최종 학습 실패: {e} → 단순 RBF 폴백")
            kernel_simple = (
                ConstantKernel(1.0, (0.01, 100.0))
                * RBF(length_scale=5.0, length_scale_bounds=(0.5, 100.0))
                + WhiteKernel(noise_level=0.5, noise_level_bounds=(1e-3, 5.0))
            )
            self._model = GaussianProcessRegressor(
                kernel=kernel_simple,
                n_restarts_optimizer=5,
                alpha=1e-1,
                normalize_y=False,
                random_state=42,
            )
            self._model.fit(X_s, y_s)

        self._fitted = True
        log.info(f"  [GP-RBF-Periodic] best_params={best_params}, CV R²={best_score:.4f}")
        log.info(f"  [GP-RBF-Periodic] kernel: {self._model.kernel_}")
        log.info(f"  [GP-RBF-Periodic] log-marginal-likelihood: "
                 f"{self._model.log_marginal_likelihood_value_:.2f}")
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        X_reduced = self._pca.transform(X_test) if self._pca is not None else X_test
        X_s = self._scaler_X.transform(X_reduced)
        pred_s, std_s = self._model.predict(X_s, return_std=True)

        pred = self._scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel()
        # 95% 예측구간 (PI) 도 로깅
        y_scale = self._scaler_y.scale_[0]
        ci_width = Z95 * std_s * y_scale
        log.info(f"  [GP-RBF-Periodic] 평균 95% CI 폭: {ci_width.mean():.2f}")

        return np.maximum(pred, 0)


# ═══════════════════════════════════════════════════════════════
# 2. Bayesian Ridge Regression
# ═══════════════════════════════════════════════════════════════

class BayesianRidgeForecaster(BaseForecaster):
    """베이지안 릿지 회귀 — 자동 정규화 + 사후분포 추정.

    사전분포: 가중치 w ~ N(0, α⁻¹I), 노이즈 ~ N(0, λ⁻¹I)
    α, λ를 타입-II 최대우도(Evidence Approximation)로 자동 추정
    장점: 과적합 자동 방지, 예측 불확실성 제공, 매우 빠름
    역학적 의의: 감염병 예측에서 BMA(Bayesian Model Averaging) 기반 앙상블의 구성요소
    """

    meta = ModelMeta(
        name="BayesianRidge",
        category="epi",
        level=3,
        min_data=40,
        description="Bayesian Ridge: 자동 정규화 + Evidence Approximation.",
        dependencies=["sklearn"],
    )

    def __init__(self):
        super().__init__()
        self._model = None
        self._scaler_X = None
        self._scaler_y = None
        self._y_train_range = None  # 0-B: for CI extrapolation cap

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> "BayesianRidgeForecaster":
        from sklearn.linear_model import BayesianRidge
        from sklearn.preprocessing import StandardScaler

        # Sprint 1.5 R5 (2026-05-26): use shared setup_xy_scalers helper
        from simulation.models.base import setup_xy_scalers
        self._scaler_X, self._scaler_y, X_s, y_s = setup_xy_scalers(X_train, y_train)

        # 0-B: remember training y range so predict can flag
        # extrapolation-blown CI widths (2026-04-20 log saw 18M-wide CI
        # on test when a few X_test rows had z-scores >> training range).
        self._y_train_range = float(np.ptp(y_train))

        # Bayesian Ridge: α(가중치 정밀도), λ(노이즈 정밀도) 자동 추정
        self._model = BayesianRidge(
            max_iter=500,
            tol=1e-6,
            alpha_1=1e-6,  # Gamma 사전분포 하이퍼파라미터
            alpha_2=1e-6,
            lambda_1=1e-6,
            lambda_2=1e-6,
            compute_score=True,  # 로그 마지날 우도 추적
        )
        self._model.fit(X_s, y_s)
        self._fitted = True

        log.info(f"  [BayesianRidge] α={self._model.alpha_:.4f}, "
                 f"λ={self._model.lambda_:.4f}")
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        X_s = self._scaler_X.transform(X_test)
        # 0-B: clip standardised X so that rows with z-scores far outside
        # training distribution do not blow up the BayesianRidge predictive
        # variance via the x'·Σ_w·x quadratic form. The point estimate is
        # unaffected (still a linear function of x); only the CI width is
        # protected from 18M-wide artefacts. Threshold is ±8σ — generous but
        # finite.
        X_s_clip = np.clip(X_s, -8.0, 8.0)
        pred_s, std_s = self._model.predict(X_s_clip, return_std=True)

        pred = self._scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel()
        y_scale = self._scaler_y.scale_[0]
        ci_width = Z95 * std_s * y_scale

        # 0-B: cap CI at 10x training range. Beyond that the interval
        # is numerically meaningless (pure extrapolation amplification) and
        # poisons downstream conformal calibration / PI comparison tables.
        if self._y_train_range and np.isfinite(self._y_train_range) and self._y_train_range > 0:
            _cap = 10.0 * self._y_train_range
            if (ci_width > _cap).any():
                _n_clip = int((ci_width > _cap).sum())
                log.warning(
                    f"  [BayesianRidge] extrapolation-blown CI: {_n_clip} test pts "
                    f"had width > 10×y-range ({ci_width.max():.1f} > {_cap:.1f}); "
                    f"clipped."
                )
                ci_width = np.minimum(ci_width, _cap)
        self._last_ci_width = ci_width  # for downstream PI consumers
        log.info(f"  [BayesianRidge] 평균 95% CI 폭: {ci_width.mean():.2f}")

        # G-303: in-model floor RETAINED (G-275 direct-use ILI≥0). Suboptimal under median-centered
        #   transforms (documented should-fix bias); phase-13 4-site + artifact floors handle ≥0.
        return np.maximum(pred, 0)

    def predict_interval(
        self, X_test: np.ndarray, alpha: float = 0.05, **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """Tier C ⑦ — native predictive interval from BayesianRidge's
 marginalized posterior (μ ± z·σ), inverse-transformed to raw space.

 Uses sklearn's `return_std=True` which returns the posterior
 predictive std dev σ(x) = √(1/λ + x'Σx) — this is *already* marginal
 over the posterior β, so no sampling is needed. The same z-clip
 and CI-cap used in predict apply here.
 """
        if not self._fitted:
            raise RuntimeError("BayesianRidge: fit() first")
        from scipy.stats import norm as _norm
        z = float(_norm.ppf(1.0 - alpha / 2.0))

        X_s = self._scaler_X.transform(X_test)
        X_s_clip = np.clip(X_s, -8.0, 8.0)
        pred_s, std_s = self._model.predict(X_s_clip, return_std=True)
        y_scale = self._scaler_y.scale_[0]
        pred = self._scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel()
        half = z * std_s * y_scale

        if self._y_train_range and np.isfinite(self._y_train_range) and self._y_train_range > 0:
            cap = 10.0 * self._y_train_range
            half = np.minimum(half, cap)

        lo = np.maximum(pred - half, 0.0)
        hi = np.maximum(pred + half, 0.0)
        return lo, hi


# ═══════════════════════════════════════════════════════════════
# 3. Negative Binomial GLM
# ═══════════════════════════════════════════════════════════════

class _LogLinkGLM(BaseForecaster):
    """Shared machinery for the two true log-link count-family GLMs.

    Why this class exists
    ---------------------
    ``NegBinGLM`` and ``PoissonAutoreg`` shipped for months as ``RidgeCV`` under a count-GLM
    name: the earlier true-GLM attempts diverged (test R² −0.27…−1.69, and −281…−1327 when
    re-measured in 2026-07-14), so the estimator was quietly swapped while the thesis kept
    describing an NB2 log link with a dispersion parameter. That is a model the code did not
    contain. This class restores the described model and makes it converge.

    Why the earlier attempts diverged, and why this one does not
    -----------------------------------------------------------
    They fed RAW features to a log link. With μ = exp(Xβ) and a test peak (100.7) 1.5× above
    the train maximum (66.9), a linear predictor extrapolates *multiplicatively* and explodes.
    The model the thesis specifies puts the LAGS IN LOGS —
    ``log μ_t = a + Σ_k ρ_k log y_{t−k} + Σ_j β_j z_j`` — which is scale-stable: doubling y
    scales μ by 2^ρ, not by e^(β·Δy). Measured on the real split, that design converges
    (test R² +0.876 for NB2, +0.876 for Poisson) where the raw-feature design does not.

    The design is rebuilt from the data alone, because ``fit`` receives arrays with no column
    names: a column is treated as a target lag iff it *is* the target shifted (checked, not
    guessed — see ``_target_lag_mask``). Those columns enter as ``log1p``; everything else is
    screened by |Pearson r| and standardised.

    Honesty note (measured, 2026-07-14): the method-of-moments dispersion on this data comes
    out ≈0, i.e. the ILI rate shows no over-dispersion beyond Poisson once the mean model is
    in place. NB2 therefore behaves almost identically to Poisson here; the dispersion
    parameter is estimated and reported rather than assumed.

    The target is a RATE (0.81–100.7, 1.7% integers), not a count, so both families are used
    as quasi-likelihoods (Poisson pseudo-ML — Gourieroux, Monfort & Trognon 1984), which is
    valid for any positive continuous response.

    Performance: O(n·p) screening + IRLS/lbfgs on ≤ 60 columns; seconds, no subprocess needed.
    Caller responsibility: the external y-transform MUST be identity (both models are in the
    META force-identity set, per_model_optimize.py) — an outer log1p on top of the internal
    log link is a double transform and inverts to nonsense.
    """

    _K_EXOG = 8          # screened non-lag covariates (CV-measured best of 8/12/20)
    _CAP_MULT = 2.0      # linear-space extrapolation backstop, retained from the salvage
    _L2_GRID = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)

    def __init__(self, k_exog: int | None = None, fixed_l2: float | None = None):
        super().__init__()
        self._k_exog = int(k_exog) if k_exog else self._K_EXOG
        self._fixed_l2 = float(fixed_l2) if fixed_l2 is not None else None
        self._lag_mask = None      # bool[p] — columns that are shifted copies of y
        self._exog_idx = None      # int[k] — screened non-lag columns
        self._scaler_X = None
        self._model = None
        self._y_train_max = 0.0
        self._dispersion = 0.0     # φ (Poisson quasi) or α (NB2)

    # ---- design ---------------------------------------------------------
    @staticmethod
    def _target_lag_mask(X: np.ndarray, y: np.ndarray, max_lag: int = 60) -> np.ndarray:
        """True for columns that are the target shifted by some lag.

        Detected, not assumed: ``fit`` gets no column names, and feeding a raw ILI lag to a
        log link is exactly what made the earlier implementation explode. A column qualifies
        iff ``col[k:] == y[:-k]`` for some k — an exact structural test, so an unrelated but
        highly correlated covariate is never mistaken for a lag.
        """
        n, p = X.shape
        mask = np.zeros(p, dtype=bool)
        upper = min(max_lag, n - 5)
        for j in range(p):
            col = X[:, j]
            for k in range(1, upper):
                if np.allclose(col[k:], y[:-k], rtol=1e-5, atol=1e-6):
                    mask[j] = True
                    break
        return mask

    def _design(self, X: np.ndarray) -> np.ndarray:
        """Design matrix: log1p(target lags) ++ screened covariates, ALL standardised.

        The lag block must be standardised too. An L2 penalty shrinks coefficients, so it is
        only comparable across columns on a common scale — leaving log1p(lag) on its raw 0–4.6
        scale while the covariates were z-scored made the penalty land unevenly and cost ~4
        points of test R2.
        """
        lag = np.log1p(np.clip(X[:, self._lag_mask], 0.0, None))
        raw = np.hstack([lag, X[:, self._exog_idx]])
        return self._scaler_X.transform(raw)

    def _prepare(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        from sklearn.preprocessing import StandardScaler

        self._lag_mask = self._target_lag_mask(X_train, y_train)
        others = np.flatnonzero(~self._lag_mask)

        O = X_train[:, others]
        Xc = O - O.mean(axis=0, keepdims=True)
        yc = y_train - float(y_train.mean())
        num = (Xc * yc[:, None]).sum(axis=0)
        den = np.sqrt((Xc ** 2).sum(axis=0) * (yc ** 2).sum() + 1e-12)
        corr = np.abs(num / np.maximum(den, 1e-12))
        k = min(self._k_exog, len(others))
        self._exog_idx = others[np.argsort(-corr)[:k]]

        lag = np.log1p(np.clip(X_train[:, self._lag_mask], 0.0, None))
        self._scaler_X = StandardScaler().fit(
            np.hstack([lag, X_train[:, self._exog_idx]]))
        self._y_train_max = float(np.max(y_train))
        return self._design(X_train)

    def _cap(self) -> float:
        return self._CAP_MULT * self._y_train_max if self._y_train_max > 0 else np.inf

    def _cv_penalty(self, Z: np.ndarray, y: np.ndarray, make) -> tuple[float, np.ndarray, np.ndarray]:
        """Rolling-origin CV on the training pool — leak-free, mirrors the pipeline's OOF.

        Returns:
            (best_L2, oof_mu, oof_y) — the out-of-fold predictions are returned because the
            dispersion must be estimated from FORECAST error, not fit error. Estimating it
            from in-sample residuals gave φ = 0.22 (less variance than Poisson, which is
            nonsense for this series) and 95% intervals that covered 56% of the test weeks.
            Out-of-fold residuals are what the interval actually has to survive.
        """
        from sklearn.model_selection import TimeSeriesSplit

        grid = (self._fixed_l2,) if self._fixed_l2 is not None else self._L2_GRID
        cap, best, best_a = self._cap(), np.inf, grid[0]
        folds = list(TimeSeriesSplit(n_splits=5).split(Z))
        oof: dict[float, tuple[np.ndarray, np.ndarray]] = {}

        for a in grid:
            mus, ys = [], []
            for tr, va in folds:
                try:
                    m = make(a).fit(Z[tr], y[tr])
                    p = np.clip(np.nan_to_num(m.predict(Z[va]), nan=0.0, posinf=cap), 0.0, cap)
                except Exception:
                    p = np.full(len(va), np.nan)
                mus.append(p)
                ys.append(y[va])
            mu, yy = np.concatenate(mus), np.concatenate(ys)
            ok = np.isfinite(mu)
            score = float(np.mean(np.abs(yy[ok] - mu[ok]))) if ok.any() else np.inf
            oof[a] = (mu, yy)
            if score < best:
                best, best_a = score, a

        mu, yy = oof[best_a]
        ok = np.isfinite(mu)
        return best_a, mu[ok], yy[ok]

    # ---- intervals ------------------------------------------------------
    def _fit_variance(self, oof_mu: np.ndarray, oof_y: np.ndarray) -> None:
        """Fit the predictive variance function Var(mu) = phi*mu + alpha*mu^2 on OOF residuals.

        The GLM's own variance function describes the *likelihood*, not the *forecast* error:
        quasi-Poisson gives Var = phi*mu, and with mu ~ 14 and an out-of-fold RMSE near 11 that
        is an order of magnitude too narrow — the first cut covered 56% of test weeks at the 95%
        level. This form nests both families (alpha = 0 is quasi-Poisson, phi = 1 is NB2) and its
        two coefficients are fitted by non-negative least squares on squared out-of-fold
        residuals, so the interval widens with the level the way the data actually does.

        Side effects: sets ``self._var_phi`` / ``self._var_alpha`` / ``self._var_floor``.
        """
        mu = np.clip(oof_mu, 1e-6, None)
        r2 = (oof_y - mu) ** 2
        A = np.column_stack([mu, mu ** 2])
        coef, *_ = np.linalg.lstsq(A, r2, rcond=None)
        phi, alpha = (float(max(c, 0.0)) for c in coef)
        if phi <= 0.0 and alpha <= 0.0:          # degenerate fit -> constant variance
            phi, alpha = float(np.mean(r2) / max(float(np.mean(mu)), 1e-6)), 0.0
        self._var_phi, self._var_alpha = phi, alpha
        self._var_floor = float(max(np.percentile(r2, 10), 1e-6))
        # R9/R10 read this attribute to build a LEAK-FREE prediction interval
        # (per_model_optimize.py:1505-1512). Out-of-fold residuals ARE that — better than the
        # in-sample fallback, which understates forecast error. Without it the model is scored
        # with pi_source="unavailable" and its WIS comes out NaN, which is exactly why
        # PoissonAutoreg carried a dash in Table 2 instead of a score.
        self.insample_residuals = [float(v) for v in (oof_y - oof_mu)]
        # R9/R10 read this attribute to build a LEAK-FREE prediction interval
        # (per_model_optimize.py:1505). Out-of-fold residuals are exactly that; without it the
        # model is scored with pi_source="unavailable" and its WIS comes out NaN — which is why
        # PoissonAutoreg carried a dash in Table 2.
        self.insample_residuals = [float(v) for v in (oof_y - oof_mu)]

    def _variance(self, mu: np.ndarray) -> np.ndarray:
        return np.maximum(self._var_phi * mu + self._var_alpha * mu ** 2, self._var_floor)

    def predict_interval(
        self, X_test: np.ndarray, alpha: float = 0.05,
        X_train_cache: Optional[np.ndarray] = None,
        y_train_cache: Optional[np.ndarray] = None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Native predictive interval from the GLM's own variance function.

        The response is a positive continuous rate, so the interval is taken from a Gamma
        predictive matched to the GLM mean and variance (shape = μ²/σ², scale = σ²/μ) rather
        than from integer Poisson/NB quantiles, which would be far too coarse at μ ≈ 5. This
        is a genuine model-based interval — the previous implementation had none, which is why
        ``PoissonAutoreg`` carried ``wis = nan`` and a dash in Table 2.

        Returns:
            (lo, hi) — both clipped to [0, 2 × y_train_max], lo ≤ hi elementwise.
        """
        from scipy.stats import gamma as _gamma

        if not self._fitted:
            raise RuntimeError(f"{self.meta.name}: fit() first")

        mu = np.clip(self.predict(X_test), 1e-6, None)
        var = np.clip(self._variance(mu), 1e-9, None)
        shape = np.clip(mu ** 2 / var, 1e-6, None)
        scale = var / mu

        cap = self._cap()
        lo = np.clip(_gamma.ppf(alpha / 2.0, shape, scale=scale), 0.0, cap)
        hi = np.clip(_gamma.ppf(1.0 - alpha / 2.0, shape, scale=scale), 0.0, cap)
        return np.minimum(lo, hi), np.maximum(lo, hi)


class NegBinGLMForecaster(_LogLinkGLM):
    """Negative-binomial (NB2) GLM with a log link — the standard surveillance count model.

    ``log μ_t = a + Σ_k ρ_k log(1 + y_{t−k}) + Σ_j β_j z_j``, with NB2 variance
    ``Var(y) = μ + α μ²``. The dispersion α is estimated from the data by method of moments
    on a Poisson pre-fit (it is *not* assumed), and the coefficients are then fitted by
    L2-regularised IRLS. Basis of the Farrington algorithm and of the ``surveillance`` R
    package; used by WHO / CDC / ECDC surveillance reports.
    """

    meta = ModelMeta(
        name="NegBinGLM",
        category="epi",
        level=5,
        min_data=60,
        description="Negative-binomial (NB2) GLM, log link, estimated dispersion.",
        dependencies=["statsmodels"],
    )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> "NegBinGLMForecaster":
        import statsmodels.api as sm

        y = np.clip(np.asarray(y_train, float), 0.0, None)
        Z = sm.add_constant(self._prepare(X_train, y), has_constant="add")

        # Provisional dispersion for the IRLS weights, from a Poisson pre-fit.
        pre = sm.GLM(y, Z, family=sm.families.Poisson()).fit_regularized(alpha=0.1, L1_wt=0.0)
        mu0 = np.clip(pre.predict(Z), 1e-6, None)
        prov = max(float(np.mean((y - mu0) ** 2 - mu0) / np.mean(mu0 ** 2)), 1e-3)

        # Each CV fold refits on that fold's rows only — an adapter that closed over a
        # whole-data fit would leak the validation rows into every candidate penalty.
        fam = sm.families.NegativeBinomial(alpha=prov)
        l2, oof_mu, oof_y = self._cv_penalty(Z, y, lambda a: _SMGlm(fam, a))
        pen = np.full(Z.shape[1], float(l2)); pen[0] = 0.0      # intercept unpenalised
        self._model = sm.GLM(y, Z, family=fam).fit_regularized(alpha=pen, L1_wt=0.0)
        self._l2 = l2

        # NB2 dispersion from OUT-OF-FOLD residuals: solve E[(y-mu)^2] = mu + alpha*mu^2.
        # In-sample residuals understate forecast error and produced 56%-covering intervals.
        m2 = np.clip(oof_mu, 1e-6, None)
        self._dispersion = max(
            float(np.mean((oof_y - m2) ** 2 - m2) / np.mean(m2 ** 2)), 1e-3)
        self._fit_variance(oof_mu, oof_y)
        self._fitted = True
        log.info(f"  [NegBinGLM] NB2 log-link: lags={int(self._lag_mask.sum())} "
                 f"exog={len(self._exog_idx)} dispersion={self._dispersion:.4f} L2={l2:g}")
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        import statsmodels.api as sm
        Z = sm.add_constant(self._design(X_test), has_constant="add")
        mu = np.nan_to_num(self._model.predict(Z), nan=0.0, posinf=self._cap(), neginf=0.0)
        return np.clip(mu, 0.0, self._cap())


class _SMGlm:
    """sklearn-shaped adapter around a statsmodels GLM, refitting on whatever rows it is given.

    ``_cv_penalty`` hands each fold its own rows; this must fit on exactly those rows, so the
    GLM is constructed inside ``fit`` rather than captured from the caller.
    """

    def __init__(self, family, l2: float):
        self._family, self._l2, self._res = family, l2, None

    def fit(self, X, y):
        import statsmodels.api as sm
        # statsmodels penalises EVERY coefficient including the constant; sklearn never does.
        # In a log link a shrunk intercept drags log(mu) toward 0, i.e. mu toward 1, which
        # under-predicts the whole series. Exempt it explicitly.
        a = np.full(X.shape[1], float(self._l2))
        a[0] = 0.0
        self._res = sm.GLM(y, X, family=self._family).fit_regularized(alpha=a, L1_wt=0.0)
        return self

    def predict(self, X):
        return self._res.predict(X)


# ═══════════════════════════════════════════════════════════════
# 4. Generalized Additive Model (GAM)
# ═══════════════════════════════════════════════════════════════

class GAMForecaster(BaseForecaster):
    """일반화 가법 모형 — 비선형 공변량 효과 모델링.

    g(E[Y]) = β₀ + Σ fⱼ(Xⱼ)  (fⱼ = 평활 함수)
    평활 함수로 스플라인 사용 → 기온-ILI, 습도-ILI 등 비선형 관계 포착.

    역학적 의의:
      - 환경역학에서 기온-사망률/질병 관계 분석의 표준 (Gasparrini et al., 2010)
      - Dlnm(distributed lag non-linear model)의 기반
      - 계절성 + 장기추세를 부드러운 곡선으로 분리
    """

    meta = ModelMeta(
        name="GAM-Spline",
        category="epi",
        level=6,
        min_data=60,
        description="Generalized Additive Model (spline smoothing). 비선형 공변량 효과.",
        dependencies=["sklearn"],
    )

    def __init__(self, n_splines: int = 10, lam: float = 0.3, topk: int = 10):
        super().__init__()
        self._model = None
        self._scaler_X = None
        self._scaler_y = None
        self._n_splines = n_splines
        self._lam = lam
        self._topk = topk
        self._feat_idx = None  # top-K feature selection mask
        self._y_log_used = True
        self._use_pygam = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> "GAMForecaster":
        from sklearn.preprocessing import StandardScaler, SplineTransformer
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import Pipeline

        # salvage (bench_salvage_v6.md V6_topK10_lam0.3, test R²=+0.3984):
        # GAM 이 p=309, n=234 전체에 스플라인 20개 적용하면 design-matrix 조건수 폭발,
        # 학습 외 구간 (2023+ ILI surge) 에서 선형 외삽이 실측치 못 따라감 (test R²=-0.16).
        # MP-PINN C-MP 교훈 → top-K |Pearson r| + log1p(y) + 작은 lam.
        # 스플라인은 top-K 피처에만 적용 → 조건수 안정, 규제(lam) 효과 극대화.
        n_train, p_orig = X_train.shape
        k = min(self._topk, p_orig)

        Xc = X_train - X_train.mean(axis=0, keepdims=True)
        yc = y_train - float(y_train.mean())
        num = (Xc * yc[:, None]).sum(axis=0)
        den = np.sqrt((Xc ** 2).sum(axis=0) * (yc ** 2).sum() + 1e-12)
        corr = np.abs(num / np.maximum(den, 1e-12))
        self._feat_idx = np.argsort(-corr)[:k]

        # transform-fix (2026-06-21): internal log1p REMOVED — fit on RAW y (identity). The single
        #   y-transform is now DATA-DRIVEN by the preproc Optuna search (per-model OOF selection),
        #   so GAM no longer log1p's y itself (that path's expm1 inverse blew up on out-of-range
        #   peaks). The upper prediction cap + 0-floor below are RETAINED (linear-space backstop).
        self._y_log_used = False
        # G-256d (2026-06-15): cap 기준 y_max 저장 (역변환 폭발 backstop, 이제 linear-space cap).
        self._y_train_max = float(np.max(y_train))
        y_fit = y_train.astype(float)

        # Sprint 1.5 R5 (2026-05-26): use shared setup_xy_scalers helper
        # (y_fit pre-transformed by caller)
        from simulation.models.base import setup_xy_scalers
        self._scaler_X, self._scaler_y, X_s, y_s = setup_xy_scalers(
            X_train[:, self._feat_idx], y_fit)

        try:
            from pygam import LinearGAM, s as _gam_s

            terms = _gam_s(0, n_splines=self._n_splines, lam=self._lam)
            for i in range(1, k):
                terms = terms + _gam_s(i, n_splines=self._n_splines, lam=self._lam)
            self._model = LinearGAM(terms)
            self._model.fit(X_s, y_s)
            self._use_pygam = True
            self._fitted = True
            log.info(f"  [GAM-Spline] top-K={k}/{p_orig}, n_splines={self._n_splines}, "
                     f"lam={self._lam} "
                     f"(topK+GAM raw-y; transform=data-driven)")
        except ImportError:
            log.info("  [GAM-Spline] pyGAM 없음 → SplineTransformer + RidgeCV fallback")
            n_knots = min(8, max(3, n_train // 40))
            self._model = Pipeline([
                ("spline", SplineTransformer(n_knots=n_knots, degree=3,
                                             include_bias=False,
                                             extrapolation="linear")),
                ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 20), cv=3)),
            ])
            self._model.fit(X_s, y_s)
            self._use_pygam = False
            self._fitted = True
            log.info(f"  [GAM-Spline] fallback: top-K={k}, knots={n_knots}")

        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        X_s = self._scaler_X.transform(X_test[:, self._feat_idx])
        pred_s = self._model.predict(X_s)
        pred = self._scaler_y.inverse_transform(
            np.asarray(pred_s).reshape(-1, 1)
        ).ravel()
        # transform-fix (2026-06-21): no internal inverse-transform (fit on raw y). The upper
        #   cap + 0-floor are PRESERVED as a linear-space extrapolation backstop — spline/GAM
        #   linear extrapolation can overshoot on out-of-range peaks (was the G-256d cap).
        _cap = 2.0 * self._y_train_max
        return np.clip(pred, 0.0, _cap)


# ═══════════════════════════════════════════════════════════════
# 5. Bayesian MCMC Regression
# ═══════════════════════════════════════════════════════════════

class BayesianMCMCForecaster(BaseForecaster):
    """베이지안 MCMC 회귀 — 사후분포 완전 샘플링.

    Metropolis-Hastings 알고리즘으로 회귀 계수의 사후분포 직접 샘플링.
    사전분포: β ~ N(0, τ²I), σ² ~ InvGamma(a, b)
    PyMC 없이도 동작하는 순수 구현.

    역학적 의의:
      - CDC FluSight 대회 상위 팀(LANL, Delphi)이 MCMC 기반
      - 불확실성 전파(propagation) 가능 → 위험 평가에 적합
      - 사전정보(prior knowledge) 반영 가능 (전문가 판단)
    """

    meta = ModelMeta(
        name="BayesianMCMC",
        category="epi",
        level=9,
        min_data=80,
        description="Bayesian MCMC (Metropolis-Hastings). 사후분포 완전 샘플링.",
        dependencies=["numpy"],
    )

    def __init__(self, n_samples: int = 5000, burnin: int = 1000, thin: int = 2):
        super().__init__()
        self._beta_samples = None
        self._sigma_samples = None
        self._scaler_X = None
        self._scaler_y = None
        self._n_samples = n_samples
        self._burnin = burnin
        self._thin = thin

    def _log_posterior(self, beta, sigma2, X, y, tau2=10.0):
        """로그 사후분포 (비례 부분만)."""
        n = len(y)
        # 가우시안 우도
        resid = y - X @ beta
        log_lik = -n / 2 * np.log(sigma2) - np.sum(resid ** 2) / (2 * sigma2)
        # 사전분포: β ~ N(0, τ²I)
        log_prior_beta = -np.sum(beta ** 2) / (2 * tau2)
        # 사전분포: σ² ~ InvGamma(2, 1) → log p(σ²) ∝ -3 log(σ²) - 1/σ²
        log_prior_sigma = -3 * np.log(sigma2) - 1.0 / sigma2
        return log_lik + log_prior_beta + log_prior_sigma

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> "BayesianMCMCForecaster":
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge

        # Sprint 1.5 R5 (2026-05-26): use shared setup_xy_scalers helper
        from simulation.models.base import setup_xy_scalers
        self._scaler_X, self._scaler_y, X_s, y_s = setup_xy_scalers(X_train, y_train)

        n, p = X_s.shape

        # 차원 축소: p >> n 문제 → 상위 피처만 사용
        self._p_use = min(p, 30)  # MCMC는 고차원에서 수렴 느림
        if p > self._p_use:
            # Ridge 계수 절대값 기준 상위 피처 선택
            ridge = Ridge(alpha=1.0).fit(X_s, y_s)
            self._top_idx = np.argsort(np.abs(ridge.coef_))[-self._p_use:]
            X_use = X_s[:, self._top_idx]
        else:
            self._top_idx = np.arange(p)
            X_use = X_s

        # 절편 추가
        X_aug = np.column_stack([np.ones(n), X_use])
        p_aug = X_aug.shape[1]

        # OLS 초기값
        # G-275 (2026-06-16, per-model 감사): collinear feature(같은 lag의 log1p/qbin/qnorm 변환 동시
        #   선택 → cond≈7.6e16 near-singular) 에서 lstsq(rcond=None) 은 작은 특이값을 안 잘라 beta_init 가
        #   |β|≈1e6 로 폭발 → MCMC 가 그 발산점에서 시작 → hold-out 외삽 시 예측 폭주(test r2 −4.35,
        #   pred 221≫y_max 67). base layer safe_lstsq(rcond=1e-6 + ridge fallback) 로 |β|≈0.7 안정.
        from simulation.models.safety import safe_lstsq
        beta_init = safe_lstsq(X_aug, y_s)
        sigma2_init = np.var(y_s - X_aug @ beta_init)

        # Adaptive Metropolis-Hastings
        beta = beta_init.copy()
        sigma2 = max(sigma2_init, 0.01)
        log_post = self._log_posterior(beta, sigma2, X_aug, y_s)

        beta_chain = np.zeros((self._n_samples, p_aug))
        sigma_chain = np.zeros(self._n_samples)

        # 제안 분포 스케일 (적응적)
        beta_scale = 0.01 * np.ones(p_aug)
        sigma_scale = 0.05
        n_accept_beta = 0
        n_accept_sigma = 0

        rng = np.random.RandomState(42)

        for i in range(self._n_samples):
            # β 업데이트 (블록)
            beta_prop = beta + rng.normal(0, beta_scale)
            log_post_prop = self._log_posterior(beta_prop, sigma2, X_aug, y_s)
            if np.log(rng.uniform()) < log_post_prop - log_post:
                beta = beta_prop
                log_post = log_post_prop
                n_accept_beta += 1

            # σ² 업데이트
            sigma2_prop = sigma2 * np.exp(rng.normal(0, sigma_scale))
            if sigma2_prop > 1e-8:
                log_post_prop = self._log_posterior(beta, sigma2_prop, X_aug, y_s)
                # Jacobian for log-normal proposal
                log_jacobian = np.log(sigma2_prop) - np.log(sigma2)
                if np.log(rng.uniform()) < log_post_prop - log_post + log_jacobian:
                    sigma2 = sigma2_prop
                    log_post = log_post_prop
                    n_accept_sigma += 1

            beta_chain[i] = beta
            sigma_chain[i] = sigma2

            # 적응: 매 200번마다 수용률 확인 후 스케일 조정
            if (i + 1) % 200 == 0 and i < self._burnin:
                accept_rate = n_accept_beta / (i + 1)
                if accept_rate < 0.2:
                    beta_scale *= 0.8
                elif accept_rate > 0.4:
                    beta_scale *= 1.2

        # Burn-in 제거 + Thinning
        post_beta = beta_chain[self._burnin::self._thin]
        post_sigma = sigma_chain[self._burnin::self._thin]

        self._beta_samples = post_beta
        self._sigma_samples = post_sigma
        self._beta_mean = post_beta.mean(axis=0)
        self._fitted = True

        accept_rate = n_accept_beta / self._n_samples
        log.info(f"  [BayesianMCMC] {len(post_beta)} posterior samples, "
                 f"accept_rate={accept_rate:.2%}, "
                 f"σ_mean={np.sqrt(post_sigma.mean()):.3f}")
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        X_s = self._scaler_X.transform(X_test)
        X_use = X_s[:, self._top_idx]
        X_aug = np.column_stack([np.ones(len(X_use)), X_use])

        # 사후 평균 예측
        pred_s = X_aug @ self._beta_mean
        pred = self._scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel()

        # 사후 예측 불확실성
        all_preds = X_aug @ self._beta_samples.T  # (n_test, n_posterior)
        pred_std_s = all_preds.std(axis=1)
        y_scale = self._scaler_y.scale_[0]
        ci_width = Z95 * pred_std_s * y_scale
        log.info(f"  [BayesianMCMC] 평균 95% CI 폭: {ci_width.mean():.2f}")

        return np.maximum(pred, 0)

    def predict_interval(
        self, X_test: np.ndarray, alpha: float = 0.05, **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """Tier C ⑦ — posterior-predictive interval directly from MCMC samples.

 For each posterior draw β_s, σ_s²: sample y_s ~ N(X·β_s, σ_s²),
 then take empirical α/2 and 1-α/2 quantiles per test row. This
 captures BOTH parameter uncertainty (posterior over β) AND
 observation noise (σ²) — the full Bayesian predictive interval.
 """
        if not self._fitted:
            raise RuntimeError("BayesianMCMC: fit() first")
        if self._beta_samples is None or self._sigma_samples is None:
            raise RuntimeError("MCMC chains missing — re-fit the model")
        X_s = self._scaler_X.transform(X_test)
        X_use = X_s[:, self._top_idx]
        X_aug = np.column_stack([np.ones(len(X_use)), X_use])

        # (n_test, n_posterior) mean surface
        mean_draws = X_aug @ self._beta_samples.T
        # observation noise per draw (sd scale in standardized y-space)
        sigma_draws = np.sqrt(np.maximum(self._sigma_samples, 1e-12))  # (n_post,)

        rng = np.random.RandomState(2026)
        noise = rng.standard_normal(mean_draws.shape) * sigma_draws[None, :]
        y_post_s = mean_draws + noise  # (n_test, n_posterior)

        lo_q = np.quantile(y_post_s, alpha / 2.0, axis=1)
        hi_q = np.quantile(y_post_s, 1.0 - alpha / 2.0, axis=1)

        y_scale = self._scaler_y.scale_[0]
        y_center = self._scaler_y.mean_[0]
        lo = lo_q * y_scale + y_center
        hi = hi_q * y_scale + y_center
        return np.maximum(lo, 0.0), np.maximum(hi, 0.0)


# ═══════════════════════════════════════════════════════════════
# 6. Poisson Autoregressive (Endemic-Epidemic)
# ═══════════════════════════════════════════════════════════════

class PoissonAutoregForecaster(_LogLinkGLM):
    """Log-linear Poisson autoregression — the endemic-epidemic surveillance decomposition.

    ``log μ_t = a + Σ_k ρ_k log(1 + y_{t−k}) + Σ_j β_j z_j`` — the model the thesis prints.
    The autoregressive block (ρ_k on the log-lags) carries the epidemic persistence; the
    screened covariate block (β_j on seasonality and exogenous drivers) carries the endemic
    baseline. Held & Paul (2012), the ``hhh4`` family used in ECDC surveillance reporting.

    This replaces a ``RidgeCV`` on raw y that had shipped under this name since the 2026-06-21
    transform fix: the identity-link Gaussian fit had no Poisson likelihood, no log link, and
    no predictive distribution — which is why the model carried ``wis = nan``. See
    :class:`_LogLinkGLM` for why the log-lag design converges where the earlier raw-feature
    log link diverged.

    Variance is Poisson quasi-likelihood, ``Var(y) = φ μ``, with φ estimated from the Pearson
    residuals (the response is a rate, not a count, so φ is not fixed at 1).
    """

    meta = ModelMeta(
        name="PoissonAutoreg",
        category="epi",
        level=7,
        min_data=60,
        description="Poisson autoregression (log link, endemic-epidemic, hhh4-style).",
        dependencies=["scikit-learn"],
    )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> "PoissonAutoregForecaster":
        from sklearn.linear_model import PoissonRegressor

        y = np.clip(np.asarray(y_train, float), 0.0, None)
        Z = self._prepare(X_train, y)

        l2, oof_mu, oof_y = self._cv_penalty(
            Z, y, lambda a: PoissonRegressor(alpha=a, max_iter=10000))
        self._model = PoissonRegressor(alpha=l2, max_iter=10000).fit(Z, y)
        self._l2 = l2

        # quasi-Poisson dispersion phi = mean((y-mu)^2 / mu), from OUT-OF-FOLD residuals.
        # The in-sample version returned phi = 0.22 — less spread than Poisson, which is not
        # a property this series has; it is an artefact of scoring the fit on its own rows.
        mu = np.clip(oof_mu, 1e-6, None)
        self._dispersion = max(float(np.mean((oof_y - mu) ** 2 / mu)), 1e-3)
        self._fit_variance(oof_mu, oof_y)

        self._fitted = True
        log.info(f"  [PoissonAutoreg] log-link: lags={int(self._lag_mask.sum())} "
                 f"exog={len(self._exog_idx)} phi={self._dispersion:.3f} L2={l2:g}")
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        mu = self._model.predict(self._design(X_test))
        mu = np.nan_to_num(mu, nan=0.0, posinf=self._cap(), neginf=0.0)
        return np.clip(mu, 0.0, self._cap())




# ═══════════════════════════════════════════════════════════════
# 모델 등록
# ═══════════════════════════════════════════════════════════════

REGISTRY.register(GaussianProcessForecaster)
REGISTRY.register(BayesianRidgeForecaster)
REGISTRY.register(NegBinGLMForecaster)
REGISTRY.register(GAMForecaster)
REGISTRY.register(BayesianMCMCForecaster)
REGISTRY.register(PoissonAutoregForecaster)

log.debug("  epi_models: 6개 역학 모델 등록 완료")
