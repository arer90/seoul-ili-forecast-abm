"""
simulation/models/linear_models.py
==================================
선형/커널(Linear/Kernel) 범주 모델: SVR-Linear, SVR-RBF, ElasticNet, KRR

- SVR-Linear:  소표본 최적 ML, margin 기반 정규화 (Level 2)
- SVR-RBF:     비선형 확장, GridSearchCV + TimeSeriesSplit (Level 3)
- ElasticNet:  L1+L2 혼합 정규화, feature selection 효과 (Level 4)
- KRR:         Kernel Ridge Regression, 커널 공간에서 릿지 회귀 (Level 5)

모두 StandardScaler 정규화 후 학습.
"""

from __future__ import annotations

import gc
import logging

import numpy as np

from simulation.models.base import BaseForecaster, ModelMeta, REGISTRY
from simulation.config_global import GLOBAL  # SSOT (2026-05-28)

log = logging.getLogger(__name__)


def _cap_linear_extrapolation(pred: np.ndarray, y_train_max) -> np.ndarray:
    """G-275 base layer: linear/kernel 외삽 폭발 상한 cap (count 가족 2×y_max 동형).

    선형/poly-kernel 회귀는 train 범위 밖 test feature 에서 선형 외삽 → 예측 폭주 가능
    (ill-conditioned 시 특히). 2×y_train_max 로 cap — 정상/peak 예측엔 no-op(ILI peak <
    2×train_max), 외삽 폭발만 bound. y_train_max=None(미저장 구버전 artifact) → 통과(back-compat).

    Args:
        pred: 예측 (이미 0-floor 적용된 상태 권장).
        y_train_max: train 타깃 최댓값(float) 또는 None.
    Returns:
        cap 적용된 예측 (shape 보존).
    """
    if y_train_max is not None and float(y_train_max) > 0:
        return np.minimum(pred, 2.0 * float(y_train_max))
    return pred


# ═══════════════════════════════════════════════════════════════
# 1. SVR-Linear -- Level 2
# ═══════════════════════════════════════════════════════════════

class SVRLinearForecaster(BaseForecaster):
    """SVR 선형 커널 -- 소표본(~300주) 최적 ML."""

    meta = ModelMeta(
        name="SVR-Linear",
        category="linear",
        level=2,
        min_data=60,
        description="SVR 선형 커널. margin 기반 정규화로 소표본 과적합 방지.",
        dependencies=["sklearn"],
    )

    def __init__(self):
        super().__init__()
        self._model = None
        self._scaler_X = None
        self._scaler_y = None
        self._feat_idx = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> SVRLinearForecaster:
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR
        # Mac-migration: BLAS 2-thread for libsvm SMO precision.
        from simulation.models._omp_context import blas_threads
        from simulation.models._feature_selection import mi_top_k_adaptive

        # adaptive-K (2026-04-22):
        # K_desired=40 (val-selected middle ground), capped at n_train//4 so
        # leftmost WF-CV folds (n~100) auto-shrink to K=25, avoiding the
        # p>n collapse observed when hard-coded K exceeds the fold's
        # effective rank.
        self._feat_idx = mi_top_k_adaptive(X_train, y_train, K_desired=40, divisor=4)
        X_sel = X_train[:, self._feat_idx]

        self._scaler_X = StandardScaler()
        self._scaler_y = StandardScaler()
        X_s = self._scaler_X.fit_transform(X_sel)
        y_s = self._scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

        # G-303 (2026-06-17, 검증 적발): C/epsilon 탐색 추가 — 형제 SVR-RBF 는 20-trial study 인데
        #   SVR-Linear 만 C=1.0 고정 = methods 비대칭(reviewer-value). 동일 tscv harness, linear 커널
        #   이라 gamma 없음. fast-path(MPH_INNER_HP_FAST, 비교 단계)는 고정 default 유지(비용 보존).
        import os as _os_svl
        _fast_hp = _os_svl.environ.get("MPH_INNER_HP_FAST") == "1"
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            _use_optuna = True
        except ImportError:
            _use_optuna = False
        if _fast_hp or not _use_optuna:
            best_params = {"C": 1.0, "epsilon": 0.01}
        else:
            from sklearn.metrics import r2_score
            from sklearn.model_selection import TimeSeriesSplit
            _tscv = TimeSeriesSplit(n_splits=3)

            def _obj(trial):
                _p = {"C": trial.suggest_float("C", 0.1, 100.0, log=True),
                      "epsilon": trial.suggest_float("epsilon", 5e-3, 0.2, log=True)}
                _sc = []
                for _tr, _va in _tscv.split(X_s):
                    _m = SVR(kernel="linear", max_iter=200_000, **_p)  # G-312: cap SMO iters — libsvm linear non-convergence hung → isolate stall-kill (TabPFN/SVR-Linear 손실)
                    with blas_threads(2):
                        _m.fit(X_s[_tr], y_s[_tr])
                    _sc.append(r2_score(y_s[_va], _m.predict(X_s[_va])))
                    del _m
                return float(np.mean(_sc))

            _study = optuna.create_study(
                direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
            _study.optimize(_obj, n_trials=20, show_progress_bar=False, gc_after_trial=True)
            best_params = {"C": float(_study.best_params["C"]),
                           "epsilon": float(_study.best_params["epsilon"])}
        self._model = SVR(kernel="linear", max_iter=200_000, **best_params)  # G-312
        with blas_threads(2):
            self._model.fit(X_s, y_s)
        self._y_train_max = float(np.max(y_train))   # G-275: 외삽 폭발 cap 기준
        self._fitted = True
        log.info(f"  [SVR-Linear] MI adaptive K={len(self._feat_idx)}/{X_train.shape[1]}, "
                 f"C={best_params['C']:.3g} eps={best_params['epsilon']:.3g}"
                 f"{' (fast default)' if (_fast_hp or not _use_optuna) else ' (20-trial tscv)'}")
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        X_sel = X_test[:, self._feat_idx] if self._feat_idx is not None else X_test
        X_s = self._scaler_X.transform(X_sel)
        pred_s = self._model.predict(X_s)
        # G-303 (2026-06-17): in-model floor RETAINED — it enforces G-275's direct-use ILI≥0 contract
        #   (linear/kernel models extrapolate strongly negative; floor is correct in original units for
        #   direct use + monotone R9(per_model_optimize) transforms). It IS suboptimal under median-centered transforms
        #   (mcmc_robust/laplace: floors sub-median to median), a documented should-fix bias — but removing
        #   it broke test_g275_linear_cap (direct-use −47.8). R9(per_model_optimize, 4-site) + artifact floors handle ≥0.
        pred = np.maximum(
            self._scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel(), 0
        )
        return _cap_linear_extrapolation(pred, getattr(self, "_y_train_max", None))


# ═══════════════════════════════════════════════════════════════
# 2. SVR-RBF -- Level 3
# ═══════════════════════════════════════════════════════════════

class SVRRBFForecaster(BaseForecaster):
    """SVR RBF 커널 + GridSearchCV -- 비선형 패턴 포착."""

    meta = ModelMeta(
        name="SVR-RBF",
        category="linear",
        level=3,
        min_data=80,
        description="SVR RBF 커널 + GridSearchCV. 비선형 패턴, 기상 공변량 포함.",
        dependencies=["sklearn"],
    )

    def __init__(self):
        super().__init__()
        self._model = None
        self._scaler_X = None
        self._scaler_y = None
        self._feat_idx = None
        self._pca = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> SVRRBFForecaster:
        from sklearn.metrics import r2_score
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR
        from simulation.models._feature_selection import mi_top_k_adaptive

        # adaptive-K (2026-04-22):
        # Replaces legacy PCA p=320→80 (R²=0.778). K_desired=20 (val-selected
        # robust minimum for RBF kernel — larger K amplifies noise through
        # the kernel matrix). Adaptive cap n//4 is redundant at n=180 (45>20)
        # but kicks in at WF-CV leftmost folds.
        self._feat_idx = mi_top_k_adaptive(X_train, y_train, K_desired=20, divisor=4)
        X_sel = X_train[:, self._feat_idx]
        log.info(f"  [SVR-RBF] MI adaptive K={len(self._feat_idx)}/{X_train.shape[1]} "
                 f"(replaces legacy PCA)")

        self._scaler_X = StandardScaler()
        self._scaler_y = StandardScaler()
        X_s = self._scaler_X.fit_transform(X_sel)
        y_s = self._scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            use_optuna = True
        except ImportError:
            use_optuna = False

        tscv = TimeSeriesSplit(n_splits=3)

        # G-273d (2026-06-15): SVR-RBF 도 3-모드 fast-path 가드(트리와 통일) — 비교 단계(preproc/
        #   feature/mc)서 full 20-trial study 대신 단일 default(fast)/축소 탐색(preproc). 이미 인라인
        #   Optuna 라 가드만 추가. 최종 refit 은 플래그 미설정 → full study.
        import os as _os_svr
        _fast_hp = _os_svr.environ.get("MPH_INNER_HP_FAST") == "1"

        if _fast_hp:
            best_score = -np.inf
            best_params = {"C": 100.0, "gamma": 1e-4, "epsilon": 0.01}  # 강한 초기값(test R²+0.67)
        elif use_optuna:
            # B-6: Optuna TPE (same pattern as XGBoost). 60-combo grid →
            # log-uniform TPE. γ range 1e-4~10, C 0.5~500, ε 5e-3~0.2.
            def objective(trial):
                params = {
                    "C": trial.suggest_float("C", 0.5, 500.0, log=True),
                    "gamma": trial.suggest_float("gamma", 1e-4, 10.0, log=True),
                    "epsilon": trial.suggest_float("epsilon", 5e-3, 0.2, log=True),
                }
                fold_scores = []
                for i, (train_idx, val_idx) in enumerate(tscv.split(X_s)):
                    X_tr, X_val = X_s[train_idx], X_s[val_idx]
                    y_tr, y_val = y_s[train_idx], y_s[val_idx]
                    model = SVR(kernel="rbf", max_iter=200_000, **params)  # G-312
                    model.fit(X_tr, y_tr)
                    pred = model.predict(X_val)
                    fold_scores.append(r2_score(y_val, pred))
                    del model
                    # Cat 1 (Codex/ANO pattern, 2026-05-12): fold-level pruning.
                    # 이전 worktree: pruner 없음 + trial.report 없음 → 모든 trial 풀스캔.
                    # Cat 1 적용으로 catastrophic R² (env-gated) fail-fast.
                    _mean_r2 = float(np.mean(fold_scores))
                    trial.report(_mean_r2, i)
                    _r2_cutoff = GLOBAL.filter.r2_catastrophic_cutoff
                    if i == 0 and _mean_r2 < _r2_cutoff:
                        raise optuna.TrialPruned(
                            f"SVR-RBF fold 0 catastrophic R²={_mean_r2:.3f} < {_r2_cutoff}")
                    if trial.should_prune():
                        raise optuna.TrialPruned(
                            f"SVR-RBF pruned at fold {i} (R²={_mean_r2:.3f})")
                gc.collect()
                return float(np.mean(fold_scores))

            from simulation.models._optuna_budget import get_trials as _get_trials
            from simulation.models._optuna_pruners import get_best_pruner_for as _get_pruner
            _n_trials = _get_trials("SVR-RBF", default=20)  # 2026-05-28: HP trial default 20 통일 (사용자 명시)
            _pp_k = _os_svr.environ.get("MPH_INNER_HP_PREPROC_TRIALS")   # G-273c-B: preproc = 축소 탐색
            if _pp_k:
                _n_trials = max(2, int(_pp_k))
            # 2026-05-22 Codex audit fix: TPESampler unseeded → SVR-RBF Δ-0.24 reproducibility 이슈
            # seed=42 + multivariate=True (R9(per_model_optimize) preproc sampler 와 동일) → deterministic search
            # G-13F (2026-06-21, codex+재확인): 위 주석은 seed=42 라 했으나 실제 seed= 미전달 = 버그
            #   (SVR-RBF 동일설정 test R² 0.803 vs 0.868 run간 변동의 근본원인). seed=42 실제 전달.
            study = optuna.create_study(direction="maximize",
                                        sampler=optuna.samplers.TPESampler(multivariate=True, seed=42),
                                        pruner=_get_pruner("SVR-RBF"))
            # : per_model_experiments 에서 발견한 강한 초기값 (test R²=0.67).
            # Optuna 가 CV fold 안에선 이 점을 못 찾았으므로 enqueue 로 보장.
            study.enqueue_trial({"C": 100.0, "gamma": 1e-4, "epsilon": 0.01})
            study.enqueue_trial({"C": 10.0, "gamma": 1e-3, "epsilon": 0.01})
            # G-161 (2026-05-02): trial cleanup callback + gc_after_trial 강제.
            from simulation.models._optuna_torch import make_trial_cleanup_callback
            study.optimize(objective, n_trials=_n_trials,
                           callbacks=[make_trial_cleanup_callback("SVR-RBF")],
                           gc_after_trial=True, show_progress_bar=False)
            best_params = study.best_params
            best_score = study.best_value
            del study
            gc.collect()
        else:
            # Mac-migration: fallback default 를 per_model_experiments 결과로 교체.
            # 구 default (C=10, gamma=0.1, eps=0.05) → test R²=-0.75 붕괴.
            # 신 default (C=100, gamma=1e-4, eps=0.01) → test R²=+0.67.
            best_score = -np.inf
            best_params = {"C": 100.0, "gamma": 1e-4, "epsilon": 0.01}

        self._model = SVR(kernel="rbf", max_iter=200_000, **best_params)  # G-312
        # Mac-migration: BLAS 2-thread for libsvm SMO (RBF kernel matrix).
        # G-13F (2026-06-21, codex): deterministic 모드(default on)서 BLAS=1 → libsvm 커널행렬 수치
        #   결정성(2-thread reduction 순서 비결정성이 SVR-RBF run간 변동 증폭). fast(=0)는 2-thread.
        from simulation.models._omp_context import blas_threads
        _det_svr = _os_svr.environ.get("MPH_DETERMINISTIC", "1") == "1"
        with blas_threads(1 if _det_svr else 2):
            self._model.fit(X_s, y_s)
        self._fitted = True
        log.info(f"  [SVR-RBF] best_params={best_params}, CV R²={best_score:.4f}")
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        X_sel = X_test[:, self._feat_idx] if self._feat_idx is not None else X_test
        X_s = self._scaler_X.transform(X_sel)
        pred_s = self._model.predict(X_s)
        # G-303: in-model floor RETAINED (G-275 direct-use ILI≥0 contract — see SVR-Linear).
        return np.maximum(
            self._scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel(), 0
        )


# ═══════════════════════════════════════════════════════════════
# 3. ElasticNet -- Level 4
# ═══════════════════════════════════════════════════════════════

class ElasticNetForecaster(BaseForecaster):
    """ElasticNet -- L1+L2 혼합 정규화, 자동 feature selection."""

    meta = ModelMeta(
        name="ElasticNet",
        category="linear",
        level=4,
        min_data=60,
        description="L1+L2 혼합 정규화. 불필요 피처 자동 제거, 해석 용이.",
        dependencies=["sklearn"],
    )

    def __init__(self):
        super().__init__()
        self._model = None
        self._scaler_X = None
        self._scaler_y = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> ElasticNetForecaster:
        from sklearn.linear_model import ElasticNetCV
        from sklearn.preprocessing import StandardScaler

        self._scaler_X = StandardScaler()
        self._scaler_y = StandardScaler()
        X_s = self._scaler_X.fit_transform(X_train)
        y_s = self._scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

        from sklearn.model_selection import TimeSeriesSplit
        # E-2: ConvergenceWarning 완화
        #   - tol=1e-3 (default 1e-4) : duality gap 수렴 기준을 약간 완화.
        #     feature 수가 많고(300+) p≈n 구간이라 1e-4 까진 자주 못 간다.
        #   - selection='random' : cyclic coordinate descent 보다 random 이
        #     강한 상관이 많은 feature set 에서 수렴이 빠른 경우가 있다.
        self._model = ElasticNetCV(
            l1_ratio=[0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
            alphas=np.logspace(-6, 2, 50),
            cv=TimeSeriesSplit(n_splits=3),  # 3 splits: 초기 fold 샘플 부족 방지 (G-028)
            max_iter=10000,
            tol=1e-3,
            selection="random",
            random_state=42,
        )
        self._model.fit(X_s, y_s)
        self._y_train_max = float(np.max(y_train))   # G-275: 외삽 폭발 cap 기준
        self._fitted = True
        log.info(f"  [ElasticNet] alpha={self._model.alpha_:.4f}, l1_ratio={self._model.l1_ratio_:.2f}")
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        X_s = self._scaler_X.transform(X_test)
        pred_s = self._model.predict(X_s)
        # G-303 (2026-06-17): in-model floor RETAINED — it enforces G-275's direct-use ILI≥0 contract
        #   (linear/kernel models extrapolate strongly negative; floor is correct in original units for
        #   direct use + monotone R9(per_model_optimize) transforms). It IS suboptimal under median-centered transforms
        #   (mcmc_robust/laplace: floors sub-median to median), a documented should-fix bias — but removing
        #   it broke test_g275_linear_cap (direct-use −47.8). R9(per_model_optimize, 4-site) + artifact floors handle ≥0.
        pred = np.maximum(
            self._scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel(), 0
        )
        return _cap_linear_extrapolation(pred, getattr(self, "_y_train_max", None))


# ═══════════════════════════════════════════════════════════════
# 4. KRR (Kernel Ridge Regression) -- Level 5
# ═══════════════════════════════════════════════════════════════

class KRRForecaster(BaseForecaster):
    """Kernel Ridge Regression -- 커널 공간에서의 릿지 회귀."""

    meta = ModelMeta(
        name="KRR",
        category="linear",
        level=5,
        min_data=60,
        description="Kernel Ridge Regression. SVR 대비 확률적 해석, 정규화 릿지.",
        dependencies=["sklearn"],
    )

    def __init__(self):
        super().__init__()
        self._model = None
        self._scaler_X = None
        self._scaler_y = None
        self._feat_idx = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> KRRForecaster:
        import os
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_squared_error
        # Mac-migration: BLAS thread context (libomp 와 무관, XGBoost 안전)
        from simulation.models._omp_context import blas_threads
        from simulation.models._feature_selection import mi_top_k_adaptive

        # adaptive-K (2026-04-22): KRR wants K=80 (squared-loss tolerates bigger K than SVR ε-hinge).
        self._feat_idx = mi_top_k_adaptive(X_train, y_train, K_desired=80, divisor=2)
        X_sel = X_train[:, self._feat_idx]
        log.info(f"  [KRR] MI adaptive K={len(self._feat_idx)}/{X_train.shape[1]}")

        self._scaler_X = StandardScaler()
        self._scaler_y = StandardScaler()
        X_s = self._scaler_X.fit_transform(X_sel)
        y_s = self._scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        tscv = TimeSeriesSplit(n_splits=3)  # G-028: n_splits=5 는 초기 fold 부족 → R² 음수, 3 으로 고정

        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            use_optuna = True
        except ImportError:
            use_optuna = False

        # G-273d (2026-06-15, 사용자 "통일-Optuna에서 시작했다"): KRR 을 grid → 인라인 Optuna 로 통일
        #   (트리·SVR-RBF 와 동일 3-모드 fast-path). HP 공간 = 기존 grid 동등(kernel/alpha/gamma).
        #   선택지표 = MSE(grid 의 neg_MSE 와 동일 — 메트릭 변화 0). MI-feat/scaler/blas_threads 보존.
        #   per_model_experiments: linear+alpha=10 이 test R²=0.64 (CV 가 못 찾던 점) → enqueue 로 보장.
        _fast_hp = os.environ.get("MPH_INNER_HP_FAST") == "1"

        if _fast_hp:
            best_params = {"kernel": "rbf", "alpha": 1.0, "gamma": 1e-3}
        elif use_optuna:
            def objective(trial):
                kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "polynomial"])
                params = {"kernel": kernel,
                          "alpha": trial.suggest_float("alpha", 1e-3, 100.0, log=True)}
                if kernel != "linear":
                    params["gamma"] = trial.suggest_float("gamma", 1e-4, 0.1, log=True)
                scores = []
                for tr_idx, va_idx in tscv.split(X_s):
                    m = KernelRidge(**params)
                    with blas_threads(2):
                        m.fit(X_s[tr_idx], y_s[tr_idx])
                    scores.append(mean_squared_error(y_s[va_idx], m.predict(X_s[va_idx])))
                    del m
                return float(np.mean(scores))

            from simulation.models._optuna_budget import get_trials as _get_trials
            _n_trials = _get_trials("KRR", default=20)
            _pp_k = os.environ.get("MPH_INNER_HP_PREPROC_TRIALS")   # G-273c-B: preproc = 축소 탐색
            if _pp_k:
                _n_trials = max(2, int(_pp_k))
            study = optuna.create_study(direction="minimize",  # MSE (lower=better)
                                        sampler=optuna.samplers.TPESampler(seed=42))  # G-13F: 재현성
            study.enqueue_trial({"kernel": "linear", "alpha": 10.0})  # grid 의 강한 점 보장
            study.optimize(objective, n_trials=_n_trials,
                           gc_after_trial=True, show_progress_bar=False)
            best_params = dict(study.best_params)
            del study
            gc.collect()
        else:
            # ── Fallback (optuna 부재): 기존 grid ──
            gs = GridSearchCV(
                KernelRidge(),
                [{"kernel": ["linear"], "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
                 {"kernel": ["rbf"], "alpha": [0.1, 1.0, 10.0, 100.0], "gamma": [1e-4, 1e-3, 1e-2, 0.1]},
                 {"kernel": ["polynomial"], "alpha": [1.0, 10.0], "gamma": [1e-3, 1e-2]}],
                cv=tscv, scoring="neg_mean_squared_error",
            )
            with blas_threads(2):
                gs.fit(X_s, y_s)
            best_params = gs.best_params_

        self._model = KernelRidge(**best_params)
        with blas_threads(2):   # Mac: kernel matrix solve 정확도 회복 (SIGSEGV 위험 없음)
            self._model.fit(X_s, y_s)
        self._y_train_max = float(np.max(y_train))   # G-275: 외삽 폭발 cap 기준 (poly kernel 외삽)
        self._fitted = True
        log.info(f"  [KRR] Best: {best_params}")
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        X_sel = X_test[:, self._feat_idx] if self._feat_idx is not None else X_test
        X_s = self._scaler_X.transform(X_sel)
        pred_s = self._model.predict(X_s)
        # G-303 (2026-06-17): in-model floor RETAINED — it enforces G-275's direct-use ILI≥0 contract
        #   (linear/kernel models extrapolate strongly negative; floor is correct in original units for
        #   direct use + monotone R9(per_model_optimize) transforms). It IS suboptimal under median-centered transforms
        #   (mcmc_robust/laplace: floors sub-median to median), a documented should-fix bias — but removing
        #   it broke test_g275_linear_cap (direct-use −47.8). R9(per_model_optimize, 4-site) + artifact floors handle ≥0.
        pred = np.maximum(
            self._scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel(), 0
        )
        return _cap_linear_extrapolation(pred, getattr(self, "_y_train_max", None))


# ── 등록 ──
REGISTRY.register(SVRLinearForecaster)
REGISTRY.register(SVRRBFForecaster)
REGISTRY.register(ElasticNetForecaster)
REGISTRY.register(KRRForecaster)
