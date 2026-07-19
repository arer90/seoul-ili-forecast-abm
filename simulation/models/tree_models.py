r"""
simulation/models/tree_models.py
================================
트리(Tree-based) 범주 모델: XGBoost, LightGBM, RandomForest

- XGBoost: XGBRegressor + TimeSeriesSplit CV + 균형 정규화 (Level 6)
- LightGBM: GBDT 경량 구현, 적정 leaf + 중간 정규화 (Level 7)
- RandomForest: 배깅 기반 앙상블, 적정 depth (Level 8)

 수정: 과도한 정규화로 인한 성능 하락 복구.
 - XGBoost: max_depth 3-6, learning_rate 0.02-0.05, reg_lambda 1-5
 - LightGBM: num_leaves 15-31, reg_lambda 1-3
 - RandomForest: max_depth 5-10, min_samples_leaf 5-15
 - TimeSeriesSplit: n_splits=3 (초기 fold 샘플 부족 방지)
"""

from __future__ import annotations

import gc
import logging
import os

import numpy as np

from simulation.models.base import BaseForecaster, ModelMeta, REGISTRY
from simulation.config_global import GLOBAL  # SSOT (2026-05-28)

log = logging.getLogger(__name__)


def _fold_wis(y_val: np.ndarray, pred: np.ndarray, y_tr: np.ndarray) -> float:
    """G-257 (2026-06-12, codex): internal tree-HP fold score = WIS (lower=better), replacing the
    per-fold R² mean. The old R² (1 − SS_res/SS_tot) divides by the fold variance, so on a
    low-variance quiet fold a tiny error explodes to R²≈−99 → HP selection dominated by quiet-fold
    NOISE, not accuracy (campaign exp_wis_vs_r2). It was also INCONSISTENT with the outer OOF
    selection objective (WIS). This computes a sigma-scaled WIS on the (transformed) CV fold —
    stable + a proper scoring rule + aligned with selection. σ = std(fold train y)."""
    from simulation.analytics.diagnostics import weighted_interval_score
    sigma = max(float(np.std(np.asarray(y_tr, dtype=float))), 1e-3)
    wis = weighted_interval_score(np.asarray(y_val, dtype=float),
                                  np.asarray(pred, dtype=float), sigma)
    return float(np.mean(wis))


# ═══════════════════════════════════════════════════════════════
# 1. XGBoost -- Level 6
# ═══════════════════════════════════════════════════════════════

class XGBoostForecaster(BaseForecaster):
    """XGBoost -- Gradient Boosting with strong regularization."""

    meta = ModelMeta(
        name="XGBoost",
        category="tree",
        level=6,
        min_data=80,
        description="XGBoost. 강한 L1/L2 정규화 + early stopping + 보수적 depth.",
        dependencies=["xgboost"],
    )

    def __init__(self):
        super().__init__()
        self._model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> XGBoostForecaster:
        import xgboost as xgb
        from sklearn.model_selection import TimeSeriesSplit

        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            use_optuna = True
        except ImportError:
            use_optuna = False

        tscv = TimeSeriesSplit(n_splits=3)

        # G-273c (2026-06-15): preproc/feature(Stage-1/2) eval 중엔 내부 HP study 생략 —
        # transform/feature 를 비교하는 단계라 단일 reasonable default + early_stop 1회로 충분
        # (중첩 HPO 폭주 제거 = 속도 ~수십×). Stage-3 최종 refit 은 _evaluate_config_hierarchical
        # 을 안 거쳐 플래그 미설정 → full HP study 정상 수행. (per_model_optimize.py:565 참조)
        _fast_hp = os.environ.get("MPH_INNER_HP_FAST") == "1"

        if _fast_hp:
            best_params = {"max_depth": 5, "learning_rate": 0.03, "reg_alpha": 0.3,
                           "reg_lambda": 1.5, "min_child_weight": 5, "subsample": 0.7,
                           "colsample_bytree": 0.6, "gamma": 0.0}
            best_score = float("nan")
        elif use_optuna:
            # ── Optuna HPO: 30 trials ( 업그레이드) ──
            # per_model_pipeline_isolated (2026-04-22):
            # A prior edit expanded max_depth 3-8 → 3-12 and enqueued a v4
            # winner (d=10 lr=0.02). In the pipeline testbench this produced
            # R²=0.360 (worse than legacy 0.82). Root cause: the
            # target-transform policy maps XGBoost → log1p; deep trees (d=10-12)
            # overfit log-space spikes and inverse expm1 amplifies the error.
            # Reverting max_depth back to 3-8 and removing the enqueue —
            # on log1p target shallower trees are the correct regularizer.
            def objective(trial):
                params = {
                    "max_depth": trial.suggest_int("max_depth", 3, 8),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 2.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
                    "min_child_weight": trial.suggest_int("min_child_weight", 3, 15),
                    "subsample": trial.suggest_float("subsample", 0.6, 0.9),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.8),
                    "gamma": trial.suggest_float("gamma", 0.0, 1.0),
                }
                fold_scores = []
                for i, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
                    X_tr, X_val = X_train[train_idx], X_train[val_idx]
                    y_tr, y_val = y_train[train_idx], y_train[val_idx]
                    model = xgb.XGBRegressor(
                        n_estimators=200, random_state=42, verbosity=0,
                        early_stopping_rounds=40, **params)  # G-268(3-way): early_stop — 200전부 학습→실효 ~50-150 (중첩 HPO 폭주 완화)
                    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                    pred = model.predict(X_val)
                    fold_scores.append(_fold_wis(y_val, pred, y_tr))
                    # : trial 모델 즉시 삭제 → OOM 방지
                    del model
                    # Cat 1 (Codex/ANO pattern, 2026-05-12): fold-level pruning.
                    # trial.report + should_prune 없으면 pruner inert. ANO repo
                    # 와 동일 패턴. Catastrophic R² (env-gated) 도 동시 fail-fast.
                    _mean_wis = float(np.mean(fold_scores))
                    trial.report(_mean_wis, i)   # G-257: WIS (minimize); should_prune respects direction
                    if trial.should_prune():
                        raise optuna.TrialPruned(
                            f"XGBoost pruned at fold {i} (WIS={_mean_wis:.3f})")
                gc.collect()
                return np.mean(fold_scores)

            # G-257 (2026-06-12, codex): Optuna sampler seed=42 — HPO 탐색까지 재현 가능.
            #        (이전: seed=None 난수. 최종 모델 학습은 항상 seed=42.)
            # : per_model_trials 예산 조회 (XGBoost D=8 → 60 권장)
            from simulation.models._optuna_budget import get_trials as _get_trials
            from simulation.models._optuna_pruners import get_best_pruner_for as _get_pruner
            _n_trials = _get_trials("XGBoost", default=20)
            _pp_k = os.environ.get("MPH_INNER_HP_PREPROC_TRIALS")   # G-273c-B: Stage-1 preproc = 축소 탐색
            if _pp_k:
                _n_trials = max(2, int(_pp_k))   # transform 비교용 작은 HP 탐색(full 추종, 단일점 발산 회피)
            study = optuna.create_study(direction="minimize",  # G-257: WIS (lower=better)
                                        sampler=optuna.samplers.TPESampler(seed=42),  # G-257: HPO 탐색도 재현 (codex)
                                        pruner=_get_pruner("XGBoost"))
            # G-161 (2026-05-02): trial cleanup callback + gc_after_trial 강제.
            # 이전 (G-158 fix 누락): callback 없음 + gc_after_trial 없음 →
            # in-process trial 종료 후 Python heap / Torch allocator 잔존 →
            # Cat 1 (tree) 1h 09m 만에 19% MEM. _trial_gpu_cleanup 가 gc 2회 +
            # malloc_trim + torch cache 비움 — heavy cleanup.
            from simulation.models._optuna_torch import make_trial_cleanup_callback
            study.optimize(objective, n_trials=_n_trials,
                           timeout=int(os.environ.get("MPH_XGB_STUDY_TIMEOUT", "300")),  # G-268: inner-study runaway 가드 (중첩 eval당; early_stop 정상시 미발동)
                           callbacks=[make_trial_cleanup_callback("XGBoost")],
                           gc_after_trial=True, show_progress_bar=False)
            best_params = study.best_params
            best_score = study.best_value
            # : study 객체 즉시 삭제 → OOM 방지
            del study
            gc.collect()
        else:
            # ── Fallback: 기존 6-combo grid search ──
            best_score = np.inf   # G-257: WIS minimize
            best_params = {}
            param_grid = [
                {"max_depth": 4, "learning_rate": 0.03, "reg_alpha": 0.5, "reg_lambda": 2.0,
                 "min_child_weight": 5, "subsample": 0.7, "colsample_bytree": 0.6, "gamma": 0.0},
                {"max_depth": 5, "learning_rate": 0.03, "reg_alpha": 0.3, "reg_lambda": 1.5,
                 "min_child_weight": 5, "subsample": 0.7, "colsample_bytree": 0.6, "gamma": 0.0},
                {"max_depth": 6, "learning_rate": 0.02, "reg_alpha": 0.5, "reg_lambda": 3.0,
                 "min_child_weight": 8, "subsample": 0.7, "colsample_bytree": 0.6, "gamma": 0.0},
                {"max_depth": 4, "learning_rate": 0.05, "reg_alpha": 0.1, "reg_lambda": 1.0,
                 "min_child_weight": 5, "subsample": 0.7, "colsample_bytree": 0.6, "gamma": 0.0},
                {"max_depth": 5, "learning_rate": 0.02, "reg_alpha": 1.0, "reg_lambda": 3.0,
                 "min_child_weight": 10, "subsample": 0.7, "colsample_bytree": 0.6, "gamma": 0.0},
                {"max_depth": 3, "learning_rate": 0.05, "reg_alpha": 0.5, "reg_lambda": 2.0,
                 "min_child_weight": 8, "subsample": 0.7, "colsample_bytree": 0.6, "gamma": 0.0},
            ]
            for params in param_grid:
                fold_scores = []
                for train_idx, val_idx in tscv.split(X_train):
                    X_tr, X_val = X_train[train_idx], X_train[val_idx]
                    y_tr, y_val = y_train[train_idx], y_train[val_idx]
                    model = xgb.XGBRegressor(
                        n_estimators=200, random_state=42, verbosity=0,
                        early_stopping_rounds=40, **params)  # G-268(3-way): early_stop — 200전부 학습→실효 ~50-150 (중첩 HPO 폭주 완화)
                    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                    pred = model.predict(X_val)
                    fold_scores.append(_fold_wis(y_val, pred, y_tr))
                avg_wis = np.mean(fold_scores)
                if avg_wis < best_score:
                    best_score = avg_wis
                    best_params = params

        # 전체 train으로 최종 학습
        val_size = max(10, int(len(y_train) * 0.15))
        X_tr, X_es = X_train[:-val_size], X_train[-val_size:]
        y_tr, y_es = y_train[:-val_size], y_train[-val_size:]

        # Optuna params에서 고정 파라미터 제외하고 사용
        self._model = xgb.XGBRegressor(
            n_estimators=200,   # 1200→200 (실측: 200=800 동일성능; early-stop 없는 XGBoost의 낭비 제거)
            random_state=42,
            verbosity=0,
            early_stopping_rounds=40,   # G-268(3-way): final-fit 도 early_stop (X_es hold-out)
            **best_params,
        )
        self._model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)
        self._fitted = True
        log.info(f"  [XGBoost] best_params={best_params}, CV WIS={best_score:.4f}")
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        # G-298 (2026-06-17, per-model 감사): NO transformed-space floor. Trees predict within the
        #   training leaf range, so under a median-centered y-transform (mcmc_robust/laplace)
        #   np.maximum(pred,0) floored legitimate sub-median predictions (transformed<0) up to the
        #   median → trough/quiet-season upward bias. The phase-13 inverse maps the tree's bounded
        #   range back to [min_y, max_y] ≥ 0, so non-negativity holds without the wrong-space clamp.
        return self._model.predict(X_test)


# ═══════════════════════════════════════════════════════════════
# 2. LightGBM -- Level 7
# ═══════════════════════════════════════════════════════════════

class LightGBMForecaster(BaseForecaster):
    """LightGBM -- 경량 Gradient Boosting, 강화 정규화."""

    meta = ModelMeta(
        name="LightGBM",
        category="tree",
        level=7,
        min_data=80,
        description="LightGBM. 보수적 num_leaves + 강한 정규화로 과적합 방지.",
        dependencies=["lightgbm"],
    )

    def __init__(self):
        super().__init__()
        self._model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> LightGBMForecaster:
        import lightgbm as lgb
        from sklearn.model_selection import TimeSeriesSplit

        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            use_optuna = True
        except ImportError:
            use_optuna = False

        tscv = TimeSeriesSplit(n_splits=3)

        # G-273c (2026-06-15): preproc/feature(Stage-1/2) eval 중엔 내부 HP study 생략 (XGBoost 동형).
        _fast_hp = os.environ.get("MPH_INNER_HP_FAST") == "1"

        if _fast_hp:
            best_params = {"num_leaves": 20, "learning_rate": 0.03, "reg_alpha": 0.3,
                           "reg_lambda": 1.5, "min_child_samples": 8,
                           "subsample": 0.7, "colsample_bytree": 0.6}
            best_score = float("nan")
        elif use_optuna:
            # ── Optuna HPO: 30 trials ( 업그레이드) ──
            def objective(trial):
                params = {
                    "num_leaves": trial.suggest_int("num_leaves", 10, 35),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 2.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
                    "min_child_samples": trial.suggest_int("min_child_samples", 3, 20),
                    "subsample": trial.suggest_float("subsample", 0.5, 0.9),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.8),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
                }
                fold_scores = []
                for i, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
                    X_tr, X_val = X_train[train_idx], X_train[val_idx]
                    y_tr, y_val = y_train[train_idx], y_train[val_idx]
                    dtrain = lgb.Dataset(X_tr, label=y_tr)
                    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
                    full_params = {
                        "objective": "regression", "metric": "rmse", "verbose": -1,
                        "seed": 42, **params,
                    }
                    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
                    model = lgb.train(full_params, dtrain, num_boost_round=200,  # 800→200 cap (early_stop 50 그대로)
                                      valid_sets=[dval], callbacks=callbacks)
                    pred = model.predict(X_val)
                    fold_scores.append(_fold_wis(y_val, pred, y_tr))
                    # : trial 모델 + Dataset 즉시 삭제 → OOM 방지
                    del model, dtrain, dval
                    # Cat 1 (2026-05-12): fold-level pruning (LightGBM)
                    _mean_wis = float(np.mean(fold_scores))
                    trial.report(_mean_wis, i)   # G-257: WIS (minimize); should_prune respects direction
                    if trial.should_prune():
                        raise optuna.TrialPruned(
                            f"LightGBM pruned at fold {i} (WIS={_mean_wis:.3f})")
                gc.collect()
                return np.mean(fold_scores)

            # G-257 (2026-06-12, codex): Optuna sampler seed=42 — HPO 탐색까지 재현 가능.
            #        (이전: seed=None 난수. 최종 모델 학습은 항상 seed=42.)
            # : per_model_trials 예산 조회 (LightGBM D=9 → 60 권장)
            from simulation.models._optuna_budget import get_trials as _get_trials
            from simulation.models._optuna_pruners import get_best_pruner_for as _get_pruner
            _n_trials = _get_trials("LightGBM", default=20)
            _pp_k = os.environ.get("MPH_INNER_HP_PREPROC_TRIALS")   # G-273c-B: Stage-1 preproc = 축소 탐색
            if _pp_k:
                _n_trials = max(2, int(_pp_k))   # transform 비교용 작은 HP 탐색(full 추종)
            study = optuna.create_study(direction="minimize",  # G-257: WIS (lower=better)
                                        sampler=optuna.samplers.TPESampler(seed=42),  # G-257: HPO 탐색도 재현 (codex)
                                        pruner=_get_pruner("LightGBM"))
            # G-161 (2026-05-02): trial cleanup callback + gc_after_trial 강제.
            from simulation.models._optuna_torch import make_trial_cleanup_callback
            study.optimize(objective, n_trials=_n_trials,
                           callbacks=[make_trial_cleanup_callback("LightGBM")],
                           gc_after_trial=True, show_progress_bar=False)
            best_params = study.best_params
            best_score = study.best_value
            # : study 객체 즉시 삭제 → OOM 방지
            del study
            gc.collect()
        else:
            # ── Fallback: 기존 6-combo grid search ──
            best_score = np.inf   # G-257: WIS minimize
            best_params = {}
            param_candidates = [
                {"num_leaves": 20, "learning_rate": 0.03, "reg_alpha": 0.3, "reg_lambda": 1.5, "min_child_samples": 8},
                {"num_leaves": 31, "learning_rate": 0.03, "reg_alpha": 0.5, "reg_lambda": 2.0, "min_child_samples": 10},
                {"num_leaves": 15, "learning_rate": 0.05, "reg_alpha": 0.1, "reg_lambda": 1.0, "min_child_samples": 5},
                {"num_leaves": 25, "learning_rate": 0.02, "reg_alpha": 0.3, "reg_lambda": 2.0, "min_child_samples": 8},
                {"num_leaves": 31, "learning_rate": 0.05, "reg_alpha": 0.5, "reg_lambda": 1.0, "min_child_samples": 5},
                {"num_leaves": 20, "learning_rate": 0.02, "reg_alpha": 1.0, "reg_lambda": 3.0, "min_child_samples": 10},
            ]
            for params in param_candidates:
                fold_scores = []
                for train_idx, val_idx in tscv.split(X_train):
                    X_tr, X_val = X_train[train_idx], X_train[val_idx]
                    y_tr, y_val = y_train[train_idx], y_train[val_idx]
                    dtrain = lgb.Dataset(X_tr, label=y_tr)
                    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
                    full_params = {
                        "objective": "regression", "metric": "rmse", "verbose": -1,
                        "subsample": 0.7, "colsample_bytree": 0.6, "seed": 42, **params}
                    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
                    model = lgb.train(full_params, dtrain, num_boost_round=200,  # 800→200 cap (early_stop 50 그대로)
                                      valid_sets=[dval], callbacks=callbacks)
                    pred = model.predict(X_val)
                    fold_scores.append(_fold_wis(y_val, pred, y_tr))
                avg_wis = np.mean(fold_scores)
                if avg_wis < best_score:
                    best_score = avg_wis
                    best_params = params

        # 최종 학습
        val_size = max(10, int(len(y_train) * 0.15))
        dtrain = lgb.Dataset(X_train[:-val_size], label=y_train[:-val_size])
        dval = lgb.Dataset(X_train[-val_size:], label=y_train[-val_size:], reference=dtrain)

        final_params = {
            "objective": "regression", "metric": "rmse", "verbose": -1,
            "seed": 42, **best_params,
        }
        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
        self._model = lgb.train(final_params, dtrain, num_boost_round=200,  # 1200→200 cap (early_stop 50 그대로)
                                valid_sets=[dval], callbacks=callbacks)
        self._fitted = True
        log.info(f"  [LightGBM] best_params={best_params}, CV WIS={best_score:.4f}")
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        # G-298 (2026-06-17): NO transformed-space floor — trees are train-range-bounded, so the
        #   wrong-space np.maximum(pred,0) floored sub-median predictions to the median under
        #   mcmc_robust/laplace. Inverse maps the bounded range to [min_y, max_y] ≥ 0. See XGBoost.
        return self._model.predict(X_test)


# ═══════════════════════════════════════════════════════════════
# 3. Random Forest -- Level 8
# ═══════════════════════════════════════════════════════════════

class RandomForestForecaster(BaseForecaster):
    """Random Forest -- 배깅 기반 트리 앙상블, 보수적 depth."""

    meta = ModelMeta(
        name="RandomForest",
        category="tree",
        level=8,
        min_data=60,
        description="Random Forest. 보수적 max_depth + min_samples로 과적합 방지.",
        dependencies=["sklearn"],
    )

    def __init__(self):
        super().__init__()
        self._model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> RandomForestForecaster:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            use_optuna = True
        except ImportError:
            use_optuna = False

        tscv = TimeSeriesSplit(n_splits=3)

        # G-273d (2026-06-15, 사용자 "통일-Optuna에서 시작했다"): RF 를 XGBoost/LightGBM 과 동일한
        #   인라인 Optuna study 로 통일. 배경: 4/27 "Tree 모델 HP suggester 표준화"가 suggest_randomforest_hp
        #   (7-HP, tpe-mv)까지 만들었으나 forecaster 배선이 끊겨 RF 만 GridSearchCV 에 남았던 간극 복원
        #   (get_hp_suggester 디스패처 dead). HP 공간 = suggest_randomforest_hp 7-HP(n_est 는 perf
        #   591366d "200~ 충분" 반영해 250 cap). fast-path 도 3-모드로 통일(XGB/LGB 와 동일 코드패턴).
        _fast_hp = os.environ.get("MPH_INNER_HP_FAST") == "1"

        if _fast_hp:
            # 비교 단계(feature/mc): 단일 reasonable default fit (study 생략, feature 선택 full 과 동일 입증)
            best_params = {"n_estimators": 200, "max_depth": 10, "min_samples_split": 2,
                           "min_samples_leaf": 3, "max_features": "sqrt", "bootstrap": True}
            best_score = float("nan")
        elif use_optuna:
            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 250),  # perf 591366d: 200~ 충분
                    "max_depth": trial.suggest_int("max_depth", 3, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                    "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.7, 1.0]),
                    "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                }
                fold_scores = []
                for i, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
                    X_tr, X_val = X_train[train_idx], X_train[val_idx]
                    y_tr, y_val = y_train[train_idx], y_train[val_idx]
                    m = RandomForestRegressor(random_state=42, n_jobs=2, **params)
                    m.fit(X_tr, y_tr)
                    pred = m.predict(X_val)
                    fold_scores.append(_fold_wis(y_val, pred, y_tr))
                    del m
                    _mean_wis = float(np.mean(fold_scores))
                    trial.report(_mean_wis, i)   # fold-level pruning (XGB/LGB 동형)
                    if trial.should_prune():
                        raise optuna.TrialPruned(
                            f"RandomForest pruned at fold {i} (WIS={_mean_wis:.3f})")
                gc.collect()
                return np.mean(fold_scores)

            from simulation.models._optuna_budget import get_trials as _get_trials
            from simulation.models._optuna_pruners import get_best_pruner_for as _get_pruner
            _n_trials = _get_trials("RandomForest", default=20)
            _pp_k = os.environ.get("MPH_INNER_HP_PREPROC_TRIALS")   # G-273c-B: Stage-1 preproc = 축소 탐색
            if _pp_k:
                _n_trials = max(2, int(_pp_k))   # transform 비교용 작은 HP 탐색(full 추종)
            from simulation.models._optuna_torch import make_trial_cleanup_callback
            study = optuna.create_study(direction="minimize",  # WIS (lower=better)
                                        sampler=optuna.samplers.TPESampler(seed=42),  # 재현
                                        pruner=_get_pruner("RandomForest"))
            study.optimize(objective, n_trials=_n_trials,
                           callbacks=[make_trial_cleanup_callback("RandomForest")],
                           gc_after_trial=True, show_progress_bar=False)
            best_params = study.best_params
            best_score = study.best_value
            del study
            gc.collect()
        else:
            # ── Fallback (optuna 부재): 기존 6-combo grid (G-039 nested n_jobs 금지) ──
            grid = GridSearchCV(
                RandomForestRegressor(random_state=42, n_jobs=2),
                {"n_estimators": [200], "max_depth": [5, 7, 10], "min_samples_leaf": [3, 5]},
                cv=tscv, scoring="neg_mean_squared_error", n_jobs=1,
            )
            grid.fit(X_train, y_train)
            best_params = grid.best_params_
            best_score = float("nan")

        # 전체 train 으로 최종 학습 (Stage-3 final = full HP; 플래그 미설정)
        self._model = RandomForestRegressor(
            random_state=42, n_jobs=2, **best_params).fit(X_train, y_train)
        self._fitted = True
        log.info(f"  [RandomForest] best_params={best_params}, CV WIS={best_score:.4f}")
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        # G-298 (2026-06-17): NO transformed-space floor — bagging is train-range-bounded, so the
        #   wrong-space np.maximum(pred,0) floored sub-median predictions to the median under
        #   mcmc_robust/laplace. Inverse maps the bounded range to [min_y, max_y] ≥ 0. See XGBoost.
        return self._model.predict(X_test)


# GradientBoostingForecaster removed 2026-05-26 (Sprint D1, MERGE-drop):
# 사용자 명시: "MERGE-drop한 모델들은 다 없애버려"
# Reason: XGBoost / LightGBM 가 sklearn GBM 보다 우수 → GradientBoosting 중복.


# ═══════════════════════════════════════════════════════════════
# 5. CatBoost -- Level 8 (G-161 fix: tree 카테고리 list 5/5 채우기)
# ═══════════════════════════════════════════════════════════════
# 사용자 지적 (2026-05-02): "CatBoost 누락 — 같은 패턴 G-156 (GradientBoosting)"
# train_by_category.sh:59 tree 카테고리 list 에 CatBoost 있는데 REGISTRY 부재.
# _optuna_samplers.py:732 suggest_catboost_hp 만 있고 모델 클래스 자체 없음.
# G-161 fix: CatBoostForecaster 클래스 + REGISTRY.register 추가 → 5/5 모두 학습.

class CatBoostForecaster(BaseForecaster):
    """CatBoost — symmetric oblivious tree, ordered boosting (Prokhorenkova 2018)."""

    meta = ModelMeta(
        name="CatBoost",
        category="tree",
        level=8,
        min_data=80,
        description="CatBoost. ordered boosting + symmetric trees + early stopping.",
        dependencies=["catboost"],
    )

    def __init__(self):
        super().__init__()
        self._model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> "CatBoostForecaster":
        """CatBoost fit with TimeSeriesSplit + Optuna HPO (G-169, G-161, D-4).

        Optuna HPO (`suggest_catboost_hp`, 9 HP) over 3-fold TimeSeriesSplit.
        Trial cleanup callback + gc_after_trial 강제 (G-161 — in-process trial
        메모리 회수). Optuna 미설치 시 default HP fallback.

        Args:
            X_train: feature matrix (n, p) — `_validate_shapes` 자동 (G-166).
            y_train: target (n,) — ILI rate, ≥0 권장.
            **kwargs: 미사용 (BaseForecaster 인터페이스 호환).

        Returns:
            self (chain용).

        Raises:
            ValueError: shape mismatch (`_validate_shapes` 통해, G-166).
            ImportError: catboost 미설치 시.

        Performance:
            - Optuna 20 trial × 3-fold = 60 fit ≈ 8-12분 (depth 2-10).
            - Final fit (iterations 최대 1200, early stop 50) ≈ 30-60초.
            - Memory peak ~500MB (n=242, depth=10).

        Side effects:
            - log.info: best_params + CV R²
            - allow_writing_files=False (CatBoost cache 파일 생성 X)

        Caller responsibility:
            - X / y shape 일치 (G-166 자동 검증).
            - n ≥ 80 (min_data, ModelMeta).

        See: G-169 (CatBoost 신규 등록), G-161 (cleanup callback),
             `_optuna_samplers.suggest_catboost_hp` (9 HP space).
        """
        from catboost import CatBoostRegressor
        from sklearn.model_selection import TimeSeriesSplit

        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            use_optuna = True
        except ImportError:
            use_optuna = False

        tscv = TimeSeriesSplit(n_splits=3)

        if use_optuna:
            from simulation.models._optuna_samplers import suggest_catboost_hp

            def objective(trial):
                params = suggest_catboost_hp(trial)
                # iterations 는 early stopping 으로 cut, 800 cap
                params["iterations"] = min(int(params.get("iterations", 200)) * 2, 400)  # 800→400 cap (early_stop 50 그대로)
                fold_scores = []
                for i, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
                    X_tr, X_val = X_train[train_idx], X_train[val_idx]
                    y_tr, y_val = y_train[train_idx], y_train[val_idx]
                    model = CatBoostRegressor(**params, allow_writing_files=False)
                    model.fit(X_tr, y_tr, eval_set=(X_val, y_val),
                              early_stopping_rounds=50, verbose=False)
                    pred = model.predict(X_val)
                    fold_scores.append(_fold_wis(y_val, pred, y_tr))
                    del model
                    # Cat 1 (2026-05-12): fold-level pruning (CatBoost)
                    _mean_wis = float(np.mean(fold_scores))
                    trial.report(_mean_wis, i)   # G-257: WIS (minimize); should_prune respects direction
                    if trial.should_prune():
                        raise optuna.TrialPruned(
                            f"CatBoost pruned at fold {i} (WIS={_mean_wis:.3f})")
                gc.collect()
                return float(np.mean(fold_scores))

            from simulation.models._optuna_budget import get_trials as _get_trials
            from simulation.models._optuna_pruners import get_best_pruner_for as _get_pruner
            _n_trials = _get_trials("CatBoost", default=20)
            # G-273d (2026-06-15): 비교 단계(preproc/fast) trial 축소 가드 — landmine 제거(CatBoost=
            #   DEFER 라 cb_* param-mapping 복잡성 회피 위해 단일-fit 대신 최소 2-trial study).
            _pp_k = os.environ.get("MPH_INNER_HP_PREPROC_TRIALS") or (
                "2" if os.environ.get("MPH_INNER_HP_FAST") == "1" else None)
            if _pp_k:
                _n_trials = max(2, int(_pp_k))
            study = optuna.create_study(direction="minimize",  # G-257: WIS (lower=better)
                                        sampler=optuna.samplers.TPESampler(seed=42),  # G-257: HPO 탐색도 재현 (codex)
                                        pruner=_get_pruner("CatBoost"))
            # G-161: trial cleanup callback + gc_after_trial 강제.
            from simulation.models._optuna_torch import make_trial_cleanup_callback
            study.optimize(objective, n_trials=_n_trials,
                           callbacks=[make_trial_cleanup_callback("CatBoost")],
                           gc_after_trial=True, show_progress_bar=False)
            best_params = study.best_params
            best_score = study.best_value
            del study
            gc.collect()
            # suggest_catboost_hp 의 cb_* prefix 를 CatBoost API 키로 매핑
            cb_key_map = {"cb_iter": "iterations", "cb_depth": "depth",
                          "cb_lr": "learning_rate", "cb_l2": "l2_leaf_reg",
                          "cb_border": "border_count",
                          "cb_bag_temp": "bagging_temperature",
                          "cb_rand_str": "random_strength",
                          "cb_grow": "grow_policy", "cb_od": "od_type"}
            best_params = {cb_key_map.get(k, k): v for k, v in best_params.items()}
            best_params["iterations"] = min(int(best_params.get("iterations", 200)) * 2, 400)  # 1200→400 cap (early_stop 50 그대로)
        else:
            best_params = {
                "iterations": 500, "depth": 6, "learning_rate": 0.05,
                "l2_leaf_reg": 3.0, "border_count": 128,
                "bagging_temperature": 0.5, "random_strength": 1.0,
                "grow_policy": "SymmetricTree",
            }
            best_score = float("nan")

        # Final fit on full train with early-stop val (last 15%)
        val_size = max(10, int(len(y_train) * 0.15))
        X_tr, X_es = X_train[:-val_size], X_train[-val_size:]
        y_tr, y_es = y_train[:-val_size], y_train[-val_size:]
        self._model = CatBoostRegressor(
            **best_params, random_seed=42, verbose=False,
            allow_writing_files=False,
        )
        self._model.fit(X_tr, y_tr, eval_set=(X_es, y_es),
                        early_stopping_rounds=50, verbose=False)
        self._fitted = True
        log.info(f"  [CatBoost] best_params={best_params}, CV WIS={best_score:.4f}")
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        """CatBoost predict — non-negative clip 강제 (ILI rate ≥ 0).

        Args:
            X_test: feature matrix (m, p) — fit 시 X_train 의 p 와 일치.
            **kwargs: 미사용.

        Returns:
            np.ndarray (m,) — non-negative ILI rate prediction.

        Raises:
            AttributeError: model 미학습 시 (`self._model is None`).

        Performance: O(m × tree_depth) — m=68, depth=10 ≈ 5ms.
        Side effects: 없음 (pure inference).
        Caller responsibility: ILI rate ≥ 0 도메인 제약은 `np.maximum(_, 0)` 강제
                              (D-5 gray-box boundary — sanitize_predictions 는 NaN/inf
                              만, 음수는 보존이라 별도 적용).

        See: G-169 (CatBoost 신규), G-159 (sanitize_predictions 와 layer 차이).
        """
        return np.maximum(self._model.predict(X_test), 0)


# ── 등록 ──
REGISTRY.register(XGBoostForecaster)
REGISTRY.register(LightGBMForecaster)
REGISTRY.register(RandomForestForecaster)
REGISTRY.register(CatBoostForecaster)
# GradientBoostingForecaster removed 2026-05-26 (Sprint D1, MERGE-drop)
