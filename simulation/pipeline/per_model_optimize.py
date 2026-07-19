"""
R9: Per-Model Individual Optimization
============================================

For each registered model, search over its OWN optimal configuration:
  • Target transform (identity / log1p / boxcox / yeo-johnson)
  • Feature subset (uses existing Optuna per-model selection if available)
  • Scaling strategy (none / standard / robust / quantile)
  • Hyperparameters (existing Optuna per-model HP space)

Goal: every model evaluated at its INDIVIDUAL BEST configuration, not at a
uniform pipeline preset. This addresses the methodological concern that
some models (e.g., NegBin GLM, Bayesian models) need different preprocessing
than others (e.g., XGBoost, DNN) — uniform configs systematically penalize
some model families.

Output flows into R10 (per_model_eval): the leaderboard reflects
"each model's best vs other models' bests" rather than "best uniform-config
model among M".

Strategy
--------
For each model M:
  1. Load cached Optuna feature selection for M (if any).
  2. Enumerate target_transforms × scalers (≤4 × ≤3 = 12 cells).
  3. For each cell, run a SHORT Optuna HP search on WF-CV (n_trials configurable,
     default 20 per cell to keep runtime manageable).
  4. Pick the cell+HP with lowest validation WIS.
  5. Persist best config: simulation/results/per_model_optimal/<M>.json
  6. R10 (per_model_eval) uses these configs to refit each model with its own best
     before computing the unified leaderboard.

This is computationally heavy. For thesis-scale runs (~50 models × 12 cells ×
20 trials = 12,000 fits), expect 4-8 hours on CPU. CLI:

  --per-model-optimize         enable R9 (per_model_optimize)
  --per-model-trials N         Optuna trials per cell (default 20)
  --per-model-transforms ...   transforms to search (default: identity,log1p,boxcox)
  --per-model-scalers ...      scalers to search (default: none,robust)

Reference: this is a structured form of "automated model selection per
model family" common in AutoML (see auto-sklearn, FLAML, TPOT). The novelty
here is applying it to epidemic forecasting where target transform choice
materially affects WIS.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Module-level GLOBAL import (Codex audit 2026-05-27 fix):
# L1281 의 optimize_one_model 가 GLOBAL.filter.* 사용. 종래 L1894 의
# function-level import (_phase12_consolidate 안에서만 binding) 는
# 다른 함수에서 NameError 발생 가능 — module scope 로 hoist.
from simulation.config_global import GLOBAL, Z95  # noqa: E402
from simulation.pipeline.training_history import save_training_record
from simulation.utils.paths import get_results_dir as _get_results_dir

log = logging.getLogger(__name__)

# G-329: small tail-amplification surcharge for asinh y-transforms in the OOF objective.
_EXTRAP_AMP_ALPHA = 0.05
_EXTRAP_AMP_THRESHOLD = 2.0


# ─── Search space constants — hierarchical 의 menu 를 단일 source of truth 로 사용 ───
# 2026-05-26: 중복 menu (R9 per_model_optimize + hierarchical) 통일 — hierarchical 만 정의.
# 이전: R9(per_model_optimize) 가 DEFAULT_TARGET_TRANSFORMS / DEFAULT_SCALERS 별도 정의 (~100줄)
# 현재: hierarchical 의 METRIC_Y_TRANSFORMS + CATEGORICAL_Y_TRANSFORMS + identity 조합
from simulation.pipeline.preproc_optuna_hierarchical import (
    METRIC_Y_TRANSFORMS as _HIER_METRIC_Y,
    CATEGORICAL_Y_TRANSFORMS as _HIER_CATEGORICAL_Y,
    STABLE_Y_TRANSFORMS as _HIER_STABLE_Y,
    METRIC_X_SCALERS as _HIER_METRIC_X,
    _apply_single_y_transform,
    _build_single_x_scaler,
)

# Full grid: identity + 7 metric (log1p/sqrt/asinh/mcmc_robust/laplace/anscombe/freeman_tukey) + 2 categorical (boxcox/yeo_johnson)
# G-254: rank/arcsine_sqrt/gaussian removed (train-bounded y-target — see preproc_optuna_hierarchical.py)
DEFAULT_TARGET_TRANSFORMS = ("identity",) + tuple(_HIER_METRIC_Y) + tuple(_HIER_CATEGORICAL_Y)
# Default scaler menu (legacy _evaluate_config flat grid).
# Note: hierarchical 의 categorical 모드는 trial path 에서 별도로 quantile/grouped 도 다룸.
DEFAULT_SCALERS = ("none",) + tuple(_HIER_METRIC_X)   # ("none", "standard", "robust", "quantile")

# G-181 (2026-05-05) + G-133/G-146 (safety):
#   MPH_STABLE_TRANSFORMS=1 → thesis/paper stable subset.
#   identity, log1p, sqrt, asinh, laplace.
#   (mcmc_robust / anscombe / freeman_tukey / boxcox / yeo_johnson excluded from active search)
#
# CAVEAT (사용자 지적 2026-05-26): STABLE_TRANSFORMS gate 는 historical fix —
#   small-sample (n=27 val) 에서 boxcox/yeo_johnson 의 lambda 추정 발산 사건 회피.
#   코드 자체에는 legacy primitives/replay support 가 보존되지만, STABLE_TRANSFORMS gate 는
#   flat-grid DEFAULT_TARGET_TRANSFORMS 와 hierarchical y/x mode sampling 을 모두 제한한다.
#   다른 데이터 / 큰 n 사용 시 → MPH_STABLE_TRANSFORMS=0 으로 release 권고
#     → Optuna 가 9 transform (7 METRIC_Y + 2 CATEGORICAL_Y) 모두 자유 sample.
#   장기 sprint candidate: sample-size 기반 자동 gating (e.g. n>50 → STABLE auto off).
if GLOBAL.training.stable_transforms:
    DEFAULT_TARGET_TRANSFORMS = ("identity",) + tuple(_HIER_STABLE_Y)


# ════════════════════════════════════════════════════════════════
# Model-aware Preprocessing Menu (2026-04-29 신규)
# ────────────────────────────────────────────────────────────────
# 사용자 통찰: "전처리 작업이 모델별로 달라야 한다"
# 이론적 근거 (학술 출처):
#   · Tree (Breiman 1984): split-based, monotonic transform 무관
#   · Linear (Friedman ESL §3.4): scale 정규화 필수
#   · Kernel (Schölkopf SVM): kernel distance 정규화 필수
#   · DL (Goodfellow §8.7): gradient flow 위해 표준화
#   · GLM (McCullagh-Nelder §2): link function 이 Y transform 내장
#   · GAM (Hastie-Tibshirani §3.2): smoothing spline 내장
#   · ARIMA (Box-Jenkins 1976): 자체 boxcox + AIC, 외부 transform 충돌
#   · Mechanistic (SEIR/PINN-large): X 안 받음
# ════════════════════════════════════════════════════════════════


# 카테고리별 적합한 preprocessing menu
# G-181 (2026-05-05) — 사용자 grill: "이전에 미리 조치하지 않았어? 왜 안되어있는데?"
#   원인: 이 menu가 모델별 transforms_y 를 1-2개로 제한 (e.g. NegBinGLM=identity 만)
#         → 학습 시 mcmc_robust/sqrt 등 새 transforms 가 시도조차 안 됨.
#   fix: 카테고리별 transforms_y 에 safe transforms (sqrt/asinh/rank/mcmc_robust/laplace) 추가.
#         학술 근거 유지하면서 더 많은 transform 시도 가능 (Optuna가 best 자동 선택).
# v19 (사용자 명시 2026-05-22): "Phase B 도 Phase A 와 비슷한 preproc Optuna 적용"
# → 11 카테고리 모두 동일 Y/X menu (Optuna 가 모델별 best 자율 학습 → 도메인 추측 X)
#
# 통일 menu (UNIFIED, 모든 모델 동일):
#   Y transforms (7 STABLE-safe, identical to DEFAULT_TARGET_TRANSFORMS with STABLE=1):
#     identity / log1p / sqrt / asinh / rank / mcmc_robust / laplace
#   X scalers (4 unified):
#     none / standard / robust / grouped
#
# 학술 정직성: per-model 도메인 추측 (예: "GLM 은 link 내장이라 Y transform 신중") 대신
#              Optuna 가 데이터 기반으로 best 선택. 통일성 + 재현성 우선.
# 2026-05-23: flat 7×4 grid (_UNIFIED_TRANSFORMS_Y × _UNIFIED_SCALERS_X) 제거.
# preproc_optuna_hierarchical.py 의 4-mode hierarchical Optuna (none/individual/group/categorical)
# 로 대체. get_model_preproc_menu() 도 함께 제거.


def _oof_regime_aggregate(wis_scores, fold_maxes=None, y_train=None) -> float:
    """OOF fold WIS 집계를 **regime-conditional mean 으로 통일** (G-265b, 2026-06-13, 3자 리뷰).

    배경: G-256b(2026-06-12)가 inline 경로의 OOF 집계를 median→regime-conditional mean 으로 바꿔
    (median 이 outbreak fold ~2/5 를 버려 peak-blind 선택하던 것 차단) peak 캠페인을 완성했으나,
    **per_model_optimize 의 champion 선택 경로 3곳(1-SE OOF helper·mc-probe·config OOF)은 여전히
    np.median(=D5 2026-05-30) 사용** → champion 선택이 peak-blind 로 남는 불일치 (codex+gemini 적발).
    본 helper 로 통일 → champion 선택도 peak-aware (G-256b 전 경로 일관).

    Args:
        wis_scores: fold 별 WIS (낮을수록 좋음).
        fold_maxes: fold 별 y_val max — regime(quiet/outbreak) 분류용. None → mean fallback.
        y_train: outbreak_level(75pct) 산출용. None → mean fallback.

    Returns:
        regime-conditional mean = 0.5·mean(quiet)+0.5·mean(elevated) (한 regime 뿐이면 mean).
        wis_scores 비면 inf.

    Side effects: none.
    """
    if not wis_scores:
        return float("inf")
    from simulation.pipeline._inline_optuna_3stage import _aggregate_oof_folds
    ol = None
    if y_train is not None and len(y_train) > 0:
        ol = float(np.percentile(np.asarray(y_train, dtype=float), 75))
    return _aggregate_oof_folds(list(wis_scores),
                                list(fold_maxes) if fold_maxes else None, ol)


def _real_wis_and_residuals(model, X_train_s, y_train, y_val, y_pred,
                            inv_fn, sigma_for_wis, mae, calib_residuals=None):
    """Calibration-aware WIS — empirical split-conformal PI from OOS (preferred) residuals.

    The selection objective minimised by Optuna. Replaces the degenerate fixed-σ Gaussian
    WIS (σ=std(y_train) was model-INDEPENDENT → WIS collapsed to point-MAE ranking; and in the
    hierarchical path `weighted_interval_score` was never even in scope → silent NameError →
    wis=mae for every trial). codex+gemini converged: empirical split-conformal WIS (Lei 2018)
    from the model's own train residuals — point accuracy + sharpness + calibration in one
    scalar, the paper-primary FluSight metric, and no fragile transform-space σ.

    Args:
        model: a FITTED estimator (already fit on X_train_s); ``.predict`` is called once more
            on X_train_s to get in-sample predictions.
        X_train_s: scaled train design matrix (n_train × p), same space the model was fit on.
        y_train: ORIGINAL-space train target (n_train,).
        y_val, y_pred: original-space val target + point forecast (n_val,).
        inv_fn: inverse-Y transform (train-fitted) mapping model output → original space.
        sigma_for_wis: fixed σ, used ONLY for the documented fallback.
        mae: precomputed val MAE, the last-resort fallback.

    Returns:
        (wis, resid_train): wis = float; resid_train = finite original-space train residuals
        (ndarray, ≥2 finite when empirical WIS was used; possibly empty on the fallback path).
        The caller reuses resid_train for the gate's empirical PICP.

    Side effects: one extra ``model.predict(X_train_s)`` (per-row; one forward pass for DNNs).
    Caller responsibility: residuals are TRAIN-only (disjoint from val) → no leakage.
    """
    from simulation.analytics.diagnostics import (
        weighted_interval_score, weighted_interval_score_empirical,
    )
    from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS
    # Q1 (codex+gemini 2026-05-30): PREFER OOS calibration residuals (prior-fold OOF, threaded
    # by the WF-CV loop) — leakage-free + realistic. In-sample train residuals (the fallback)
    # are OPTIMISTIC (model fits its own rows → PI too narrow → over-confident). Used only for
    # fold 0 / non-CV callers / when <2 finite OOF residuals.
    resid = None
    if calib_residuals is not None:
        resid = np.asarray(calib_residuals, dtype=float).ravel()
        resid = resid[np.isfinite(resid)]
    if resid is None or resid.size < 2:
        try:
            y_pred_train = np.asarray(inv_fn(model.predict(X_train_s)), dtype=float).ravel()
            resid = np.asarray(y_train, dtype=float).ravel() - y_pred_train
            resid = resid[np.isfinite(resid)]
        except Exception:
            resid = np.asarray([], dtype=float)
    if resid.size >= 2:
        try:
            wis = float(np.mean(weighted_interval_score_empirical(
                y_val, y_pred, residuals=resid, alphas=FLUSIGHT_ALPHAS)))
            if np.isfinite(wis):
                return wis, resid
        except Exception:
            pass
    # Documented fallback (rare: train-predict failed / <2 finite residuals): fixed-σ
    # Gaussian WIS — NOT a silent swallow (the empirical path above is the contract).
    try:
        wis = float(np.mean(weighted_interval_score(
            y_val, y_pred, sigma_for_wis, alphas=FLUSIGHT_ALPHAS)))
    except Exception:
        wis = mae
    return wis, resid


def _in_range_fold_metrics(y_train, y_val, y_pred, residuals, fallback_wis: float) -> dict:
    """OOF fold metrics on validation rows whose target is within that fold's train range."""
    yt = np.asarray(y_train, dtype=float).ravel()
    yv = np.asarray(y_val, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    if yt.size == 0 or yv.size == 0 or yp.size != yv.size:
        return {"wis_in_range": float(fallback_wis), "r2_in_range": float("nan"),
                "n_in_range": 0}
    cap = float(np.nanmax(yt))
    mask = np.isfinite(yv) & np.isfinite(yp) & (yv <= cap + 1e-12)
    n_ir = int(np.sum(mask))
    if n_ir < 2:
        return {"wis_in_range": float(fallback_wis), "r2_in_range": float("nan"),
                "n_in_range": n_ir}

    yv_ir, yp_ir = yv[mask], yp[mask]
    resid = np.asarray(residuals if residuals is not None else [], dtype=float).ravel()
    resid = resid[np.isfinite(resid)]
    wis_ir = float(fallback_wis)
    try:
        from simulation.analytics.diagnostics import weighted_interval_score_empirical
        from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS
        if resid.size >= 2:
            wis_ir = float(np.mean(weighted_interval_score_empirical(
                yv_ir, yp_ir, residuals=resid, alphas=FLUSIGHT_ALPHAS)))
        else:
            wis_ir = float(np.mean(np.abs(yp_ir - yv_ir)))
    except Exception:
        wis_ir = float(np.mean(np.abs(yp_ir - yv_ir)))

    ss_tot = float(np.sum((yv_ir - float(np.mean(yv_ir))) ** 2))
    r2_ir = (1.0 - float(np.sum((yp_ir - yv_ir) ** 2)) / ss_tot
             if ss_tot > 1e-12 else float("nan"))
    return {"wis_in_range": wis_ir, "r2_in_range": r2_ir, "n_in_range": n_ir}


def _oof_selection_score(cell: dict) -> float:
    """Selection score for one OOF fold: in-range WIS composite plus bounded R2 guard."""
    import os as _os
    try:
        wis = float(cell.get("wis", float("inf")))
    except (TypeError, ValueError):
        wis = float("inf")
    if not np.isfinite(wis):
        return float("inf")
    try:
        alpha = float(_os.environ.get("MPH_OOF_IN_RANGE_ALPHA", "0.7"))
    except (TypeError, ValueError):
        alpha = 0.7
    alpha = min(1.0, max(0.0, alpha))
    try:
        wis_ir = float(cell.get("wis_in_range", wis))
    except (TypeError, ValueError):
        wis_ir = wis
    wis_ir = wis_ir if np.isfinite(wis_ir) else wis
    score = alpha * wis_ir + (1.0 - alpha) * wis

    try:
        floor = float(_os.environ.get("MPH_OOF_IN_RANGE_R2_FLOOR", "0.90"))
        beta = float(_os.environ.get("MPH_OOF_IN_RANGE_R2_PENALTY", "0.25"))
        max_gap = float(_os.environ.get("MPH_OOF_IN_RANGE_R2_MAX_GAP", "1.0"))
    except (TypeError, ValueError):
        floor, beta, max_gap = 0.90, 0.25, 1.0
    r2_ir = cell.get("r2_in_range", float("nan"))
    n_ir = int(cell.get("n_in_range", 0) or 0)
    if beta > 0 and n_ir >= 4 and np.isfinite(r2_ir) and float(r2_ir) < floor:
        gap = min(max(floor - float(r2_ir), 0.0), max(0.0, max_gap))
        score += max(abs(score), 1.0) * beta * gap
    return float(score)


def _fold_variance_penalize(score: float, fold_scores: list[float]) -> float:
    """Optional OOF fold-variance penalty; default is a small coefficient."""
    import os as _os
    try:
        coef = float(_os.environ.get("MPH_OOF_WIS_VAR_PENALTY", "0.05"))
    except (TypeError, ValueError):
        coef = 0.05
    vals = np.asarray([v for v in fold_scores if np.isfinite(v)], dtype=float)
    if coef <= 0 or vals.size < 2 or not np.isfinite(score):
        return float(score)
    denom = max(abs(float(np.mean(vals))), 1e-9)
    return float(score) * (1.0 + coef * float(np.std(vals, ddof=1)) / denom)


def _rolling_or_static_predict_oof(model, X_val_s, y_train, y_val, y_train_t):
    """G-337: sequence 모델 OOF 선택도 rolling-origin 1-step(공정) — preproc/mc/feature/HP 가 static
    collapse 기준이 아니라 최종 eval(_refit_and_predict_test)과 동일 1-step 기준으로 고르게 한다.

    문제: static multi-step OOF 는 sequence 모델(classic-ts·foundation·pf)을 collapse 시켜 transform/
    feature/HP 를 잘못 선택. feature 모델은 lag 로 이미 1-step 이라 무관.
    해법: y_observed = transform(y_val). train→y_train_t 의 affine map(polyfit deg-1)으로 변환 —
        identity(y_train_t==y_train)면 (1,0)=raw, affine transform(N-BEATS=mcmc_robust/TiDE=laplace)이면 정확.
    대상: supports_rolling_eval(classic-ts/epi/FusedEpi/N-HiTS) ∪ supports_transform_rolling(N-BEATS/TiDE).
    그 외 = static. 실패 시 static fallback(절대 안 깨짐).

    Args: model(fitted), X_val_s, y_train(raw), y_val(raw), y_train_t(model 학습공간).
    Returns: y_pred_t (transform 공간 — 호출자가 transform_inv 로 복원).
    """
    from simulation.models.base import supports_rolling_eval, supports_transform_rolling
    try:
        if (y_val is None or len(y_val) != len(X_val_s)
                or not (supports_rolling_eval(model) or supports_transform_rolling(model))):
            return model.predict(X_val_s)
        _yt = np.asarray(y_train, dtype=float).ravel()
        _ytt = np.asarray(y_train_t, dtype=float).ravel()
        if len(_yt) >= 2 and len(_yt) == len(_ytt):
            _a, _b = np.polyfit(_yt, _ytt, 1)          # train→transform affine (identity → (1,0)=raw)
            _y_obs_t = _a * np.asarray(y_val, dtype=float).ravel() + _b
        else:
            _y_obs_t = np.asarray(y_val, dtype=float).ravel()
        return model.predict(X_val_s, y_observed=_y_obs_t)
    except Exception:
        return model.predict(X_val_s)


def _evaluate_config(
    factory_fn,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    transform_name: str,
    scaler_name: str,
    feature_indices: Optional[list[int]] = None,
    sigma_for_wis: float = 1.0,
    feature_cols: Optional[list[str]] = None,   # 2026-04-28: grouped preproc
    optuna_trial: Optional[object] = None,       # 2026-04-29: preproc Optuna
    calib_residuals: Optional[np.ndarray] = None,  # Q1: prior-fold OOF residuals → PI calibration
    _fast_inner: bool = False,   # G-273c: comparison 단계(feature-stability/mc-probe) → tree 내부 HP study 생략
) -> dict:
    """Train a model with given (transform, scaler, feature_subset) on
    train, evaluate on val by WIS. Returns metrics.

    [2026-04-28] env MPH_GROUPED_PREPROC=1 시 feature_cols 기반 그룹별
    ColumnTransformer 사용 (글로벌 single-scaler 대안).

    [2026-04-29] optuna_trial 전달 + scaler_name='grouped_optuna' 시:
                  그룹별 (log_op, scale_op) 를 trial 에서 suggest 하는
                  build_grouped_preprocessor_optuna() 사용. MPH_PREPROC_OPTUNA=1
                  활성화.
    """
    # Feature subset
    if feature_indices is not None:
        X_train_use = X_train[:, feature_indices]
        X_val_use = X_val[:, feature_indices]
        # feature_cols subset
        feat_names_use = (
            [feature_cols[i] for i in feature_indices]
            if feature_cols is not None else None
        )
    else:
        X_train_use, X_val_use = X_train, X_val
        feat_names_use = feature_cols

    # Target transform
    y_train_t, transform_inv, _ = _apply_single_y_transform(y_train, transform_name)

    # 2026-04-28: Grouped preprocessor (env-gated)
    use_grouped = (
        GLOBAL.training.grouped_preproc
        and feat_names_use is not None
        and len(feat_names_use) == X_train_use.shape[1]
        and scaler_name in ("grouped", "robust", "standard")  # grouped 는 새 sentinel
    )
    # 2026-04-29: Optuna preproc 모드 (scaler_name == "grouped_optuna")
    if (scaler_name == "grouped_optuna"
            and optuna_trial is not None
            and feat_names_use is not None):
        try:
            from simulation.models.grouped_preprocessor import (
                build_grouped_preprocessor_optuna,
            )
            sc = build_grouped_preprocessor_optuna(feat_names_use, optuna_trial)
            X_train_s = sc.fit_transform(X_train_use)
            X_val_s = sc.transform(X_val_use)
        except Exception:
            # fallback: fixed grouped
            try:
                from simulation.models.grouped_preprocessor import build_grouped_preprocessor
                sc = build_grouped_preprocessor(feat_names_use)
                X_train_s = sc.fit_transform(X_train_use)
                X_val_s = sc.transform(X_val_use)
            except Exception:
                from sklearn.preprocessing import RobustScaler
                sc = RobustScaler()
                X_train_s = sc.fit_transform(X_train_use)
                X_val_s = sc.transform(X_val_use)
    elif use_grouped and scaler_name == "grouped":
        try:
            from simulation.models.grouped_preprocessor import build_grouped_preprocessor
            sc = build_grouped_preprocessor(feat_names_use)
            X_train_s = sc.fit_transform(X_train_use)
            X_val_s = sc.transform(X_val_use)
        except Exception:
            # fallback: robust
            from sklearn.preprocessing import RobustScaler
            sc = RobustScaler()
            X_train_s = sc.fit_transform(X_train_use)
            X_val_s = sc.transform(X_val_use)
    elif scaler_name in ("none", "standard", "robust", "quantile"):
        # 2026-05-26: hierarchical._build_single_x_scaler 로 통일 (Sprint 1.D)
        sc = _build_single_x_scaler(scaler_name)
        if sc is None:
            X_train_s, X_val_s = X_train_use, X_val_use
        else:
            X_train_s = sc.fit_transform(X_train_use)
            X_val_s = sc.transform(X_val_use)
    else:
        X_train_s, X_val_s = X_train_use, X_val_use

    # Fit & predict
    # G-273c (2026-06-15): _fast_inner=True (feature-stability / mc-probe 비교 단계에서 호출) 시
    #   tree forecaster 내부 HP study 생략(단일 reasonable default + early_stop). 비교 단계엔 충분
    #   (HP 풀튜닝은 Stage-3 최종 refit). _oof_cv_metrics·final 경로는 default False → full HP 유지.
    #   flag 는 이 fit 동안만, try/finally 로 복원(누수 0).
    import os as _os_ec
    model = factory_fn()
    _prev_fast_ec = _os_ec.environ.get("MPH_INNER_HP_FAST")
    if _fast_inner:
        _os_ec.environ["MPH_INNER_HP_FAST"] = "1"
    try:
        # G-291 (2026-06-17, 3자 감사): OOF fold 에 feature_names 전달 — OverseasTransfer encoder/DL lag1
        #   인덱스 탐색에 필요(미전달 시 transfer skip). 미사용 모델은 **kwargs 로 무시(무해).
        model.fit(X_train_s, y_train_t, feature_names=feat_names_use)
        # G-332b/G-337: rolling-eval/transform-rolling 모델의 OOF 도 최종 eval 과 동일 rolling 1-step
        #   (static multi-step 은 sequence 모델 collapse → transform/feature/HP 잘못 선택). 공유 헬퍼.
        y_pred_t = _rolling_or_static_predict_oof(model, X_val_s, y_train, y_val, y_train_t)
    except Exception as e:
        # P2 (R8 2026-05-28): trial 간 GPU mem 해제 (DNN VRAM 단편화 방지).
        del model
        from simulation.utils.memory_cleanup import cleanup_gpu_memory
        cleanup_gpu_memory()
        return {"wis": float("inf"), "mae": float("inf"), "error": str(e)}
    finally:
        if _fast_inner:
            if _prev_fast_ec is None:
                _os_ec.environ.pop("MPH_INNER_HP_FAST", None)
            else:
                _os_ec.environ["MPH_INNER_HP_FAST"] = _prev_fast_ec
    # G-298 (2026-06-17): ILI≥0 도메인 floor in ORIGINAL units (별도 단계 — sanitize 책임 X).
    #   Replaces the wrong-space np.maximum that trees applied BEFORE the inverse (which floored
    #   sub-median predictions to the median under mcmc_robust/laplace). Selection OOF must floor
    #   identically to _refit_and_predict_test so selection==eval.
    y_pred = np.maximum(np.asarray(transform_inv(y_pred_t), dtype=np.float64), 0.0)

    # G-231 (2026-05-22): α-blend 완전 제거 (사용자 명시 "안 사용한다").

    # Metrics
    err = y_pred - y_val
    mae = float(np.mean(np.abs(err)))
    # 2026-05-30 (B+Q1): real calibration-aware WIS — prior-fold OOF residuals (calib_residuals)
    # when the CV loop threads them, else the model's own in-sample train residuals.
    wis, resid_train = _real_wis_and_residuals(
        model, X_train_s, y_train, y_val, y_pred, transform_inv, sigma_for_wis, mae,
        calib_residuals=calib_residuals)
    _ir_metrics = _in_range_fold_metrics(y_train, y_val, y_pred, resid_train, wis)

    # G-FIX(2026-05-28): OOF 진단/selection 용 r2/mape/pi95_coverage (champion=best-WIS, gate 제거 2026-06-05).
    # 2026-05-30 (B: real-WIS): PICP95 도 model 자신의 train 잔차 95% empirical band 사용 →
    # 이전 fixed-σ band (model-INDEPENDENT) 제거. resid_train <2 finite 면 fixed-σ fallback.
    _yv = np.asarray(y_val, dtype=float).ravel()
    _ss_tot = float(np.sum((_yv - _yv.mean()) ** 2))
    _r2 = (1.0 - float(np.sum(err ** 2)) / _ss_tot) if _ss_tot > 1e-12 else float("nan")
    _nz = np.abs(_yv) > 1e-9
    _mape = (float(np.mean(np.abs(err[_nz] / _yv[_nz])) * 100.0)
             if _nz.any() else float("nan"))
    if resid_train.size >= 2:
        _q95 = float(np.quantile(np.abs(resid_train), 0.95))
        _pi95 = float(np.mean(np.abs(err) <= _q95))
    else:
        _pi95 = float(np.mean(np.abs(err) <= Z95 * max(float(sigma_for_wis), 1e-9)))

    _result = {
        "wis": wis,
        "mae": mae,
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": _r2,
        "mape": _mape,
        "pi95_coverage": _pi95,
        "transform": transform_name,
        "scaler": scaler_name,
        "n_features": X_train_use.shape[1],
        **_ir_metrics,
        # Q1: OOS val residuals → next WF fold's PI calibration (transient, "_" = not serialized)
        "_val_residuals": err,
    }
    _result["selection_wis"] = _oof_selection_score(_result)
    # P2 (R8 2026-05-28): trial 간 GPU mem 해제 (DNN VRAM 단편화 방지). 결과는 float 만 보유.
    del model
    from simulation.utils.memory_cleanup import cleanup_gpu_memory
    cleanup_gpu_memory()
    return _result


# ════════════════════════════════════════════════════════════════
# v16 (2026-05-22): Hierarchical preproc 통합 (사용자 명시 A2 + B1)
# ────────────────────────────────────────────────────────────────
# 2026-05-26 (Sprint 1.5 R2): R9(per_model_optimize) 의 6-bucket classifier 제거 →
#   hierarchical._categorize_feature_groups 의 19-bucket promoted 호출.
# 이전 (R9 per_model_optimize local): lag / weather / mobility / seasonal / exog / other (6)
# 현재 (hierarchical):  advanced / cyclic / spectral / discrete / lag_ili /
#                       rmean / weather / fcst_weather / disease_count /
#                       mobility_rt / mobility / search_trend / vaccine /
#                       health_resource / claims / epi_indicator / binary /
#                       composite / other (19, fine-grained)
# ════════════════════════════════════════════════════════════════
from simulation.pipeline.preproc_optuna_hierarchical import (
    _categorize_feature_groups,  # noqa: F401  (re-export for backward compat callers)
)


def _amplification_surcharge(
    wis: float,
    y_train_max: float,
    y_state: Optional[dict] = None,
    alpha: float = _EXTRAP_AMP_ALPHA,
    threshold: float = _EXTRAP_AMP_THRESHOLD,
) -> float:
    """G-329: multiply WIS by a small surcharge for high-Jacobian asinh inverse tails."""
    try:
        if y_state is None or not np.isfinite(float(wis)):
            return wis
        names = []
        y_mode = y_state.get("y_mode")
        if y_mode == "individual":
            names.append(y_state.get("y_individual"))
        elif y_mode == "group":
            names.extend(y_state.get("y_group_chain", []))
        if "asinh" not in names:
            return wis
        amplification_factor = float(np.cosh(np.arcsinh(max(float(y_train_max), 0.0))))
        if np.isfinite(amplification_factor) and amplification_factor > threshold:
            return float(wis) * (1.0 + float(alpha) * (amplification_factor - 1.0))
    except Exception:
        pass
    return wis


def _sanity_penalize_wis(wis: float, y_pred: np.ndarray, y_train_max: float,
                         sanity_mult: Optional[float] = None, penalty: float = 1e4,
                         y_state: Optional[dict] = None) -> float:
    """G-256c: inflate a fold's WIS when its predictions explode far past the historical range.

    A nonlinear-inverse y-transform (log1p→expm1, sqrt→x²) on an extrapolating model (linear/NN)
    overshoots to 100s–1000s× the data (controlled experiment: ceiling 440–990796, r2 down to
    -2e8). The plain OOF objective averages this rare blow-up away (one bad fold outweighed by
    in-range gains), so Optuna keeps picking the exploding transform even though identity is in
    the pool. This guard adds a dominating penalty whenever ``max(y_pred) > sanity_mult × train
    max`` — making the blow-up visible in the objective so Optuna selects a safe transform on
    its own (no hard pool restriction needed). A legitimate epidemic peak is ~1.5× the train max;
    >3× (default) is pathological (no ILI season reaches 200). Env: ``MPH_SANITY_PRED_MULT``.

    Args:
        wis: the fold's weighted interval score (lower = better).
        y_pred: original-unit predictions on the val fold.
        y_train_max: max target seen in this fold's training data.
        sanity_mult: explosion threshold as a multiple of train max (default env or 3.0).
        penalty: base penalty magnitude (scaled by the explosion ratio).

    G-329 adds a smaller tail-amplification surcharge for asinh before hard explosion happens;
    the G-256c range explosion penalty is preserved and applied first.

    Returns:
        ``wis`` unchanged if predictions are in a sane range and transform has no asinh tail
        amplification, else a penalized value.
    """
    score = float(wis)
    if sanity_mult is None:
        import os
        try:
            sanity_mult = float(os.environ.get("MPH_SANITY_PRED_MULT", "3.0"))
        except (TypeError, ValueError):
            sanity_mult = 3.0
    try:
        cap = sanity_mult * max(float(y_train_max), 1.0)
        pmax = float(np.max(y_pred)) if np.size(y_pred) else 0.0
        if np.isfinite(pmax) and pmax > cap:
            score = score + penalty * (pmax / cap)   # dominates; scales with how far it blew up
    except Exception:
        pass
    return _amplification_surcharge(score, y_train_max, y_state)


def _evaluate_config_hierarchical(
    factory_fn,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    optuna_trial: object,    # required for hierarchical sampling
    feature_indices=None,
    sigma_for_wis: float = 1.0,
    feature_cols=None,
    max_chain_length: int = 2,
    calib_residuals=None,    # Q1: prior-fold OOF residuals → PI calibration (else in-sample)
    extrapolation_safe: bool = False,   # G-256: linear/NN/GAM → linear-inverse y only
    force_y_identity: bool = False,     # G-300: model owns y-transform → R9 y = identity
    force_x_identity: bool = False,     # G-301: USES_FEATURES=False → R9 x = none (no waste)
    restrict_centered_y: bool = False,  # G-303: in-model 0-floor → exclude laplace/mcmc_robust
) -> dict:
    """Hierarchical preproc 사용한 _evaluate_config 의 alternative.

    v15 의 preproc_optuna_hierarchical 모듈 사용:
      - Y: none / individual / group(metric_only 또는 categorical)
      - X: none / individual / group(metric_only ColumnTransformer 또는 categorical)
      - Inverse: train-fitted state 만 사용 (no data leakage)

    Return shape 은 _evaluate_config 와 호환:
      {wis, mae, rmse, transform, scaler, n_features, ...}
      + 추가: hier_y_state, hier_x_state (Champion artifact serialization 용)
    """
    from simulation.pipeline.preproc_optuna_hierarchical import (
        suggest_y_preproc, suggest_x_scaler,
    )

    # ───── feature selection (v20 final fix: defensive validation)
    # numpy array → list 변환 (truthiness ambiguity 방지)
    if feature_indices is not None:
        try:
            feature_indices = list(feature_indices)
        except TypeError:
            feature_indices = None
    has_feat_idx = bool(feature_indices) and len(feature_indices) > 0

    try:
        X_train_use = X_train[:, feature_indices] if has_feat_idx else np.asarray(X_train)
        X_val_use = X_val[:, feature_indices] if has_feat_idx else np.asarray(X_val)
    except Exception as e:
        return {"wis": float("inf"), "mae": float("inf"), "error": f"feature_idx: {e}",
                "transform": "HIER_FAIL_FEAT", "scaler": "HIER_FAIL_FEAT",
                "n_features": 0}

    # Shape validation: X_train_use vs y_train row 일치 필수
    y_train_arr = np.asarray(y_train).ravel()
    y_val_arr = np.asarray(y_val).ravel()
    if X_train_use.shape[0] != y_train_arr.shape[0]:
        return {"wis": float("inf"), "mae": float("inf"),
                "error": f"shape_mismatch: X_train {X_train_use.shape[0]} vs y_train {y_train_arr.shape[0]}",
                "transform": "HIER_FAIL_SHAPE", "scaler": "HIER_FAIL_SHAPE",
                "n_features": X_train_use.shape[1] if X_train_use.ndim > 1 else 0}
    if X_val_use.shape[0] != y_val_arr.shape[0]:
        return {"wis": float("inf"), "mae": float("inf"),
                "error": f"shape_mismatch: X_val {X_val_use.shape[0]} vs y_val {y_val_arr.shape[0]}",
                "transform": "HIER_FAIL_SHAPE_VAL", "scaler": "HIER_FAIL_SHAPE_VAL",
                "n_features": X_val_use.shape[1] if X_val_use.ndim > 1 else 0}

    feat_names_use = None
    if has_feat_idx and feature_cols is not None:
        try:
            feat_names_use = [feature_cols[i] for i in feature_indices]
        except (IndexError, TypeError):
            feat_names_use = list(feature_cols)[:X_train_use.shape[1]] if feature_cols else None
    elif feature_cols is not None:
        feat_names_use = list(feature_cols)[:X_train_use.shape[1]]

    # ───── Hierarchical Y preproc
    try:
        y_train_t, inv_y_fn, y_state = suggest_y_preproc(
            optuna_trial, y_train_arr, max_chain_length=max_chain_length,
            extrapolation_safe=extrapolation_safe,
            force_y_identity=force_y_identity,
            restrict_centered_y=restrict_centered_y,
        )
        # Verify y_train_t shape matches input
        if y_train_t.shape[0] != y_train_arr.shape[0]:
            raise ValueError(f"Y transform changed shape: {y_train_arr.shape} → {y_train_t.shape}")
    except Exception as e:
        return {"wis": float("inf"), "mae": float("inf"), "error": f"y_preproc: {e}",
                "transform": "HIER_FAIL", "scaler": "HIER_FAIL",
                "n_features": X_train_use.shape[1] if X_train_use.ndim > 1 else 0}

    # ───── Hierarchical X preproc (feature_groups from feat_names_use)
    feature_groups = None
    if feat_names_use is not None:
        feature_groups = _categorize_feature_groups(feat_names_use)

    try:
        X_train_s, X_val_s, x_scaler, x_state = suggest_x_scaler(
            optuna_trial, X_train_use, X_val_use, feature_groups=feature_groups,
            force_x_identity=force_x_identity,
        )
    except Exception as e:
        return {"wis": float("inf"), "mae": float("inf"), "error": f"x_preproc: {e}",
                "transform": f"HIER_{y_state.get('y_mode', '?')}", "scaler": "HIER_FAIL",
                "n_features": X_train_use.shape[1]}

    # ───── Pre-fit shape validation (v20 final defensive)
    try:
        X_train_s = np.asarray(X_train_s)
        X_val_s = np.asarray(X_val_s)
        if X_train_s.shape[0] != y_train_t.shape[0]:
            return {"wis": float("inf"), "mae": float("inf"),
                    "error": f"post_preproc_shape: X_train_s {X_train_s.shape[0]} vs y_train_t {y_train_t.shape[0]}",
                    "transform": f"HIER_{y_state.get('y_mode', '?')}",
                    "scaler": f"HIER_{x_state.get('x_mode', '?')}",
                    "n_features": X_train_s.shape[1] if X_train_s.ndim > 1 else 0}
        if X_val_s.shape[0] != y_val_arr.shape[0]:
            return {"wis": float("inf"), "mae": float("inf"),
                    "error": f"post_preproc_shape: X_val_s {X_val_s.shape[0]} vs y_val {y_val_arr.shape[0]}",
                    "transform": f"HIER_{y_state.get('y_mode', '?')}",
                    "scaler": f"HIER_{x_state.get('x_mode', '?')}",
                    "n_features": X_val_s.shape[1] if X_val_s.ndim > 1 else 0}
    except Exception as e:
        return {"wis": float("inf"), "mae": float("inf"), "error": f"shape_check: {e}",
                "transform": "HIER_FAIL_SHAPE", "scaler": "HIER_FAIL_SHAPE",
                "n_features": 0}

    # ───── Model fit + predict (transformed space)
    # G-273c (2026-06-15): Stage-1/2(preproc·feature) eval 은 transform/feature 를 *비교* 하는
    # 단계다. 그런데 tree forecaster 의 fit() 은 매 호출마다 내부 HP Optuna study(×CV)를 통째로
    # 재실행한다 → preproc pure-grid(7-transform×1 OOF, G-335) × HP 내부 study(~20-50 trial × 3 CV) ≈ 수천 번
    # boosting fit = XGBoost 61분의 구조적 원인. 이 플래그를 fit 동안만 켜서 tree forecaster 가
    # 내부 HP study 를 건너뛰고 단일 reasonable default + early_stop 으로 1회만 학습하게 한다
    # (transform 비교엔 충분; HP 풀 튜닝은 Stage-3 최종 refit 에서만). 최종 refit(_refit_and_predict_*)
    # 은 이 함수를 거치지 않으므로 플래그 미설정 → 항상 full HP. 챔피언 TabPFN 등 foundation 은
    # 내부 HP study 자체가 없어 무영향.
    import os as _os_eh
    model = factory_fn()
    # G-273c-B (2026-06-15, 실증 후): Stage-1 preproc 는 *transform 선택* 단계 — 단일 고정점(fast)이
    #   XGBoost서 full(identity)과 다른 transform(sqrt)을 골랐음(validate_fastpath_selection.py:
    #   XGBoost Exp1 불일치, LightGBM/RF 일치·feature subset 3/3 ρ=1.0). 단일점 대신 *작은 HP
    #   탐색*(k=5, 트리만)으로 full 추종. feature-stability·mc-probe(_fast_inner=단일점)는 full 과
    #   동일 입증이라 그대로. 트리 fit 이 MPH_INNER_HP_PREPROC_TRIALS 를 읽어 study trial 수를 축소.
    _pp_k = _os_eh.environ.get("MPH_PREPROC_INNER_TRIALS", "5")
    _prev_pp = _os_eh.environ.get("MPH_INNER_HP_PREPROC_TRIALS")
    _os_eh.environ["MPH_INNER_HP_PREPROC_TRIALS"] = _pp_k
    try:
        # G-291 (2026-06-17, 3자 감사): OOF fold 에 feature_names 전달 — OverseasTransfer encoder/DL lag1
        #   인덱스 탐색에 필요(미전달 시 transfer skip). 미사용 모델은 **kwargs 로 무시(무해).
        model.fit(X_train_s, y_train_t, feature_names=feat_names_use)
        # G-337: preproc/HP 선택 OOF 도 sequence 모델은 rolling 1-step(static collapse 회피 = transform/
        #   HP 를 공정 1-step 기준으로 선택). 공유 헬퍼(identity→raw, affine transform→정확).
        y_pred_t = _rolling_or_static_predict_oof(model, X_val_s, y_train, y_val, y_train_t)
    except Exception as e:
        return {"wis": float("inf"), "mae": float("inf"), "error": str(e),
                "transform": f"HIER_{y_state.get('y_mode', '?')}",
                "scaler": f"HIER_{x_state.get('x_mode', '?')}",
                "n_features": X_train_use.shape[1] if X_train_use.ndim > 1 else 0}
    finally:
        if _prev_pp is None:
            _os_eh.environ.pop("MPH_INNER_HP_PREPROC_TRIALS", None)
        else:
            _os_eh.environ["MPH_INNER_HP_PREPROC_TRIALS"] = _prev_pp

    # ───── Inverse Y to original units
    try:
        y_pred = np.maximum(np.asarray(inv_y_fn(y_pred_t)).ravel(), 0.0)  # G-298: ILI≥0 원공간 floor
    except Exception as e:
        return {"wis": float("inf"), "mae": float("inf"), "error": f"inverse: {e}",
                "transform": f"HIER_{y_state.get('y_mode', '?')}",
                "scaler": f"HIER_{x_state.get('x_mode', '?')}",
                "n_features": X_train_use.shape[1]}

    # Sanitize NaN/inf (G-159: NaN/None/±inf → 0.0 만, 정상 값 보존)
    # Codex audit 2026-05-27 fix: 종래 finite_median 대체는 G-159 위반 —
    # sanitize_predictions 의 표준 보장 (NaN/inf → 0.0) 와 일관성 깨짐.
    # Hierarchical Optuna trial 의 inverse-Y NaN 도 표준 sanitize 적용.
    from simulation.models.safety import sanitize_predictions
    y_pred = sanitize_predictions(y_pred, nonneg=False)

    # ───── Metrics
    err = y_pred - y_val_arr
    mae = float(np.mean(np.abs(err)))
    # 2026-05-30 (B: real-WIS): calibration-aware WIS from the model's own residuals.
    # ALSO fixes a silent bug — `weighted_interval_score` was never imported in THIS
    # function's scope → NameError → the bare `except` had been returning wis=mae for EVERY
    # hierarchical (production) trial under a false "wis" label. Verified 2026-05-30.
    # Q1: prior-fold OOF residuals (calib_residuals) when the CV loop threads them, else in-sample.
    wis, resid_train = _real_wis_and_residuals(
        model, X_train_s, y_train_arr, y_val_arr, y_pred, inv_y_fn, sigma_for_wis, mae,
        calib_residuals=calib_residuals)
    # G-256c (2026-06-12): explosion guard — make the OOF objective SEE a transform blow-up so
    # Optuna picks a safe one ITSELF (user-preferred: full pool incl. identity, no hard restrict).
    # log1p+expm1 on an extrapolating model overshoots to 100s-1000s (Part A r2 -9 to -2e8); this
    # inflates the fold's WIS past any in-range gain so the exploding config loses selection.
    wis = _sanity_penalize_wis(wis, y_pred, float(np.max(y_train_arr)), y_state=y_state)
    _ir_metrics = _in_range_fold_metrics(y_train_arr, y_val_arr, y_pred, resid_train, wis)

    # OOF r2/mape/pi95 (empirical band) — 진단/selection parity with the flat path (gate 제거 2026-06-05).
    _ss_tot = float(np.sum((y_val_arr - y_val_arr.mean()) ** 2))
    _r2 = (1.0 - float(np.sum(err ** 2)) / _ss_tot) if _ss_tot > 1e-12 else float("nan")
    _nz = np.abs(y_val_arr) > 1e-9
    _mape = (float(np.mean(np.abs(err[_nz] / y_val_arr[_nz])) * 100.0)
             if _nz.any() else float("nan"))
    if resid_train.size >= 2:
        _pi95 = float(np.mean(np.abs(err) <= float(np.quantile(np.abs(resid_train), 0.95))))
    else:
        _pi95 = float(np.mean(np.abs(err) <= Z95 * max(float(sigma_for_wis), 1e-9)))

    _result = {
        "wis": wis,
        "mae": mae,
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": _r2,
        "mape": _mape,
        "pi95_coverage": _pi95,
        "transform": f"HIER_{y_state.get('y_mode', '?')}",   # marker
        "scaler": f"HIER_{x_state.get('x_mode', '?')}",        # marker
        "n_features": X_train_use.shape[1],
        **_ir_metrics,
        "hier_y_state": y_state,   # Champion artifact serialization
        "hier_x_state": x_state,   # Champion artifact serialization
        # Q1: OOS val residuals → next WF fold's PI calibration (transient, "_" = not serialized)
        "_val_residuals": err,
    }
    _result["selection_wis"] = _oof_selection_score(_result)
    return _result


# 2026-05-26: R9(per_model_optimize) 의 _apply_transform / _apply_single_y_transform 정의 제거.
# 모든 호출은 simulation/pipeline/preproc_optuna_hierarchical.py 의
# `_apply_single_y_transform` 을 사용 (top-level import L60-67).
# 이전 (≈100 lines): identity/log1p/boxcox/yeo_johnson/sqrt/asinh/gaussian/
#                    mcmc_robust/laplace/rank 분기 + G-146 inverse caps —
#                    hierarchical 과 거의 동일 logic 의 별도 copy.
# 현재: hierarchical 만 정의 (단일 source of truth).


def _hier_replay_preproc(
    hier_frozen_params: dict,
    hier_max_chain_length: int,
    y_train_pool: np.ndarray,
    X_tr: np.ndarray,
    X_te: np.ndarray,
    feat_names_use: Optional[list[str]],
) -> tuple:
    """Replay a sampled hierarchical preproc, RE-FIT on the full (train+val) pool (G-233).

    Stage-1 preproc Optuna records its choice as a mode marker (``transform="HIER_<mode>"``)
    plus the raw Optuna ``trial.params`` (``hier_frozen_params``). The flat refit path applies
    the y-transform by name and so raised ``ValueError: Unknown Y transform: HIER_*``. This
    re-applies the EXACT sampled hierarchical preproc via an optuna ``FixedTrial`` over the
    frozen params (same pattern as ``_oof_cv_wis_hier``), fitting on the full pool and
    transforming the test slab. The y-inverse is the picklable state-based
    ``apply_y_preproc_inverse_only(.., hier_y_state)`` so reported metrics == R10/Pinf
    inference reproduction.

    Returns ``(y_tr_t, transform_inv, transform_inv_obj, X_tr_s, X_te_s, fitted_scaler,
    transform_name, hier_y_state)``. Raises on FixedTrial/fit errors (caller degrades).
    """
    import optuna as _opt
    from simulation.pipeline.preproc_optuna_hierarchical import (
        suggest_y_preproc, suggest_x_scaler, apply_y_preproc_inverse_only,
        _categorize_feature_groups,
    )
    fixed = _opt.trial.FixedTrial(dict(hier_frozen_params))
    y_pool = np.asarray(y_train_pool, dtype=np.float64).ravel()
    y_tr_t, _closure_inv, hier_y_state = suggest_y_preproc(
        fixed, y_pool, max_chain_length=hier_max_chain_length)
    feature_groups = (_categorize_feature_groups(feat_names_use)
                      if feat_names_use is not None else None)
    X_tr_s, X_te_s, fitted_scaler, _x_state = suggest_x_scaler(
        fixed, X_tr, X_te, feature_groups=feature_groups)

    def transform_inv(yt, _st=hier_y_state):
        return apply_y_preproc_inverse_only(yt, _st)

    transform_name = "HIER_" + str(hier_y_state.get("y_mode", "?"))
    return (y_tr_t, transform_inv, None, X_tr_s, X_te_s,
            fitted_scaler, transform_name, hier_y_state)


def _do_no_harm_select(
    model_name: str,
    r9_config: dict,
    baseline_config: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    score_fn=None,
    factory_fn=None,
    feature_indices=None,
    feature_cols=None,
    baseline_indices=None,
    sigma: float = 1.0,
    margin: Optional[float] = None,
    tail_frac: float = 0.25,
    min_tail: int = 8,
    test_context_max: Optional[float] = None,
) -> tuple[dict, bool]:
    """Leak-free do-no-harm floor: keep the R9 config unless an identity baseline beats it on the
    most-recent TAIL of the train+val pool.

    transform-fix (2026-06-21) PART E — AUGMENTS the G-328c baseline-floor. G-328c compares R9 vs
    baseline on the full forward-holdout (X_val). This helper sharpens the comparison to the
    most-recent TAIL of the train+val pool — a leak-free proxy for the (sealed) test slab's
    out-of-range regime — so an R9 config that only wins on quiet early folds but would extrapolate
    badly is floored back to identity. NEVER touches the sealed test / y_test.

    Args:
        model_name: registry name. Ensembles (``Ensemble-*``) are skipped (kept on R9).
        r9_config: the R9-optimized config dict (transform/scaler/preproc_optuna_params).
        baseline_config: the identity baseline config dict (transform="identity", scaler="none").
        X_train, y_train: train pool (rows in time order; the tail is the most-recent slice).
        X_val, y_val: forward-holdout pool, appended after train to form the train+val pool. The
            tail is taken from the END of this combined pool (most recent observations).
        score_fn: optional injected scorer ``(cfg, X_tr, y_tr, X_ev, y_ev) -> float`` (lower=better);
            if None, a default WIS scorer built on ``_refit_and_predict_test`` + ``factory_fn`` is used.
        factory_fn: model factory (required when score_fn is None).
        feature_indices/feature_cols: R9 feature subset + names (for the default scorer).
        baseline_indices: BASIC-feature indices for the identity baseline (for the default scorer).
        sigma: WIS sigma (= std(y_train)).
        margin: relative WIS improvement the baseline must beat R9 by to trigger fallback
            (default = ``MPH_DO_NO_HARM_MARGIN`` env, else 0.05). Within-margin → keep R9 (overfit guard).
        tail_frac: fraction of the train+val pool used as the most-recent evaluation tail.
        min_tail: minimum tail size; below this the floor is skipped (kept on R9).
        test_context_max: optional max(y_test) for the audit log only (NOT used for selection).

    Returns:
        (chosen_config, fell_back) — the config to use and whether the baseline floor fired.

    Performance: 2 model refits on the tail (O(tail) fit cost). Side effects: logs an audit line
        (tail_max vs train_max vs optional test context). Caller responsibility: y_train/y_val in
        time order so the tail is genuinely the most-recent window.
    """
    # ensembles: their config is decided on the R2 (baseline) ensemble path (no preproc transform) → skip
    if model_name and str(model_name).startswith("Ensemble-"):
        return r9_config, False

    if margin is None:
        try:
            import os as _os
            margin = float(_os.environ.get("MPH_DO_NO_HARM_MARGIN", "0.05"))
        except Exception:
            margin = 0.05

    # build the train+val pool (time order: train then val) and carve the most-recent tail.
    try:
        X_pool = np.vstack([X_train, X_val])
        y_pool = np.concatenate([y_train, y_val])
    except Exception as _stack_exc:
        log.warning(f"  [do-no-harm] {model_name} skip: pool stack failed: {_stack_exc}")
        return r9_config, False

    n_pool = len(y_pool)
    n_tail = int(round(tail_frac * n_pool))
    if n_pool < min_tail * 2 or n_tail < min_tail:
        # pool too small to split into a meaningful fit-window + tail → skip (do no harm).
        return r9_config, False

    split = n_pool - n_tail
    X_fit, y_fit = X_pool[:split], y_pool[:split]
    X_eval, y_eval = X_pool[split:], y_pool[split:]

    train_max = float(np.max(y_fit)) if len(y_fit) else 0.0
    tail_max = float(np.max(y_eval)) if len(y_eval) else 0.0

    # default WIS scorer (production path): refit on the fit-window, score WIS on the tail.
    if score_fn is None:
        if factory_fn is None:
            log.warning(f"  [do-no-harm] {model_name} skip: no score_fn/factory_fn")
            return r9_config, False

        def _default_score(cfg, X_tr, y_tr, X_ev, y_ev):
            _idx = baseline_indices if cfg is baseline_config else feature_indices
            res = _refit_and_predict_test(
                factory_fn,
                transform_name=cfg.get("transform", "identity"),
                scaler_name=cfg.get("scaler", "none"),
                hier_frozen_params=cfg.get("preproc_optuna_params"),
                X_train_pool=X_tr, y_train_pool=y_tr,
                X_test=X_ev, y_test=y_ev,
                feature_indices=_idx, feature_cols=feature_cols,
                sigma_for_wis=sigma)
            w = res.get("wis") if isinstance(res, dict) else None
            return float(w) if isinstance(w, (int, float)) and np.isfinite(w) else float("inf")

        score_fn = _default_score

    try:
        r9_wis = float(score_fn(r9_config, X_fit, y_fit, X_eval, y_eval))
        bl_wis = float(score_fn(baseline_config, X_fit, y_fit, X_eval, y_eval))
    except Exception as _score_exc:
        log.warning(f"  [do-no-harm] {model_name} skip: scoring failed: {_score_exc}")
        return r9_config, False

    fell_back = (np.isfinite(r9_wis) and np.isfinite(bl_wis)
                 and bl_wis < r9_wis * (1.0 - margin))
    _ctx = (f", test_max={test_context_max:.2f}" if isinstance(test_context_max, (int, float))
            else "")
    log.info(
        f"  [do-no-harm] {model_name}: tail_max={tail_max:.2f} vs train_max={train_max:.2f}{_ctx} "
        f"| R9 tail-WIS={r9_wis:.3f} baseline={bl_wis:.3f} margin={margin:.2f} → "
        f"{'baseline (floor fired)' if fell_back else 'R9'}")
    return (baseline_config, True) if fell_back else (r9_config, False)


def _symmetric_rolling_eval(
    factory_fn, transform_name, scaler_name,
    X_pool, y_pool, X_test, y_test,
    feature_indices=None, feature_cols=None,
    hier_frozen_params=None, sigma_for_wis: float = 1.0,
) -> dict:
    """§8.6 symmetric refit — 각 test origin i 를 (pool ⊕ 관측 test[:i]) 로 frozen-config 전체
    재fit(transform/scaler/model) → 1-step 예측. ARIMA rolling 과 *진짜 동일*한 정보집합/파라미터
    갱신(ML fit-once 비대칭 제거). ``_refit_and_predict_test`` 를 origin 마다 재사용(standalone
    재구현 금지 = R²−1.45 burn 회피). ``MPH_SYMMETRIC_REEVAL=1`` 전용.

    Returns: {predictions (n_test,), wis, mae, rmse, r2, n, _symmetric:True}.
    Performance: n_test × full-refit (느림 — 빠른 모델만 실용; foundation/DL 은 호출자가 게이트).
    Side effects: origin 마다 모델 1회 fit (격리 없음 — 빠른 모델 전용).
    Caller responsibility: y_test = 관측 실측(미래 누수 0 — test[:i] 만 사용, i 미포함).
    """
    import numpy as _np
    Xp = _np.asarray(X_pool, dtype=float); yp = _np.asarray(y_pool, dtype=float).ravel()
    Xt = _np.asarray(X_test, dtype=float); yt = _np.asarray(y_test, dtype=float).ravel()
    n = len(Xt)
    preds = _np.full(n, _np.nan, dtype=float)
    for i in range(n):
        if i == 0:
            Xg, yg = Xp, yp
        else:
            Xg = _np.vstack([Xp, Xt[:i]])          # pool ⊕ 관측 test[:i] (i 미포함 = 인과)
            yg = _np.concatenate([yp, yt[:i]])
        try:
            r = _refit_and_predict_test(
                factory_fn, transform_name=transform_name, scaler_name=scaler_name,
                hier_frozen_params=hier_frozen_params,
                X_train_pool=Xg, y_train_pool=yg,
                X_test=Xt[i:i + 1], y_test=yt[i:i + 1],
                feature_indices=feature_indices, feature_cols=feature_cols,
                sigma_for_wis=sigma_for_wis)
            p = r.get("predictions")
            if p is not None and len(_np.asarray(p).ravel()) >= 1:
                preds[i] = float(_np.asarray(p).ravel()[0])
        except Exception:
            pass                                    # origin 실패 → nan (valid mask 로 제외)
    valid = _np.isfinite(preds)
    if int(valid.sum()) < 5:    # full-metric SSOT n≥5 (downstream test_metrics 보존)
        return {"error": "symmetric: <5 valid origins", "predictions": preds.tolist(), "_symmetric": True}
    ya, yb = yt[valid], preds[valid]
    # ★ full 129-key SSOT 를 symmetric 예측에 적용 (subset 반환 시 downstream 깨짐 — 129 보존).
    try:
        from simulation.pipeline.phase_evaluator import evaluate_predictions_full as _epf
        out = dict(_epf(_np.asarray(ya, dtype=float), _np.asarray(yb, dtype=float),
                        sigma=float(sigma_for_wis), y_train_pool=_np.asarray(y_pool, dtype=float),
                        phase_id="symmetric_reeval"))
    except Exception:
        out = {}
    ss_res = float(_np.sum((ya - yb) ** 2)); ss_tot = float(_np.sum((ya - _np.mean(ya)) ** 2))
    out.setdefault("rmse", float(_np.sqrt(_np.mean((ya - yb) ** 2))))
    out.setdefault("mae", float(_np.mean(_np.abs(ya - yb))))
    out.setdefault("r2", float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"))
    if "wis" not in out:
        try:
            from simulation.analytics.diagnostics import weighted_interval_score
            out["wis"] = float(_np.mean(_np.asarray(weighted_interval_score(ya, yb, sigma_for_wis), dtype=float)))
        except Exception:
            out["wis"] = out["mae"]
    out["predictions"] = preds.tolist()
    out["n"] = int(valid.sum()); out["_symmetric"] = True
    return out


def _multi_seed_metrics(
    factory_fn, transform_name, scaler_name,
    X_pool, y_pool, X_test, y_test,
    feature_indices=None, feature_cols=None,
    hier_frozen_params=None, sigma_for_wis: float = 1.0,
    seeds=(42, 1, 2, 3, 4),
) -> dict:
    """#13 재현성 — 동일 frozen-config 를 N seed 로 재fit → test WIS/R² mean±SD.

    deterministic 모델(seed 고정·random_state=42) → SD≈0 = **재현성 증명**.
    비결정 모델(unseeded BLAS/TPE 등) → SD>0 = **플래그**. ``_refit_and_predict_test`` 재사용.
    챔피언 보고용(전 모델 X). Returns: {wis_mean, wis_sd, r2_mean, r2_sd, n_seeds, wis_per_seed, seeds}.
    """
    import numpy as _np
    wis_l, r2_l = [], []
    for s in seeds:
        _np.random.seed(int(s))
        try:
            import torch as _t; _t.manual_seed(int(s))
        except Exception:
            pass
        try:
            r = _refit_and_predict_test(
                factory_fn, transform_name=transform_name, scaler_name=scaler_name,
                hier_frozen_params=hier_frozen_params,
                X_train_pool=X_pool, y_train_pool=y_pool, X_test=X_test, y_test=y_test,
                feature_indices=feature_indices, feature_cols=feature_cols,
                sigma_for_wis=sigma_for_wis)
            if "error" not in r:
                w, r2 = r.get("wis"), r.get("r2")
                if isinstance(w, (int, float)) and _np.isfinite(w):
                    wis_l.append(float(w))
                if isinstance(r2, (int, float)) and _np.isfinite(r2):
                    r2_l.append(float(r2))
        except Exception:
            pass

    def _ms(a):
        return (float(_np.mean(a)), float(_np.std(a))) if a else (float("nan"), float("nan"))
    wm, ws = _ms(wis_l)
    rm, rs = _ms(r2_l)
    return {"wis_mean": wm, "wis_sd": ws, "r2_mean": rm, "r2_sd": rs,
            "n_seeds": len(wis_l), "wis_per_seed": wis_l, "seeds": list(seeds)}


def _direct_multihorizon_eval(
    factory_fn, transform_name, scaler_name,
    X_pool, y_pool, X_test, y_test,
    feature_indices=None, feature_cols=None,
    hier_frozen_params=None, sigma_for_wis: float = 1.0, H: int = 4,
) -> dict:
    """③ direct multi-horizon — h=1..H 각각 **별도 모델**(X[t]→y[t+h], recursion 없음, 오차누적 0).

    FluSight 1-4주. h=1 만은 persistence 지배(§6.1 trivial), h≥2 가 변별. 누수 0: X[t]의 lag 는
    origin t 관측만, t..t+h 사이 y 미사용. valid test = pool_end+i+h < n 인 origin. ``_refit_and_predict_test`` 재사용.
    Returns: {h: {wis, r2, mae, n}} per horizon.
    """
    import numpy as _np
    Xp = _np.asarray(X_pool, dtype=float); yp = _np.asarray(y_pool, dtype=float).ravel()
    Xt = _np.asarray(X_test, dtype=float); yt = _np.asarray(y_test, dtype=float).ravel()
    full_y = _np.concatenate([yp, yt])
    res = {}
    for h in range(1, H + 1):
        if len(yp) <= h:
            continue
        Xtr_h, ytr_h = Xp[:-h], yp[h:]                       # train: X[t] → y[t+h] (pool 내)
        Xte_h, yte_h = [], []
        for i in range(len(Xt)):                              # test origin i 에서 h-ahead
            gt = len(yp) + i + h                               # target 글로벌 idx
            if gt < len(full_y):
                Xte_h.append(Xt[i]); yte_h.append(full_y[gt])
        if len(yte_h) < 5:
            continue
        try:
            r = _refit_and_predict_test(
                factory_fn, transform_name=transform_name, scaler_name=scaler_name,
                hier_frozen_params=hier_frozen_params,
                X_train_pool=Xtr_h, y_train_pool=ytr_h,
                X_test=_np.asarray(Xte_h, dtype=float), y_test=_np.asarray(yte_h, dtype=float),
                feature_indices=feature_indices, feature_cols=feature_cols, sigma_for_wis=sigma_for_wis)
            res[h] = {"wis": r.get("wis"), "r2": r.get("r2"), "mae": r.get("mae"), "n": len(yte_h)}
        except Exception as e:   # noqa: BLE001
            res[h] = {"error": str(e)[:60]}
    return res


def _native_interval_wis(model, X_test, y_test, levels=None) -> dict:
    """native-NB→WIS (#17) — 모델의 ``predict_quantiles`` (count 보정 분위)로 WIS·PICP 계산.

    NegBin/SeirCount 등 native count interval 보유 모델용. ★fairness(G-318): 이건 **SECONDARY
    artifact**(챔피언 선정 metric 아님 — 전 모델 동일 empirical-residual WIS 유지가 공정). figure/
    ABM/논문 count-PI 용. ``predict_quantiles`` 없으면 None. Returns: {native_wis, native_picp95, _native_nb}.
    """
    import numpy as _np
    if not hasattr(model, "predict_quantiles"):
        return None
    if levels is None:
        levels = (0.025, 0.25, 0.5, 0.75, 0.975)
    try:
        q = model.predict_quantiles(_np.asarray(X_test, dtype=float), levels=levels)
    except Exception:
        return None
    yt = _np.asarray(y_test, dtype=float).ravel()
    med = _np.asarray(q.get(0.5), dtype=float)
    # WIS = (1/(K+0.5))[0.5|y-med| + Σ (α/2)·IS_α]   (Bracher 2021)
    pairs = [(0.025, 0.975, 0.05), (0.25, 0.75, 0.5)]   # (lo, hi, alpha)
    total = 0.5 * _np.abs(yt - med)
    K = 0
    for lo, hi, alpha in pairs:
        if lo in q and hi in q:
            L = _np.asarray(q[lo], dtype=float); U = _np.asarray(q[hi], dtype=float)
            width = U - L
            pen = (2.0 / alpha) * (L - yt) * (yt < L) + (2.0 / alpha) * (yt - U) * (yt > U)
            total = total + (alpha / 2.0) * (width + pen); K += 1
    wis = float(_np.mean(total / (K + 0.5))) if K else float("nan")
    picp95 = (float(_np.mean((yt >= _np.asarray(q[0.025], float)) & (yt <= _np.asarray(q[0.975], float))))
              if (0.025 in q and 0.975 in q) else float("nan"))
    return {"native_wis": wis, "native_picp95": picp95, "_native_nb": True}


def _refit_and_predict_test(
    factory_fn,
    transform_name: str,
    scaler_name: str,
    X_train_pool: np.ndarray, y_train_pool: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    feature_indices: Optional[list[int]] = None,
    sigma_for_wis: float = 1.0,
    return_fitted_model: bool = False,
    feature_cols: Optional[list[str]] = None,   # 2026-04-28: grouped preproc
    # G-181 (2026-05-05) — Ensemble val_predictions path fix:
    val_predictions_dict: Optional[dict] = None,    # base 모델 val 예측 dict
    val_actual: Optional[np.ndarray] = None,         # 실제 val (for ensemble weighting)
    test_predictions_dict: Optional[dict] = None,    # base 모델 test 예측 dict
    is_ensemble: bool = False,                        # ensemble 카테고리 flag
    # audit Stage 1.1 (Task #13.b/c cascade, 2026-05-27) — KDCA threshold input
    viral_positivity_train: Optional[np.ndarray] = None,  # WHO FluNet KR positivity
    hier_frozen_params: Optional[dict] = None,        # G-233: HIER preproc replay params
    hier_max_chain_length: int = 2,
) -> dict:
    """Refit a model with chosen (transform, scaler) on the FULL train+val
    pool, then predict on the TEST slab. This is the proper evaluation step
    after the val-grid search (which only chose the config).

    When ``return_fitted_model=True`` the returned dict carries enough
    state under ``_artifact_state`` to rebuild the **exact** training-time
    pipeline at inference time without seeing y_train again:

      _fitted_model      → trained estimator
      _artifact_state    → {
          "transform_name": ...,
          "transform_inv_obj": λ for boxcox / fitted PowerTransformer for yeo_johnson / None,
          "fitted_scaler":  fitted sklearn scaler instance | None,
          "feature_indices": [...] | None,
      }

    R9 then bundles these into a ``ChampionArtifact`` and pickles
    that as ``models/<name>.pt`` — R10 (per_model_eval) just loads the artifact and
    calls ``.predict(X)``.

    Returns (on success): {predictions, wis, mae, rmse, r2, n,
                            _fitted_model?, _artifact_state?}.
    """
    from simulation.analytics.diagnostics import weighted_interval_score
    from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS

    # G-309 (3자 감사 #4): recode global-summary features (quantile/threshold/interaction) leakage-free
    #   — [pool; test] 를 train_end=len(pool) 기준 재코딩(test era 통계 미사용, baseline/wfcv 와 정합).
    #   feature/scaling 슬라이싱 전에 적용. 해당 컬럼 부재 시 no-op (동작 불변).
    if feature_cols is not None and X_test is not None and len(X_test) > 0:
        _np_pool = len(X_train_pool)
        _yc = np.concatenate([np.asarray(y_train_pool, dtype=float).ravel(),
                              np.zeros(len(X_test))])
        _Xc = _recode_advanced_per_fold(np.vstack([X_train_pool, X_test]),
                                        _yc, feature_cols, _np_pool)
        X_train_pool, X_test = _Xc[:_np_pool], _Xc[_np_pool:]

    # Feature subset (same as during val search)
    if feature_indices is not None:
        X_tr = X_train_pool[:, feature_indices]
        X_te = X_test[:, feature_indices]
    else:
        X_tr, X_te = X_train_pool, X_test

    # Target transform + scaler. ``transform_state`` carries fitted params (boxcox λ /
    # PowerTransformer) for artifact replay. G-233: HIER replay when params present; else
    # flat-by-name (a "HIER_*" marker WITHOUT params degrades to identity — never raises).
    feat_names_use = feature_cols
    if feature_cols is not None and feature_indices is not None:
        feat_names_use = [feature_cols[i] for i in feature_indices]
    _hier_y_state = None
    fitted_scaler = None
    if hier_frozen_params:
        try:
            (y_tr_t, transform_inv, transform_inv_obj, X_tr_s, X_te_s,
             fitted_scaler, transform_name, _hier_y_state) = _hier_replay_preproc(
                hier_frozen_params, hier_max_chain_length,
                y_train_pool, X_tr, X_te, feat_names_use)
            transform_state = {}
        except Exception as _he:
            return {"error": f"hier_replay: {_he}"}
    else:
        _tn = "identity" if str(transform_name).startswith("HIER_") else transform_name
        y_tr_t, transform_inv, transform_state = _apply_single_y_transform(
            y_train_pool, _tn)
        transform_inv_obj = (
            transform_state.get("lambda") if _tn == "boxcox"
            else transform_state.get("power_transformer") if _tn == "yeo_johnson"
            else None
        )
        transform_name = _tn
        # G-165: "grouped_optuna" 도 grouped 와 동일 처리 (no-scale 학습/추론 불일치 방지).
        if (scaler_name in ("grouped", "grouped_optuna")) and feat_names_use is not None:
            try:
                from simulation.models.grouped_preprocessor import build_grouped_preprocessor
                sc = build_grouped_preprocessor(feat_names_use)
                X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)
                fitted_scaler = sc
            except Exception:
                from sklearn.preprocessing import RobustScaler
                sc = RobustScaler()
                X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)
                fitted_scaler = sc
        elif scaler_name in ("none", "standard", "robust", "quantile"):
            sc = _build_single_x_scaler(scaler_name)
            if sc is None:
                X_tr_s, X_te_s = X_tr, X_te
            else:
                X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)
                fitted_scaler = sc
        else:
            X_tr_s, X_te_s = X_tr, X_te

    # G-160 + G-166 (2026-05-02): _refit_and_predict_test 가 직접 model.fit 호출.
    # fit_predict 안 거치므로 shape validation 명시적 호출 필요.
    # tree_models.py:86 IndexError "inconsistent samples [235, 200]" 의 직접 호출자.
    from simulation.models.base import (_validate_shapes, sanitize_predictions,
                                         supports_rolling_eval, supports_transform_rolling)
    try:
        _validate_shapes(X_tr_s, y_tr_t, X_test=X_te_s,
                         name=f"refit_{transform_name}_{scaler_name}")
    except ValueError as ve:
        return {"error": f"shape_validation: {ve}"}

    model = factory_fn()
    try:
        # G-181 (2026-05-05) — Ensemble val_predictions path fix
        if is_ensemble and val_predictions_dict and test_predictions_dict:
            # Ensemble fit_predict 직접 호출 (val_predictions/val_actual + model_predictions kwargs)
            _val_actual = val_actual if val_actual is not None else y_tr_t
            try:
                # 1. fit (val_predictions + val_actual 로 가중치 학습)
                model.fit(X_tr_s, y_tr_t,
                          val_predictions=val_predictions_dict,
                          val_actual=_val_actual)
                # 2. predict (model_predictions = base 모델 test 예측)
                y_pred_t = model.predict(X_te_s,
                                         model_predictions=test_predictions_dict)
                log.info(f"  [phase13-ensemble] {len(val_predictions_dict)} base models → ensemble pred")
            except Exception as ens_e:
                # Ensemble 실패 시 fallback: base 모델 평균 (median)
                log.warning(f"  [phase13-ensemble] fit_predict 실패: {ens_e} → median fallback")
                preds_arr = np.array(list(test_predictions_dict.values()))
                y_pred_t = np.median(preds_arr, axis=0) if preds_arr.size else np.zeros(len(X_te_s))
        else:
            # G-299 (2026-06-17): thread feature_names into the REPORTED/DEPLOY refit (was only
            #   passed at the selection/OOF sites :345/:607). OverseasTransfer's encoder locates
            #   ili_rate_lag1-4 by name; without it the refit/deploy silently ran encoder-OFF
            #   (feature-only) while selection ran encoder-ON → selection≠eval on the shipped model.
            #   All models tolerate the kwarg (**kwargs), same as the OOF sites.
            model.fit(X_tr_s, y_tr_t, feature_names=feat_names_use)
            # G-321 (2026-06-19, 사용자): META classic-ts(ARIMA/SARIMA/SARIMAX/Theta/FluSight) 는
            #   rolling-origin 1-step(관측 과거로 1주씩 예측) = feature 모델의 predict(X_test) 1-step 과
            #   동일 task = 공정 평가. identity transform 이라 raw y_test = 모델 학습공간. 단일원점
            #   forecast(len)=68주 외삽→mean-revert→불공정 음수(A/B: ARIMA −0.89→+0.92 등). 그 외 모델
            #   은 기존 predict(X)(feature=실 lag 로 이미 1-step).
            if supports_rolling_eval(model) and y_test is not None and len(y_test) == len(X_te_s):
                y_pred_t = model.predict(X_te_s, y_observed=np.asarray(y_test, dtype=float))
            elif supports_transform_rolling(model) and y_test is not None and len(y_test) == len(X_te_s):
                # G-337 (2026-06-24): transform-space sequence(N-BEATS=mcmc_robust/TiDE=laplace) — 인코더가
                #   transform 공간이라 raw y_observed 주면 공간불일치 폭발(실측 −9~−16). [train_pool;test] 동시
                #   transform(affine m·s = train 269-지배 fit) 후 test 슬라이스 = transformed y_observed →
                #   공정 rolling(collapse 회피: N-BEATS −0.84→−0.21, TiDE −0.08→+0.20; 약체=구조적). 실패→static.
                try:
                    from simulation.pipeline.preproc_optuna_hierarchical import _apply_single_y_transform as _afy_tr
                    _tnr = str(transform_name)
                    _tnr = _tnr[5:] if _tnr.startswith("HIER_") else _tnr
                    # G-349 (2026-06-25, 감사 P1): 'none'=identity alias. _apply_single_y_transform 은 'identity'만
                    #   인식('none'→ValueError) → strip-only 가 y_mode='none'(N-BEATS/N-HiTS/TiDE 실측)서 ValueError
                    #   →static fallback = 선정(rolling)≠평가(static) 비대칭. sibling :1308 과 동일 정규화로 통일.
                    _tnr = "identity" if _tnr in ("none", "") else _tnr
                    _cat_t, _, _ = _afy_tr(np.concatenate([
                        np.asarray(y_train_pool, dtype=float).ravel(),
                        np.asarray(y_test, dtype=float)]), _tnr)
                    y_pred_t = model.predict(
                        X_te_s, y_observed=np.asarray(_cat_t[len(y_train_pool):], dtype=float))
                except Exception as _tre:
                    log.warning(f"  [phase13] {getattr(getattr(model, 'meta', None), 'name', '?')} "
                                f"transform-rolling 실패({_tre}) → static fallback")
                    y_pred_t = model.predict(X_te_s)
            else:
                y_pred_t = model.predict(X_te_s)
    except Exception as e:
        return {"error": str(e)}
    # G-159: predict 결과 sanitize (NaN/inf → 0.0 만, 정상 값 보존)
    # (sanitize_predictions 는 line 523 에서 이미 import)
    y_pred_t = sanitize_predictions(y_pred_t)
    y_pred = np.asarray(transform_inv(y_pred_t), dtype=np.float64)
    y_pred = sanitize_predictions(y_pred, nonneg=True)   # G-298: ILI≥0 원공간 도메인 floor

    # G-181 (2026-05-05) — 사용자 grill: "예상 외 대답 → 세이프 모드"
    # Auto-fallback: transform 적용 후에도 pred 발산 (max > y_train_max × 100) 또는
    #   NaN/inf 비율 > 50% → identity transform 으로 자동 retry.
    if GLOBAL.ops.safe_mode_auto:
        try:
            _y_pred_finite = y_pred[np.isfinite(y_pred)]
            _y_max_train = float(np.nanmax(y_train_pool)) if len(y_train_pool) else 100.0
            _frac_invalid = 1.0 - (len(_y_pred_finite) / max(1, len(y_pred)))
            _is_diverged = (
                len(_y_pred_finite) == 0 or
                _y_pred_finite.max() > _y_max_train * 100.0 or
                _frac_invalid > 0.5
            )
            if _is_diverged and transform_name != "identity":
                log.warning(f"  [phase13] {transform_name}/{scaler_name} 발산 감지 "
                            f"(max={_y_pred_finite.max() if len(_y_pred_finite) else float('nan'):.1f}, "
                            f"y_max_train={_y_max_train:.1f}, invalid={_frac_invalid:.1%}) → "
                            f"identity fallback retry")
                # Identity transform 으로 재학습 (1회)
                y_tr_id, inv_id, _ = _apply_single_y_transform(y_train_pool, "identity")
                _model_id = factory_fn()
                _model_id.fit(X_tr_s, y_tr_id)
                _y_pred_id = _model_id.predict(X_te_s)
                # G-303 (2026-06-17, 검증 적발): nonneg=True — 이 identity-retry 가 :917 에서 floored
                #   y_pred 를 덮어쓰므로 여기서도 ILI≥0 원공간 floor 를 적용(다른 inverse 사이트와 동형).
                _y_pred_id = sanitize_predictions(
                    np.asarray(inv_id(_y_pred_id), dtype=np.float64), nonneg=True)
                # 새 pred 가 더 안전하면 채택
                _id_finite = _y_pred_id[np.isfinite(_y_pred_id)]
                if len(_id_finite) > 0 and (
                    len(_id_finite) == len(_y_pred_id) and
                    _id_finite.max() <= _y_max_train * 10.0
                ):
                    log.info(f"  [phase13] identity fallback PASS — pred max={_id_finite.max():.1f}")
                    y_pred = _y_pred_id
                    model = _model_id
                    transform_name = "identity_fallback"
        except Exception as _safe_e:
            log.warning(f"  [phase13] safe mode auto-fallback exception: {_safe_e}")

    # G-324 (2026-06-19, 3-AI 검토 + 사용자 #5 일반화 우려): hard cap 를 transform-조건부로.
    #   외삽 blowup 은 **비선형 역변환**(log1p→expm1·sqrt→x²·asinh→sinh·boxcox/yeo_johnson power)서만
    #   발생(TiDE pred=669 vs y_max~95 → R²=-115). identity/linear/affine(none/laplace/mcmc) 모델은
    #   발산 없음 → cap 제외(미래 novel-peak 일반화 제한 해소; ILI>3×train_max 정당 outbreak 도 통과).
    #   uncapped shadow 보존(정직 eval — capped 가 점수 부풀리는지 가시화, 3-AI 권고).
    _y_pred_uncapped = np.asarray(y_pred, dtype=np.float64).copy()
    _NONLINEAR_INV = {"log1p", "sqrt", "asinh", "boxcox", "yeo_johnson"}
    if str(transform_name) in _NONLINEAR_INV:
        try:
            import os as _os_cap
            _cap_mult = float(_os_cap.environ.get("MPH_PRED_CAP_MULT", "3.0"))
            _y_max_tr = float(np.nanmax(y_train_pool)) if len(y_train_pool) else 100.0
            _cap = max(_y_max_tr * _cap_mult, 1.0)
            _yp = np.asarray(y_pred, dtype=np.float64)
            _n_capped = int(np.sum(_yp > _cap))
            if _n_capped:
                y_pred = np.minimum(_yp, _cap)
                log.info(f"  [phase13] pred hard-cap({transform_name} 비선형역변환): {_n_capped} preds "
                         f"→ ≤{_cap:.1f} (={_cap_mult}×y_max_train). identity/linear=cap 제외(G-324).")
        except Exception as _cap_e:
            log.warning(f"  [phase13] pred cap exception: {_cap_e}")

    # G-168 (2026-05-02): ~60 metric 다 계산 (R10 per_model_eval 와 align). 이전: 5-7 metric.
    # G-167: PICP95/80/50 추가 — multi-criteria filter 의 4 번째 criteria 활성화.
    # audit Stage 1.1 (2026-05-27): viral_positivity_train 전달 → KDCA threshold primary.
    out = _compute_full_metrics(
        y_test=y_test, y_pred=y_pred,
        sigma_for_wis=sigma_for_wis,
        y_train_pool=y_train_pool,
        viral_positivity_train=viral_positivity_train,
    )
    # G-326 (2026-06-19, 사용자: 전체 eval 통일 R+P): R9 test_metrics 가 68-subset 이던 것을 full
    #   129-metric SSOT(evaluate_predictions_full) 로 확장 — 기존 키 우선 merge(r2/wis 정의 불변,
    #   신규 metric(auprc·c_index·brier·cost_skill·calibration 등)만 추가). 선정은 oof_wis 라 무영향.
    try:
        from simulation.pipeline.phase_evaluator import evaluate_predictions_full as _epf
        _full129 = _epf(np.asarray(y_test, dtype=np.float64), np.asarray(y_pred, dtype=np.float64),
                        sigma=float(sigma_for_wis),
                        y_train_pool=np.asarray(y_train_pool, dtype=np.float64),
                        phase_id="r9_per_model")
        out = {**_full129, **out}   # 기존(out) 우선 — 정의 보존, 신규만 추가
    except Exception as _e129:   # noqa: BLE001
        log.debug(f"  [phase13] full-129 SSOT merge skip: {_e129}")
    # G-354 (2026-06-25, P1 감사 #4): leak-free in-sample residual (R10 PI 반폭 출처).
    #   옛 R10 else-branch 가 test 점에 self-calibrate(y_test-pred)하던 것을 대체할 누수-free 출처.
    #   ★rolling 모델(FusedEpi/classic-ts/epi/transform-rolling)은 static model.predict(X_tr_s)가
    #     rolling 1-step 오차분포와 불일치(TiRex base obs=None → 상수 base) → model 이 fit() 단계서
    #     산출한 native leak-free 잔차(insample_residuals/_calib_residuals)를 우선 노출.
    #   ★비-rolling(tree/linear/kernel/deep-static)만 train-pool 1-step 정적 예측 잔차
    #     (y_train_pool - transform_inv(model.predict(X_tr_s))) — 동일 sanitize/inverse 순서로 레짐 정합.
    #   추가 refit 0회(native 재사용 또는 predict 1회). test 슬라이스 절대 미접근.
    _res_is_out = None
    try:
        _is_rolling = bool(supports_rolling_eval(model) or supports_transform_rolling(model))
    except Exception:   # noqa: BLE001
        _is_rolling = False
    try:
        _nat = getattr(model, "insample_residuals", None)
        if callable(_nat):
            _nat = _nat()
        if _nat is None:
            _nat = getattr(model, "_calib_residuals", None)
        if _nat is not None:
            _na = np.asarray(_nat, dtype=np.float64).ravel()
            _na = _na[np.isfinite(_na)]
            if len(_na) >= 2:
                _res_is_out = _na.tolist()
        if _res_is_out is None and not _is_rolling:
            _y_is_t = sanitize_predictions(np.asarray(model.predict(X_tr_s), dtype=np.float64))
            _y_is = sanitize_predictions(np.asarray(transform_inv(_y_is_t), dtype=np.float64),
                                         nonneg=True)
            _res_is = np.asarray(y_train_pool, dtype=np.float64).ravel() - _y_is
            _res_is = _res_is[np.isfinite(_res_is)]
            if len(_res_is) >= 2:
                _res_is_out = _res_is.tolist()
    except Exception as _ires_e:   # noqa: BLE001
        log.debug(f"  [phase13] in-sample residual skip "
                  f"({getattr(getattr(model, 'meta', None), 'name', '?')}): {_ires_e}")
        _res_is_out = None
    out["insample_residuals"] = _res_is_out
    out["predictions"] = y_pred.tolist()
    # G-324: uncapped shadow (정직 eval — capped 가 점수 부풀리는지 비교 가능). 비-capped 모델은 == predictions.
    out["predictions_uncapped"] = _y_pred_uncapped.tolist()
    out["was_capped"] = bool(not np.array_equal(_y_pred_uncapped, np.asarray(y_pred, dtype=np.float64)))

    if return_fitted_model:
        out["_fitted_model"] = model  # popped before JSON serialization
        out["_artifact_state"] = {     # for ChampionArtifact bundling
            "transform_name": transform_name,
            "transform_inv_obj": transform_inv_obj,
            "hier_y_state": _hier_y_state,   # G-233: picklable HIER inverse for inference
            "fitted_scaler": fitted_scaler,
            "feature_indices": (list(feature_indices) if feature_indices is not None
                                else None),
        }
    return out


def _build_deploy_artifact(
    factory_fn, best: dict, feature_indices, feature_cols,
    mc_method: str, mc_state, model_name: str, save_dir,
    X_pool, y_pool, X_test, y_test, X_real=None, y_real=None,
) -> Optional[str]:
    """Q5 / G-276: 배포(inference)용 artifact — champion config 동결 + 전체 관측데이터 재학습.

    eval artifact ``<name>.pt`` (train+val fit) 는 hold-out metric 재현용으로 **동결**한다.
    배포는 train+val+test+real 전부로 재학습한 ``<name>_deploy.pt`` 를 쓴다 — 가장 최근
    (= 계절 AR 에서 가장 정보량 큰) 관측까지 반영해 운영 forecast 신선도를 확보. 선택/HP/
    transform 은 hold-out 에서 frozen (누수 없음: generalization 측정은 끝났고 배포 fit 에만
    그 주들을 추가; 표준 deployment 관행). ``_refit_and_predict_test`` 의 검증된 preproc/fit
    경로를 그대로 재사용(dummy test predict 는 버림) → eval 경로와 0 divergence.

    Env: ``MPH_DEPLOY_REFIT=0`` → 비활성(eval .pt 만).

    Args: 대부분 champion artifact 와 동일. X_real/y_real 은 service-zone(없으면 생략).
    Returns: 저장된 ``<name>_deploy.pt`` 경로(str) | None (비활성/실패).
    Side effects: ``save_dir/<name>_deploy.pt`` 1개 write. Performance: 모델당 1회 추가 fit.
    """
    import os as _os_d
    if _os_d.environ.get("MPH_DEPLOY_REFIT", "1") == "0":
        return None
    # G-293 (2026-06-17, 3자 감사): 앙상블은 base 모델 예측 결합이라 단독 _refit_and_predict_test
    #   (is_ensemble/val_predictions 미전달)로는 fit 불가 → deploy refit skip. 배포는 base 모델 deploy 로.
    if str(model_name).startswith("Ensemble-"):
        log.info(f"  [phase13-deploy] {model_name} 앙상블 — deploy refit skip(base 모델 결합)")
        return None
    from pathlib import Path
    pairs = [(X_pool, y_pool), (X_test, y_test), (X_real, y_real)]
    Xs = [np.asarray(bx) for bx, by in pairs if bx is not None and len(np.asarray(bx))]
    ys = [np.asarray(by) for bx, by in pairs if by is not None and len(np.asarray(by))]
    if not Xs:
        return None
    X_full = np.vstack(Xs)
    y_full = np.concatenate(ys)
    try:
        res = _refit_and_predict_test(
            factory_fn,
            transform_name=best.get("transform", "identity"),
            scaler_name=best.get("scaler", "none"),
            hier_frozen_params=best.get("preproc_optuna_params"),
            X_train_pool=X_full, y_train_pool=y_full,
            X_test=X_full[-2:], y_test=y_full[-2:],   # dummy predict (버림) — fitted_model 만 필요
            feature_indices=feature_indices, feature_cols=feature_cols,
            return_fitted_model=True,
        )
    except Exception as _de:
        log.warning(f"  [phase13-deploy] {model_name} refit-full 실패: {_de}")
        return None
    dm = res.get("_fitted_model")
    if dm is None or "error" in res:
        log.warning(f"  [phase13-deploy] {model_name} deploy fit none ({res.get('error')})")
        return None
    ast = res.get("_artifact_state", {})
    from simulation.utils.model_artifact import make_artifact
    art = make_artifact(
        model=dm,
        transform_name=ast.get("transform_name", best.get("transform", "identity")),
        transform_inv_obj=ast.get("transform_inv_obj"),
        hier_y_state=ast.get("hier_y_state"),
        fitted_scaler=ast.get("fitted_scaler"),
        feature_indices=ast.get("feature_indices"),
        mc_method=mc_method, mc_state=mc_state,
        config={"transform": best.get("transform", "identity"),
                "scaler": best.get("scaler", "none"),
                "n_features": (len(feature_indices) if feature_indices is not None
                               else best.get("n_features")),
                "deploy": True, "n_train_full": int(len(y_full))},
        meta={"phase": "phase13_deploy_refit", "n_train_full": int(len(y_full)),
              "note": "fit on train+val+test+real; metrics live on eval (.pt) artifact"},
        model_name=model_name,
    )
    out = Path(save_dir) / f"{model_name}_deploy.pt"
    try:
        out.write_bytes(art.to_pickle_bytes())
        log.info(f"  [phase13-deploy] {model_name} deploy artifact "
                 f"(n_full={len(y_full)} vs eval {len(y_pool)}) → {out.name}")
        return str(out)
    except Exception as _we:
        log.warning(f"  [phase13-deploy] {model_name} deploy save 실패: {_we}")
        return None


# ─────────────────────────────────────────────────────────────────
# Phase C.6 + C.7 (sprint 2026-05-06): Real-Slab Forecasting + ACI
# ─────────────────────────────────────────────────────────────────
def _refit_and_predict_real(
    factory_fn,
    transform_name: str,
    scaler_name: str,
    X_train_pool: np.ndarray, y_train_pool: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    X_real: np.ndarray, y_real: np.ndarray,
    feature_indices: Optional[list[int]] = None,
    sigma_for_wis: float = 1.0,
    feature_cols: Optional[list[str]] = None,
    aci_alpha_star: float = 0.05,
    aci_gamma: float = 0.05,
    test_residuals: Optional[np.ndarray] = None,
    hier_frozen_params: Optional[dict] = None,   # G-233: HIER preproc replay (per-step)
    hier_max_chain_length: int = 2,
) -> dict:
    """Real-Slab Forecasting (methodology §4.1) — rolling-origin 1-step-ahead
    + Adaptive Conformal Inference (Gibbs & Candès 2021 NeurIPS, ACI).

    Phase C.6 + C.7 (sprint 2026-05-06). Service-zone evaluation that
    complements `_refit_and_predict_test` (research zone, n_test=68).

    For each i ∈ [0, n_real):
        train_window = (X_train_pool ⊕ X_test) ⊕ X_real[:i]   (사용자 명시)
        runner = factory_fn(); runner.fit(transform(y), scale(X))
        y_pred_i = runner.predict(X_real[i:i+1])
        ACI: predict_interval(y_pred_i) → observe(y_real[i]) → update α_{i+1}

    Args:
        factory_fn: model factory returning a fresh instance per step.
        transform_name: target transform ("identity", "log1p", "sqrt", ...).
        scaler_name: feature scaler ("none", "standard", "robust", "grouped").
            Note: "grouped" is partially supported via passthrough; the
            full grouped pipeline lives in `_refit_and_predict_test`.
        X_train_pool: train+val features (n_train + n_val rows).
        y_train_pool: train+val target.
        X_test: test features (n_test ~ 68).
        y_test: test target.
        X_real: real features (n_real ~ 8) — service zone.
        y_real: real target (ground-truth, observed).
        feature_indices: per-model feature subset (None = all).
        sigma_for_wis: σ for WIS / cold-start ACI calibrate.
        feature_cols: original feature names (for grouped preproc).
        aci_alpha_star: ACI target miscoverage (0.05 → 95% PI).
        aci_gamma: ACI step size (Gibbs 2021 §3, default 0.05).
        test_residuals: optional in-sample residuals (y_test - test_pred)
            for ACI calibrate. None → cold-start with `sigma_for_wis * 1.96`.

    Returns:
        dict with:
          - predictions: list[float] (n_real rolling-origin predictions)
          - pi95_lo, pi95_hi: list[float] (ACI horizon-aware bands)
          - real_metrics: {mae, rmse, picp95, pi95_mean_width,
                           peak_hit_week_diff, aci_realized_coverage, n}
          - aci_alpha_history: list[float] (length n_real + 1)
          - error: str (set on failure; predictions/PI 미보장)

    Performance: O(n_real × model_fit_time).
        Simple model (GLM, GAM, KRR) ~5 s / step.
        Deep model (DNN, TCN, TimesNet) ~1-2 min / step.
        Foundation (TimesFM-2.5, TiRex) ~30 s / step.
    Side effects: 0 — fresh model + scaler instances per step.
    Caller responsibility: X_real shape match feature space; y_real ground
        truth observed; transform_name & scaler_name from best_config.
    """
    from simulation.analytics.conformal_aci import AdaptiveConformal
    from simulation.models.base import sanitize_predictions

    try:
        n_real = int(len(y_real))
        if n_real == 0:
            return {"error": "y_real empty"}

        # In-sample (research zone) = train+val+test
        X_in = np.vstack([X_train_pool, X_test])
        y_in = np.concatenate([y_train_pool, y_test])

        predictions: list[float] = []

        # Rolling-origin loop (methodology §4.1 표준)
        for i in range(n_real):
            if i > 0:
                X_window = np.vstack([X_in, X_real[:i]])
                y_window = np.concatenate([y_in, y_real[:i]])
            else:
                X_window = X_in
                y_window = y_in

            # G-309 (3자 감사 #4): recode global-summary features leakage-free at THIS rolling origin —
            #   [window; step] 를 train_end=len(window) 기준 재코딩(step 의 quantile/threshold/interaction 이
            #   미래 통계 미사용, baseline/real_eval 와 정합). 컬럼 부재 시 no-op (동작 불변).
            _step_full = X_real[i:i + 1]
            if feature_cols is not None:
                _te = len(X_window)
                _ycat = np.concatenate([np.asarray(y_window, dtype=float).ravel(), np.zeros(1)])
                _Xcat = _recode_advanced_per_fold(np.vstack([X_window, _step_full]),
                                                  _ycat, feature_cols, _te)
                X_window = _Xcat[:_te]
                _step_full = _Xcat[_te:_te + 1]

            # feature subset
            if feature_indices is not None and len(feature_indices) > 0:
                X_w = X_window[:, feature_indices]
                X_step = _step_full[:, feature_indices]
            else:
                X_w = X_window
                X_step = _step_full

            # G-299 (2026-06-17): subset feature names — hoisted so the rolling refit can thread
            #   feature_names into runner.fit (OverseasTransfer encoder locates ili_rate_lag1-4 by
            #   name; the rolling path previously passed only feature_cols → encoder OFF here too).
            _feat_names_step = (
                [feature_cols[j] for j in feature_indices]
                if (feature_cols is not None and feature_indices is not None
                    and len(feature_indices) > 0)
                else feature_cols)

            # target transform + scaler (G-233: HIER replay per-step when params present;
            # else flat-by-name — a "HIER_*" marker without params degrades to identity).
            if hier_frozen_params:
                (y_w_t, transform_inv, _tio, X_w_s, X_step_s,
                 _fs, _tn, _hys) = _hier_replay_preproc(
                    hier_frozen_params, hier_max_chain_length,
                    y_window, X_w, X_step, _feat_names_step)
            else:
                _tn = "identity" if str(transform_name).startswith("HIER_") else transform_name
                y_w_t, transform_inv, _ = _apply_single_y_transform(y_window, _tn)
                X_w_s, X_step_s = X_w, X_step
                if scaler_name in ("standard", "robust", "quantile"):
                    sc = _build_single_x_scaler(scaler_name)
                    if sc is not None:
                        X_w_s = sc.fit_transform(X_w)
                        X_step_s = sc.transform(X_step)
                # "none" / "grouped" / others → passthrough (X_w_s = X_w)

            # fit + predict
            runner = factory_fn()
            try:
                # G-303 (2026-06-17, 검증 적발): rolling 은 fit_predict 경로로 가고 base 가
                #   **kwargs 를 fit 으로 forward → feature_names 를 여기서 전달해야 OverseasTransfer
                #   encoder 가 켜진다(이전 G-299 는 도달 못 하는 except 줄에만 넣어 INERT 였음).
                pred_t = runner.fit_predict(
                    X_train=X_w_s, y_train=y_w_t,
                    X_test=X_step_s,
                    feature_cols=feature_cols,
                    feature_names=_feat_names_step,
                )
            except (TypeError, AttributeError):
                runner.fit(X_w_s, y_w_t, feature_names=_feat_names_step)   # G-299/303: encoder names
                pred_t = runner.predict(X_step_s)

            # inverse transform + sanitize (G-159: NaN/inf → 0.0)
            pred = transform_inv(np.asarray(pred_t, dtype=np.float64).flatten())[0]
            pred = float(sanitize_predictions(np.array([pred]), nonneg=True)[0])   # G-298: ILI≥0
            predictions.append(pred)

        # ACI sequential application (Gibbs & Candès 2021)
        aci = AdaptiveConformal(alpha_star=aci_alpha_star, gamma=aci_gamma)
        if test_residuals is not None and len(test_residuals) > 0:
            aci.calibrate(np.asarray(test_residuals, dtype=np.float64))
        else:
            cold = np.array([float(sigma_for_wis * Z95)] * 10, dtype=np.float64)
            aci.calibrate(cold)

        pi95_lo: list[float] = []
        pi95_hi: list[float] = []
        for i, y_pred in enumerate(predictions):
            lo, hi = aci.predict_interval(y_pred)
            pi95_lo.append(float(lo))
            pi95_hi.append(float(hi))
            aci.update(float(y_real[i]))

        # Section B (real slab) descriptive metrics — methodology §5
        # n=8 small-sample: R² unstable (paper §6.1 caveat),
        # MAPE/SMAPE percentile-based — forecasting 학술 표준
        y_real_arr = np.asarray(y_real, dtype=np.float64)
        pred_arr = np.asarray(predictions, dtype=np.float64)
        pi_lo_arr = np.asarray(pi95_lo, dtype=np.float64)
        pi_hi_arr = np.asarray(pi95_hi, dtype=np.float64)

        # R² (small-n caveat — descriptive only, paper §6.1)
        ss_res = float(np.sum((y_real_arr - pred_arr) ** 2))
        ss_tot = float(np.sum((y_real_arr - y_real_arr.mean()) ** 2))
        r2_real = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-9 else float("nan")
        # MAPE — clip min y to 0.01 (zero-division 차단)
        y_safe = np.maximum(np.abs(y_real_arr), 0.01)
        mape = float(np.mean(np.abs((y_real_arr - pred_arr) / y_safe)) * 100.0)
        # SMAPE — symmetric (Hyndman & Athanasopoulos 2021), bound [0, 200]
        denom = np.abs(y_real_arr) + np.abs(pred_arr) + 1e-9
        smape = float(np.mean(2.0 * np.abs(y_real_arr - pred_arr) / denom) * 100.0)
        # WAPE — weighted absolute percentage error (FluSight-friendly)
        wape = float(np.sum(np.abs(y_real_arr - pred_arr))
                     / max(np.sum(np.abs(y_real_arr)), 1e-9) * 100.0)

        real_metrics = {
            "mae": float(np.mean(np.abs(y_real_arr - pred_arr))),
            "rmse": float(np.sqrt(np.mean((y_real_arr - pred_arr) ** 2))),
            "r2": r2_real,                           # n=8 descriptive caveat
            "mape": mape,                            # %, percentile-based
            "smape": smape,                          # %, symmetric [0, 200]
            "wape": wape,                            # %, weighted (FluSight)
            "picp95": float(np.mean((y_real_arr >= pi_lo_arr)
                                     & (y_real_arr <= pi_hi_arr))),
            "pi95_mean_width": float(np.mean(pi_hi_arr - pi_lo_arr)),
            "peak_hit_week_diff": int(np.argmax(pred_arr) - np.argmax(y_real_arr)),
            "aci_realized_coverage": float(aci.realized_coverage),
            "n": n_real,
        }

        return {
            "predictions": [float(p) for p in predictions],
            "pi95_lo": [float(x) for x in pi95_lo],
            "pi95_hi": [float(x) for x in pi95_hi],
            "real_metrics": real_metrics,
            "aci_alpha_history": [float(a) for a in aci.alpha_history],
        }

    except Exception as e:
        log.warning(f"  [phase13] _refit_and_predict_real failed: {e}")
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────
# G-168 + G-167 (2026-05-02): ~60 metric helper — R10 (per_model_eval) 와 align
# Phase C1 partial (2026-05-12): extracted to simulation/pipeline/metric_eval.py
# Re-exported here under the original _compute_full_metrics name to preserve
# 모든 caller (_evaluate_config, _refit_and_predict_*, _oof_cv_wis, etc.).
# ─────────────────────────────────────────────────────────────────
from simulation.pipeline.metric_eval import compute_full_metrics as _compute_full_metrics



def _recode_advanced_per_fold(X, y, feature_cols, train_end):
    """G-309 (3자 감사 #4, 2026-06-18): R9(per_model_optimize) OOF/refit 의 global-summary feature 누수 차단.

    quantile(``*_qbin``/``*_qnorm``)·``above_threshold``·``{src}_ili`` interaction 은 R1(data) 에서
    GLOBAL 통계(= test+real era 포함)로 한 번 코딩된다. baseline/wfcv/real_eval 은 fold 마다
    ``train_end`` 까지로 **재코딩**하여 누수를 막지만 R9(per_model_optimize/_inline) 은
    그러지 않아 OOF 선택·test/real refit 이 미래-정보 코딩 feature 를 썼다. 이 helper 가 그 3개
    recoder 를 ``X[:train_end]`` 기준으로 적용한다. 해당 컬럼이 없으면(non-MPH_ADVANCED_FEATURES)
    각 recoder 가 silently no-op → 동작 불변.

    Args:
        X: 전체 feature 공간 design matrix (n × p; feature-indexed 이전).
        y: target (above_threshold 의 ``2·median(y[:train_end])`` 용).
        feature_cols: X 컬럼명 (None → no-op).
        train_end: 재코딩 reference 로 쓸 행 범위 [:train_end].

    Returns:
        global-summary 컬럼이 fold-local 로 재코딩된 X (변경 시 새 배열, 없으면 동일 객체). X 불변.
    """
    if feature_cols is None:
        return X
    try:
        from simulation.pipeline.wfcv import (
            _recode_quantile_features_per_fold as _rq,
            _recode_above_threshold_per_fold as _rt,
            _recode_interaction_features_per_fold as _ri,
        )
    except Exception:
        return X
    Xf = _rq(X, feature_cols, train_end)
    Xf = _rt(Xf, y, feature_cols, train_end)
    Xf = _ri(Xf, feature_cols, train_end)
    return Xf


def _oof_cv_wis(
    factory_fn,
    X_train: np.ndarray,
    y_train: np.ndarray,
    transform_name: str,
    scaler_name: str,
    feature_indices: Optional[list[int]] = None,
    sigma_for_wis: float = 1.0,
    feature_cols: Optional[list[str]] = None,
    n_folds: int = 5,
    return_folds: bool = False,
) -> float:
    """Walk-Forward CV → mean OOF WIS on training data only.

    return_folds=True → returns (median_wis, per_fold_wis_list) for a 1-SE/parsimony rule
    (nested size-path guard); default False preserves the scalar return for all existing callers.

    [2026-04-28] Best 결정을 val 27 single → OOF (~150-200 weeks 누적) 로 변경.
    val noise 에 overfit 하던 문제 해결.

    Walk-forward (no peeking):
      n_total = len(X_train)
      fold_size = n_total // (n_folds + 1)
      for k in 1..n_folds:
          end_train = k * fold_size
          end_val   = (k+1) * fold_size
          train on [:end_train], validate on [end_train:end_val]
    """
    from simulation.analytics.diagnostics import weighted_interval_score
    from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS

    n = len(X_train)
    if n < (n_folds + 1) * 10:
        return (float("inf"), []) if return_folds else float("inf")    # too small for CV

    fold_size = n // (n_folds + 1)
    wis_scores = []
    fold_maxes: list = []     # G-265b: regime-conditional 집계용 fold y_val max (wis 와 동일 guard)
    prev_resid = None     # Q1 (2026-05-30): prior-fold OOS residuals → this fold's PI calibration
    for k in range(1, n_folds + 1):
        end_tr = k * fold_size
        end_va = (k + 1) * fold_size if k < n_folds else n
        # G-309 (3자 감사 #4): fold-local recode of global-summary features (quantile/threshold/
        #   interaction) using ONLY [:end_tr] → OOF 선택이 build-time GLOBAL(test+real era) 코딩
        #   누수 미사용 (baseline/wfcv/real_eval 와 정합). 해당 컬럼 부재 시 no-op.
        _Xf = _recode_advanced_per_fold(X_train, y_train, feature_cols, end_tr)
        X_tr = _Xf[:end_tr]
        y_tr = y_train[:end_tr]
        X_va = _Xf[end_tr:end_va]
        y_va = y_train[end_tr:end_va]
        if len(X_va) < 4:
            continue
        cell = _evaluate_config(
            factory_fn, X_tr, y_tr, X_va, y_va,
            transform_name=transform_name, scaler_name=scaler_name,
            feature_indices=feature_indices,
            sigma_for_wis=max(float(np.std(y_tr)), 1e-3),
            feature_cols=feature_cols,
            calib_residuals=prev_resid,   # Q1: leakage-free (fold 0 → in-sample fallback)
            _fast_inner=True,   # G-273c: feature-stability 후보 비교 → 내부 HP study 생략(유일 호출처)
        )
        if "error" not in cell and np.isfinite(cell["wis"]):
            wis_scores.append(_oof_selection_score(cell))
            fold_maxes.append(float(np.max(y_va)))   # G-265b: regime 분류 (wis 와 동일 guard = 정렬)
        _vr = cell.get("_val_residuals")
        if _vr is not None and np.size(_vr) >= 2:
            prev_resid = np.asarray(_vr, dtype=float).ravel()   # calibrate the NEXT fold

    # G-265b (2026-06-13, 3자 리뷰): median→regime-conditional mean 통일 (G-256b 를 champion 선택
    # 경로에도 적용 — median 이 outbreak fold(~2/5) 를 버려 peak-blind 선택하던 것 차단). 1-SE rule 은
    # raw per-fold list 로 그대로(점수만 regime-aware).
    _agg = _fold_variance_penalize(_oof_regime_aggregate(wis_scores, fold_maxes, y_train),
                                   wis_scores)
    return (_agg, list(wis_scores)) if return_folds else _agg


def _oof_cv_metrics(
    factory_fn,
    X_train: np.ndarray,
    y_train: np.ndarray,
    transform_name: str,
    scaler_name: str,
    feature_indices: Optional[list[int]] = None,
    sigma_for_wis: float = 1.0,
    feature_cols: Optional[list[str]] = None,
    n_folds: int = 5,
    hier_frozen_params: Optional[dict] = None,   # G-233: HIER preproc replay (champion gate)
    hier_max_chain_length: int = 2,
) -> dict:
    """Walk-Forward CV → mean OOF {wis, r2, mape, pi95_coverage}.

    G-FIX(2026-05-28): OOF 진단/selection 지표 (champion = best-WIS; 4-criteria gate
    제거 2026-06-05). `_oof_cv_wis` 와 **동일 fold split + 동일 flat `_evaluate_config`**
    경로 → config 선택(oof_cv)과 게이트가 일관. per-fold metric 의 평균.

    Args:
        factory_fn ~ feature_cols: `_oof_cv_wis` 와 동일.
        n_folds: walk-forward fold 수 (default 3).

    Returns:
        {"wis","r2","mape","pi95_coverage","n_folds_used"}.
        CV 불가(n 부족)/전 fold 실패 시 wis=inf, 나머지 NaN, n_folds_used=0.

    Performance: n_folds 회 refit. **champion config 당 1회** 호출(gate 전용) —
        선택 루프(매 config)와 달리 모델당 1회라 비용 작음.
    Caller responsibility: pool(train+val) 만 넘길 것 (test slab 누출 금지).
    """
    n = len(X_train)
    if n < (n_folds + 1) * 10:
        return {"wis": float("inf"), "r2": float("nan"),
                "mape": float("nan"), "pi95_coverage": float("nan"),
                "n_folds_used": 0}

    fold_size = n // (n_folds + 1)
    # G-334 (2026-06-22): fold-불변 inverse-cap 기준 = 전체 train max (floor OOF 도 grid 와 동일 통일).
    try:
        from simulation.pipeline.preproc_optuna_hierarchical import set_y_ref_max as _set_yrm
        _set_yrm(float(np.max(np.asarray(y_train, dtype=np.float64))))
    except Exception:
        pass
    acc: dict[str, list[float]] = {"wis": [], "r2": [], "mape": [], "pi95_coverage": []}
    prev_resid = None     # Q1 (2026-05-30): prior-fold OOS residuals → this fold's PI calibration
    for k in range(1, n_folds + 1):
        end_tr = k * fold_size
        end_va = (k + 1) * fold_size if k < n_folds else n
        X_tr, y_tr = X_train[:end_tr], y_train[:end_tr]
        X_va, y_va = X_train[end_tr:end_va], y_train[end_tr:end_va]
        if len(X_va) < 4:
            continue
        if hier_frozen_params:
            # G-233: replay HIER preproc per fold (OOF selection metric must match HIER config)
            import optuna as _opt_m
            try:
                cell = _evaluate_config_hierarchical(
                    factory_fn, X_tr, y_tr, X_va, y_va,
                    optuna_trial=_opt_m.trial.FixedTrial(dict(hier_frozen_params)),
                    feature_indices=feature_indices,
                    sigma_for_wis=max(float(np.std(y_tr)), 1e-3),
                    feature_cols=feature_cols,
                    max_chain_length=hier_max_chain_length,
                    calib_residuals=prev_resid,   # Q1: leakage-free gate PICP (fold 0 fallback)
                )
            except Exception:
                continue
        else:
            cell = _evaluate_config(
                factory_fn, X_tr, y_tr, X_va, y_va,
                transform_name=transform_name, scaler_name=scaler_name,
                feature_indices=feature_indices,
                sigma_for_wis=max(float(np.std(y_tr)), 1e-3),
                feature_cols=feature_cols,
                calib_residuals=prev_resid,   # Q1: leakage-free gate PICP (fold 0 fallback)
            )
        if "error" in cell:
            continue
        for _m in acc:
            _v = cell.get(_m)
            if _v is not None and np.isfinite(_v):
                acc[_m].append(float(_v))
        _vr = cell.get("_val_residuals")
        if _vr is not None and np.size(_vr) >= 2:
            prev_resid = np.asarray(_vr, dtype=float).ravel()   # calibrate the NEXT fold

    out = {_m: (float(np.mean(_vs)) if _vs
                else (float("inf") if _m == "wis" else float("nan")))
           for _m, _vs in acc.items()}
    out["n_folds_used"] = len(acc["wis"])
    return out




def optimize_one_model(
    model_name: str,
    factory_fn,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    feature_indices: Optional[list[int]] = None,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    feature_cols: Optional[list[str]] = None,
    X_real: Optional[np.ndarray] = None,
    y_real: Optional[np.ndarray] = None,
    val_predictions_dict: Optional[dict] = None,       # G-181: ensemble base-model val preds
    test_predictions_dict: Optional[dict] = None,      # G-181: ensemble base-model test preds
    db_fingerprint: Optional[dict] = None,             # G-235: DB state hash for comparison integrity
    mc_method: str = "none",                           # G-232/G-234: mc filter used before training
    mc_state: Optional[Any] = None,                   # G-232/G-234: mc filter state for inference replay
    viral_positivity_train: Optional[np.ndarray] = None,  # audit Stage 1.1 (2026-05-27) KDCA threshold input
) -> dict:
    """Run hierarchical preproc Optuna for a single model. Returns
    best preproc config + val score + test refit + test metrics.

    Meta-ensemble models (FluSight-Ensemble, Phase-Adaptive) skip Optuna
    and use identity×none fixed (not transform/scaler targets).

    [2026-04-28] MPH_BEST_BY=oof_cv → best 결정을 OOF (3-fold WF-CV) WIS 로
                 변경 (val 27 의 noise overfit 방지). default=val_wis.
    [2026-05-23] flat 7×4 transform×scaler grid 제거 → hierarchical Optuna (G-233).
    [2026-05-24] transforms/scalers/n_trials_per_cell 파라미터 제거 (dead params).
    """
    # ════════════════════════════════════════════════════════════════
    # 옛 frozen Stage-2 feature subset LOADER (LEGACY opt-in 전용)
    # ────────────────────────────────────────────────────────────────
    # 이전 (2026-04-29~2026-05-25): Stage1(phase0a)+Stage2(phase0b) candidate 비교 (MPH_USE_3STAGE).
    # 2026-05-26: phase0a archive — Stage2 만 로드.
    # 현재 (2026-06-01, codex closure): 아래 LOADER 는 **MPH_LEGACY_PERMODEL_FEATURES 로 HARD-GATE**.
    #   기본 경로(env 미설정) = 이 블록 SKIP → feature 선택은 STABILITY(preproc-first, 아래 L1356+) 단독.
    #   frozen subset 은 stale/degenerate(n/p≈2.8)라 폐기됨 — silent 재활성 방지.
    # ════════════════════════════════════════════════════════════════
    stage2_data = None
    # 2026-06-01 (codex closure blocker): 옛 frozen stage2_feature_optuna LOADER 를 LEGACY opt-in 으로
    #   HARD-GATE. 기본 경로는 STABILITY(preproc-first, 아래 L1356+) 단독. 이전엔 파일 존재 시 무조건
    #   로드 → 파일 재생성 시 STABILITY 를 silent override 위험. 현재 파일 archive 라 runtime 동작 불변
    #   (candidates=[], stage2_data=None — 이미 그 경로), 단 미래 silent 재활성 차단. 복원=MPH_LEGACY_PERMODEL_FEATURES=1.
    import os as _os_s2load
    if feature_cols is not None and _os_s2load.environ.get("MPH_LEGACY_PERMODEL_FEATURES"):
        # NB: do NOT re-import Path here — a function-local `from pathlib import Path`
        # makes `Path` local to the WHOLE function, so the champion-log block (~L1687,
        # outside this legacy-gated branch) hits UnboundLocalError when this branch is
        # skipped (default) → every model's .pt write silently fails (G-251/G-252).
        # Module-level Path (L51) already covers all uses.
        import json as _js
        from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)

        # SSOT MPH_OUTPUT_ROOT: 각 reader 는 ITS writer 의 base 와 짝을 맞춘다.
        #   det (stage2_feature_optuna): writer _inline_optuna_3stage = get_results_dir() → redirect-aware.
        #   legacy (optuna_feat_sel_*): writer run_optuna_feature_selection SAVE_DIR = get_results_dir() → redirect-aware (2026-05-29 정정).
        s2_det_p = get_results_dir() / "stage2_feature_optuna" / f"{model_name}.json"
        # 2026-05-28 Format mismatch fix (사용자 명시 "미완료된 것들 해줘"):
        # R3 (external) actual entry (tools/run_optuna_feature_selection.py) 가
        # optuna_feat_sel_<lowercase_key>.json (다른 path + 다른 keys) 출력.
        # adapter 함수로 R9 (per_model_optimize) schema 변환.
        from simulation.pipeline._inline_optuna_3stage import _model_to_optuna_key as _m2k
        _legacy_key = _m2k(model_name)
        s2_legacy_p = get_results_dir() / f"optuna_feat_sel_{_legacy_key}.json"  # SSOT: SAVE_DIR=get_results_dir() (2026-05-29)

        def _adapt_legacy_stage2(legacy: dict) -> dict:
            """tools/ legacy format → R9 (per_model_optimize) expected schema."""
            return {
                "best_feature_subset": legacy.get("selected_features", []),
                "n_selected": legacy.get("n_features_selected", 0),
                "n_features_pool_after_drop": legacy.get("n_features_total", len(feature_cols)),
                # RMSE proxy for OOF WIS (lower better, 같은 direction)
                "best_score_oof_wis": legacy.get("best_rmse", float("inf")),
                "_format_legacy": "tools/run_optuna_feature_selection.py",
                "_legacy_strategy": legacy.get("strategy", "?"),
            }

        candidates = []
        for _tag, _p in (("deterministic", s2_det_p),):
            if _p.exists():
                try:
                    _d2 = _js.loads(_p.read_text())
                    candidates.append((_tag, _p, _d2["best_score_oof_wis"]))
                except Exception:
                    pass
        # legacy fallback (R3 external actual output) — adapter 사용
        if not candidates and s2_legacy_p.exists():
            try:
                _legacy_raw = _js.loads(s2_legacy_p.read_text())
                _legacy_adapted = _adapt_legacy_stage2(_legacy_raw)
                # legacy 의 adapted dict 을 stage2_feature_optuna/<MODEL>.json 으로 cache
                # — 다음 phase13 호출 시 직접 candidates 로 picked up
                s2_det_p.parent.mkdir(parents=True, exist_ok=True)
                s2_det_p.write_text(_js.dumps(_legacy_adapted, indent=2, default=str))
                candidates.append(("legacy_adapted", s2_det_p, _legacy_adapted["best_score_oof_wis"]))
                log.info(f"  [phase13] {model_name} Stage 2 legacy adapter applied: "
                          f"{s2_legacy_p.name} → {s2_det_p.name}")
            except Exception as _le:
                log.warning(f"  [phase13] {model_name} legacy adapter failed: {_le}")

        if candidates:
            # OOF_WIS 최소 선택 (deterministic 우선, 없으면 legacy_adapted)
            candidates.sort(key=lambda x: x[2] if np.isfinite(x[2]) else float("inf"))
            best_method, s2_path, best_oof = candidates[0]
            try:
                stage2_data = _js.loads(s2_path.read_text())
                log.info(f"  [phase13] {model_name} Stage 2 selected: method={best_method} "
                         f"(OOF_WIS={best_oof:.3f}, candidates={len(candidates)})")
                log.info(f"    Stage 2: {stage2_data['n_selected']}/{stage2_data['n_features_pool_after_drop']} selected")
                # Feature subset → indices
                best_features = set(stage2_data["best_feature_subset"])
                stage2_indices = [i for i, c in enumerate(feature_cols) if c in best_features]
                if stage2_indices:
                    feature_indices = stage2_indices
                    log.info(f"    Stage 2 feature_indices: {len(feature_indices)} columns")
                    # G-242 ③: gated feature-neighborhood (default off). MPH_FEAT_NEIGHBORHOOD_K>0
                    # enlarges the frozen pre-stage subset by its k nearest |corr| neighbours so
                    # the per-model preproc/HP + mc(④) search can explore just beyond it.
                    # 3-LLM caution: joint feature search overfits at n≈349 → default 0.
                    import os as _os_fn
                    _feat_k = int(_os_fn.environ.get("MPH_FEAT_NEIGHBORHOOD_K", "0") or "0")
                    if _feat_k > 0:
                        _expanded = _expand_feature_neighborhood(
                            feature_indices, X_train, y_train, _feat_k)
                        if len(_expanded) > len(feature_indices):
                            log.info(f"    G-242 ③ feature-neighborhood K={_feat_k}: "
                                     f"{len(feature_indices)} → {len(_expanded)} columns")
                            feature_indices = _expanded
            except Exception as _3se:
                log.warning(f"  [phase13] {model_name} Stage 2 로드 실패: {_3se}")
                stage2_data = None

    # G-242 (codex+gemini review 2026-05-30): clip a stale original-space feature_indices
    # to the current (possibly mc-reduced) X width before it reaches preproc/HP. Under
    # mc=pca the Stage-2 name-intersection above empties, leaving the incoming
    # original-space index list → X[:, idx] would IndexError on PC columns. No-op for
    # none/vif/corr (in-range indices unchanged).
    _n_cols_now = int(np.asarray(X_train).shape[1])
    _fi_before = None if feature_indices is None else len(feature_indices)
    feature_indices = _clip_feature_indices(feature_indices, _n_cols_now)
    if _fi_before is not None and _fi_before != (0 if feature_indices is None else len(feature_indices)):
        log.warning(f"  [phase13] {model_name} feature_indices clipped "
                    f"{_fi_before}→{'all-cols' if feature_indices is None else len(feature_indices)} "
                    f"(out-of-range vs X cols={_n_cols_now}; likely mc-reduced/pca)")

    # 2026-04-29: ARIMA family + SEIR mechanistic 추가 — grid bypass
    # 이유:
    #   ARIMA family: 자체 SARIMAX 가 boxcox + AIC grid 사용, X 안 받음.
    #                 R9 (per_model_optimize) 의 transform×scaler grid 가 무의미.
    #   SEIR family:  ODE 직접 풀이 (β, γ, σ), X 안 받음.
    #                 R9 grid 가 무의미.
    META_MODELS = {
        # ── 기존 (ensemble meta) ──
        "FluSight-Ensemble", "Phase-Adaptive",
        # ── 2026-04-29 추가: ARIMA family ──
        # G-331 (2026-06-21 저녁, 재학습 데이터): "SARIMA" RE-ADDED to META. 앞선 transform-fix 가
        #   제거(내부 log1p un-hardcode → 데이터-주도 preproc transform)했으나, 같은 처지의 count/TS
        #   형제가 데이터-주도 transform 이 68-주 peak-외삽에서 폭발함을 입증(PoissonAutoreg test
        #   R²=-347, preds 669 vs data ~100). SARIMA 는 ARIMA-family(형제 ARIMA/SARIMAX 이미 META)
        #   → identity×none 으로 일관성 + 폭발 안전.
        "ARIMA", "SARIMAX", "SARIMA",
        # ── G-288 (2026-06-17, 3자 감사): Theta(univariate)·FluSight-Baseline(persistence)도 feature/
        #    transform 무시 → preproc grid(G-335, 순수grid) 무의미 → force-identity/skip.
        "Theta", "FluSight-Baseline",
        # ── G-292 (2026-06-17, 3자 감사): active ensemble 7종은 base 예측 결합 모델 — R9(per_model_optimize) preproc
        #    Optuna 가 val_predictions 부재로 25/25 trial 전부 실패(test_r2=None, 175 trial 낭비). champion 은
        #    R2(baseline) 앙상블 경로서 산출(R9 무관) → META 로 skip(낭비 제거, champion 무영향).
        "Ensemble-NNLS", "Ensemble-NNLS-Filtered", "Ensemble-BMA", "Ensemble-InvRMSE",
        "Ensemble-Diversity", "Ensemble-Adaptive", "Ensemble-ResidualAR",
        # ── 2026-04-29 추가: SEIR mechanistic ──
        "Bayesian-SEIR", "Metapop-SEIR", "SEIR-V2-Forced",  # registry 이름 (class 이름 SEIRForcedForecaster → 수정)
        # ── 2026-04-29 추가: Rt-Augmented (Ridge on Rt 역추정) ──
        "Rt-Augmented",
        # ── G-319 (2026-06-19, 전체 라인업 감사 wjrh3mf5m): count/renewal 모델은 y-transform
        #    (log1p/sqrt/asinh) 받으면 곱셈 renewal(epiestim y*100)·NegBin round(hhh4 np.round)가
        #    변환공간서 실행 후 역변환 = 수학적 무효(hhh4 zero_frac=0.41+pmax=100.4 실측 동시 collapse+폭발).
        #    y 는 ILI-rate 원공간이어야 link/count 구조 유효 → META 로 identity×none 고정.
        "EpiEstim", "Wallinga-Teunis", "hhh4-equivalent", "TSIR",
        # G-331 (2026-06-21 저녁, 재학습 데이터): "PoissonAutoreg" RE-ADDED to META. 앞선 transform-fix
        #   가 제거(내부 log-AR un-hardcode → 데이터-주도 transform)하되 "PoissonAutoreg 는 현재 작동
        #   (+0.466)이라 제외 — 재학습 결과 보고 필요시 추가"라고 명시 보류했음. 재학습이 결론:
        #   preproc 가 HIER_individual 선택 → 68-주 test R²=-347(preds max 669 vs data ~100 = 역변환
        #   폭발). count 모델(Poisson/NB family)은 identity 강제 필수 — 외부 transform 이 peak-외삽서 폭발.
        "PoissonAutoreg",
        # ── G-319b (2026-06-19, 동일 감사): NB-GLM/count 도 내부 log1p(V6 RidgeCV) 또는 round(y)
        #    내장 → runner transform 과 double-transform/round 깨짐 → 역변환 폭발(NegBinGLM pmax=133.9,
        #    GLARMA=167.3 실측). identity 강제로 내부 link 정상화.
        # G-331 (2026-06-21 저녁, 재학습 데이터): "NegBinGLM" RE-ADDED to META. NegBinGLM-Glum 과
        #   byte-identical(둘 다 V6 RidgeCV fallback — 진짜 glum NB 는 macOS SEGFAULT)인데 Glum 만 META
        #   였음 → 같은 모델을 두 방식(하나 identity, 하나 데이터-주도 transform)으로 처리=비일관 + 데이터
        #   -주도 경로가 regress(test 0.927→0.830). twin(NegBinGLM-Glum)과 count-family 폭발 안전
        #   (PoissonAutoreg 참조)에 맞춰 identity 로 재고정.
        "GLARMA", "NegBinGLM-Glum", "NegBinGLM-V7", "NegBinGLM",
        # ── G-319g (2026-06-19, per-model Optuna 효과분석 wat75yc87): EARS-C1/C2/C3 는 tail-mean
        #    detection baseline — HP space 전무(grep 확인)·X feature 미사용 → preproc100+feature+HP
        #    Optuna 100+trial 전부 무효(no-op 낭비 ~수십분/모델). EpiEstim/Theta 와 동급 → META 등재.
        #    공정성: spurious-overfit config 선택 위험 제거(Optuna 검색공간 없음).
        "EARS-C1", "EARS-C2", "EARS-C3",
    }
    # R9 (per_model_optimize): hierarchical preproc Optuna (G-233, 2026-05-23).
    # flat 7×4 transform×scaler grid 제거 → Optuna 4-mode (none/individual/group/categorical) 단독.
    # META_MODELS (ARIMA/SEIR 등) 는 Optuna skip → identity×none 고정.
    sigma = max(float(np.std(y_train)), 1e-3)
    best = None
    trial_results: list[dict] = []   # Optuna trial result accumulator (renamed from 'grid')
    best_by = GLOBAL.training.best_by
    use_preproc_optuna = (
        feature_cols is not None
        and model_name not in META_MODELS
    )

    # ── preproc-FIRST staged order (MPH_PREPROC_FIRST, default 1 since 2026-06-01) ──
    # 사용자 순서: Stage-1 preproc on FULL features → Stage-2 PRINCIPLED feature 선택
    #   (feature_select_corr1se: STABILITY selection — |corr| 재표본 빈도(B subsample × 점수 상위
    #    inner_k 빈도 ≥ π). n-adaptive 점수: 작은 n=|corr|, n≥epv×p=model importance.
    #    옛 |corr| top-k / 1-SE size-search 는 폐기·제거됨(2026-06-01 codex+Gemini 청소).
    # 이게 frozen pre-stage feature map (stale/degenerate, n/p≈2.8) 을 대체. default 1 근거:
    #   실데이터 A/B 순서 중요(VIF Jaccard 0.50) + binary 0/1-over-399 과적합·비현실(3자+실측).
    # Stage-2 OOF callback = _evaluate_config 로 Stage-1 preproc 공간서 per-fold WIS+SE
    #   (prior-fold 잔차 = 누수0). _preproc_first_done=True 면 아래 feature-loaded 경로 skip.
    _preproc_first_done = False
    import os as _os_pf
    # 2026-06-15 (사용자 "GAT 너무 느려"): PyG graph-NN(GAT/GCN)은 macOS 강제 in-process
    # (_INPROCESS_OVERRIDE_DARWIN) + fit ~14s → preproc 100 × OOF 5 ≈ ~110분/모델.
    # preproc(transform×scaler) 민감도가 linear/tree보다 훨씬 낮으므로 graph 모델만 cap한다
    # (다른 모델은 전역 preproc_trials 유지 → 품질 무손상). MPH_PREPROC_TRIALS_GRAPH 로 튜닝(default 12).
    _preproc_n = GLOBAL.training.preproc_trials
    if "GAT" in model_name.upper() or "GCN" in model_name.upper():
        _preproc_n = min(_preproc_n, int(_os_pf.environ.get("MPH_PREPROC_TRIALS_GRAPH", "12")))
        if _preproc_n < GLOBAL.training.preproc_trials:
            log.info(f"  [phase13] {model_name} graph-NN preproc cap → n_trials={_preproc_n} "
                     f"(전역 {GLOBAL.training.preproc_trials}; graph는 preproc 저민감)")
    if use_preproc_optuna and _os_pf.environ.get("MPH_PREPROC_FIRST", "1") == "1":
        try:
            from simulation.pipeline._inline_optuna_3stage import _stage1_preproc_optuna_inline
            from simulation.pipeline.feature_select_corr1se import (
                select_features_stability, make_model_importance_fn, feature_guard_keep,
                build_nested_size_path, select_size_path_1se, resolve_feature_path,
                derive_min_keep_from_stability)
            # Stage 1: preproc Optuna on FULL feature set (feature_indices=None) → best transform/scaler
            best, trial_results = _stage1_preproc_optuna_inline(
                model_name=model_name, factory_fn=factory_fn,
                X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                feature_indices=None, feature_cols=feature_cols, sigma=sigma,
                n_trials=_preproc_n, best_by=best_by,
            )
            _tf = best.get("transform", "identity"); _sc = best.get("scaler", "none")
            # Stage 2: FEATURE OPTIMIZATION = STABILITY SELECTION (Meinshausen-Bühlmann 2010).
            #   7-way bake-off + codex/gemini 1위: 재표본 빈도 기반 → 한 OOF split 과적합 회피(robust),
            #   n 커질수록 강화(scalable). per-k OOF size-search(1-SE 과소적합)·binary(과적합)·forward 능가.
            #   n-adaptive size: subsample=n//2, inner_k=n//20(EPV), 출력 size=빈도≥π 창발(dynamic).
            #   n-adaptive 점수 (C, 2026-06-01): n < epv×p = |corr|(filter, global) — 현 n=242 동작 보존;
            #     n ≥ epv×p (massive) = importance_fn(적용 모델 coef_/importances_/permutation) per-model 자동.
            #     사용자: "data 작다고 무시 말고 massive 대비." threshold 도출(하드코드 n 아님). log1p y 공간.
            _ylog_fs = np.log1p(np.clip(np.asarray(y_train, float).ravel(), 0, None))
            _imp_fn = make_model_importance_fn(factory_fn)   # massive n 에서만 활성 (작은 n 미호출)
            # Phase A (음수 R² 근본수정): target 자기회귀 lag(ili_rate_lag1-52)을 STABILITY 선택에
            #   강제 포함 — 순수 |corr| screen 이 AR backbone 을 drop 하면 forecaster 가 최근 ILI 신호
            #   0 → 음수 R² 붕괴(v2 retrain 회복으로 입증). feature_cols 는 _apply_mc_columns 가
            #   X_train 에 정렬(mc=none 원본/vif·corr 축소 동일) → name→index 안전.
            _ar_lag_mandatory = {c for c in (feature_cols or [])
                                 if isinstance(c, str) and c.startswith("ili_rate_lag")}
            _sel = select_features_stability(X_train, _ylog_fs, epv_ratio=20, seed=42,
                                             importance_fn=_imp_fn,
                                             feature_names=feature_cols,
                                             mandatory=_ar_lag_mandatory)
            if _sel.get("n_forced_mandatory"):
                log.info(f"  [phase13] {model_name} AR-lag force-include +"
                         f"{_sel['n_forced_mandatory']} (음수R² 붕괴 방지)")
            _sel_idx = _clip_feature_indices(
                _sel["selected_indices"], int(np.asarray(X_train).shape[1]))
            # 2026-06-16 (G-274, per-model 감사): STABILITY 가 mc-축소 후 X 에서 일부 모델
            #   (CQR-LightGBM·SVR-Linear·DNN-Conformal·GAT·TCN·TabularDNN·TiDE·BayesianRidge·
            #   OverseasTransfer) 을 [0,2,4] 3-feature 로 collapse → 굶주림(CQR-LightGBM valWIS 10.89,
            #   TabularDNN 12.92). binary guard(아래)도 full(전체 feature) OOF 평가가 작은 fold(n≈40≪p)
            #   모델 fit ValueError 로 죽어 _sel_idx fallback → collapse 미복구. _full_idx(전체) fallback 은
            #   fit 불가라 오답. → STABILITY freq 상위 _floor 보장(healthy sibling 12 와 일치). collapse(<_min)
            #   에만 발동(legit 11/12 선택 무손상; tree 도 정상 12 면 미발동 — CQR-LightGBM 만 tree collapse).
            #   mechanistic(SEIR=X무시) 제외. Default floor is now STABILITY's EPV-derived
            #   inner_k (= n_pool//epv_ratio), not a fixed feature-count constant.
            #   Explicit legacy overrides remain via MPH_FEAT_MIN_KEEP / MPH_FEAT_FLOOR.
            _floor = derive_min_keep_from_stability(_sel, int(np.asarray(X_train).shape[1]))
            _floor_min = _floor
            if model_name not in ("BayesianMCMC",) and len(_sel_idx) < _floor_min:
                _freq_fl = np.asarray(_sel.get("stability", []), float)
                if _freq_fl.size:
                    _p_fl = int(np.asarray(X_train).shape[1])
                    _topf = sorted(int(j) for j in np.argsort(_freq_fl)[::-1][:min(_floor, _freq_fl.size)])
                    if len(_topf) > len(_sel_idx):
                        log.info(f"  [phase13] {model_name} STABILITY floor: "
                                 f"{len(_sel_idx)}→{len(_topf)} (freq-top, collapse<{_floor_min} 복구)")
                        _sel_idx = _clip_feature_indices(_topf, _p_fl)
            # Stage-2 GUARD (사용자 명시 2026-06-01 "각 단계는 이전 단계 대비 개선 보장"):
            #   feature 선택(subset)은 이전 단계(full feature) 대비 OOF-WIS 를 ≥MPH_FEAT_MARGIN
            #   개선할 때만 유지, 아니면 full feature 복원. mc(Stage 3) margin-guard 와 동형.
            #   ILI(AR-지배)처럼 선택이 full 대비 개선 못 하면(실측 full OOF 0.847 ≥ sel 0.839)
            #   자동 full = "개선 보장" 위반 방지. full=feature_indices=None 으로 평가(전체).
            _full_idx = list(range(int(np.asarray(X_train).shape[1])))
            _feat_margin = float(_os_pf.environ.get("MPH_FEAT_MARGIN", "0.02"))
            # PARSIMONY default (사용자 2026-06-01): subset 기본 유지, full 은 subset 이 ≥margin 명백히
            #   나쁠 때만 복원. "개선 안 되면 full(399) 쏟기"가 불편 → 간결한 subset 우선(Part2: 9≈399 정확도).
            #   MPH_FEAT_PARSIMONY=0 시 strict(개선시만 subset, 동등→full) 로 회귀.
            _parsimony = _os_pf.environ.get("MPH_FEAT_PARSIMONY", "1") == "1"
            # NESTED size-path (opt-in, MPH_FEAT_PATH=nested; codex+Gemini 2026-06-01): binary
            #   {subset, full} 가 "너무 극단"이라는 사용자 우려 → π ladder(0.8/0.6/0.4)+full 의 nested
            #   사다리에서 1-SE/parsimony 로 per-model 선택. nested = 제약된 search → 작은 n 의
            #   select-on-OOF overfit 억제(unordered 메뉴는 overfit, bake-off 실증). default=binary(현행).
            # FAMILY-AWARE (사용자 "dl/modern-ts는?" 2026-06-01): deep-NN(category=='dl' — dl-tabular
            #   + modern-ts TCN/N-BEATS/PatchTST/DeepAR/TFT 전부 'dl', 24모델)은 작은-fold OOF 불신뢰
            #   (n≈40/fold underfit → OOF≫test; TabularDNN 실측 nested 손해) → nested 요청돼도 binary
            #   (stability-anchored, robust). resolve_feature_path() 가 gate.
            try:
                from simulation.models.base import REGISTRY as _REG_FP
                _cat_fp = (getattr(getattr(_REG_FP.get(model_name), "meta", None), "category", "") or "")
            except Exception:
                _cat_fp = ""
            # G-295 (2026-06-17, per-model 감사): the kernel feature-floor below gated on
            #   `_cat_fp == "kernel"`, but KRR/SVR-Linear/SVR-RBF all carry meta.category="linear"
            #   (only CATEGORY_MODELS family == "kernel") → the gate NEVER fired and the floor was
            #   dead code, re-exposing the 3-feature starvation it was added (G-274) to block. Resolve
            #   the floor against the CATEGORY_MODELS family (the SSOT for the kernel grouping).
            try:
                from simulation.models.registry import CATEGORY_MODELS as _CATM_FP
                _fam_fp = next((f for f, mm in _CATM_FP.items() if model_name in mm), "")
            except Exception:
                _fam_fp = ""
            _feat_path = resolve_feature_path(
                _os_pf.environ.get("MPH_FEAT_PATH", "binary"), category=_cat_fp, model_name=model_name)
            # G-294 (2026-06-17): the Stage-2 feature guard must score OOF under the SAME
            #   hierarchical preproc Stage-1 chose. best["transform"]/["scaler"] are mode MARKERS
            #   ("HIER_<mode>"), so routing them to the flat _oof_cv_wis raised
            #   ValueError("Unknown Y transform: HIER_*") (preproc_optuna_hierarchical.py:342) →
            #   caught by `except` below → ALWAYS subset fallback ⇒ both the nested 1-SE size-path
            #   and the binary "각 단계 개선" margin guard were silently dead for every HIER model.
            #   Replay the frozen preproc per fold via _oof_cv_wis_hier (same FixedTrial path as
            #   Stage-1 + matching extrapolation_safe). Flat path retained only for non-HIER configs.
            from simulation.pipeline._inline_optuna_3stage import _oof_cv_wis_hier as _oof_hier
            _hier_pp = best.get("preproc_optuna_params") if isinstance(best, dict) else None
            _es_guard = bool(best.get("_extrap_safe", False)) if isinstance(best, dict) else False

            def _guard_oof(_fi, return_folds=False):
                if _hier_pp:
                    return _oof_hier(factory_fn, X_train, y_train, _hier_pp,
                                     feature_indices=_fi, feature_cols=feature_cols,
                                     extrapolation_safe=_es_guard, return_folds=return_folds)
                return _oof_cv_wis(factory_fn, X_train, y_train, _tf, _sc,
                                   feature_indices=_fi, feature_cols=feature_cols,
                                   return_folds=return_folds)
            try:
                if _feat_path == "nested":
                    _p = int(np.asarray(X_train).shape[1])
                    # 2026-06-15 (per-model 감사): kernel(KRR/SVR)은 min_keep=1 nested 사다리가
                    #   1-SE parsimony 로 3-feature collapse → KRR test R²0.30(설계 K=80 굶주림),
                    #   SVR-Linear 0.24. kernel category 만 floor 상향(env MPH_KERNEL_FEAT_FLOOR,
                    #   default 15) — 다른 category(linear/dl/tree) 무영향. floor 는 최소 후보만
                    #   올리고 1-SE 가 여전히 best 선택(SVR-RBF 0.871 등 무손상).
                    # G-328b root fix: feature floor is data-derived from STABILITY inner_k
                    # (EPV rule), so 1-SE cannot collapse to k=1 without a hardcoded count.
                    _mk = derive_min_keep_from_stability(_sel, _p)
                    if _fam_fp == "kernel":   # G-295: CATEGORY family (kernel models = meta.category 'linear')
                        _mk = max(_mk, min(int(np.ceil(np.sqrt(_p))), _p))
                    _cands = [_clip_feature_indices(c, _p) for c in
                              build_nested_size_path(_sel["stability"], _p,
                                                     pi_levels=(0.8, 0.6, 0.4), min_keep=_mk)]
                    _means, _folds_list, _sizes = [], [], []
                    for _c in _cands:
                        _fi = None if len(_c) >= _p else _c     # full → None (전체)
                        _m, _fl = _guard_oof(_fi, return_folds=True)   # G-294: HIER replay
                        _means.append(_m); _folds_list.append(_fl); _sizes.append(len(_c))
                    _pick = select_size_path_1se(_means, _sizes, fold_scores=_folds_list,
                                                 margin=_feat_margin, se_mult=1.0)
                    feature_indices = _cands[_pick]
                    # G-355 (설계#8): 선정=배포 일치 — 챔피언 선정 OOF 는 배포될 SUBSET 의 OOF 여야 한다.
                    #   Stage-1(feature_indices=None)이 best['oof_wis']/folds 를 FULL pool 로 채웠고
                    #   guard 가 feature_indices 를 subset 으로 줄였으나 그 subset OOF 를 안 기록 →
                    #   return(oof_wis/oof_wis_folds)이 full-pool 수치 보고(선정≠배포). _means[_pick]/
                    #   _folds_list[_pick] = 배포 subset 의 OOF·fold 벡터(이미 산출, 추가 compute 0).
                    if np.isfinite(_means[_pick]):
                        best["oof_wis"] = float(_means[_pick])
                        best["oof_wis_folds"] = (list(_folds_list[_pick])
                                                 if _folds_list[_pick] else None)
                        best["_oof_wis_source"] = "subset_guard_nested"
                    _best_oof = min((m for m in _means if np.isfinite(m)), default=float("inf"))
                    _fk = (f"NESTED sizes={_sizes} → k={_sizes[_pick]} "
                           f"(oof {_means[_pick]:.3f}, best {_best_oof:.3f}, 1-SE/parsimony)")
                else:
                    # G-355 (설계#8): fold 벡터까지 포착(return_folds=True) — 선정 OOF·fold 안정성(G-339
                    #   _oof_fold_cv tiebreaker)을 배포될 feature set 으로 기록하려면 folds 필요.
                    #   OOF 평가 횟수 불변(이미 내부서 fold 계산); means 만 guard 에 전달(scalar 계약 유지).
                    _oof_full, _folds_full = _guard_oof(None, return_folds=True)   # G-294 HIER (full)
                    _oof_sel, _folds_sel = _guard_oof(_sel_idx, return_folds=True) # G-294 HIER (subset)
                    if feature_guard_keep(_oof_full, _oof_sel, _feat_margin, prefer_subset=_parsimony):
                        feature_indices = _sel_idx
                        _picked_oof, _picked_folds = _oof_sel, _folds_sel
                        _fk = f"SUBSET(n={len(_sel_idx)}, oof {_oof_sel:.3f} vs full {_oof_full:.3f})"
                    else:
                        feature_indices = _full_idx                 # subset 명백 열위 → full 복원
                        _picked_oof, _picked_folds = _oof_full, _folds_full
                        _fk = f"FULL(n={len(_full_idx)}; subset oof {_oof_sel:.3f} 명백열위 vs full {_oof_full:.3f})"
                    # G-355: 배포 feature set 의 OOF 를 챔피언 선정값으로 기록(선정=배포 일치)
                    if np.isfinite(_picked_oof):
                        best["oof_wis"] = float(_picked_oof)
                        best["oof_wis_folds"] = list(_picked_folds) if _picked_folds else None
                        best["_oof_wis_source"] = "subset_guard_binary"
            except Exception as _fg_err:
                feature_indices = _sel_idx                       # guard 실패 → subset (back-compat)
                _fk = f"SUBSET(guard 실패: {type(_fg_err).__name__})"
            _preproc_first_done = True
            log.info(
                f"  [phase13] {model_name} PREPROC-FIRST + STABILITY+GUARD({_feat_path if _feat_path == 'nested' else ('parsimony' if _parsimony else 'strict')}): "
                f"tf={_tf}/{_sc}, {_fk} (mode={_sel['mode']}, inner_k={_sel['inner_k']}, "
                f"margin={_feat_margin}, mb_min_n={_sel['model_based_min_n']})")
        except Exception as _pf_err:
            log.warning(
                f"  [phase13] {model_name} preproc-first+corr1se 실패 → 기본 순서 fallback: {_pf_err}")
            best = None
            _preproc_first_done = False

    # META_MODELS 또는 feature_cols 없음 → Optuna skip, identity×none 고정
    if not use_preproc_optuna:
        log.info(f"  [phase13] {model_name} = meta/mechanistic 또는 feature_cols 없음 "
                 f"→ identity×none 고정")
        best = {"transform": "identity", "scaler": "none"}
        # ── G-332 (2026-06-21, codex blocker): META/identity 모델은 preproc trial 이 없어
        #    best["oof_wis"] 가 비어 → 직렬화서 inf → G-318 챔피언 selector(rerank_champion.py:47
        #    `continue` / per_model_eval inf-후순위)가 epi 챔피언 후보(NegBinGLM-Glum·ARIMA·
        #    PoissonAutoreg·NegBinGLM·hhh4·EpiEstim)를 통째로 silent 제외. identity config 의
        #    5-fold WF-OOF WIS(non-META 와 동일 _oof_cv_wis 추정기·동일 sigma·동일 fold)를 계산해
        #    champion-eligible 화 = 공정 비교. SEIR mechanistic/ensemble 은 비-후보(자체 champion
        #    경로 G-292)+OOF 고비용 → skip(inf 유지=report-only). 실패도 inf 유지(do-no-harm).
        _skip_meta_oof = (str(model_name).startswith("Ensemble-")
                          or "SEIR" in str(model_name)
                          or model_name in {"Rt-Augmented", "FluSight-Ensemble", "Phase-Adaptive"})
        if not _skip_meta_oof:
            try:
                _meta_oof = _oof_cv_wis(
                    factory_fn, X_train, y_train, "identity", "none",
                    feature_indices=feature_indices, sigma_for_wis=sigma,
                    feature_cols=feature_cols,
                    n_folds=GLOBAL.training.oof_folds)   # G-332b: CLI fold override 존중(non-META 일치)
                if isinstance(_meta_oof, (int, float)) and np.isfinite(_meta_oof):
                    best["oof_wis"] = float(_meta_oof)
                    best["_oof_wis_source"] = "meta_identity_oof"
                    log.info(f"  [G-332 meta-oof] {model_name}: identity OOF-WIS="
                             f"{_meta_oof:.3f} → champion-eligible")
                else:
                    log.info(f"  [G-332 meta-oof] {model_name}: OOF 비유한 → inf(report-only)")
            except Exception as _moe:
                log.warning(f"  [G-332 meta-oof] {model_name} OOF 실패 → inf 유지: {_moe}")

    if use_preproc_optuna:
        try:
            import optuna as _opt_lib
            _opt_lib.logging.set_verbosity(_opt_lib.logging.WARNING)
        except ImportError:
            log.warning(f"  [phase13] optuna 없음 → {model_name} identity fallback")
            use_preproc_optuna = False
            best = {"transform": "identity", "scaler": "none"}

    if use_preproc_optuna and not _preproc_first_done:
        n_preproc_trials = _preproc_n
        log.info(f"  [phase13] optimizing {model_name} "
                 f"(Optuna preproc mode, n_trials={n_preproc_trials})")

        # 2026-05-28 (사용자 명시 design A B1): preproc Optuna logic 이동.
        # Origin: 본 함수 안 75 LOC (_preproc_objective + study create + best find).
        # 이동 후: simulation.pipeline._inline_optuna_3stage._stage1_preproc_optuna_inline.
        from simulation.pipeline._inline_optuna_3stage import _stage1_preproc_optuna_inline
        best, trial_results = _stage1_preproc_optuna_inline(
            model_name=model_name,
            factory_fn=factory_fn,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            feature_indices=feature_indices,
            feature_cols=feature_cols,
            sigma=sigma,
            n_trials=n_preproc_trials,
            best_by=best_by,
        )

    # 2026-05-23: flat 7×4 grid (for tf in transforms: for sc in scalers:) 제거됨.
    # preproc 선택은 위 Optuna 경로 (4-mode hierarchical: none/individual/group/categorical) 단독.
    # 실패 시 identity×none fallback (위 exception handler 처리).
    if use_preproc_optuna:
        # R9 training-history task: persist finalized per-model Optuna trials.
        # G-269b: 보조기능(history)이 load-bearing 챔피언 학습을 죽이면 안 됨 — 실패 시 log+continue (G-237 역방향 방지).
        try:
            save_training_record(
                model_name=model_name,
                scope="pooled",
                record_type="optuna_trial",
                history_obj=trial_results,
                out_dir=_get_results_dir() / "training_history",
                params_json=json.dumps(best.get("preproc_optuna_params", {}), default=str),
            )
        except Exception as _hist_exc:
            log.warning("training-history(optuna) 저장 실패 — 학습 계속: %s", _hist_exc)

    # ── Test refit: refit on (train+val) with best config, predict on test
    #    AND save fitted model with champion-challenger logic ──
    test_metrics: dict = {}
    refit_test_predictions: Optional[list] = None
    _insample_residuals: Optional[list] = None   # G-354: leak-free R10 PI 출처 (config dict 가 if-블록 밖 → 상단 init 으로 NameError 회피)
    champion_decision: str = "skipped"
    real_result: Optional[dict] = None   # G-FIX (2026-05-24): init before outer if — prevents NameError when X_test is None
    if X_test is not None and y_test is not None and len(y_test) > 0:
        try:
            X_pool = np.vstack([X_train, X_val])
            y_pool = np.concatenate([y_train, y_val])
            # G-181 (2026-05-05) — Ensemble val_predictions path fix:
            # ensemble 카테고리 감지 시 base 모델 dict 전달.
            _is_ensemble = False
            _val_pred_dict = None
            _test_pred_dict = None
            try:
                _meta = factory_fn().meta
                if getattr(_meta, 'category', '') == 'meta':
                    _is_ensemble = True
                    # G-181 fix (2026-05-24): explicit named params from function signature.
                    # 이전: 'kwargs' in dir() → 항상 False → dict 항상 empty.
                    _val_pred_dict = val_predictions_dict or {}
                    _test_pred_dict = test_predictions_dict or {}
            except Exception:
                pass
            # 진단 전용(default 미설정=무영향): MPH_FORCE_FEATURE_INDICES=232,282,... → feature 선택
            #   결과를 특정 인덱스로 덮어씀(과거 config 재현·디버깅용). HP 는 선택 feature 로 이미 튜닝됐을
            #   수 있음(TabPFN 등 HP-light 모델엔 무해). 잘못된 인덱스는 _refit 의 shape-guard 가 잡음.
            import os as _osff
            _ffi = _osff.environ.get("MPH_FORCE_FEATURE_INDICES", "").strip()
            if _ffi:
                try:
                    feature_indices = [int(_x) for _x in _ffi.split(",") if _x.strip()]
                    log.info(f"  [FORCE_FEAT] {model_name}: feature_indices 강제 "
                             f"{len(feature_indices)}개 (진단)")
                except Exception as _ffe:
                    log.warning(f"  [FORCE_FEAT] {model_name} parse 실패: {_ffe}")
            # ── G-12V (2026-06-21, codex+rigor 재설계, 옛 G-328c/PART E 대체): do-no-harm floor 를
            #    단일 27-주 val(G-132 'n=27 single 거절' 위반·고분산) + 중첩 tail 게이트(상관·joint
            #    error 통제 0)에서 **5-fold walk-forward OOF**(R9 를 고른 동일 추정기)로 교체. R9 는
            #    OOF-best 라 대개 floor 미발동 → 단일-val 의 잘못된 override(GAM: val→identity test
            #    0.483 < OOF→log1p 0.656) 차단. baseline(identity+BASIC)이 OOF-WIS 에서 margin(5%)
            #    이상 우수할 때만 floor = 진짜 do-no-harm. 빈 BASIC indices = 전체 feature 를 BASIC 으로
            #    오평가하던 silent fallback(codex) → hard skip. floored config 에 finite oof_wis 기록
            #    (G-318 selector 자격). 옛 tail-floor 는 audit-only 로 강등(중첩 게이트 제거).
            if (not _is_ensemble and not str(model_name).startswith("Ensemble-") and feature_cols):
                try:
                    import os as _osf
                    from simulation.pipeline.baseline import BASIC_FEATURE_COLS as _BFC
                    _bidx = [feature_cols.index(c) for c in _BFC if c in feature_cols]
                    if _osf.environ.get("MPH_VAL_FLOOR", "1") != "1":
                        # G-12V (codex 비용 게이트): floor = 모델당 2× 5-fold OOF. default on(정의 run
                        #   ~5-10% 추가, correctness). fast/탐색 모드는 MPH_VAL_FLOOR=0 으로 비용 절감.
                        log.info(f"  [G-12V OOF-floor] {model_name} skip: MPH_VAL_FLOOR=0 (비용 절감)")
                    elif len(_bidx) < 2:
                        log.info(f"  [G-12V OOF-floor] {model_name} skip: BASIC feature "
                                 f"indices 부재({len(_bidx)}) — silent all-feature 오평가 방지(hard skip)")
                    else:
                        _margin = float(_osf.environ.get("MPH_DO_NO_HARM_MARGIN", "0.05"))
                        # R9 config 5-fold OOF (HIER preproc replay) vs baseline(identity+BASIC) OOF
                        _r9_oof = _oof_cv_metrics(
                            factory_fn, X_train, y_train,
                            best.get("transform", "identity"), best.get("scaler", "none"),
                            feature_indices=feature_indices, feature_cols=feature_cols,
                            hier_frozen_params=best.get("preproc_optuna_params")).get("wis")
                        _bl_oof = _oof_cv_metrics(
                            factory_fn, X_train, y_train, "identity", "none",
                            feature_indices=_bidx, feature_cols=feature_cols,
                            hier_frozen_params=None).get("wis")
                        _both_ok = (isinstance(_r9_oof, (int, float)) and np.isfinite(_r9_oof)
                                    and isinstance(_bl_oof, (int, float)) and np.isfinite(_bl_oof))
                        _floor_fire = _both_ok and (_bl_oof < _r9_oof * (1.0 - _margin))
                        log.info(f"  [G-12V OOF-floor] {model_name}: R9 OOF-WIS "
                                 f"{round(_r9_oof, 3) if _both_ok else _r9_oof} vs baseline "
                                 f"{round(_bl_oof, 3) if _both_ok else _bl_oof} (margin {_margin}) → "
                                 f"{'baseline 채택(do-no-harm)' if _floor_fire else 'R9 유지'}")
                        if _floor_fire:
                            best = {"transform": "identity", "scaler": "none",
                                    "preproc_optuna_params": None,
                                    "oof_wis": float(_bl_oof), "_floored_oof": True}
                            feature_indices = _bidx
                except Exception as _bf_exc:
                    log.warning(f"  [G-12V OOF-floor] {model_name} skip: {_bf_exc}")
            test_result = _refit_and_predict_test(
                factory_fn,
                transform_name=best.get("transform", "identity"),
                scaler_name=best.get("scaler", "none"),
                # G-233: replay HIER preproc (None for flat/META → flat path, unchanged)
                hier_frozen_params=best.get("preproc_optuna_params"),
                X_train_pool=X_pool, y_train_pool=y_pool,
                X_test=X_test, y_test=y_test,
                feature_indices=feature_indices,
                feature_cols=feature_cols,   # G-FIX (2026-06-03): 누락 시 _refit_and_predict_test 의
                                              # feat_names_use=None → replay feature_groups=None → "all_features"
                                              # 로 범주화 → 검색의 실제-group(x_group_<name>) frozen params 와 불일치
                                              # → grouped preproc FixedTrial replay "x_group_all_features not found"
                                              # → refit_test_predictions=None. 검색(L440)과 동일 feat_names 전달로 일치.
                sigma_for_wis=sigma,
                return_fitted_model=True,
                # G-181: ensemble dict (caller 가 kwargs 로 전달)
                val_predictions_dict=_val_pred_dict,
                val_actual=y_val if _is_ensemble else None,
                test_predictions_dict=_test_pred_dict,
                is_ensemble=_is_ensemble,
                # audit Stage 1.1 (cascade #1, 2026-05-27) — KDCA threshold input
                viral_positivity_train=viral_positivity_train,
            )
            # §8.6 symmetric 재평가 (MPH_SYMMETRIC_REEVAL=1) — ML fit-once 비대칭 제거:
            #   전 모델을 매 origin frozen-config 재fit(ARIMA rolling 과 진짜 동일). EVAL 예측/metric
            #   만 대체(deploy artifact = fit-once 전체학습 유지). ensemble/foundation 제외는 호출 비용상.
            import os as _os_sym
            if (_os_sym.environ.get("MPH_SYMMETRIC_REEVAL", "0").strip() == "1"
                    and not _is_ensemble and X_test is not None and len(X_test) > 0
                    and "error" not in test_result):
                try:
                    _sym = _symmetric_rolling_eval(
                        factory_fn, best.get("transform", "identity"), best.get("scaler", "none"),
                        X_pool, y_pool, X_test, y_test,
                        feature_indices=feature_indices, feature_cols=feature_cols,
                        hier_frozen_params=best.get("preproc_optuna_params"), sigma_for_wis=sigma)
                    if _sym.get("n", 0) > 0 and _sym.get("predictions") is not None:
                        for _k in ("_fitted_model", "_artifact_state"):
                            if _k in test_result:
                                _sym[_k] = test_result[_k]     # deploy artifact(fit-once) 보존
                        test_result = _sym
                        log.info(f"  [§8.6 symmetric] {model_name}: refit-per-origin "
                                 f"wis={round(_sym.get('wis', float('nan')), 3)} "
                                 f"r2={round(_sym.get('r2', float('nan')), 3)} (비대칭 fit-once 대체)")
                except Exception as _se:
                    log.warning(f"  [§8.6 symmetric] {model_name} skip: {_se} → fit-once 유지")

            # ②③④ 보조 진단 metric (게이트 — 비용상 default off, 최종 eval 시 env=1로 on). do-no-harm.
            if (not _is_ensemble and X_test is not None and len(X_test) > 0
                    and "error" not in test_result):
                _tr_nm = best.get("transform", "identity"); _sc_nm = best.get("scaler", "none")
                _dkw = dict(feature_indices=feature_indices, feature_cols=feature_cols,
                            hier_frozen_params=best.get("preproc_optuna_params"), sigma_for_wis=sigma)
                if _os_sym.environ.get("MPH_MULTI_SEED", "0").strip() == "1":
                    try:
                        test_result["multi_seed"] = _multi_seed_metrics(
                            factory_fn, _tr_nm, _sc_nm, X_pool, y_pool, X_test, y_test, **_dkw)  # #13 재현성
                    except Exception as _e:
                        log.warning(f"  [#13 multi-seed] {model_name} skip: {_e}")
                if _os_sym.environ.get("MPH_MULTI_HORIZON", "0").strip() == "1":
                    try:
                        test_result["multi_horizon"] = _direct_multihorizon_eval(
                            factory_fn, _tr_nm, _sc_nm, X_pool, y_pool, X_test, y_test, **_dkw)   # ③ decay
                    except Exception as _e:
                        log.warning(f"  [③ multi-horizon] {model_name} skip: {_e}")
                _fm = test_result.get("_fitted_model")
                if (_os_sym.environ.get("MPH_NATIVE_NB_WIS", "0").strip() == "1"
                        and _fm is not None and hasattr(_fm, "predict_quantiles")):
                    try:
                        _Xt = X_test[:, feature_indices] if feature_indices is not None else X_test
                        _nb = _native_interval_wis(_fm, _Xt, y_test)                              # ④ count 구간
                        if _nb:
                            test_result["native_nb"] = _nb
                    except Exception as _e:
                        log.warning(f"  [④ native-NB-WIS] {model_name} skip: {_e}")

            if "error" not in test_result:
                fitted_model = test_result.pop("_fitted_model", None)
                artifact_state = test_result.pop("_artifact_state", {}) or {}
                # G-168 (2026-05-02): ~60 metric 모두 보존 (이전: 5-키 추출 → 정보 손실).
                # _refit_and_predict_test 는 compute_full_metrics 통해 ~60 metric 다 반환.
                # 'predictions' 만 따로 빼고 나머지 metric 키 모두 test_metrics 에.
                refit_test_predictions = test_result.pop("predictions", None)
                _insample_residuals = test_result.pop("insample_residuals", None)  # G-354
                test_metrics = dict(test_result)  # 모든 metric 보존
                # R9 training-history task: deep module detects DL/Lightning/closed-form.
                # G-269b: history 실패가 champion-challenger 블록을 건너뛰면 안 됨 — log+continue.
                try:
                    save_training_record(
                        model_name=model_name,
                        scope="pooled",
                        record_type="auto",
                        history_obj={"model": fitted_model, "metrics": test_metrics},
                        out_dir=_get_results_dir() / "training_history",
                        params_json=json.dumps(best.get("preproc_optuna_params", {}), default=str),
                    )
                except Exception as _hist_exc:
                    log.warning("training-history(fitted) 저장 실패 — 학습 계속: %s", _hist_exc)
                # G-168/G-167: ~60 metric 중 핵심 지표 7개 + PI coverage log
                _picp95 = test_metrics.get("pi95_coverage", float("nan"))
                _mape = test_metrics.get("mape", float("nan"))
                log.info(
                    f"  [phase13] {model_name} TEST (refit on train+val): "
                    f"WIS={test_metrics['wis']:.3f} "
                    f"MAE={test_metrics['mae']:.3f} "
                    f"R²={test_metrics['r2']:.3f} "
                    f"MAPE={_mape:.1f}% "
                    f"PICP95={_picp95:.3f} (n={test_metrics['n']})"
                )

                # R8.2 (2026-05-26): full 129-key SSOT eval on refit test predictions.
                # Trajectory: R9 trial-best → refit → R10 (per_model_eval) SSOT.
                # Provides paper_top{2,3,5,10}_complete at refit state (g175_*_pass 제거 2026-06-05).
                try:
                    from simulation.pipeline.phase_evaluator import evaluate_predictions_full
                    if refit_test_predictions is not None:
                        _y_arr = np.asarray(y_test, dtype=np.float64)
                        _p_arr = np.asarray(refit_test_predictions, dtype=np.float64)
                        _mask = np.isfinite(_y_arr) & np.isfinite(_p_arr)
                        if _mask.sum() >= 5:
                            full_r8 = evaluate_predictions_full(
                                y_test=_y_arr[_mask],
                                y_pred=_p_arr[_mask],
                                residuals=(_y_arr[_mask] - _p_arr[_mask]),
                                sigma=sigma,
                                y_train_pool=y_pool,
                                threshold=GLOBAL.filter.alert_threshold,
                                phase_id=f"phase12_refit_{model_name}",
                                enable_bootstrap_ci=False,
                            )
                            test_metrics["phase_eval_r8"] = full_r8
                except Exception as _e:
                    test_metrics["phase_eval_r8_err"] = str(_e)

                # ─────────────────────────────────────────────────
                # Phase C.6 + C.7 (sprint 2026-05-06): service zone (real slab)
                # rolling-origin 1-step-ahead + ACI Gibbs2021 — methodology §4.1
                # ─────────────────────────────────────────────────
                real_result = None
                if (X_real is not None and y_real is not None
                        and len(y_real) > 0):
                    try:
                        _test_resid = None
                        if refit_test_predictions is not None:
                            _test_resid = (np.asarray(y_test, dtype=np.float64)
                                            - np.asarray(refit_test_predictions,
                                                         dtype=np.float64))
                        real_result = _refit_and_predict_real(
                            factory_fn,
                            transform_name=best.get("transform", "identity"),
                            scaler_name=best.get("scaler", "none"),
                            X_train_pool=X_pool, y_train_pool=y_pool,
                            X_test=X_test, y_test=y_test,
                            X_real=X_real, y_real=y_real,
                            feature_indices=feature_indices,
                            sigma_for_wis=sigma,
                            feature_cols=feature_cols,
                            test_residuals=_test_resid,
                            # G-233: HIER preproc replay per-step (None for flat → unchanged)
                            hier_frozen_params=best.get("preproc_optuna_params"),
                        )
                        if "error" not in real_result:
                            _rm = real_result["real_metrics"]
                            log.info(
                                f"  [phase13] {model_name} REAL (service zone, "
                                f"n={_rm['n']}): MAE={_rm['mae']:.3f} "
                                f"PICP95={_rm['picp95']:.3f} "
                                f"peak_Δw={_rm['peak_hit_week_diff']:+d}"
                            )
                        else:
                            log.warning(f"  [phase13] {model_name} REAL failed: "
                                         f"{real_result['error']}")
                    except Exception as ree:
                        log.warning(f"  [phase13] {model_name} REAL exception: "
                                     f"{ree}")
                        real_result = {"error": str(ree)}
                # Champion-challenger: bundle (model + scaler + transform_state +
                # feature_indices) into a ChampionArtifact so R10 (per_model_eval) can
                # reproduce the exact pipeline on new X. Save .pt only if the
                # bundle beats the current champion — G-258 (2026-06-12, codex): selection score
                # = leakage-free OOF-WIS (not test_wis). test was being used to pick which re-train
                # bundle to keep → test-as-selection bias. test metrics kept in meta for reporting.
                if fitted_model is not None:
                    try:
                        from simulation.utils.champion_log import ChampionLog
                        from simulation.utils.model_artifact import make_artifact
                        # G-176 (2026-05-05): ensemble fallback default
                        artifact = make_artifact(
                            model=fitted_model,
                            transform_name=artifact_state.get("transform_name",
                                                                best.get("transform", "identity")),
                            transform_inv_obj=artifact_state.get("transform_inv_obj"),
                            # G-233: HIER inference replays via this state (not transform_name)
                            hier_y_state=artifact_state.get("hier_y_state"),
                            fitted_scaler=artifact_state.get("fitted_scaler"),
                            feature_indices=artifact_state.get("feature_indices"),
                            feature_cols=None,
                            # G-232/G-234 (2026-05-25): store mc filter state so R10 (per_model_eval)
                            # can replay filter on full X_inference before apply_features()
                            mc_method=mc_method,
                            mc_state=mc_state,
                            config={
                                "transform": best.get("transform", "identity"),
                                "scaler":    best.get("scaler", "none"),
                                "n_features": (len(feature_indices) if feature_indices is not None
                                               else best.get("n_features")),
                            },
                            # G-168 (2026-05-02): meta 에 ~60 metric 다 들어감
                            # (champion_log audit + R12 comprehensive deep-dive 활용).
                            # 안전한 float 변환 (NaN 도 허용 — JSON null 직렬화).
                            meta={**{f"test_{k}": (
                                        float(v) if isinstance(v, (int, float)) and v == v else None
                                     )
                                     for k, v in test_metrics.items()
                                     if k != "predictions"},
                                  "phase": "phase13_per_model_optimize"},
                            model_name=model_name,   # ← tier 라벨 자동
                        )
                        # Champion 승격 = 순수 best-WIS (사용자 명시 2026-06-05: 4-criteria
                        #   완전 제거 — R²/MAPE/PICP 게이트·진단 모두 삭제. WIS 가 유일 기준.
                        #   개별 R²/MAPE/PICP 는 129-key eval 에 일반 metric 으로만 존재).
                        # G-176 (2026-05-05): ensemble fallback default
                        # .pt SSOT: mirror config.get_model_dir() (config not in
                        # scope here; GLOBAL is) so the champion lands where R11 (shap)
                        # and Pinf (inference) load it — env-set → <root>/
                        # results/models_pt, else project-local ./models. Removes the
                        # silent CWD-relative write + env-set split (was Path("models")).
                        from simulation.config_global import GLOBAL as _G_md
                        _md = (Path(_G_md.paths.output_root) / "results" / "models_pt"
                               if _G_md.paths.output_root else Path("models"))
                        cl = ChampionLog(
                            models_dir=_md,
                            log_path=_md / "champion_log.json",
                        )
                        # G-258: leakage-free champion score = OOF-WIS (selection metric), not
                        # test_wis. Fall back to test_wis only if OOF is missing/non-finite.
                        _oof_wis = best.get("oof_wis", float("inf"))
                        _champ_score = (float(_oof_wis) if np.isfinite(_oof_wis)
                                        else float(test_metrics["wis"]))
                        decision = cl.propose(
                            name=model_name,
                            pickle_bytes=artifact.to_pickle_bytes(),
                            new_score=_champ_score,
                            metric="oof_wis",
                            lower_better=True,
                            config={
                                "transform": best.get("transform", "identity"),
                                "scaler":    best.get("scaler", "none"),
                                "n_features": (len(feature_indices) if feature_indices is not None
                                               else best.get("n_features")),
                                "artifact":   "ChampionArtifact",
                            },
                            extra_metrics={"test_wis": float(test_metrics["wis"]),
                                           "mae":  test_metrics["mae"],
                                           "r2":   test_metrics["r2"],
                                           "rmse": test_metrics["rmse"]},
                            db_fingerprint=db_fingerprint,  # G-235
                        )
                        champion_decision = decision
                        # Q5 / G-276: 이 모델이 새 champion 이면 배포용 artifact 도 전체-데이터 재학습.
                        #   eval .pt(train+val)=metric 동결, _deploy.pt=train+val+test+real=운영 forecast.
                        if decision in ("promoted", "no_current"):
                            _build_deploy_artifact(
                                factory_fn, best, feature_indices, feature_cols,
                                mc_method, mc_state, model_name, _md,
                                X_pool, y_pool, X_test, y_test, X_real, y_real,
                            )
                    except Exception as ce:
                        log.warning(f"  [phase13] {model_name} champion-log "
                                     f"failed: {ce}")
            else:
                log.warning(f"  [phase13] {model_name} test refit failed: "
                             f"{test_result['error']}")
        except Exception as e:
            log.warning(f"  [phase13] {model_name} test refit exception: {e}")

    # transform-fix follow-up (2026-06-21, PART G): the do-no-harm/baseline-floor (~L2282/2305) and
    #   the META/mechanistic skip (~L2184) set `best` to a metric-bare {transform:identity,...} dict,
    #   so val_metrics.oof_wis below defaulted to float('inf'). The G-318 champion selector
    #   (rerank_champion._load / per_model_eval.select_champion_g318) SKIPS non-finite oof_wis → in
    #   the 2026-06-21 run 35/49 models (incl count/epi BayesianMCMC/NegBinGLM/Poisson) were silently
    #   dropped from the OOF shortlist, leaving a winner's-curse champion. Backfill the model's best
    #   COMPLETED-trial OOF-WIS as the finite shortlist signal (deployed config may differ from the
    #   trial, but this restores shortlist ELIGIBILITY; hold-out test WIS still arbitrates the final
    #   champion). 0-trial META baselines stay non-finite by design.
    _bof = best.get("oof_wis")
    if (not isinstance(_bof, (int, float))) or (not np.isfinite(_bof)):
        _fin_trials = [t for t in (trial_results or [])
                       if isinstance(t, dict) and isinstance(t.get("oof_wis"), (int, float))
                       and np.isfinite(t.get("oof_wis"))]
        if _fin_trials:
            _bt = min(_fin_trials, key=lambda t: t["oof_wis"])
            best["oof_wis"] = float(_bt["oof_wis"])
            if (not isinstance(best.get("wis"), (int, float))
                    or not np.isfinite(best.get("wis", float("nan")))):
                _bw = _bt.get("wis")
                if isinstance(_bw, (int, float)):
                    best["wis"] = float(_bw)
            best["_oof_wis_source"] = "min_trial_backfill"   # PART G audit flag (override visibility)

    # G-176 (2026-05-05): ensemble fallback default
    return {
        "model": model_name,
        "best_config": {
            "transform": best.get("transform", "identity"),
            "scaler": best.get("scaler", "none"),
            # 2026-06-15 (per-model 감사): n_features = 실제 선택 feature 수(len(feature_indices)),
            #   post-mc pool-width(398/348/300/52) stale 아님(49/52 불일치, 챔피언 포함).
            #   ARIMA/SARIMA 등 feature_indices None 모델은 best.get 유지.
            "n_features": (len(feature_indices) if feature_indices is not None
                           else best.get("n_features")),
            "feature_indices": feature_indices,
            # G-FIX (2026-06-03): HIER preproc replay 파라미터 영속화 — ensemble(NNLS)이 base 모델
            # val 예측을 재현하려면 hier 상태 필요. 이전엔 best_config 에 transform/scaler 이름만 있어
            # _hier_replay_preproc 재현 불가. 추가 계산 0(이미 best 에 있음) → main 무영향.
            "preproc_optuna_params": best.get("preproc_optuna_params"),
        },
        "val_metrics": {"wis": best.get("wis", float("nan")),
                         # G-307 (3자 감사 #1, 2026-06-18): OOF-CV WIS(5-fold WF-CV expanding-window)
                         #   = cross-model 챔피언 **선정** 메트릭(누수-free; G-258 champion-challenger 와 동일
                         #   키, G-132 single-val 금지). 위 'wis'(best['wis'])는 단일 train→val split = 보고용.
                         #   지금까지 반환 dict 에 미노출이라 R10 이 hold-out test 로 선정(winner's curse).
                         "oof_wis": best.get("oof_wis", float("inf")),
                         # G-339 (2026-06-24): per-fold OOF 벡터 carry — leak-free 챔피언의 1-SE
                         #   band + fold 안정성 tiebreaker(per_model_eval.select_champion_g318) 용.
                         #   결손(META/0-trial)이면 None → 챔피언 selector 가 margin+parsimony 로 graceful.
                         "oof_wis_folds": best.get("oof_wis_folds"),
                         # G-354 (2026-06-25, P1 감사 #4): leak-free in-sample residual (train-pool
                         #   fit error 또는 model native conformal cal-split 잔차). R10 PI 반폭 보정의
                         #   누수-free 출처 — test-residual(y_test-pred) self-calibration 대체.
                         "insample_residuals": _insample_residuals,
                         "mae": best.get("mae", float("nan")),
                         "rmse": best.get("rmse")},
        "test_metrics": test_metrics,                      # ← TEST slab metrics ★
        "refit_test_predictions": refit_test_predictions,  # ← test predictions
        # Phase C.6 + C.7 (sprint 2026-05-06): service zone (real slab) ★
        "refit_real_predictions": (real_result.get("predictions")
                                    if isinstance(real_result, dict)
                                    and "predictions" in real_result else None),
        # ILI rate is nonneg — clip the deployed lower band at 0 (the point preds are
        # already sanitized nonneg; an un-clipped ACI band otherwise emits negative
        # lower bounds that violate the rate domain and over-widen the interval).
        "refit_real_pi95_lo": ([max(0.0, float(v)) for v in real_result["pi95_lo"]]
                                if isinstance(real_result, dict)
                                and isinstance(real_result.get("pi95_lo"), (list, tuple))
                                else None),
        "refit_real_pi95_hi": (real_result.get("pi95_hi")
                                if isinstance(real_result, dict)
                                and "pi95_hi" in real_result else None),
        "real_metrics": (real_result.get("real_metrics")
                          if isinstance(real_result, dict)
                          and "real_metrics" in real_result else None),
        "aci_alpha_history": (real_result.get("aci_alpha_history")
                               if isinstance(real_result, dict)
                               and "aci_alpha_history" in real_result else None),
        # Back-compat: legacy "best_metrics" field still mirrors val_metrics
        # G-176 (2026-05-05): ensemble fallback default
        "best_metrics": {"wis": best.get("wis", float("nan")),
                          "mae": best.get("mae", float("nan")),
                          "rmse": best.get("rmse")},
        "champion_decision": champion_decision,  # promoted | kept_current | no_current
        "optuna_trial_results": trial_results,
    }


# ════════════════════════════════════════════════════════════════
# G-234 (2026-05-24): MPH_MULTICOLLINEARITY=auto 자동 선택
# ════════════════════════════════════════════════════════════════
# Phase A/B 별도 worker 불필요 — run_per_model_optimize() 내부에서 4-method 비교 후 최적 선택.
# 대표 모델 5개 × 4 method (none/vif/corr/pca) quick fit+predict → median WIS 최소 method.
# cache: simulation/results/mc_method_auto_selected.json (동일 파이프라인 재실행 시 재사용).

_AUTO_MC_PROBE_PREFERRED = [
    "XGBoost", "LightGBM", "ElasticNet", "KRR", "SVR-Linear",
    "RandomForest", "BayesianRidge",
]
_AUTO_MC_N_PROBE = 5
_AUTO_MC_CACHE_NAME = "mc_method_auto_selected.json"


def _auto_select_mc_method(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_cols: "list[str] | None",
    model_factories: dict,
    *,
    out_dir: "Path | str | None" = None,
    force: bool = False,
) -> str:
    """4-method multicollinearity 자동 비교 → best method 반환 (G-234).

    none / vif / corr / pca 를 대표 모델 5개로 빠르게 비교.
    Optuna 없이 단일 fit+predict → val WIS 기준 median 최소 method 선택.

    Args:
        X_train: 학습 feature array (n_train × p)
        y_train: 학습 타겟 (n_train,)
        X_val:   검증 feature array (n_val × p)
        y_val:   검증 타겟 (n_val,)
        feature_cols: 컬럼 이름 목록 (VIF/corr 에 필요)
        model_factories: {name: callable} — R9 (per_model_optimize) model_factories 전달
        out_dir: cache JSON 저장 경로
        force: True 시 cache 무시 후 재실행

    Returns:
        best_method: 'none' | 'vif' | 'corr' | 'pca'

    Side effects:
        out_dir / mc_method_auto_selected.json 에 비교 결과 저장
    """
    import datetime as _dt
    from simulation.pipeline.mc_filter_stage3 import (
        apply_multicollinearity_filter,
    )
    if out_dir is None:  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
        from simulation.utils.paths import get_results_dir
        out_dir = get_results_dir()

    cache_path = Path(out_dir) / _AUTO_MC_CACHE_NAME

    # ── cache hit ──────────────────────────────────────────────
    if not force and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            method = cached.get("best_method", "none")
            log.info(
                f"  [phase13-auto] mc_method cache hit: '{method}' "
                f"(probe={cached.get('probe_models', [])})"
            )
            return method
        except Exception:
            pass

    # ── 대표 모델 선택 ────────────────────────────────────────
    probe_names = [n for n in _AUTO_MC_PROBE_PREFERRED if n in model_factories]
    probe_names = probe_names[:_AUTO_MC_N_PROBE]
    if not probe_names:
        probe_names = list(model_factories.keys())[:_AUTO_MC_N_PROBE]

    log.info(
        f"  [phase13-auto] 4-method 비교 시작 "
        f"(probe={probe_names}, n_val={len(y_val)})"
    )

    method_median: dict[str, float] = {}

    for mc_method in ("none", "vif", "corr", "pca"):
        # ── filter 적용 ──────────────────────────────────────
        try:
            if mc_method == "none":
                Xtr, Xva = X_train, X_val
                fc = feature_cols
            else:
                _Xte_dummy = np.zeros((1, X_train.shape[1]))
                Xtr, Xva, _, mc_state, mc_meta = apply_multicollinearity_filter(
                    X_train, X_val, _Xte_dummy, y_train,
                    feature_cols=list(feature_cols) if feature_cols else None,
                    method=mc_method,
                )
                if mc_method in ("vif", "corr") and feature_cols:
                    fc = [feature_cols[i] for i in mc_state]
                elif mc_method == "pca":
                    fc = [f"PC{i+1}" for i in range(Xtr.shape[1])]
                else:
                    fc = feature_cols
                n_kept = mc_meta.get("n_kept", Xtr.shape[1])
                log.info(
                    f"  [phase13-auto] {mc_method}: {n_kept} features kept"
                )
        except Exception as _fe:
            log.warning(f"  [phase13-auto] {mc_method} filter failed: {_fe} — skip")
            continue

        # ── probe 모델별 fit+predict → WIS ─────────────────
        wis_list: list[float] = []
        for mname in probe_names:
            try:
                model = model_factories[mname]()
                y_pred_va, _, sigma_va = model.fit_predict(Xtr, y_train, Xva)

                # WIS 계산 (simple normal-quantile PI)
                from simulation.analytics.diagnostics import (
                    weighted_interval_score,
                )
                from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS
                _sigma = float(np.nanstd(y_train) or 1.0) if sigma_va is None else None
                _sig = np.full(len(y_val), _sigma) if sigma_va is None else sigma_va
                _wis = weighted_interval_score(
                    y_val, y_pred_va, _sig, alphas=FLUSIGHT_ALPHAS
                )
                # G-FIX (2026-05-24): weighted_interval_score returns per-sample array.
                # float(array) raises ValueError when n_val>1 → silent discard of all WIS.
                # Fix: mean over samples before scalar cast.
                if _wis is not None:
                    _wis_scalar = float(np.mean(_wis)) if hasattr(_wis, '__len__') else float(_wis)
                    if not np.isnan(_wis_scalar):
                        wis_list.append(_wis_scalar)
            except Exception as _me:
                log.debug(f"  [phase13-auto] {mc_method}/{mname}: {_me}")

        if wis_list:
            med = float(np.median(wis_list))
            method_median[mc_method] = med
            log.info(
                f"  [phase13-auto] {mc_method}: median_WIS={med:.4f} "
                f"({len(wis_list)}/{len(probe_names)} 모델 성공)"
            )
        else:
            log.warning(
                f"  [phase13-auto] {mc_method}: 모든 probe 모델 실패 → 제외"
            )

    if not method_median:
        log.warning(
            "  [phase13-auto] 모든 method 실패 → fallback 'none'"
        )
        return "none"

    best_method = min(method_median, key=method_median.__getitem__)
    log.info(
        f"  [phase13-auto] 최적 method: '{best_method}' "
        f"(scores: {{{', '.join(f'{m}={v:.3f}' for m, v in sorted(method_median.items()))}}})"
    )

    # ── cache 저장 ────────────────────────────────────────────
    try:
        cache_data = {
            "best_method": best_method,
            "method_median_wis": method_median,
            "probe_models": probe_names,
            "timestamp": _dt.datetime.now().isoformat(),
        }
        cache_path.write_text(json.dumps(cache_data, indent=2))
    except Exception as _ce:
        log.debug(f"  [phase13-auto] cache 저장 실패: {_ce}")

    # ── CSV 저장 (4-method 비교, 가시성 + G-234 감사용) ────────────────
    # columns: method | median_wis | selected
    # none/vif/corr/pca 각 방법의 probe WIS 기록 → pipeline 재현 시 방법 선택 근거 확인.
    try:
        import csv as _csv_mc
        csv_path = Path(out_dir) / "mc_method_comparison.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as _f_mc:
            _wmc = _csv_mc.DictWriter(
                _f_mc, fieldnames=["method", "median_wis", "selected"]
            )
            _wmc.writeheader()
            for _m_mc, _wis_mc in sorted(method_median.items()):
                _wmc.writerow({
                    "method": _m_mc,
                    "median_wis": round(_wis_mc, 4),
                    "selected": "Y" if _m_mc == best_method else "",
                })
        log.info(f"  [phase13-auto] mc_method_comparison.csv → {csv_path}")
    except Exception as _ce_csv:
        log.debug(f"  [phase13-auto] CSV 저장 실패: {_ce_csv}")

    return best_method


def _oof_cv_wis_with_mc(
    factory_fn,
    X_train: np.ndarray,
    y_train: np.ndarray,
    transform_name: str,
    scaler_name: str,
    mc_method: str,
    feature_cols: Optional[list[str]] = None,
    n_folds: int = 2,
) -> float:
    """Walk-forward OOF mean WIS with PER-FOLD multicollinearity filtering (G-242).

    Unlike `_oof_cv_wis` (fixed `feature_indices` subset), this re-fits the mc filter
    INSIDE each fold's train split, so the OOF estimate is honest for *all* methods —
    in particular ``pca`` is a fitted transform (not an index subset), and fitting it
    once on the whole train pool would leak future structure into earlier folds.

    Args:
        factory_fn: callable() -> BaseForecaster
        X_train: train feature array (n × p), chronological order
        y_train: train target (n,)
        transform_name / scaler_name: preproc passed through to `_evaluate_config`
        mc_method: 'none' | 'vif' | 'corr' | 'pca'
        feature_cols: column names (needed by vif/corr)
        n_folds: walk-forward folds (default 2 for the cheap pre-pass)

    Returns:
        mean OOF WIS over folds, or float('inf') if data too small / all folds failed.

    Side effects: none. Performance: n_folds default-HP fits. No test data touched.
    """
    from simulation.pipeline.mc_filter_stage3 import apply_multicollinearity_filter

    n = len(X_train)
    if n < (n_folds + 1) * 10:
        return float("inf")
    fold_size = n // (n_folds + 1)
    wis_scores: list[float] = []
    fold_maxes: list[float] = []   # G-265b: regime-conditional 집계용 fold y_val max
    for k in range(1, n_folds + 1):
        end_tr = k * fold_size
        end_va = (k + 1) * fold_size if k < n_folds else n
        X_tr, y_tr = X_train[:end_tr], y_train[:end_tr]
        X_va, y_va = X_train[end_tr:end_va], y_train[end_tr:end_va]
        if len(X_va) < 4:
            continue
        if mc_method == "none":
            Xtr_f, Xva_f = X_tr, X_va
        else:
            try:
                _dummy = np.zeros((1, X_tr.shape[1]))
                Xtr_f, Xva_f, _, _st, _mt = apply_multicollinearity_filter(
                    X_tr, X_va, _dummy, y_tr,
                    feature_cols=list(feature_cols) if feature_cols else None,
                    method=mc_method,
                )
            except Exception:
                continue
        cell = _evaluate_config(
            factory_fn, np.asarray(Xtr_f), y_tr, np.asarray(Xva_f), y_va,
            transform_name=transform_name, scaler_name=scaler_name,
            feature_indices=None,
            sigma_for_wis=max(float(np.std(y_tr)), 1e-3),
            feature_cols=None,
            _fast_inner=True,   # G-273c: mc-probe = method 비교 → 내부 HP study 생략
        )
        if "error" not in cell and np.isfinite(cell.get("wis", float("inf"))):
            wis_scores.append(float(cell["wis"]))
            fold_maxes.append(float(np.max(y_va)))   # G-265b: regime 분류 (wis 와 동일 guard)
    # G-265b (2026-06-13, 3자 리뷰): median→regime-conditional mean 통일 (G-256b champion 경로 적용).
    return _oof_regime_aggregate(wis_scores, fold_maxes, y_train)


# ── G-236/G-249 (2026-06-10): R9 (per_model_optimize) per-model subprocess isolation ─────────
# The WF-CV path (MultiModelRunner) isolates every individual model category in a
# subprocess (_SUBPROCESS_CATEGORIES incl. 'epi') so in-process OMP/BLAS thread
# accumulation can't crash a run (G-236). R9 BYPASSED that: the mc probe and
# the optimize loop fit models directly in the parent. After torch+lightgbm+
# statsmodels load their own libomp, an IRLS-heavy epi model (GLARMA: two NegBin
# GLM.fit) tips the polluted OMP runtime → "OMP: Error #179 pthread_mutex_init"
# (a PROCESS abort, uncatchable by try/except). These helpers route each model's
# fits through a fresh child (clean OMP state, crash-contained) reusing the SAME
# G-236 category decision.
def _phase13_isolate_model(model_name: str) -> bool:
    """True iff this model's R9 (per_model_optimize) fits should run in an isolated subprocess.

    Reuses the G-236 category gate (``_should_use_subprocess``): the SAME models
    isolated in WF-CV (dl/tree/linear/ts/epi/physics; NOT 'meta' ensembles; macOS
    PyG/MPS forced in-process) are isolated here. OFF when ``MPH_PHASE13_ISOLATE=0``.

    Args:
        model_name: registered model name.
    Returns:
        bool — isolate this model's fits in a subprocess.
    Side effects: none (reads env + REGISTRY, both immutable post-load).
    """
    try:
        from simulation.pipeline._phase13_isolation import phase13_isolation_enabled
        if not phase13_isolation_enabled():
            return False
        from simulation.models.base import REGISTRY
        from simulation.models.runner import _should_use_subprocess
        cls = REGISTRY.get(model_name)
        if cls is None:
            return False
        return bool(_should_use_subprocess(cls.meta.category, model_name))
    except Exception:
        return False


def _probe_one_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    transform_name: str = "identity",
    scaler_name: str = "standard",
    feature_cols: Optional[list[str]] = None,
    n_folds: int = 2,
    factory=None,
) -> dict:
    """One model's none/vif/corr/pca OOF-WIS cells + best method (G-242 ④, one group).

    Extracted verbatim from ``_compare_mc_per_model`` so the identical computation
    runs both in-process and inside an isolated subprocess worker — zero behaviour
    change when isolation is off.

    Args:
        model_name: registered model name (rebuilds the factory when isolated).
        X_train / y_train: chronological train pool.
        transform_name / scaler_name: common preproc for a fair mc comparison.
        feature_cols: column names (vif/corr).
        n_folds: OOF folds.
        factory: optional ``callable() -> forecaster``; rebuilt from REGISTRY when
            None (the subprocess worker passes None — lambdas don't pickle).
    Returns:
        ``{"best": method, "cells": {method: {oof_wis, insample_wis, overfit_gap,
        overfit_ratio, n_kept}}}``. "best" = min finite OOF WIS, else "none".
    Performance: ~4 × n_folds default-HP fits. Side effects: none.
    """
    from simulation.pipeline.mc_filter_stage3 import apply_multicollinearity_filter
    if factory is None:
        from simulation.models.base import REGISTRY
        _cls = REGISTRY.get(model_name)
        if _cls is None:
            return {"best": "none", "cells": {}, "__error__": f"{model_name} not registered"}
        factory = (lambda c=_cls: c())
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    cells: dict[str, dict] = {}
    for method in ("none", "vif", "corr", "pca"):
        oof = _oof_cv_wis_with_mc(
            factory, X_train, y_train, transform_name, scaler_name,
            method, feature_cols=feature_cols, n_folds=n_folds,
        )
        insample = float("inf")
        n_kept = int(X_train.shape[1])
        try:
            if method == "none":
                Xtr_f = X_train
            else:
                _d = np.zeros((1, X_train.shape[1]))
                Xtr_f, _, _, _st, _mt = apply_multicollinearity_filter(
                    X_train, X_train, _d, y_train,
                    feature_cols=list(feature_cols) if feature_cols else None,
                    method=method,
                )
                n_kept = int(_mt.get("n_kept", np.asarray(Xtr_f).shape[1]))
            _cell = _evaluate_config(
                factory, np.asarray(Xtr_f), y_train, np.asarray(Xtr_f), y_train,
                transform_name=transform_name, scaler_name=scaler_name,
                feature_indices=None,
                sigma_for_wis=max(float(np.std(y_train)), 1e-3),
                feature_cols=None,
                _fast_inner=True,   # G-273c: mc-probe insample baseline → 내부 HP study 생략
            )
            if "error" not in _cell:
                insample = float(_cell.get("wis", float("inf")))
        except Exception:
            pass
        _both = np.isfinite(oof) and np.isfinite(insample)
        gap = (oof - insample) if _both else float("nan")
        ratio = (oof / insample) if (_both and insample > 1e-9) else float("nan")
        cells[method] = {"oof_wis": oof, "insample_wis": insample,
                         "overfit_gap": gap, "overfit_ratio": ratio, "n_kept": n_kept}
    finite = {m: c["oof_wis"] for m, c in cells.items() if np.isfinite(c["oof_wis"])}
    best = min(finite, key=finite.__getitem__) if finite else "none"
    return {"best": best, "cells": cells}


def _ensure_registry_populated() -> None:
    """Populate REGISTRY in a FRESH subprocess (the child re-imports nothing by default).

    Without this, ``REGISTRY.get(name)`` returns None in the child → every isolated
    model silently falls back to 'not registered'. Idempotent + cached.
    """
    try:
        from simulation.models.registry import verify_registry_coverage
        verify_registry_coverage(force_import=True)
    except Exception:
        pass


def _mc_probe_worker(payload: dict) -> dict:
    """Subprocess entry for one model's mc probe (addressed by run_isolated)."""
    _ensure_registry_populated()
    return _probe_one_model(
        payload["mname"], payload["X_train"], payload["y_train"],
        transform_name=payload.get("transform_name", "identity"),
        scaler_name=payload.get("scaler_name", "standard"),
        feature_cols=payload.get("feature_cols"),
        n_folds=int(payload.get("n_folds", 2)),
    )


def _optimize_worker(payload: dict) -> dict:
    """Subprocess entry for one model's full ``optimize_one_model`` (via run_isolated).

    Rebuilds the factory from REGISTRY (lambdas don't pickle) and runs the SAME
    ``optimize_one_model`` in a fresh process; GLOBAL config + REGISTRY re-initialise
    from inherited env. Returns the ``optimize_one_model`` result dict verbatim.
    """
    _ensure_registry_populated()
    from simulation.models.base import REGISTRY
    _cls = REGISTRY.get(payload["mname"])
    _factory = (lambda c=_cls: c()) if _cls is not None else None
    return optimize_one_model(
        payload["mname"], _factory,
        payload["X_train"], payload["y_train"],
        payload["X_val"], payload["y_val"],
        feature_indices=payload.get("feature_indices"),
        X_test=payload.get("X_test"), y_test=payload.get("y_test"),
        feature_cols=payload.get("feature_cols"),
        X_real=payload.get("X_real"), y_real=payload.get("y_real"),
        db_fingerprint=payload.get("db_fingerprint"),
        mc_method=payload.get("mc_method", "none"),
        mc_state=payload.get("mc_state"),
        viral_positivity_train=payload.get("viral_positivity_train"),
    )


def _compare_mc_per_model(
    model_factories: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    feature_cols: Optional[list[str]] = None,
    out_dir: "Path | str | None" = None,
    transform_name: str = "identity",
    scaler_name: str = "standard",
    n_folds: int = 2,
    force: bool = False,
) -> "tuple[dict, str, list]":
    """Per-model none/vif/corr/pca comparison → OOF WIS + overfit gap → CSV (G-242 ④).

    Generalizes the GLOBAL `_auto_select_mc_method` to **per-model** visibility with two
    upgrades the user asked for: (1) honest **per-fold OOF WIS** instead of a single
    val-slab fit, (2) an explicit **overfit gap** (oof_wis − insample_wis) per cell so
    one can SEE which method controls overfitting for each model.

    For each model × method:
        oof_wis      = `_oof_cv_wis_with_mc` (per-fold mc, no leakage)
        insample_wis = full-train fit → predict-train WIS (optimistic baseline)
        overfit_gap  = oof_wis − insample_wis   (larger ⇒ more overfit)
        n_kept       = features surviving the filter
    selected = 'Y' on the per-model min OOF WIS.

    Args:
        model_factories: {name: callable() -> forecaster} to compare (caller scopes the
            set — probes by default, all-registered when MPH_MC_COMPARE_ALL=1).
        X_train / y_train: chronological train pool.
        feature_cols: column names (vif/corr).
        out_dir: results dir (SSOT MPH_OUTPUT_ROOT default).
        transform_name / scaler_name: common preproc for a fair mc comparison (the real
            per-model preproc is searched later in R9 (per_model_optimize); mc choice is ~orthogonal).
        n_folds / force / cache: cheap pre-pass, cached on mc_per_model_selection.csv.

    Returns:
        (per_model_best: {model: method}, global_best: str, rows: list[dict])
        global_best = method minimizing the MEDIAN OOF WIS across models (stable global
        choice for the non-restructured apply path).

    Side effects: writes <out_dir>/mc_per_model_selection.csv. No test data touched.
    Performance: ~ len(models) × 4 × n_folds default-HP fits.
    """
    import csv as _csv
    if out_dir is None:
        from simulation.utils.paths import get_results_dir
        out_dir = get_results_dir()
    csv_path = Path(out_dir) / "mc_per_model_selection.csv"
    meta_path = csv_path.with_suffix(".meta.json")

    # ── fingerprint: invalidate the cache when the model set / data shape / feature names
    #    / preproc / folds change. Codex G-242 review caught that a stale CSV could
    #    silently drive the mc=auto choice (the silent-failure class — G-237). ──
    import hashlib as _hl
    _fp = "|".join([
        ",".join(sorted(model_factories.keys())),
        str(int(X_train.shape[0])), str(int(X_train.shape[1])),
        _hl.md5((",".join(feature_cols) if feature_cols else "").encode()).hexdigest()[:8],
        str(n_folds), str(transform_name), str(scaler_name),
    ])

    # ── cache hit (fingerprint must match) ─────────────────────
    if not force and csv_path.exists() and meta_path.exists():
        try:
            if json.loads(meta_path.read_text()).get("fingerprint") == _fp:
                cached = list(_csv.DictReader(csv_path.open(encoding="utf-8")))
                if cached:
                    per_model_best = {
                        r["model"]: r["method"] for r in cached if r.get("selected") == "Y"
                    }
                    medians: dict[str, list[float]] = {}
                    for r in cached:
                        try:
                            medians.setdefault(r["method"], []).append(float(r["oof_wis"]))
                        except (ValueError, KeyError):
                            pass
                    med = {m: float(np.median(v)) for m, v in medians.items() if v}
                    gbest = min(med, key=med.__getitem__) if med else "none"
                    log.info(f"  [mc-per-model] cache hit (fingerprint OK): "
                             f"{len(per_model_best)} models, global='{gbest}'")
                    return per_model_best, gbest, cached
            else:
                log.info("  [mc-per-model] fingerprint mismatch → recompute (stale cache ignored)")
        except Exception:
            pass

    rows: list[dict] = []
    per_model_best: dict[str, str] = {}
    import os as _os_p13
    from simulation.pipeline._phase13_isolation import run_isolated
    _probe_timeout = float(_os_p13.environ.get("MPH_MC_PROBE_TIMEOUT", "900"))
    _n_probe = len(model_factories)
    for i_m, (mname, factory) in enumerate(model_factories.items(), 1):
        _mc_t0 = time.time()
        log.info(f"  [mc-per-model] {i_m}/{_n_probe} probing {mname} …")
        # G-236/G-249: isolate each model's probe in a fresh process when its category
        # is OMP-fragile — a child OMP #179 abort is CONTAINED here instead of killing
        # the whole mc pre-pass (which, pre-save, would lose ALL R9 per_model_optimize work).
        if _phase13_isolate_model(mname):
            _pres = run_isolated(
                "simulation.pipeline.per_model_optimize:_mc_probe_worker",
                {"mname": mname, "X_train": np.asarray(X_train),
                 "y_train": np.asarray(y_train),
                 "transform_name": transform_name, "scaler_name": scaler_name,
                 "feature_cols": list(feature_cols) if feature_cols else None,
                 "n_folds": int(n_folds)},
                timeout=_probe_timeout, stall_timeout=300.0, label=f"mc:{mname}",
            )
            if (_pres.get("__crashed__") or _pres.get("__worker_error__")
                    or _pres.get("__error__")):
                log.warning(
                    f"  [mc-per-model] {mname} probe isolated-fail "
                    f"({_pres.get('reason') or _pres.get('__worker_error__') or _pres.get('__error__')}) "
                    f"→ mc='none' fallback (run continues)")
                cells, best = {}, "none"
            else:
                cells, best = _pres.get("cells", {}), _pres.get("best", "none")
        else:
            _pres = _probe_one_model(
                mname, X_train, y_train, transform_name=transform_name,
                scaler_name=scaler_name, feature_cols=feature_cols,
                n_folds=n_folds, factory=factory)
            cells, best = _pres.get("cells", {}), _pres.get("best", "none")
        per_model_best[mname] = best
        # G-251 observability: per-model heartbeat — progress + none/vif/corr/pca OOF-WIS
        # comparison + selection + elapsed (fast elapsed ⇒ VIF disk-cache hit; slow ⇒ a
        # recompute). The mc pre-pass was previously silent in the parent log → 1h+ blind.
        _mc_wis = " ".join(f"{m}={cells[m].get('oof_wis', float('nan')):.3f}"
                           for m in ("none", "vif", "corr", "pca")
                           if isinstance(cells.get(m), dict))
        log.info(f"  [mc-per-model] {i_m}/{_n_probe} {mname}: "
                 f"{_mc_wis or '(no cells)'} → {best} [{time.time() - _mc_t0:.1f}s]")
        for method, c in cells.items():
            rows.append({"model": mname, "method": method, **c,
                         "selected": "Y" if method == best else ""})

    # ── global aggregate (median OOF WIS) — stable choice for non-restructured apply ──
    method_medians: dict[str, float] = {}
    for method in ("none", "vif", "corr", "pca"):
        vals = [r["oof_wis"] for r in rows
                if r["method"] == method and np.isfinite(r["oof_wis"])]
        if vals:
            method_medians[method] = float(np.median(vals))
    global_best = min(method_medians, key=method_medians.__getitem__) if method_medians else "none"
    # Gemini G-242 review: log the margin so a human sees if the auto-pick is decisive or a
    # coin-flip at n≈349 (medians within ~1% ⇒ the "winner" is noise, not signal).
    if method_medians:
        log.info("  [mc-per-model] method median OOF WIS: " +
                 ", ".join(f"{m}={v:.3f}" for m, v in sorted(method_medians.items(), key=lambda t: t[1])) +
                 f"  → global_best='{global_best}'")

    # ── write CSV + fingerprint sidecar ────────────────────────
    # G-253b (사용자 요청): per-model rank(1=best oof_wis) 컬럼 + best-method 요약 CSV.
    from collections import defaultdict as _dd_rank
    _by_model = _dd_rank(list)
    for r in rows:
        _by_model[r["model"]].append(r)
    _rank: dict = {}
    _summary_rows: list = []
    for _mn, _mrows in _by_model.items():
        _valid = [r for r in _mrows
                  if isinstance(r.get("oof_wis"), float) and np.isfinite(r["oof_wis"])]
        _ordered = sorted(_valid, key=lambda r: r["oof_wis"])  # 1 = 최저 WIS = best
        for _i, r in enumerate(_ordered, 1):
            _rank[(_mn, r["method"])] = _i
        if _ordered:
            _best = _ordered[0]
            _none = next((r for r in _mrows if r["method"] == "none"), None)
            _nw = (_none["oof_wis"] if _none and isinstance(_none.get("oof_wis"), float)
                   and np.isfinite(_none["oof_wis"]) else None)
            _summary_rows.append({
                "model": _mn, "best_method": _best["method"],
                "best_oof_wis": round(_best["oof_wis"], 4),
                "none_oof_wis": round(_nw, 4) if _nw is not None else "",
                "margin_vs_none": round(_nw - _best["oof_wis"], 4) if _nw is not None else "",
                "n_kept_best": _best["n_kept"],
            })
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=[
                "model", "method", "rank", "oof_wis", "insample_wis",
                "overfit_gap", "overfit_ratio", "n_kept", "selected"])
            w.writeheader()
            for r in rows:
                def _fmt(x):
                    return round(x, 4) if isinstance(x, float) and np.isfinite(x) else ""
                w.writerow({
                    "model": r["model"], "method": r["method"],
                    "rank": _rank.get((r["model"], r["method"]), ""),
                    "oof_wis": _fmt(r["oof_wis"]), "insample_wis": _fmt(r["insample_wis"]),
                    "overfit_gap": _fmt(r["overfit_gap"]),
                    "overfit_ratio": _fmt(r.get("overfit_ratio", float("nan"))),
                    "n_kept": r["n_kept"], "selected": r["selected"],
                })
        # G-253b: 모델당 1행 best-method 요약 (best/none/margin, best-WIS 오름차순)
        _summary_path = Path(out_dir) / "mc_best_method_summary.csv"
        with _summary_path.open("w", newline="", encoding="utf-8") as _sf:
            sw = _csv.DictWriter(_sf, fieldnames=[
                "model", "best_method", "best_oof_wis", "none_oof_wis",
                "margin_vs_none", "n_kept_best"])
            sw.writeheader()
            for sr in sorted(_summary_rows, key=lambda d: d["best_oof_wis"]):
                sw.writerow(sr)
        meta_path.write_text(json.dumps(
            {"fingerprint": _fp, "global_best": global_best,
             "method_medians": method_medians, "n_models": len(per_model_best)}, indent=2))
        log.info(f"  [mc-per-model] {len(per_model_best)} models × 4 methods → {csv_path} "
                 f"(global_best='{global_best}')")
    except Exception as _e:
        log.debug(f"  [mc-per-model] CSV write failed: {_e}")

    return per_model_best, global_best, rows


def _expand_feature_neighborhood(
    base_indices: list[int],
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
) -> list[int]:
    """Enlarge a frozen feature subset by its k nearest NON-selected neighbours (G-242 ③).

    "Neighbourhood" = the k highest ``|corr-with-target|`` features the feature pre-stage
    left out. The pre-stage subset is ALWAYS kept verbatim (it stays a subset of the
    result), so the frozen choice is never lost — we only *offer* the per-model preproc/HP
    search and the per-model multicollinearity filter (④) a few extra candidates to accept
    or prune. This is the gated, **default-off** relaxation of the otherwise-frozen
    selection.

    Design note (3-LLM, 2026-05-30): aggressive per-trial Optuna feature-mask search
    overfits validation noise at n≈349 (one layer above G-132). So ③ deliberately uses a
    *static enlargement* — selection within it is delegated to model regularization + the
    ④ mc filter, NOT to a wider Optuna mask search. Raise K only as n grows.

    Args:
        base_indices: frozen pre-stage feature indices (always ⊆ the return).
        X_train: train feature array (n × p).
        y_train: train target (n,).
        k: neighbourhood size. k<=0 → base_indices unchanged (default behaviour).

    Returns:
        sorted(base_indices ∪ {top-k non-selected by |corr|}); len ≤ len(base)+k.

    Side effects: none. Performance: O(p·n) one-pass corr. Caller responsibility:
        indices valid for X_train columns.
    """
    if k <= 0 or X_train is None or getattr(X_train, "ndim", 0) != 2:
        return list(base_indices)
    base = {int(i) for i in base_indices}
    p = int(X_train.shape[1])
    cands = [i for i in range(p) if i not in base]
    if not cands:
        return sorted(base)
    y = np.asarray(y_train, dtype=float).ravel()
    scored: list[tuple[float, int]] = []
    for i in cands:
        col = np.asarray(X_train[:, i], dtype=float).ravel()
        # Codex G-242 review: mask non-finite pairs so a single NaN/inf doesn't silently
        # demote an otherwise-useful feature to corr 0 (and never feed NaN to corrcoef).
        m = np.isfinite(col) & np.isfinite(y)
        if int(m.sum()) < 3 or float(np.std(col[m])) < 1e-12 or float(np.std(y[m])) < 1e-12:
            c = 0.0
        else:
            c = abs(float(np.corrcoef(col[m], y[m])[0, 1]))
            if not np.isfinite(c):
                c = 0.0
        scored.append((c, i))
    scored.sort(key=lambda t: t[0], reverse=True)
    return sorted(base | {i for _, i in scored[:k]})


def _clip_feature_indices(feature_indices, n_cols):
    """Drop feature indices outside [0, n_cols); empty ⇒ None (= use all columns). G-242.

    Guards the per-model path against a STALE original-space index list applied to an
    mc-reduced X. codex+gemini review (2026-05-30) found that under mc=pca the per-model
    Stage-2 name-intersection empties (original names ∉ PC1..PCk), so the incoming
    original-space `feature_indices` survives and `X[:, feature_indices]` would IndexError
    on the PC-column array. Clipping is a no-op for in-range indices (none/vif/corr path).

    Args:
        feature_indices: list[int] | None — candidate column indices.
        n_cols: width of the (post-mc) feature matrix.

    Returns:
        in-range subset (list[int]), or None when nothing is in range (caller uses all cols).

    Side effects: none.
    """
    if feature_indices is None:
        return None
    valid = [int(i) for i in feature_indices if 0 <= int(i) < int(n_cols)]
    return valid if valid else None


from simulation.utils.resource_tracker import track_resources


def _per_model_mc_choice(pm_rows, model_name, *, fallback="none", rel_margin=0.02):
    """Margin-guarded per-model multicollinearity method from the ④ comparison rows (G-242 A).

    The user wants mc handled PER-MODEL, not one global method. ④ (`_compare_mc_per_model`)
    already measures per-model OOF WIS per method; this picks each model's best — but only when
    it beats 'none' by ≥ rel_margin (relative). Otherwise 'none' (the simplest filter): an
    overfit guard at n≈349 where the mc CHOICE itself can overfit a noisy 2-fold OOF. D3
    refuted clean family rules (mc effect is data-dependent), so this is data-measured, not a
    tree/linear heuristic. Mirrors the 1-SE / Occam 'prefer simpler unless clearly better' rule.

    Args:
        pm_rows: list of dicts {"model","method","oof_wis"} — the mc_per_model_selection.csv rows.
        model_name: the model to choose for.
        fallback: method to use when `model_name` is absent from the rows (the global choice).
        rel_margin: minimum relative OOF-WIS improvement over 'none' to deviate from 'none'.

    Returns:
        one of "none"|"vif"|"corr"|"pca". 'none' on unclear benefit; `fallback` if unmeasured.
    """
    by_method: dict[str, float] = {}
    for r in pm_rows or []:
        if r.get("model") == model_name:
            try:
                by_method[str(r["method"])] = float(r["oof_wis"])
            except (KeyError, ValueError, TypeError):
                continue
    finite = {m: w for m, w in by_method.items() if np.isfinite(w)}
    if not finite:
        return fallback
    best_m = min(finite, key=finite.get)
    none_wis = finite.get("none")
    if none_wis is None:
        return best_m                       # no 'none' baseline measured → trust the data
    if best_m == "none":
        return "none"
    improve = (none_wis - finite[best_m]) / abs(none_wis) if none_wis != 0 else 0.0
    return best_m if improve >= rel_margin else "none"


def _apply_mc_columns(method, X_train, X_val, X_test, X_real, y_train, feature_cols):
    """Apply ONE multicollinearity filter to all design matrices consistently (G-242 A).

    Per-model apply path's column handler — mirrors run_per_model_optimize's global apply so a model can
    use its OWN method. 'none' → pass-through (state None). vif/corr drop collinear columns
    (kept-index remap of feature_cols + X_real); pca transforms to PC# components (X_real via
    the fitted scaler+pca). mc_state is what the R10 (per_model_eval) artifact stores to replay at inference.

    Args:
        method: "none"|"vif"|"corr"|"pca".
        X_train/X_val/X_test/X_real: design matrices (X_test/X_real may be None).
        y_train: target (vif/corr supervision is unused; passed through to the filter API).
        feature_cols: column names aligned to X_train's columns.

    Returns:
        (X_train_f, X_val_f, X_test_f, X_real_f, feature_cols_f, mc_state, mc_meta).

    Raises: nothing — propagates the filter's own exceptions to the caller (which logs + skips).
    """
    if method == "none":
        return X_train, X_val, X_test, X_real, feature_cols, None, {}
    from simulation.pipeline.mc_filter_stage3 import apply_multicollinearity_filter
    _X_te_for = X_test if X_test is not None else np.zeros((1, X_train.shape[1]))
    X_tr_f, X_val_f, X_te_f, mc_state, mc_meta = apply_multicollinearity_filter(
        X_train, X_val, _X_te_for, y_train,
        feature_cols=list(feature_cols) if feature_cols else None, method=method)
    X_test_f = X_te_f if X_test is not None else None
    X_real_f, fcols_f = X_real, feature_cols
    if method in ("vif", "corr"):
        kept = mc_state  # list[int] absolute col indices
        fcols_f = [feature_cols[i] for i in kept] if feature_cols else []
        if X_real is not None:
            X_real_f = X_real[:, kept]
    elif method == "pca":
        fcols_f = [f"PC{i+1}" for i in range(X_tr_f.shape[1])]
        if X_real is not None and mc_state is not None:
            X_real_f = mc_state["pca"].transform(
                mc_state["scaler"].transform(np.asarray(X_real, dtype=np.float64)))
    return X_tr_f, X_val_f, X_test_f, X_real_f, fcols_f, mc_state, mc_meta


@track_resources("phase13_per_model_optimize")
def run_per_model_optimize(
    phase1: dict,
    all_results: dict,
    config,
) -> dict:
    """Per-model individual optimization via hierarchical preproc Optuna (G-233).

    Uses the in-sample train/val split for the search; the final test-slab
    evaluation happens in R10 (per_model_eval, which now reads the per-model optimal
    configs persisted here).

    Args:
      phase1: dict with X_all, y_all, n_train, n_val, pool_end
      all_results: pipeline outputs (for cached per-model feature selection)
      config: pipeline config

    Returns: {"per_model_configs": {model: best_config}, "elapsed": float}
    """
    t0 = time.time()
    if not bool(getattr(config, "per_model_optimize", False)) and \
       not bool(getattr(config.split, "per_model_optimize", False)):
        log.info("  [phase13] per-model optimization disabled (set "
                 "--per-model-optimize to enable)")
        return {"skipped": True, "reason": "disabled", "elapsed": 0.0}

    # audit Stage 3.1 (cascade #3, 2026-05-27) — Multi-seed wrapper (env-gated)
    # MPH_MULTI_SEED_RUN=1 → 5 seed sequential runs for stability analysis
    # default: single seed (42) — backward compat
    try:
        from simulation.analytics.multi_seed import (
            lock_global_seeds, multi_seed_enabled, get_seed_list, build_seed_manifest,
        )
        _seed = 42
        lock_global_seeds(_seed, log=True)
        _seed_manifest = build_seed_manifest(_seed)
        if multi_seed_enabled():
            log.info(f"  [phase13] MPH_MULTI_SEED_RUN=1 detected — "
                      f"multi-seed wrapper available (seeds: {get_seed_list()})")
            log.info(f"  [phase13] Note: full multi-seed integration is post-cascade "
                      f"sub-task; current run uses single seed manifest")
        else:
            log.info(f"  [phase13] single-seed manifest: {_seed_manifest['global_seed']} "
                      f"(set MPH_MULTI_SEED_RUN=1 for stability analysis)")
    except Exception as _seed_e:
        log.warning(f"  [phase13] seed lock fail (non-fatal): {_seed_e}")

    X_all = phase1["X_all"]
    y_all = phase1["y_all"]
    feature_cols = phase1["feature_cols"]
    n_train = phase1["n_train"]
    n_val = phase1["n_val"]
    n_test = phase1.get("n_test", 0)
    pool_end = phase1.get("pool_end", n_train + n_val)

    X_train = X_all[:n_train]
    y_train = y_all[:n_train]
    X_val   = X_all[n_train:n_train + n_val]
    y_val   = y_all[n_train:n_train + n_val]
    # Test slab (HWP §3): in-sample idx [pool_end:n] = last 68 weeks
    X_test  = X_all[pool_end:pool_end + n_test] if n_test > 0 else None
    y_test  = y_all[pool_end:pool_end + n_test] if n_test > 0 else None
    # Real slab (service zone, Phase C.6 2026-05-06): phase1 가 별 key 로 잘라 saved
    # (X_all 은 in-sample 만 — train+val+test; phase1_data.py:189-190 참조)
    # methodology §4.1 rolling-origin 1-step-ahead — Phase C.6/C.7 가 사용
    X_real = phase1.get("real_X", None)
    y_real = phase1.get("real_y", None)
    n_real = int(len(y_real)) if y_real is not None else 0

    # G-232/G-234: default before conditional override below
    _mc_state: Optional[Any] = None  # will be set when filter is applied

    # G-232 (2026-05-23): multicollinearity filter in R9 (per_model_optimize) trials.
    # G-234 (2026-05-24): MPH_MULTICOLLINEARITY=auto → 4-method 자동 비교 후 최적 선택.
    # Phase A/B 별도 worker 없이 파이프라인 내부에서 자동 처리.
    from simulation.config_global import GLOBAL as _GCFG_mc  # SSOT (2026-05-28)
    _mc_env = _GCFG_mc.training.multicollinearity

    # ── G-242 ④: per-model none/vif/corr/pca comparison (visibility + save) ──────────
    # DECOUPLED from the apply choice: runs when mc=auto (needed for the choice) OR
    # MPH_MC_COMPARE != "0" (default ON) — so mc_per_model_selection.csv appears even under
    # the default config (mc='none'), without changing what gets applied. Cheap + cached.
    # model_factories 아직 미구성 → probe factories (MPH_MC_COMPARE_ALL=1 → 전체 53).
    import os as _os_mc
    # G-242 A (2026-05-30; default→per-model 2026-05-31 per 사용자 "왜 global로 하는거야?").
    # per-model mc APPLY is now THE default (user-mandated + demo-verified: global='pca' would
    # HURT 5/7 models). Set MPH_MC_PER_MODEL=0 only for the legacy global path. When on (default)
    # the ④ compare covers ALL models so each has a measured choice, applied per-model in the
    # loop, margin-guarded (deviate from 'none' only on ≥MPH_MC_MARGIN gain → n≈349 overfit safe).
    _per_model_mc = _os_mc.environ.get("MPH_MC_PER_MODEL", "1") == "1"

    def _build_probe_factories():
        from simulation.models.base import REGISTRY as _AUTO_REG
        if _os_mc.environ.get("MPH_MC_COMPARE_ALL", "0") == "1" or _per_model_mc:
            _allm = _AUTO_REG.get_all()
            _force_all = _os_mc.environ.get("MPH_MC_COMPARE_ALL", "0") == "1"
            # ── active panel 로 제한 (2026-06-04) ──────────────────────────────────────
            # worker 는 active CATEGORY_MODELS(53)만 최적화하는데 get_all() 은 DEFER 11
            #   (DeepAR/RNN/TFT/BayesianMCMC/CoxPH/PROPHET/…)까지 포함(registered≠active) →
            #   그들 mc 는 worker 가 안 써서 probe 가 무관 모델(특히 비싼 DL)을 헛 fit = 낭비.
            #   CATEGORY_MODELS(사용자 prune SSOT, 53) ∩ CLI --models 로 제한. ALL=1 이면 override.
            from simulation.models.registry import CATEGORY_MODELS as _CATM
            _active = set(m for v in _CATM.values() for m in v)
            _cli = set(getattr(config, "_selected_models", None) or [])
            if _cli:
                _active &= _cli
            # USES_FEATURES=False (TimesFM/TiRex): X 무시 → mc 4방법 동일 예측 = 비교 무의미(irrelevance).
            #   probe 제외 → mc='none' fallback(_per_model_mc_choice L2328). 불필요 fit 8× 회피.
            def _keep_probe(_n, _c):
                if not getattr(_c, "USES_FEATURES", True):
                    return False                        # 피처 무시(foundation) → mc 무관
                return _force_all or (_n in _active)     # active(worker) 대상만 비교
            fac = {_n: (lambda c=_c: c()) for _n, _c in _allm.items() if _keep_probe(_n, _c)}
            _n_feat = sum(1 for _c in _allm.values() if not getattr(_c, "USES_FEATURES", True))
            _n_nonact = sum(1 for _n, _c in _allm.items()
                            if getattr(_c, "USES_FEATURES", True) and not _force_all and _n not in _active)
            log.info(f"  [phase13] G-242 probe → {len(fac)} models "
                     f"(active≤53 기준; 피처무시 {_n_feat} 제외, non-active/DEFER {_n_nonact} 제외)")
            return fac
        fac = {}
        for _pn in _AUTO_MC_PROBE_PREFERRED:
            _cls = _AUTO_REG.get(_pn)
            if _cls is not None:
                fac[_pn] = (lambda c=_cls: c())
                if len(fac) >= _AUTO_MC_N_PROBE:
                    break
        return fac

    _save_dir = Path(getattr(config, "save_dir", "simulation/results"))
    _global_best = None
    _pm_rows: list = []     # G-242 A: per-model ④ rows for the margin-guarded per-model choice
    _want_compare = (_mc_env == "auto") or _per_model_mc or (_os_mc.environ.get("MPH_MC_COMPARE", "1") != "0")
    if _want_compare:
        try:
            _pm_best, _global_best, _pm_rows = _compare_mc_per_model(
                _build_probe_factories(), X_train, y_train,
                feature_cols=list(feature_cols) if feature_cols else None,
                out_dir=_save_dir, n_folds=2,
            )
            log.info(
                f"  [phase13] G-242 per-model mc compare → mc_per_model_selection.csv "
                f"({len(_pm_best)} models, global_best='{_global_best}')"
            )
        except Exception as _cmp_err:
            log.warning(f"  [phase13] G-242 per-model compare 실패: {_cmp_err}")

    # ── mc method 결정: auto → OOF global aggregate (val-single fallback); 그 외 → config ──
    if _mc_env == "auto":
        if _global_best is not None:
            _mc_method = _global_best
        else:
            try:
                _mc_method = _auto_select_mc_method(
                    X_train, y_train, X_val, y_val,
                    feature_cols=list(feature_cols) if feature_cols else None,
                    model_factories=_build_probe_factories(), out_dir=_save_dir,
                )
            except Exception as _auto_err:
                log.warning(f"  [phase13] auto select 실패 → none fallback: {_auto_err}")
                _mc_method = "none"
        log.info(f"  [phase13] auto → multicollinearity='{_mc_method}'")
    else:
        _mc_method = _mc_env

    # G-242 A: in per-model mode the filter is applied PER-MODEL inside the loop (below), so
    # the global one-shot apply is skipped here — X_train/feature_cols stay UNFILTERED.
    if _mc_method != "none" and not _per_model_mc:
        try:
            from simulation.pipeline.mc_filter_stage3 import (
                apply_multicollinearity_filter,
            )
            _X_te_for_filter = (X_test if X_test is not None
                                else np.zeros((1, X_train.shape[1])))
            _X_tr_f, _X_val_f, _X_te_f, _mc_state_new, _mc_meta = (
                apply_multicollinearity_filter(
                    X_train, X_val, _X_te_for_filter, y_train,
                    feature_cols=list(feature_cols) if feature_cols else None,
                    method=_mc_method,
                )
            )
            X_train = _X_tr_f
            X_val   = _X_val_f
            if X_test is not None:
                X_test = _X_te_f
            # G-232/G-234: promote to outer var for artifact save (phase14 replay)
            _mc_state = _mc_state_new
            if _mc_method in ("vif", "corr"):
                _kept = _mc_state  # list[int] of absolute col indices
                feature_cols = ([feature_cols[i] for i in _kept]
                                if feature_cols else [])
                if X_real is not None:
                    X_real = X_real[:, _kept]
            elif _mc_method == "pca":
                feature_cols = [f"PC{i+1}" for i in range(X_train.shape[1])]
                if X_real is not None and _mc_state is not None:
                    _sc  = _mc_state["scaler"]
                    _pca = _mc_state["pca"]
                    X_real = _pca.transform(_sc.transform(
                        np.asarray(X_real, dtype=np.float64)))
            log.info(
                f"  [phase13] G-232 multicollinearity filter '{_mc_method}': "
                f"{_mc_meta.get('n_kept', '?')} features kept "
                f"(dropped {_mc_meta.get('n_dropped', '?')}, "
                f"{_mc_meta.get('runtime_s', 0):.1f}s)"
            )
        except Exception as _mc_err:
            log.warning(
                f"  [phase13] multicollinearity filter '{_mc_method}' failed: "
                f"{_mc_err} — continuing without filter (G-232 partial)"
            )

    # Get model factories from REGISTRY (same approach as Phase 12)
    # 2026-05-12 Codex Q1 Risk 2 fix: 이전 12 모듈 import → registry under-load
    # (33 vs 70 expected). `--resume-from 12` direct 호출 시 modern_ts /
    # graph / mech / foundation / ensemble 모듈 누락 → silent shrink.
    # 정정: registry.verify_registry_coverage(force_import=True) 가 19 모듈 sweep
    # 보장 (runner._import_all_models 동등). 실패해도 graceful (return ok=False).
    try:
        from simulation.models.base import REGISTRY
        from simulation.models.registry import verify_registry_coverage
        try:
            _coverage = verify_registry_coverage(force_import=True)
            log.info(f"  [phase13] REGISTRY coverage: {_coverage['total_registered']} "
                     f"models registered (expected {_coverage['total_expected']}, "
                     f"missing {len(_coverage['missing'])})")
        except Exception as _ce:
            # Fallback to legacy 12-module list if verify_registry_coverage fails
            log.warning(f"  [phase13] verify_registry_coverage failed: {_ce} — "
                        "fallback to legacy 12-module import")
            for _m in ("epi_models", "dl_models", "tree_models", "linear_models",
                       "negbin_glm", "graph_models", "phase_ensemble",
                       "conformal", "cqr_models", "bayesian_seir",
                       "seir_forced", "pinn_model"):
                try:
                    __import__(f"simulation.models.{_m}")
                except Exception:
                    pass
        # Restrict to models selected via --models filter, else all registered
        selected = getattr(config, "_selected_models", None) or []
        if selected:
            model_names = [n for n in selected if REGISTRY.get(n) is not None]
        else:
            model_names = list(REGISTRY.get_all().keys())
        # Build factories: each callable returns a fresh model instance
        # All models run R9 (per_model_optimize) preproc Optuna — OPTUNA_ISOLATE=1 + MPH_LIGHTNING_MAX_TIME_PER_MODEL
        # provide per-trial OOM/timeout protection (SLOW_MODELS filter removed 2026-05-24).
        model_factories: dict = {}
        for n in model_names:
            cls = REGISTRY.get(n)
            if cls is not None:
                def _factory(cls=cls):
                    return cls()
                model_factories[n] = _factory
        log.info(f"  [phase13] {len(model_factories)} models → preproc Optuna")
    except Exception as e:
        log.error(f"  [phase13] could not access model factories: {e}")
        return {"skipped": True, "reason": f"factory_unavailable: {e}",
                "elapsed": time.time() - t0}

    # Cached per-model feature selections (from Optuna feature selection phase)
    per_model_feat = (all_results.get("phase2", {}).get("per_model_feature_map", {})
                      or all_results.get("external", {}).get("feature_selection_log", {})
                      or {})

    out_dir = Path(getattr(config, "save_dir", "simulation/results")) / "per_model_optimal"
    out_dir.mkdir(parents=True, exist_ok=True)

    # G-235 (2026-05-24): DB fingerprint — compute ONCE before the training loop.
    # Embedded in each champion record so compare_v1_v2 can verify data parity.
    _db_fp: Optional[dict] = None
    try:
        from simulation.utils.db_fingerprint import compute_db_fingerprint
        _db_fp = compute_db_fingerprint()
        log.info(
            "  [phase13] DB fingerprint: %s (tables: %d)",
            _db_fp.get("combined_sha256", "?"),
            len(_db_fp.get("tables", {})),
        )
    except Exception as _fp_err:
        log.warning("  [phase13] DB fingerprint failed (non-fatal): %s", _fp_err)

    per_model_configs: dict = {}
    # G-237 (2026-05-27, 사용자 명시 "중간 끊김 재시작"): per-model skip-if-exists.
    # 52 모델 중 30번 segfault/끊김 시 재시작에서 1-29 다시 학습 방지.
    # Optuna study DB warm-start 만으로는 trial 결과만 영속 — JSON 결과 재작성 비용.
    # MPH_FORCE_REDO_PHASE13=1 로 강제 재학습 (env config 큰 변경 시).
    _force_redo_p12 = GLOBAL.ops.force_redo_phase13
    for mname, factory in model_factories.items():
        _json_path = out_dir / f"{mname}.json"
        if _json_path.exists() and not _force_redo_p12:
            try:
                per_model_configs[mname] = json.loads(_json_path.read_text())
                log.info(f"  [phase13] {mname} already optimized — skip (MPH_FORCE_REDO_PHASE13=1 to redo)")
                continue
            except Exception as _skip_err:
                log.warning(f"  [phase13] {mname} JSON load failed → redo: {_skip_err}")
                # fall through to re-optimize
        try:
            feat_info = per_model_feat.get(mname, {})
            feat_idx = (feat_info.get("feature_indices") if isinstance(feat_info, dict)
                         else None)
            # Per-model design matrices + mc choice (G-242 A). Default path: locals == the
            # global vars → optimize_one_model is called with identical args (zero regression).
            # Per-model mode: apply each model's OWN margin-guarded mc to the unfiltered X.
            _X_tr_m, _X_val_m, _X_te_m = X_train, X_val, X_test
            _X_real_m, _fcols_m = X_real, feature_cols
            _mc_method_m, _mc_state_m = _mc_method, _mc_state
            if _per_model_mc:
                _mc_method_m = _per_model_mc_choice(
                    _pm_rows, mname, fallback=_mc_method,
                    rel_margin=float(_os_mc.environ.get("MPH_MC_MARGIN", "0.02")))
                if _mc_method_m != "none":
                    try:
                        (_X_tr_m, _X_val_m, _X_te_m, _X_real_m, _fcols_m,
                         _mc_state_m, _mm) = _apply_mc_columns(
                            _mc_method_m, X_train, X_val, X_test, X_real,
                            y_train, feature_cols)
                        log.info(f"  [phase13] G-242 per-model mc '{mname}' → "
                                 f"'{_mc_method_m}' ({_mm.get('n_kept', '?')} kept)")
                    except Exception as _pmm_err:
                        log.warning(f"  [phase13] per-model mc '{mname}'/'{_mc_method_m}' "
                                    f"failed → none: {_pmm_err}")
                        _mc_method_m, _mc_state_m = "none", None
                        _X_tr_m, _X_val_m, _X_te_m = X_train, X_val, X_test
                        _X_real_m, _fcols_m = X_real, feature_cols
                else:
                    log.info(f"  [phase13] G-242 per-model mc '{mname}' → 'none' (guard)")
            # G-311 (2026-06-18): OverseasTransfer's transfer encoder resolves ili_rate_lag1-4 BY
            #   NAME; mc=pca renames features to PCs → name lookup fails → transfer silently
            #   degrades to feature-only. Force name-preserving mc='none' so the encoder engages.
            #   (mc 고정 isolation — the transfer A/B toggles transfer on/off at constant preproc.)
            from simulation.pipeline.preproc_optuna_hierarchical import model_requires_named_features
            if model_requires_named_features(mname) and _mc_method_m != "none":
                log.info(f"  [phase13] G-311 '{mname}' needs named features → mc forced 'none' "
                         f"(was '{_mc_method_m}')")
                _mc_method_m, _mc_state_m = "none", None
                _X_tr_m, _X_val_m, _X_te_m = X_train, X_val, X_test
                _X_real_m, _fcols_m = X_real, feature_cols
            # G-236/G-249: run the full optimize in a fresh child for OMP-fragile
            # categories — each child loads only ONE model's stack (no torch+lightgbm+
            # statsmodels mix), so the OMP #179 abort can't recur, and a child death is
            # CONTAINED (parent logs + continues to the next model). 'meta' ensembles +
            # macOS PyG/MPS stay in-process (unchanged path).
            if _phase13_isolate_model(mname):
                from simulation.pipeline._phase13_isolation import run_isolated as _run_iso
                res = _run_iso(
                    "simulation.pipeline.per_model_optimize:_optimize_worker",
                    {"mname": mname, "X_train": _X_tr_m, "y_train": y_train,
                     "X_val": _X_val_m, "y_val": y_val, "feature_indices": feat_idx,
                     "X_test": _X_te_m, "y_test": y_test, "feature_cols": _fcols_m,
                     "X_real": _X_real_m, "y_real": y_real, "db_fingerprint": _db_fp,
                     "mc_method": _mc_method_m, "mc_state": _mc_state_m,
                     "viral_positivity_train": phase1.get("viral_positivity_train")},
                    timeout=float(_os_mc.environ.get("MPH_PHASE13_ISOLATE_TIMEOUT", "5400")),
                    # G-312: foundation models (TabPFN/TiRex/TimesFM) run long INFERENCE stretches
                    #   without writing child.log → the 900s stall-guard false-killed TabPFN (43min,
                    #   the prior champion) mid-inference. Give quiet-foundation 3× stall tolerance;
                    #   the hard `timeout` still bounds a true hang. (SVR-Linear hang fixed via G-312
                    #   max_iter, so it is not extended here.)
                    stall_timeout=(float(_os_mc.environ.get("MPH_PHASE13_ISOLATE_STALL", "900"))
                                   * (3.0 if mname in {"TabPFN", "TiRex", "TimesFM-2.5"} else 1.0)),
                    label=mname,
                )
                if res.get("__crashed__") or res.get("__worker_error__"):
                    log.warning(
                        f"  [phase13] {mname} isolated optimize FAILED "
                        f"({res.get('reason') or res.get('__worker_error__')}) → skip model, "
                        f"run continues (resume retries this one)")
                    continue
            else:
                res = optimize_one_model(
                    mname, factory,
                    _X_tr_m, y_train, _X_val_m, y_val,
                    feature_indices=feat_idx,
                    X_test=_X_te_m, y_test=y_test,   # ← test slab for refit eval
                    feature_cols=_fcols_m,            # 2026-04-28: grouped preproc
                    X_real=_X_real_m, y_real=y_real, # Phase C.6 (2026-05-06): service zone
                    db_fingerprint=_db_fp,           # G-235: attach DB state hash
                    mc_method=_mc_method_m,          # G-232/G-234/A: per-model for artifact replay
                    mc_state=_mc_state_m,            # G-232/G-234/A: filter state (list[int] or dict)
                    # audit Stage 1.1 (cascade #1, 2026-05-27) — KDCA threshold input
                    viral_positivity_train=phase1.get("viral_positivity_train"),
                )
            # G-237b (2026-06-15): refit-null 을 silent 집계하지 않고 loud 가시화. ensemble(meta)은
            #   base-pred plumbing(#8 deferred)로 per_model refit-null 이 known/허용(R10 per_model_eval 평가
            #   정상) → 제외. 그 외 모델의 test r2 None = loud banner(보고 무결성, 제외는 안 함).
            try:
                _tm_g237 = res.get("test_metrics") if isinstance(res, dict) else None
                if (not str(mname).startswith("Ensemble-")
                        and (not isinstance(_tm_g237, dict) or _tm_g237.get("r2") is None)):
                    log.error(f"  [G-237][REFIT-NULL] {mname}: test refit 산출 없음 "
                              f"(silent 집계 아님 — 보고 무결성 경고; #5/#6 후에도 발생 시 조사)")
            except Exception:
                pass
            per_model_configs[mname] = res
            (out_dir / f"{mname}.json").write_text(json.dumps(res, indent=2, default=str))
            # G-253 (A): R9(per_model_optimize) OPTIMIZED 예측을 canonical per-model CSV로도 기록
            # (split/idx/y_true/y_pred), R2(baseline) CSV를 덮어씀 →
            # results/csv/predictions_<name>.csv = 최종 최적화 예측 (web·감사 동일 참조).
            # parent 에서 작성 (res 가 예측 보유) → 격리-child cwd 무관.
            try:
                from simulation.utils.paths import get_results_dir as _grd
                _csvd = _grd() / "csv"; _csvd.mkdir(parents=True, exist_ok=True)
                _safe = mname.replace(" ", "_").replace("/", "_")
                _rows = ["split,idx,y_true,y_pred"]
                # eval CSV = test slab 전용 (모델 평가). "real"(서비스존 operational
                # forecast)은 학습/평가가 아니라 배포(ABM/ARIA/web)용 → json 의
                # refit_real_predictions + 배포 경로에만 둔다. 여기 섞지 않음 (사용자 지적).
                _otp = res.get("refit_test_predictions") if isinstance(res, dict) else None
                if _otp is not None and y_test is not None and len(y_test) == len(_otp):
                    for _i, (_a, _p) in enumerate(zip(np.asarray(y_test, float),
                                                       np.asarray(_otp, float))):
                        _rows.append(f"test,{_i},{float(_a)},{float(_p)}")
                if len(_rows) > 1:
                    (_csvd / f"predictions_{_safe}.csv").write_text(
                        "\n".join(_rows) + "\n", encoding="utf-8")
            except Exception as _ce:
                log.debug(f"  [phase13] {mname} optimized-pred CSV failed: {_ce}")
        except Exception as e:
            log.warning(f"  [phase13] {mname} optimization failed: {e}")

    # G-162 (2026-05-02): summary 의 scalers/transforms 정직성 fix.
    # Hierarchical preproc Optuna 가 선택한 실제 값만 보고 (caller arg 제거됨 2026-05-24).
    actual_scalers: set = set()
    actual_transforms: set = set()
    for c in per_model_configs.values():
        if not isinstance(c, dict):
            continue
        for cell in c.get("optuna_trial_results", []) or []:
            if isinstance(cell, dict):
                if cell.get("scaler"):
                    actual_scalers.add(cell["scaler"])
                if cell.get("transform"):
                    actual_transforms.add(cell["transform"])

    # Persist roll-up
    summary = {
        "n_models_optimized": len(per_model_configs),
        "transforms_actually_searched": sorted(actual_transforms),
        "scalers_actually_searched":    sorted(actual_scalers),
        # back-compat aliases
        "transforms_searched": sorted(actual_transforms),
        "scalers_searched":    sorted(actual_scalers),
        # G-235: DB fingerprint embedded in summary for cross-run comparison
        "db_fingerprint": (
            {"combined_sha256": _db_fp.get("combined_sha256"),
             "computed_at":     _db_fp.get("computed_at"),
             "db_path":         _db_fp.get("db_path")}
            if _db_fp is not None else None
        ),
        "best_per_model": {
            m: {"transform": c["best_config"]["transform"],
                "scaler":    c["best_config"]["scaler"],
                # val metrics (config-selection criterion)
                "val_wis":   c.get("val_metrics", {}).get("wis"),
                "val_mae":   c.get("val_metrics", {}).get("mae"),
                # TEST metrics (proper evaluation slab — primary leaderboard)
                "test_wis":  c.get("test_metrics", {}).get("wis"),
                "test_mae":  c.get("test_metrics", {}).get("mae"),
                "test_r2":   c.get("test_metrics", {}).get("r2"),
                "test_n":    c.get("test_metrics", {}).get("n"),}
            for m, c in per_model_configs.items()
        },
        "elapsed_sec": time.time() - t0,
        "note": ("val metrics chose the (transform × scaler) config; "
                  "test_* metrics are the FINAL evaluation on the held-out "
                  "test slab (n=68, HWP §3) after refitting with chosen config "
                  "on full train+val pool."),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    log.info(f"  [phase13] optimized {len(per_model_configs)} models "
             f"in {summary['elapsed_sec']:.1f}s → {out_dir}")

    # Package E (2026-04-29): R9 (per_model_optimize) Pass 2 — 통합 통계 audit
    # 사용자 제안 — "phase13에서 평가" 영구 표준
    consolidation = _phase12_consolidate(per_model_configs, out_dir)

    return {"per_model_configs": per_model_configs, "summary": summary,
             "consolidation": consolidation,
             "out_dir": str(out_dir), "elapsed": time.time() - t0}


# Package E (2026-04-29): R9 (per_model_optimize) Pass 2 — Final consolidation
def _phase12_consolidate(per_model_configs: dict, out_dir: Path) -> dict:
    """R9 (per_model_optimize) 끝에 호출 — 통합 통계 audit 자동 실행.

    각 모델의 best config 결과를 모아 multi-criteria filter (5중) +
    statistical_audit (Fisher z, DM, MCS, Bootstrap, Mondrian) 적용.

    출력:
        STATISTICAL_AUDIT.md  (TRIPOD+AI 정합 보고서)
        STATISTICAL_AUDIT.json
        consolidation_summary.json (R9 per_model_optimize 산출 요약)

    Fail-safe: 실패 시 학습 결과 그대로 보존 (warning 만 log).
    """
    log.info("=" * 60)
    log.info("R9 Pass 2: Final consolidation (statistical audit)")
    log.info("=" * 60)

    try:
        from simulation.scripts.statistical_audit import (
            audit_prediction_model, mcs_test, render_md, _r2,
        )
        from simulation.config_global import GLOBAL
        import numpy as np

        # 1. Per-model audit
        audits = []
        baseline_pred = None
        # baseline (persistence) 우선 찾기
        for name, cfg in per_model_configs.items():
            if name.lower() == "persistence":
                baseline_pred = np.asarray(
                    cfg.get("refit_test_predictions",          # G-FIX (2026-05-24): was "test_predictions" (wrong key)
                            cfg.get("test_metrics", {}).get("predictions", []))  # predictions popped at line 1379
                )
                if len(baseline_pred) == 0:
                    baseline_pred = None
                break

        for name, cfg in per_model_configs.items():
            try:
                tm = cfg.get("test_metrics", {})
                yt = np.asarray(tm.get("y_true", cfg.get("y_true", [])))
                yp = np.asarray(tm.get("predictions", cfg.get("refit_test_predictions", [])))  # G-FIX key
                if len(yt) == 0 or len(yp) == 0 or len(yt) != len(yp):
                    continue

                yl = np.asarray(tm.get("pi_lower")) if tm.get("pi_lower") is not None else None
                yu = np.asarray(tm.get("pi_upper")) if tm.get("pi_upper") is not None else None
                bp = baseline_pred if baseline_pred is not None and len(baseline_pred) == len(yt) else None
                a = audit_prediction_model(
                    name=name, y_true=yt, y_pred=yp,
                    y_lower=yl, y_upper=yu,
                    baseline_pred=bp,
                    n_boot=500,  # phase13 안이라 약간 줄임
                )
                audits.append(a)
            except Exception as _e:
                log.warning(f"  [consolidate] {name} audit 실패: {_e}")
                continue

        # 2. MCS pairwise
        mcs_summary = {}
        if len(audits) >= 2:
            losses = {}
            for name, cfg in per_model_configs.items():
                tm = cfg.get("test_metrics", {})
                yt = np.asarray(tm.get("y_true", cfg.get("y_true", [])))
                yp = np.asarray(tm.get("predictions", cfg.get("refit_test_predictions", [])))  # G-FIX key
                if len(yt) and len(yt) == len(yp):
                    losses[name] = (yt - yp) ** 2
            if losses:
                mcs_result = mcs_test(losses, alpha=0.05, n_boot=300)
                mcs_summary = {
                    "survivors": mcs_result["survivors"],
                    "mcs_size": mcs_result["mcs_size"],
                    "alpha": 0.05,
                }
                for a in audits:
                    a.mcs = {
                        "in_mcs": a.model_name in mcs_result["survivors"],
                        "mcs_size": mcs_result["mcs_size"],
                    }

        # 3. STATISTICAL_AUDIT.md 생성
        from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
        _audit_dir = get_results_dir()
        md_path = _audit_dir / "STATISTICAL_AUDIT.md"
        json_path = _audit_dir / "STATISTICAL_AUDIT.json"

        from dataclasses import asdict
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_md(audits, []), encoding="utf-8")

        payload = {
            "ts": time.time(),
            "phase": "phase12_consolidation",
            "n_models_audited": len(audits),
            "mcs": mcs_summary,
            "audits": [asdict(a) for a in audits],
        }
        json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

        log.info(f"  ✓ STATISTICAL_AUDIT.md 생성 (audited {len(audits)} 모델)")
        if mcs_summary.get("mcs_size"):
            log.info(f"  ✓ MCS@5%: {mcs_summary['mcs_size']} 모델 (best 통계 superset)")

        return {
            "ok": True,
            "n_audited": len(audits),
            "filter_pass": payload["filter_pass_count"],
            "mcs_size": mcs_summary.get("mcs_size"),
            "md_path": str(md_path),
            "json_path": str(json_path),
        }

    except Exception as _e:
        # Fail-safe: 학습 결과 보존, audit 만 skip
        log.warning(f"  ⚠ R9 consolidation 실패: {type(_e).__name__}: {_e}")
        log.warning(f"  ⚠ scripts/audit_and_retrain.sh 수동 실행 필요")
        return {"ok": False, "error": str(_e)}


# back-compat aliases (2026-06-02 semantic rename — 옛 run_phaseN)
run_phase13 = run_per_model_optimize
