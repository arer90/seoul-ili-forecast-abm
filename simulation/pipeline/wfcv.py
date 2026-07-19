"""
R4: Walk-Forward CV + Inline Optuna + Recursive Forecasting
=================================================================
V16 핵심 모듈. V15의 치명적 버그(BUG-1) 수정:
- 모델별 Optuna 선택 피처를 WF-CV 내에서 사용
- Inline Optuna: WF-CV fold 내부에서 피처 탐색
- retune_every: N fold마다 Optuna 재실행 (효율성)
- warm-start: 이전 fold 결과를 다음 Optuna 초기값으로
"""
from simulation.config_global import GLOBAL  # SSOT (2026-05-28)
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline

log = logging.getLogger(__name__)


def _generate_wf_folds(n_total: int, min_train: int, step: int,
                       holdout_start: Optional[int] = None) -> List[Tuple[int, int, int]]:
    """Walk-Forward fold 생성. 반환: [(train_end, val_start, val_end), ...]

    S0-1 fix: if `holdout_start` is given, no fold's val_end may exceed it,
    guaranteeing the conformal holdout slab is never touched by WF-CV.
    """
    cap = holdout_start if holdout_start is not None else n_total
    cap = max(min(cap, n_total), min_train)
    folds = []
    pos = min_train
    while pos < cap:
        val_end = min(pos + step, cap)
        folds.append((pos, pos, val_end))
        pos += step
    return folds


# ── S1-1 / P1-5: fold-wise quantile bin recalculation ──────────────────
# builder.py line 867 computes `_qe_train_end = int(len(df) * train_ratio)`
# once, which leaks fold-future distributional info into early-fold
# *_qbin / *_qnorm columns. For strict causal WF-CV we recompute these
# columns per fold using only X_all[:train_end] as the reference window.
# Source columns must be present in feature_cols for this to work;
# otherwise the raw values are kept (behaviour identical to ).
_QUANTILE_SPECS: Tuple[Tuple[str, int], ...] = (
    ("ili_rate_lag1", 10),
    ("temp_avg", 8),
)


# ── S1-1 full fix : fold-wise epidemic-phase threshold ──────────
# transforms._add_epidemic_phase_features computes
#     baseline = median(ili[:int(n*0.8)]); threshold = baseline * 2
# at BUILD time. For early WF-CV folds where train_end << 0.8·n the
# threshold then bakes in distributional info from the *future* of the
# fold's own train window. We recompute `above_threshold` per fold using
# only y_all[:train_end] as the reference window. The other two
# epidemic-phase columns (`consec_rise`, `season_cum_ili`) are already
# causal w.r.t. the raw target (`np.roll(_, 1)` shift), so they do not
# need a fold-wise recode.
_ABOVE_THRESHOLD_COL: str = "above_threshold"


def _recode_quantile_features_per_fold(
    X_all: np.ndarray,
    feature_cols: List[str],
    train_end: int,
) -> np.ndarray:
    """Rebuild *_qbin / *_qnorm using ONLY X_all[:train_end] as reference.

 Returns a (possibly new) ndarray with only the 4 quantile columns
 updated; the original X_all is never mutated. If the source column
 or either target column is missing, that pair is silently skipped
 (callers keep behaviour).
 """
    if train_end < 10:
        return X_all  # too small for stable quantiles; skip

    fc_index = {name: i for i, name in enumerate(feature_cols)}
    X_out = X_all
    copied = False
    for src_name, n_bins in _QUANTILE_SPECS:
        qbin_name = f"{src_name}_qbin"
        qnorm_name = f"{src_name}_qnorm"
        if (src_name not in fc_index
                or qbin_name not in fc_index
                or qnorm_name not in fc_index):
            continue
        src_idx = fc_index[src_name]
        qbin_idx = fc_index[qbin_name]
        qnorm_idx = fc_index[qnorm_name]

        src_tr = X_all[:train_end, src_idx]
        mask = ~np.isnan(src_tr)
        if mask.sum() < n_bins:
            continue
        bins = np.quantile(src_tr[mask], np.linspace(0, 1, n_bins + 1))
        bins = np.unique(bins)
        if len(bins) < 2:
            continue

        src_full = X_all[:, src_idx]
        qbin_new = np.searchsorted(bins[1:-1], src_full).astype(float)
        denom = max(len(bins) - 2, 1)
        qnorm_new = qbin_new / float(denom)

        if not copied:
            X_out = X_all.copy()
            copied = True
        X_out[:, qbin_idx] = qbin_new
        X_out[:, qnorm_idx] = qnorm_new

    return X_out


# ── S1-1 : interaction-feature max-normalization fold-recode ──────
# transforms._add_interaction_features builds `{src}_ili = (src /
# (src.max() + eps)) * ili_rate_lag1`.  The `src.max()` is a *global*
# summary at build time, so it leaks a single scalar statistic of the
# full series into every fold (CAUSALITY_AUDIT §2.B).  The relative
# feature ordering is preserved by monotonicity, but absolute scale
# drifts between folds.  Here we rebuild the interaction column using
# `max(src[:train_end])` and the (already-causal) `ili_rate_lag1`
# multiplier.
#
# Spec row: (output_col, source_col, eps).  Multiplier is always
# `ili_rate_lag1`.  Three interaction features use other formulas
# (`cold_ili` → clip+abs, `humid_ili` → /100, `peak_ratio_ili` →
# ratio, `emp_contact_ili` → ratio) and are intentionally left out of
# this spec; only the straight `src / (src.max() + eps) * lag1` pattern
# is covered.  `er_burden_ili` is handled by the inverse-max block below.
_INTERACTION_SPECS: Tuple[Tuple[str, str, float], ...] = (
    ("inflow_ili",        "pop_inflow",             1.0),
    ("subway_ili",        "subway_total_avg",       1.0),
    ("bus_ili",           "bus_total_avg",          1.0),
    ("wp_inflow_ili",     "wp_commuter_inflow",     1.0),
    ("hs_congestion_ili", "hs_congestion_ratio",    1.0),
    ("rt_subcrowd_ili",   "rt_sub_acml_total_avg",  1e-6),
    ("rt_roadcong_ili",   "rt_road_cong_avg",       1e-6),
    ("rt_nonresnt_ili",   "rt_popdet_nonresnt_avg", 1e-6),
    ("rt_highrisk_ili",   "rt_popdet_highrisk_age", 1e-6),
)
# S1-1 full-close: inverse-max interaction specs.
# Formula: er_inv = 1/clip(src, 0.1); er_inv_norm = er_inv/(max(er_inv[:train_end])+eps);
#          out = er_inv_norm * lag1
# (matches transforms.py:193-197 build-time formula but with fold-local max)
_INVERSE_MAX_INTERACTION_SPECS: Tuple[Tuple[str, str, float, float], ...] = (
    ("er_burden_ili", "er_bed_avg", 0.1, 1e-6),
)
_INTERACTION_MULTIPLIER_COL: str = "ili_rate_lag1"


def _recode_interaction_features_per_fold(
    X_all: np.ndarray,
    feature_cols: List[str],
    train_end: int,
) -> np.ndarray:
    """Rebuild `{src}_ili` interaction features using ONLY
    X_all[:train_end, src_idx].max() as the denominator.

    Returns a (possibly new) ndarray with only the interaction columns
    updated; `X_all` is never mutated.  Silently skips specs whose
    source or output column is missing (e.g. the interaction block was
    disabled at build time).  Returns the input unchanged when
    `train_end < 10` or when the multiplier column is absent.
    """
    if train_end < 10:
        return X_all
    if _INTERACTION_MULTIPLIER_COL not in feature_cols:
        return X_all
    fc_index = {name: i for i, name in enumerate(feature_cols)}
    mult_idx = fc_index[_INTERACTION_MULTIPLIER_COL]
    lag1_filled = np.nan_to_num(X_all[:, mult_idx], nan=0.0)

    X_out = X_all
    copied = False
    for out_name, src_name, eps in _INTERACTION_SPECS:
        if src_name not in fc_index or out_name not in fc_index:
            continue
        src_idx = fc_index[src_name]
        out_idx = fc_index[out_name]
        src_filled = np.nan_to_num(X_all[:, src_idx], nan=0.0)
        src_tr = src_filled[:train_end]
        denom = float(src_tr.max()) + eps
        if denom <= 0:
            continue
        new_interact = (src_filled / denom) * lag1_filled
        if not copied:
            X_out = X_all.copy()
            copied = True
        X_out[:, out_idx] = new_interact

    # S1-1 full-close: inverse-max interactions (er_burden_ili).
    for out_name, src_name, clip_lo, eps in _INVERSE_MAX_INTERACTION_SPECS:
        if src_name not in fc_index or out_name not in fc_index:
            continue
        src_idx = fc_index[src_name]
        out_idx = fc_index[out_name]
        src_filled = np.nan_to_num(X_all[:, src_idx], nan=1.0)
        # 1 / clip(src, clip_lo)
        src_clipped = np.where(src_filled < clip_lo, clip_lo, src_filled)
        inv = 1.0 / src_clipped
        inv_tr = inv[:train_end]
        denom = float(inv_tr.max()) + eps
        if denom <= 0:
            continue
        inv_norm = inv / denom
        new_interact = inv_norm * lag1_filled
        if not copied:
            X_out = X_all.copy()
            copied = True
        X_out[:, out_idx] = new_interact
    return X_out


def _recode_above_threshold_per_fold(
    X_all: np.ndarray,
    y_all: np.ndarray,
    feature_cols: List[str],
    train_end: int,
) -> np.ndarray:
    """S1-1 : recompute `above_threshold` with `threshold = 2·median(y_all[:train_end])`.

 Mirrors transforms._add_epidemic_phase_features exactly — same lag-1
 roll semantics, same `above_rolled[0] = 0` convention — but replaces
 the fixed 80%-split median with a fold-specific one. `consec_rise`
 and `season_cum_ili` do not depend on the threshold and are left
 untouched (the build-time values are already causal).

 If the column is absent from `feature_cols` (e.g. the epidemic_phase
 block was disabled at build time) the input array is returned
 unchanged. If `train_end` is too small for a stable median, the
 build-time values are kept.
 """
    if _ABOVE_THRESHOLD_COL not in feature_cols:
        return X_all
    if train_end < 10:
        return X_all
    y = np.asarray(y_all, dtype=np.float64)
    base_src = y[:train_end]
    base_src = base_src[~np.isnan(base_src)]
    if len(base_src) == 0:
        return X_all
    baseline = float(np.median(base_src))
    threshold = baseline * 2.0
    above = (y > threshold).astype(np.float64)
    above_rolled = np.roll(above, 1)
    above_rolled[0] = 0.0
    col_idx = feature_cols.index(_ABOVE_THRESHOLD_COL)
    X_out = X_all.copy()
    X_out[:, col_idx] = above_rolled
    return X_out


def _resolve_model_features_strict(per_model_features, feature_cols, model_name):
    """S1-6 fix: resolve per-model feature subset by NAME with strict
    validation. Raises ValueError if any requested feature is missing from
    the current feature_cols (catches silent column-reorder bugs).
    Returns (feat_idx, X_selector) where X_selector is a list of integer
    column indices in feature_cols order.
    """
    if not per_model_features:
        return None
    missing = [f for f in per_model_features if f not in feature_cols]
    if missing:
        raise ValueError(
            f"[{model_name}] per_model_features has {len(missing)} features "
            f"not present in feature_cols (e.g. {missing[:5]!r}). "
            f"This usually means feature ordering drifted between "
            f"Optuna selection and R4 (wfcv). Rerun with --force to rebuild."
        )
    return [feature_cols.index(f) for f in per_model_features]

def _compute_fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """단일 fold의 메트릭 계산."""
    if len(y_true) == 0:
        return {}
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    # MAPE (0 보호)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    else:
        mape = None
    return {"r2": r2, "rmse": rmse, "mae": mae, "mape": mape}


def _aggregate_oof_metrics(y_all: np.ndarray, oof_preds: np.ndarray) -> dict:
    """OOF 예측값의 전체 메트릭 계산."""
    valid = ~np.isnan(oof_preds)
    if valid.sum() == 0:
        return {}
    y_v = y_all[valid]
    p_v = oof_preds[valid]
    metrics = _compute_fold_metrics(y_v, p_v)
    metrics["n_folds"] = int(valid.sum())
    # Temporal stability: 전반 vs 후반
    mid = valid.sum() // 2
    idx = np.where(valid)[0]
    if mid > 10:
        early_y = y_all[idx[:mid]]
        early_p = oof_preds[idx[:mid]]
        late_y = y_all[idx[mid:]]
        late_p = oof_preds[idx[mid:]]
        em = _compute_fold_metrics(early_y, early_p)
        lm = _compute_fold_metrics(late_y, late_p)
        metrics["temporal_stability"] = {
            "early_r2": em.get("r2", 0),
            "late_r2": lm.get("r2", 0),
            "stable": abs(em.get("r2", 0) - lm.get("r2", 0)) < 0.15,
        }

    # R8.2 (2026-05-26): full 134-key SSOT eval via phase_evaluator.
    # OOF predictions = single-model context. Multi-model rankings NaN-fill.
    # Reviewer-defensible: same metric set as R8 (scoring) SSOT for trajectory.
    try:
        from simulation.pipeline.phase_evaluator import evaluate_predictions_full
        residuals_oof = y_v - p_v   # OOF residuals (in-sample for σ)
        full_r8 = evaluate_predictions_full(
            y_test=y_v, y_pred=p_v,
            residuals=residuals_oof,
            y_train_pool=None,   # MASE = NaN (R4 wfcv fold doesn't have separate train pool)
            threshold=GLOBAL.filter.alert_threshold,
            phase_id="phase6_wfcv",
            enable_bootstrap_ci=False,   # opt-out (R4 wfcv inline cost concern)
        )
        metrics["phase_eval_r8"] = full_r8
    except Exception as _e:
        # Backwards-compat: don't break existing 4-key flow if phase_evaluator fails
        metrics["phase_eval_r8_err"] = str(_e)
    return metrics


def _make_default_model_factories() -> dict:
    """WF-CV용 기본 모델 팩토리."""
    from sklearn.linear_model import ElasticNet as _EN
    from sklearn.kernel_ridge import KernelRidge as _KRR
    from sklearn.ensemble import (
        RandomForestRegressor as _RF,
        GradientBoostingRegressor as _GBR,
        AdaBoostRegressor as _Ada,
        ExtraTreesRegressor as _ET,
    )

    def _svr():
        return Pipeline([("scaler", StandardScaler()),
                         ("svr", SVR(kernel="linear", C=10, epsilon=0.01))])
    return {
        "SVR-Linear": _svr,
        "ElasticNet": lambda: Pipeline([("s", StandardScaler()),
                                         ("en", _EN(alpha=0.01, l1_ratio=0.5, max_iter=2000))]),
        "KRR": lambda: Pipeline([("s", StandardScaler()),
                                   ("krr", _KRR(alpha=1.0, kernel="rbf"))]),
        "RandomForest": lambda: _RF(n_estimators=300, max_depth=7,
                                     min_samples_leaf=3, random_state=42, n_jobs=2),
        "XGBoost": lambda: __import__("xgboost", fromlist=["XGBRegressor"]).XGBRegressor(
            n_estimators=200, max_depth=7, learning_rate=0.02,
            subsample=0.7, colsample_bytree=0.6, random_state=42, verbosity=0),
        "GradientBoosting": lambda: _GBR(n_estimators=200, max_depth=5,
                                          learning_rate=0.05, subsample=0.8, random_state=42),
        "AdaBoost": lambda: _Ada(n_estimators=200, learning_rate=0.02, random_state=42),
        "ExtraTrees": lambda: _ET(n_estimators=300, max_depth=7,
                                    min_samples_leaf=3, random_state=42, n_jobs=2),
    }


def run_wfcv_single_model(
    X_all: np.ndarray,
    y_all: np.ndarray,
    feature_cols: List[str],
    model_name: str,
    model_factory: Callable,
    config,
    per_model_features: Optional[List[str]] = None,
    memory_guard=None,
    holdout_start: Optional[int] = None,
    do_pca: bool = False,
) -> dict:
    """단일 모델에 대한 WF-CV 실행.

    핵심 수정 (BUG-1 fix): per_model_features가 있으면 해당 피처만 사용.
    S0-1 fix: WF-CV folds stop at `holdout_start`; after WF-CV, one
    "final" model is refit on X_all[:holdout_start] and used to predict
    X_all[holdout_start:] as the conformal holdout prediction.
    S1-6 fix: per_model_features resolved by strict name check.
    G-232a (2026-05-25): do_pca=True → per-fold PCA (fit on train fold only,
    transform val/holdout) applied after feature subsetting. Leakage-free.
    """
    n_total = len(y_all)
    folds = _generate_wf_folds(n_total, config.wfcv.min_train_weeks,
                               config.wfcv.step_size,
                               holdout_start=holdout_start)
    oof_preds = np.full(n_total, np.nan)

    # ★ BUG-1 FIX + S1-6: strict name-based feature resolution
    feat_idx = _resolve_model_features_strict(per_model_features, feature_cols, model_name)
    if feat_idx:
        _feat_tag = f"{len(feat_idx)}개 (Optuna 선택, name-validated)"
    else:
        _feat_tag = f"{X_all.shape[1]}개 (전체/baseline)"
    if do_pca:
        _feat_tag += " + PCA(95% variance, per-fold)"
    log.info(f"    [{model_name}] 피처: {_feat_tag}")

    fold_metrics = []
    # F3: CV+ (Barber+2021) support — per-fold predictions on the
    # holdout slab so R7 (intervals) can aggregate order-statistic intervals.
    # Shape ends up (n_folds_completed, n_holdout). Only collected when
    # a holdout slab exists; no-op for legacy holdout_start=None runs.
    n_holdout = (n_total - holdout_start) if (holdout_start is not None
                                              and holdout_start < n_total) else 0
    fold_holdout_preds: list[np.ndarray] = []  # len = n_folds_completed
    fold_val_indices: list[tuple[int, int]] = []  # (val_start, val_end) per completed fold
    for fold_idx, (train_end, val_start, val_end) in enumerate(folds):
        # S1-1 P1-5: fold-wise quantile recoding using only [:train_end]
        X_fold_all = _recode_quantile_features_per_fold(X_all, feature_cols, train_end)
        # S1-1 : fold-wise epidemic-phase threshold — replaces the
        #            global median(y[:0.8·n]) with median(y[:train_end]).
        X_fold_all = _recode_above_threshold_per_fold(
            X_fold_all, y_all, feature_cols, train_end,
        )
        # S1-1 : fold-wise interaction max-norm — replaces global
        #            src.max() with src[:train_end].max() (CAUSALITY_AUDIT §2.B).
        X_fold_all = _recode_interaction_features_per_fold(
            X_fold_all, feature_cols, train_end,
        )
        X_use = X_fold_all[:, feat_idx] if feat_idx else X_fold_all
        X_tr = X_use[:train_end]
        y_tr = y_all[:train_end]
        X_va = X_use[val_start:val_end]
        y_va = y_all[val_start:val_end]

        # G-232a: per-fold PCA (fit on X_tr only → no val/holdout leakage)
        _fold_pca_state = None
        if do_pca and X_tr.shape[1] >= 2:
            try:
                from sklearn.preprocessing import StandardScaler as _SS
                from sklearn.decomposition import PCA as _PCA
                _sc = _SS()
                _pc = _PCA(n_components=0.95, random_state=42, svd_solver="full")
                X_tr = _pc.fit_transform(_sc.fit_transform(X_tr))
                X_va = _pc.transform(_sc.transform(X_va))
                _fold_pca_state = (_sc, _pc)
            except Exception as _pca_e:
                log.debug(f"    [{model_name}] fold PCA failed: {_pca_e} — using raw features")
                _fold_pca_state = None

        if len(y_va) < 1:
            continue

        try:
            model = model_factory()
            model.fit(X_tr, y_tr)
            pred = model.predict(X_va)
            pred = np.maximum(pred, 0)  # 음수 방지
            oof_preds[val_start:val_end] = pred
            fm = _compute_fold_metrics(y_va, pred)
            fold_metrics.append(fm)
            # F3: also predict the holdout slab with this fold-model.
            # X_use uses the same fold-wise recoding as training, so the
            # holdout rows see the same transformation the fold saw.
            # G-232a: if PCA was applied this fold, transform holdout slab too.
            if n_holdout > 0:
                try:
                    if _fold_pca_state is not None:
                        _sc_f, _pc_f = _fold_pca_state
                        _X_ho = _pc_f.transform(_sc_f.transform(X_use[holdout_start:]))
                    else:
                        _X_ho = X_use[holdout_start:]
                    hp_fold = model.predict(_X_ho)
                    hp_fold = np.maximum(np.asarray(hp_fold, dtype=np.float64), 0)
                    fold_holdout_preds.append(hp_fold)
                    fold_val_indices.append((val_start, val_end))
                except Exception as _ehp:
                    log.debug(
                        f"    [{model_name}] Fold {fold_idx} holdout 예측 실패: {_ehp}"
                    )
        except Exception as e:
            log.debug(f"    [{model_name}] Fold {fold_idx} 실패: {e}")
        # 메모리 관리
        if memory_guard and config.memory.gc_after_each_fold:
            memory_guard.check_and_gc(f"{model_name} fold {fold_idx}")

    overall = _aggregate_oof_metrics(y_all, oof_preds)

    # S0-1 fix: final-model holdout prediction for split conformal
    holdout_preds = None
    if holdout_start is not None and holdout_start < n_total:
        try:
            # P1-5: holdout refit also uses fold-wise quantile recoding at train_end=holdout_start
            X_h_all = _recode_quantile_features_per_fold(X_all, feature_cols, holdout_start)
            # S1-1 : same fold-wise above_threshold recode at holdout train_end
            X_h_all = _recode_above_threshold_per_fold(
                X_h_all, y_all, feature_cols, holdout_start,
            )
            # S1-1 : holdout refit also uses fold-wise interaction max-norm
            X_h_all = _recode_interaction_features_per_fold(
                X_h_all, feature_cols, holdout_start,
            )
            X_use_h = X_h_all[:, feat_idx] if feat_idx else X_h_all
            # G-232a: holdout refit PCA — fit on [:holdout_start], transform [holdout_start:]
            if do_pca and X_use_h.shape[1] >= 2:
                try:
                    from sklearn.preprocessing import StandardScaler as _SS
                    from sklearn.decomposition import PCA as _PCA
                    _sc_h = _SS()
                    _pc_h = _PCA(n_components=0.95, random_state=42, svd_solver="full")
                    _X_tr_h = _pc_h.fit_transform(_sc_h.fit_transform(X_use_h[:holdout_start]))
                    _X_ho_h = _pc_h.transform(_sc_h.transform(X_use_h[holdout_start:]))
                    X_use_h_tr = _X_tr_h
                    X_use_h_ho = _X_ho_h
                except Exception as _hpca_e:
                    log.debug(f"    [{model_name}] holdout PCA failed: {_hpca_e}")
                    X_use_h_tr = X_use_h[:holdout_start]
                    X_use_h_ho = X_use_h[holdout_start:]
            else:
                X_use_h_tr = X_use_h[:holdout_start]
                X_use_h_ho = X_use_h[holdout_start:]
            final_model = model_factory()
            final_model.fit(X_use_h_tr, y_all[:holdout_start])
            hp = final_model.predict(X_use_h_ho)
            holdout_preds = np.maximum(np.asarray(hp, dtype=np.float64), 0)
            log.info(
                f"    [{model_name}] holdout refit: {holdout_start}→{n_total} "
                f"({n_total - holdout_start} pts)"
            )
        except Exception as e:
            log.warning(f"    [{model_name}] holdout refit 실패: {e}")

    # F3: stack per-fold holdout predictions into a (K, H) matrix
    # for CV+. None if no holdout slab or no fold succeeded.
    fold_holdout_matrix = None
    if fold_holdout_preds:
        try:
            fold_holdout_matrix = np.vstack(fold_holdout_preds)
        except Exception as _e:
            log.debug(f"    [{model_name}] fold_holdout stack 실패: {_e}")

    return {
        "oof_preds": oof_preds,
        "holdout_preds": holdout_preds,
        "fold_holdout_preds": fold_holdout_matrix,   # F3: (K, H) or None
        "fold_val_indices": fold_val_indices,        # F3: per-fold val window
        "overall_metrics": overall,
        "n_folds_completed": len(fold_metrics),
        "fold_metrics": fold_metrics,
    }


def run_wfcv_with_inline_optuna(
    X_all: np.ndarray,
    y_all: np.ndarray,
    feature_cols: List[str],
    model_name: str,
    model_factory: Callable,
    config,
    memory_guard=None,
    holdout_start: Optional[int] = None,
) -> dict:
    """Inline Optuna + WF-CV 통합.

    retune_every fold마다 Optuna로 피처 재탐색.
    warm-start: 이전 best를 다음 Optuna 초기값으로 사용.
    """
    folds = _generate_wf_folds(len(y_all), config.wfcv.min_train_weeks,
                               config.wfcv.step_size,
                               holdout_start=holdout_start)
    n_total = len(y_all)
    oof_preds = np.full(n_total, np.nan)

    best_features = None
    best_params = None
    retune_count = 0
    _optuna_import_failed = False          # G-126: 한 번만 경고
    # F3: CV+ per-fold holdout predictions (inline-Optuna path).
    n_holdout = (n_total - holdout_start) if (holdout_start is not None
                                              and holdout_start < n_total) else 0
    fold_holdout_preds: list[np.ndarray] = []
    fold_val_indices: list[tuple[int, int]] = []
    for fold_idx, (train_end, val_start, val_end) in enumerate(folds):
        # Inline Optuna retune
        if (not _optuna_import_failed
                and (fold_idx % config.wfcv.retune_every == 0 or best_features is None)):
            try:
                from simulation.tools.run_optuna_feature_selection import run_inline_for_model
                result = run_inline_for_model(
                    model_name,
                    X_data=X_all[:train_end],
                    y_data=y_all[:train_end],
                    feat_cols=feature_cols,
                    n_trials=config.optuna.trials,
                    cv_folds=config.optuna.cv_folds,
                    strategy=config.optuna.strategy,
                )
                if result and isinstance(result, tuple) and len(result) >= 2:
                    _, best_result = result
                    if isinstance(best_result, dict):
                        new_features = best_result.get("best_features", [])
                        if new_features:
                            best_features = new_features
                            retune_count += 1
                            log.info(f"    [{model_name}] Fold {fold_idx}: Optuna retune #{retune_count}, "
                                     f"{len(best_features)}개 피처")
            except ImportError:
                # G-126: 모듈 누락은 첫 fold 에서만 경고, 이후 skip
                if not _optuna_import_failed:
                    log.warning(f"    [{model_name}] inline Optuna 모듈 없음 — 전체 피처로 진행")
                    _optuna_import_failed = True
            except Exception as e:
                log.warning(f"    [{model_name}] Fold {fold_idx} Optuna 실패: {e}")

        # 피처 서브셋 적용 + S1-1 fold-wise recodes (quantile + above_threshold + interaction)
        X_fold_all = _recode_quantile_features_per_fold(X_all, feature_cols, train_end)
        X_fold_all = _recode_above_threshold_per_fold(
            X_fold_all, y_all, feature_cols, train_end,
        )
        X_fold_all = _recode_interaction_features_per_fold(
            X_fold_all, feature_cols, train_end,
        )
        if best_features:
            feat_idx = [feature_cols.index(f) for f in best_features if f in feature_cols]
            X_use = X_fold_all[:, feat_idx] if feat_idx else X_fold_all
        else:
            X_use = X_fold_all
        X_tr = X_use[:train_end]
        y_tr = y_all[:train_end]
        X_va = X_use[val_start:val_end]
        y_va = y_all[val_start:val_end]

        if len(y_va) < 1:
            continue

        try:
            model = model_factory()
            model.fit(X_tr, y_tr)
            pred = np.maximum(model.predict(X_va), 0)
            oof_preds[val_start:val_end] = pred
            # F3: per-fold holdout prediction for CV+
            if n_holdout > 0:
                try:
                    hp_fold = model.predict(X_use[holdout_start:])
                    hp_fold = np.maximum(np.asarray(hp_fold, dtype=np.float64), 0)
                    fold_holdout_preds.append(hp_fold)
                    fold_val_indices.append((val_start, val_end))
                except Exception as _ehp:
                    log.debug(
                        f"    [{model_name}] Fold {fold_idx} holdout 예측 실패 (inline): {_ehp}"
                    )
        except Exception as e:
            log.debug(f"    [{model_name}] Fold {fold_idx} 학습 실패: {e}")

        if memory_guard and fold_idx % 50 == 0:
            memory_guard.check_and_gc(f"{model_name} inline fold {fold_idx}")

    overall = _aggregate_oof_metrics(y_all, oof_preds)
    overall["retune_count"] = retune_count

    # S0-1 fix: inline 모드에서도 holdout refit 수행
    # P1-5: holdout refit 도 fold-wise quantile recoding 적용 (train_end=holdout_start)
    n_total = len(y_all)
    holdout_preds = None
    if holdout_start is not None and holdout_start < n_total:
        try:
            X_h_all = _recode_quantile_features_per_fold(X_all, feature_cols, holdout_start)
            X_h_all = _recode_above_threshold_per_fold(
                X_h_all, y_all, feature_cols, holdout_start,
            )
            X_h_all = _recode_interaction_features_per_fold(
                X_h_all, feature_cols, holdout_start,
            )
            X_use = X_h_all
            if best_features:
                feat_idx = [feature_cols.index(f) for f in best_features if f in feature_cols]
                if feat_idx:
                    X_use = X_h_all[:, feat_idx]
            final_model = model_factory()
            final_model.fit(X_use[:holdout_start], y_all[:holdout_start])
            hp = final_model.predict(X_use[holdout_start:])
            holdout_preds = np.maximum(np.asarray(hp, dtype=np.float64), 0)
            log.info(f"    [{model_name}] holdout refit: {holdout_start}→{n_total} "
                     f"({n_total - holdout_start} pts)")
        except Exception as e:
            log.warning(f"    [{model_name}] holdout refit 실패: {e}")

    # F3: stack per-fold holdout preds for CV+ (inline mode).
    fold_holdout_matrix = None
    if fold_holdout_preds:
        try:
            fold_holdout_matrix = np.vstack(fold_holdout_preds)
        except Exception as _e:
            log.debug(f"    [{model_name}] fold_holdout stack 실패 (inline): {_e}")

    return {
        "oof_preds": oof_preds,
        "holdout_preds": holdout_preds,
        "fold_holdout_preds": fold_holdout_matrix,   # F3: (K, H) or None
        "fold_val_indices": fold_val_indices,        # F3
        "overall_metrics": overall,
        "best_features": best_features,
        "n_retunes": retune_count,
    }


from simulation.utils.resource_tracker import track_resources


@track_resources("phase6_wfcv")
def run_wfcv(X_all, y_all, feature_cols, config,
               per_model_feature_map: Optional[Dict[str, List[str]]] = None,
               memory_guard=None,
               holdout_start: Optional[int] = None,
               per_model_pca_feat_set: Optional[Dict[str, List[str]]] = None) -> dict:
    """R4 (wfcv) 전체 실행.

    Args:
        per_model_feature_map: {model_name: [feature_names]} — external/inline Optuna 결과
        holdout_start: S0-1 — index at which the conformal holdout begins.
            Folds stop at this index; a final model is refit on [:holdout_start]
            and used to predict [holdout_start:] as conformal test preds.
        per_model_pca_feat_set: {model_name: [feature_names]} — G-232a PCA path.
            Models in this dict apply per-fold PCA (fit on train, transform val)
            instead of direct feature subsetting. Passed through to run_wfcv_single_model.
    """
    from .utils.logging_util import phase_banner, fmt_time
    phase_banner("R4", "Walk-Forward CV + Inline Optuna")
    t0 = time.time()
    model_factories = _make_default_model_factories()

    # C-step: PAPER_PRIMARY_11 only mode
    #   해당 flag 가 켜지면 registry 의 PAPER_PRIMARY_11 과 factory 이름의
    #   교집합만 WF-CV 로 돌리고, step_size 도 paper-primary 전용으로 상향
    #   (fold 수 감축 → 학습 시간 단축, 결과 변동성 감소).
    if getattr(config.wfcv, "paper_primary_only", False):
        try:
            from simulation.models.registry import PAPER_PRIMARY_11
            primary_names = {name for name, _ in PAPER_PRIMARY_11}
            filtered = {k: v for k, v in model_factories.items() if k in primary_names}
            dropped = sorted(set(model_factories) - set(filtered))
            if filtered:
                log.info(
                    f"  [C-step] paper_primary_only=True → "
                    f"{len(filtered)}/{len(model_factories)} 모델만 실행 "
                    f"(제외: {dropped})"
                )
                model_factories = filtered
            else:
                log.warning(
                    "  [C-step] paper_primary_only=True 이지만 factory 교집합 0 → "
                    "전체 factory 실행 fallback"
                )
            # step_size 도 paper-primary 모드 값으로 override (원본 보존)
            _orig_step = config.wfcv.step_size
            config.wfcv.step_size = int(config.wfcv.step_size_paper_primary)
            log.info(
                f"  [C-step] step_size {_orig_step} → {config.wfcv.step_size} "
                f"(paper-primary fold 압축 모드)"
            )
        except Exception as _e:
            log.warning(f"  [C-step] PAPER_PRIMARY_11 적용 실패: {_e} — 기본 factory 로 진행")

    # partial-refit: --models CLI 로 선택한 모델만 WF-CV 실행.
    #   R2 baseline 의 include_only 와 동일한 시맨틱. 여기서 dict 만 좁히고
    #   실제 sidecar 병합(이전 OOF 유지 + 새 모델 덮어쓰기) 는 pipeline/runner.py
    #   의 R4 (wfcv) save 블록에서 처리.
    _selected = getattr(config, "_selected_models", None) or []
    if _selected:
        keep = set(_selected)
        filtered = {k: v for k, v in model_factories.items() if k in keep}
        dropped = sorted(set(model_factories) - set(filtered))
        if filtered:
            log.info(
                f"  [partial] --models 필터 적용: "
                f"{len(filtered)}/{len(model_factories)} 모델만 WF-CV 실행 "
                f"(유지: {sorted(filtered)}, 제외: {dropped})"
            )
            model_factories = filtered
        else:
            log.warning(
                f"  [partial] --models={_selected} 와 factory 교집합 0 → "
                "R4 (wfcv) 에서는 해당 모델이 factory 에 없음. "
                "(GAM-Spline/PoissonAutoreg/NegBinGLM 등은 baseline/external REGISTRY(R2/R3) 전용이며 "
                "R4 (wfcv) factory 에 포함되지 않음) — R4 (wfcv) 전체 skip."
            )
            model_factories = {}

    wf_results = {}
    holdout_predictions: Dict[str, np.ndarray] = {}
    # F3: per-model per-fold holdout predictions for CV+.
    fold_holdout_map: Dict[str, np.ndarray] = {}
    fold_val_indices_map: Dict[str, list] = {}

    # 모드별 실행
    optuna_mode = config.optuna.mode

    for model_name, model_factory in model_factories.items():
        log.info(f"    [{model_name}] Walk-Forward 실행 중...")

        if optuna_mode == "inline":
            # Inline Optuna: WF-CV 내부에서 피처 탐색
            result = run_wfcv_with_inline_optuna(
                X_all, y_all, feature_cols, model_name, model_factory,
                config, memory_guard, holdout_start=holdout_start,
            )
        else:
            # Baseline 또는 External: 사전 지정 피처 사용 (없으면 None = 전체 feature).
            # feature 선택은 R9 (per_model_optimize) 에서만 (2026-06-01); 여기 map 은 external 모드만 채움.
            model_features = None
            if per_model_feature_map and model_name in per_model_feature_map:
                model_features = per_model_feature_map[model_name]
            # G-232a: PCA path — per-fold PCA applied inside run_wfcv_single_model
            _do_pca_p7 = bool(
                per_model_pca_feat_set and model_name in per_model_pca_feat_set
            )

            try:
                result = run_wfcv_single_model(
                    X_all, y_all, feature_cols, model_name, model_factory,
                    config, per_model_features=model_features,
                    memory_guard=memory_guard,
                    holdout_start=holdout_start,
                    do_pca=_do_pca_p7,
                )
            except ValueError as ve:
                # S1-6: strict feature validation raised
                log.error(f"    [{model_name}] feature mismatch: {ve}")
                continue

        if result.get("holdout_preds") is not None:
            holdout_predictions[model_name] = result["holdout_preds"]

        # F3: collect per-fold holdout matrix if produced by the
        # WF-CV loop so R7 (intervals) can run CV+ / jackknife+ intervals.
        _fhp = result.get("fold_holdout_preds")
        if _fhp is not None:
            fold_holdout_map[model_name] = _fhp
            fold_val_indices_map[model_name] = result.get("fold_val_indices", [])

        m = result.get("overall_metrics", {})
        mape_s = f"MAPE={m.get('mape', 0):.2f}%" if m.get("mape") else "MAPE=N/A"
        log.info(f"    [{model_name:15s}]  R²={m.get('r2',0):.4f}  {mape_s}  "
                 f"RMSE={m.get('rmse',0):.2f}  ({result.get('n_folds_completed', m.get('n_folds', 0))} folds)")

        wf_results[model_name] = {
            "overall_metrics": m,
            "oof_preds": result.get("oof_preds"),
        }

        if memory_guard:
            memory_guard.check_and_gc(f"after {model_name} WF-CV")
    # 정렬된 결과 출력
    log.info("")
    log.info("  --- Walk-Forward CV 결과 (MAPE 순) ---")
    sorted_results = sorted(wf_results.items(),
                            key=lambda x: x[1]["overall_metrics"].get("mape", 999))
    for name, r in sorted_results:
        m = r["overall_metrics"]
        mape_s = f"{m.get('mape',0):.2f}%" if m.get("mape") else "N/A"
        log.info(f"    {name:18s}  R²={m.get('r2',0):.4f}  MAPE={mape_s}  RMSE={m.get('rmse',0):.2f}")

    elapsed = time.time() - t0
    log.info(f"  ✓ R4 wfcv 완료 [{fmt_time(elapsed)}]")

    # ──────────────────────────────────────────────────────────────────
    # Stage 4 — epi-validity gate
    # ──────────────────────────────────────────────────────────────────
    # Cheap post-hoc check over each model's OOF + holdout forecasts.
    # Flags biologically implausible outputs (Rt jumps, summer peaks,
    # prediction NaN/negatives, sub-0.3 Rt elimination floor, …).
    # Default = flag only; `--epi-validity-strict` sets the exclude flag.
    epi_gate_cfg = getattr(config, "epi_validity", None)
    epi_gate: dict = {}
    if epi_gate_cfg is None or epi_gate_cfg.enabled:
        try:
            from simulation.verifier.epi_validity import run_epi_validity_gate

            # BUG-F fix: OOF 는 min_train(120) 이전 + holdout 구간에
            #   NaN 이 박힌 길이 344 벡터. 과거엔 146 개 NaN 을 그대로 gate 로
            #   넘겨서 8/8 모델이 "146 NaN values" flag 됐다. np.isfinite 로
            #   필터링해 실제 예측 지점만 검사.
            model_outputs: Dict[str, dict] = {}
            for name, res in wf_results.items():
                oof = res.get("oof_preds")
                hold = holdout_predictions.get(name)
                preds = None
                if oof is not None and hold is not None:
                    preds = np.concatenate([np.asarray(oof), np.asarray(hold)])
                elif oof is not None:
                    preds = np.asarray(oof)
                elif hold is not None:
                    preds = np.asarray(hold)
                if preds is not None:
                    preds = np.asarray(preds, dtype=float)
                    preds = preds[np.isfinite(preds)]
                    if preds.size > 0:
                        model_outputs[name] = {"predictions": preds}

            strict = bool(epi_gate_cfg.strict_exclude) if epi_gate_cfg else False
            epi_gate = run_epi_validity_gate(model_outputs, strict_exclude=strict)

            n_fail = sum(1 for v in epi_gate.values() if v.get("status") == "fail")
            n_exc = sum(1 for v in epi_gate.values()
                        if v.get("exclude_from_ensemble"))
            log.info(
                f"  · epi-validity gate: {len(epi_gate)} models scanned, "
                f"{n_fail} flagged, {n_exc} marked exclude"
                + (" (strict_exclude=ON)" if strict else " (flag-only)")
            )
            for name, rep in epi_gate.items():
                if rep.get("status") == "fail":
                    viols = "; ".join(rep.get("violations", [])[:3])
                    log.info(f"    ⚠ {name}: {viols}")
        except Exception as e:  # defensive — gate must never kill training
            log.warning(f"  · epi-validity gate skipped (error): {e}")
            epi_gate = {"_error": str(e)}
    else:
        log.info("  · epi-validity gate disabled (config.epi_validity.enabled=False)")

    # Sprint 3 EDA sidecar (2026-05-26) — non-fatal, atomic write
    try:
        import numpy as _np
        from pathlib import Path as _Path
        from .eda_writer import write_phase_eda
        # OOF predictions full-length; restrict to non-NaN portion + match y_all length
        _oof_preds = {
            k: _np.asarray(v.get("oof_preds"))
            for k, v in wf_results.items()
            if v.get("oof_preds") is not None
        }
        if _oof_preds:
            # All OOF arrays should match y_all length; eda_writer filters mismatches.
            write_phase_eda(
                phase_id=7, phase_tag="wfcv",
                y_true=_np.asarray(y_all).ravel(),
                predictions=_oof_preds,
                save_dir=_Path(getattr(config, "save_dir", "simulation/results")) / "eda",
                extra_meta={"holdout_start": int(holdout_start) if holdout_start else None,
                             "n_models": len(_oof_preds)},
            )
    except Exception as _eda_e:
        log.debug(f"  [R4 wfcv] EDA sidecar skipped: {_eda_e}")

    return {
        "wf_results": {k: {kk: vv for kk, vv in v.items() if kk != "oof_preds"}
                       for k, v in wf_results.items()},
        "oof_predictions": {k: v.get("oof_preds") for k, v in wf_results.items()},
        "holdout_predictions": holdout_predictions,   # S0-1
        "holdout_start": holdout_start,                # S0-1
        # F3: per-fold holdout prediction matrices + fold val windows
        # for CV+ / jackknife+ conformal in R7 (intervals).
        "fold_holdout_predictions": fold_holdout_map,
        "fold_val_indices": fold_val_indices_map,
        "epi_validity_gate": epi_gate,                 # Stage 4
        "elapsed": time.time() - t0,
    }


# back-compat aliases (2026-06-02 semantic rename — 옛 run_phaseN)
run_phase6 = run_wfcv
