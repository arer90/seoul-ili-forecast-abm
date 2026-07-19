"""Inline 3-stage Optuna — R9/R10 공유 module (2026-05-28).

사용자 명시 (2026-05-27 design A): R10 per_model_eval (research) + R9 per_model_optimize
(service) 공유 3-stage Optuna entry — preproc → feature → HP. 두 phase 가 같은 module
호출하되 mode 별 validation 분리.

설계 의도:
  - R10 per_model_eval = research용 (paper / 학술 분석, stricter validation)
  - R9 per_model_optimize = service용 (production / real-time inference, oof_cv default)
  - 공유 logic = 3-stage Optuna (preproc 100 + feature 20 + HP 20 default)
  - 분리 logic = validation mode (CV folds, metric, gate threshold)

3-stage 흐름 (per model):
  Stage 1: preproc Optuna (transform × scaler search, n_trials=100)
           - feature=all, HP=default
           - validation = WF-CV WIS minimize
  Stage 2: feature Optuna (feature_subset search, n_trials=20)
           - preproc=Stage 1 best, HP=default
           - validation = WF-CV WIS minimize
  Stage 3: HP Optuna (model internal, n_trials=20)
           - preproc=Stage 1 best, feature=Stage 2 best
           - factory_fn 안의 자체 Optuna (XGBoost/LightGBM/CatBoost default=20)
           - validation = model 자체 TimeSeriesSplit (3-fold)

mode 분리:
  service (R9 per_model_optimize):
    - validation = oof_cv (3-fold WF-CV) WIS minimize
    - output: simulation/results/per_model_optimal/<MODEL>.json
    - 목적: real-time inference champion 선택
  research (R10 per_model_eval):
    - validation = stricter (5-fold or LORO CV, MCS membership)
    - output: simulation/results/per_model_research/<MODEL>.json
    - 목적: paper-grade evaluation + statistical comparison

References:
  - PHASE_AUDIT §S9.3 nested Optuna 구조
  - SPRINT_REVIEW.md §7.2 R9/R10 design
  - 사용자 명시 (2026-05-28): "R10 per_model_eval research / R9 per_model_optimize service 분리"
"""

from __future__ import annotations

import json
import logging
import os
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from simulation.config_global import GLOBAL  # SSOT (2026-05-28)
from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)

log = logging.getLogger(__name__)


def _aggregate_oof_folds(scores, fold_maxes=None, outbreak_level=None) -> float:
    """Aggregate per-fold OOF scores into one selection number (lower = better).

    G-255 (2026-06-12): replaced ``np.median``. The old median (D5, 2026-05-30, single-bad-fold
    robustness) silently DISCARDED the outbreak folds — with 5 expanding walk-forward folds only
    ~2 of them contain an epidemic peak (the rest are quiet low-flu folds), so the median lands
    on a quiet fold and a peak-blind config wins selection; the Seoul ILI test slab IS a peak
    (100.7 > train 66.9) → collapse (LightGBM r2=0.313). Part B (exp_peak_extrapolation.py):
    mean ≥ median always, +0.15 CatBoost.

    G-256b (2026-06-12, codex+gemini): REGIME-CONDITIONAL when fold context is supplied. Plain
    mean weights folds by error MAGNITUDE, so the few high-error outbreak folds can dominate (a
    config that is poor in quiet weeks — 90% of operation — but lucky on the peak could win).
    When ``fold_maxes`` + ``outbreak_level`` are given, split folds into quiet (val max ≤ level)
    vs elevated (> level) and give the two REGIMES equal weight: neither the many quiet folds
    nor the few outbreak folds dominate. Falls back to plain mean (G-255) when fold context is
    absent or only one regime is present. Single-glitch robustness: non-finite folds are dropped.

    Args:
        scores: per-fold scores (WIS / error). Finite-filtered here defensively.
        fold_maxes: per-fold validation-target max (same order as ``scores``); enables the
            regime split. ``None`` → plain mean.
        outbreak_level: ILI level above which a fold is "elevated" (e.g. 75th pct of y_train).
            ``None`` → plain mean.

    Returns:
        Regime-balanced (or plain) mean of finite scores, or ``float('inf')`` if none finite.
    """
    if fold_maxes is None:
        fold_maxes = [None] * len(list(scores)) if not hasattr(scores, "__len__") else [None] * len(scores)
    pairs = [(float(s), (None if m is None else float(m)))
             for s, m in zip(scores, fold_maxes) if np.isfinite(s)]
    if not pairs:
        return float("inf")
    vals = [s for s, _ in pairs]
    if outbreak_level is None or any(m is None for _, m in pairs):
        return float(np.mean(vals))                                   # G-255 plain mean
    quiet = [s for s, m in pairs if m <= outbreak_level]
    elevated = [s for s, m in pairs if m > outbreak_level]
    if not quiet or not elevated:
        return float(np.mean(vals))                                   # single regime → plain mean
    return float(0.5 * np.mean(quiet) + 0.5 * np.mean(elevated))      # G-256b regime-balanced


# ────────────────────────────────────────────────────────────────────────────
# Stage 2 — Feature subset Optuna (phase0b logic 이동, 2026-05-28)
# ────────────────────────────────────────────────────────────────────────────
# Origin: simulation/pipeline/phase3_feature_optuna.py (244 LOC, deprecation alias 화)
# 의도 (사용자 명시 2026-05-28 design A): R9/R10 공유 module 안에 logic.
# Helpers: tools/run_optuna_feature_selection.py 의 _fit_predict / _compute_score /
#          _default_hp / MANDATORY_FEATURES_EXACT / _COMMON_KEY_MAP 에 의존.

def _model_to_optuna_key(model_name: str) -> str:
    """모델 이름 → optuna feature selection key (tools/run_optuna_feature_selection.py).

    G-236 후속 (2026-05-29, Codex+Gemini): 이전엔 `_COMMON_KEY_MAP`(19, DL/sequence
    누락)을 써서 TCN/N-BEATS/PatchTST/iTransformer/Mamba/TimesNet 등이 silent 하게
    "elasticnet" proxy 로 떨어짐(부적합 feature 선택). `_CATEGORY_KEY_MAP`(29)는 이들을
    "dnn" proxy 로 매핑(MODELS_QUICK/REPRESENTATIVE 에 'dnn' study 존재 = 보장) + builder
    `_OPTUNA_MODEL_MAP_INDIVIDUAL` 와 일치. unknown(stat/epi 등)은 elasticnet fallback +
    WARN(더 이상 silent X — G-159 "no silent default").
    """
    import logging
    _log = logging.getLogger(__name__)
    try:
        from simulation.tools.run_optuna_feature_selection import _CATEGORY_KEY_MAP
        key = _CATEGORY_KEY_MAP.get(model_name)
        if key is None:
            _log.warning("[G-236후속] _model_to_optuna_key: '%s' 미매핑 → elasticnet "
                         "proxy (feature-optuna 키 누락 — 신규/rename 모델이면 _CATEGORY_KEY_MAP 갱신)",
                         model_name)
            return "elasticnet"
        return key
    except ImportError:
        return "elasticnet"


def _wf_cv_wis_simple(X: np.ndarray, y: np.ndarray, model_name: str,
                      n_folds: int = 5) -> float:
    """Simple WF-CV WIS for Stage 2 feature Optuna (proxy 평가).

    Origin: phase3_feature_optuna._wf_cv_wis_simple (L200-236).
    """
    n = len(y)
    if n < (n_folds + 1) * 10:
        return float("inf")
    from simulation.tools.run_optuna_feature_selection import (
        _fit_predict, _compute_score, _default_hp,
    )
    model_key = _model_to_optuna_key(model_name)
    hp = _default_hp(model_key)

    fold_size = n // (n_folds + 1)
    scores = []
    fold_maxes: list[float] = []   # G-256b: per-fold val max for regime-conditional aggregation
    outbreak_level = float(np.percentile(np.asarray(y, dtype=float).ravel(), 75))
    for k in range(1, n_folds + 1):
        end_tr = k * fold_size
        end_va = (k + 1) * fold_size if k < n_folds else n
        if end_va - end_tr < 4:
            continue
        try:
            pred = _fit_predict(model_key, hp, X[:end_tr], y[:end_tr], X[end_tr:end_va])
            try:
                pred_in = _fit_predict(
                    model_key, hp, X[:end_tr - fold_size], y[:end_tr - fold_size],
                    X[end_tr - fold_size:end_tr],
                )
                resid = y[end_tr - fold_size:end_tr] - pred_in
            except Exception:
                resid = np.diff(y[:end_tr])
            score = _compute_score(y[end_tr:end_va], pred, resid, "wis")
            if np.isfinite(score):
                scores.append(score)
                fold_maxes.append(float(np.asarray(y[end_tr:end_va], dtype=float).max()))
        except Exception:
            continue
    # G-255/G-256b (2026-06-12): regime-balanced mean over folds (median discarded outbreak folds
    # → peak-blind selection → test-peak collapse). Regime split keeps the few outbreak folds
    # from being drowned out and the many quiet folds from being ignored. See helper.
    return _aggregate_oof_folds(scores, fold_maxes, outbreak_level)


def _stage2_feature_optuna_inline(
    phase1: dict,
    model_names: list[str],
    n_trials_per_model: int = 20,    # 2026-05-28 사용자 명시 budget (was 30)
    output_dir: Optional[Path] = None,
) -> dict:
    """Stage 2 (옛 Phase 0b/1.5b): per-model feature subset Optuna search on raw X.

    Origin: phase3_feature_optuna.run_phase3_feature_optuna (L51-197).
    Inline in 2026-05-28 (사용자 명시 design A — R9/R10 공유 module).

    Args:
        phase1: R1 data 결과 (X_all, y_all, feature_cols, n_train, pool_end)
        model_names: 처리할 모델 list (53 model individual)
        n_trials_per_model: 모델당 Optuna trials (default 20, 사용자 명시 budget)
        output_dir: Stage 2 결과 저장 위치 (None → get_results_dir()/stage2_feature_optuna, MPH_OUTPUT_ROOT 존중)

    Returns:
        {model_name: {best_feature_subset, best_score_oof_wis, ...}}

    Performance: O(n_models × n_trials × WF_CV_cost)
    Side effects: writes ``output_dir/<model>.json`` per model + ``_summary.json``
    """
    import optuna
    from optuna.samplers import TPESampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore")

    if output_dir is None:
        output_dir = get_results_dir() / "stage2_feature_optuna"
    output_dir.mkdir(parents=True, exist_ok=True)

    X_all = phase1["X_all"]
    y_all = phase1["y_all"]
    feature_cols = phase1["feature_cols"]
    n_train = phase1["n_train"]
    pool_end = phase1.get("pool_end", n_train + phase1.get("n_val", 0))

    X_pool = X_all[:pool_end]
    y_pool = y_all[:pool_end]

    try:
        from simulation.tools.run_optuna_feature_selection import MANDATORY_FEATURES_EXACT
        mandatory_set = MANDATORY_FEATURES_EXACT
    except ImportError:
        mandatory_set = set()

    log.info(f"  [Stage 2 feature] {len(model_names)} models, "
             f"X_pool={X_pool.shape}, n_trials={n_trials_per_model}, "
             f"mandatory={len(mandatory_set)}")

    results = {}
    for model_name in model_names:
        X_pool_processed = X_pool
        kept_cols = list(feature_cols)
        n_kept = len(kept_cols)
        log.info(f"    [{model_name}] raw X: X={X_pool.shape}")

        mand_mask = np.array([c in mandatory_set for c in kept_cols], dtype=bool)
        n_mandatory = int(mand_mask.sum())

        def _objective(trial):
            mask = np.zeros(n_kept, dtype=bool)
            for i, col in enumerate(kept_cols):
                if mand_mask[i]:
                    mask[i] = True
                else:
                    mask[i] = trial.suggest_categorical(f"use_{col}", [True, False])
            n_sel = mask.sum()
            if n_sel < max(5, n_mandatory + 3):
                return float("inf")
            X_sel = X_pool_processed[:, mask]
            score = _wf_cv_wis_simple(X_sel, y_pool, model_name, n_folds=3)
            penalty = 0.001 * n_sel
            return score + penalty

        try:
            study = optuna.create_study(
                study_name=f"stage2_feat_{model_name}_{int(time.time())}",
                direction="minimize",
                sampler=TPESampler(multivariate=True, n_startup_trials=5, seed=42),  # G-13F: 재현성
                pruner=__import__(
                    "simulation.models._optuna_pruners",
                    fromlist=["get_pruner_for_stage"]
                ).get_pruner_for_stage("stage2"),
            )
            # G-161 parity (R8 2026-05-28): stage1(L434) 과 동일하게 trial 간 GC.
            study.optimize(_objective, n_trials=n_trials_per_model,
                            show_progress_bar=False, gc_after_trial=True)
            best_value = float(study.best_value)

            best_params = study.best_params
            best_mask = mand_mask.copy()
            for i, col in enumerate(kept_cols):
                if not mand_mask[i]:
                    best_mask[i] = best_params.get(f"use_{col}", False)
            best_features = [kept_cols[i] for i in range(n_kept) if best_mask[i]]
            n_selected = int(best_mask.sum())
        except Exception as e:
            log.warning(f"    [{model_name}] Stage 2 Optuna 실패: {e}")
            best_value = float("inf")
            best_features = kept_cols   # fallback: 전체
            best_params = {}
            n_selected = n_kept

        result = {
            "model": model_name,
            "best_score_oof_wis": best_value,
            "best_feature_subset": best_features,
            "best_params": {k: v for k, v in best_params.items() if k.startswith("use_")},
            "n_features_pool_after_drop": n_kept,
            "n_mandatory": n_mandatory,
            "n_selected": n_selected,
            "n_total_original": len(feature_cols),
            "trials_completed": n_trials_per_model,
            "saved_at": datetime.utcnow().isoformat() + "Z",
        }
        out_path = output_dir / f"{model_name}.json"
        out_path.write_text(json.dumps(result, indent=2, default=str))
        log.info(f"    [{model_name}] best_oof_wis={best_value:.3f}, "
                 f"selected={n_selected}/{n_kept}")
        results[model_name] = result

    summary = {
        "n_models": len(model_names),
        "models_processed": list(results.keys()),
        "n_trials_per_model": n_trials_per_model,
        "saved_at": datetime.utcnow().isoformat() + "Z",
    }
    (output_dir / "_summary.json").write_text(json.dumps(summary, indent=2))
    return results


def load_stage2_features(model_name: str,
                          input_dir: Optional[Path] = None
                          ) -> Optional[dict]:
    """Stage 2 결과 file load (R9 per_model_optimize 가 사용).

    Origin: phase3_feature_optuna.load_phase0b_features (L248-255).
    input_dir=None → get_results_dir()/stage2_feature_optuna (writer 와 동일 base, MPH_OUTPUT_ROOT 존중).
    """
    if input_dir is None:
        input_dir = get_results_dir() / "stage2_feature_optuna"
    p = input_dir / f"{model_name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


# Backward-compat aliases (phase0b file 이 호출하던 이름)
run_phase3_feature_optuna = _stage2_feature_optuna_inline
load_phase0b_features = load_stage2_features


# ────────────────────────────────────────────────────────────────────────────
# Stage 2 adapter — single-model array-based (B4f, 2026-05-28)
# ────────────────────────────────────────────────────────────────────────────
# 의도 (사용자 명시 design A B4f): _stage2_feature_optuna_inline 의 signature
# (phase1: dict, model_names: list) 가 research mode 의 single-model array input
# 과 불일치. adapter 함수로 wrap.

def _stage2_feature_optuna_per_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_cols: list[str],
    n_trials: int = 20,
    output_dir: Optional[Path] = None,
) -> tuple[Optional[list[int]], dict]:
    """Stage 2 adapter for research mode (single model, array input).

    Wraps _stage2_feature_optuna_inline by constructing phase1-like dict.

    Args:
        model_name: 모델 이름
        X_train/y_train/X_val/y_val: train + val arrays
        feature_cols: feature 이름 list (len = X.shape[1])
        n_trials: Optuna trial 수 (default 20, 사용자 명시 budget)
        output_dir: 결과 file dir (default = per_model_research_features/)

    Returns:
        (feature_indices, result_dict)
        - feature_indices: best feature subset indices (None on failure)
        - result_dict: stage2 metadata (best_feature_subset, best_score_oof_wis, n_selected, ...)

    Performance: ~30-60s (20 trial × 3-fold proxy WF-CV).
    """
    if output_dir is None:
        output_dir = get_results_dir() / "per_model_research_features"

    # Concatenate train + val for Stage 2 pool (feature optuna needs full pool)
    X_pool = np.vstack([X_train, X_val])
    y_pool = np.concatenate([y_train, y_val])

    # fake phase1 dict (adapter contract)
    fake_phase1 = {
        "X_all": X_pool,
        "y_all": y_pool,
        "feature_cols": list(feature_cols) if feature_cols else [],
        "n_train": len(X_train),
        "pool_end": len(X_pool),
    }

    try:
        results = _stage2_feature_optuna_inline(
            phase1=fake_phase1,
            model_names=[model_name],
            n_trials_per_model=n_trials,
            output_dir=output_dir,
        )
    except Exception as e:
        log.warning(f"  [stage2_adapter] {model_name} failed: {e}")
        return None, {"error": str(e)}

    result = results.get(model_name, {})
    best_features = result.get("best_feature_subset", [])
    if not best_features or not feature_cols:
        return None, result

    # feature_subset (names) → feature_indices
    feat_set = set(best_features)
    feat_indices = [i for i, c in enumerate(feature_cols) if c in feat_set]
    return feat_indices, result


def _oof_cv_wis_hier(
    factory_fn,
    X_train: np.ndarray,
    y_train: np.ndarray,
    frozen_params: dict,
    *,
    feature_indices=None,
    feature_cols=None,
    n_folds: int = 5,
    max_chain_length: int = 2,
    extrapolation_safe: bool = False,   # G-256: linear/NN/GAM → linear-inverse y only
    return_folds: bool = False,         # G-294: (agg, per_fold_list) for nested 1-SE size-path guard
    force_y_identity: bool = False,     # G-300: model owns y-transform → phase-13 y = identity
    force_x_identity: bool = False,     # G-301: USES_FEATURES=False → phase-13 x = none (no waste)
    restrict_centered_y: bool = False,  # G-303: in-model 0-floor → exclude laplace/mcmc_robust
    optuna_trial=None,                  # Stage-1 preproc pruner: fold-level report/should_prune
) -> float:
    """Walk-forward OOF mean WIS that REPLAYS a frozen hierarchical preproc per fold (D4 fix).

    ``return_folds=True`` → returns ``(agg_wis, per_fold_wis_list)`` (mirrors flat
    ``_oof_cv_wis``) so the Stage-2 feature guard's nested size-path can apply its
    1-SE/parsimony rule; default ``False`` keeps the scalar return for existing callers.

    The Stage-1 preproc objective previously scored a FIXED identity/robust OOF, so it was
    blind to the trial's sampled transform/scaler → every trial tied → arbitrary (trial-0)
    preproc selection. This re-applies the SAME sampled hierarchical preproc — via an optuna
    ``FixedTrial`` over the trial's ``frozen_params`` (verified to replay deterministically)
    — INSIDE each walk-forward fold, re-fit on that fold's train only (no leakage). So OOF
    now varies with the preproc and the objective can distinguish configs.

    Args:
        factory_fn: callable() -> forecaster.
        X_train, y_train: chronological train pool.
        frozen_params: the Optuna ``trial.params`` defining the sampled hierarchical preproc.
        feature_indices / feature_cols: per-model feature subset passthrough.
        n_folds: walk-forward folds. max_chain_length: hierarchical chain cap.

    Returns:
        mean OOF WIS over folds, or float('inf') if data too small / all folds failed.

    Side effects: none. No test/holdout data touched (train-pool walk-forward only).
    """
    import optuna as _opt
    from simulation.pipeline.per_model_optimize import (
        _evaluate_config_hierarchical, _fold_variance_penalize, _oof_selection_score,
    )

    n = len(X_train)
    if n < (n_folds + 1) * 10:
        return (float("inf"), []) if return_folds else float("inf")
    fold_size = n // (n_folds + 1)
    # G-334 (2026-06-22): fold-불변 inverse-cap 기준 = 전체 train max. 모든 fold 가 동일 cap 을 써서
    #   작은 fold 의 y_max 로 asinh 등 상위 PI 가 잘려 OOF 가 부풀던 회귀(1.901→4.344) 제거.
    try:
        from simulation.pipeline.preproc_optuna_hierarchical import set_y_ref_max as _set_yrm
        _set_yrm(float(np.max(np.asarray(y_train, dtype=np.float64))))
    except Exception:
        pass
    scores: list[float] = []
    fold_maxes: list[float] = []   # G-256b: per-fold val max for regime-conditional aggregation
    outbreak_level = float(np.percentile(np.asarray(y_train, dtype=float).ravel(), 75))
    prev_resid = None     # Q1 (2026-05-30): prior-fold OOS residuals → this fold's PI calibration
    # G-309 (3자 감사 #4): fold-local recode of global-summary features (quantile/threshold/
    #   interaction) — phase-13 의 MAIN selection(stage-1 preproc·stage-2 feature 가 이 hier OOF 사용)
    #   도 build-time GLOBAL(test+real era) 코딩 누수 미사용. lazy import = per_model_optimize ↔ _inline
    #   순환 회피. 해당 컬럼 부재 시 no-op (동작 불변).
    from simulation.pipeline.per_model_optimize import _recode_advanced_per_fold as _rcd
    for k in range(1, n_folds + 1):
        end_tr = k * fold_size
        end_va = (k + 1) * fold_size if k < n_folds else n
        _Xf = _rcd(X_train, y_train, feature_cols, end_tr)
        X_tr, y_tr = _Xf[:end_tr], y_train[:end_tr]
        X_va, y_va = _Xf[end_tr:end_va], y_train[end_tr:end_va]
        if len(X_va) < 4:
            continue
        try:
            fixed = _opt.trial.FixedTrial(dict(frozen_params))
            cell = _evaluate_config_hierarchical(
                factory_fn, X_tr, y_tr, X_va, y_va, optuna_trial=fixed,
                feature_indices=feature_indices,
                sigma_for_wis=max(float(np.std(y_tr)), 1e-3),
                feature_cols=feature_cols, max_chain_length=max_chain_length,
                calib_residuals=prev_resid,   # Q1: leakage-free (fold 0 → in-sample fallback)
                extrapolation_safe=extrapolation_safe,
                force_y_identity=force_y_identity,
                force_x_identity=force_x_identity,
                restrict_centered_y=restrict_centered_y,
            )
        except Exception:
            continue
        if "error" not in cell and np.isfinite(cell.get("wis", float("inf"))):
            scores.append(_oof_selection_score(cell))
            fold_maxes.append(float(np.asarray(y_va, dtype=float).max()))
        _vr = cell.get("_val_residuals")
        if _vr is not None and np.size(_vr) >= 2:
            prev_resid = np.asarray(_vr, dtype=float).ravel()   # calibrate the NEXT fold
        if optuna_trial is not None and scores:
            interim = _aggregate_oof_folds(scores, fold_maxes, outbreak_level)
            optuna_trial.report(float(interim), step=int(k))
            if optuna_trial.should_prune():
                raise _opt.TrialPruned()
    # G-255/G-256b (2026-06-12): regime-balanced mean over folds. Median discarded the outbreak
    # folds (~2/5 contain a peak) → peak-blind configs won → collapse on the test peak. Mean
    # counts every fold; regime split (quiet vs elevated) keeps the few outbreak folds from
    # being drowned out AND keeps the many quiet folds from being ignored. See helper.
    _agg = _fold_variance_penalize(_aggregate_oof_folds(scores, fold_maxes, outbreak_level),
                                   scores)
    return (_agg, list(scores)) if return_folds else _agg


def _make_preproc_plateau_stop(patience: int) -> Callable:
    """Optuna study 콜백: best 가 `patience` trial 연속 무개선이면 study.stop().

    배경: preproc study 는 pruner 를 배정하지만 ``_preproc_objective`` 가 fold 별
    ``trial.report()``/``should_prune()`` 를 호출하지 않아 pruner 가 inert
    (대조: tree_models.py:112-116 은 정확히 호출). 그 결과 best oof_wis 가 trial 1-5 에
    plateau 해도 n_trials 전부 소진. 본 콜백이 study-level 적응형 조기종료를 복원한다.

    direction=minimize → "개선" = objective 가 엄격히 더 낮아짐. best trial 은 항상
    완료된 뒤 멈추므로 늦은-plateau 챔피언층(TabPFN·NegBinGLM·ElasticNet·XGBoost)은
    끝까지 가고, 조기-plateau 모델(FluSight·EARS·KRR·N-BEATS)만 일찍 종료 → 품질 무손상.

    Args:
        patience: 무개선 COMPLETE trial 연속 수. 이 수에 도달하면 study.stop().
    Returns:
        ``callback(study, trial)`` — ``study.optimize(callbacks=[...])`` 용.
    Performance: O(1)/trial. Side effects: 임계 도달 시 study.stop() 호출.
    Caller responsibility: study direction 이 minimize 여야 함(preproc=minimize).
    """
    state = {"best": float("inf"), "since": 0}

    def _cb(study, trial) -> None:
        v = getattr(trial, "value", None)
        if v is None:                       # 실패/pruned trial → 무개선으로도 안 셈
            return
        if float(v) < state["best"] - 1e-12:
            state["best"] = float(v)
            state["since"] = 0
        else:
            state["since"] += 1
            if state["since"] >= patience:
                study.stop()

    return _cb


# ────────────────────────────────────────────────────────────────────────────
# Stage 1 — Preproc Optuna (R9 per_model_optimize logic 이동, 2026-05-28)
# ────────────────────────────────────────────────────────────────────────────
# Origin: simulation/pipeline/phase13_per_model_optimize.py:1040-1115 (75 LOC).
# 의도 (사용자 명시 2026-05-28 design A B1): R9/R10 공유 module 안에 logic.

def _pick_masked_best_preproc(trial_results, best_by):
    """G-308 (3자 감사 #2, 2026-06-18): error-trial 을 MASKING 하며 best Stage-1 preproc trial 을
    고르고, preproc_optuna_params 를 **선택된 trial 자신의 params** 로 바인딩한다.

    버그: ``study.best_params`` 는 Optuna 내부 best(objective 반환 ``oof_wis`` 기준)다. error trial
    (``_evaluate_config`` 실패 → 'error' 키)이라도 독립 ``_oof_cv_wis_hier`` 가 유한 oof 를 얻으면
    Optuna best 로 지목될 수 있다. 아래 argmin 은 그런 error trial 을 inf 로 마스킹하므로 ``best`` 는
    유효하지만, ``study.best_params`` 를 바인딩하면 refit 에서 그 ERROR trial 의 preproc 를 replay
    (→ N-HiTS/TiDE refit-null). 마스킹된 best 의 자기 trial_params 를 써야 정합.

    Args:
        trial_results: per-trial cell 리스트. 각 cell 은 ``oof_wis``/``wis``, 선택적 'error',
            그리고 ``trial_params`` (Optuna ``trial.params`` 스냅샷) 를 가진다.
        best_by: "oof_cv" → oof_wis 기준; 그 외 → wis.

    Returns:
        (best_cell, best_idx). best_cell["preproc_optuna_params"] 가 자기 trial_params 로 설정됨.
    """
    _sk = "oof_wis" if best_by == "oof_cv" else "wis"
    _scores = [(c.get(_sk, float("inf")) if "error" not in c else float("inf"))
               for c in trial_results]
    best_idx = int(np.argmin(_scores))
    # G-329g (2026-06-20, 3AI 최종 preproc ②): identity-margin 게이트. transform/scaler 가
    #   identity(y_mode=none) + no-scale(x_mode=none) baseline 을 relative margin(MPH_PREPROC_MARGIN,
    #   default 2%) 이상 genuine 하게 이겨야만 채택; 아니면 identity 유지. identity-anchor(G-329c, trial-0)
    #   와 짝 — baseline 을 노이즈 마진으로 이긴 transform/grouped-scaler 채택(=OOF 과적합·X grouped
    #   과소예측) 차단. mc(MPH_MC_MARGIN=0.02)와 동일 철학. 누수 0(OOF only). forced none 은 키 부재 →
    #   default "none" 로 baseline 판정.
    def _is_id(_c):
        _p = _c.get("trial_params") or {}
        return (_p.get("y_mode", "none") == "none"
                and _p.get("x_mode", "none") == "none")
    _margin = float(os.environ.get("MPH_PREPROC_MARGIN", "0.02"))
    # 진단 전용 노브(default off): MPH_PREPROC_FORCE_ARGMIN=1 → identity-revert 게이트 우회,
    #   순수 argmin(OOF-최저) 채택. 1-SE 가 보수적으로 identity 로 되돌린 transform 이 hold-out
    #   test 에서 실제로 더 나은지(=1-SE 비용) A/B 측정용. 학습 default 동작은 1-SE 그대로(=off).
    _force_argmin = os.environ.get("MPH_PREPROC_FORCE_ARGMIN", "0").strip() == "1"
    _id_idx = next((i for i, c in enumerate(trial_results)
                    if "error" not in c and _is_id(c) and np.isfinite(_scores[i])), None)
    if (not _force_argmin and _id_idx is not None and _id_idx != best_idx
            and np.isfinite(_scores[best_idx])):
        # G-333 (2026-06-22, flat-grid): 고정 2% margin → fold-paired 1-SE. 비-identity transform 은
        #   OOF 가 identity 를 1 fold-SE 이상 유의하게 이길 때만 채택 → 노이즈급 우세 차단(DNN 12.28 vs
        #   12.95 = fold 노이즈 내 → identity 유지 → ~0.9 회복; TabPFN 진짜 큰 마진이면 transform 유지).
        #   fold 벡터(oof_wis_folds) 부재(legacy/TPE 경로) 시 옛 2% relative margin fallback.
        _id_f = trial_results[_id_idx].get("oof_wis_folds")
        _bt_f = trial_results[best_idx].get("oof_wis_folds")
        if (isinstance(_id_f, (list, tuple)) and isinstance(_bt_f, (list, tuple))
                and len(_id_f) == len(_bt_f) and len(_id_f) >= 2):
            _d = np.asarray(_id_f, dtype=np.float64) - np.asarray(_bt_f, dtype=np.float64)
            _se = float(np.std(_d, ddof=1) / np.sqrt(len(_d)))
            if not (float(np.mean(_d)) > _se):   # best 가 1-SE 이상 유의하게 못 이김
                best_idx = _id_idx               # → identity 유지 (노이즈 동률)
        elif _scores[best_idx] >= _scores[_id_idx] * (1.0 - _margin):
            best_idx = _id_idx                   # fold 부재 → 옛 2% margin fallback
    # 진단 전용(default 미설정=무영향): MPH_FORCE_Y_INDIVIDUAL=<name> → 해당 y-transform trial
    #   강제 선택(OOF/1-SE 무관). 과거 특정 transform(예: 0.927 의 asinh) 재현·디버깅용.
    _force_yi = os.environ.get("MPH_FORCE_Y_INDIVIDUAL", "").strip()
    if _force_yi:
        for _i, _c in enumerate(trial_results):
            if "error" in _c:
                continue
            _p = _c.get("trial_params") or {}
            if _p.get("y_mode") == "individual" and _p.get("y_individual") == _force_yi:
                best_idx = _i
                break
    best = trial_results[best_idx]
    best["preproc_optuna_params"] = best.get("trial_params") or {}
    return best, best_idx


def _seed_y_transform_trials(
    study: Any,
    model_name: str,
    *,
    force_y_identity: bool,
    force_x_identity: bool,
    restrict_centered: bool,
    n_trials: int,
    grid_mode: bool = False,
) -> int:
    """Enqueue one startup trial per y-transform (+ identity anchor) before TPE.

    transform-fix (2026-06-21) PART D — EXTENDS the G-329c identity anchor: with the internal
    y-transforms un-hardcoded (PART A) and the preproc search enabled for those models (PART C),
    the single y-transform is now selected by the data-driven preproc OOF. To stop a transform
    being missed by the TPE random-startup lottery (G-280's exact failure mode for log1p), we seed
    ONE individual trial per stable y-transform up front. TPE then exploits from a complete prior.

    Args:
        study: an optuna study (or any object with ``enqueue_trial(params, skip_if_exists=)``).
        model_name: registry model name (only used for log context).
        force_y_identity: if True, the model owns its y-transform → seed ONLY the identity anchor.
        force_x_identity: if True, the model ignores X → omit the x_mode key (already forced none).
        restrict_centered: if True (transformed-zero-floor model), exclude the 2 median-centered
            transforms (laplace/mcmc_robust) so the in-model 0-floor stays correct (G-303).
        n_trials: stage-1 budget. Seeds are capped to leave ≥1 trial for TPE exploitation.

    Returns:
        The number of trials successfully enqueued.

    Performance: O(len(STABLE_Y_TRANSFORMS)) enqueue calls (≤6), negligible.
    Side effects: enqueues trials into ``study`` (Optuna persistence). Never raises — each enqueue
        is wrapped in try/except so a broken study cannot break stage-1 (caller relies on this).
    Caller responsibility: pass the same force_*/restrict flags used to build the objective so the
        seeded params replay validly (FixedTrial picks the frozen choice from the offered set).
    """
    from simulation.pipeline.preproc_optuna_hierarchical import (
        STABLE_Y_TRANSFORMS as _STABLE_Y_FOR_SEED,
        _NONCENTERED_STABLE_Y,
    )

    enqueued = 0

    def _enq(params: dict) -> None:
        nonlocal enqueued
        try:
            study.enqueue_trial(params, skip_if_exists=True)
            enqueued += 1
        except Exception:
            pass

    # ① identity anchor (G-329c) — always.
    _anchor: dict = {}
    if not force_y_identity:
        _anchor["y_mode"] = "none"
    # G-346 (2026-06-25): force_x_identity 여도 x_mode='none' 를 항상 기록. 옛 가드(if not force_x_identity)는
    #   x_mode 를 omit 했고, 그게 영속 preproc_optuna_params 로 들어가 refit FixedTrial replay 가 'x_mode' 키를
    #   못 찾아 HIER_FAIL→oof_wis=inf (foundation TiRex/TimesFM-2.5/DLinear). 'none' = force_x_identity
    #   early-return(preproc_optuna_hierarchical:885-887)의 passthrough 와 동일값이라 정합. y_mode 와 대칭.
    _anchor["x_mode"] = "none"
    if _anchor:
        _enq(_anchor)

    # ② one individual trial per y-transform — only when y is NOT model-owned. Budget-capped:
    #    seeds (anchor + per-transform) must leave ≥1 trial for TPE, so cap the transform count.
    if not force_y_identity:
        pool = list(_NONCENTERED_STABLE_Y if restrict_centered else _STABLE_Y_FOR_SEED)
        # G-333 (2026-06-22, flat-grid): grid_mode → seed EVERY transform (caller sets optimize
        #   n_trials=enqueued → pure grid, sampler never fires). legacy → leave ≥1 for TPE.
        budget = len(pool) if grid_mode else max(0, n_trials - enqueued - 1)
        for t in pool[:budget]:
            seed = {"y_mode": "individual", "y_individual": t, "x_mode": "none"}  # G-346: x_mode 항상 포함
            _enq(seed)

    return enqueued


def _stage1_preproc_optuna_inline(
    model_name: str,
    factory_fn: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_indices: Optional[list[int]] = None,
    feature_cols: Optional[list[str]] = None,
    sigma: float = 1.0,
    n_trials: int = 100,
    best_by: str = "oof_cv",
) -> tuple[dict, list]:
    """Stage 1 preproc Optuna (transform × scaler hierarchical search).

    Origin: phase13_per_model_optimize.py:1040-1115 (2026-05-28 logic 이동).

    Args:
        model_name: 모델 이름
        factory_fn: 모델 factory (returns BaseForecaster)
        X_train/y_train/X_val/y_val: train + val arrays
        feature_indices: fixed feature subset (Stage 2 결과 or None)
        feature_cols: feature 이름 list
        sigma: WIS scoring sigma (= std(y_train))
        n_trials: Optuna trials (default 100, MPH_PREPROC_TRIALS env override)
        best_by: "oof_cv" (default G-132) or "val"

    Returns:
        (best_cell, trial_results) — best dict {transform, scaler, wis, oof_wis, ...}
        + 모든 trial 결과 list.

    Performance: O(n_trials × model_fit_cost). ~100 trial × 30-60s = ~50-100 min per model.
    Side effects: Optuna study DB 영속 (warm-start), trial cleanup callback (G-161).
    """
    import optuna as _opt_lib

    # R9 per_model_optimize 의 helpers import (logic 의존성)
    from simulation.pipeline.per_model_optimize import (
        _evaluate_config_hierarchical,
    )
    from simulation.models._optuna_pruners import get_best_pruner_for as _pc_get_pruner

    import os as _os
    from simulation.pipeline.preproc_optuna_hierarchical import (
        model_needs_linear_inverse_y, model_applies_internal_y_transform,
        model_floors_at_transformed_zero,
    )
    # G-256/G-256c: the hard pool restriction is OPT-IN (default OFF). The full pool — including
    # identity — stays available, and the G-256c sanity penalty in the OOF objective makes Optuna
    # avoid blow-ups on its own (user-preferred). Set MPH_FORCE_LINEAR_INVERSE_Y=1 to additionally
    # hard-restrict extrapolating families as a belt-and-suspenders backstop.
    # 2026-06-15 (전수 A-영향 확인): 글로벌 MPH_FORCE_LINEAR_INVERSE_Y=1 은 restrict 가족 전체
    #   (foundation/linear 포함)에 적용 → 비선형변환으로 양수를 낸 챔피언 TabPFN(0.927)·NegBinGLM·
    #   ElasticNet 까지 흔듦(전수표 확인). per-model allow-list(MPH_LINEAR_INVERSE_MODELS="DNN,TCN,...")로
    #   음수-폭발 모델에만 한정 → 챔피언·working 무손상. 글로벌 플래그도 보존(belt-and-suspenders).
    _lin_list = {s.strip() for s in
                 _os.environ.get("MPH_LINEAR_INVERSE_MODELS", "").split(",") if s.strip()}
    _extrap_safe = (model_needs_linear_inverse_y(model_name)
                    and (_os.environ.get("MPH_FORCE_LINEAR_INVERSE_Y", "0") == "1"
                         or model_name in _lin_list))
    # G-300: models with an intrinsic y-transform (log-link GLMs / pf log1p) get RAW y from
    #   phase-13 (y_mode forced "none") to avoid a double y-transform. Recorded in frozen params.
    _force_y_identity = model_applies_internal_y_transform(model_name)
    # G-301: USES_FEATURES=False models (TimesFM-2.5/TiRex) ignore X → force x_mode="none" to skip
    #   the wasted x-scaler search (every x_mode gives identical output). y-search unaffected.
    try:
        from simulation.models.base import REGISTRY as _REG_xf
        _cls_xf = _REG_xf.get(model_name)
        _force_x_identity = (_cls_xf is not None and not getattr(_cls_xf, "USES_FEATURES", True))
    except Exception:
        _force_x_identity = False
    # G-303: models with an in-model transformed-zero floor (SVR/ElasticNet/KRR/BayesianRidge)
    #   exclude median-centered y-transforms (laplace/mcmc_robust) so the floor stays correct.
    _restrict_centered = model_floors_at_transformed_zero(model_name)
    # G-333 (2026-06-22, flat-grid 재설계): pure-grid preproc. seed 가 {identity + STABLE_Y} 전부
    #   enqueue + optimize n_trials=enqueued → Optuna sampler 미발동 = 공정 grid(각 transform 1회).
    #   DNN 38×HIER_individual vs 3×identity 같은 TPE-exploitation coverage skew 제거. 선택은 1-SE
    #   (identity 기본). MPH_PREPROC_GRID=0 → legacy TPE Optuna 복귀(fast/탐색 모드).
    _grid_mode = os.environ.get("MPH_PREPROC_GRID", "1").strip() == "1"

    _opt_lib.logging.set_verbosity(_opt_lib.logging.WARNING)
    log.info(f"  [stage1_preproc] {model_name} n_trials={n_trials} best_by={best_by}"
             f"{' [extrapolation_safe y]' if _extrap_safe else ''}"
             f"{' [y=identity: internal transform]' if _force_y_identity else ''}")

    trial_results: list = []
    # transform-fix (2026-06-21, PART F): count of enqueued startup SEED trials (identity anchor +
    #   one per stable y-transform). Seed trials are EXEMPT from fold-level pruning (below) so every
    #   y-transform gets a FULL fair OOF evaluation. Diagnosis (codex+workflow): asinh/laplace/
    #   mcmc_robust were pruned mid-fold in ~1/3 of models → their best transform never competed
    #   (e.g. TabPFN asinh R²0.927 lost to a defaulted identity 0.892). Set after the seed call below;
    #   read by the objective closure at exec time (objective only reads → no nonlocal needed).
    _n_seeded = 0

    def _preproc_objective(trial):
        cell_local = _evaluate_config_hierarchical(
            factory_fn, X_train, y_train, X_val, y_val,
            optuna_trial=trial,
            feature_indices=feature_indices, sigma_for_wis=sigma,
            feature_cols=feature_cols,
            max_chain_length=GLOBAL.training.hier_max_chain,
            extrapolation_safe=_extrap_safe,
            force_y_identity=_force_y_identity,
            force_x_identity=_force_x_identity,
            restrict_centered_y=_restrict_centered,
        )
        # B4a (2026-05-28): research_5fold mode = stricter 5-fold WF-CV (paper-grade)
        # 2026-05-31 (사용자 명시, 학위논문 제출용): oof_cv fold 수 = GLOBAL.training.oof_folds
        # (기본 5; MPH_OOF_FOLDS / --oof-folds). research_5fold 은 명시 5 유지.
        _n_folds = 5 if best_by == "research_5fold" else GLOBAL.training.oof_folds
        if best_by in ("oof_cv", "research_5fold"):
            try:
                # D4 fix (2026-05-30): replay the SAMPLED hierarchical preproc per fold so the
                # OOF reflects it. Was _oof_cv_wis(transform="identity", scaler="robust") =
                # FIXED → preproc-blind → all trials tied → arbitrary (trial-0) selection.
                _oof_agg, _oof_folds = _oof_cv_wis_hier(
                    factory_fn, X_train, y_train, trial.params,
                    feature_indices=feature_indices, feature_cols=feature_cols,
                    n_folds=_n_folds, max_chain_length=GLOBAL.training.hier_max_chain,
                    extrapolation_safe=_extrap_safe,
                    force_y_identity=_force_y_identity,
                    force_x_identity=_force_x_identity,
                    restrict_centered_y=_restrict_centered,
                    # transform-fix (2026-06-21, PART F): seed trials (trial.number < _n_seeded) skip
                    #   fold-level pruning → each y-transform gets a full fair OOF. TPE trials prune as
                    #   usual. _oof_cv_wis_hier only report()/should_prune() when optuna_trial is not
                    #   None (its L500-504), so None = no prune for the enqueued transform seeds.
                    optuna_trial=(None if trial.number < _n_seeded else trial),
                    return_folds=True,   # G-333: per-fold WIS → _pick_masked_best_preproc 의 1-SE 비교
                )
                cell_local["oof_wis"] = _oof_agg
                cell_local["oof_wis_folds"] = list(_oof_folds) if _oof_folds else None
            except _opt_lib.TrialPruned:
                raise
            except Exception:
                cell_local["oof_wis"] = float("inf")
        cell_local["trial_params"] = dict(trial.params)   # G-308 (감사#2): masked-best replay 정합
        trial_results.append(cell_local)
        score_key = "oof_wis" if best_by in ("oof_cv", "research_5fold") else "wis"
        return cell_local.get(score_key, cell_local["wis"])

    # ─────────────────────────────────────────────────────────────────────────
    # G-335 (2026-06-22, 사용자 KISS): flat-grid 은 고정 enumeration 이라 Optuna(study/sampler/pruner)
    #   가 불필요한 오버헤드 — grid_mode 에선 study 없이 **순수 loop**. {identity + STABLE_Y} config 는
    #   _seed_y_transform_trials(grid_mode=True) 의 enqueue 를 mock collector 로 가로채 그대로 빌드
    #   (파라미터 포맷 단일소스 = drift 0). 각 config = _evaluate_config_hierarchical(val) +
    #   _oof_cv_wis_hier(OOF, optuna_trial=None=no prune). 기존 Optuna-grid 와 **결과 동일**(같은 OOF
    #   함수 호출) + sampler-seed·pruning·비결정성 제거 + 투명. legacy TPE 는 아래 else(MPH_PREPROC_GRID=0).
    if _grid_mode:
        try:
            class _GridCollector:
                def __init__(self):
                    self.cfgs: list = []

                def enqueue_trial(self, params, skip_if_exists=True):
                    self.cfgs.append(dict(params))

            _gc = _GridCollector()
            _seed_y_transform_trials(
                _gc, model_name,
                force_y_identity=_force_y_identity, force_x_identity=_force_x_identity,
                restrict_centered=_restrict_centered, n_trials=n_trials, grid_mode=True)
            _grid_cfgs = _gc.cfgs or [{}]   # force_y AND force_x → 빈 dict → identity 1개(안전)
            from optuna.trial import FixedTrial as _FixedT
            _n_folds_g = 5 if best_by == "research_5fold" else GLOBAL.training.oof_folds
            for _gp in _grid_cfgs:
                cell_g = _evaluate_config_hierarchical(
                    factory_fn, X_train, y_train, X_val, y_val,
                    optuna_trial=_FixedT(dict(_gp)),
                    feature_indices=feature_indices, sigma_for_wis=sigma,
                    feature_cols=feature_cols,
                    max_chain_length=GLOBAL.training.hier_max_chain,
                    extrapolation_safe=_extrap_safe, force_y_identity=_force_y_identity,
                    force_x_identity=_force_x_identity, restrict_centered_y=_restrict_centered)
                if best_by in ("oof_cv", "research_5fold"):
                    try:
                        _agg_g, _folds_g = _oof_cv_wis_hier(
                            factory_fn, X_train, y_train, dict(_gp),
                            feature_indices=feature_indices, feature_cols=feature_cols,
                            n_folds=_n_folds_g, max_chain_length=GLOBAL.training.hier_max_chain,
                            extrapolation_safe=_extrap_safe, force_y_identity=_force_y_identity,
                            force_x_identity=_force_x_identity,
                            restrict_centered_y=_restrict_centered,
                            optuna_trial=None, return_folds=True)   # None = no fold-pruning (pure grid)
                        cell_g["oof_wis"] = _agg_g
                        cell_g["oof_wis_folds"] = list(_folds_g) if _folds_g else None
                    except Exception:
                        cell_g["oof_wis"] = float("inf")
                cell_g["trial_params"] = dict(_gp)
                trial_results.append(cell_g)
            best, best_idx = _pick_masked_best_preproc(trial_results, best_by)
            best["_extrap_safe"] = bool(_extrap_safe)
            best["scaler"] = best.get("scaler", "grouped_optuna")
            log.info(f"  [stage1_preproc] {model_name} PURE-GRID (no Optuna): "
                     f"{len(trial_results)} configs → transform={best.get('transform')} "
                     f"OOF_WIS={best.get('oof_wis', float('nan')):.3f}")
        except Exception as _ge:
            log.warning(f"  [stage1_preproc] {model_name} pure-grid fail → identity: {_ge}")
            best, trial_results = {"transform": "identity", "scaler": "none"}, []
        return best, trial_results

    # ── legacy TPE Optuna path (MPH_PREPROC_GRID=0, fast/탐색 모드) ──
    # G-280 (2026-06-16, 사용자): preproc n_startup 을 **비율-기반**으로. 사건: default 10 random +
    #   plateau-stop(25) → log1p 가 random 탐색서 거의 안 뽑혀(실측 53모델 중 1개만 선택) TPE 가
    #   exploit 못 함 = transform 커버리지 실패(TPE 수렴=절대개수 와 별개 문제). 비율로 올려 STABLE
    #   pool(transform×mode) 전체가 충분히 sampled 되게 보장. env MPH_OPTUNA_STARTUP_RATIO(0.4).
    _su_ratio = float(os.environ.get("MPH_OPTUNA_STARTUP_RATIO", "0.4"))
    _su_floor = int(os.environ.get("MPH_OPTUNA_STARTUP_MIN", "15"))
    _n_startup = max(_su_floor, int(_su_ratio * n_trials))
    _n_startup = min(_n_startup, max(1, n_trials - 1))   # TPE 에 ≥1 trial 남김
    log.info(f"  [stage1_preproc] {model_name} n_startup={_n_startup}/{n_trials} "
             f"(ratio {_su_ratio}) — transform 커버리지 보장")
    try:
        study = _opt_lib.create_study(
            study_name=f"stage1_preproc_{model_name}_{int(time.time())}",
            direction="minimize",
            sampler=_opt_lib.samplers.TPESampler(
                multivariate=True, group=True,
                n_startup_trials=_n_startup,   # G-280: 비율-기반 (커버리지)
                seed=42,                       # G-13F (2026-06-21, codex): R9 preproc 선택 재현성
                warn_independent_sampling=False,
            ),
            pruner=_pc_get_pruner(model_name, min_resource=1, max_resource=20),
        )
        # G-329c (2026-06-20): identity (no-transform, no-scale) enqueued as anchor → baseline always
        #   a candidate + TPE prior not biased to non-identity.
        # transform-fix (2026-06-21) PART D: EXTEND the single identity anchor to seed ONE startup
        #   trial per y-transform too (helper above), so the data-driven OOF evaluates every candidate
        #   transform instead of relying on the TPE random-startup lottery (G-280's log1p coverage
        #   failure). Budget-capped, skip_if_exists, try/except per enqueue — _force_y_identity models
        #   still get only the identity anchor. (force_*_identity keys omitted = already-forced none.)
        _n_seeded = _seed_y_transform_trials(   # PART F: capture seed count → prune-exempt seeds
            study, model_name,
            force_y_identity=_force_y_identity,
            force_x_identity=_force_x_identity,
            restrict_centered=_restrict_centered,
            n_trials=n_trials,
            grid_mode=_grid_mode,               # G-333: seed ALL transforms (pure grid)
        )
        # G-161: trial cleanup callback
        try:
            from simulation.models._optuna_torch import (
                make_trial_cleanup_callback as _mk_cleanup,
            )
            _callbacks = [_mk_cleanup(model_name)]
        except Exception:
            _callbacks = []
        # 2026-06-15 (3-LLM 검증 + 사용자 승인): preproc Optuna pruner 가 trial.report()
        # 부재로 inert(tree_models.py:112-116 엔 있음) → 조기-plateau 모델이 n_trials 낭비.
        # 적응형 plateau-stop 으로 best 무개선 K trial 후 종료(best trial 보존=챔피언 무손상,
        # 늦은-plateau 는 끝까지). MPH_PREPROC_PLATEAU_PATIENCE 튜닝(default 25; 0=비활성).
        # G-333: grid 모드는 seed 만 도므로(고정 개수) plateau-stop 불필요·유해(9번째 transform 잘림) → off.
        _pp_pat = 0 if _grid_mode else int(os.environ.get("MPH_PREPROC_PLATEAU_PATIENCE", "25"))
        if _pp_pat > 0:
            # G-329f (2026-06-20, 3AI feature/HP 워크플로 P-1): plateau-stop 발동 복원. 기존
            #   `max(_pp_pat, _n_startup+5)` 가 post-startup trial(n_trials−n_startup) < patience 를
            #   만들어 plateau.stop() 영구 미발동(60−30=30 < 35) = dead. 무손상 절감 dead. 수정: patience
            #   를 post-startup 보다 작게 cap → 조기-plateau 모델 종료 가능. startup 보호는 plateau 카운터가
            #   improvement 마다 reset 되어 random startup 중엔 사실상 안 멈춤(G-280 의도 유지). 늦은-
            #   plateau 챔피언은 best 갱신 지속 → 끝까지(영향 0).
            _pp_pat = max(8, min(_pp_pat, n_trials - _n_startup - 2))
            _callbacks.append(_make_preproc_plateau_stop(_pp_pat))
        # G-333: grid 모드 = enqueued seeds(=_n_seeded)만 실행 → Optuna sampler 미발동 = 순수 grid.
        #   ⚠ 방어(자체검증): force_y AND force_x 동시 모델은 anchor 도 빈dict → _n_seeded=0 →
        #   optimize(n_trials=0) crash. max(1,…) 로 최소 1 trial(objective 가 forced-identity 적용)=안전.
        _opt_n = max(1, _n_seeded) if _grid_mode else n_trials
        study.optimize(_preproc_objective, n_trials=_opt_n,
                       show_progress_bar=False,
                       gc_after_trial=True,
                       callbacks=_callbacks)
        # best cell 찾기
        # 2026-06-15 (per-model 감사): error 트리얼(_evaluate_config 가 wis=inf+'error' 반환)이
        #   독립 _oof_cv_wis_hier 에서 유한 oof_wis 를 얻어 argmin 으로 invalid 선발 → 그 preproc
        #   config 로 refit 시 N-HiTS/TiDE refit-null. line 474 패턴 미러 = error 트리얼 inf 마스킹.
        # G-308 (3자 감사 #2): masked best_idx 와 동일 trial 의 params 사용 (study.best_params 금지 —
        #   Optuna 내부 best 가 error trial 을 가리킬 수 있어 invalid preproc 를 refit replay = refit-null).
        best, best_idx = _pick_masked_best_preproc(trial_results, best_by)
        # G-294 (2026-06-17): the Stage-2 feature guard replays this preproc per fold via
        #   _oof_cv_wis_hier; it must match Stage-1's extrapolation-safe setting (single source).
        best["_extrap_safe"] = bool(_extrap_safe)
        log.info(f"  [stage1_preproc] {model_name} best: "
                  f"transform={best.get('transform')} val_WIS={best.get('wis'):.3f} "
                  + (f"OOF_WIS={best.get('oof_wis', float('nan')):.3f} "
                     if best_by == "oof_cv" else "")
                  + f"MAE={best.get('mae'):.3f}")
        best["scaler"] = best.get("scaler", "grouped_optuna")  # preserve actual HIER_x mode
    except Exception as _opte:
        log.warning(f"  [stage1_preproc] {model_name} Optuna fail → identity fallback: {_opte}")
        best = {"transform": "identity", "scaler": "none"}
        trial_results = []

    return best, trial_results


__all__ = [
    "run_3stage_optuna",
    "_stage1_preproc_optuna_inline",
    "_stage2_feature_optuna_inline",
    "load_stage2_features",
    "run_phase3_feature_optuna",     # backward-compat
    "load_phase0b_features",          # backward-compat
]


def _preproc_first_select(
    model_name: str,
    factory_fn: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    feature_cols: Optional[list[str]] = None,
    n_trials_preproc: int = 100,
    n_trials_feature: int = 20,
    best_by: str = "oof_cv",
) -> "tuple[dict, Optional[list[int]]]":
    """D1=b preproc-FIRST staged selection (user's order: preproc → feature).

    Stage 1 preproc Optuna on the FULL feature set → best model-appropriate preproc; then
    Stage 2 target-aware feature selection (inline) → subset. This is the user's preferred
    sequence and REPLACES the separate pre-stage feature load (so feature selection comes
    AFTER baseline/preproc, not as a pre-stage). Per the D2 empirical test the order is
    ~performance-neutral, so this is a conceptual/architecture alignment, not a perf change.

    Both sub-stages are train-pool only (no test/holdout leakage — _stage1 uses OOF-CV with
    the D4 per-fold preproc replay; _stage2 uses its own WF-CV WIS proxy).

    Args:
        model_name / factory_fn: model identity + builder.
        X_train/y_train/X_val/y_val: train-pool arrays.
        feature_cols: column names (Stage 2 needs them).
        n_trials_preproc / n_trials_feature: per-stage Optuna budgets.
        best_by: "oof_cv" (service) | "research_5fold" (research).

    Returns:
        (best_preproc_cell, feature_indices) — best_preproc_cell has transform/scaler/
        preproc_optuna_params; feature_indices is the Stage-2 subset (None ⇒ all columns).

    Side effects: Optuna studies (warm-start DB), Stage-2 JSON. No test data touched.
    """
    sigma = max(float(np.std(np.asarray(y_train).ravel())), 1e-3)
    best_preproc, _ = _stage1_preproc_optuna_inline(
        model_name=model_name, factory_fn=factory_fn,
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
        feature_indices=None,            # ← FULL features (preproc-first, not a loaded subset)
        feature_cols=feature_cols, sigma=sigma,
        n_trials=n_trials_preproc, best_by=best_by,
    )
    feat_indices, _meta = _stage2_feature_optuna_per_model(
        model_name=model_name,
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
        feature_cols=list(feature_cols) if feature_cols else [],
        n_trials=n_trials_feature,
    )
    return best_preproc, feat_indices


def run_3stage_optuna(
    model_name: str,
    factory_fn: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_cols: Optional[list[str]] = None,
    n_train: Optional[int] = None,
    mode: str = "service",                  # "service" (R9 per_model_optimize) | "research" (R10 per_model_eval)
    n_trials_preproc: int = 100,            # Stage 1 trials
    n_trials_feature: int = 20,             # Stage 2 trials
    n_trials_hp: int = 20,                  # Stage 3 trials (model internal)
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    X_real: Optional[np.ndarray] = None,
    y_real: Optional[np.ndarray] = None,
    feature_indices_init: Optional[list[int]] = None,
    viral_positivity_train: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> dict:
    """R9/R10 공유 3-stage Optuna entry.

    Args:
        model_name: 모델 이름 (registry key)
        factory_fn: 모델 factory 함수 (returns BaseForecaster instance)
        X_train/y_train/X_val/y_val: train + val arrays
        feature_cols: feature 이름 list (Stage 2 feature_subset 학습에 사용)
        n_train: training set 길이 (CV split 계산용)
        mode: "service" (R9 per_model_optimize, oof_cv val, real-time champion)
              | "research" (R10 per_model_eval, stricter CV, paper-grade)
        n_trials_preproc/feature/hp: 각 stage 의 Optuna trial 수
        X_test/y_test: held-out test slab (refit eval, mode="service" 필수)
        X_real/y_real: service-zone real slab (mode="service" 만)
        feature_indices_init: 초기 feature subset (default = all)
        viral_positivity_train: KDCA threshold input (audit S1.1 cascade)

    Returns:
        {
          "best_preproc": {transform, scaler, ...},
          "best_feature_indices": [...],
          "best_hp": {...},
          "val_metrics": {wis, mae, ...},
          "test_metrics": {wis, mae, ...} (mode="service" 만),
          "mode": "service" | "research",
          "elapsed_sec": float,
        }

    Raises:
        ValueError: shape mismatch / invalid mode / factory_fn 실패

    Performance: O(n_trials × WF_CV_cost). Mode="service" 기본 ~10-30 min per model.
                 Mode="research" stricter validation 으로 ~2x 시간.
    Side effects: 결과 파일 저장 (mode 별 다른 위치).
    Caller responsibility: X_train/y_train shape 일치, feature_cols length = X.shape[1].

    Status: SKELETON (2026-05-28 사용자 명시 design A 진행 중).
    TODO:
        - [ ] Stage 1 preproc Optuna 실제 구현 (R9 per_model_optimize _preproc_objective 복사 + module화)
        - [ ] Stage 2 feature Optuna 실제 구현 (phase0b run_phase3_feature_optuna 통합)
        - [ ] Stage 3 HP Optuna 호출 (factory_fn 안의 자체 Optuna 가 자동 처리)
        - [ ] mode="research" validation 분리 (5-fold CV, MCS membership 추가)
        - [ ] R9 / R10 의 호출 site 변경
    """
    if mode not in ("service", "research"):
        raise ValueError(f"mode must be 'service' or 'research', got {mode!r}")

    log.info(
        f"  [_inline_optuna_3stage] {model_name} mode={mode} "
        f"trials=(preproc={n_trials_preproc}, feature={n_trials_feature}, hp={n_trials_hp})"
    )

    if mode == "research":
        # 2026-05-28 B4: research mode — stricter validation for paper-grade analysis.
        # Stage 1+2+3 sequential, stricter CV (5-fold default, MCS membership 추가).
        # Note: 현재 skeleton wrapper — per_model_optimize.optimize_one_model delegate. 다음 sprint
        # 에 stricter validation (5-fold + MCS + bootstrap CI per fold) 구현.
        return _run_research_mode(
            model_name=model_name, factory_fn=factory_fn,
            X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
            feature_cols=feature_cols, n_train=n_train,
            n_trials_preproc=n_trials_preproc,
            n_trials_feature=n_trials_feature,
            n_trials_hp=n_trials_hp,
            X_test=X_test, y_test=y_test,
            feature_indices_init=feature_indices_init,
            viral_positivity_train=viral_positivity_train,
            **kwargs,
        )

    # mode == "service" — R9 per_model_optimize 의 real-time inference champion (현재 동작)
    return _run_service_mode(
        model_name=model_name, factory_fn=factory_fn,
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
        feature_cols=feature_cols, n_train=n_train,
        n_trials_preproc=n_trials_preproc,
        n_trials_feature=n_trials_feature,
        n_trials_hp=n_trials_hp,
        X_test=X_test, y_test=y_test,
        X_real=X_real, y_real=y_real,
        feature_indices_init=feature_indices_init,
        viral_positivity_train=viral_positivity_train,
        **kwargs,
    )


def _run_service_mode(
    model_name: str, factory_fn: Callable,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    **kwargs: Any,
) -> dict:
    """Service mode (R9 per_model_optimize): delegate to per_model_optimize.optimize_one_model.

    Real-time inference champion 선택. oof_cv WIS minimize. per_model_optimal/<MODEL>.json.
    """
    from simulation.pipeline.per_model_optimize import optimize_one_model

    result = optimize_one_model(
        model_name=model_name,
        factory_fn=factory_fn,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_indices=kwargs.get("feature_indices_init"),
        X_test=kwargs.get("X_test"),
        y_test=kwargs.get("y_test"),
        feature_cols=kwargs.get("feature_cols"),
        X_real=kwargs.get("X_real"),
        y_real=kwargs.get("y_real"),
        viral_positivity_train=kwargs.get("viral_positivity_train"),
    )
    if isinstance(result, dict):
        result["_inline_3stage_mode"] = "service"
        result["_inline_3stage_n_trials"] = {
            "preproc": kwargs.get("n_trials_preproc", 100),
            "feature": kwargs.get("n_trials_feature", 20),
            "hp": kwargs.get("n_trials_hp", 20),
        }
    return result


def _run_research_mode(
    model_name: str, factory_fn: Callable,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    feature_cols: Optional[list[str]] = None,
    n_train: Optional[int] = None,
    n_trials_preproc: int = 100,
    n_trials_feature: int = 20,
    n_trials_hp: int = 20,
    feature_indices_init: Optional[list[int]] = None,
    **kwargs: Any,
) -> dict:
    """Research mode (R10 per_model_eval): stricter validation for paper-grade analysis.

    설계 (원래 2026-05-28 design A → ★Stage1 G-335로 갱신 2026-06-22):
        Stage 1: preproc **pure-grid** (G-335: Optuna study/sampler/pruner 제거, 7-transform × 1 OOF + 1-SE;
                 원래 "Optuna n_trials=100 transform×scaler"는 superseded; legacy Optuna는 MPH_PREPROC_GRID=0만)
        Stage 2: feature STABILITY nested + 1-SE (dedicated per model)
        Stage 3: HP Optuna (model internal, automatic via factory_fn)

    Stricter validation (vs service mode):
        - 5-fold WF-CV (instead of 3-fold) — 다음 sprint B4-full
        - MCS_{90} membership 추가 — 다음 sprint
        - Bootstrap CI per fold — 다음 sprint
        - Conditional calibration check — 다음 sprint

    Output: simulation/results/per_model_research/<MODEL>.json
    (R9 per_model_optimize 의 per_model_optimal/ 와 별개 evaluation track)

    Status (2026-05-28 B4a 완료): Stage 1 stricter 5-fold WF-CV ✓.
        B4b/c/d/e/f = 진행 중 (Bootstrap CI per fold + MCS + Calibration + R10 per_model_eval wiring + Stage 2 adapter).
    """
    import json
    sigma = max(float(np.std(y_train)), 1e-3)
    research_result: dict = {
        "model": model_name,
        "_mode": "research",
        "_validation": "stricter (5-fold WF-CV)",
    }

    # ── Stage 1: preproc Optuna with 5-fold WF-CV (B4a 2026-05-28) ──
    try:
        log.info(
            f"  [research] {model_name} Stage 1 preproc Optuna "
            f"(n_trials={n_trials_preproc}, best_by=research_5fold)"
        )
        best_preproc, trial_results = _stage1_preproc_optuna_inline(
            model_name=model_name,
            factory_fn=factory_fn,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            feature_indices=feature_indices_init,
            feature_cols=feature_cols,
            sigma=sigma,
            n_trials=n_trials_preproc,
            best_by="research_5fold",   # NEW B4a
        )
        research_result["stage1_best_preproc"] = {
            "transform": best_preproc.get("transform"),
            "scaler": best_preproc.get("scaler"),
            "wis": best_preproc.get("wis"),
            "oof_wis_5fold": best_preproc.get("oof_wis"),
        }
        research_result["stage1_status"] = "completed (5-fold WF-CV)"
    except Exception as e:
        log.warning(f"  [research] {model_name} Stage 1 failed: {e}")
        research_result["stage1_error"] = str(e)
        best_preproc = {"transform": "identity", "scaler": "none"}
        trial_results = []

    # ── Stage 2: feature subset Optuna (B4f 2026-05-28 — adapter 완료) ──
    try:
        log.info(
            f"  [research] {model_name} Stage 2 feature Optuna "
            f"(n_trials={n_trials_feature}, per-model dedicated)"
        )
        feat_indices, stage2_meta = _stage2_feature_optuna_per_model(
            model_name=model_name,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            feature_cols=list(feature_cols) if feature_cols else [],
            n_trials=n_trials_feature,
        )
        research_result["stage2_feature_indices"] = feat_indices
        research_result["stage2_n_selected"] = stage2_meta.get("n_selected", 0)
        research_result["stage2_best_oof_wis"] = stage2_meta.get("best_score_oof_wis")
        research_result["stage2_status"] = "completed (Stage 2 adapter B4f)"
    except Exception as e:
        log.warning(f"  [research] {model_name} Stage 2 failed: {e}")
        research_result["stage2_error"] = str(e)
        research_result["stage2_feature_indices"] = feature_indices_init

    # ── Stage 3: HP Optuna (model internal, automatic via factory_fn) ──
    research_result["stage3_status"] = (
        f"model internal HP Optuna (factory_fn 자체, default={n_trials_hp} trial)"
    )

    # ── B4b: trial-level percentile CI (5-fold WF-CV sample variance) ──
    # Note: 진정한 Bootstrap CI per fold 는 refit + test prediction 필요 (B4e R10 per_model_eval wiring 후).
    # 현재는 Stage 1 trial_results 의 WIS distribution → percentile CI (≈ preproc search variance).
    try:
        _trial_wis = [
            float(c.get("wis"))
            for c in (trial_results or [])
            if isinstance(c, dict) and np.isfinite(c.get("wis", float("nan")))
        ]
        if len(_trial_wis) >= 10:
            research_result["b4b_trial_percentile_ci"] = {
                "ci_lo_2p5": float(np.percentile(_trial_wis, 2.5)),
                "ci_hi_97p5": float(np.percentile(_trial_wis, 97.5)),
                "median": float(np.median(_trial_wis)),
                "n_trials_valid": len(_trial_wis),
                "_method": "trial-level percentile (Stage 1 WIS sample, 5-fold WF-CV)",
            }
        else:
            research_result["b4b_trial_percentile_ci"] = {
                "_status": f"insufficient trials ({len(_trial_wis)} < 10)",
            }
    except Exception as e:
        research_result["b4b_error"] = str(e)

    # ── B4c: MCS_{90} membership (multi-model, R10 per_model_eval wiring 의존) ──
    # MCS 는 N model 의 loss_matrix 가 필요 — single model 에서 의미 X.
    # R10 per_model_eval 의 research mode hook (B4e) 가 53 model loss_matrix 구성 후 compute_mcs 호출.
    research_result["b4c_mcs_status"] = (
        "single-model run — MCS 는 R10 per_model_eval multi-model wiring (B4e) 시 활성. "
        "loss_matrix = stack(model 별 best per-trial WIS) → compute_mcs(alpha=0.10) → mcs_members."
    )

    # ── B4d: Conditional calibration (regime별 PIT, refit test 의존) ──
    # PIT (Probability Integral Transform) 는 test prediction 의 분포 가 uniform[0,1] 인지 검사.
    # regime (pre/during/post-COVID) 별 분리 — R6 dm_test 의 regime-split mask (pre/during/post/global) 활용.
    # 현재: best preproc 만 결정, refit test predictions 없음 → B4e wiring 후 활성.
    research_result["b4d_calibration_status"] = (
        "refit test 필요 — R10 per_model_eval wiring (B4e) 후 활성. "
        "conditional_calibration.compute_pit_by_regime(y_test, y_pred, regimes) 호출 예정."
    )

    # ── B4e: R10 per_model_eval wiring (pending) ──
    research_result["_pending_b4e"] = (
        "R10 run_per_model_eval 안에서 factory_fn registry 가져와서 "
        "각 model 별 run_3stage_optuna(mode='research') 호출. "
        "결과 → per_model_research/<MODEL>.json 누적 → MCS / Bootstrap CI 최종 계산."
    )

    research_result["_inline_3stage_n_trials"] = {
        "preproc": n_trials_preproc,
        "feature": n_trials_feature,
        "hp": n_trials_hp,
    }

    # 산출 저장 (MPH_OUTPUT_ROOT 존중 — get_results_dir, 2026-05-29)
    _out_dir = get_results_dir() / "per_model_research"
    _out_dir.mkdir(parents=True, exist_ok=True)
    (_out_dir / f"{model_name}.json").write_text(
        json.dumps(research_result, indent=2, default=str)
    )
    return research_result


__all__ = ["run_3stage_optuna"]
