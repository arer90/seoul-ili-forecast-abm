"""
R3: External Optuna 피처 선택
====================================
사전에 simulation/tools/run_optuna_feature_selection.py로 생성된 JSON 결과를 로딩하거나,
external 모드로 3-fold light CV Optuna를 실행.
"""
import logging
from simulation.config_global import GLOBAL  # SSOT (2026-05-28)
import json
import time
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List

log = logging.getLogger(__name__)


# Optuna 모델명 매핑 (기존 _OPTUNA_MODEL_MAP_INDIVIDUAL 호환)
_OPTUNA_MODEL_MAP = {
    "ElasticNet": "ElasticNet", "KRR": "KRR", "SVR-Linear": "SVR-Linear",
    "SVR-RBF": "SVR-RBF", "RandomForest": "RandomForest", "XGBoost": "XGBoost",
    "GradientBoosting": "GradientBoosting", "LightGBM": "LightGBM",
    "DNN": "DNN", "DNN-Optuna": "DNN", "TabularDNN-Lite": "DNN",
    "N-BEATS": "N-BEATS", "N-HiTS": "N-HiTS",
    "TCN": "TCN", "TCN-Optuna": "TCN", "TFT": "TFT",
    "PatchTST": "PatchTST", "iTransformer": "iTransformer",
    "Mamba": "Mamba", "TiDE": "TiDE", "TimesNet": "TimesNet",
    "DeepAR": "DeepAR", "RNN": "RNN",
    "GCN": "GCN", "GAT": "GAT",
    "GAM-Spline": "GAM", "GP-RBF-Periodic": "GP-RBF-Periodic",
    "BayesianRidge": "BayesianRidge", "BayesianMCMC": "BayesianMCMC",
    "NegBinGLM": "NegBinGLM", "PoissonAutoreg": "PoissonAutoreg",
}


def load_optuna_json(model_name: str, json_dir: str) -> Optional[dict]:
    """External Optuna JSON 결과 로딩."""
    optuna_key = _OPTUNA_MODEL_MAP.get(model_name, model_name)
    patterns = [
        f"optuna_feat_sel_{optuna_key}.json",
        f"optuna_all_strategies_{optuna_key}.json",
    ]
    for pattern in patterns:
        path = Path(json_dir) / pattern
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
    return None


def get_best_features_from_json(data: dict, feature_cols: list) -> Tuple[List[str], str]:
    """JSON 결과에서 최적 전략의 피처 리스트 추출."""
    if not data:
        return feature_cols, "fallback"

    best_strategy = data.get("best_strategy", "")
    best_result = data.get(best_strategy, data)

    if isinstance(best_result, dict):
        features = best_result.get("best_features", best_result.get("features", []))
        if features:
            valid = [f for f in features if f in feature_cols]
            if valid:
                return valid, best_strategy
    return feature_cols, "fallback"


from simulation.utils.resource_tracker import track_resources


@track_resources("phase5_external")
def run_external(X_all, y_all, feature_cols, config) -> dict:
    """External Optuna 실행 또는 JSON 캐시 로딩 후 학습."""
    from simulation.models.runner import MultiModelRunner
    from simulation.models.target_transform import (
        get_preset, get_per_model_strategy, TargetTransformer,
    )

    t0 = time.time()
    n = len(y_all)
    # 2026-05-12 G-187 (사용자 명시): 모든 phase 가 _compute_split_sizes() HWP path 사용.
    # 이전 LEGACY path (n_train = int(n * train_ratio) = 70%) → 200/35/102 mismatch.
    # 정정: HWP §3 표준 = 242/27/68 (in_sample=337) — phase1_data 와 일관.
    from simulation.pipeline.data import compute_split_indices
    n_train, _n_val_phase22, _n_test_phase22 = compute_split_indices(n, config)

    # AUDIT 2026-06-01 (누수 fix): recode(quantile/threshold/interaction)를 per-model slice 前에 적용.
    # 이전: per-model slice(X_m = X_all[:, feat_idx], 아래 loop)가 recode 前이라 external-선택
    #   recode-target feature(_qbin/_ili/above_threshold)가 build-time(global-stat) 값으로 누수.
    #   train_end 경계서 먼저 recode → 모든 경로(per-model slice + 기본 X_train/test)가 train-only 값.
    try:
        from simulation.pipeline.wfcv import (
            _recode_quantile_features_per_fold,
            _recode_above_threshold_per_fold,
            _recode_interaction_features_per_fold,
        )
        _feat_pre = list(feature_cols) if feature_cols else []
        if _feat_pre and n_train >= 10:
            X_all = _recode_quantile_features_per_fold(X_all, _feat_pre, n_train)
            X_all = _recode_above_threshold_per_fold(X_all, y_all, _feat_pre, n_train)
            X_all = _recode_interaction_features_per_fold(X_all, _feat_pre, n_train)
            log.info(f"  [S1-1] external: fold-wise recode @ train_end={n_train} (pre-slice, 누수fix)")
    except Exception as _re:
        log.warning(f"  [S1-1] external recode 실패 (계속 진행): {_re}")

    json_dir = config.optuna.external_json_dir or str(config.get_save_dir())
    log.info(f"  External Optuna JSON 디렉토리: {json_dir}")

    # 모델별 피처 로딩
    per_model_features = {}
    feature_selection_log = {}

    # Bug-A fix: R3도 --models 필터(_selected_models)를 존중한다.
    # 과거엔 R2/R4 만 include_only 를 적용해 3개 모델 retry 할 때도
    # R3 external 에서 54개 전체를 재학습하는 중복이 있었음.
    _selected = getattr(config, "_selected_models", None) or []
    if _selected:
        model_names = [m for m in _OPTUNA_MODEL_MAP.keys() if m in _selected]
        log.info(
            f"  [partial] R3 제한 (_selected_models={_selected}): "
            f"{len(model_names)}/{len(_OPTUNA_MODEL_MAP)} 모델만 재학습"
        )
    else:
        model_names = list(_OPTUNA_MODEL_MAP.keys())  # G-253: 66-baseline reference 유지 (철회)
    for model_name in model_names:
        data = load_optuna_json(model_name, json_dir)
        if data:
            features, strategy = get_best_features_from_json(data, feature_cols)
            feat_idx = sorted([feature_cols.index(f) for f in features if f in feature_cols])
            # B1 (G-186 후속, 2026-05-12): shape validation 추가.
            # feat_idx empty 또는 out-of-range 시 skip — caller bug 0초 차단.
            if not feat_idx:
                log.warning(f"  [{model_name}] external optuna json features 0/{len(features)} "
                              f"matched feature_cols → skip R3 (default features 사용)")
                continue
            if max(feat_idx) >= X_all.shape[1]:
                log.warning(f"  [{model_name}] feat_idx max {max(feat_idx)} >= "
                              f"X_all.shape[1] {X_all.shape[1]} → skip R3")
                continue
            X_m = X_all[:, feat_idx]
            # G-187 (2026-05-12): HWP §3 표준 split — train/val/test 명시.
            # phase1_data._compute_split_sizes 와 일관 (242/27/68 in-sample).
            # 이전 G-186 fix (LEGACY path) 는 200/35/102 였음 — paper와 mismatch.
            per_model_features[model_name] = (
                X_m[:n_train],                              # train (242)
                X_m[n_train:n_train + _n_val_phase22],     # val (27)
                X_m[n_train + _n_val_phase22:],            # test (68)
            )
            feature_selection_log[model_name] = {
                "strategy": strategy, "n_features": len(features)
            }
        # JSON 없으면 MI fallback (per_model_features에 안 넣으면 runner가 기본 피처 사용)

    log.info(f"  ✓ {len(per_model_features)}개 모델 Optuna 피처 로딩 완료")

    # 학습
    # G-305 (2026-06-17, 사용자 원칙 "x/y 변환은 preproc·HP Optuna 에서만"): external(R3)은
    #   Optuna 로 FEATURE 만 선택하고 y-변환은 고정 preset(log1p)을 걸었다 → 변환이 Optuna 탐색이
    #   아니므로 원칙 위반. baseline(G-305)과 동형으로 pipeline y-변환을 none(raw)으로 override.
    #   y/x 변환은 오직 R9 preproc/HP Optuna. covid_strategy 는 preset 유지.
    _tt_preset, covid_strategy = get_preset(config.preset)
    tt = TargetTransformer(method="none")        # G-305: NO pipeline y-transform (raw)
    per_model_map = {}                           # G-305: per-model 고정 변환 제거 → 전 모델 raw

    # MultiModelRunner : covid_strategy 는 내부 통합됨 — kwarg 로 안 받음
    # : feature_names 전달해 native DL 활성화
    # Bug-A fix (2026-04-25): per_model_features 가 8개로 줄어도 runner 가
    # REGISTRY 전체 (29 모델) 를 학습했음. include_only 로 진짜 필터 적용.
    # 영향: --models 13개 명시 시 학습 시간 ~50% 단축 (3h41m → 1.5~2h 예상).
    # Bug-B fix (2026-06-08): include_only must honor the explicit --models filter
    # (_selected) even when NO external Optuna JSON exists (fresh run → per_model_features
    # empty → previously _filter_keys=None → runner ran ALL 56 models despite --models).
    # _selected (config._selected_models) is the user's authoritative model restriction;
    # per_model_features only narrows which models get external-optuna features loaded.
    _filter_keys = (list(per_model_features.keys()) if per_model_features
                    else (_selected or None))
    runner = MultiModelRunner(
        target_transformer=tt,
        per_model_transform=per_model_map,
        per_model_features=per_model_features,
        feature_names=list(feature_cols) if feature_cols else None,
        include_only=_filter_keys,
    )
    if _filter_keys:
        log.info(f"  [Bug-A fix] include_only 진짜 필터 적용: "
                  f"{len(_filter_keys)}개 모델만 외부 학습")

    # G-187 (2026-05-12): n_val 은 위 _compute_split_sizes 에서 받은 _n_val_phase22 사용.
    # 이전 LEGACY: n_val = int(n_train * 0.15) → 잘못 (HWP 27 vs LEGACY 36).
    n_val = _n_val_phase22

    # (recode 는 위 per-model slice 前으로 이동 — AUDIT 2026-06-01 누수fix; 여기 중복 제거)

    # G-187 (2026-05-12): HWP §3 표준 split — n_train=242 / n_val=27 / n_test=68.
    # 이전 LEGACY (n_train=235): X_train=X_all[:200], X_val=[200:235], X_test=[235:].
    # 정정 HWP: X_train=X_all[:242], X_val=[242:269], X_test=[269:337].
    X_train = X_all[:n_train]
    y_train = y_all[:n_train]
    X_val = X_all[n_train:n_train + n_val]
    y_val = y_all[n_train:n_train + n_val]
    X_test = X_all[n_train + n_val:]
    y_test = y_all[n_train + n_val:]

    results = runner.run(
        X_train, y_train, X_val, y_val, X_test, y_test,
        run_ensembles=config.training.run_ensembles,
        save_models=config.training.save_models,
        save_dir=str(config.get_model_dir()),
    )

    elapsed = time.time() - t0
    log.info(f"  ✓ External Optuna 학습 완료: {len(results)}개 모델, {elapsed:.0f}s")

    # R8.3 (2026-05-26): full 134-key SSOT eval on external Optuna test predictions.
    # Trajectory: R2 baseline → external → R4 OOF → ... → R8 final.
    # Toggle via env MPH_FULL_EVAL_TRAJECTORY (default '1' = enabled).
    from simulation.config_global import GLOBAL as _GCFG  # SSOT (2026-05-28)
    if _GCFG.filter.full_eval_trajectory:
        try:
            from simulation.pipeline.phase_evaluator import evaluate_predictions_full
            _ind = results.get("individual_results", {}) if isinstance(results, dict) else {}
            for _mname, _mres in _ind.items():
                if not isinstance(_mres, dict):
                    continue
                _tp = _mres.get("test_pred")
                if _tp is None:
                    continue
                _y_arr = np.asarray(y_test, dtype=np.float64)
                _p_arr = np.asarray(_tp, dtype=np.float64)
                _mask = np.isfinite(_y_arr) & np.isfinite(_p_arr)
                if _mask.sum() < 5:
                    continue
                try:
                    full_r8 = evaluate_predictions_full(
                        y_test=_y_arr[_mask],
                        y_pred=_p_arr[_mask],
                        residuals=(_y_arr[_mask] - _p_arr[_mask]),
                        y_train_pool=np.asarray(y_train, dtype=np.float64),
                        threshold=GLOBAL.filter.alert_threshold,
                        phase_id=f"R3_external_{_mname}",
                        enable_bootstrap_ci=False,
                    )
                    _mres["phase_eval_r8"] = full_r8
                except Exception as _e:
                    _mres["phase_eval_r8_err"] = str(_e)
        except Exception as _e:
            log.debug(f"  [phase5_external] phase_eval_r8 wiring skipped: {_e}")

    # : plots + CSVs (matplotlib + plotly + seaborn)
    try:
        from simulation.pipeline.plotting import generate_all as _plot_all
        _plot_manifest = _plot_all(
            runner_result=results,
            y_val=y_val, y_test=y_test,
            output_root=str(config.get_save_dir()),
            tag="phase5_external",
        )
    except Exception as _pe:
        log.warning(f"  [plot] external generate_all failed: {_pe}")
        _plot_manifest = {"status": "error", "error": str(_pe)}

    # Sprint 3 EDA sidecar (2026-05-26) — non-fatal, atomic write
    try:
        import numpy as _np
        from pathlib import Path as _Path
        from .eda_writer import write_phase_eda
        _test_preds = {
            _m: _np.asarray(_r.get("test_pred"))
            for _m, _r in (results.get("individual_results", {}) or {}).items()
            if _r.get("test_pred") is not None
        }
        if _test_preds:
            write_phase_eda(
                phase_id=2, phase_tag="external",
                y_true=y_test, predictions=_test_preds,
                save_dir=_Path(config.get_save_dir()) / "eda",
                extra_meta={"n_models": len(results),
                             "feature_selection_log": bool(feature_selection_log)},
            )
    except Exception as _eda_e:
        log.debug(f"  [phase5_external] EDA sidecar skipped: {_eda_e}")

    return {
        "model_results": results,
        "feature_selection_log": feature_selection_log,
        "n_models": len(results),
        "elapsed": elapsed,
        "plot_manifest": _plot_manifest,
    }


# back-compat aliases (2026-06-02 semantic rename — 옛 run_phaseN)
run_phase5_external = run_external
