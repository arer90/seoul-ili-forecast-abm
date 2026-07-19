"""
R2: Baseline 학습 (BASIC-feature reference)
================================================
기본 피처(lag + 계절성)만, 기본 하이퍼파라미터로 모든 모델 학습.
"단순 lag 모델 대비 최적화 모델이 우수한가?" 를 보는 기준선 — full-feature 가 아님
(G-240, 2026-05-30 사용자 설계: 이전 full-feature reference 를 basic 으로 교체).
"""
import logging
from simulation.config_global import GLOBAL  # SSOT (2026-05-28)
import time
import numpy as np

log = logging.getLogger(__name__)

# SSOT (G-240, 2026-05-30): baseline 은 BASIC 피처(lag + 계절성)만 사용 — 단순 lag
# 모델 대비 최적화 모델 우위를 보는 reference. 없는 컬럼은 런타임에 자동 skip
# (lag52 는 include_seasonal_extra 에 따라 부재 가능). week-of-year 는 raw 정수가
# 없으므로 sin/cos_month + Fourier 가 그 신호를 담당.
BASIC_FEATURE_COLS = [
    "ili_rate_lag1", "ili_rate_lag2", "ili_rate_lag4", "ili_rate_lag52",
    "sin_month", "cos_month",
    "fourier_sin_h1", "fourier_cos_h1", "fourier_sin_h2", "fourier_cos_h2",
    "fourier_sin_h3", "fourier_cos_h3",
    "season_idx",
]


from simulation.utils.resource_tracker import track_resources


@track_resources("phase4_baseline")
def run_baseline(X_all, y_all, feature_cols, config) -> dict:
    """Baseline 학습 실행.

    Returns: {model_results: {name: {r2, rmse, mape, ...}}, elapsed}
    """
    from simulation.models.runner import MultiModelRunner
    from simulation.models.target_transform import (
        get_preset, get_per_model_strategy, TargetTransformer,
    )

    t0 = time.time()
    n = len(y_all)
    # Single source of truth — same indices phase1 published.
    from .data import compute_split_indices
    n_train, n_val, n_test = compute_split_indices(n, config)
    if config.split.use_validation:
        # 4-way: train | val | test (val carved by helper)
        n_train_actual = n_train  # alias kept for legacy lines below
    else:
        # WF-CV: collapse val into train, then carve last 15% as inner-val
        n_train = n_train + n_val
        n_val = int(n_train * 0.15)
        n_train_actual = n_train - n_val

    # S1-1 full-close: rebuild train-boundary-dependent interaction /
    # above_threshold / quantile features using max/median/bin computed only
    # over X_all[:n_train]. Without this, R2 test split leaks through
    # the global max() normalization baked in at build time.
    _train_end_for_recode = n_train_actual if not config.split.use_validation else n_train
    try:
        from simulation.pipeline.wfcv import (
            _recode_quantile_features_per_fold,
            _recode_above_threshold_per_fold,
            _recode_interaction_features_per_fold,
        )
        _feat = list(feature_cols) if feature_cols else []
        if _feat and _train_end_for_recode >= 10:
            X_all = _recode_quantile_features_per_fold(X_all, _feat, _train_end_for_recode)
            X_all = _recode_above_threshold_per_fold(X_all, y_all, _feat, _train_end_for_recode)
            X_all = _recode_interaction_features_per_fold(X_all, _feat, _train_end_for_recode)
            log.info(f"  [S1-1] fold-wise recode applied @ train_end={_train_end_for_recode}")
    except Exception as _re:
        log.warning(f"  [S1-1] recode 실패 (계속 진행, leakage caveat): {_re}")

    # G-240 (2026-05-30): baseline = BASIC features only (lag + seasonal). A simple
    # "beat a naive lag model" reference — NOT a full-feature run. Column-slice happens
    # AFTER the leakage recode (which only touches quantile/threshold/interaction cols
    # absent from the basic set → no-op here). All basic cols are shift-based / calendar
    # → causal, leakage-safe. Missing cols (e.g. lag52) auto-skip.
    _basic = [c for c in BASIC_FEATURE_COLS if feature_cols and c in feature_cols]
    if _basic:
        _bidx = np.array([list(feature_cols).index(c) for c in _basic])
        X_all = X_all[:, _bidx]
        feature_cols = _basic
        log.info(f"  [baseline] BASIC feature subset: {len(_basic)}/{len(BASIC_FEATURE_COLS)} cols")
    else:
        log.warning("  [baseline] BASIC_FEATURE_COLS none matched feature_cols — full set fallback")

    if config.split.use_validation:
        # 4-way HWP: [ train (n_train) | val (n_val) | test (n_test) ]
        pool_end = n_train + n_val
        X_train, y_train = X_all[:n_train], y_all[:n_train]
        X_val,   y_val   = X_all[n_train:pool_end], y_all[n_train:pool_end]
        X_test,  y_test  = X_all[pool_end:],        y_all[pool_end:]
    else:
        # WF-CV: train_actual + inner-val + test, total = n
        X_train, y_train = X_all[:n_train_actual], y_all[:n_train_actual]
        X_val,   y_val   = X_all[n_train_actual:n_train], y_all[n_train_actual:n_train]
        X_test,  y_test  = X_all[n_train:],        y_all[n_train:]

    log.info(f"  Baseline 학습: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    log.info(f"  피처 수: {X_all.shape[1]} (basic: lag + 계절성)")

    # Target Transform
    # G-305 (2026-06-17, 사용자 원칙 "x/y 변환은 preproc·HP Optuna 에서만 — 그 외 절대 X"):
    #   baseline 은 pipeline-level y 변환을 적용하지 않는다 → 순수 raw y + BASIC feature reference.
    #   변환은 오직 R9 preproc/HP Optuna 가 담당. 모델 내부 전처리(scaler·log-link 등)는 각
    #   모델 책임으로 유지(이건 "변환"이 아니라 모델 정의). covid_strategy 는 preset 에서 유지.
    _tt_preset, covid_strategy = get_preset(config.preset)
    tt = TargetTransformer(method="none")        # G-305: NO pipeline y-transform (raw)
    per_model_map = {}                           # G-305: per-model 고정 변환 제거 → 전 모델 raw

    log.info(f"  프리셋: {config.preset} (G-305: pipeline transform override → none/raw), "
             f"Target=none, COVID={covid_strategy.mode}")

    # MultiModelRunner : covid_strategy 는 내부 통합됨 — kwarg 로 안 받음
    # : feature_names 전달 (DL 입력 컬럼 정합)
    # : --models CLI 필터를 include_only 로 전달 (partial refit).
    # G-253 (2026-06-11 정정): baseline = REGISTRY 전체(66) = active 54 + deferred 12
    # 의 **공정 baseline reference**. deferred 의 성능 비교는 이 66-baseline 으로만 가능
    # (실증: BayesianMCMC r2=0.746 #16/66 → active 승격 근거였음). 53/54-제한 시도는
    # MultiModelRunner include_only 가 hard-restrict 안 해 무효 + reference 손실이라 철회.
    # 53(→54) 은 phase 13 OPTIMIZE 스코프(CATEGORY_MODELS); baseline 은 별개 reference.
    _selected = getattr(config, "_selected_models", None) or []
    runner = MultiModelRunner(
        target_transformer=tt,
        per_model_transform=per_model_map,
        per_model_features=None,  # Baseline: 위에서 basic 으로 slice 완료 → 전 모델 동일 set
        feature_names=list(feature_cols) if feature_cols else None,
        include_only=_selected,
    )
    if _selected:
        log.info(f"  [partial] --models 필터: {_selected}")

    results = runner.run(
        X_train, y_train, X_val, y_val, X_test, y_test,
        run_ensembles=config.training.run_ensembles,
        save_models=config.training.save_models,
        save_dir=str(config.get_model_dir()),
    )

    elapsed = time.time() - t0
    log.info(f"  ✓ Baseline 학습 완료: {len(results)}개 모델, {elapsed:.0f}s")

    # R8.2 (2026-05-26): full 134-key SSOT eval on baseline test predictions.
    # Trajectory: R2 raw baseline → R4 OOF → ... → R8 final SSOT.
    # Earliest snapshot in the chain — useful for comparing baseline vs tuned trajectory.
    # R8.3 (2026-05-26): MPH_FULL_EVAL_TRAJECTORY toggle (default '1' = enabled).
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
                    phase_id=f"R2_baseline_{_mname}",
                    enable_bootstrap_ci=False,
                )
                _mres["phase_eval_r8"] = full_r8
            except Exception as _e:
                _mres["phase_eval_r8_err"] = str(_e)
    except Exception as _e:
        log.debug(f"  [phase4_baseline] phase_eval_r8 wiring skipped: {_e}")

    # partial refit A: R2 sidecar.
    #   --models 로 일부만 재학습하더라도 이전 full-run 의 나머지 모델 결과를
    #   병합해 downstream 리포트가 전체 46 모델을 다 보이도록.
    try:
        import pickle
        from pathlib import Path
        _p2_sidecar = Path(config.get_save_dir()) / "phase4_baseline_sidecar.pkl"
        _partial = bool(_selected)
        _merged_ind: dict = {}
        _merged_ens: dict = {}
        if _partial and _p2_sidecar.exists():
            try:
                with _p2_sidecar.open("rb") as _f:
                    _prev = pickle.load(_f)
                _merged_ind.update(_prev.get("individual_results", {}) or {})
                _merged_ens.update(_prev.get("ensemble_results", {}) or {})
                log.info(
                    f"  [partial] phase4 sidecar 로드: "
                    f"individual {len(_merged_ind)}개, ensemble {len(_merged_ens)}개 유지"
                )
            except Exception as _pe:
                log.warning(f"  [partial] phase4 sidecar 로드 실패 → 새로 작성: {_pe}")
        # 이번 run 결과로 덮어쓰기
        _merged_ind.update(results.get("individual_results", {}) or {})
        _merged_ens.update(results.get("ensemble_results", {}) or {})
        with _p2_sidecar.open("wb") as _f:
            pickle.dump({
                "individual_results": _merged_ind,
                "ensemble_results": _merged_ens,
            }, _f)
        # 다운스트림 병합 뷰 반영 (partial 에서만 — full 은 자기 결과 그대로)
        if _partial:
            results = dict(results)
            results["individual_results"] = _merged_ind
            results["ensemble_results"] = _merged_ens
        log.info(
            f"  [partial] phase4 sidecar 저장: {_p2_sidecar} "
            f"(individual {len(_merged_ind)}개, ensemble {len(_merged_ens)}개"
            + (", partial merge" if _partial else "")
            + ")"
        )
    except Exception as _se:
        log.warning(f"  [partial] phase4 sidecar 저장 실패: {_se}")

    # : plots + CSVs (matplotlib + plotly + seaborn)
    try:
        from simulation.pipeline.plotting import generate_all as _plot_all
        _plot_manifest = _plot_all(
            runner_result=results,
            y_val=y_val, y_test=y_test,
            output_root=str(config.get_save_dir()),
            tag="phase4_baseline",
        )
    except Exception as _pe:
        log.warning(f"  [plot] baseline generate_all failed: {_pe}")
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
                phase_id=2, phase_tag="baseline",
                y_true=y_test, predictions=_test_preds,
                save_dir=_Path(config.get_save_dir()) / "eda",
                extra_meta={"split": {"n_train": len(X_train),
                                       "n_val": len(X_val),
                                       "n_test": len(X_test)}},
            )
    except Exception as _eda_e:
        log.debug(f"  [phase4_baseline] EDA sidecar skipped: {_eda_e}")

    return {
        "model_results": results,
        "n_models": len(results),
        "elapsed": elapsed,
        "split": {"n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test)},
        "plot_manifest": _plot_manifest,
    }


# back-compat aliases (2026-06-02 semantic rename — 옛 run_phaseN)
run_phase4_baseline = run_baseline
