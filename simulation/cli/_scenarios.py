"""SCENARIOS + ALL_MODELS — extracted from __main__.py.

Phase C2 fix (2026-05-12): cmd_train / cmd_train_all (now in
cli/training_commands.py) reference SCENARIOS dict + ALL_MODELS dict
defined at __main__ module level. Extraction missed these globals →
NameError when training launched (Day 9 regression).

This module is the single source of truth for both. __main__.py and
cli/training_commands.py both import from here.

Public API:
    SCENARIOS  — dict[str, dict]: training scenario defs (base / baseline /
                 full / full_light / quick-test / dl-only / ml-only / stat-only /
                 diagnostics-only / wfcv-only / optuna_hp / optuna_feature /
                 optuna_joint / optuna_hp_then_feature / aggressive)
    ALL_MODELS — dict[str, list[str]]: model categories (DL / ML / STAT /
                 EPI / ENSEMBLE) — model-name lists per family
"""
from __future__ import annotations


SCENARIOS = {
    # : 순수 defaults (아무 override도 없이 config 의 기본값만 사용).
    #   preset=aggressive, optuna_mode=none, optuna_strategy=hp_then_feature,
    #   epochs=100, early_stopping_patience=10, wfcv.step_size=1.
    #   Smoke/첫 실행용. 튜닝·시나리오 특성이 필요 없을 때.
    "base": {
        "desc": "Pure defaults (no overrides — config.py 기본값만 사용)",
    },
    "baseline": {
        "desc": "Baseline (Optuna off, all models, full 9 phases)",
        "optuna_mode": "none",
        # P0-1: aggressive(=log1p) preset turned DNN/TabularDNN into
        # saturation-at-cap predictors (expm1 exploded → clipped to 100.4)
        # and dragged XGB/LGBM from 0.85 → 0.55. Default to conservative
        # (no global transform); DNN still gets log1p via PER_MODEL_TRANSFORM.
        "preset": "conservative",
    },
    "optuna-external": {
        "desc": "External Optuna feature selection + training",
        "optuna_mode": "external",
        "optuna_trials": 100,
    },
    "optuna-inline": {
        "desc": "Inline Optuna (WF-CV integrated feature selection)",
        "optuna_mode": "inline",
        "optuna_trials": 100,
    },
    "full": {
        # 2026-06-02 (codex+Gemini NO-GO fix): 옛 external/pre-pipeline feature-Optuna 제거.
        # B2/B4 + eval-features 재설계 → feature 선택 = R9(per_model_optimize) STABILITY 전용, R2-R8 = BASIC.
        # 옛 optuna_mode="all"+rerun_feature_optuna=True 의 3 결함을 동시 제거:
        #   (a) run_optuna_feature_selection 의 split-전 target-corr prefilter = LEAKAGE,
        #   (b) R3(external) 이 full X_all 로 학습(BASIC 위반),
        #   (c) external→per_model_feature_map(full feature명)이 R4(WF-CV) BASIC X_eval(13컬럼)과 충돌.
        # HP·feature 최적화는 R9 per-model 전담. R2 baseline + R4 WF-CV = BASIC 비교 패널.
        "desc": "Full scenario (R9 per-model optimize, BASIC R2-R8, conformal holdout=26)",
        "optuna_mode": "none",
        "optuna_trials": 100,
        # conformal_holdout=26: honest split-conformal (full_light parity) — R7(intervals) 의
        # "oof_internal_split_OPTIMISTIC" 폴백 차단 (PI/WIS/PICP/best-WIS in-sample-optimistic 방지).
        "conformal_holdout_weeks": 26,
    },
    # Stage 3 : Mid-weight preset for the paper-primary sweep.
    #   - Full-fit epochs 200 with early_stopping_patience 20, batch 32
    #   - Optuna "all" mode, trials=30, inline_epochs=50 per trial
    #   - Targets ~6-9h on a single GPU for the PAPER_PRIMARY_11 set;
    #     dyn_cap (S0-3 fix in runner.py) may reduce trials further if
    #     n_train is small.
    "full_light": {
        # 2026-06-02 (codex+Gemini NO-GO fix): full 과 동일 — 옛 external/pre-pipeline feature-Optuna
        # 제거 (leakage prefilter + R3(external) full-X + per_model_feature_map↔BASIC 충돌). feature 선택 =
        # R9(per_model_optimize) STABILITY 전용. optuna_mode all→none, rerun_feature_optuna/feature_optuna_* 삭제.
        "desc": "Full-light (R9 per-model optimize, BASIC R2-R8, epochs 200, holdout=26) -- 6-9h GPU",
        "optuna_mode": "none",
        "optuna_trials": 30,
        "epochs": 200,
        "early_stopping_patience": 20,
        # baseline (PICP=80.77%) 재현 위한 26주 holdout 강제 (config default 0 → 명시 주입). PI A/B 비교 필수.
        "conformal_holdout_weeks": 26,
    },
    "quick-test": {
        "desc": "Quick test (lite mode, 20 trials, 30 epochs)",
        "optuna_mode": "external",
        "lite": True,
    },
    "dl-only": {
        "desc": "DL models only (17 — dl-tabular + dl-seq + modern-ts + graph)",
        "optuna_mode": "inline", "epochs": 200,
        "models": [
            # dl-tabular
            "DNN", "DNN-Optuna", "TabularDNN-Lite",
            # dl-seq
            "TCN", "TCN-Optuna",
            # modern-ts
            "PatchTST", "iTransformer", "Mamba", "TimesNet",
            "N-BEATS", "N-HiTS", "TiDE", "TFT", "DeepAR", "RNN",
            # graph
            "GCN", "GAT",
        ],
    },
    "ml-only": {
        # 2026-05-12 fix (Codex): stale 'SVR/Ridge/ExtraTrees/AdaBoost/KNN' 제거 → live registry 11 ML
        "desc": "ML models only (11 — tree + kernel + other; XGBoost / LightGBM / GAM / KRR / etc.)",
        "optuna_mode": "external", "optuna_trials": 100,
        "models": [
            "XGBoost", "LightGBM", "RandomForest", "GradientBoosting", "CatBoost",
            "KRR", "SVR-Linear", "SVR-RBF",
            "GAM-Spline", "GP-RBF-Periodic", "BayesianMCMC",
        ],
    },
    "stat-only": {
        # 2026-05-12 fix (Codex): stale 'Prophet/ExponentialSmoothing/PoissonGLM' 제거 → live registry 8
        "desc": "Statistical models only (8 — ts + linear; SARIMAX / NegBinGLM / ElasticNet / etc.)",
        "models": [
            "ARIMA", "SARIMA", "SARIMAX",
            "ElasticNet", "BayesianRidge", "NegBinGLM", "NegBinGLM-V7", "PoissonAutoreg",
        ],
    },
    "diagnostics-only": {
        # resume_from 은 R/P 라벨로 적는다. 정수 인덱스는 PHASES 순서가 바뀔 때마다
        # 조용히 어긋난다 — 실제로 7(=R8 scoring)이 들어가 있어 R5·R6·R7 을 건너뛰었다.
        # training_commands.py 는 int 면 PHASES[i][0], str 이면 그대로 통과시킨다.
        "desc": "Re-run diagnostics + downstream (R5 onward, reuse trained models)",
        "resume_from": "R5",
    },
    "wfcv-only": {
        # 같은 버그: 6(=R7 intervals)이 들어가 있어 R4·R5·R6 을 건너뛰었다.
        "desc": "Re-run from Walk-Forward CV (R4)",
        "resume_from": "R4",
    },
    # Stage-3: Optuna-strategy scenarios. All four use inline mode
    # (WF-CV integrated) with trials=30 + moderate preset as the working
    # baseline. `optuna_strategy` routes to the right search topology inside
    # simulation/models/optuna_search.py.
    "optuna_hp": {
        "desc": "Stage-3: HP-only Optuna (mandatory_only strategy)",
        "optuna_mode": "inline",
        "optuna_trials": 30,
        "optuna_strategy": "mandatory_only",
        "preset": "moderate",
    },
    "optuna_feature": {
        "desc": "Stage-3: feature-only Optuna (feature_only strategy)",
        "optuna_mode": "inline",
        "optuna_trials": 30,
        "optuna_strategy": "feature_only",
        "preset": "moderate",
    },
    "optuna_joint": {
        "desc": "Stage-3: joint HP+feature Optuna (joint strategy)",
        "optuna_mode": "inline",
        "optuna_trials": 30,
        "optuna_strategy": "joint",
        "preset": "moderate",
    },
    "optuna_hp_then_feature": {
        "desc": "Stage-3: two-phase HP->feature Optuna (default strategy)",
        "optuna_mode": "inline",
        "optuna_trials": 30,
        "optuna_strategy": "hp_then_feature",
        "preset": "moderate",
    },
    # Stage-3 aggressive sweep: full_light budget + two-phase Optuna on all
    # models (no paper-primary gating here; combine with --paper-primary-only
    # at runner level if wanted).
    "aggressive": {
        # 2026-06-02 (codex+Gemini NO-GO fix): full 과 동일 stale external/pre-pipeline feature-Optuna
        # 제거 (leakage + R3(external) full-X + per_model_feature_map↔BASIC 충돌). feature 선택 = R9 전용.
        # + conformal_holdout_weeks=26 추가 (기존 누락 → R7(intervals) OPTIMISTIC PI 폴백 위험; full/full_light parity).
        "desc": "Stage-3 aggressive (R9 per-model optimize, BASIC R2-R8, moderate preset, holdout=26)",
        "optuna_mode": "none",
        "optuna_trials": 30,
        "epochs": 200,
        "early_stopping_patience": 20,
        "preset": "moderate",
        "conformal_holdout_weeks": 26,
    },
}

#: ALL_MODELS — dynamically mirrors registry.CATEGORY_MODELS (SSOT).
#: 2026-05-26 Sprint prune: 78 → 50 (Codex 제안) → 실제 착지 53 active (user KEEP override). Group mapping
#: rebuilt from live `simulation.models.registry.CATEGORY_MODELS` to eliminate the
#: 53/65 drift that the Gemini consistency audit (H-1) flagged. Future REGISTRY
#: edits propagate automatically; no manual ALL_MODELS sync needed.
def _build_all_models() -> dict[str, list[str]]:
    """Group CATEGORY_MODELS into the 5+1 broad families used by SCENARIOS.

    Mapping (post-2026-05-26 prune):
      DL         = dl-tabular + modern-ts + graph                 (~21 models)
      ML         = tree + kernel + other                           (~8 models)
      STAT       = ts + linear                                     (~9 models)
      EPI        = epi-extended                                    (~7 models)
      FOUNDATION = foundation                                      (~3 — TimesFM-2.5/TiRex/OverseasTransfer)
      ENSEMBLE   = ensemble                                        (~6 models)
      CQR        = cqr                                             (~3 models)
    """
    from simulation.models.registry import CATEGORY_MODELS
    cm = CATEGORY_MODELS
    return {
        "DL":         list(cm.get("dl-tabular", []) +
                           cm.get("modern-ts",  []) +
                           cm.get("graph",      [])),
        "ML":         list(cm.get("tree",   []) +
                           cm.get("kernel", []) +
                           cm.get("other",  [])),
        "STAT":       list(cm.get("ts",     []) +
                           cm.get("linear", [])),
        "EPI":        list(cm.get("epi-extended", [])),
        "FOUNDATION": list(cm.get("foundation", [])),
        "ENSEMBLE":   list(cm.get("ensemble", [])),
        "CQR":        list(cm.get("cqr", [])),
    }


ALL_MODELS = _build_all_models()


__all__ = ["SCENARIOS", "ALL_MODELS"]

