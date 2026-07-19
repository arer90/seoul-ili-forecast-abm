"""R11 (xai) — XAI (SHAP + Permutation Importance) on R9 (per_model_optimize) champion models.

사용자 명시 (2026-05-28): "shap에서 내가 아는 shap형태나 feature importance가 아니라
그냥 형태만 나오는 것 같아. 원래 shap은 11,12단계 이후로 완성된 모델에서 shap을 하는거
아니야? xai로 의미로 말이야?"

문제 (R11 (xai) 기존):
- runner.py L874 = R11 (xai) dispatch (R9 (per_model_optimize) 의 L930 전).
- R11 (xai) 가 R9 (per_model_optimize) champion .pt 가져오려 하지만 미완료 → model_dir 비어있음.
- Tree 5 모델만 (XGBoost/LightGBM/RandomForest/GradientBoosting/ExtraTrees) — DL/Graph/ARIMA 미적용.

해결 (R11 (xai) 신규):
- runner.py 의 R9 (per_model_optimize) (L990) 후 dispatch.
- per_model_optimal/<MODEL>.json + champion bundle 로 SHAP/permutation importance.
- 모델 종류별 explainer 선택:
    tree (XGBoost/LightGBM/RandomForest/GradientBoosting/CatBoost): shap.TreeExplainer
    dl (DNN/TCN/PatchTST/iTransformer/Mamba/TimesNet/N-BEATS/N-HiTS/TiDE):
        shap.GradientExplainer (작은 sample) 또는 PermutationImportance (cheaper).
    sklearn (ElasticNet/BayesianRidge/SVR-*/KRR/GAM-Spline):
        sklearn.inspection.permutation_importance
    arima/sarima/seir/bayesian-mcmc/poisson-autoreg: skip (의미 X — model-internal AR 구조)
    ensemble: meta importance (component weights → contribution)

Output:
    simulation/results/phase15_xai/
    ├── <MODEL>/
    │   ├── shap_values.npy            (tree/dl)
    │   ├── importance.csv             (feature × score)
    │   ├── summary_plot.png           (bar/dot)
    │   └── waterfall_plot.png         (per-prediction example)
    ├── _summary.json                  (model_count, types, total_features 등)
    └── _ranking.csv                   (모델 ranking by mean |shap|)

Status (2026-05-28): SKELETON — body 다음 turn.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path


from simulation.utils.resource_tracker import track_resources

log = logging.getLogger(__name__)


# Model type → explainer 매핑 (사용자 명시 design)
TREE_MODELS = {"XGBoost", "LightGBM", "RandomForest", "GradientBoosting", "CatBoost", "ExtraTrees"}
SKLEARN_LINEAR_MODELS = {"ElasticNet", "BayesianRidge", "Ridge", "Lasso",
                          "SVR-Linear", "SVR-RBF", "KRR", "GAM-Spline",
                          "NegBinGLM", "PoissonAutoreg"}
DL_MODELS = {"DNN", "DNN-Optuna", "TabularDNN", "TabularDNN-Lite",
             "TCN", "TCN-Optuna", "PatchTST", "iTransformer", "Mamba", "TimesNet",
             "N-BEATS", "N-HiTS", "TiDE"}
GRAPH_MODELS = {"GAT", "GCN"}
SKIP_MODELS = {"ARIMA", "SARIMA", "SARIMAX", "Bayesian-SEIR", "Metapop-SEIR",
                "SEIR-V2-Forced", "Rt-Augmented", "BayesianMCMC",
                # G-261 (2026-06-13): Chronos-2 / Chronos-2-FT 제거 — Chronos retire.
                #   active foundation(TimesFM-2.5/TiRex/OverseasTransfer)도 feature-free → SHAP skip.
                "TimesFM-2.5", "TiRex", "OverseasTransfer",
                "FluSight-Ensemble", "Phase-Adaptive", "FluSight-Baseline",
                # Ensemble = component aggregation, no per-feature SHAP
                "Ensemble-NNLS", "Ensemble-NNLS-Filtered", "Ensemble-BMA",
                "Ensemble-InvRMSE", "Ensemble-Diversity", "Ensemble-Adaptive",
                "Ensemble-ResidualAR"}


def _classify_model(model_name: str) -> str:
    """모델 → explainer type 분류 ('tree' | 'linear' | 'dl' | 'graph' | 'skip')."""
    if model_name in TREE_MODELS:
        return "tree"
    if model_name in SKLEARN_LINEAR_MODELS:
        return "linear"
    if model_name in DL_MODELS:
        return "dl"
    if model_name in GRAPH_MODELS:
        return "graph"
    return "skip"


@track_resources("phase15_xai")
def run_xai(phase1: dict, all_results: dict, config) -> dict:
    """R11 (xai) — XAI on R9 (per_model_optimize) champion models.

    Args:
        phase1: R1 (data) 결과 (X_all, y_all, feature_cols).
        all_results: pipeline outputs — R9 (per_model_optimize) 결과 (per_model_configs) 필요.
        config: pipeline config (save_dir).

    Returns:
        {
            "out_dir": str,
            "n_models_processed": int,
            "n_models_skipped": int,
            "models_by_type": {tree: N, linear: N, dl: N, graph: N, skip: N},
            "elapsed": float,
            "resource_tracker": {...},   # @track_resources 자동 첨부
        }

    Status (2026-05-28): SKELETON — body 다음 sprint full 구현.
    """
    from .utils.logging_util import phase_banner
    phase_banner("R11", "XAI (SHAP + Permutation Importance) — P1 champion 활용")

    t0 = time.time()

    # R9 (per_model_optimize) 결과 가져옴 (AUDIT 2026-06-01: runner 는 "per_model_optimize" 키로 저장 — 죽은 키
    #   "phase12"/"per_model_opt" 만 읽어 항상 skip 했음. 실제 키 우선.)
    _p13 = all_results.get("per_model_optimize") or all_results.get("phase12") or all_results.get("per_model_opt") or {}
    per_model_configs = _p13.get("per_model_configs", {})
    if not per_model_configs:
        log.warning("  [phase15-xai] per_model_optimize 결과 없음 → XAI skip (skeleton)")
        return {"skipped": True, "reason": "no_per_model_optimize_results", "elapsed": time.time() - t0}

    # Output dir
    save_dir = Path(getattr(config, "save_dir", "simulation/results"))
    out_dir = save_dir / "phase15_xai"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Classify models
    by_type: dict[str, list[str]] = {"tree": [], "linear": [], "dl": [], "graph": [], "skip": []}
    for mname in per_model_configs.keys():
        by_type[_classify_model(mname)].append(mname)

    log.info(f"  [phase15-xai] classification: tree={len(by_type['tree'])} "
             f"linear={len(by_type['linear'])} dl={len(by_type['dl'])} "
             f"graph={len(by_type['graph'])} skip={len(by_type['skip'])}")

    # TODO (다음 sprint full B4 follow-up):
    # 1. Tree models: shap.TreeExplainer (champion .pt load → shap_values → importance.csv + summary_plot.png)
    # 2. Linear (sklearn): sklearn.inspection.permutation_importance
    # 3. DL models: shap.GradientExplainer or shap.DeepExplainer
    # 4. Graph (GAT/GCN): permutation importance (SHAP 직접 지원 X)
    # 5. Skip (ARIMA/SEIR/Foundation/Ensemble): N/A

    log.info("  [phase15-xai] model-type classification (real SHAP + 4-axis = shap_analysis.run_shap)")

    # Save classification summary
    import json
    summary = {
        "out_dir": str(out_dir),
        "n_models_target": len(per_model_configs),
        "models_by_type": {k: len(v) for k, v in by_type.items()},
        "models_target_by_type": {k: sorted(v) for k, v in by_type.items()},
        "status": "model-type classifier (real SHAP is shap_analysis.run_shap)",
        "note": ("The per-family SHAP body (TreeExplainer/Linear-permutation/Deep/"
                 "Kernel) + the 4-axis explanation (feature/input/output/model, "
                 "xai_explanation.json) are DONE in simulation.pipeline.shap_analysis."
                 "run_shap — both run in R11 (xai). This function provides the "
                 "model→explainer-type classification summary only."),
        "elapsed": time.time() - t0,
    }
    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info(f"  [phase15-xai] summary → {out_dir / '_summary.json'}")

    return summary


__all__ = ["run_xai", "_classify_model"]


# back-compat aliases (2026-06-02 semantic rename — 옛 run_phaseN)
run_phase15_xai = run_xai
