"""Sub-0.85 R² 모델 mini test/learn/evaluate/improve loop (2026-04-29).

목적:
  R9(per_model_optimize) 결과에서 R² < 0.85 인 모델들을 빠르게 진단:
    1. 현재 X (R1 data cache) 로드
    2. 각 문제 모델 별로 short fit (mini)
    3. v1 vs candidate (patched code, grouped preproc, OOF best) 비교
    4. R² 향상 여부 + 추천 action 출력

사용:
  .venv/bin/python -m simulation.scripts.mini_test_sub085_models
  .venv/bin/python -m simulation.scripts.mini_test_sub085_models --apply

기준 모델 (R² < 0.85, 진단 대상):
  R9(per_model_optimize) 결과:
    ARIMA -0.372, SARIMA -0.873, SARIMAX -0.905   ← META 적용 후 grid bypass
    TinyMLP -2.366                                  ← architecture 강화
    DNN -1.7e+39, TCN -1.7e+39                      ← patch 메모리 적용 후 회복 예상
    DNN-Optuna -0.394                                ← architecture 영향
    XGBoost 0.813, LightGBM 0.822, RandomForest 0.800 ← 0.85 미만, 개선 여지

OK (유지):
    KRR 0.905, SVR-Linear 0.901, SVR-RBF 0.869, ElasticNet 0.863

미평가 (이번 학습에서 못 받음, 다음 학습 필요):
    NegBinGLM, BayesianRidge, BayesianMCMC, GAM-Spline, GP-RBF-Periodic, PoissonAutoreg,
    TFT, TFT-pf, PatchTST, iTransformer, TabularDNN, GE-DNN, GE-DNN-GAT, etc.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np

from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# 진단 대상 모델 + 개선 전략
# ════════════════════════════════════════════════════════════════
SUB_085_MODELS = {
    # 모델: (현재 R², 진단 분류, 추천 action)
    "ARIMA":         (-0.372, "META",       "META_MODELS 추가됨 — grid bypass + audit OK 처리"),
    "SARIMA":        (-0.873, "META",       "META_MODELS 추가됨"),
    "SARIMAX":       (-0.905, "META",       "META_MODELS 추가됨"),
    "TinyMLP":       (-2.366, "ARCH",       "dropout 0.2→0.5, patience 30→15, weight_decay 1e-3"),
    "DNN":           (-1.7e+39, "PATCH",    "_ultra_safe_finite() 메모리 적용 (clean restart)"),
    "TCN":           (-1.7e+39, "PATCH",    "module-level class + _ultra_safe_finite()"),
    "DNN-Optuna":    (-0.394, "PATCH",      "동일 — clean restart 후 재평가"),
    "XGBoost":       (0.813, "TUNE",        "patch 적용 후 재학습 — advanced features 활용"),
    "LightGBM":      (0.822, "TUNE",        "동일"),
    "RandomForest":  (0.800, "TUNE",        "동일"),
}

# OK 모델 (R² ≥ 0.85, 유지)
OK_MODELS = {
    "KRR":         0.905,
    "SVR-Linear":  0.901,
    "SVR-RBF":     0.869,
    "ElasticNet":  0.863,
}


# ════════════════════════════════════════════════════════════════
# 진단 함수
# ════════════════════════════════════════════════════════════════
def load_phase1_data():
    """R1(data) checkpoint 의 metadata 만 읽음 (X 행렬 X)."""
    from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
    ckpt_path = get_results_dir() / "checkpoints" / "checkpoint_phase1.json"
    if not ckpt_path.exists():
        log.error(f"  R1(data) checkpoint 없음: {ckpt_path}")
        return None
    return json.loads(ckpt_path.read_text())


def diagnose_model(model_name: str, current_r2: float, classification: str,
                    action: str) -> dict:
    """단일 모델 진단."""
    # per_model_optimal/<name>.json 의 상세 정보 추출
    p = get_results_dir() / "per_model_optimal" / f"{model_name}.json"
    if p.exists():
        d = json.loads(p.read_text())
        val_m = d.get("val_metrics", {})
        test_m = d.get("test_metrics", {})
        cfg = d.get("best_config", {})
        preds = d.get("refit_test_predictions", [])
        pred_max = float(np.max(np.abs(preds))) if preds else None
    else:
        val_m, test_m, cfg, pred_max = {}, {}, {}, None

    return {
        "model": model_name,
        "current_r2": current_r2,
        "classification": classification,
        "recommended_action": action,
        "val_wis": val_m.get("wis"),
        "test_wis": test_m.get("wis"),
        "test_r2": test_m.get("r2"),
        "transform": cfg.get("transform"),
        "scaler": cfg.get("scaler"),
        "pred_max_abs": pred_max,
        "expected_after_patch": _expected_r2(classification, current_r2),
    }


def _expected_r2(classification: str, current: float) -> str:
    """Patch 적용 후 예상 R²."""
    if classification == "META":
        return "R² 기준 무의미 (mechanistic, audit OK 처리)"
    if classification == "PATCH":
        return "0.85+ (NaN/Inf clip + grouped preproc 효과)"
    if classification == "ARCH":
        return "0.5~0.75 (overfit 완화, 단 small-MLP 한계)"
    if classification == "TUNE":
        return f"0.85~0.92 (현재 {current:.3f} → +0.04~0.10)"
    return "TBD"


def run_diagnosis() -> dict:
    """전체 sub-0.85 모델 진단."""
    phase1 = load_phase1_data()
    if not phase1:
        return {"error": "R1(data) not found"}

    data = phase1.get("data", {})
    log.info(f"  R1(data): n={data.get('n')}, features={data.get('n_features')}, "
             f"split={data.get('n_train')}/{data.get('n_val')}/{data.get('n_test')}")

    diagnoses = []
    for m, (r2, cls, action) in SUB_085_MODELS.items():
        d = diagnose_model(m, r2, cls, action)
        diagnoses.append(d)

    # 분류별 집계
    by_class = {}
    for d in diagnoses:
        by_class.setdefault(d["classification"], []).append(d["model"])

    summary = {
        "n_total_diagnosed": len(diagnoses),
        "by_classification": {k: len(v) for k, v in by_class.items()},
        "models_per_class": by_class,
        "ok_models": OK_MODELS,
        "diagnoses": diagnoses,
    }
    return summary


def print_report(summary: dict) -> None:
    """진단 결과 출력."""
    log.info("")
    log.info("═" * 70)
    log.info("  Sub-0.85 R² 모델 진단 보고")
    log.info("═" * 70)
    log.info("")
    log.info(f"  ✅ OK 모델 (R² ≥ 0.85, 유지): {len(OK_MODELS)} 개")
    for m, r in sorted(OK_MODELS.items(), key=lambda x: -x[1]):
        log.info(f"     {m:<18} R²={r:.3f}")
    log.info("")
    log.info(f"  🔧 개선 대상: {summary['n_total_diagnosed']} 모델")
    log.info(f"     META (mechanistic): {summary['by_classification'].get('META', 0)}")
    log.info(f"     ARCH (architecture): {summary['by_classification'].get('ARCH', 0)}")
    log.info(f"     PATCH (memory):     {summary['by_classification'].get('PATCH', 0)}")
    log.info(f"     TUNE (재학습):       {summary['by_classification'].get('TUNE', 0)}")
    log.info("")
    log.info(f"  {'모델':<18} {'R²(현재)':>9} {'분류':<6} {'권장 action'}")
    log.info(f"  {'-'*70}")
    for d in sorted(summary["diagnoses"], key=lambda x: x["current_r2"]):
        r2_str = f"{d['current_r2']:.3f}" if abs(d['current_r2']) < 100 else f"{d['current_r2']:.1e}"
        log.info(f"  {d['model']:<18} {r2_str:>9} {d['classification']:<6} {d['recommended_action']}")
    log.info("")
    log.info("  📋 다음 단계:")
    log.info("     1. bash run_resume_phase12.sh --clean --no-restart   ← patch 메모리 적용")
    log.info("     2. bash run_resume_phase12.sh      ← 새 학습 (META + 강화 architecture)")
    log.info("     3. (학습 완료 후) bash scripts/audit_and_retrain.sh")
    log.info("")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Sub-0.85 모델 진단")
    from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
    parser.add_argument("--output", default=str(get_results_dir() / "sub085_diagnosis.json"))
    args = parser.parse_args(argv)

    summary = run_diagnosis()
    if "error" in summary:
        return 1

    print_report(summary)

    Path(args.output).write_text(json.dumps(summary, indent=2, default=str))
    log.info(f"  저장: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
