"""2-Stage Pipeline orchestrator + mini test (2026-05-26).

흐름:
  Stage 2 (R9 per_model_optimize 의 feature-선택 sub-stage): per-model feature Optuna on raw X
    └─ 저장: simulation/results/stage2_feature_optuna/<model>.json

  Stage 3 (R9 per_model_optimize 의 HP sub-stage): HP search with hierarchical preproc + Stage 2 fixed
    └─ R9 per_model_optimize 가 stage2_feature_optuna/ 결과를 자동 로드

  Pipeline summary:
    └─ 저장: simulation/results/pipeline_results.json

2026-05-26 변경 — phase0a archive:
  이전 (run_3stage_pipeline.py): Stage 1 (phase0a_preproc_decision) + Stage 2 + Stage 3
  현재 (run_2stage_pipeline.py): Stage 2 + Stage 3 (preproc 는 R9 per_model_optimize 의 hierarchical 이 trial 마다 결정)

사용:
  .venv/bin/python -m simulation.scripts.run_2stage_pipeline
  .venv/bin/python -m simulation.scripts.run_2stage_pipeline --models LightGBM,ElasticNet
  .venv/bin/python -m simulation.scripts.run_2stage_pipeline --n-trials 10 --test
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_phase1_data() -> Optional[dict]:
    """R1 data 데이터 로드 (cache 활용)."""
    try:
        from simulation.pipeline.config import PipelineConfig
        from simulation.pipeline.data import run_data
        cfg = PipelineConfig()
        cfg.data.use_fe_cache = True
        os.environ.setdefault("MPH_ADVANCED_FEATURES", "1")
        return run_data(cfg)
    except Exception as e:
        log.error(f"  R1 data 로드 실패: {e}")
        return None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="2-Stage Pipeline (feature Optuna → R9 per_model_optimize HP)")
    parser.add_argument("--models", default="LightGBM,ElasticNet,KRR,SVR-RBF",
                         help="comma-separated 모델 list")
    parser.add_argument("--n-trials", type=int, default=10,
                         help="Stage 2 모델당 Optuna trials (default 10)")
    parser.add_argument("--test", action="store_true",
                         help="mini test mode (소량 trials, 빠른 검증)")
    parser.add_argument("--skip-stage2", action="store_true",
                         help="R9 feature-선택 sub-stage Optuna skip")
    args = parser.parse_args(argv)

    if args.test:
        args.n_trials = 5

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    log.info(f"  2-Stage Pipeline 시작: models={model_names}, n_trials={args.n_trials}")
    if args.test:
        log.info("  TEST MODE — 소량 trials")

    # R1 data 로드
    log.info("")
    log.info("━" * 70)
    log.info("  R1 data: 데이터 + Feature Engineering 로드")
    log.info("━" * 70)
    phase1 = load_phase1_data()
    if phase1 is None:
        return 1
    X = phase1["X_all"]
    y = phase1["y_all"]
    feature_cols = phase1["feature_cols"]
    log.info(f"  X shape: {X.shape}, features: {len(feature_cols)}")
    log.info(f"  splits: train={phase1['n_train']}, "
             f"val={phase1.get('n_val', 0)}, test={phase1.get('n_test', 0)}")

    pipeline_results = {
        "started_at": datetime.utcnow().isoformat() + "Z",
        "models": model_names,
        "phase1": {
            "n_total": int(phase1.get("n", len(y))),
            "n_features": len(feature_cols),
            "n_train": phase1["n_train"],
            "n_val": phase1.get("n_val", 0),
            "n_test": phase1.get("n_test", 0),
        },
        "stages": {},
    }

    t0 = time.time()

    # Stage 2: per-model feature Optuna on raw X (R9 per_model_optimize hierarchical handles preproc).
    if not args.skip_stage2:
        log.info("")
        log.info("━" * 70)
        log.info(f"  Stage 2 (R9 feature-선택 sub-stage): Feature Optuna (n_trials={args.n_trials})")
        log.info("━" * 70)
        # 2026-05-28 (사용자 명시 design A): phase0b → _inline_optuna_3stage 이동
        from simulation.pipeline._inline_optuna_3stage import (
            _stage2_feature_optuna_inline as run_phase3_feature_optuna,
        )
        t_s2 = time.time()
        stage2 = run_phase3_feature_optuna(
            phase1, model_names, n_trials_per_model=args.n_trials,
        )
        elapsed_s2 = time.time() - t_s2
        log.info(f"  Stage 2 완료: {elapsed_s2:.1f}s")
        pipeline_results["stages"]["stage2_feature"] = {
            "elapsed_s": elapsed_s2,
            "n_models_processed": len(stage2),
            "models": {
                m: {
                    "best_score_oof_wis": v.get("best_score_oof_wis"),
                    "n_selected": v.get("n_selected"),
                    "n_total": v.get("n_features_pool_after_drop"),
                }
                for m, v in stage2.items()
            },
        }
    else:
        log.info("  Stage 2 skipped (--skip-stage2)")

    # Stage 3: R9 per_model_optimize HP Optuna — 별도 entry (python -m simulation train --resume-from 12)
    log.info("")
    log.info("━" * 70)
    log.info("  Stage 3 (R9 per_model_optimize HP sub-stage): HP Optuna — 별도 entry 로 실행")
    log.info("━" * 70)
    log.info("  Stage 2 결과는 stage2_feature_optuna/ 에 저장됨.")
    log.info("  R9 per_model_optimize 가 자동 로드하며 preproc 은 hierarchical 이 trial 마다 결정.")

    # Pipeline summary 저장
    pipeline_results["completed_at"] = datetime.utcnow().isoformat() + "Z"
    pipeline_results["total_elapsed_s"] = time.time() - t0
    from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
    out_path = get_results_dir() / "pipeline_results.json"
    out_path.write_text(json.dumps(pipeline_results, indent=2, default=str))
    log.info("")
    log.info("━" * 70)
    log.info(f"  2-Stage Pipeline 완료 ({pipeline_results['total_elapsed_s']:.1f}s)")
    log.info("━" * 70)
    log.info(f"  결과:")
    log.info(f"    simulation/results/stage2_feature_optuna/<model>.json")
    log.info(f"    simulation/results/pipeline_results.json (종합)")
    log.info("")
    log.info("  요약:")
    if "stage2_feature" in pipeline_results["stages"]:
        s2 = pipeline_results["stages"]["stage2_feature"]
        log.info(f"    Stage 2: {s2['elapsed_s']:.1f}s, "
                 f"{s2['n_models_processed']} models")
        for m, v in s2.get("models", {}).items():
            score = v.get("best_score_oof_wis")
            score_str = f"{score:.3f}" if isinstance(score, (int, float)) and np.isfinite(score) else "n/a"
            log.info(f"      {m:<18} OOF_WIS={score_str}, "
                     f"selected={v.get('n_selected')}/{v.get('n_total')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
