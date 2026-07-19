"""문제 모델만 선택적으로 재학습 (2026-04-28).

`audit_problem_models.py` 가 식별한 S0/S1 (선택적으로 S2) 모델을
patched advanced_transforms (NaN/Inf strict sanitize 적용) 으로 재학습.

워크플로우:
  1. simulation/results/problem_models_audit.json 로드
  2. (옵션) R1 (data) re-run — patched code 로 features 재생성
  3. 각 문제 모델 별로 optimize_one_model() 재실행
       - MPH_GROUPED_PREPROC=1
       - MPH_BEST_BY=oof_cv
       - MPH_STABLE_TRANSFORMS=1
  4. 결과를 simulation/results/per_model_optimal_v2/<name>.json 저장
  5. 비교 리포트 생성: v1 vs v2 (R², WIS, MAE)

CLI:
  .venv/bin/python -m simulation.scripts.retrain_problem_models
  .venv/bin/python -m simulation.scripts.retrain_problem_models \\
      --include-s2 \\
      --regenerate-phase1
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)


logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# 환경 강제 설정 (patched code 활용)
# ════════════════════════════════════════════════════════════════
def _force_env() -> None:
    """patched workflow 의 핵심 env 변수 강제 설정."""
    defaults = {
        "MPH_GROUPED_PREPROC": "1",         # ColumnTransformer 적용
        "MPH_BEST_BY": "oof_cv",             # OOF WIS minimum
        "MPH_STABLE_TRANSFORMS": "1",        # Y target = identity/log1p only
        "MPH_ADVANCED_FEATURES": "1",        # 12 카테고리 신규 features
        "MPH_PRESET": "production",
        "MPH_PRUNER": "hyperband",
    }
    for k, v in defaults.items():
        if k not in os.environ:
            os.environ[k] = v
    log.info("  Env 적용:")
    for k in defaults:
        log.info(f"    {k} = {os.environ[k]}")


# ════════════════════════════════════════════════════════════════
# R1 (data) 재생성 (옵션, --regenerate-phase1)
# ════════════════════════════════════════════════════════════════
def regenerate_phase1() -> dict:
    """R1 (data) cache 무효화 + 재실행.

    Returns: phase1 dict (X_all, y_all, feature_cols, n_train/val/test, pool_end)
    """
    from simulation.pipeline.data import run_data
    from simulation.pipeline.runner import build_cli_parser
    from simulation.pipeline.config import PipelineConfig

    log.info("  R1 (data) 재생성 (patched advanced_transforms 적용) ...")
    ckpt = get_results_dir() / "checkpoints" / "checkpoint_phase1.json"
    cache_dir = get_results_dir()
    if ckpt.exists():
        backup = ckpt.with_suffix(f".json.bak_retrain_{int(time.time())}")
        ckpt.rename(backup)
        log.info(f"  R1 (data) checkpoint 백업: {backup.name}")
    for f in cache_dir.glob("fe_cache_*.parquet"):
        f.unlink()
    log.info("  fe_cache_*.parquet 정리 완료")

    # build_cli_parser has no --scenario flag (removed simulation.utils.config);
    # absence = default full pipeline. scenario is a Stage-3 tag, irrelevant to
    # R1 (data) regeneration.
    args = build_cli_parser().parse_args([
        "--per-model-optimize",
        "--weather-mode", "hybrid",
    ])
    config = PipelineConfig.from_cli(args)
    phase1 = run_data(config)
    return phase1


def load_phase1_from_checkpoint() -> dict:
    """이미 캐시된 R1 (data) 사용 (재생성 안 함)."""
    # G-150 fix: simulation.utils.config 는 deprecated, runner.build_cli_parser 사용
    from simulation.pipeline.runner import build_cli_parser
    from simulation.pipeline.config import PipelineConfig
    from simulation.pipeline.data import run_data
    parser = build_cli_parser()
    # build_cli_parser has no --scenario flag; absence = default full pipeline.
    args = parser.parse_args([
        "--per-model-optimize",
        "--weather-mode", "hybrid",
    ])
    config = PipelineConfig.from_cli(args)
    log.info("  R1 (data) cache 사용 (regenerate=False)")
    return run_data(config)


# ════════════════════════════════════════════════════════════════
# 모델 factory 로드
# ════════════════════════════════════════════════════════════════
def build_factories(model_names: list[str]) -> dict[str, Any]:
    """REGISTRY 에서 model factory 로드 + 사전 import 강제."""
    from simulation.models.base import REGISTRY
    for _m in ("epi_models", "dl_models", "tree_models", "linear_models",
                "negbin_glm", "graph_models", "phase_ensemble",
                "conformal", "cqr_models", "bayesian_seir",
                "seir_forced", "pinn_model"):
        try:
            __import__(f"simulation.models.{_m}")
        except Exception as e:
            log.debug(f"  import 실패 (skip): {_m}: {e}")

    out = {}
    missing = []
    for n in model_names:
        spec = REGISTRY.get(n)
        if spec is None:
            missing.append(n)
            continue
        # factory 는 spec.factory 또는 spec 자체가 callable
        factory = getattr(spec, "factory", None) or spec
        out[n] = factory
    if missing:
        log.warning(f"  REGISTRY 에 없는 모델: {missing}")
    return out


# ════════════════════════════════════════════════════════════════
# 단일 모델 재학습
# ════════════════════════════════════════════════════════════════
def retrain_one(
    model_name: str,
    factory_fn,
    phase1: dict,
    output_dir: Path,
) -> dict:
    """단일 문제 모델 재학습 (hierarchical preproc Optuna — transforms/scalers 자동 선택)."""
    from simulation.pipeline.per_model_optimize import optimize_one_model

    X_all = phase1["X_all"]
    y_all = phase1["y_all"]
    feature_cols = phase1["feature_cols"]
    n_train = phase1["n_train"]
    n_val = phase1["n_val"]
    n_test = phase1.get("n_test", 0)
    pool_end = phase1.get("pool_end", n_train + n_val)

    X_train = X_all[:n_train]
    y_train = y_all[:n_train]
    X_val = X_all[n_train:n_train + n_val]
    y_val = y_all[n_train:n_train + n_val]
    X_test = X_all[pool_end:pool_end + n_test] if n_test > 0 else None
    y_test = y_all[pool_end:pool_end + n_test] if n_test > 0 else None

    t0 = time.time()
    log.info(f"  [{model_name}] 재학습 시작 "
             f"(train={n_train}, val={n_val}, test={n_test})")
    try:
        res = optimize_one_model(
            model_name, factory_fn,
            X_train, y_train, X_val, y_val,
            X_test=X_test, y_test=y_test,
            feature_cols=feature_cols,
        )
        elapsed = time.time() - t0
        res["elapsed_s"] = elapsed
        # 결과 저장
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{model_name}.json"
        out_path.write_text(json.dumps(res, indent=2, default=str))
        log.info(f"  [{model_name}] ✓ 완료 "
                 f"({elapsed:.0f}s) → {out_path.name}")
        return res
    except Exception as e:
        log.error(f"  [{model_name}] ✗ 실패: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "model": model_name}


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="문제 모델 선택적 재학습")
    parser.add_argument("--audit-path",
                         default=str(get_results_dir() / "problem_models_audit.json"),
                         help="audit_problem_models.py 출력 JSON")
    parser.add_argument("--include-s2", action="store_true",
                         help="S2 (mild) 모델도 재학습 (default: S0+S1만)")
    parser.add_argument("--regenerate-phase1", action="store_true",
                         help="R1 (data) cache 무효화 + 재실행 (patched features)")
    parser.add_argument("--output-dir",
                         default=None,
                         help="재학습 결과 저장 위치 (default: get_results_dir()/per_model_optimal_v2, MPH_OUTPUT_ROOT)")
    parser.add_argument("--models", default="",
                         help="콤마구분 모델 명시 (audit JSON 무시)")
    parser.add_argument("--max-models", type=int, default=None,
                         help="처음 N 개만 재학습 (디버그)")
    args = parser.parse_args(argv)

    _force_env()

    # ─ 재학습 대상 결정 ────────────────────────────────
    if args.models:
        targets = [m.strip() for m in args.models.split(",") if m.strip()]
        log.info(f"  명시 모델: {targets}")
    else:
        audit_p = Path(args.audit_path)
        if not audit_p.exists():
            log.error(f"  audit JSON 없음: {audit_p}")
            log.error(f"  먼저 실행: .venv/bin/python -m simulation.scripts.audit_problem_models")
            return 1
        audit = json.loads(audit_p.read_text())
        targets = list(audit.get("retrain_candidates", []))
        if args.include_s2:
            targets += list(audit.get("consider_retrain", []))
        log.info(f"  audit 결과 대상: {len(targets)} 모델 "
                 f"(S0+S1={len(audit.get('retrain_candidates', []))}, "
                 f"S2={len(audit.get('consider_retrain', []))})")

    if args.max_models:
        targets = targets[:args.max_models]
        log.info(f"  --max-models {args.max_models} 적용 → {len(targets)} 모델")

    if not targets:
        log.warning("  재학습 대상 없음 (모두 OK!)")
        return 0

    # ─ R1 (data) ───────────────────────────────────────
    if args.regenerate_phase1:
        phase1 = regenerate_phase1()
    else:
        phase1 = load_phase1_from_checkpoint()
    log.info(f"  R1 (data) OK: n={len(phase1['y_all'])}, "
             f"features={phase1['n_features']}, "
             f"split={phase1['n_train']}/{phase1['n_val']}/"
             f"{phase1.get('n_test', 0)}")

    # ─ Factory ─────────────────────────────────────────
    factories = build_factories(targets)
    log.info(f"  Factory 로드: {len(factories)}/{len(targets)} 모델")

    # ─ 재학습 루프 ──────────────────────────────────────
    output_dir = Path(args.output_dir) if args.output_dir else get_results_dir() / "per_model_optimal_v2"
    results: dict = {}
    t_total = time.time()
    for i, mname in enumerate(targets, 1):
        log.info("")
        log.info(f"═══ [{i}/{len(targets)}] {mname} ═══")
        if mname not in factories:
            log.warning(f"  [{mname}] factory 없음 — skip")
            continue
        res = retrain_one(mname, factories[mname], phase1, output_dir)
        results[mname] = res
        gc.collect()

    elapsed_total = time.time() - t_total
    log.info("")
    log.info(f"═══ 재학습 완료 ({elapsed_total/60:.1f}분) ═══")

    # 요약 저장
    summary_p = output_dir / "_retrain_summary.json"
    summary = {
        "total_attempts": len(targets),
        "success": [m for m, r in results.items() if "error" not in r],
        "failed": [m for m, r in results.items() if "error" in r],
        "elapsed_total_s": elapsed_total,
        "env": {
            k: os.environ.get(k) for k in (
                "MPH_GROUPED_PREPROC", "MPH_BEST_BY", "MPH_STABLE_TRANSFORMS",
                "MPH_ADVANCED_FEATURES", "MPH_PRUNER", "MPH_PRESET")
        },
    }
    summary_p.write_text(json.dumps(summary, indent=2, default=str))
    log.info(f"  요약: {summary_p}")
    log.info(f"  성공: {len(summary['success'])} / {len(targets)}")
    if summary["failed"]:
        log.warning(f"  실패: {summary['failed']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
