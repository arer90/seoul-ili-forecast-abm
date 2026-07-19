"""Factorial ablation — single-cell phase-13 runner (격리 subprocess entry, 2026-06-02).

cell당 **별도 subprocess** 로 실행해야 하는 이유: GLOBAL config = frozen dataclass(재현성 #5) →
in-process 변이 불가. env(MPH_PREPROC_OPTUNA/HP_OPTUNA_TRIALS/PHASE13_FEATURE_POOL)는 GLOBAL
**생성(import) 시점**에만 읽힘 → fresh process 만이 cell 토글을 반영. (ablation_factorial 모듈 참조.)

이 스크립트 = 한 cell 의 phase-13(per_model_optimize) 만 panel 에 실행. 상위 phases(1-12) 는
재현성-격리상 불필요 — phase-13 은 phase1(data) 만 입력. per_model_optimal/{model}.json 에
refit_test_predictions 기록(상위 orchestrator 가 수집 → factorial_effects).

env (cell_run_env 가 설정):
  MPH_OUTPUT_ROOT=<cell dir>   ← main run 과 격리 (save_dir/Optuna/per_model_optimal)
  MPH_PREPROC_OPTUNA / MPH_HP_OPTUNA_TRIALS / MPH_PHASE13_FEATURE_POOL  ← cell 토글
  OPTUNA_ISOLATE=1             ← G-158 child memory 회수

사용:
  MPH_OUTPUT_ROOT=/tmp/cell_101 MPH_PREPROC_OPTUNA=1 MPH_HP_OPTUNA_TRIALS=0 \
    MPH_PHASE13_FEATURE_POOL=basic OPTUNA_ISOLATE=1 \
    .venv/bin/python -m simulation.scripts.factorial_cell_runner XGBoost,DNN
"""
import sys


def main(argv) -> int:
    """panel 에 cell phase-13 실행. argv[1] = 콤마구분 model 이름.

    Returns: 0 성공, 2 인자오류, 1 실행오류 (loud crash 대신 코드 — orchestrator 가 cell skip).
    """
    if len(argv) < 2 or not argv[1].strip():
        print("usage: factorial_cell_runner <model1,model2,...>", file=sys.stderr)
        return 2
    panel = [m.strip() for m in argv[1].split(",") if m.strip()]

    import os
    from pathlib import Path
    from simulation.pipeline.config import PipelineConfig
    from simulation.pipeline.data import run_data
    from simulation.pipeline.per_model_optimize import run_per_model_optimize

    cfg = PipelineConfig()                 # GLOBAL.paths.output_root(MPH_OUTPUT_ROOT) → save_dir(격리)
    cfg._selected_models = panel           # --models 와 동일 효과 (L2615 restrict)
    cfg.per_model_optimize = True          # ★ phase-13 활성 (CLI --per-model-optimize 동등; 없으면 skipped:disabled)
    try:
        cfg.split.per_model_optimize = True   # guard L2420 가 둘 다 검사 (방어)
    except Exception:
        pass
    # 공유 FE 캐시 강제 — cells 가 main 의 캐시(simulation/cache) hit → 80M행 DB 재로드/FE 재계산 회피.
    # 출력(save_dir/Optuna)은 MPH_OUTPUT_ROOT 격리 유지. 동일 feature config → read-only hit (main 무영향).
    cfg.data.cache_dir = str(Path(__file__).resolve().parents[1] / "cache")
    phase1 = run_data(cfg)

    # feature 토글 (runner A1 flag 복제, L1094): BASIC 면 phase1 을 BASIC(lag+계절성 13) 슬라이스 후 13.
    # factorial_cell_runner 는 run_per_model_optimize 직접호출 → runner A1 flag 우회 → 여기서 명시 복제.
    if os.environ.get("MPH_PHASE13_FEATURE_POOL", "full").strip().lower() == "basic":
        from simulation.pipeline.runner import _resolve_eval_features
        _Xb, _fcb, _ = _resolve_eval_features(phase1["X_all"], phase1["feature_cols"], eval_basic=True)
        phase1 = {**phase1, "X_all": _Xb, "feature_cols": _fcb}

    res = run_per_model_optimize(phase1, {}, cfg)   # phase-13: per_model_optimal/{model}.json 기록
    n = len(res.get("per_model_configs", {}))
    print(f"FACTORIAL_CELL_DONE models={n} save_dir={cfg.save_dir} "
          f"feature_pool={os.environ.get('MPH_PHASE13_FEATURE_POOL', 'full')} "
          f"n_features={phase1['X_all'].shape[1]}")
    return 0 if n > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
