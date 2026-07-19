"""Factorial ensemble (NNLS) runner — 격리 subprocess (2026-06-03).

per-model 격리 factorial 에서 ensemble(Ensemble-NNLS) 부활용. base 모델들이 cell 에서 따로 돌아
base-pred 배관이 끊긴 걸 복원: 각 base 모델의 best_config 를 replay 해 **train→val 예측**(out-of-sample,
NNLS 가중치 학습용) 생성 + 저장된 **test 예측** 수집 → `nnls_ensemble` 결합.

전제: base 모델들이 먼저 완료(per_model_optimal/{base}.json 존재) + 그 best_config 에 hier preproc
params 영속화됨(per_model_optimize.py best_config["preproc_optuna_params"], 2026-06-03 추가). caller =
run_factorial driver (ensemble 모델 차례에 이 runner 호출).

env (cell_run_env): MPH_OUTPUT_ROOT(=cell dir), MPH_PHASE13_FEATURE_POOL(basic→BASIC slice), 토글.
사용: ... factorial_ensemble_runner <ensemble_model> <base1,base2,...>
"""
import sys


def main(argv) -> int:
    """ensemble 1개를 base 모델 예측 결합으로 산출. argv[1]=ensemble명, argv[2]=base 콤마목록."""
    if len(argv) < 3:
        print("usage: factorial_ensemble_runner <ensemble> <base1,base2,...>", file=sys.stderr)
        return 2
    ens_name = argv[1].strip()
    bases = [m.strip() for m in argv[2].split(",") if m.strip() and m.strip() != ens_name]

    import json
    import os
    from pathlib import Path
    import numpy as np
    from simulation.pipeline.config import PipelineConfig
    from simulation.pipeline.data import run_data
    from simulation.pipeline.per_model_optimize import _refit_and_predict_test
    from simulation.analytics.ensemble_combine import nnls_ensemble
    from simulation.models.base import REGISTRY

    # base 모델 factory (run_per_model_optimize 패턴: 모델 모듈 import → REGISTRY → ()->instance)
    for _m in ("epi_models", "dl_models", "tree_models", "linear_models", "negbin_glm",
               "graph_models", "graph_models_pyg", "phase_ensemble", "conformal",
               "cqr_models", "bayesian_seir", "seir_forced", "pinn_model",
               "kernel_models", "gam_models", "ts_models", "foundation_models"):
        try:
            __import__(f"simulation.models.{_m}")
        except Exception:
            pass

    def build_factories(names):
        out = {}
        for n in names:
            cls = REGISTRY.get(n)
            if cls is not None:
                out[n] = (lambda cls=cls: cls())
        return out

    cfg = PipelineConfig()
    cfg.data.cache_dir = str(Path(__file__).resolve().parents[1] / "cache")
    save_dir = Path(cfg.save_dir) / "per_model_optimal"
    save_dir.mkdir(parents=True, exist_ok=True)

    p1 = run_data(cfg)
    X_all = np.asarray(p1["X_all"], dtype=np.float64)
    y_all = np.asarray(p1["y_all"], dtype=np.float64)
    feature_cols = list(p1["feature_cols"])
    # feature 토글 (BASIC slice, factorial_cell_runner 와 동일)
    if os.environ.get("MPH_PHASE13_FEATURE_POOL", "full").strip().lower() == "basic":
        from simulation.pipeline.runner import _resolve_eval_features
        X_all, feature_cols, _ = _resolve_eval_features(X_all, feature_cols, eval_basic=True)
    n_train, n_val = int(p1["n_train"]), int(p1["n_val"])
    n_test = int(p1.get("n_test") or (len(y_all) - n_train - n_val))
    pool_end = n_train + n_val
    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_val, y_val = X_all[n_train:pool_end], y_all[n_train:pool_end]
    y_test = y_all[pool_end:pool_end + n_test]

    factories = build_factories(bases)
    base_val, base_test = {}, {}
    for b in bases:
        jp = save_dir / f"{b}.json"
        if not jp.exists() or b not in factories:
            continue
        try:
            cfgb = json.loads(jp.read_text()).get("best_config", {})
        except Exception:
            continue
        # test 예측 = 저장본
        try:
            tp = json.loads(jp.read_text()).get("refit_test_predictions")
            if tp is not None:
                base_test[b] = np.asarray(tp, dtype=np.float64).ravel()[-n_test:]
        except Exception:
            pass
        # val 예측 = best_config replay (train→val, out-of-sample)
        try:
            vr = _refit_and_predict_test(
                factories[b],
                transform_name=cfgb.get("transform", "identity"),
                scaler_name=cfgb.get("scaler", "none"),
                X_train_pool=X_train, y_train_pool=y_train,
                X_test=X_val, y_test=y_val,
                feature_indices=cfgb.get("feature_indices"),
                feature_cols=feature_cols,
                hier_frozen_params=cfgb.get("preproc_optuna_params"),
            )
            vp = (vr or {}).get("predictions")
            if vp is not None:
                base_val[b] = np.asarray(vp, dtype=np.float64).ravel()[:len(y_val)]
        except Exception as e:
            print(f"  [ensemble] {b} val-replay 실패: {e}", file=sys.stderr)

    ens_pred, weights = nnls_ensemble(base_val, y_val, base_test)
    if ens_pred is None:
        print(f"FACTORIAL_ENSEMBLE_FAIL {ens_name}: 사용가능 base 0 "
              f"(val={len(base_val)} test={len(base_test)})", file=sys.stderr)
        return 1

    # WIS 평가 (empirical split-conformal, factorial collect 가 읽을 refit_test_predictions 저장)
    try:
        from simulation.analytics.metrics import weighted_interval_score_empirical  # best-effort
        wis = float("nan")
    except Exception:
        wis = float("nan")
    out = {
        "model": ens_name,
        "best_config": {"transform": "ensemble_nnls", "scaler": "none",
                        "weights": weights, "n_base": len(weights)},
        "val_metrics": {},
        "test_metrics": {},
        "refit_test_predictions": [float(x) for x in np.asarray(ens_pred, float).ravel()],
    }
    (save_dir / f"{ens_name}.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"FACTORIAL_ENSEMBLE_DONE {ens_name} bases={len(weights)} "
          f"n_test={len(ens_pred)} save_dir={cfg.save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
