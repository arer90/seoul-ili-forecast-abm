"""전 모델 baseline(R2, BASIC x/y) 전수 감사 — 음수 R² 모델 식별 (loop-engineering 입력).

사용자 철학(2026-06-19): baseline = 모든 모델 기본 x/y 로 양수 floor(약해도 OK), R9 = transform/feature/
HP 최적화. classic-ts(rolling G-321) 외 foundation/pf/deep 도 양수여야. 이 스크립트가 ground-truth
baseline R² 를 산출 → 음수 목록을 진단 workflow 가 소비.

faithful = 파이프라인 run_baseline 그대로(BASIC feature 내부선택, raw y, MultiModelRunner subprocess).
출력: simulation/results/_baseline_audit_r2.json + stdout 정렬 표.

run: KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 .venv/bin/python -m simulation.scripts.audit_baseline_all
"""
import json
import os
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("MPH_EVAL_FEATURES", "basic")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")


def main():
    from simulation.pipeline.config import PipelineConfig
    from simulation.pipeline.data import run_data
    from simulation.pipeline.baseline import run_baseline

    cfg = PipelineConfig()
    try:
        cfg.optuna.mode = "none"   # baseline only (no Optuna)
    except Exception:
        pass

    d = run_data(cfg)
    print(f"[audit] data: n={d['n']} feat={d['n_features']}", flush=True)
    baseline = run_baseline(d["X_all"], d["y_all"], d["feature_cols"], cfg)

    # run_baseline 반환 = {"model_results": runner_result, ...}; runner_result 안에 individual_results.
    _mr = baseline.get("model_results", {})
    _ind = _mr.get("individual_results", {}) if isinstance(_mr, dict) else {}
    out = {}
    for name, r in _ind.items():
        tm = (r.get("test_metrics") or {})
        out[name] = {
            "r2": tm.get("r2"),
            "rmse": tm.get("rmse"),
            "error": r.get("error"),
            "category": r.get("category"),
        }

    os.makedirs("simulation/results", exist_ok=True)
    path = "simulation/results/_baseline_audit_r2.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)

    def _key(kv):
        v = kv[1]["r2"]
        return v if isinstance(v, (int, float)) else -999.0

    print(f"\n[audit] DONE {len(out)} models → {path}\n", flush=True)
    print(f"  {'model':24}{'baseline_r2':>12}  {'status'}")
    neg = []
    for name, v in sorted(out.items(), key=_key):
        r2 = v["r2"]
        if isinstance(r2, (int, float)):
            status = "✅" if r2 >= 0 else "❌ NEGATIVE"
            if r2 < 0:
                neg.append(name)
            print(f"  {name:24}{r2:>12.4f}  {status}")
        else:
            status = f"❌ ERROR: {v['error']}"
            neg.append(name)
            print(f"  {name:24}{'(none)':>12}  {status}")
    print(f"\n[audit] NEGATIVE/ERROR ({len(neg)}): {neg}", flush=True)


if __name__ == "__main__":
    main()
