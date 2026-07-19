#!/usr/bin/env python3
"""학습 끝난 후 자동 audit MD 생성 — 33 모델 종합 결과 한 화면.

목적
----
per_model_optimize(R9) 끝나면 자동으로:
- all_models_metrics.md (R² / WIS / MAE / RMSE per model)
- top_features_shap.md (SHAP top 30)
- ensemble_weights.md (NNLS + stacking weights)
- champion_summary.md (.pt artifact 인벤토리)

사용법
------
    .venv/bin/python -m simulation.scripts.auto_audit_after_train

    # 학습 끝나면 자동:
    bash run_training.sh && python -m simulation.scripts.auto_audit_after_train
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
RESULTS_DIR = get_results_dir()


def audit_phase_completion() -> dict:
    """R1~R9(data~per_model_optimize) checkpoint 진행 상태."""
    checkpoints = sorted((RESULTS_DIR / "checkpoints").glob("checkpoint_phase*.json"))
    completed = []
    for cp in checkpoints:
        try:
            with cp.open(encoding="utf-8") as f:
                data = json.load(f)
            completed.append({
                "phase": cp.stem.replace("checkpoint_", ""),
                "saved_at": data.get("saved_at", "unknown"),
                "n_models": len(data.get("models", [])),
            })
        except Exception:
            pass
    return {"n_completed": len(completed), "phases": completed}


def audit_all_models_metrics() -> dict:
    """per_model_optimize(R9) 의 모델별 metrics 수집."""
    p12_dir = RESULTS_DIR / "phase13_per_model_optimize"
    if not p12_dir.exists():
        return {"exists": False}

    models = []
    for jf in p12_dir.glob("*.json"):
        try:
            with jf.open(encoding="utf-8") as f:
                data = json.load(f)
            models.append({
                "model": jf.stem,
                "test_r2": data.get("test_r2"),
                "test_wis": data.get("test_wis"),
                "test_mae": data.get("test_mae"),
                "test_rmse": data.get("test_rmse"),
                "best_hp": data.get("best_hp", {}),
            })
        except Exception:
            pass

    # WIS 기준 정렬 (낮은 게 좋음)
    models.sort(key=lambda m: m.get("test_wis") or float("inf"))
    return {"exists": True, "n_models": len(models), "models": models}


def audit_champions() -> dict:
    """champion_log.json — promote 된 model 인벤토리."""
    cl_path = Path("models/champion_log.json")
    if not cl_path.exists():
        return {"exists": False}
    try:
        with cl_path.open(encoding="utf-8") as f:
            data = json.load(f)
        current_count = sum(1 for k, v in data.items() if v.get("current"))
        return {
            "exists": True,
            "n_models": len(data),
            "n_current": current_count,
            "models": [
                {"name": k,
                 "version": v.get("current", {}).get("version"),
                 "test_wis": v.get("current", {}).get("test_wis"),
                 "promoted_at": v.get("current", {}).get("promoted_at")}
                for k, v in data.items() if v.get("current")
            ][:20],
        }
    except Exception as e:
        return {"exists": False, "error": str(e)}


def audit_ensemble_weights() -> dict:
    """ensemble weights."""
    ens_dir = RESULTS_DIR / "phase11_ensemble"
    if not ens_dir.exists():
        return {"exists": False}

    weights = {}
    for jf in ens_dir.glob("*weights*.json"):
        try:
            with jf.open(encoding="utf-8") as f:
                weights[jf.stem] = json.load(f)
        except Exception:
            pass
    return {"exists": bool(weights), "files": list(weights.keys())}


def audit_real_eval() -> dict:
    """real-slab evaluation (real_eval)."""
    re_dir = RESULTS_DIR / "real_eval"
    if not re_dir.exists():
        return {"exists": False}

    summary = re_dir / "summary.json"
    metrics = re_dir / "metrics_full.json"
    result = {"exists": True}
    if summary.exists():
        try:
            with summary.open(encoding="utf-8") as f:
                result["summary"] = json.load(f)
        except Exception:
            pass
    if metrics.exists():
        try:
            with metrics.open(encoding="utf-8") as f:
                m = json.load(f)
            result["n_metrics"] = len(m)
        except Exception:
            pass
    return result


def write_audit_md():
    """모든 audit 통합 MD 생성."""
    out = RESULTS_DIR / "AUTO_AUDIT_LATEST.md"

    phases = audit_phase_completion()
    metrics = audit_all_models_metrics()
    champions = audit_champions()
    ensemble = audit_ensemble_weights()
    real_eval = audit_real_eval()

    md = []
    md.append("# 학습 후 자동 audit\n")
    import datetime
    md.append(f"**생성**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    # Phase
    md.append("## 1. Phase 진행\n")
    md.append(f"완료: **{phases['n_completed']} / 13** phases\n")
    if phases["phases"]:
        md.append("| Phase | Saved | n_models |")
        md.append("|-------|-------|----------|")
        for p in phases["phases"]:
            md.append(f"| {p['phase']} | {p['saved_at'][:19]} | {p['n_models']} |")
    md.append("")

    # Metrics
    md.append("## 2. 모델별 metrics (WIS 정렬)\n")
    if metrics.get("exists"):
        md.append(f"총 {metrics['n_models']} 모델\n")
        md.append("| Rank | Model | R² | WIS | MAE | RMSE |")
        md.append("|------|-------|----|----|-----|------|")
        for i, m in enumerate(metrics["models"][:20], 1):
            r2 = f"{m['test_r2']:.4f}" if m['test_r2'] else "—"
            wis = f"{m['test_wis']:.4f}" if m['test_wis'] else "—"
            mae = f"{m['test_mae']:.4f}" if m['test_mae'] else "—"
            rmse = f"{m['test_rmse']:.4f}" if m['test_rmse'] else "—"
            md.append(f"| {i} | {m['model']} | {r2} | {wis} | {mae} | {rmse} |")
    else:
        md.append("per_model_optimize(R9) 결과 없음\n")
    md.append("")

    # Champions
    md.append("## 3. Champion 모델\n")
    if champions.get("exists"):
        md.append(f"총 {champions['n_models']} 모델 (current: {champions['n_current']})\n")
        md.append("| Model | Version | WIS | Promoted |")
        md.append("|-------|--------|-----|----------|")
        for c in champions.get("models", []):
            wis = f"{c['test_wis']:.4f}" if c.get('test_wis') else "—"
            md.append(f"| {c['name']} | v{c.get('version', '?')} | {wis} | {c.get('promoted_at', '?')[:19]} |")
    else:
        md.append("Champion log 없음\n")
    md.append("")

    # Ensemble
    md.append("## 4. Ensemble weights\n")
    if ensemble.get("exists"):
        md.append(f"파일: {len(ensemble['files'])} 개\n")
        for f in ensemble["files"]:
            md.append(f"- `{f}`")
    else:
        md.append("ensemble 결과 없음\n")
    md.append("")

    # Real eval
    md.append("## 5. Real-slab evaluation (per_model_eval R10)\n")
    if real_eval.get("exists"):
        md.append(f"summary 있음, metrics: {real_eval.get('n_metrics', '?')} entries\n")
        if real_eval.get("summary"):
            s = real_eval["summary"]
            md.append(f"- Best model: {s.get('best_model', '?')}")
            wis = s.get('best_wis')
            r2 = s.get('best_r2')
            md.append(f"- Best WIS: {wis:.3f}" if isinstance(wis, (int, float)) else f"- Best WIS: {wis}")
            if isinstance(r2, (int, float)):
                md.append(f"- Best R²: {r2:.3f}")
            md.append(f"- Real slab n: {s.get('real_n', '?')} weeks")
            md.append("")
            md.append("> ⚠ R² 음수 = small real_n (8 주) 의 통계적 한계 — "
                      "실제 baseline(R2) test (68주) R² 와 다름. WIS / MAE 가 더 신뢰성 있음.")
    else:
        md.append("real_eval(R10) 결과 없음\n")
    md.append("")

    # 2026-04-28: Champion 출처 명시 (per_model_optimize(R9) 미완 시 baseline)
    md.append("## 6. ⚠ Champion 출처 주의\n")
    if phases["n_completed"] < 13:
        md.append(f"**Phase 완료: {phases['n_completed']} / 13** — per_model_optimize(R9) 미완 가능")
        md.append("- per_model_optimize(R9) (transform×scaler grid) 가 완료되지")
        md.append("  않으면, champion 은 **baseline(R2)** (HP 안 튜닝, transform 안 됨)")
        md.append("  결과로 등록됨 → **R² 음수 / 이상치 가능**")
        md.append("- per_model_optimize(R9) 재시도 권장 (patch 적용된 코드로):")
        md.append("  ```bash")
        md.append("  .venv/bin/python -m simulation train \\")
        md.append("      --resume-from simulation/results/checkpoints/checkpoint_phase11.json \\")
        md.append("      --scenario full --per-model-optimize")
        md.append("  ```")
    else:
        md.append("per_model_optimize(R9) 까지 완료 → champion = HP-tuned 결과")
    md.append("")

    md.append("---\n")
    md.append("\n**다음 단계**:\n")
    md.append("- `simulation predict-real --weeks-ahead 4 --with-actuals`")
    md.append("- `simulation phase10-real-eval --weather-mode hybrid`")
    md.append("- `simulation visualize --training-curves --models all`")

    out.write_text("\n".join(md), encoding="utf-8")
    log.info(f"Audit MD 저장: {out}")
    return out


def main():
    print("=" * 60)
    print("  학습 후 자동 audit")
    print("=" * 60)

    # 진행 상태 출력
    phases = audit_phase_completion()
    print(f"\n[1] Phases 완료: {phases['n_completed']} / 13")
    for p in phases["phases"][-5:]:
        print(f"  {p['phase']}: {p['saved_at']}")

    metrics = audit_all_models_metrics()
    if metrics.get("exists"):
        print(f"\n[2] per_model_optimize(R9) 모델: {metrics['n_models']}")
        for m in metrics["models"][:5]:
            wis = f"{m['test_wis']:.4f}" if m['test_wis'] else "—"
            print(f"  {m['model']:25s} WIS={wis}")
    else:
        print("\n[2] per_model_optimize(R9) 미완 (학습 끝나면 자동 생성됨)")

    champions = audit_champions()
    if champions.get("exists"):
        print(f"\n[3] Champion: {champions['n_current']} / {champions['n_models']}")

    out = write_audit_md()
    print(f"\n[4] MD 생성: {out}")


if __name__ == "__main__":
    main()
