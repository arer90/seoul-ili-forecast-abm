"""문제 모델 자동 탐지 (2026-04-28).

R9 (per_model_optimize) 의 per_model_optimal/*.json + champion_log.json 결과를 읽고
다음 기준으로 문제 모델을 식별:

  S0 (severe):
    - test_r2 가 NaN/Inf
    - test_r2 < 0 (mean baseline 보다 못함)
    - test_predictions 에 NaN/Inf 포함
    - test_predictions 절대값 max > 100 (ILI 도메인 outlier)

  S1 (moderate):
    - test_r2 < 0.7  (baseline 0.85 대비 크게 낮음)
    - test_wis > 3 × val_wis (severe overfitting)

  S2 (mild):
    - test_r2 in [0.7, 0.8) (개선 여지 있음)
    - val_wis << test_wis 인 경우 (val/test gap 큼)

출력: simulation/results/problem_models_audit.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from simulation.config_global import GLOBAL  # SSOT (2026-05-28)
from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# 진단 기준
# ════════════════════════════════════════════════════════════════
PROBLEM_THRESHOLDS = {
    "test_r2_floor": 0.0,         # S0: 음수 R²
    "test_r2_low": 0.7,           # S1: 0.7 미만 = baseline (0.85) 대비 낮음
    "test_r2_mild": 0.8,          # S2: 0.8 미만 = 개선 여지
    "val_test_gap_severe": 3.0,   # test_wis > 3× val_wis = severe overfit
    "val_test_gap_mild": 2.0,     # 2x = mild overfit
    "extreme_pred_abs": 100.0,    # |predictions| > 100 = ILI 도메인 이상
    # ── audit Stage 3.3 (cascade #4, 2026-05-27) — champion / MCS cutoff (기존 R² 보강).
    # backward-compat — flag False default 시 기존 behavior 유지. env override:
    #   MPH_AUDIT_USE_CHAMPION_GATE=1 → champion_eligible 고려
    #   MPH_AUDIT_USE_MCS=1           → MCS_{90} 비-membership → S1+
    # (4-criteria/g175 strict cutoff 제거 2026-06-05 — champion = best-WIS.)
    "use_champion_gate":   False,
    "use_mcs_membership":  False,
}


def _audit_severity_bump(model_metrics: dict, base_severity: str) -> tuple[str, list[str]]:
    """audit Stage 3.3 — champion / MCS cutoff 으로 severity bump (g175 제거 2026-06-05; 현재 미사용).

    Args:
        model_metrics: per-model JSON dict (from R9 per_model_optimize result).
        base_severity: 기존 R²-based 진단 ("OK", "S0", "S1", "S2").

    Returns:
        (new_severity, additional_reasons)
    """
    reasons = []
    severity = base_severity

    use_champion = (PROBLEM_THRESHOLDS.get("use_champion_gate", False)
                    or GLOBAL.ops.audit_use_champion_gate)
    use_mcs = (PROBLEM_THRESHOLDS.get("use_mcs_membership", False)
               or GLOBAL.ops.audit_use_mcs)

    if use_champion:
        eligible = model_metrics.get("champion_eligible")
        if eligible is False:
            failed = []
            for k in ["alert_f1_pass", "lead_time_pass", "picp95_ci_lower_pass"]:
                if model_metrics.get(k) is False:
                    failed.append(k.replace("_pass", ""))
            if failed:
                reasons.append(f"champion_gate_fail ({'/'.join(failed)})")
                if severity == "OK":
                    severity = "S1"

    if use_mcs:
        mcs_member = model_metrics.get("mcs_90_member")
        if mcs_member is False:
            reasons.append("not in MCS_{90}")
            if severity == "OK":
                severity = "S2"

    return severity, reasons


def _check_predictions_finite(preds: list) -> tuple[bool, str]:
    """예측 NaN/Inf/extreme 검사."""
    if preds is None or len(preds) == 0:
        return False, "no_predictions"
    arr = np.asarray(preds, dtype=np.float64)
    if not np.isfinite(arr).all():
        return False, f"NaN/Inf in {(~np.isfinite(arr)).sum()}/{len(arr)} predictions"
    max_abs = float(np.max(np.abs(arr)))
    if max_abs > PROBLEM_THRESHOLDS["extreme_pred_abs"]:
        return False, f"extreme |pred|={max_abs:.1f} (>100)"
    return True, ""


def diagnose_one_model(json_path: Path) -> dict[str, Any]:
    """단일 모델 결과 JSON 진단."""
    try:
        d = json.loads(json_path.read_text())
    except Exception as e:
        return {"error": f"JSON 읽기 실패: {e}", "severity": "S0"}

    name = d.get("model") or json_path.stem
    val_m = d.get("val_metrics", {}) or {}
    test_m = d.get("test_metrics", {}) or {}
    val_wis = val_m.get("wis")
    test_wis = test_m.get("wis")
    test_r2 = test_m.get("r2")
    test_mae = test_m.get("mae")
    test_n = test_m.get("n", 0)
    preds = d.get("refit_test_predictions") or []
    config = d.get("best_config", {}) or {}

    reasons: list[str] = []
    severity = "OK"

    # S0 — severe (definitely problem)
    if test_r2 is None or not np.isfinite(test_r2 if isinstance(test_r2, (int, float)) else float("nan")):
        reasons.append("test_r2_missing_or_nan")
        severity = "S0"
    else:
        if test_r2 < PROBLEM_THRESHOLDS["test_r2_floor"]:
            reasons.append(f"test_r2_negative ({test_r2:.3f})")
            severity = "S0"

    pred_ok, pred_reason = _check_predictions_finite(preds)
    if not pred_ok:
        reasons.append(f"pred_problem: {pred_reason}")
        severity = "S0"

    # S1 — moderate (worth retraining)
    if test_r2 is not None and isinstance(test_r2, (int, float)) and np.isfinite(test_r2):
        if PROBLEM_THRESHOLDS["test_r2_floor"] <= test_r2 < PROBLEM_THRESHOLDS["test_r2_low"]:
            reasons.append(f"test_r2_low ({test_r2:.3f} < 0.7)")
            if severity == "OK":
                severity = "S1"

    # G-156 (사용자 명시 2026-05-02): val R²/val_wis 무시 — test 만 평가
    # 이전: val_wis vs test_wis gap 으로 overfit 판정 → val small (n=27) noise 영향
    # 수정: test 직접 평가 (R², MAPE, WIS, PICP) 만 사용. val_test_gap 검사 제거.
    # 단 val_wis 정보는 reasons 에 단순 기록 (참고용).
    if GLOBAL.filter.use_val_test_gap:
        # 옛 동작 유지하려면 env=1
        if val_wis is not None and test_wis is not None and val_wis > 0:
            gap = float(test_wis) / float(val_wis)
            if gap >= PROBLEM_THRESHOLDS["val_test_gap_severe"]:
                reasons.append(f"severe_val_test_gap (test/val WIS = {gap:.2f}x)")
                if severity == "OK":
                    severity = "S1"
            elif gap >= PROBLEM_THRESHOLDS["val_test_gap_mild"]:
                reasons.append(f"mild_val_test_gap (test/val WIS = {gap:.2f}x)")
                if severity == "OK":
                    severity = "S2"

    # S2 — mild (improvement room)
    if test_r2 is not None and isinstance(test_r2, (int, float)) and np.isfinite(test_r2):
        if (PROBLEM_THRESHOLDS["test_r2_low"] <= test_r2
                < PROBLEM_THRESHOLDS["test_r2_mild"] and severity == "OK"):
            reasons.append(f"test_r2_mild ({test_r2:.3f} < 0.8)")
            severity = "S2"

    return {
        "model": name,
        "severity": severity,
        "reasons": reasons,
        "val_wis": float(val_wis) if val_wis is not None else None,
        "test_wis": float(test_wis) if test_wis is not None else None,
        "test_mae": float(test_mae) if test_mae is not None else None,
        "test_r2": float(test_r2) if isinstance(test_r2, (int, float)) else None,
        "test_n": int(test_n),
        "config": {
            "transform": config.get("transform"),
            "scaler": config.get("scaler"),
            "n_features": config.get("n_features"),
        },
        "champion_decision": d.get("champion_decision"),
    }


def run_audit(
    per_model_dir: Optional[Path] = None,  # None → get_results_dir()/per_model_optimal (MPH_OUTPUT_ROOT)
    output_path: Optional[Path] = None,
) -> dict:
    """전체 audit + 결과 저장."""
    if per_model_dir is None:
        per_model_dir = get_results_dir() / "per_model_optimal"
    if output_path is None:  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
        output_path = get_results_dir() / "problem_models_audit.json"
    if not per_model_dir.exists():
        log.error(f"  per_model_dir 없음: {per_model_dir}")
        return {"error": "per_model_dir not found"}

    files = sorted(per_model_dir.glob("*.json"))
    # 메타데이터 파일 제외 ('summary', '_*.json' 등)
    files = [f for f in files
             if not f.stem.startswith("_") and f.stem.lower() != "summary"]
    if not files:
        log.error(f"  per_model_optimal 비어있음: {per_model_dir}")
        return {"error": "no per_model results"}

    log.info(f"  검사 대상: {len(files)} 모델 (메타데이터 제외)")
    diagnoses = [diagnose_one_model(f) for f in files]

    by_severity = {"S0": [], "S1": [], "S2": [], "OK": []}
    for d in diagnoses:
        by_severity.setdefault(d["severity"], []).append(d)

    summary = {
        "n_total": len(diagnoses),
        "n_S0_severe": len(by_severity["S0"]),
        "n_S1_moderate": len(by_severity["S1"]),
        "n_S2_mild": len(by_severity["S2"]),
        "n_OK": len(by_severity["OK"]),
        "thresholds": PROBLEM_THRESHOLDS,
        "by_severity": by_severity,
        "retrain_candidates": (
            [d["model"] for d in by_severity["S0"]]
            + [d["model"] for d in by_severity["S1"]]
        ),
        "consider_retrain": [d["model"] for d in by_severity["S2"]],
    }

    output_path.write_text(json.dumps(summary, indent=2, default=str))
    log.info(f"  저장: {output_path}")

    log.info("")
    log.info("══════ 요약 ══════")
    log.info(f"  S0 (severe):    {summary['n_S0_severe']:3d} 모델 — 반드시 재학습")
    log.info(f"  S1 (moderate):  {summary['n_S1_moderate']:3d} 모델 — 재학습 권장")
    log.info(f"  S2 (mild):      {summary['n_S2_mild']:3d} 모델 — 검토 후 재학습")
    log.info(f"  OK:             {summary['n_OK']:3d} 모델 — 그대로 유지")
    log.info("")
    log.info(f"  재학습 후보 (S0+S1): {len(summary['retrain_candidates'])} 모델")
    for m in summary["retrain_candidates"][:20]:
        d = next((x for x in diagnoses if x["model"] == m), None)
        if d:
            log.info(f"    [{d['severity']}] {m:<25} R²={d['test_r2']} reasons={d['reasons']}")

    return summary


if __name__ == "__main__":
    import sys
    out = run_audit()
    sys.exit(0 if "error" not in out else 1)
