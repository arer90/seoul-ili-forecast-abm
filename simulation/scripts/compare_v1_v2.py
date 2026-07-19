"""v1 (원본 학습) vs v2 (재학습) 비교 리포트 (2026-04-28).

디렉토리:
  v1: simulation/results/per_model_optimal/
  v2: simulation/results/per_model_optimal_v2/

산출물:
  simulation/results/v1_vs_v2_comparison.{json,md}

비교 metric:
  • test R² / WIS / MAE / RMSE
  • val→test gap (overfitting)
  • config (transform, scaler, n_features)
  • predictions 의 NaN/Inf 검증
  • Decision: PROMOTE (v2 better) / KEEP_V1 (v1 better) / TIE

Decision 기준:
  • PROMOTE_V2: v2.test_r2 > v1.test_r2 + 0.05  AND  v2.test_wis < v1.test_wis * 0.9
  • PROMOTE_V2_R2: v2.test_r2 > v1.test_r2 + 0.10  (R² 큰 개선만 있어도 promote)
  • KEEP_V1: v1.test_r2 > v2.test_r2 + 0.02  (v1 가 분명히 좋을 때)
  • TIE: 그 외

CLI:
  .venv/bin/python -m simulation.scripts.compare_v1_v2
  .venv/bin/python -m simulation.scripts.compare_v1_v2 --apply  # promote 자동 적용
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# Decision 기준
# ════════════════════════════════════════════════════════════════
DECISION_THRESHOLDS = {
    "promote_r2_delta": 0.05,         # v2.r2 - v1.r2 > 0.05 → 충분한 개선
    "promote_r2_big_delta": 0.10,     # v2.r2 - v1.r2 > 0.10 → 큰 개선 (WIS 무시)
    "promote_wis_ratio": 0.90,        # v2.wis < v1.wis × 0.9 → WIS 도 개선
    "keep_v1_r2_delta": 0.02,         # v1.r2 - v2.r2 > 0.02 → v1 유지
}


def _load_one(json_path: Path) -> Optional[dict]:
    if not json_path.exists():
        return None
    try:
        return json.loads(json_path.read_text())
    except Exception as e:
        log.warning(f"  로드 실패: {json_path.name}: {e}")
        return None


def _extract_metrics(d: dict) -> dict:
    """JSON 에서 핵심 metric 추출."""
    val = d.get("val_metrics") or {}
    test = d.get("test_metrics") or {}
    cfg = d.get("best_config") or {}
    preds = d.get("refit_test_predictions") or []
    arr = np.asarray(preds, dtype=np.float64) if preds else np.array([])
    pred_finite = bool(np.isfinite(arr).all()) if arr.size else None
    pred_max_abs = float(np.max(np.abs(arr))) if arr.size else None

    return {
        "val_wis": val.get("wis"),
        "val_mae": val.get("mae"),
        "test_wis": test.get("wis"),
        "test_mae": test.get("mae"),
        "test_rmse": test.get("rmse"),
        "test_r2": test.get("r2"),
        "test_n": test.get("n"),
        "transform": cfg.get("transform"),
        "scaler": cfg.get("scaler"),
        "n_features": cfg.get("n_features"),
        "pred_finite": pred_finite,
        "pred_max_abs": pred_max_abs,
    }


def _decide(v1: dict, v2: dict) -> tuple[str, str]:
    """v1, v2 metric dict 비교 → (decision, reason).

    audit Stage 3.3 (Task #21, 2026-05-27) 권장 보강:
        - paired DM test on WIS series (Diebold & Mariano 1995)
        - Cohen's d (effect size)
        - 95% CI lower bound > 0 → BETTER
        - cutoff: d ≥ 0.2 AND CI_lower > 0
    실제 paired DM 은 audit script 가 별도 호출 (predictions array 필요).
    본 함수는 metric dict 만으로 결정 — point estimate based fallback.
    """
    r1 = v1.get("test_r2")
    r2 = v2.get("test_r2")
    w1 = v1.get("test_wis")
    w2 = v2.get("test_wis")

    # v2 또는 v1 의 R²/WIS 가 NaN/None 이면 비교 불가
    if r2 is None or not isinstance(r2, (int, float)) or not np.isfinite(r2):
        return "RETRAIN_FAILED", "v2.r2 is NaN/None"
    if r1 is None or not isinstance(r1, (int, float)) or not np.isfinite(r1):
        return "PROMOTE_V2", "v1.r2 is NaN/None, v2 is finite"

    # v2 의 predictions 가 폭주
    if v2.get("pred_finite") is False:
        return "RETRAIN_FAILED", "v2 predictions have NaN/Inf"
    if v2.get("pred_max_abs") and v2["pred_max_abs"] > 100:
        return "RETRAIN_PARTIAL", f"v2 pred_max_abs={v2['pred_max_abs']:.1f} (>100)"

    delta_r2 = r2 - r1
    delta_wis = (w2 - w1) if (w1 is not None and w2 is not None) else None

    # audit Stage 3.3 — paired DM 결과가 v1/v2 dict 에 있으면 우선 적용
    # expected keys: "paired_dm_p_value", "wis_diff_ci_lower", "cohens_d"
    if v2.get("paired_dm_p_value") is not None and v2.get("wis_diff_ci_lower") is not None:
        p_dm = float(v2["paired_dm_p_value"])
        ci_lo = float(v2["wis_diff_ci_lower"])
        d_cohen = float(v2.get("cohens_d", 0.0))
        # WIS lower = better: ΔWIS = v1 - v2 (positive = v2 better)
        if p_dm < 0.05 and ci_lo > 0 and d_cohen >= 0.2:
            return "PROMOTE_V2", (
                f"paired DM significant (p={p_dm:.4f}) + "
                f"CI lower>0 ({ci_lo:.3f}) + Cohen's d={d_cohen:.2f}"
            )
        elif p_dm >= 0.05 and abs(d_cohen) < 0.2:
            return "TIE", (
                f"paired DM not significant (p={p_dm:.4f}) + small effect d={d_cohen:.2f}"
            )
        elif d_cohen < -0.2:
            return "KEEP_V1", f"v1 better (Cohen's d={d_cohen:.2f})"

    # Fallback: point estimate based (기존 logic)
    if delta_r2 > DECISION_THRESHOLDS["promote_r2_big_delta"]:
        return "PROMOTE_V2", f"R² 개선 큼 (Δ={delta_r2:+.3f})"

    if delta_r2 > DECISION_THRESHOLDS["promote_r2_delta"]:
        if w1 and w2 and w2 < w1 * DECISION_THRESHOLDS["promote_wis_ratio"]:
            return "PROMOTE_V2", f"R²+WIS 모두 개선 (ΔR²={delta_r2:+.3f}, ΔWIS={w2-w1:+.3f})"
        return "PROMOTE_V2_R2_ONLY", f"R² 개선 (Δ={delta_r2:+.3f}), WIS 비슷"

    if -DECISION_THRESHOLDS["keep_v1_r2_delta"] < delta_r2 < DECISION_THRESHOLDS["promote_r2_delta"]:
        return "TIE", f"비슷 (ΔR²={delta_r2:+.3f})"

    return "KEEP_V1", f"v1 가 더 좋음 (ΔR²={delta_r2:+.3f})"


def _extract_fingerprint(d: dict) -> Optional[str]:
    """Extract combined_sha256 from a per-model result JSON, or None if absent."""
    if not isinstance(d, dict):
        return None
    fp = d.get("db_fingerprint") or (d.get("summary", {}) or {}).get("db_fingerprint")
    if isinstance(fp, dict):
        return fp.get("combined_sha256")
    return None


def run_comparison(
    v1_dir: Optional[Path] = None,  # None → get_results_dir()/per_model_optimal (MPH_OUTPUT_ROOT)
    v2_dir: Optional[Path] = None,  # None → get_results_dir()/per_model_optimal_v2
    output_md: Optional[Path] = None,
    output_json: Optional[Path] = None,
    require_same_data: bool = False,
) -> dict:
    """v1 vs v2 비교 + 리포트.

    Args:
        v1_dir:           Directory with baseline (v1) per-model JSON results.
        v2_dir:           Directory with retrain (v2) per-model JSON results.
        output_md:        Markdown report path.
        output_json:      JSON summary path.
        require_same_data: If True, abort comparison for models where v1/v2
                          DB fingerprints differ (they trained on different data
                          snapshots — comparison is not apples-to-apples).
                          Default False: warn only.

    Returns:
        Summary dict with ``by_decision``, ``rows``, and ``data_parity_warnings``.
    """
    if output_md is None:  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
        output_md = get_results_dir() / "v1_vs_v2_comparison.md"
    if output_json is None:
        output_json = get_results_dir() / "v1_vs_v2_comparison.json"
    if v1_dir is None:
        v1_dir = get_results_dir() / "per_model_optimal"
    if v2_dir is None:
        v2_dir = get_results_dir() / "per_model_optimal_v2"
    if not v1_dir.exists() or not v2_dir.exists():
        log.error(f"  디렉토리 없음: v1={v1_dir.exists()}, v2={v2_dir.exists()}")
        return {"error": "dir not found"}

    v2_files = sorted(v2_dir.glob("*.json"))
    if not v2_files:
        log.error(f"  v2 결과 없음: {v2_dir}")
        return {"error": "no v2 results"}

    data_parity_warnings: list[str] = []
    rows = []
    for vf in v2_files:
        if vf.name.startswith("_"):    # _retrain_summary.json 등 메타 skip
            continue
        name = vf.stem
        d2 = _load_one(vf)
        d1 = _load_one(v1_dir / f"{name}.json")
        if d2 is None:
            continue
        m2 = _extract_metrics(d2)
        m1 = _extract_metrics(d1) if d1 else None

        # G-235: DB fingerprint parity check
        fp1 = _extract_fingerprint(d1) if d1 else None
        fp2 = _extract_fingerprint(d2)
        parity_ok = True
        if fp1 and fp2 and fp1 != fp2:
            msg = (f"{name}: DB fingerprint mismatch (v1={fp1[:12]} ≠ v2={fp2[:12]}) "
                   f"— trained on different data snapshots")
            log.warning("  [compare] %s", msg)
            data_parity_warnings.append(msg)
            parity_ok = False
            if require_same_data:
                rows.append({
                    "model": name, "decision": "DATA_MISMATCH",
                    "reason": msg, "v1": m1, "v2": m2,
                    "data_parity_ok": False,
                })
                continue

        if m1 is None:
            decision, reason = "NEW_V2_ONLY", "v1 결과 없음"
        else:
            decision, reason = _decide(m1, m2)

        rows.append({
            "model": name,
            "decision": decision,
            "reason": reason,
            "v1": m1,
            "v2": m2,
            "data_parity_ok": parity_ok,
        })

    # ─ 요약 ─
    by_decision = {}
    for r in rows:
        by_decision.setdefault(r["decision"], []).append(r["model"])

    summary = {
        "n_compared": len(rows),
        "by_decision": {k: len(v) for k, v in by_decision.items()},
        "decision_lists": by_decision,
        "thresholds": DECISION_THRESHOLDS,
        "rows": rows,
        # G-235: data parity — list of models where v1/v2 used different DB snapshots
        "data_parity_warnings": data_parity_warnings,
        "n_data_parity_mismatches": len(data_parity_warnings),
    }
    output_json.write_text(json.dumps(summary, indent=2, default=str))
    log.info(f"  JSON: {output_json}")

    # ─ Markdown 리포트 ─
    md_lines = [
        "# v1 vs v2 모델 비교 리포트",
        "",
        f"**생성**: 2026-04-28  ·  **비교 모델**: {len(rows)} 개",
        "",
        "## 결정 요약",
        "",
        "| 결정 | 개수 | 의미 |",
        "|------|----:|------|",
        f"| **PROMOTE_V2** | {len(by_decision.get('PROMOTE_V2', []))} | v2 가 R²+WIS 모두 개선 → 승격 권장 |",
        f"| PROMOTE_V2_R2_ONLY | {len(by_decision.get('PROMOTE_V2_R2_ONLY', []))} | v2 R² 개선, WIS 는 비슷 |",
        f"| TIE | {len(by_decision.get('TIE', []))} | 비슷 (선택 자유) |",
        f"| **KEEP_V1** | {len(by_decision.get('KEEP_V1', []))} | v1 가 더 좋음 → v1 유지 |",
        f"| RETRAIN_FAILED | {len(by_decision.get('RETRAIN_FAILED', []))} | v2 에러/폭주 |",
        f"| RETRAIN_PARTIAL | {len(by_decision.get('RETRAIN_PARTIAL', []))} | v2 부분 성공 (조건부) |",
        f"| NEW_V2_ONLY | {len(by_decision.get('NEW_V2_ONLY', []))} | v1 결과 없음 |",
        "",
        "## 모델별 비교",
        "",
        "| 모델 | 결정 | v1 R² | v2 R² | ΔR² | v1 WIS | v2 WIS | v1 config | v2 config |",
        "|------|------|------:|------:|----:|------:|------:|-----------|-----------|",
    ]
    # decision priority sort
    DECISION_ORDER = {
        "PROMOTE_V2": 0, "PROMOTE_V2_R2_ONLY": 1, "TIE": 2,
        "KEEP_V1": 3, "NEW_V2_ONLY": 4,
        "RETRAIN_PARTIAL": 5, "RETRAIN_FAILED": 6,
    }
    rows_sorted = sorted(rows, key=lambda r: DECISION_ORDER.get(r["decision"], 99))
    def _fmt(v, fmt=".3f"):
        if v is None:
            return "n/a"
        if isinstance(v, (int, float)) and np.isfinite(v):
            return format(v, fmt)
        return str(v)

    for r in rows_sorted:
        v1 = r["v1"] or {}
        v2 = r["v2"]
        r1 = v1.get("test_r2")
        r2_v = v2.get("test_r2")
        if (isinstance(r1, (int, float)) and isinstance(r2_v, (int, float))
                and np.isfinite(r1) and np.isfinite(r2_v)):
            delta = f"{(r2_v - r1):+.3f}"
        else:
            delta = "n/a"

        r1_s = _fmt(r1, ".3f")
        r2_s = _fmt(r2_v, ".3f")
        w1_s = _fmt(v1.get("test_wis"), ".2f")
        w2_s = _fmt(v2.get("test_wis"), ".2f")
        cfg1 = f"{v1.get('transform')}/{v1.get('scaler')}"
        cfg2 = f"{v2.get('transform')}/{v2.get('scaler')}"

        md_lines.append(
            f"| {r['model']} | **{r['decision']}** | "
            f"{r1_s} | {r2_s} | {delta} | "
            f"{w1_s} | {w2_s} | {cfg1} | {cfg2} |"
        )
    md_lines.append("")
    md_lines.append("## Decision 기준")
    md_lines.append("")
    md_lines.append("```")
    for k, v in DECISION_THRESHOLDS.items():
        md_lines.append(f"  {k}: {v}")
    md_lines.append("```")
    md_lines.append("")
    md_lines.append("## 적용 방법")
    md_lines.append("")
    md_lines.append("```bash")
    md_lines.append("# PROMOTE_V2 모델만 자동 적용 (per_model_optimal/ 덮어쓰기, 백업 보관)")
    md_lines.append(".venv/bin/python -m simulation.scripts.compare_v1_v2 --apply")
    md_lines.append("```")
    output_md.write_text("\n".join(md_lines))
    log.info(f"  Markdown: {output_md}")

    log.info("")
    log.info("══════ 결정 분포 ══════")
    for k in DECISION_ORDER:
        n = len(by_decision.get(k, []))
        if n > 0:
            log.info(f"  {k:<22} {n:3d} 모델")

    return summary


def apply_promotions(
    summary: dict,
    v1_dir: Optional[Path] = None,  # None → get_results_dir()/per_model_optimal (MPH_OUTPUT_ROOT)
    v2_dir: Optional[Path] = None,  # None → get_results_dir()/per_model_optimal_v2
) -> int:
    """PROMOTE_V2 / PROMOTE_V2_R2_ONLY 모델을 v1 위치에 복사 (백업 보관)."""
    if v1_dir is None:
        v1_dir = get_results_dir() / "per_model_optimal"
    if v2_dir is None:
        v2_dir = get_results_dir() / "per_model_optimal_v2"
    promote_models = (
        summary.get("decision_lists", {}).get("PROMOTE_V2", [])
        + summary.get("decision_lists", {}).get("PROMOTE_V2_R2_ONLY", [])
    )
    if not promote_models:
        log.info("  적용할 PROMOTE 모델 없음")
        return 0

    backup_dir = v1_dir.parent / f"per_model_optimal_v1_backup_{int(__import__('time').time())}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"  v1 백업: {backup_dir}")

    n_applied = 0
    for m in promote_models:
        v1_p = v1_dir / f"{m}.json"
        v2_p = v2_dir / f"{m}.json"
        if not v2_p.exists():
            continue
        if v1_p.exists():
            shutil.copy(v1_p, backup_dir / f"{m}.json")
        shutil.copy(v2_p, v1_p)
        log.info(f"  ✓ {m} 승격 적용")
        n_applied += 1

    log.info(f"  적용 완료: {n_applied}/{len(promote_models)}")
    log.info(f"  롤백: per_model_optimal/ ← {backup_dir}")
    return n_applied


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="v1 vs v2 비교 리포트")
    parser.add_argument("--apply", action="store_true",
                         help="PROMOTE_V2 모델을 자동 적용 (v1 백업 + v2 복사)")
    parser.add_argument("--v1-dir", default=None,
                         help="default: get_results_dir()/per_model_optimal (MPH_OUTPUT_ROOT)")
    parser.add_argument("--v2-dir", default=None,
                         help="default: get_results_dir()/per_model_optimal_v2")
    args = parser.parse_args(argv)

    summary = run_comparison(
        v1_dir=Path(args.v1_dir) if args.v1_dir else None,
        v2_dir=Path(args.v2_dir) if args.v2_dir else None,
    )
    if "error" in summary:
        return 1

    if args.apply:
        log.info("")
        log.info("═══ PROMOTE_V2 적용 ═══")
        apply_promotions(
            summary,
            v1_dir=Path(args.v1_dir) if args.v1_dir else None,
            v2_dir=Path(args.v2_dir) if args.v2_dir else None,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
