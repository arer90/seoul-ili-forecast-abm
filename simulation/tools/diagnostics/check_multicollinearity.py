#!/usr/bin/env python3
"""다중공선성 (multicollinearity) 자동 검사 + 완화 도구.

목적
----
- Pearson correlation > threshold pair 자동 검출
- VIF (Variance Inflation Factor) — feature 별 다른 features 와의 R² 측정
- Condition number κ — design matrix 의 수치 안정성
- 자동 mitigation 제안 — perfect collinearity (corr=1.0) feature 제거

사용법
------
    .venv/bin/python -m simulation.tools.check_multicollinearity
    .venv/bin/python -m simulation.tools.check_multicollinearity --strict
    .venv/bin/python -m simulation.tools.check_multicollinearity --vif-threshold 10
    .venv/bin/python -m simulation.tools.check_multicollinearity --auto-fix
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# ══════════════════════════════════════════════════════════
# 1. VIF (Variance Inflation Factor)
# ══════════════════════════════════════════════════════════

def compute_vif(X: np.ndarray, j: int) -> float:
    """X[:, j] 의 VIF = 1 / (1 - R²(X[:, j] ~ X[:, others])).

    VIF > 10 = multicollinearity 의심
    VIF > 100 = severe multicollinearity (sklearn lstsq numerical 안전 한계)
    """
    others = [i for i in range(X.shape[1]) if i != j]
    Xo = X[:, others]
    y_j = X[:, j]
    try:
        coef, *_ = np.linalg.lstsq(Xo, y_j, rcond=None)
        pred = Xo @ coef
        ss_res = float(np.sum((y_j - pred) ** 2))
        ss_tot = float(np.sum((y_j - y_j.mean()) ** 2)) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        return 1.0 / max(1e-12, 1.0 - r2)
    except Exception:
        return float("nan")


def vif_table(X: np.ndarray, feature_cols: list[str],
              skip_const: bool = True) -> list[tuple[float, str, int]]:
    """모든 feature 의 VIF 계산. (vif, name, idx) 리스트, 내림차순."""
    results = []
    for i, name in enumerate(feature_cols):
        col = X[:, i]
        if skip_const and np.std(col) < 1e-10:
            continue
        v = compute_vif(X, i)
        if np.isfinite(v):
            results.append((float(v), name, i))
    results.sort(reverse=True)
    return results


# ══════════════════════════════════════════════════════════
# 2. Pairwise correlation
# ══════════════════════════════════════════════════════════

def high_corr_pairs(X: np.ndarray, feature_cols: list[str],
                    threshold: float = 0.95
                    ) -> list[tuple[float, str, str, float]]:
    """|corr| > threshold 인 pair 모두 반환 (signed, names, abs_r)."""
    corr = np.corrcoef(X.T)
    np.fill_diagonal(corr, 0.0)
    n = len(feature_cols)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            r = corr[i, j]
            if np.isnan(r):
                continue
            if abs(r) >= threshold:
                pairs.append((abs(r), feature_cols[i], feature_cols[j], r))
    pairs.sort(reverse=True)
    return pairs


# ══════════════════════════════════════════════════════════
# 3. Condition number
# ══════════════════════════════════════════════════════════

def condition_number(X: np.ndarray) -> tuple[float, str]:
    """X 의 condition number = σ_max / σ_min.

    >1e10 : 심각 (numerical instability)
    >1e6  : 주의 (regularization 권장)
    <1e6  : 정상
    """
    try:
        sv = np.linalg.svd(X, compute_uv=False)
        kappa = float(sv[0] / max(sv[-1], 1e-300))
    except Exception:
        return float("nan"), "compute_failed"

    if kappa > 1e10:
        return kappa, "severe"
    if kappa > 1e6:
        return kappa, "warning"
    return kappa, "ok"


# ══════════════════════════════════════════════════════════
# 4. Auto-mitigation 제안
# ══════════════════════════════════════════════════════════

def suggest_removals(X: np.ndarray, feature_cols: list[str],
                     mandatory: set[str] = None,
                     vif_threshold: float = 100.0,
                     corr_threshold: float = 0.99,
                     ) -> dict:
    """제거 권장 features (자동).

    규칙 (보수적):
      1. 상수 (std < 1e-10) — 무조건 제거 권장
      2. corr ≥ 0.999 인 pair 중 mandatory 가 아닌 쪽 제거
      3. VIF > vif_threshold 인 mandatory 는 제거 안 하지만 경고 표시
    """
    mandatory = mandatory or set()
    suggestions = {
        "constant": [],     # 무조건 제거
        "perfect_pair": [], # corr=1.0 (또는 ≥0.999) — 자동 제거 가능
        "high_vif_keep": [], # VIF 높지만 mandatory → 유지 + 경고
        "high_vif_remove": [], # VIF 높고 not mandatory → 제거 권장
    }

    # 1. constant
    for i, name in enumerate(feature_cols):
        if np.std(X[:, i]) < 1e-10:
            suggestions["constant"].append(name)

    # 2. near-perfect pair
    pairs = high_corr_pairs(X, feature_cols, threshold=corr_threshold)
    seen = set(suggestions["constant"])
    for r, a, b, _ in pairs:
        if a in seen or b in seen:
            continue
        # mandatory 가 우선 — non-mandatory 가 제거
        if a in mandatory and b not in mandatory:
            suggestions["perfect_pair"].append((b, a, r))
            seen.add(b)
        elif b in mandatory and a not in mandatory:
            suggestions["perfect_pair"].append((a, b, r))
            seen.add(a)
        elif a not in mandatory and b not in mandatory:
            # 둘 다 non-mandatory: 알파벳 순으로 두 번째 제거
            keep, drop = sorted([a, b])
            suggestions["perfect_pair"].append((drop, keep, r))
            seen.add(drop)
        else:
            # 둘 다 mandatory: 경고만 (자동 제거 안 함)
            suggestions["high_vif_keep"].append((a, b, r))

    # 3. VIF
    vifs = vif_table(X, feature_cols)
    for v, name, _ in vifs:
        if v < vif_threshold:
            break
        if name in seen:
            continue
        if name in mandatory:
            suggestions["high_vif_keep"].append((name, "(mandatory)", v))
        else:
            suggestions["high_vif_remove"].append((name, v))
            seen.add(name)

    return suggestions


# ══════════════════════════════════════════════════════════
# 5. Main
# ══════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strict", action="store_true",
                    help="VIF > threshold 또는 perfect pair 발견 시 exit 1")
    p.add_argument("--vif-threshold", type=float, default=100.0,
                    help="VIF 경고 임계 (default 100)")
    p.add_argument("--corr-threshold", type=float, default=0.99,
                    help="자동 제거 권장 corr 임계 (default 0.99)")
    p.add_argument("--auto-fix", action="store_true",
                    help="JSON 으로 auto-fix 권장 features 출력")
    from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
    p.add_argument("--out", type=str,
                    default=str(get_results_dir() / "multicollinearity_audit.json"),
                    help="JSON 저장 경로")
    args = p.parse_args()

    log.info("=" * 60)
    log.info("  다중공선성 자동 검사 (VIF + Correlation + Condition)")
    log.info("=" * 60)

    try:
        from simulation.tools.run_optuna_feature_selection import (
            load_data, MANDATORY_FEATURES_EXACT,
        )
    except Exception as e:
        log.error(f"Import 실패: {e}")
        return 2

    log.info("Loading data...")
    X, y, feature_cols = load_data()
    log.info(f"Data: n={len(y)}, d={len(feature_cols)}")

    # mandatory subset
    mand_idx = [i for i, c in enumerate(feature_cols) if c in MANDATORY_FEATURES_EXACT]
    X_mand = X[:, mand_idx]
    mand_names = [feature_cols[i] for i in mand_idx]
    log.info(f"Mandatory: {len(mand_idx)} / {len(feature_cols)}")
    log.info("")

    # ① 전체 X 의 condition number
    kappa, status = condition_number(X)
    log.info(f"① Condition number κ (전체)")
    log.info(f"   κ = {kappa:.2e}  [{status}]")
    if status == "severe":
        log.warning("   → numerical instability 위험 — regularization 강화 또는 feature 축소 권장")

    # ② Mandatory 만 condition
    kappa_m, status_m = condition_number(X_mand)
    log.info(f"   κ (mandatory only) = {kappa_m:.2e}  [{status_m}]")
    log.info("")

    # ③ Top correlated pairs
    pairs = high_corr_pairs(X, feature_cols, threshold=0.95)
    log.info(f"② Top |corr| ≥ 0.95 pairs (전체 {len(pairs)}개)")
    for r, a, b, signed in pairs[:15]:
        flag = "🔴" if r >= 0.999 else ("🟡" if r >= 0.97 else "🟢")
        log.info(f"   {flag} |r|={r:.4f} ({signed:+.4f})  {a:30s} ↔ {b}")
    if len(pairs) > 15:
        log.info(f"   ... and {len(pairs) - 15} more")
    log.info("")

    # ④ VIF
    log.info(f"③ Top VIF (>= {args.vif_threshold})")
    vifs = vif_table(X, feature_cols)
    n_severe = sum(1 for v, _, _ in vifs if v > args.vif_threshold)
    n_high = sum(1 for v, _, _ in vifs if 10 < v <= args.vif_threshold)
    for v, name, _ in vifs[:15]:
        if v < args.vif_threshold:
            break
        flag = "🔴" if v > 1000 else "🟡"
        is_mand = "★" if name in MANDATORY_FEATURES_EXACT else " "
        v_str = f"{v:.1e}" if v > 1e6 else f"{v:.1f}"
        log.info(f"   {flag}{is_mand} VIF={v_str:>10s}  {name}")
    log.info(f"   요약: VIF>{args.vif_threshold} {n_severe}개  /  10<VIF≤{args.vif_threshold} {n_high}개  / 총 {len(vifs)}개")
    log.info("")

    # ⑤ Auto-fix 제안
    sug = suggest_removals(X, feature_cols, MANDATORY_FEATURES_EXACT,
                           vif_threshold=args.vif_threshold,
                           corr_threshold=args.corr_threshold)
    log.info("④ Auto-fix 권장")
    if sug["constant"]:
        log.warning(f"   상수 (std<1e-10) 제거 ({len(sug['constant'])}개): {sug['constant']}")
    if sug["perfect_pair"]:
        log.warning(f"   Near-perfect pair 제거 권장 ({len(sug['perfect_pair'])}):")
        for drop, keep, r in sug["perfect_pair"][:10]:
            log.warning(f"     - drop {drop} (|r|={r:.4f} with {keep})")
    if sug["high_vif_keep"]:
        log.info(f"   VIF 높지만 mandatory 라 유지 ({len(sug['high_vif_keep'])}):")
        for entry in sug["high_vif_keep"][:5]:
            if len(entry) == 3:
                a, b, r = entry
                log.info(f"     - {a:30s} ↔ {b} ({r if isinstance(r, float) else 'high'})")
    if sug["high_vif_remove"]:
        log.warning(f"   VIF>{args.vif_threshold} 비-mandatory 제거 권장 ({len(sug['high_vif_remove'])}):")
        for name, v in sug["high_vif_remove"][:10]:
            log.warning(f"     - {name} (VIF={v:.1f})")

    # ⑥ JSON 저장
    if args.auto_fix:
        import json
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({
                "kappa_full": float(kappa),
                "kappa_mandatory": float(kappa_m),
                "n_high_corr_pairs": len(pairs),
                "top_corr_pairs": [
                    {"a": a, "b": b, "abs_r": float(r), "r": float(s)}
                    for r, a, b, s in pairs[:30]
                ],
                "top_vif": [
                    {"feature": name, "vif": float(v), "is_mandatory": name in MANDATORY_FEATURES_EXACT}
                    for v, name, _ in vifs[:30]
                ],
                "suggestions": {
                    "constant": sug["constant"],
                    "perfect_pair": [{"drop": d, "keep": k, "abs_r": float(r)}
                                     for d, k, r in sug["perfect_pair"]],
                    "high_vif_remove": [{"feature": n, "vif": float(v)}
                                        for n, v in sug["high_vif_remove"]],
                },
            }, f, indent=2, ensure_ascii=False)
        log.info(f"")
        log.info(f"JSON 저장: {out_path}")

    # exit code
    n_critical = (len(sug["constant"]) + len(sug["perfect_pair"]) +
                  len(sug["high_vif_remove"]))
    log.info("")
    log.info("=" * 60)
    log.info(f"  Summary: {n_critical} critical issues  (constant={len(sug['constant'])}, "
             f"perfect_pair={len(sug['perfect_pair'])}, high_vif={len(sug['high_vif_remove'])})")
    log.info("=" * 60)

    if args.strict and n_critical > 0:
        log.error("--strict: 1 이상의 critical issue → exit 1")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
