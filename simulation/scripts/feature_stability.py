"""Feature stability — Kuncheva index for WF-CV (Tier 1 #3).

여러 fold 에서 같은 feature subset 이 선택되는지 측정.
불안정한 feature 자동 식별 → drop 하면 R² +0.01-0.02 가능.

ENGINEERING_PRINCIPLES.md §원칙 #5 (재현성): Kuncheva index 정량 — fold 간 일관성.

근거:
    Kuncheva LI (2007). "A Stability Index for Feature Selection".
    Proceedings of IASTED Int. Conf. on AI and Applications.

산식:
    Kuncheva index =  (Σ |S_i ∩ S_j| - r²/k) / (r - r²/k)
    where:
      S_i, S_j = feature subsets in fold i, j
      r = average subset size
      k = total feature count
      pairwise sum over i<j

    Range: [-1, 1]
      1.0  = perfectly stable (모든 fold 같은 subset)
      0.0  = random
      -1.0 = anti-correlated

    Threshold (실용):
      > 0.75  매우 안정 — 사용
      0.50-0.75 안정 — 사용 가능
      < 0.50  불안정 — drop 권장

사용:
    .venv/bin/python -m simulation.scripts.feature_stability \\
        --subsets simulation/results/optuna_feature_selection.db \\
        --model dnn \\
        --threshold 0.5
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np


def kuncheva_index(subsets: list[set], total_features: int) -> float:
    """Kuncheva 2007 stability index.

    Args:
        subsets: list of feature subsets (각 fold)
        total_features: total feature count k

    Returns:
        index in [-1, 1]
    """
    n = len(subsets)
    if n < 2:
        return 1.0

    sizes = [len(s) for s in subsets]
    r = np.mean(sizes)
    k = total_features

    pairs = 0
    sum_intersect = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            inter = len(subsets[i] & subsets[j])
            sum_intersect += inter
            pairs += 1

    avg_inter = sum_intersect / pairs

    # Kuncheva: I = (avg_inter - r²/k) / (r - r²/k)
    denom = r - (r * r) / k
    if denom < 1e-9:
        return 0.0
    return float((avg_inter - (r * r) / k) / denom)


def feature_frequency(subsets: list[set]) -> dict[str, float]:
    """Per-feature selection frequency across folds."""
    n = len(subsets)
    if n == 0:
        return {}
    all_features = set().union(*subsets)
    return {
        f: sum(1 for s in subsets if f in s) / n
        for f in all_features
    }


def extract_subsets_from_optuna(
    db_path: str,
    study_pattern: str = "feat",
) -> dict[str, list[set]]:
    """Optuna DB 에서 study 별 best trial 의 feature subset 추출.

    Returns:
        {study_name: [subset_fold_1, subset_fold_2, ...]}
    """
    from simulation.database import safe_connect  # G-116 (2026-05-29)
    conn = safe_connect(db_path)
    studies = conn.execute(
        "SELECT study_id, study_name FROM studies WHERE study_name LIKE ?",
        (f"%{study_pattern}%",)
    ).fetchall()

    out: dict[str, list[set]] = {}
    for study_id, study_name in studies:
        # Best trial per study
        best_trial = conn.execute("""
            SELECT t.trial_id FROM trials t
            JOIN trial_values tv ON tv.trial_id = t.trial_id
            WHERE t.study_id = ? AND tv.value IS NOT NULL
            ORDER BY tv.value ASC LIMIT 1
        """, (study_id,)).fetchone()
        if not best_trial:
            continue

        # Extract use_* boolean params
        params = conn.execute("""
            SELECT param_name, param_value FROM trial_params
            WHERE trial_id = ?
        """, (best_trial[0],)).fetchall()

        subset = {p_name for p_name, p_val in params
                  if p_name.startswith("use_") and p_val > 0.5}
        if subset:
            out.setdefault(study_name.split("_")[0], []).append(subset)
    return out


def main():
    ap = argparse.ArgumentParser()
    from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
    ap.add_argument("--db", default=str(get_results_dir() / "optuna_feature_selection.db"))
    ap.add_argument("--pattern", default="feat",
                    help="study name pattern (default: 'feat' for fold studies)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="stability threshold (Kuncheva)")
    ap.add_argument("--out", default=str(get_results_dir() / "feature_stability.json"))
    args = ap.parse_args()

    if not Path(args.db).exists():
        print(f"✗ Optuna DB 없음: {args.db}")
        return 1

    subsets_per_model = extract_subsets_from_optuna(args.db, args.pattern)
    if not subsets_per_model:
        print(f"✗ '{args.pattern}' 패턴 매칭 study 없음")
        return 1

    # Total features estimate (largest subset 의 크기 × 1.2)
    all_features = set()
    for subsets in subsets_per_model.values():
        for s in subsets:
            all_features |= s
    k = max(len(all_features), 100)

    print(f"전체 feature pool: {k}개")
    print(f"분석 모델: {len(subsets_per_model)}개")
    print()

    summary = {}
    for model, subsets in sorted(subsets_per_model.items()):
        if len(subsets) < 2:
            print(f"  {model:20s}: fold 1개 — skip")
            continue
        idx = kuncheva_index(subsets, total_features=k)
        freq = feature_frequency(subsets)

        # 안정 / 불안정 feature 분리
        stable = [f for f, p in freq.items() if p >= 0.8]
        unstable = [f for f, p in freq.items() if p < 0.4]

        verdict = "🟢 안정" if idx > 0.75 else "🟡 보통" if idx > 0.5 else "🔴 불안정"
        print(f"  {model:20s}: Kuncheva = {idx:+.3f}  {verdict}  "
              f"(stable={len(stable)}, unstable={len(unstable)})")
        summary[model] = {
            "kuncheva": idx,
            "n_folds": len(subsets),
            "n_stable_features": len(stable),
            "n_unstable_features": len(unstable),
            "verdict": verdict,
            "stable_features": stable[:20],   # top 20
            "unstable_features": unstable[:20],
        }

    Path(args.out).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n✓ 저장: {args.out}")
    print(f"\n해석:")
    print(f"  > 0.75  매우 안정 — 모델 사용 권장")
    print(f"  0.50~0.75 안정 — 사용 가능")
    print(f"  < {args.threshold:.2f}  불안정 — feature subset 재검토 권장")
    return 0


if __name__ == "__main__":
    sys.exit(main())
