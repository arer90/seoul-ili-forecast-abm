"""각 feature 의 실제 분포 측정 → 데이터 기반 transform 권장 (2026-04-29).

목적: 그룹별 transform 매핑 (`_GROUP_OPTIONS`) 이 실제 데이터에 맞는지 empirical
      검증. 추측 (textbook) vs 측정 (real data) 비교.

측정 통계:
  · n_obs, n_unique, missing_rate
  · min, max, mean, median, std
  · skewness, kurtosis (분포 모양)
  · neg_ratio (음수 비율)
  · zero_ratio (0 비율 — sparsity)
  · p99/p1 dynamic range (heavy-tail 정도)
  · Shapiro-Wilk p (정규성)

데이터 기반 transform 권장 (decision rule):
  - 음수 있음 (neg_ratio > 0):
      - skew 큼 (|skew|>1) → yeo_johnson
      - else → standard or arcsinh
  - 모두 양수, 0 많음 (zero_ratio > 0.3):
      - count-like (mean > 1) → anscombe
      - count-like (mean < 1) → freeman_tukey
  - 모두 양수, skewed (skew > 2):
      - log1p
  - 모두 양수, mild skew (1 < skew < 2):
      - sqrt
  - bounded [0, 1] (max ≤ 1):
      - arcsine_sqrt
  - cyclic-like (range [-1, 1]):
      - passthrough
  - binary (n_unique == 2):
      - passthrough
  - 정규에 가까움 (|skew| < 0.5):
      - standard

출력:
  simulation/results/feature_distribution_profile.json   (모든 feature 통계)
  simulation/results/feature_transform_recommendations.md (그룹별 비교)

사용:
  .venv/bin/python -m simulation.scripts.profile_feature_distributions
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# 1. 분포 통계 측정
# ════════════════════════════════════════════════════════════════
def profile_one_feature(name: str, x: np.ndarray) -> dict:
    """한 feature 의 분포 통계 측정."""
    x = np.asarray(x, dtype=np.float64)
    n_obs = len(x)
    finite = np.isfinite(x)
    x_clean = x[finite]
    if len(x_clean) < 3:
        return {"name": name, "error": "insufficient finite obs"}

    stats = {
        "name": name,
        "n_obs": int(n_obs),
        "n_finite": int(len(x_clean)),
        "n_unique": int(len(np.unique(x_clean))),
        "missing_rate": float((n_obs - len(x_clean)) / n_obs),
        "min": float(np.min(x_clean)),
        "max": float(np.max(x_clean)),
        "mean": float(np.mean(x_clean)),
        "median": float(np.median(x_clean)),
        "std": float(np.std(x_clean)),
        "skew": float(_skewness(x_clean)),
        "kurt": float(_kurtosis(x_clean)),
        "neg_ratio": float(np.mean(x_clean < 0)),
        "zero_ratio": float(np.mean(np.abs(x_clean) < 1e-9)),
    }
    # Dynamic range
    p1 = float(np.percentile(x_clean, 1))
    p99 = float(np.percentile(x_clean, 99))
    stats["p1"] = p1
    stats["p99"] = p99
    stats["dynamic_range"] = float(p99 - p1)

    # Bounded?
    stats["is_bounded_unit"] = bool(stats["min"] >= 0 and stats["max"] <= 1.001)
    stats["is_bounded_neg1_pos1"] = bool(stats["min"] >= -1.001 and stats["max"] <= 1.001)
    stats["is_binary"] = bool(stats["n_unique"] == 2)
    stats["is_constant"] = bool(stats["n_unique"] == 1)

    return stats


def _skewness(x: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    m = np.mean(x)
    s = np.std(x) + 1e-12
    return float(np.mean(((x - m) / s) ** 3))


def _kurtosis(x: np.ndarray) -> float:
    if len(x) < 4:
        return 0.0
    m = np.mean(x)
    s = np.std(x) + 1e-12
    return float(np.mean(((x - m) / s) ** 4) - 3.0)


# ════════════════════════════════════════════════════════════════
# 2. 데이터 기반 transform 권장
# ════════════════════════════════════════════════════════════════
def recommend_transform(stats: dict) -> dict:
    """분포 통계 → transform 권장 (decision rule)."""
    if "error" in stats:
        return {"primary": "none", "reason": "no data"}
    if stats["is_constant"]:
        return {"primary": "drop", "reason": "constant feature"}
    if stats["is_binary"]:
        return {"primary": "passthrough", "reason": "binary 0/1"}
    if stats["is_bounded_neg1_pos1"] and abs(stats["mean"]) < 0.1:
        return {"primary": "passthrough", "reason": "cyclic-like [-1, 1]"}
    if stats["is_bounded_unit"]:
        return {"primary": "arcsine_sqrt", "reason": "proportion [0, 1]"}

    skew = stats["skew"]
    neg = stats["neg_ratio"]
    zero = stats["zero_ratio"]
    mean = stats["mean"]

    # 음수 있음
    if neg > 0.05:
        if abs(skew) > 1.5:
            return {"primary": "yeo_johnson",
                    "secondary": ["arcsinh"],
                    "reason": f"mixed-sign + skewed (neg={neg:.0%}, |skew|={abs(skew):.2f})"}
        if abs(skew) > 0.5:
            return {"primary": "arcsinh",
                    "secondary": ["yeo_johnson"],
                    "reason": f"mixed-sign + mild skew (neg={neg:.0%})"}
        return {"primary": "standard",
                "secondary": ["robust", "arcsinh"],
                "reason": f"mixed-sign + symmetric (neg={neg:.0%})"}

    # 모두 양수, sparse zero
    if zero > 0.3:
        if mean > 1.0:
            return {"primary": "anscombe",
                    "secondary": ["freeman_tukey", "log1p"],
                    "reason": f"Poisson-like, mean={mean:.2f} (zero={zero:.0%})"}
        return {"primary": "freeman_tukey",
                "secondary": ["anscombe", "log1p"],
                "reason": f"low-mean Poisson, mean={mean:.2f} (zero={zero:.0%})"}

    # 모두 양수, skewed
    if skew > 2.5:
        return {"primary": "log1p",
                "secondary": ["sqrt", "yeo_johnson"],
                "reason": f"strongly right-skewed (skew={skew:.2f})"}
    if skew > 1.0:
        return {"primary": "sqrt",
                "secondary": ["log1p", "yeo_johnson"],
                "reason": f"moderately right-skewed (skew={skew:.2f})"}

    # 모두 양수, near-normal
    return {"primary": "standard",
            "secondary": ["robust", "log1p"],
            "reason": f"near-normal positive (skew={skew:.2f})"}


# ════════════════════════════════════════════════════════════════
# 3. 그룹 가정 vs 데이터 비교
# ════════════════════════════════════════════════════════════════
def compare_with_group_assumption(profile: list[dict]) -> dict:
    """현재 그룹별 매핑 vs 데이터 기반 권장 비교."""
    from simulation.models.grouped_preprocessor import classify_feature, _GROUP_OPTIONS

    by_group = {}
    for p in profile:
        if "error" in p:
            continue
        g = classify_feature(p["name"])
        by_group.setdefault(g, []).append(p)

    comparison = []
    for group, members in sorted(by_group.items(), key=lambda x: -len(x[1])):
        # 그룹 안에서 권장 transform 의 mode (가장 흔한 것)
        recommendations = [recommend_transform(m).get("primary", "none") for m in members]
        from collections import Counter
        rec_counter = Counter(recommendations)
        most_common = rec_counter.most_common(3)

        # 그룹의 분포 통계 평균
        group_skew = float(np.median([m.get("skew", 0) for m in members]))
        group_neg = float(np.mean([m.get("neg_ratio", 0) for m in members]))
        group_zero = float(np.mean([m.get("zero_ratio", 0) for m in members]))

        # 현재 _GROUP_OPTIONS 의 권장
        current_options = _GROUP_OPTIONS.get(group, {})
        current_log_options = current_options.get("log_op", [])

        comparison.append({
            "group": group,
            "n_features": len(members),
            "data_skew_median": group_skew,
            "data_neg_ratio": group_neg,
            "data_zero_ratio": group_zero,
            "data_recommended_top3": [(t, n) for t, n in most_common],
            "current_log_options": current_log_options,
            "agreement": _check_agreement(most_common, current_log_options),
        })
    return {"by_group": comparison}


def _check_agreement(data_recommendations, current_options) -> str:
    """데이터 권장 vs 현재 옵션 일치 여부."""
    if not data_recommendations:
        return "n/a"
    primary_rec = data_recommendations[0][0]
    if primary_rec in current_options:
        return f"✓ '{primary_rec}' 가 현재 menu 에 있음"
    return f"✗ '{primary_rec}' 가 현재 menu 에 없음 (추가 필요)"


# ════════════════════════════════════════════════════════════════
# 4. 메인
# ════════════════════════════════════════════════════════════════
def main(argv: Optional[list[str]] = None) -> int:
    # R1(data) 데이터 로드 — checkpoint + fe_cache 활용 (재계산 회피)
    log.info("  R1(data) 데이터 로드 중...")
    from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
    ckpt_path = get_results_dir() / "checkpoints" / "checkpoint_phase1.json"
    if not ckpt_path.exists():
        log.error(f"  R1(data) checkpoint 없음: {ckpt_path}")
        return 1
    ckpt = json.loads(ckpt_path.read_text())
    data = ckpt.get("data", {})
    feature_cols = data.get("feature_cols", [])
    n = data.get("n", 337)
    log.info(f"  R1(data): n={n}, features={len(feature_cols)}")

    # X 행렬 직접 빌드 (builder.py 호출)
    try:
        from simulation.pipeline.config import PipelineConfig
        from simulation.pipeline.data import run_data
        cfg = PipelineConfig()
        cfg.data.use_fe_cache = True   # cache 사용 (빠름)
        # advanced features 환경변수 적용
        import os
        os.environ.setdefault("MPH_ADVANCED_FEATURES", "1")
        phase1 = run_data(cfg)
        X = phase1["X_all"]
        feature_cols = phase1["feature_cols"]
    except Exception as e:
        log.error(f"  R1(data) 재계산 실패: {e}. fe_cache 사용 시도...")
        # fallback: fe_cache parquet 직접 읽기
        try:
            import polars as pl
            from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
            cache_path = get_results_dir() / "feature_cache.parquet"
            if cache_path.exists():
                df = pl.read_parquet(cache_path)
                feature_cols = [c for c in df.columns if c not in ("ili_rate",)]
                X = df.select(feature_cols).to_numpy()
                log.info(f"  fe_cache 사용: shape={X.shape}")
            else:
                log.error(f"  fe_cache 도 없음: {cache_path}")
                return 1
        except Exception as e2:
            log.error(f"  fallback 실패: {e2}")
            return 1
    log.info(f"  X shape: {X.shape}, features: {len(feature_cols)}")
    log.info(f"  분포 profiling 시작...")

    profile = []
    for i, name in enumerate(feature_cols):
        stats = profile_one_feature(name, X[:, i])
        stats["recommendation"] = recommend_transform(stats)
        profile.append(stats)

    # 통계 출력
    log.info("")
    log.info("  ── 권장 transform 분포 ──")
    rec_count = {}
    for p in profile:
        r = p.get("recommendation", {}).get("primary", "?")
        rec_count[r] = rec_count.get(r, 0) + 1
    for r, n in sorted(rec_count.items(), key=lambda x: -x[1]):
        log.info(f"    {r:<18} {n:>4} features")

    # 그룹 가정 vs 데이터 비교
    log.info("")
    log.info("  ── 그룹 가정 vs 데이터 권장 비교 ──")
    comparison = compare_with_group_assumption(profile)

    log.info("")
    log.info(f"  {'그룹':<18} {'n':>4} {'skew_med':>9} {'neg%':>5} {'zero%':>6}  {'데이터 권장':<28} 일치?")
    log.info(f"  {'-'*100}")
    for c in comparison["by_group"]:
        rec_str = ", ".join([f"{t}({n})" for t, n in c["data_recommended_top3"]])
        log.info(f"  {c['group']:<18} {c['n_features']:>4} "
                  f"{c['data_skew_median']:>9.2f} "
                  f"{c['data_neg_ratio']*100:>5.0f} "
                  f"{c['data_zero_ratio']*100:>6.0f}  "
                  f"{rec_str:<28} {c['agreement']}")

    # 저장
    out = {
        "n_features": len(feature_cols),
        "recommendation_distribution": rec_count,
        "comparison": comparison,
        "profile": profile,
    }
    from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
    (get_results_dir() / "feature_distribution_profile.json").write_text(
        json.dumps(out, indent=2, default=str)
    )
    log.info("")
    log.info(f"  저장: simulation/results/feature_distribution_profile.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
