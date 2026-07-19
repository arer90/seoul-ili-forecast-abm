"""Rich features helper — V3-V5 mini test 검증된 24개 파생변수.

Mini test V5 결과:
  Basic 5 features: 모든 DL/TS 모델이 reference baseline (R²=0.876) 미달
  Rich 24 features: DNN R²=0.94, TinyMLP R²=0.93 → reference baseline 능가

V3 진단 (mini_test_v3_dnn_tcn):
  DNN-1단계 (identity, basic 5): R²=0.87 (reference baseline 거의)
  DNN-2단계 (identity, rich 24): R²=0.94 (reference baseline 능가) ← +0.07 향상

원인:
  Basic 5 (lag1/lag2/lag5/sin_52/cos_52) 만으로는 ILI 의 단기 dynamics 만 학습.
  Rich 24 는 추가:
    - lag3, lag4, lag6, lag7 (autoregressive 풍부화)
    - rmean_4/8/12 (단기/중기/장기 trend)
    - rstd_4/8/12 (volatility — 정점 시즌)
    - diff1, diff2 (1차/2차 미분 — 가속도)
    - sin_26, cos_26, sin_13 (다중 주기 — 분기/계절)
    - winter (40-12주 indicator)
    - log_lag1 (큰 값 안정화)
    - week_norm, lag1_high (positional + threshold)

사용:
    from simulation.models.rich_features import build_rich_features
    X, feat_names = build_rich_features(y, week_seq)
"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple


def build_rich_features(y: np.ndarray, week_seq: np.ndarray,
                          n_drop: int = 7) -> Tuple[np.ndarray, List[str]]:
    """24개 파생변수 생성 — V3-V5 mini test 검증.

    Args:
        y:         target time series (n,)
        week_seq:  week index array (n,) — seasonal sin/cos 용
        n_drop:    앞쪽 invalid lag rows 제거 (default 7 = max lag)

    Returns:
        (X[n_drop:], feature_names) — feature matrix + name list
    """
    y = np.asarray(y, dtype=float)
    weeks = np.asarray(week_seq, dtype=float)
    n = len(y)
    if n != len(weeks):
        raise ValueError(f"y ({n}) and weeks ({len(weeks)}) length mismatch")

    features = {}
    # 1) lag1-7 (autoregressive)
    for k in range(1, 8):
        features[f"lag{k}"] = np.roll(y, k)

    # 2) Rolling stats (lag1 기반 → leakage 방지)
    def roll_mean(arr, w):
        return np.array([np.mean(arr[max(0, i - w + 1):i + 1]) for i in range(len(arr))])
    def roll_std(arr, w):
        return np.array([np.std(arr[max(0, i - w + 1):i + 1]) if i >= 1 else 0.0
                          for i in range(len(arr))])
    y_lag = np.roll(y, 1)
    for w in [4, 8, 12]:
        features[f"rmean_{w}"] = roll_mean(y_lag, w)
        features[f"rstd_{w}"] = roll_std(y_lag, w)

    # 3) Differences (가속도)
    features["diff1"] = features["lag1"] - features["lag2"]
    features["diff2"] = features["lag1"] - 2 * features["lag2"] + features["lag5"]

    # 4) Multi-period seasonal (52/26/13주)
    features["sin_52"] = np.sin(2 * np.pi * weeks / 52.0)
    features["cos_52"] = np.cos(2 * np.pi * weeks / 52.0)
    features["sin_26"] = np.sin(2 * np.pi * weeks / 26.0)
    features["cos_26"] = np.cos(2 * np.pi * weeks / 26.0)
    features["sin_13"] = np.sin(2 * np.pi * weeks / 13.0)

    # 5) Calendar / threshold
    features["winter"] = ((weeks >= 40) | (weeks <= 12)).astype(float)
    features["log_lag1"] = np.log1p(np.clip(features["lag1"], 0, None))
    features["week_norm"] = (weeks - 26) / 26.0
    # G-186 (2026-05-06 Codex audit Q2): leakage path warning
    # `np.percentile(y, 75)` = full-series percentile — train+test 통합 분포에서 계산
    # → 학습 시 train period 만으로 percentile 계산해야 학술 정직 (look-ahead 차단)
    # 현재 active path 검증 필요: rich_features 가 cache 에 통합되면 leakage risk
    # 다음 학습 시 → walk-forward percentile (train-only) 사용 권장
    # 참조: docs/NEXT_SPRINT_CHECKLIST_66_MODELS_PAPER_GRADE.md P1-J
    features["lag1_high"] = (features["lag1"] > np.percentile(y, 75)).astype(float)

    feat_names = list(features.keys())
    X = np.column_stack([features[name] for name in feat_names])

    # invalid lag rows 제거
    if n_drop > 0:
        X = X[n_drop:]

    return X, feat_names


def basic_features(y: np.ndarray, week_seq: np.ndarray,
                    n_drop: int = 5) -> Tuple[np.ndarray, List[str]]:
    """Basic 5 features (reference baseline 비교용)."""
    y = np.asarray(y, dtype=float)
    weeks = np.asarray(week_seq, dtype=float)
    features = {
        "lag1": np.roll(y, 1),
        "lag2": np.roll(y, 2),
        "lag5": np.roll(y, 5),
        "sin_52": np.sin(2 * np.pi * weeks / 52.0),
        "cos_52": np.cos(2 * np.pi * weeks / 52.0),
    }
    feat_names = list(features.keys())
    X = np.column_stack([features[name] for name in feat_names])
    if n_drop > 0:
        X = X[n_drop:]
    return X, feat_names


__all__ = ["build_rich_features", "basic_features"]
