"""Q4 / G-277: 계층(within / out-of-train-range) metric 보고 회귀 가드.

full metric 은 primary 로 유지(외삽 점 삭제=cherry-pick 금지). y_true > train_max 인
외삽 점을 분리해 within-range r2/mae 를 별도 보고 — "정상 구간은 잘 맞춘다"를 정직하게.
"""
from __future__ import annotations

import numpy as np

from simulation.pipeline.metric_eval import (
    compute_full_metrics, _stratified_range_metrics, _STRAT_KEYS,
)


def test_split_counts_and_peak():
    y_train = np.array([10.0, 30.0, 66.0])          # train_max = 66
    a = np.array([20.0, 50.0, 66.0, 90.0, 100.7])   # 2개 외삽(>66)
    p = np.array([22.0, 48.0, 60.0, 70.0, 75.0])
    d = _stratified_range_metrics(a, p, y_train)
    assert d["n_within_range"] == 3
    assert d["n_out_of_range"] == 2
    assert abs(d["frac_out_of_range"] - 0.4) < 1e-9
    assert abs(d["out_of_range_max_obs"] - 100.7) < 1e-6


def test_within_r2_uses_only_within():
    y_train = np.array([10.0, 66.0])
    # within 점들은 완벽 예측 → within_r2 ≈ 1.0, 외삽은 크게 틀려도 within_r2 영향 X
    a = np.array([20.0, 40.0, 60.0, 100.0])
    p = np.array([20.0, 40.0, 60.0, 10.0])   # 마지막(외삽)만 틀림
    d = _stratified_range_metrics(a, p, y_train)
    assert d["within_range_r2"] > 0.99, d
    assert d["within_range_mae"] < 1e-6
    assert d["out_of_range_mae"] > 80.0       # 외삽 점은 크게 틀림


def test_none_train_pool_empty():
    a = np.array([1.0, 2.0]); p = np.array([1.0, 2.0])
    assert _stratified_range_metrics(a, p, None) == {}


def test_compute_full_metrics_includes_strat_keys():
    rng = np.random.RandomState(0)
    y_train = np.abs(rng.randn(100) * 5 + 20)
    a = np.abs(rng.randn(60) * 5 + 20)
    p = a + rng.randn(60)
    out = compute_full_metrics(a, p, sigma_for_wis=5.0, y_train_pool=y_train)
    for k in _STRAT_KEYS:
        assert k in out, f"compute_full_metrics 에 {k} 누락"
    # full r2 는 그대로 존재(보조 metric 이 primary 대체 X)
    assert "r2" in out and np.isfinite(out["r2"])


def test_empty_schema_has_strat_keys():
    """no-valid edge case(empty schema)도 동일 key set (contract 일관)."""
    from simulation.pipeline.metric_eval import _empty_metrics_schema
    sch = _empty_metrics_schema(n=0)
    for k in _STRAT_KEYS:
        assert k in sch, f"empty schema 에 {k} 누락 (contract drift)"
