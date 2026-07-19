"""FusedEpi peak-aware loss/eval smoke test (peak_weighting.py).

공중보건 자원배치 = peak 정확도 → outbreak/peak 구간을 upweight한 모델-비종속 평가.

불변식 (사양):
  1. peak 구간 가중 ↑ (peak_weight > base_weight, 정점에만).
  2. peak_rmse ≠ overall_rmse (비균일 가중 시).
  3. weights 음수 없음 (모든 출력 ≥ 0).
  4. 평탄(분산 0) 시계열 → 균일 가중.
  5. shape 일치 (입력 n = 출력 n).
  6. leak-free / 보존: weights 균일이면 peak_aware_wis = 표준 WIS 평균.

실행: .venv/bin/python -m pytest tests/test_peak_weighting.py -x -q
"""
import numpy as np
import pytest

from simulation.analytics.peak_weighting import (
    peak_weights,
    weighted_metrics,
    peak_aware_wis,
    DEFAULT_ALPHAS,
)

ALPHAS = [0.05, 0.20, 0.50]


def _epi_curve(n=68, seed=0):
    """서울 ILI 형 단일-정점 유행 곡선 (원공간, 비음)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = 2.0 + 0.5 * np.sin(2 * np.pi * t / 52.0)
    peak = 12.0 * np.exp(-((t - 45) ** 2) / (2 * 6.0 ** 2))  # week ~45 정점
    y = np.clip(base + peak + rng.normal(0, 0.3, n), 0, None)
    return y


def _bounds_from_pred(pred, half_widths):
    """pred ± hw_α 대칭 구간 dict 생성."""
    return {a: (pred - hw, pred + hw) for a, hw in half_widths.items()}


# --- case 1: peak 구간 가중 ↑ (불변식 1) ----------------------------------- #
def test_peak_region_upweighted():
    y = _epi_curve()
    w = peak_weights(y, quantile=0.9, peak_weight=3.0, base_weight=1.0)
    assert w.shape == y.shape
    peak_idx = y >= np.quantile(y, 0.9)
    # peak 시점은 모두 3.0, 그 외 1.0.
    assert np.allclose(w[peak_idx], 3.0)
    assert np.allclose(w[~peak_idx], 1.0)
    assert w[np.argmax(y)] == 3.0  # 진짜 정점은 반드시 peak.


# --- case 2: 음수 없음 + edge 가중치 (불변식 3) ---------------------------- #
def test_weights_non_negative_and_validation():
    y = _epi_curve(seed=1)
    w = peak_weights(y, peak_weight=5.0, base_weight=0.0)
    assert np.all(w >= 0.0)
    assert w.min() == 0.0 and w.max() == 5.0
    with pytest.raises(ValueError):
        peak_weights(y, peak_weight=-1.0)          # 음수 가중치 거부
    with pytest.raises(ValueError):
        peak_weights(y, quantile=1.5)              # quantile 범위 밖
    with pytest.raises(ValueError):
        peak_weights(np.array([]))                 # 빈 입력


# --- case 3: 평탄 시계열 → 균일 가중 (불변식 4) ---------------------------- #
def test_flat_series_uniform_weights():
    y_flat = np.full(40, 7.0)
    w = peak_weights(y_flat, quantile=0.9, peak_weight=3.0, base_weight=1.0)
    assert np.allclose(w, 1.0)          # 평탄 → 전부 base, peak 없음
    # NaN 섞인 평탄도 폭발 없이 base.
    y_nan = y_flat.copy()
    y_nan[5] = np.nan
    w2 = peak_weights(y_nan, base_weight=1.0)
    assert np.all(np.isfinite(w2)) and np.allclose(w2, 1.0)


# --- case 4: peak_rmse ≠ overall_rmse (불변식 2) --------------------------- #
def test_peak_rmse_differs_from_overall():
    rng = np.random.default_rng(2)
    y = _epi_curve(seed=2)
    # 정점에서 더 큰 오차를 주입한 예측 (peak 오차 ≫ 평시 오차).
    pred = y.copy()
    peak_idx = y >= np.quantile(y, 0.9)
    pred[peak_idx] += rng.normal(0, 4.0, peak_idx.sum())
    pred[~peak_idx] += rng.normal(0, 0.1, (~peak_idx).sum())
    w = peak_weights(y, quantile=0.9, peak_weight=3.0)

    m = weighted_metrics(y, pred, w)
    assert set(m) >= {"peak_rmse", "peak_mae", "overall_rmse", "overall_mae",
                      "peak_skill", "n_peak"}
    assert m["n_peak"] > 0
    # peak 구간 오차가 크므로 peak_rmse > overall_rmse 이고 서로 다르다.
    assert m["peak_rmse"] != m["overall_rmse"]
    assert m["peak_rmse"] > m["overall_rmse"]


# --- case 5: shape 일치 + 검증 (불변식 5) --------------------------------- #
def test_shape_consistency_and_validation():
    y = _epi_curve(seed=3)
    w = peak_weights(y)
    assert w.shape == (len(y),)
    pred = y + 0.5
    with pytest.raises(ValueError):
        weighted_metrics(y, pred[:-1], w)          # 길이 불일치 fail-fast
    with pytest.raises(ValueError):
        weighted_metrics(y, pred, -np.ones_like(w))  # 음수 weights 거부


# --- case 6: weights 균일 → peak_aware_wis == 표준 WIS 평균 (불변식 6/보존) - #
def test_uniform_weights_recovers_standard_wis():
    y = _epi_curve(seed=4)
    pred = y + 0.3
    hw = {a: float(np.quantile(np.abs(y - pred), 1 - a)) + 1.0 for a in ALPHAS}
    bounds = _bounds_from_pred(pred, hw)

    # 가중 (peak upweight) vs 균일.
    w_uniform = np.ones_like(y)
    w_peak = peak_weights(y, quantile=0.9, peak_weight=4.0)

    wis_uniform = peak_aware_wis(y, bounds, ALPHAS, w_uniform, median=pred)

    # 균일 가중 = 표준 WIS 의 산술평균과 동일해야 함 (보존 불변식).
    from simulation.analytics.peak_weighting import _per_point_wis
    pp = _per_point_wis(y, bounds, ALPHAS, pred)
    assert np.isclose(wis_uniform, float(np.mean(pp)), rtol=1e-9)

    # peak-가중은 정점 오차를 무겁게 → 정점 예측을 망치면 peak-WIS 가 더 크게 반응.
    pred_bad = pred.copy()
    peak_idx = y >= np.quantile(y, 0.9)
    pred_bad[peak_idx] -= 6.0
    bounds_bad = _bounds_from_pred(pred_bad, hw)
    wis_peak_bad = peak_aware_wis(y, bounds_bad, ALPHAS, w_peak, median=pred_bad)
    wis_unif_bad = peak_aware_wis(y, bounds_bad, ALPHAS, w_uniform, median=pred_bad)
    assert wis_peak_bad > wis_unif_bad  # peak 가중이 정점 악화에 더 민감


# --- case 7: empty alphas / 결정성 (edge + reproducibility) ---------------- #
def test_empty_alphas_nan_and_determinism():
    y = _epi_curve(seed=5)
    pred = y + 0.2
    bounds = _bounds_from_pred(pred, {a: 2.0 for a in ALPHAS})
    w = peak_weights(y)
    # bounds 에 없는 level 만 요청 → NaN.
    out = peak_aware_wis(y, bounds, [0.99], w, median=pred)
    assert np.isnan(out)
    # 결정성: 동일 seed 두 번 → bit-identical.
    w_a = peak_weights(y, seed=123)
    w_b = peak_weights(y, seed=123)
    assert np.array_equal(w_a, w_b)


# --- case 8: peak_skill 정의 + DEFAULT_ALPHAS K=11 ------------------------- #
def test_peak_skill_and_default_alphas():
    y = _epi_curve(seed=6)
    # 완벽 예측 → peak_skill = 1.0 (model MSE 0).
    w = peak_weights(y, quantile=0.85, peak_weight=3.0)
    m_perfect = weighted_metrics(y, y.copy(), w)
    assert np.isclose(m_perfect["peak_skill"], 1.0, atol=1e-9)
    assert m_perfect["peak_rmse"] == 0.0

    # DEFAULT_ALPHAS = FluSight K=11.
    assert len(DEFAULT_ALPHAS) == 11
    assert DEFAULT_ALPHAS[0] == 0.02 and DEFAULT_ALPHAS[-1] == 0.90
