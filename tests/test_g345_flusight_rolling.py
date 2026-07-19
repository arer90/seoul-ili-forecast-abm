"""G-345 (2026-06-24, 감사 P1-2): FluSight-Baseline rolling persistence — relative-WIS 분모 정직화.

FluSight 는 ROLLING_EVAL_MODELS 멤버인데 predict 가 y_observed 를 무시 → 68주 frozen flat(last_obs).
relative-WIS 분모(FluSight WIS)가 비현실적으로 약해져 타 모델 skill 과대(분모 능가 인플레).
Mathis 2024 표준(median=지난주 관측) rolling random-walk persistence 로 정정.

macOS: per-file.
"""
import numpy as np

from simulation.models.flusight_baseline import FluSightQuantileBaseline


def _fit(y_train):
    m = FluSightQuantileBaseline()
    m.fit(np.zeros((len(y_train), 1)), np.asarray(y_train, float))
    return m


def test_rolling_persistence_uses_prev_week():
    """rolling: 주 i median = y_observed[i-1], 주 0 = train 마지막 관측."""
    m = _fit([10.0, 11.0, 12.0])      # last_obs = 12
    yo = np.array([20.0, 30.0, 40.0, 50.0])
    pred = m.predict(np.zeros((4, 1)), y_observed=yo)
    assert pred[0] == 12.0            # 주 0 = train 마지막
    assert pred[1] == 20.0            # 주 1 = yo[0]
    assert pred[2] == 30.0            # 주 2 = yo[1]
    assert pred[3] == 40.0            # 주 3 = yo[2]


def test_static_without_y_observed_is_flat():
    """y_observed 없으면 flat last_obs (back-compat)."""
    m = _fit([10.0, 11.0, 12.0])
    pred = m.predict(np.zeros((4, 1)))
    assert np.allclose(pred, 12.0)


def test_rolling_is_leak_free():
    """주 i 예측은 y_observed[:i] 에만 의존 — 미래(test[k:])를 바꿔도 앞 예측 불변."""
    m = _fit([5.0, 6.0, 7.0])
    yo = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    base = m.predict(np.zeros((5, 1)), y_observed=yo.copy())
    yo2 = yo.copy(); yo2[3:] = 999.0     # 미래 교란
    pert = m.predict(np.zeros((5, 1)), y_observed=yo2)
    assert np.allclose(base[:4], pert[:4])   # 주 0..3 은 yo[:3] 만 → 불변


def test_rolling_beats_flat_on_trending_series():
    """추세 series 에서 rolling persistence 가 flat 보다 점예측 우수(분모가 strawman 아님 입증)."""
    rng = np.random.RandomState(0)
    y = np.cumsum(np.abs(rng.randn(60))) + 10      # 단조 증가 추세
    m = _fit(y[:40])
    yo = y[40:]
    roll = m.predict(np.zeros((len(yo), 1)), y_observed=yo)
    flat = np.full(len(yo), y[39])
    assert np.mean(np.abs(roll - yo)) < np.mean(np.abs(flat - yo)), "rolling 이 flat 보다 우수해야"


def test_truncate_at_zero():
    m = _fit([1.0, 2.0, 3.0])
    yo = np.array([-5.0, -3.0, 1.0])     # 음수 관측 가드
    pred = m.predict(np.zeros((3, 1)), y_observed=yo)
    assert np.all(pred >= 0.0)
