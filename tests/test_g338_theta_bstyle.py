"""G-338: Theta B-style fit-once rolling (symmetric-refit §8.6) regression guard.

사용자 (2026-06-24): symmetric refit 을 B(fit-once + rolling-observed-feed, 배포 충실)로 통일.
검증 결과 패널 대부분이 이미 B (statsmodels append(refit=False)·epi observed-feed); Theta 만
유일하게 base rolling_1step 으로 매 origin 재추정(A-style)이었음 → fixed-α/seasonal/trend 로 교체.

이 테스트가 지키는 불변식:
  1. fit 후 B-state(α/seasonal/trend) 가 한 번 추정됨.
  2. rolling predict 가 finite·nonneg·정상 R².
  3. B(fit-once) ≈ 옛 A(per-origin refit) — α window-stable 라 max|Δ| 작음.
  4. leak-free: pred[i] 가 y_observed[i:] 에 불변.
  5. 짧은/degenerate 시계열 → base refit fallback (크래시 없음).
"""
import numpy as np
import pytest

from simulation.models.ts_models import ThetaForecaster
from simulation.models.base import supports_rolling_eval, TimeSeriesForecaster


def _ili_like(n=337, seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    season = 15 * np.sin(2 * np.pi * t / 52) + 8 * np.sin(2 * np.pi * t / 26)
    return np.clip(20 + 0.01 * t + season + rng.normal(0, 3, n), 0, None)


def _r2(yt, yp):
    yt, yp = np.asarray(yt), np.asarray(yp)
    return 1 - ((yt - yp) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()


def _fit():
    y = _ili_like()
    tr, te = y[:269], y[269:]
    m = ThetaForecaster().fit(np.zeros((269, 3)), tr)
    return m, tr, te


def test_b_state_estimated_once():
    m, _, _ = _fit()
    assert m._b_alpha is not None, "B-state α not estimated in fit_series"
    assert 0.0 < m._b_alpha < 1.0
    assert len(m._b_cycle) == m._period
    assert m._b_n == 269


def test_rolling_finite_nonneg_sane():
    m, _, te = _fit()
    preds = m.predict(np.zeros((68, 3)), y_observed=te)
    assert np.all(np.isfinite(preds)) and np.all(preds >= 0)
    assert _r2(te, preds) > 0.7   # ILI-like rolling 1-step should be strong


def test_b_equivalent_to_a_refit():
    """B(fit-once) ≈ A(per-origin refit) — α window-stable 라 차이 작음."""
    m, tr, te = _fit()
    preds_b = m.predict(np.zeros((68, 3)), y_observed=te)
    m_a = ThetaForecaster().fit(np.zeros((269, 3)), tr)        # fresh — base refit mutates B-state
    preds_a = TimeSeriesForecaster.rolling_1step(m_a, te)
    gap = float(np.max(np.abs(preds_a - preds_b)))
    assert gap < 5.0, f"B deviates from A-refit by {gap:.2f} (>5)"
    assert abs(_r2(te, preds_a) - _r2(te, preds_b)) < 0.05


def test_leak_free():
    """pred[i] 가 y_observed[i:] 에 불변 = leak-free 1-step."""
    m, _, te = _fit()
    preds = m.predict(np.zeros((68, 3)), y_observed=te)
    te2 = te.copy()
    te2[40:] = 999.0
    preds2 = m.predict(np.zeros((68, 3)), y_observed=te2)
    assert np.allclose(preds[:40], preds2[:40]), "future obs leaked into past prediction"


def test_no_state_mutation():
    """rolling predict 가 B-state(level/n) 를 변형하지 않음(반복 호출 동일)."""
    m, _, te = _fit()
    lvl0, n0 = m._b_level, m._b_n
    p1 = m.predict(np.zeros((68, 3)), y_observed=te)
    p2 = m.predict(np.zeros((68, 3)), y_observed=te)
    assert m._b_level == lvl0 and m._b_n == n0
    assert np.allclose(p1, p2)


def test_short_series_fallback():
    """짧은 시계열 → B-state None → base refit fallback, 크래시 없음."""
    y = _ili_like()
    ms = ThetaForecaster().fit(np.zeros((60, 3)), y[:60])
    ps = ms.predict(np.zeros((10, 3)), y_observed=y[60:70])
    assert len(ps) == 10 and np.all(np.isfinite(ps))


def test_eval_wiring_and_single_origin():
    m, _, _ = _fit()
    assert supports_rolling_eval(m), "Theta must be in ROLLING_EVAL_MODELS"
    fc = m.forecast(5)                                          # single-origin path no regression
    assert len(fc) == 5 and np.all(np.isfinite(fc))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
