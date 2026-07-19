"""G-365: model-agnostic adaptive conformal PI (Conformal-PID) for R10.

Reproduction (2026-06-26): R10 split-conformal PI under-covers all models (median 0.67)
because it calibrates on in-sample residuals that under-estimate out-of-sample (epidemic-peak)
error (in-sample σ 1/3~1/5 of test). FusedEpi's adaptive conformal reaches 0.926; the user
asked to apply adaptive conformal to ALL models.

Fix: adaptive_conformal_bounds — model-agnostic PID (Angelopoulos 2024) that widens the interval
at test peaks using PAST rolling observations (leak-free). wis_from_bounds scores it.
"""
import numpy as np

from simulation.analytics.adaptive_conformal import (
    adaptive_conformal_bounds, wis_from_bounds, _pid_adjust,
)

ALPHAS = [0.05, 0.20, 0.50]


def _shift_case(seed=0, n=68):
    rng = np.random.default_rng(seed)
    pred = np.full(n, 10.0)
    y = np.full(n, 10.0) + rng.normal(0, 1.5, n)
    y[40:55] += np.linspace(5, 25, 15)               # late distribution-shift peak
    cal_res = rng.normal(0, 2.0, 200)                # optimistic (small) in-sample residuals
    hw = {a: float(np.quantile(np.abs(cal_res), 1 - a)) for a in ALPHAS}
    return pred, y, cal_res, hw


def test_adaptive_improves_coverage_on_shift():
    pred, y, cal_res, hw = _shift_case()
    static_cov = float(np.mean((y >= pred - hw[0.05]) & (y <= pred + hw[0.05])))
    b = adaptive_conformal_bounds(pred, hw, cal_res, y, ALPHAS, window=20, ki=0.3)
    lo, hi = b[0.05]
    adapt_cov = float(np.mean((y >= lo) & (y <= hi)))
    assert adapt_cov > static_cov          # 분포이동서 adaptive 가 widen → 개선
    assert adapt_cov >= 0.85               # nominal 0.95 근처로 회복


def test_adaptive_not_overexpand_on_stationary():
    """평상시(분포이동 없음) adaptive 가 과확장하지 않음 (~nominal)."""
    pred, _, cal_res, hw = _shift_case()
    rng = np.random.default_rng(1)
    y2 = np.full(68, 10.0) + rng.normal(0, 2.0, 68)
    b = adaptive_conformal_bounds(pred, hw, cal_res, y2, ALPHAS, window=20, ki=0.3)
    lo, hi = b[0.05]
    cov = float(np.mean((y2 >= lo) & (y2 <= hi)))
    assert 0.88 <= cov <= 1.0              # 과소도 과대도 아님


def test_leak_free_uses_past_only():
    """step i 구간은 과거 obs 만 사용 — y[i] 변경이 구간[<=i] 에 영향 없어야(미래 미사용)."""
    pred, y, cal_res, hw = _shift_case()
    b1 = adaptive_conformal_bounds(pred, hw, cal_res, y, ALPHAS, window=20, ki=0.3)
    y2 = y.copy(); y2[60] += 100.0         # 후반 한 점만 교란
    b2 = adaptive_conformal_bounds(pred, hw, cal_res, y2, ALPHAS, window=20, ki=0.3)
    lo1, hi1 = b1[0.05]; lo2, hi2 = b2[0.05]
    assert np.allclose(lo1[:60], lo2[:60]) and np.allclose(hi1[:60], hi2[:60])  # i<60 불변


def test_wis_from_bounds_finite():
    pred, y, cal_res, hw = _shift_case()
    b = adaptive_conformal_bounds(pred, hw, cal_res, y, ALPHAS, window=20, ki=0.3)
    w = wis_from_bounds(y, b, ALPHAS, median=pred)
    assert w.shape == (len(y),) and np.all(np.isfinite(w)) and np.all(w >= 0)


def test_pid_empty_scores_safe():
    """init_scores 빈 배열 — 0초 죽지 않고 base 구간 반환."""
    qlo = np.full(10, 5.0); qhi = np.full(10, 15.0); obs = np.full(10, 10.0)
    nlo, nhi = _pid_adjust(qlo, qhi, obs, [], beta=0.95, target=0.05)
    assert nlo.shape == (10,) and np.all(nhi >= nlo)


def test_online_conformal_no_seed_leak_free():
    """G-365c: in-sample 잔차 없이(pure online) 작동 + leak-free (미래 미사용)."""
    from simulation.analytics.adaptive_conformal import online_conformal_bounds
    pred, y, _, _ = _shift_case()
    b1 = online_conformal_bounds(pred, y, ALPHAS, init_residuals=None, window=20)
    assert 0.05 in b1 and b1[0.05][0].shape == (len(y),)        # seed 없이도 산출
    y2 = y.copy(); y2[60] += 100.0
    b2 = online_conformal_bounds(pred, y2, ALPHAS, init_residuals=None, window=20)
    lo1, hi1 = b1[0.05]; lo2, hi2 = b2[0.05]
    assert np.allclose(lo1[:60], lo2[:60]) and np.allclose(hi1[:60], hi2[:60])  # 과거 불변


def test_online_conformal_recovers_coverage():
    """pure online 이 분포이동서 커버리지 회복(~nominal)."""
    from simulation.analytics.adaptive_conformal import online_conformal_bounds
    pred, y, _, _ = _shift_case()
    b = online_conformal_bounds(pred, y, ALPHAS, init_residuals=None, window=20, ki=0.3)
    lo, hi = b[0.05]
    cov = float(np.mean((y >= lo) & (y <= hi)))
    assert cov >= 0.80     # cold-start 있어도 회복
