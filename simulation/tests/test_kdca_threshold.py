"""Test KDCA epidemic threshold helper (audit Stage 1.1, Task #13).

Verifies:
    - identify_non_epidemic_weeks correctness (KDCA: positivity<2%, ≥2 consec weeks)
    - compute_kdca_epidemic_threshold:
        * primary KDCA mean+2SD when viral_positivity provided
        * fallback q70-lowest proxy when viral_positivity None
        * fallback when all weeks epidemic (no non-epi periods)
        * NaN-safe (no raise)
    - KDCA 2024-25 / 2025-26 reference values (paper cross-verification target)

Reference: Kang SK, Son WS, Kim BI (2024) doi:10.3346/jkms.2024.39.e40 (PMID 38288541)
"""
from __future__ import annotations

import numpy as np

try:
    import pytest
except ImportError:
    pytest = None  # type: ignore[assignment]

from simulation.analytics.kdca_threshold import (
    KDCA_DEFAULT_THRESHOLD_2024_25,
    KDCA_DEFAULT_THRESHOLD_2025_26,
    WEEKS_PER_SEASON,
    compute_kdca_epidemic_threshold,
    get_kdca_season_reference,
    identify_non_epidemic_weeks,
)


# ────────────────────────────────────────────────────────────────────
# identify_non_epidemic_weeks
# ────────────────────────────────────────────────────────────────────


def test_identify_non_epidemic_empty():
    assert len(identify_non_epidemic_weeks(np.array([]))) == 0
    assert len(identify_non_epidemic_weeks(None)) == 0  # type: ignore[arg-type]


def test_identify_non_epidemic_all_low():
    """All weeks < 2% → all non-epidemic."""
    pos = np.full(10, 0.01)  # 1%
    mask = identify_non_epidemic_weeks(pos)
    assert mask.sum() == 10
    assert mask.all()


def test_identify_non_epidemic_all_high():
    """All weeks >= 2% → none non-epidemic."""
    pos = np.full(10, 0.05)  # 5%
    mask = identify_non_epidemic_weeks(pos)
    assert mask.sum() == 0


def test_identify_non_epidemic_min_consec_2():
    """≥2 consec weeks required. Isolated week skipped."""
    pos = np.array([0.01, 0.05, 0.01, 0.05, 0.01, 0.01, 0.05])
    #               low  high low  high low  low  high
    # consec low runs: [0..0]=1, [2..2]=1, [4..5]=2  → only last 2 are non-epi
    mask = identify_non_epidemic_weeks(pos, min_consec_weeks=2)
    expected = np.array([False, False, False, False, True, True, False])
    np.testing.assert_array_equal(mask, expected)


def test_identify_non_epidemic_nan_treated_as_low():
    """NaN positivity → treated as non-epidemic (conservative)."""
    pos = np.array([np.nan, np.nan, 0.05, 0.01, 0.01])
    mask = identify_non_epidemic_weeks(pos, min_consec_weeks=2)
    # NaN, NaN = 2 consec low → non-epi; 0.05 = high; 0.01, 0.01 = 2 consec low → non-epi
    expected = np.array([True, True, False, True, True])
    np.testing.assert_array_equal(mask, expected)


# ────────────────────────────────────────────────────────────────────
# compute_kdca_epidemic_threshold
# ────────────────────────────────────────────────────────────────────


def test_compute_threshold_none_train_pool():
    """None train_pool → all-NaN output, method=fallback_q70."""
    out = compute_kdca_epidemic_threshold(None)  # type: ignore[arg-type]
    assert np.isnan(out["threshold"])
    assert out["method"] == "fallback_q70"
    assert out["n_nonepi_weeks"] == 0
    assert "Kang SK" in out["reference"]


def test_compute_threshold_empty_pool():
    out = compute_kdca_epidemic_threshold(np.array([]))
    assert np.isnan(out["threshold"])


def test_compute_threshold_fallback_q70_when_no_positivity():
    """viral_positivity=None → fallback q70 mean+2SD."""
    np.random.seed(42)
    y = np.random.gamma(2, 2, 156)  # 3 seasons
    out = compute_kdca_epidemic_threshold(y)
    assert out["method"] == "fallback_q70"
    assert np.isfinite(out["threshold"])
    assert np.isfinite(out["threshold_q70"])
    assert out["threshold"] == out["threshold_q70"]  # primary = fallback


def test_compute_threshold_kdca_primary_when_positivity_provided():
    """viral_positivity 제공 + non-epi 가능 → KDCA primary."""
    np.random.seed(42)
    y = np.random.gamma(2, 2, 156)
    pos = np.random.uniform(0, 0.05, 156)
    pos[:30] = 0.005  # 첫 30주 = clearly non-epidemic (<2%)
    out = compute_kdca_epidemic_threshold(y, viral_positivity_train=pos)
    assert out["method"] == "kdca"
    assert out["n_nonepi_weeks"] >= 30  # at least the 첫 30주
    assert np.isfinite(out["threshold"])
    assert np.isfinite(out["mean_nonepi"])
    assert np.isfinite(out["sd_nonepi"])
    # KDCA formula: mean + 2*SD
    expected = out["mean_nonepi"] + 2.0 * out["sd_nonepi"]
    assert abs(out["threshold"] - expected) < 1e-6


def test_compute_threshold_kdca_fallback_when_all_epidemic():
    """모든 weeks viral_pos >= 2% → no non-epi → fallback q70."""
    np.random.seed(42)
    y = np.random.gamma(2, 2, 156)
    pos = np.full(156, 0.05)  # 모두 5% (epidemic)
    out = compute_kdca_epidemic_threshold(y, viral_positivity_train=pos)
    assert out["method"] == "fallback_q70"
    assert out["n_nonepi_weeks"] == 0
    assert np.isfinite(out["threshold"])  # fallback works


def test_compute_threshold_truncate_to_n_seasons():
    """n_seasons * 52 보다 긴 pool 은 마지막 N season 만 사용."""
    np.random.seed(42)
    y = np.random.gamma(2, 2, 500)  # ~9.6 seasons
    pos = np.random.uniform(0, 0.05, 500)
    out = compute_kdca_epidemic_threshold(y, viral_positivity_train=pos, n_seasons=3)
    assert out["n_seasons_used"] == 3
    # 3 * 52 = 156 weeks 만 사용됨 (last)


def test_compute_threshold_nan_safe():
    """y_train_pool 에 NaN 있어도 raise X."""
    np.random.seed(42)
    y = np.random.gamma(2, 2, 156)
    y[::10] = np.nan
    pos = np.random.uniform(0, 0.05, 156)
    pos[:30] = 0.005
    out = compute_kdca_epidemic_threshold(y, viral_positivity_train=pos)
    assert np.isfinite(out["threshold"])  # NaN-safe


def test_compute_threshold_n_seasons_short_pool():
    """매우 짧은 pool 도 모두 사용 (n_seasons window 보다 짧음)."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # 5 weeks only
    out = compute_kdca_epidemic_threshold(y, n_seasons=3)
    # 짧지만 fallback 산출 가능 (data 4 이상)
    assert out["method"] == "fallback_q70"
    assert np.isfinite(out["threshold"])


# ────────────────────────────────────────────────────────────────────
# KDCA 공시 reference cross-verification
# ────────────────────────────────────────────────────────────────────


def test_kdca_reference_constants():
    """Kang SK, Son WS, Kim BI (2024) + KDCA weekly surveillance reports."""
    assert KDCA_DEFAULT_THRESHOLD_2024_25 == 8.6
    assert KDCA_DEFAULT_THRESHOLD_2025_26 == 9.1


def test_get_kdca_season_reference():
    assert get_kdca_season_reference("2024-25") == 8.6
    assert get_kdca_season_reference("2025-26") == 9.1
    assert get_kdca_season_reference("unknown") is None


def test_constants():
    assert WEEKS_PER_SEASON == 52


# ────────────────────────────────────────────────────────────────────
# Integration: metric_eval.compute_full_metrics with KDCA
# ────────────────────────────────────────────────────────────────────


def test_compute_full_metrics_with_kdca_threshold():
    """metric_eval 의 compute_full_metrics 가 viral_positivity 전달 받아 KDCA primary 사용."""
    from simulation.pipeline.metric_eval import compute_full_metrics

    np.random.seed(42)
    y_test = np.random.gamma(2, 2, 68)
    y_pred = y_test + np.random.normal(0, 0.5, 68)
    y_train = np.random.gamma(2, 2, 156)  # 3 seasons
    pos_train = np.random.uniform(0, 0.05, 156)
    pos_train[:30] = 0.005

    # KDCA primary
    m_kdca = compute_full_metrics(
        y_test, y_pred,
        sigma_for_wis=1.0,
        y_train_pool=y_train,
        viral_positivity_train=pos_train,
        threshold_method="kdca",
    )
    assert "r2" in m_kdca
    assert "wis" in m_kdca

    # q70 sensitivity
    m_q70 = compute_full_metrics(
        y_test, y_pred,
        sigma_for_wis=1.0,
        y_train_pool=y_train,
        viral_positivity_train=pos_train,
        threshold_method="q70",
    )
    # 두 method 의 R²/WIS 는 동일 (threshold 와 무관)
    assert m_kdca["r2"] == m_q70["r2"]
    assert m_kdca["wis"] == m_q70["wis"]
    # alert metric 은 threshold 에 따라 다를 수 있음
    # (값 없을 수 있어서 finite check 만)


def test_compute_full_metrics_backward_compat():
    """기존 caller (viral_positivity 미전달) 도 작동 — backward compat."""
    from simulation.pipeline.metric_eval import compute_full_metrics

    np.random.seed(42)
    y_test = np.random.gamma(2, 2, 68)
    y_pred = y_test + np.random.normal(0, 0.5, 68)
    y_train = np.random.gamma(2, 2, 156)

    # 기존 signature (no viral_positivity)
    m = compute_full_metrics(y_test, y_pred, sigma_for_wis=1.0, y_train_pool=y_train)
    assert "r2" in m  # 정상 산출
    assert "wis" in m
    # threshold_method default = "kdca", viral_pos None → fallback_q70 자동 적용


if __name__ == "__main__":
    if pytest is not None:
        pytest.main([__file__, "-v"])
    else:
        # Inline runner (no pytest)
        import sys
        tests = [name for name in dir() if name.startswith("test_")]
        passed = failed = 0
        for name in sorted(tests):
            fn = globals()[name]
            try:
                fn()
                print(f"  PASS  {name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL  {name}: {type(e).__name__}: {e}")
                failed += 1
        print(f"\n{passed} passed, {failed} failed")
        sys.exit(0 if failed == 0 else 1)
