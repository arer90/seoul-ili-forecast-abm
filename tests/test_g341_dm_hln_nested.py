"""G-341: DM HLN 소표본보정 + nested-pair 가드, G-340: relative-WIS vs baseline. regression guard.

외부 reviewer (2026-06-24): (1) 메인 R6 DM 이 asymptotic Normal → HLN 소표본 보정으로 통일,
(2) DM 은 nested 모델쌍(ARIMA⊂SARIMA⊂SARIMAX·Poisson⊂NegBin)에 무효 → 플래그+win-count 제외,
(3) raw WIS 외 Hub 표준 relative-WIS(vs FluSight-Baseline, Mathis 2024) 배선.
"""
import numpy as np
import pytest

from simulation.pipeline.dm_test import _dm_test, _is_nested_pair
from simulation.models.flusight_baseline import compute_relative_wis
from simulation.analytics.ablation_stats import hln_dm_pvalue


def test_dm_uses_hln_pvalue():
    """_dm_test 의 p-value == hln_dm_pvalue(제곱오차) = HLN 소표본 보정 SSOT."""
    rng = np.random.default_rng(1)
    e1 = rng.normal(0, 1.0, 50)
    e2 = rng.normal(0, 2.0, 50)
    r = _dm_test(e1, e2)
    p_direct = hln_dm_pvalue(e1 ** 2, e2 ** 2, h=1)
    assert abs(r["p_value"] - p_direct) < 1e-9, "DM p-value 가 HLN SSOT 와 불일치"
    assert r["statistic"] < 0 and r["p_value"] < 0.05, "더 정확한 e1 이 유의하게 우위"


def test_dm_small_n_guard():
    assert _dm_test(np.ones(5), np.zeros(5))["p_value"] == 1.0   # n<10


def test_dm_identical_not_significant():
    rng = np.random.default_rng(2)
    e = rng.normal(0, 1.5, 40)
    assert _dm_test(e.copy(), e.copy())["p_value"] > 0.5


def test_nested_pair_detection():
    assert _is_nested_pair("ARIMA", "SARIMA")
    assert _is_nested_pair("ARIMA", "SARIMAX")
    assert _is_nested_pair("SARIMA", "SARIMAX")
    assert _is_nested_pair("PoissonAutoreg", "NegBinGLM")
    assert _is_nested_pair("NegBinGLM", "NegBinGLM-Glum")
    # 비-nested (서로 다른 family)
    assert not _is_nested_pair("XGBoost", "RandomForest")
    assert not _is_nested_pair("ARIMA", "XGBoost")
    assert not _is_nested_pair("TabPFN", "NegBinGLM")


def test_relative_wis_vs_baseline():
    """relative WIS = gmean(model)/gmean(baseline), <1 = baseline 능가 (Mathis 2024)."""
    rw = compute_relative_wis([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
    assert abs(rw["relative_wis"] - 0.5) < 1e-6      # 모델이 baseline 의 절반 오차
    rw2 = compute_relative_wis([4.0, 4.0], [2.0, 2.0])
    assert rw2["relative_wis"] > 1.0                  # 모델이 baseline 보다 나쁨


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
