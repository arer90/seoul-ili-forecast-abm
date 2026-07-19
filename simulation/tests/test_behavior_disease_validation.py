"""SCI-급 validation TDD — 앙상블 WIS/CRPS/coverage + HLN-DM·bootstrap 유의성.

real ILI 대비 adaptive vs static 비교가 통계적으로 valid 한지. macOS: run PER-FILE.
"""
import numpy as np
import pytest

from simulation.abm.behavior_disease_validation import (
    wis_per_week, crps_per_week, pi_coverage, point_metrics, dm_hln,
    bootstrap_ci_mean, validate_arms,
    fit_nb_dispersion, observation_ensemble, validate_arms_calibrated,
)


def _curve(T=30):
    t = np.arange(T)
    return 10.0 + 8.0 * np.exp(-((t - 15) ** 2) / 40.0)   # 종형 유행곡선


def test_wis_perfect_is_near_zero():
    y = _curve()
    reps = np.tile(y, (40, 1)) + 1e-9
    assert np.mean(wis_per_week(y, reps)) < 1e-3


def test_crps_perfect_near_zero():
    y = _curve()
    reps = np.tile(y, (40, 1))
    assert abs(np.mean(crps_per_week(y, reps))) < 1e-6


def test_coverage_and_point():
    y = _curve()
    rng = np.random.default_rng(0)
    reps = y[None, :] + rng.normal(0, 1.0, (200, len(y)))
    assert pi_coverage(y, reps, 0.05) > 0.9            # 넓은 앙상블 → 95% 커버
    pm = point_metrics(y, np.tile(y, (10, 1)))
    assert pm["rmse"] < 1e-9 and pm["mae"] < 1e-9      # median=y


def test_dm_hln_detects_better():
    rng = np.random.default_rng(1)
    loss_a = np.abs(rng.normal(0, 0.5, 30))            # 작은 손실
    loss_b = np.abs(rng.normal(0, 0.5, 30)) + 3.0      # 큰 손실
    stat, p = dm_hln(loss_a, loss_b)
    assert stat < 0 and p < 0.05                       # a 가 유의 우수


def test_shape_validation():
    y = _curve()
    with pytest.raises(ValueError):
        wis_per_week(y, np.ones((5,)))                 # 1-D reps
    with pytest.raises(ValueError):
        validate_arms(y, np.ones((5, 30)), np.ones((5, 20)))  # T mismatch


def test_validate_arms_adaptive_significantly_better():
    """adaptive 앙상블이 y 에 밀착(unbiased), static 은 편향·산포 → adaptive 유의 우수."""
    y = _curve(40)
    rng = np.random.default_rng(7)
    reps_a = y[None, :] + rng.normal(0, 0.6, (60, 40))             # 정확
    reps_s = (y[None, :] * 0.4 + 5.0) + rng.normal(0, 4.0, (60, 40))  # 편향+산포
    r = validate_arms(y, reps_a, reps_s, n_boot=1000, seed=0)
    assert r["adaptive"]["wis"] < r["static"]["wis"]
    assert r["significance"]["delta_wis_mean"] < 0
    assert r["significance"]["dm_p_value"] < 0.05
    assert r["significance"]["bootstrap_ci95_hi"] < 0
    assert r["significance"]["adaptive_significantly_better"] is True


def test_validate_arms_no_difference_not_significant():
    """동일 분포 두 arm → 유의차 없음 (false positive 가드)."""
    y = _curve(40)
    rng = np.random.default_rng(3)
    reps_a = y[None, :] + rng.normal(0, 1.0, (60, 40))
    reps_s = y[None, :] + rng.normal(0, 1.0, (60, 40))
    r = validate_arms(y, reps_a, reps_s, n_boot=1000, seed=0)
    assert r["significance"]["adaptive_significantly_better"] is False


def test_auc_roc_and_c_index_added_and_discriminate():
    """AUC-ROC(이상유행 탐지)·C-index(순위 일치) 추가 — 정확한 arm 이 평탄 arm 보다 우수."""
    y = _curve(40)                                   # 종형: 순위·outbreak 구조 존재
    rng = np.random.default_rng(5)
    reps_a = y[None, :] + rng.normal(0, 0.5, (60, 40))            # 정확 → 순위·outbreak 추종
    reps_s = np.full((60, 40), float(y.mean())) + rng.normal(0, 0.5, (60, 40))  # 평탄 → 무변별
    r = validate_arms(y, reps_a, reps_s, threshold=float(np.median(y)), n_boot=500)
    a, s = r["adaptive"], r["static"]
    assert "auc_roc" in a and "c_index" in a         # 추가됨
    assert 0.0 <= a["auc_roc"] <= 1.0 and 0.0 <= a["c_index"] <= 1.0
    assert a["c_index"] > s["c_index"] + 0.1         # 정확 arm 이 순위 더 잘 맞춤
    assert a["auc_roc"] > s["auc_roc"] + 0.1         # 정확 arm 이 outbreak 더 잘 변별
    assert abs(s["c_index"] - 0.5) < 0.15            # 평탄 arm ≈ 무변별(0.5)


def test_fit_dispersion_and_observation_ensemble():
    """φ 적합 + NegBin 앙상블이 n_draws 무관 calibrated 구간을 준다 (coverage 붕괴 X)."""
    import numpy as np
    y = _curve(40)
    phi = fit_nb_dispersion(y, y)                 # μ=y → φ 적합
    assert phi > 0
    rng = np.random.default_rng(0)
    reps = observation_ensemble(y, phi, 400, rng)
    assert reps.shape == (400, 40)
    cov = pi_coverage(y, reps, 0.05)
    assert cov > 0.8                              # μ=y → 95% PI 가 y 를 잘 덮음 (붕괴 X)


def test_calibrated_validation_fixes_coverage_collapse():
    """관측노이즈 fold → coverage 붕괴(0) 해결. adaptive(μ≈y) > static(편향)."""
    import numpy as np
    y = _curve(40)
    r = validate_arms_calibrated(y, mu_adaptive=y, mu_static=0.6 * y + 3.0,
                                 n_draws=400, seed=1, threshold=float(np.median(y)))
    assert r["adaptive"]["coverage95"] > 0.8      # 붕괴 아님 (seed-spread 였으면 ~0)
    assert r["adaptive"]["wis"] < r["static"]["wis"]   # 정확 arm 우수
    assert r["dispersion"]["phi_adaptive"] > 0
    assert "auc_roc" in r["adaptive"] and "c_index" in r["adaptive"]
