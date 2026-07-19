"""In-run 동적 agent 할당 TDD — 유효N 스케일·strata 재할당·불편 보존 재샘플.

사용자 제안(2026-06-05) "변화에 따라 agent 수·내용 동적". macOS: run PER-FILE.
"""
import numpy as np
import pytest

from simulation.abm.adaptive_allocation import (
    AdaptiveAllocator, allocate_by_activity, resample_weighted,
)


def test_target_n_scales_with_prevalence():
    al = AdaptiveAllocator(base_n=8000, max_n=32000, floor_n=2000, sensitivity=1.0)
    peak = 0.1
    assert al.target_n(0.0, peak) == 2000           # 저활동 → floor
    assert al.target_n(0.1, peak) == 32000          # peak → max
    assert 2000 < al.target_n(0.05, peak) < 32000   # 중간
    assert al.target_n(0.02, peak) < al.target_n(0.08, peak)   # monotone↑
    assert al.target_n(0.5, peak) == 32000          # over-peak clamp


def test_target_n_peak_ref_zero_uses_base():
    assert AdaptiveAllocator().target_n(0.05, 0.0) == AdaptiveAllocator().base_n


def test_allocator_validation():
    with pytest.raises(ValueError):
        AdaptiveAllocator(floor_n=5000, base_n=3000)   # floor > base
    with pytest.raises(ValueError):
        AdaptiveAllocator(sensitivity=0.0)


def test_allocate_sums_floor_and_proportional():
    act = np.array([100.0, 10.0, 1.0, 0.0])         # gu0 최고활동, gu3 0
    out = allocate_by_activity(act, budget=1000, floor_frac=0.05)
    assert int(out.sum()) == 1000                    # 합 = budget
    assert np.all(out >= 1)                           # floor
    assert out[0] > out[1] > out[2]                  # 활동 비례
    assert out[3] >= 1                                # 활동0 strata 도 floor 유지(소실 방지)


def test_allocate_zero_activity_uniform():
    out = allocate_by_activity(np.zeros(4), budget=400, floor_frac=0.05)
    assert int(out.sum()) == 400
    assert out.max() - out.min() <= 1                # 활동0 → ~균등


def test_allocate_validation():
    with pytest.raises(ValueError):
        allocate_by_activity(np.array([1.0, 2.0]), budget=1)   # budget < strata


def test_resample_conserves_population():
    w = np.array([10.0, 5.0, 1.0, 0.0, 84.0])        # P=100
    idx, nw = resample_weighted(w, target_count=50, rng=np.random.default_rng(0))
    assert idx.shape == (50,)
    assert abs(50 * nw - 100.0) < 1e-9               # Σ(new weight)=P 보존
    assert np.all((0 <= idx) & (idx < 5))


def test_resample_unbiased():
    """E[복제수_i] ∝ w_i (particle-filter 불편) — 다회 평균 검증."""
    w = np.array([1.0, 3.0, 6.0])                    # share 0.1/0.3/0.6
    M = 100
    counts = np.zeros(3)
    for s in range(400):
        idx, _ = resample_weighted(w, M, np.random.default_rng(s))
        for i in range(3):
            counts[i] += int(np.sum(idx == i))
    share = counts / counts.sum()
    assert np.allclose(share, [0.1, 0.3, 0.6], atol=0.03)


def test_resample_validation():
    with pytest.raises(ValueError):
        resample_weighted(np.array([1.0]), 50, None)              # rng 필수
    with pytest.raises(ValueError):
        resample_weighted(np.zeros(3), 10, np.random.default_rng(0))   # 합 0
