"""Sobol 전역민감도 (분산 분해) smoke test.

불변식 (검증 대상):
  1. 선형 가산모델 y=Σaᵢxᵢ — S1ᵢ 가 해석해 aᵢ²/Σaⱼ² 에 근사.
  2. ΣS1 ≤ 1 (가산이면 ≈1).
  3. ST ≥ S1 (모든 파라미터, 추정 노이즈 허용 tolerance).
  4. 가산모델 → ST ≈ S1 (상호작용 0).
  5. 상호작용 모델 → ST > S1 (상호작용 존재).
  6. 결정성 — 같은 seed 두 호출 byte-identical.
  7. shape / 키 구조 / n_evals = n*(p+2).
  8. edge / leak-free — 상수 출력은 모든 지수 0; 불량 bounds 는 ValueError.

ABM kernel 직접 호출 없이 합성 test 함수로 전부 검증 (모델-비종속).
실행: `.venv/bin/python -m pytest tests/test_sobol_sensitivity.py -x -q` (macOS 파일단위).
"""
from __future__ import annotations

import numpy as np
import pytest

from simulation.abm.sobol_sensitivity import sobol_indices, sobol_rank


# ── 합성 모델 ────────────────────────────────────────────────────────────────
def _additive(coeffs):
    """y = Σ coeffs[i] * x_i ; x_i ∈ [0,1]."""
    def fn(p: dict) -> float:
        return float(sum(c * p[f"x{i}"] for i, c in enumerate(coeffs)))
    bounds = {f"x{i}": (0.0, 1.0) for i in range(len(coeffs))}
    return fn, bounds


def _interaction(p: dict) -> float:
    """y = x0 + x1 + 4·x0·x1 — 강한 상호작용항."""
    return p["x0"] + p["x1"] + 4.0 * p["x0"] * p["x1"]


_N = 4096  # 합성 함수 노이즈 작음 → 4096 으로 수렴 확보


# ── 1. 선형 가산: S1 해석해 근사 ────────────────────────────────────────────
def test_additive_s1_matches_analytic():
    """y=3x0+1x1+0x2 → S1 ∝ aᵢ²; 해석해 aᵢ²/Σaⱼ² 에 근사."""
    fn, bounds = _additive([3.0, 1.0, 0.0])
    res = sobol_indices(fn, bounds, n_samples=_N, seed=0)
    denom = 9.0 + 1.0 + 0.0
    analytic = {"x0": 9.0 / denom, "x1": 1.0 / denom, "x2": 0.0}
    for k, exp in analytic.items():
        assert abs(res["S1"][k] - exp) < 0.05, (k, res["S1"][k], exp)


# ── 2. ΣS1 ≤ 1 (가산 → ≈1) ──────────────────────────────────────────────────
def test_sum_s1_le_one_additive():
    # 평균-센터링 first-order 추정량 → 작은 계수 섞여도 n=4096 에서 ΣS1≈1.
    fn, bounds = _additive([2.0, 1.0, 0.5])
    res = sobol_indices(fn, bounds, n_samples=_N, seed=1)
    total = sum(res["S1"].values())
    assert total <= 1.0 + 0.05, total          # ≤ 1 (노이즈 tolerance)
    assert total > 0.90, total                 # 가산 → ≈ 1


# ── 3. ST ≥ S1 (모든 파라미터) ──────────────────────────────────────────────
def test_st_ge_s1_all_params():
    res = sobol_indices(_interaction, {"x0": (0.0, 1.0), "x1": (0.0, 1.0)},
                        n_samples=_N, seed=2)
    for name in res["names"]:
        assert res["ST"][name] >= res["S1"][name] - 1e-6, (
            name, res["ST"][name], res["S1"][name])


# ── 4. 가산모델 → ST ≈ S1 ───────────────────────────────────────────────────
def test_additive_st_approx_s1():
    fn, bounds = _additive([3.0, 1.0, 2.0])
    res = sobol_indices(fn, bounds, n_samples=_N, seed=3)
    for name in res["names"]:
        assert abs(res["ST"][name] - res["S1"][name]) < 0.06, (
            name, res["ST"][name], res["S1"][name])


# ── 5. 상호작용 모델 → ST > S1 ──────────────────────────────────────────────
def test_interaction_st_strictly_gt_s1():
    res = sobol_indices(_interaction, {"x0": (0.0, 1.0), "x1": (0.0, 1.0)},
                        n_samples=_N, seed=4)
    # 상호작용항 4·x0·x1 → 두 파라미터 모두 ST 가 S1 보다 뚜렷이 큼
    for name in res["names"]:
        assert res["ST"][name] - res["S1"][name] > 0.01, (
            name, res["ST"][name], res["S1"][name])
    # 가산 분해 누설: ΣS1 < 1 (상호작용이 분산을 흡수)
    assert sum(res["S1"].values()) < 0.98


# ── 6. 결정성 ───────────────────────────────────────────────────────────────
def test_determinism_same_seed():
    fn, bounds = _additive([1.0, 2.0, 3.0])
    r1 = sobol_indices(fn, bounds, n_samples=512, seed=7)
    r2 = sobol_indices(fn, bounds, n_samples=512, seed=7)
    np.testing.assert_array_equal(r1["S1_array"], r2["S1_array"])
    np.testing.assert_array_equal(r1["ST_array"], r2["ST_array"])
    # 다른 seed 는 (일반적으로) 다른 값
    r3 = sobol_indices(fn, bounds, n_samples=512, seed=8)
    assert not np.array_equal(r1["S1_array"], r3["S1_array"])


# ── 7. shape / 키 구조 / n_evals ────────────────────────────────────────────
def test_shape_and_keys():
    fn, bounds = _additive([1.0, 1.0, 1.0, 1.0])  # p=4
    n = 256
    res = sobol_indices(fn, bounds, n_samples=n, seed=0)
    assert res["names"] == ["x0", "x1", "x2", "x3"]
    assert res["S1_array"].shape == (4,)
    assert res["ST_array"].shape == (4,)
    assert set(res["S1"]) == set(bounds)
    assert set(res["ST"]) == set(bounds)
    assert res["n_samples"] == n
    assert res["n_evals"] == n * (4 + 2)
    assert res["var_Y"] > 0.0


# ── 7b. rank 정렬 ───────────────────────────────────────────────────────────
def test_sobol_rank_orders_by_importance():
    fn, bounds = _additive([5.0, 1.0, 3.0])  # x0 > x2 > x1
    res = sobol_indices(fn, bounds, n_samples=_N, seed=0)
    ranked = sobol_rank(res, by="ST")
    names_in_order = [name for name, _ in ranked]
    assert names_in_order[0] == "x0"            # 최대 영향
    assert names_in_order[-1] == "x1"           # 최소 영향
    # 값은 내림차순
    vals = [v for _, v in ranked]
    assert vals == sorted(vals, reverse=True)
    # by="S1" 도 동작
    ranked_s1 = sobol_rank(res, by="S1")
    assert ranked_s1[0][0] == "x0"


# ── 8. edge: 상수 출력 → 모든 지수 0 ────────────────────────────────────────
def test_constant_output_zero_indices():
    def const(p: dict) -> float:
        return 42.0
    bounds = {"x0": (0.0, 1.0), "x1": (0.0, 1.0)}
    res = sobol_indices(const, bounds, n_samples=128, seed=0)
    assert res["var_Y"] == 0.0
    assert all(v == 0.0 for v in res["S1"].values())
    assert all(v == 0.0 for v in res["ST"].values())


# ── 8b. edge: 불량 입력 → ValueError ────────────────────────────────────────
def test_invalid_inputs_raise():
    fn, bounds = _additive([1.0, 1.0])
    with pytest.raises(ValueError):
        sobol_indices(fn, {}, n_samples=64)            # 빈 bounds
    with pytest.raises(ValueError):
        sobol_indices(fn, {"x0": (1.0, 1.0)}, n_samples=64)  # low == high
    with pytest.raises(ValueError):
        sobol_indices(fn, {"x0": (2.0, 1.0)}, n_samples=64)  # low > high
    with pytest.raises(ValueError):
        sobol_indices(fn, bounds, n_samples=1)          # n < 2
    res = sobol_indices(fn, bounds, n_samples=64, seed=0)
    with pytest.raises(ValueError):
        sobol_rank(res, by="bogus")                     # 잘못된 by
