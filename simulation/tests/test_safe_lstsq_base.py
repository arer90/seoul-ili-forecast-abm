"""G-275 base layer: safe_lstsq 수치 robust 최소제곱 회귀 가드.

ill-conditioned design(collinear)에서 lstsq(rcond=None)이 |β|≈1e6 로 폭발하는
G-275 부류를 base layer 에서 차단. well-conditioned 면 일반 lstsq 와 동일 결과.
"""
from __future__ import annotations

import numpy as np

from simulation.models.safety import safe_lstsq


def _collinear_design(n=200, seed=0):
    rng = np.random.RandomState(seed)
    base = np.abs(rng.randn(n) * 5 + 20)
    A = np.column_stack([
        np.ones(n), base, np.log1p(base),
        np.clip(np.round(base / 5), 0, None),
        (base - base.mean()) / (base.std() + 1e-9),
    ])
    y = base * 1.2 + rng.randn(n) * 2 + 5
    return A, y


def test_wellconditioned_matches_lstsq():
    """well-conditioned: safe_lstsq ≈ np.linalg.lstsq (무손실)."""
    rng = np.random.RandomState(1)
    A = np.column_stack([np.ones(100), rng.randn(100, 3)])
    y = A @ np.array([1.0, 2.0, -1.0, 0.5]) + rng.randn(100) * 0.1
    ref, *_ = np.linalg.lstsq(A, y, rcond=None)
    got = safe_lstsq(A, y)
    assert np.allclose(got, ref, atol=1e-3), f"well-conditioned 불일치: {got} vs {ref}"


def test_collinear_bounded_coef():
    """핵심: near-singular(cond≫1e12)서 |β| bounded (rcond=None 은 1e6 폭발)."""
    A, y = _collinear_design()
    assert np.linalg.cond(A) > 1e10, "테스트 design 이 ill-conditioned 하지 않음"
    none_coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    safe_coef = safe_lstsq(A, y)
    # rcond=None 은 이 design 에서 |β| 폭발, safe 는 O(1~10)
    assert np.all(np.isfinite(safe_coef))
    assert np.abs(safe_coef).max() < 100.0, f"safe_lstsq |β| 폭발: {np.abs(safe_coef).max():.2e}"


def test_singular_ridge_fallback():
    """완전 특이(중복 컬럼)에도 finite (ridge fallback)."""
    n = 50
    col = np.linspace(0, 1, n)
    A = np.column_stack([np.ones(n), col, col, col])   # col 3× 중복 = rank-deficient
    y = 2 * col + 1 + np.random.RandomState(2).randn(n) * 0.01
    coef = safe_lstsq(A, y)
    assert np.all(np.isfinite(coef)), "특이행렬서 non-finite"
    pred = A @ coef
    assert np.all(np.isfinite(pred))
    assert np.abs(pred - y).mean() < 1.0, "ridge fallback 적합 실패"


def test_multitarget_shape():
    """b 가 (n, k) 여도 (p, k) 반환."""
    rng = np.random.RandomState(3)
    A = np.column_stack([np.ones(80), rng.randn(80, 2)])
    B = rng.randn(80, 4)
    coef = safe_lstsq(A, B)
    assert coef.shape == (3, 4)
    assert np.all(np.isfinite(coef))
