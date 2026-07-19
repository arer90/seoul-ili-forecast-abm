"""Smoke tests: FusedEpi 불확실성 분해 (aleatoric/epistemic).

불변식 검증:
    - aleatoric + epistemic ≈ total (분산의 전체 법칙)
    - 멤버 불일치 큰 시점 → epistemic ↑
    - 단일 멤버 → epistemic ≈ 0
    - shape 일치 ((m, n) → (n,))
    - 음수 없음 (분산은 비음)
    - flag = epistemic 상위 분위 (자료부족 경고)
"""
import numpy as np
import pytest

from simulation.analytics.uncertainty_decomp import (
    decompose_uncertainty,
    flag_high_epistemic,
)


def test_total_variance_law_holds():
    """aleatoric + epistemic == total (법칙)."""
    rng = np.random.default_rng(0)
    preds = rng.normal(10.0, 3.0, size=(8, 50))
    res_var = np.full(50, 4.0)
    d = decompose_uncertainty(preds, residual_var=res_var)
    assert np.allclose(d["aleatoric"] + d["epistemic"], d["total"])
    # epistemic = 멤버 간 표본분산과 일치
    assert np.allclose(d["epistemic"], np.var(preds, axis=0, ddof=0))


def test_shape_preserved_and_nonneg():
    """(m, n) → 각 성분 (n,); 모든 성분 비음."""
    rng = np.random.default_rng(1)
    m, n = 12, 337  # 서울 25구 ILI 337주 시나리오
    preds = rng.normal(5.0, 2.0, size=(m, n))
    d = decompose_uncertainty(preds, residual_var=np.full(n, 1.0))
    for key in ("epistemic", "aleatoric", "total", "epistemic_frac"):
        assert d[key].shape == (n,), key
        assert np.all(d[key] >= 0.0), key
    assert d["n_members"] == m and d["n_steps"] == n


def test_high_disagreement_raises_epistemic():
    """멤버 의견이 갈리는 시점(분포이동)에서 epistemic 증가."""
    n = 60
    base = np.full((6, n), 10.0)
    base += np.random.default_rng(2).normal(0, 0.3, size=(6, n))  # 평상시 소폭
    # 시점 40-50: 멤버마다 크게 다른 예측 (자료부족·외삽 의견불일치)
    spread = np.linspace(-15, 15, 6)[:, None]
    base[:, 40:50] += spread
    d = decompose_uncertainty(base)
    peak_epi = d["epistemic"][40:50].mean()
    calm_epi = d["epistemic"][:40].mean()
    assert peak_epi > 10 * calm_epi  # 의견불일치 구간 epistemic 압도적으로 큼


def test_single_member_zero_epistemic():
    """단일 멤버 → epistemic ≈ 0 (멤버 간 분산 없음)."""
    preds = np.full((1, 30), 7.0)
    d = decompose_uncertainty(preds, residual_var=2.0)
    assert np.allclose(d["epistemic"], 0.0)
    # aleatoric 만 total 을 이룸 (residual_var scalar broadcast)
    assert np.allclose(d["aleatoric"], 2.0)
    assert np.allclose(d["total"], 2.0)
    assert np.allclose(d["epistemic_frac"], 0.0)


def test_member_vars_take_precedence():
    """member_vars 주어지면 멤버 평균이 aleatoric (residual_var 무시)."""
    preds = np.zeros((4, 10))  # epistemic = 0
    mv = np.full((4, 10), 5.0)
    d = decompose_uncertainty(preds, member_vars=mv, residual_var=np.full(10, 999.0))
    assert np.allclose(d["aleatoric"], 5.0)  # member_vars 우선, residual_var 무시
    assert np.allclose(d["epistemic"], 0.0)


def test_flag_high_epistemic_marks_data_scarce():
    """epistemic 높은 시점 플래그 = 자료부족 경고 (상위 분위만 True)."""
    n = 50
    base = np.full((5, n), 8.0) + np.random.default_rng(3).normal(0, 0.2, size=(5, n))
    base[:, 30:35] += np.linspace(-10, 10, 5)[:, None]  # 자료부족 외삽 구간
    d = decompose_uncertainty(base)
    flags = flag_high_epistemic(d, quantile=0.8)
    assert flags.shape == (n,) and flags.dtype == bool
    # 진짜 자료부족(외삽) 구간 전부 플래그됨
    assert flags[30:35].all()
    # 상위 분위만 플래그 (threshold>=0 분위 strict 초과) → 전체의 ~20% 이하
    assert flags.sum() <= int(round((1.0 - 0.8) * n))
    # 플래그된 시점의 epistemic 이 비-플래그보다 항상 큼 (분위 의미 보존)
    assert d["epistemic"][flags].min() > d["epistemic"][~flags].max()


def test_determinism_repeatable():
    """동일 입력 → 동일 출력 (결정성)."""
    rng = np.random.default_rng(7)
    preds = rng.normal(0, 1, size=(6, 40))
    d1 = decompose_uncertainty(preds, residual_var=np.full(40, 1.5))
    d2 = decompose_uncertainty(preds, residual_var=np.full(40, 1.5))
    for key in ("epistemic", "aleatoric", "total", "epistemic_frac"):
        assert np.array_equal(d1[key], d2[key]), key


def test_edge_validation_raises():
    """edge: 잘못된 입력 → fail-loud ValueError."""
    with pytest.raises(ValueError):
        decompose_uncertainty(np.zeros(10))            # 1-D 아님
    with pytest.raises(ValueError):
        decompose_uncertainty(np.zeros((3, 5)), residual_var=np.full(4, 1.0))  # n 불일치
    with pytest.raises(ValueError):
        decompose_uncertainty(np.zeros((3, 5)), residual_var=np.full(5, -1.0))  # 음수
    with pytest.raises(ValueError):
        flag_high_epistemic({"epistemic": np.ones(5)}, quantile=1.5)  # 분위 범위 밖
    with pytest.raises(ValueError):
        flag_high_epistemic({"total": np.ones(5)})     # epistemic 키 없음


def test_empty_flags_safe():
    """edge: 빈 epistemic → 빈 bool 배열 (0초 죽지 않음)."""
    flags = flag_high_epistemic({"epistemic": np.array([])})
    assert flags.shape == (0,) and flags.dtype == bool
