"""TDD — ablation-ladder 통계 코어 (HLN-DM + Holm, 3자 검증 설계 2026-06-02).

run PER-FILE on macOS: KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 pytest <thisfile>
"""
import numpy as np
import pytest

from simulation.analytics.ablation_stats import (
    hln_dm_pvalue, holm_correction, ladder_deltas, factorial_effects)


def test_hln_dm_detects_clear_winner():
    rng = np.random.default_rng(42)
    n = 68
    loss_a = np.abs(rng.normal(0, 1, n))          # A 작은 손실
    loss_b = loss_a + 1.5                          # B 일관되게 큼 → A 유의하게 우세
    p = hln_dm_pvalue(loss_a, loss_b)
    assert p < 0.05, f"명백한 차이는 유의해야: p={p}"


def test_hln_dm_noise_not_significant():
    rng = np.random.default_rng(7)
    n = 68
    base = np.abs(rng.normal(0, 1, n))
    loss_a = base
    loss_b = base + rng.normal(0, 0.01, n)        # ~0.005 수준 미미차 (노이즈)
    p = hln_dm_pvalue(loss_a, loss_b)
    assert p > 0.05, f"미미차(노이즈)는 비유의여야: p={p}"


def test_hln_more_conservative_than_uncorrected_small_n():
    # HLN 보정은 소표본서 p 를 키움(보수적) → 같은 데이터로 보정계수<1 → |DM*|<|DM| → p 큼
    rng = np.random.default_rng(1)
    n = 20
    a = np.abs(rng.normal(0, 1, n)); b = a + 0.4
    p_hln = hln_dm_pvalue(a, b, h=1)
    # 보정 없는 DM p 수동 계산
    from scipy import stats as st
    d = b - a; dm = d.mean() / np.sqrt(d.var(ddof=0) / n)
    p_uncorr = 2 * (1 - st.norm.cdf(abs(dm)))
    assert p_hln >= p_uncorr - 1e-9, "HLN 은 소표본서 더 보수적(p 큼)이어야"


def test_hln_degenerate_returns_one():
    assert hln_dm_pvalue([1.0, 1.0], [1.0, 1.0]) == 1.0       # n<3
    assert hln_dm_pvalue([1, 2, 3], [1, 2, 3]) == 1.0          # 0 차이


def test_holm_correction_basic():
    # 3개 p: 0.01, 0.02, 0.04 → Holm: 0.01*3=0.03, 0.02*2=0.04, 0.04*1=0.04(monotone)
    adj = holm_correction([0.01, 0.02, 0.04])
    assert abs(adj[0] - 0.03) < 1e-9
    assert abs(adj[1] - 0.04) < 1e-9
    assert abs(adj[2] - 0.04) < 1e-9
    # monotone (정렬 순)
    assert adj[0] <= adj[1] <= adj[2]


def test_holm_caps_at_one_and_empty():
    assert holm_correction([]) == []
    assert all(p <= 1.0 for p in holm_correction([0.5, 0.6, 0.9]))


def test_ladder_classifies_improve_vs_ns():
    rng = np.random.default_rng(3)
    n = 68
    y = rng.normal(10, 2, n)
    # A0 큰 오차, A1 명백 개선, A2 미미(노이즈), A3 미미
    A0 = y + rng.normal(0, 3, n)
    A1 = y + rng.normal(0, 0.5, n)        # A0→A1 명백 개선
    A2 = A1 + rng.normal(0, 0.01, n)      # A1→A2 미미 (feature 노이즈)
    A3 = A2 + rng.normal(0, 0.01, n)      # A2→A3 미미 (AR 노이즈)
    res = ladder_deltas({"A0": A0, "A1": A1, "A2": A2, "A3": A3}, y)
    steps = {s["label"]: s for s in res["steps"]}
    assert steps["ΔHP/preproc"]["better"] == "improve", "A0→A1 명백개선은 improve"
    assert steps["ΔFeature"]["better"] == "ns", "A1→A2 미미는 ns(노이즈)"
    assert res["n"] == n


def test_factorial_main_effect_and_interaction():
    """2^3 요인설계: HP 주효과 명백 + feature 는 HP 있을 때만 도움(상호작용)."""
    rng = np.random.default_rng(11)
    n = 68
    y = rng.normal(10, 2, n)
    base_err = rng.normal(0, 3, n)          # 공통 noise (paired)
    cells = {}
    for p in (0, 1):
        for hh in (0, 1):
            for f in (0, 1):
                err = base_err.copy()
                if hh:
                    err = err * 0.3          # HP on → 오차 크게 감소 (주효과)
                if f and hh:
                    err = err * 0.5          # feature 는 HP on 일 때만 추가 개선 (상호작용)
                cells[(p, hh, f)] = y + err
    res = factorial_effects(cells, y)
    eff = {m["factor"]: m for m in res["main"]}
    assert eff["hp"]["effect"] > 0 and eff["hp"]["sig"] == "yes", "HP 주효과 유의해야"
    inter = {d["pair"]: d["effect"] for d in res["interactions"]}
    assert inter["hp:feature"] > 0, "feature 는 HP on 일 때 더 도움(시너지 >0)"
    assert res["n"] == n


def test_factorial_neutral_factor_not_significant():
    rng = np.random.default_rng(5)
    n = 68
    y = rng.normal(5, 1, n); base = rng.normal(0, 1, n)
    # preproc 완전 무효(어느 cell이든 동일 noise) → 주효과 ~0, 비유의
    cells = {(p, hh, f): y + base for p in (0, 1) for hh in (0, 1) for f in (0, 1)}
    res = factorial_effects(cells, y)
    eff = {m["factor"]: m for m in res["main"]}
    assert eff["preproc"]["sig"] == "no", "무효 요인은 비유의"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
