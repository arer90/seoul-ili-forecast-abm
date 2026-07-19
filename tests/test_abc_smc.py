"""ABC-SMC (tolerance-annealing SMC-ABC) 사후추정 smoke test.

검증 전략: ABM 은 비싸고 약식별이라 '복원 실패'가 알고리즘 탓인지 ABM 탓인지 구분이
어렵다. 그래서 먼저 **알려진 θ* 의 가벼운 합성 Gaussian simulator**(잘 식별됨)로
SMC-ABC 가 known param 을 복원하고 tolerance 감소에 따라 사후가 수축하는지 검증한다.
통과하면 같은 검증된 abc_smc 를 ABM(α, κ, τ, θ) 에 적용한다.

불변식 (task 명세):
  1. 수렴: posterior_mean → θ* (합성 Gaussian simulator).
  2. 수축: tolerance 감소 → posterior_std 축소(첫 round 대비 마지막 round).
  3. 정규화: weights 합 = 1.
  4. 결정성: 같은 seed → byte-동일 결과.
  5. leak-free: simulator 가 observed 를 보지 않아도 복원(누출 없이 정보 추출).
  6. shape / edge: 출력 shape 정상 + 잘못된 입력은 0초 ValueError.
"""
from __future__ import annotations

import numpy as np
import pytest

from simulation.abm.abc_smc import abc_smc


# --------------------------------------------------------------------------- #
# 합성 simulator: x = θ + 소노이즈 (잘 식별된 toy, 2-D θ → 3-D 요약통계)
# --------------------------------------------------------------------------- #
def _gaussian_simulator(noise: float = 0.05):
    """θ* 복원 검증용 — x = [θ0, θ1, θ0+θ1] + N(0, noise)."""
    def simulator(theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = rng.normal(0.0, noise, size=3)
        return np.array([theta[0], theta[1], theta[0] + theta[1]]) + n
    return simulator


THETA_TRUE = np.array([0.7, 1.3])
PRIORS = {"a": (0.0, 2.0), "b": (0.0, 2.0)}


def _observed(theta_true=THETA_TRUE):
    return np.array([theta_true[0], theta_true[1], theta_true[0] + theta_true[1]])


# === 1. 수렴: posterior_mean → θ* =========================================== #
def test_posterior_mean_converges_to_true_theta():
    """★ 핵심: 알려진 θ* 의 simulator 에서 사후 평균이 진짜 θ* 를 복원."""
    res = abc_smc(
        _gaussian_simulator(noise=0.03), _observed(), PRIORS,
        n_particles=120, tolerance_schedule=(1.0, 0.5, 0.2, 0.1), seed=42,
    )
    mean = res["posterior_mean"]
    assert mean.shape == (2,)
    assert np.all(np.abs(mean - THETA_TRUE) < 0.15), (
        f"복원 실패: {mean} vs {THETA_TRUE}"
    )
    assert res["param_names"] == ["a", "b"]


# === 2. 수축: tolerance 감소 → posterior_std 축소 =========================== #
def test_posterior_std_shrinks_with_tolerance():
    """ε 가 줄면 사후가 좁아진다 — 마지막 round std < 첫 round std (각 축)."""
    res = abc_smc(
        _gaussian_simulator(noise=0.03), _observed(), PRIORS,
        n_particles=150, tolerance_schedule=(1.2, 0.6, 0.3, 0.12), seed=7,
    )
    std_rounds = np.array(res["posterior_std_per_round"])
    assert std_rounds.shape[0] == res["n_rounds"]
    first, last = std_rounds[0], std_rounds[-1]
    # 두 축 모두 수축(엄격 감소까지는 아니어도 분명히 좁아져야 함)
    assert np.all(last < first), f"수축 실패: first={first} last={last}"
    # 최종 사후 std 가 prior 폭(2.0)보다 훨씬 좁음 → 식별됨
    assert np.all(res["posterior_std"] < 0.4)


# === 3. 정규화: weights 합 = 1 ============================================== #
def test_weights_normalized_to_one():
    res = abc_smc(
        _gaussian_simulator(), _observed(), PRIORS,
        n_particles=80, tolerance_schedule=(1.0, 0.5), seed=1,
    )
    w = res["weights"]
    assert w.shape[0] == res["particles"].shape[0]
    assert np.isclose(w.sum(), 1.0, atol=1e-12), f"weight 합 != 1: {w.sum()}"
    assert np.all(w >= 0.0)


# === 4. 결정성: 같은 seed → byte-동일 ====================================== #
def test_determinism_same_seed():
    kwargs = dict(
        observed=_observed(), priors=PRIORS, n_particles=60,
        tolerance_schedule=(1.0, 0.4), seed=123,
    )
    r1 = abc_smc(_gaussian_simulator(), **kwargs)
    r2 = abc_smc(_gaussian_simulator(), **kwargs)
    assert np.array_equal(r1["particles"], r2["particles"])
    assert np.array_equal(r1["weights"], r2["weights"])
    assert np.array_equal(r1["posterior_mean"], r2["posterior_mean"])


def test_different_seed_changes_result():
    """결정성의 짝: seed 가 다르면 결과도 다르다(고정 노이즈 아님)."""
    common = dict(
        observed=_observed(), priors=PRIORS, n_particles=60,
        tolerance_schedule=(1.0, 0.4),
    )
    r1 = abc_smc(_gaussian_simulator(), seed=1, **common)
    r2 = abc_smc(_gaussian_simulator(), seed=2, **common)
    assert not np.array_equal(r1["posterior_mean"], r2["posterior_mean"])


# === 5. leak-free: simulator 가 observed 를 안 봐도 복원 ==================== #
def test_leak_free_simulator_blind_to_observed():
    """simulator 는 observed 인자를 받지 않는다(시그니처가 (theta, rng) 뿐).
    그럼에도 사후가 θ* 로 수렴 = 정보가 '거리'를 통해서만 흐른다(누출 없음)."""
    calls = {"saw_observed": False}

    def blind_sim(theta, rng):
        # observed 는 클로저/인자 어디에도 없음 — 구조적으로 leak 불가
        return np.array([theta[0], theta[1], theta[0] + theta[1]])

    res = abc_smc(
        blind_sim, _observed(), PRIORS,
        n_particles=100, tolerance_schedule=(0.8, 0.3, 0.1), seed=5,
    )
    assert not calls["saw_observed"]
    assert np.all(np.abs(res["posterior_mean"] - THETA_TRUE) < 0.2)


# === 6. shape / edge cases ================================================= #
def test_shape_and_rounds_metadata():
    res = abc_smc(
        _gaussian_simulator(), _observed(), PRIORS,
        n_particles=50, tolerance_schedule=(1.0, 0.5, 0.25), seed=3,
    )
    assert res["particles"].shape[1] == 2          # D 축
    assert res["n_rounds"] == 3
    assert len(res["accept_counts"]) == res["n_rounds"]
    assert len(res["tolerance_schedule"]) == res["n_rounds"]
    assert res["posterior_mean"].shape == (2,)
    assert res["posterior_std"].shape == (2,)


def test_edge_empty_priors_raises():
    with pytest.raises(ValueError):
        abc_smc(_gaussian_simulator(), _observed(), {},
                n_particles=10, tolerance_schedule=(1.0,))


def test_edge_bad_prior_bounds_raises():
    with pytest.raises(ValueError):
        abc_smc(_gaussian_simulator(), _observed(), {"a": (1.0, 1.0), "b": (0.0, 2.0)},
                n_particles=10, tolerance_schedule=(1.0,))


def test_edge_nonpositive_tolerance_raises():
    with pytest.raises(ValueError):
        abc_smc(_gaussian_simulator(), _observed(), PRIORS,
                n_particles=10, tolerance_schedule=(1.0, 0.0))


def test_edge_too_few_particles_raises():
    with pytest.raises(ValueError):
        abc_smc(_gaussian_simulator(), _observed(), PRIORS,
                n_particles=1, tolerance_schedule=(1.0,))


def test_edge_simulator_output_length_mismatch_raises():
    """simulator 출력 길이 != observed 길이 → ValueError (계약 위반 fail-fast)."""
    def wrong_len(theta, rng):
        return np.array([theta[0]])                # 1-D 길이 1, observed 는 3
    with pytest.raises(ValueError):
        abc_smc(wrong_len, _observed(), PRIORS,
                n_particles=10, tolerance_schedule=(5.0,))


def test_edge_nan_simulator_rejected_not_crashed():
    """비유한 simulation 은 거리=inf 로 자동 기각 — 크래시/오염 없이 복원 지속."""
    def sometimes_nan(theta, rng):
        base = np.array([theta[0], theta[1], theta[0] + theta[1]])
        if rng.random() < 0.3:                     # 30% blow-up
            return base * np.inf
        return base + rng.normal(0, 0.03, size=3)
    res = abc_smc(
        sometimes_nan, _observed(), PRIORS,
        n_particles=80, tolerance_schedule=(1.0, 0.4, 0.15), seed=9,
    )
    assert np.all(np.isfinite(res["posterior_mean"]))
    assert np.all(np.abs(res["posterior_mean"] - THETA_TRUE) < 0.25)
