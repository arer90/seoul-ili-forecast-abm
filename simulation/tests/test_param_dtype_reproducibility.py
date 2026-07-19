"""TDD — float32 per-agent param이 재현성/정확도를 깨는가? (외부평가 메모리 권고 검증).

사용자 요청: 내가 '하면 안 된다'고 *주장*한 float32를 TDD로 *사실 확인*. 주장강등 금지.
가설: float32 param 저장은 메모리 절반 절감하나, 비트동일 재현성(max|Δ|=0)을 깰 수 있다.
이 test가 (1) float32 자기-재현성, (2) float32 vs float64 출력 발산을 측정해 채택/기각을
데이터로 결정한다. 채택 기준 = peak 상대차 < 1%(과학적 회귀 tol).
"""
import inspect

import numpy as np

from simulation.abm.agent_kernel import run_agent_world


def _run(dtype, seed=42, N=5000):
    return run_agent_world(N, T_days=80, beta=0.35, sigma=0.2, gamma=0.1,
                           delta=0.001, nu=0.0, global_seed=seed, theta_sd=0.2,
                           param_dtype=dtype)


def test_default_is_float32_bit_identical():
    """기본값 float32(TDD 검증 후 전환) — float64와 비트동일이라 기존 결과·재현성 불변."""
    assert inspect.signature(run_agent_world).parameters["param_dtype"].default is np.float32


def test_float32_is_self_reproducible():
    """① float32도 같은 seed면 비트동일 (재현성=결정성은 dtype 무관)."""
    a = np.asarray(_run(np.float32)["I"], float)
    b = np.asarray(_run(np.float32)["I"], float)
    assert np.array_equal(a, b), "float32가 같은 seed서도 비결정적이면 안 됨"


import pytest


@pytest.mark.parametrize("cfg", [
    {}, {"theta_sd": 0.3, "beta": 0.45}, {"alpha_mean": 1.5, "theta_sd": 0.1},
    {"beta": 0.6, "delta": 0.005}, {"N": 8000, "theta_sd": 0.25}])
def test_float32_bit_identical_to_float64(cfg):
    """② ★ TDD 결과(내 '하면 안 됨' 주장 반증): float32 param이 float64와 **비트동일**.
    per-agent param이 float32여도 하위 hazard/binomial 계산이 float64로 promote →
    난수 stream·감염 결과 불변 → 동일 궤적. 5 config 전부 max|Δ|=0 검증 (재현성 보존)."""
    base = dict(N=4000, T_days=70, beta=0.35, sigma=0.2, gamma=0.1, delta=0.001,
                nu=0.0, global_seed=42, theta_sd=0.2)
    base.update(cfg)
    f64 = np.asarray(run_agent_world(param_dtype=np.float64, **base)["I"], float)
    f32 = np.asarray(run_agent_world(param_dtype=np.float32, **base)["I"], float)
    assert np.array_equal(f64, f32), (
        f"float32≠float64 in {cfg} → 재현성 깸. (현재는 비트동일 = 메모리 절감 안전)")
