"""Smoke tests for the ABM neural surrogate (simulation/abm/surrogate.py).

느린 agent kernel(run_agent_world)을 빠른 NN 으로 emulate 하는 surrogate 의 불변식 검증:
  - generate_training_data 의 shape / count 보존(>=0, 인구합 보존) / 결정성
  - ABMSurrogate fit/predict shape · 단일벡터 · 음수 clip
  - held-out R² > 0 (학습됨 증명, leak-free: test 파라미터는 학습 미사용)
  - surrogate predict 가 kernel 보다 빠름(실측 배율 > 1)
  - edge: unknown param 거부, fit 전 predict 거부
  - sklearn fallback 이 같은 인터페이스로 동작

macOS: per-file 실행 (`.venv/bin/python -m pytest tests/test_abm_surrogate.py -x -q`).
"""
from __future__ import annotations

import numpy as np
import pytest

from simulation.abm.surrogate import (
    ABMSurrogate,
    COMPARTMENTS,
    PARAM_NAMES,
    TORCH_AVAILABLE,
    generate_training_data,
    surrogate_vs_kernel_speedup,
)

# 소규모 PoC 설정(빠른 테스트). N/T 는 동역학이 살아있을 만큼만.
N_AGENTS = 1500
T_DAYS = 40


# --- 1. 학습 데이터 생성: shape / 보존 / 결정성 -------------------------------
def test_generate_training_data_shapes():
    """params (n,d) + trajectories (n,T,k) 정확한 shape."""
    X, Y = generate_training_data(8, seed=1, N_agents=N_AGENTS, T_days=T_DAYS)
    assert X.shape == (8, len(PARAM_NAMES))
    assert Y.shape == (8, T_DAYS, len(COMPARTMENTS))
    assert np.all(np.isfinite(X)) and np.all(np.isfinite(Y))


def test_generate_training_data_counts_nonneg_and_conserved():
    """count 는 모두 >= 0 이고, 매 시점 compartment 합 = N_agents (보존)."""
    X, Y = generate_training_data(6, seed=2, N_agents=N_AGENTS, T_days=T_DAYS)
    assert np.all(Y >= 0), "compartment count 는 음수일 수 없다"
    totals = Y.sum(axis=2)  # (n, T) — S+E+I+R+V+D
    assert np.allclose(totals, N_AGENTS), "매 시점 인구 합은 N_agents 로 보존"


def test_generate_training_data_deterministic():
    """같은 seed/인자 = 비트 동일 (np.random.default_rng 결정성)."""
    a = generate_training_data(5, seed=7, N_agents=N_AGENTS, T_days=T_DAYS)
    b = generate_training_data(5, seed=7, N_agents=N_AGENTS, T_days=T_DAYS)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    # 다른 seed = 달라야(파라미터 샘플이 바뀜)
    c = generate_training_data(5, seed=8, N_agents=N_AGENTS, T_days=T_DAYS)
    assert not np.array_equal(a[0], c[0])


# --- 2. ABMSurrogate fit/predict 기본 ----------------------------------------
def test_surrogate_fit_predict_shape_and_single_vector():
    """predict 가 (n,T,k) 반환, 단일 param 벡터는 (T,k), 음수 없음."""
    X, Y = generate_training_data(20, seed=3, N_agents=N_AGENTS, T_days=T_DAYS)
    sur = ABMSurrogate(seed=0, max_epochs=120).fit(X, Y)
    pred = sur.predict(X)
    assert pred.shape == (20, T_DAYS, len(COMPARTMENTS))
    assert np.all(pred >= 0), "예측 count 는 0 으로 clip"
    single = sur.predict(X[0])
    assert single.shape == (T_DAYS, len(COMPARTMENTS))
    # 단일벡터 ≈ 배치 행 (torch float32 BLAS 경로 차이로 ~1e-5 상대오차 허용)
    assert np.allclose(single, pred[0], rtol=1e-3, atol=1e-2)


# --- 3. held-out R² > 0 (학습됨 증명, leak-free) ------------------------------
def test_surrogate_heldout_r2_positive_leakfree():
    """학습/평가 파라미터를 시드로 분리(leak-free). held-out 궤적 R² > 0 = 학습됨."""
    X_tr, Y_tr = generate_training_data(40, seed=100, N_agents=N_AGENTS, T_days=T_DAYS)
    X_te, Y_te = generate_training_data(12, seed=999, N_agents=N_AGENTS, T_days=T_DAYS)
    # leak-free 보증: test 파라미터가 train 에 없음
    for row in X_te:
        assert not np.any(np.all(np.isclose(X_tr, row), axis=1)), "test param leaked into train"
    sur = ABMSurrogate(seed=0).fit(X_tr, Y_tr)
    r2 = sur.score_r2(X_te, Y_te)
    assert r2 > 0.0, f"held-out R² 가 양수여야 학습 증명; got {r2}"


# --- 4. 속도: surrogate predict 가 kernel 보다 빠름 ---------------------------
def test_surrogate_faster_than_kernel():
    """실측 speedup > 1 (surrogate 가 kernel 보다 빠름) + R² > 0 동시 확인."""
    res = surrogate_vs_kernel_speedup(
        n_train=40, n_test=10, seed=0, N_agents=N_AGENTS, T_days=T_DAYS, n_timing_repeats=3
    )
    assert res["speedup"] > 1.0, f"surrogate 가 kernel 보다 빨라야 함; got {res}"
    assert res["r2"] > 0.0, f"held-out R² 양수; got {res}"
    assert res["surrogate_time_s"] < res["kernel_time_s"]


# --- 5. edge: 잘못된 입력 거부 -----------------------------------------------
def test_generate_rejects_unknown_param_and_bad_sizes():
    with pytest.raises(ValueError):
        generate_training_data(4, seed=0, param_names=("beta", "BOGUS"))
    with pytest.raises(ValueError):
        generate_training_data(0, seed=0)
    with pytest.raises(ValueError):
        generate_training_data(3, seed=0, T_days=0)


def test_predict_before_fit_and_bad_dims_raise():
    sur = ABMSurrogate()
    with pytest.raises(RuntimeError):
        sur.predict(np.zeros(len(PARAM_NAMES)))
    X, Y = generate_training_data(6, seed=4, N_agents=N_AGENTS, T_days=T_DAYS)
    sur.fit(X, Y)
    with pytest.raises(ValueError):
        sur.predict(np.zeros((2, len(PARAM_NAMES) + 3)))  # wrong width
    # fit shape guards
    sur2 = ABMSurrogate()
    with pytest.raises(ValueError):
        sur2.fit(X[:3], Y)  # row mismatch


# --- 6. sklearn fallback parity ----------------------------------------------
def test_sklearn_fallback_same_interface():
    """prefer_torch=False → sklearn 백엔드로도 fit/predict + R²>0 동작(인터페이스 동일)."""
    X_tr, Y_tr = generate_training_data(40, seed=55, N_agents=N_AGENTS, T_days=T_DAYS)
    X_te, Y_te = generate_training_data(12, seed=5555, N_agents=N_AGENTS, T_days=T_DAYS)
    sur = ABMSurrogate(seed=0, prefer_torch=False).fit(X_tr, Y_tr)
    assert sur.backend == "sklearn"
    assert sur.n_jobs <= 2, "ENGINEERING_PRINCIPLES.md §2: n_jobs <= 2"
    pred = sur.predict(X_te)
    assert pred.shape == (12, T_DAYS, len(COMPARTMENTS))
    assert sur.score_r2(X_te, Y_te) > 0.0
