"""TDD — phase 4-12 BASIC eval-features 분리 (사용자 2026-06-02).

runner._resolve_eval_features 가 phase 4-12 용 BASIC subset 을 정확히 슬라이스하고
(phase 13 은 caller 가 full 을 넘기므로 분리), eval_basic=False / BASIC 부재 시 안전하게
full 로 fallback 함을 고정.

run PER-FILE on macOS: KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 pytest <thisfile>
"""
import numpy as np
import pytest

from simulation.pipeline.runner import _resolve_eval_features


def _toy(n=8, extra=("noise_a", "noise_b", "ili_rate_lag3", "rt_x")):
    from simulation.pipeline.baseline import BASIC_FEATURE_COLS
    cols = list(BASIC_FEATURE_COLS) + list(extra)
    X = np.arange(n * len(cols)).reshape(n, len(cols)).astype(float)
    return X, cols, list(BASIC_FEATURE_COLS)


def test_basic_slices_only_basic_cols():
    X, cols, basic = _toy()
    Xe, cols_e, idx = _resolve_eval_features(X, cols, eval_basic=True)
    # 정확히 BASIC 컬럼만 (lag+계절성), noise/lag3/rt 제외
    assert cols_e == basic, f"BASIC subset 만이어야: got {cols_e}"
    assert idx == list(range(len(basic))), f"BASIC 가 앞쪽 인덱스: {idx}"
    assert Xe.shape == (X.shape[0], len(basic)), f"X_eval shape: {Xe.shape}"
    # 슬라이스가 원본 컬럼값 보존
    assert np.array_equal(Xe, X[:, idx])


def test_basic_excludes_noise_and_extra_features():
    X, cols, basic = _toy(extra=("weather_temp", "ili_rate_lag3", "subway_ili"))
    Xe, cols_e, idx = _resolve_eval_features(X, cols, eval_basic=True)
    for extra in ("weather_temp", "ili_rate_lag3", "subway_ili"):
        assert extra not in cols_e, f"{extra} 가 BASIC 에 들어감 (제외돼야)"
    assert len(cols_e) == len(basic)


def test_full_when_eval_basic_false():
    """MPH_EVAL_FEATURES=full → full 그대로, basic_idx=None (phase 13 와 동일 full)."""
    X, cols, _ = _toy()
    Xe, cols_e, idx = _resolve_eval_features(X, cols, eval_basic=False)
    assert idx is None
    assert cols_e == cols and Xe.shape == X.shape
    assert Xe is X  # 슬라이스 안 함 (동일 객체)


def test_full_fallback_when_no_basic_cols():
    """feature_cols 에 BASIC 이 하나도 없으면 안전하게 full (basic_idx=None)."""
    cols = ["x0", "x1", "x2"]
    X = np.zeros((5, 3))
    Xe, cols_e, idx = _resolve_eval_features(X, cols, eval_basic=True)
    assert idx is None and cols_e == cols and Xe.shape == X.shape


def test_phase13_independence_full_pool_preserved():
    """phase 13 분리 검증: eval_basic 여부와 무관하게 caller 가 full(X_all) 을 13 에 넘기면
    13 은 full pool. helper 는 4-12 용 X_eval 만 반환하고 원본 X_all 을 변형하지 않음."""
    X, cols, basic = _toy()
    X_orig = X.copy()
    Xe, _, idx = _resolve_eval_features(X, cols, eval_basic=True)
    assert np.array_equal(X, X_orig), "원본 X_all 이 변형됨 (13 의 full pool 오염)"
    assert Xe.shape[1] < X.shape[1], "X_eval(BASIC) 가 full 보다 좁아야"


def test_custom_basic_cols():
    X, cols, _ = _toy()
    Xe, cols_e, idx = _resolve_eval_features(X, cols, eval_basic=True,
                                             basic_cols=["ili_rate_lag1", "sin_month"])
    assert set(cols_e) == {"ili_rate_lag1", "sin_month"}


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
