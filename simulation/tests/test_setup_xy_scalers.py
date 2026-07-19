"""Sprint α R1 (2026-05-26): setup_xy_scalers SoT tests.

base.setup_xy_scalers 는 8 dl_models + 3 epi_models + 1 overseas_transfer 의
StandardScaler X+y bootstrap 의 SoT. Pickle ABI 가 ChampionArtifact (7 consumer
including server/mcp_epi.py:1164) 에 의존하므로 attribute 이름 + 반환 순서 잠금.
"""
from __future__ import annotations

import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler

from simulation.models.base import setup_xy_scalers


def test_returns_4_tuple_in_canonical_order():
    """Pickle ABI: 정확히 (sx, sy, X_s, y_s) 4-tuple — caller 순서 의존."""
    X = np.random.randn(20, 3)
    y = np.random.randn(20)
    result = setup_xy_scalers(X, y)
    assert len(result) == 4
    sx, sy, X_s, y_s = result
    assert isinstance(sx, StandardScaler)
    assert isinstance(sy, StandardScaler)
    assert isinstance(X_s, np.ndarray)
    assert isinstance(y_s, np.ndarray)


def test_X_scaled_zero_mean_unit_std():
    """X 가 standard scaling 적용 — column-wise mean≈0, std≈1."""
    rng = np.random.default_rng(42)
    X = rng.normal(loc=5.0, scale=3.0, size=(100, 4))
    y = rng.normal(size=100)
    sx, sy, X_s, y_s = setup_xy_scalers(X, y)
    assert np.allclose(X_s.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(X_s.std(axis=0), 1, atol=1e-10)


def test_y_scaled_1d_not_2d():
    """y_s 는 ravel 된 1-D (sklearn .reshape(-1, 1) 후 .ravel())."""
    X = np.random.randn(10, 2)
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    sx, sy, X_s, y_s = setup_xy_scalers(X, y)
    assert y_s.ndim == 1
    assert len(y_s) == len(y)


def test_scalers_are_independent_instances():
    """sx 와 sy 가 별개 — 같은 객체 X."""
    X = np.random.randn(10, 2)
    y = np.random.randn(10)
    sx, sy, _, _ = setup_xy_scalers(X, y)
    assert sx is not sy


def test_inference_replay_via_stored_scalers():
    """fitted scalers 가 새 X / y_pred 에 적용 가능 (champion replay 패턴)."""
    rng = np.random.default_rng(1)
    X_train = rng.normal(size=(50, 3))
    y_train = rng.normal(size=50)
    sx, sy, X_s, y_s = setup_xy_scalers(X_train, y_train)

    # New inference X
    X_new = rng.normal(size=(5, 3))
    X_new_s = sx.transform(X_new)
    assert X_new_s.shape == (5, 3)

    # Inverse y (champion .pt replay 패턴)
    y_pred_s = np.array([0.5, -1.2, 0.0, 1.0, -0.5])
    y_pred = sy.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
    assert y_pred.shape == (5,)


def test_scalers_are_picklable():
    """ChampionArtifact 에 pickle 저장됨 — fitted state 보존."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 2))
    y = rng.normal(size=30)
    sx, sy, _, _ = setup_xy_scalers(X, y)
    # Round-trip
    sx_p = pickle.loads(pickle.dumps(sx))
    sy_p = pickle.loads(pickle.dumps(sy))
    # Pickled scaler 가 동일 transform 결과
    X_test = rng.normal(size=(3, 2))
    assert np.allclose(sx.transform(X_test), sx_p.transform(X_test))


def test_accepts_2d_y_input():
    """y 가 (n, 1) shape 도 처리 (caller flexibility)."""
    X = np.random.randn(10, 2)
    y_2d = np.random.randn(10, 1)
    sx, sy, X_s, y_s = setup_xy_scalers(X, y_2d)
    assert y_s.ndim == 1
    assert len(y_s) == 10


def test_accepts_list_input():
    """X / y 가 list 도 처리 (np.asarray 변환)."""
    X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    y = [10.0, 20.0, 30.0]
    sx, sy, X_s, y_s = setup_xy_scalers(X, y)
    assert X_s.shape == (3, 2)
    assert y_s.shape == (3,)
