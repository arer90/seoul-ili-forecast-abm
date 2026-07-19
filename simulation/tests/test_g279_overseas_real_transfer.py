"""G-279 (2026-06-16, 3자 감사): OverseasTransfer 진짜 transfer + cap 회귀 가드.

옛 버전: encoder 가 forward 에 안 쓰여 phantom(무기여) + 출력 cap 無 → rolling r2 −107 폭발.
신: encoder(단일 ILI 시퀀스 동역학) 가 Seoul lag1-4 시퀀스를 받아 embedding → features concat
    → head. 출력 2×y_max cap.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from simulation.models.overseas_transfer import OverseasTransferForecaster

_FNAMES = ["ili_rate_lag1", "ili_rate_lag2", "ili_rate_lag3", "ili_rate_lag4",
           "feat_a", "feat_b", "feat_c", "feat_d"]


def _fit_small(seed=0, **kw):
    rng = np.random.RandomState(seed)
    X = np.abs(rng.randn(120, 8) * 3 + 10).astype(np.float32)
    y = np.abs(X[:, 0] * 1.5 + rng.randn(120) * 2 + 8).astype(np.float32)
    m = OverseasTransferForecaster(epochs_pretrain=3, epochs_finetune=5,
                                   hidden_dim=16, batch_size=16, **kw)
    m.fit(X[:90], y[:90], feature_names=_FNAMES)
    return m, X, y


def test_encoder_wired_into_model():
    """encoder 가 forward 모델의 자식 (phantom 아님)."""
    m, _, _ = _fit_small()
    assert "encoder" in dict(m._model.named_children()), "encoder 미배선 (phantom)"
    assert m._lag_indices == [3, 2, 1, 0], f"lag 시퀀스 인덱스(oldest→newest) 오류: {m._lag_indices}"


def test_output_capped_no_explosion():
    """외삽서 출력이 2×y_max 로 bounded (옛 버전 pred 669 폭발 회귀 가드)."""
    m, X, _ = _fit_small()
    pred = m.predict(X[90:] * 4)   # 강한 외삽
    assert np.all(np.isfinite(pred))
    assert pred.max() <= 2.0 * m._y_max + 1e-3, f"폭발: {pred.max()} > 2×{m._y_max}"
    assert pred.min() >= 0.0


def test_encoder_contributes_to_prediction():
    """lag 시퀀스 교란 → 예측 변화 = encoder 가 실제 기여 (phantom 이면 불변)."""
    m, X, _ = _fit_small(seed=1)
    Xte = X[90:100].copy()
    p1 = m.predict(Xte)
    Xte2 = Xte.copy()
    Xte2[:, [0, 1, 2, 3]] += 8.0   # lag1-4 (ILI 시퀀스) 교란
    p2 = m.predict(Xte2)
    assert not np.allclose(p1, p2), "lag 교란에 예측 불변 = encoder 미기여(phantom)"


def test_no_lag_features_graceful():
    """lag features 없으면 transfer 생략(feature-only) — crash 안 함."""
    rng = np.random.RandomState(2)
    X = np.abs(rng.randn(100, 5) * 3 + 10).astype(np.float32)
    y = np.abs(X[:, 0] * 1.5 + 8).astype(np.float32)
    m = OverseasTransferForecaster(epochs_pretrain=2, epochs_finetune=3, hidden_dim=8)
    m.fit(X[:80], y[:80], feature_names=["a", "b", "c", "d", "e"])  # lag 없음
    assert m._lag_indices == []
    pred = m.predict(X[80:])
    assert np.all(np.isfinite(pred)) and pred.min() >= 0
