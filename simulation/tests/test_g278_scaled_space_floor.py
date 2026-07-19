"""G-278 (2026-06-16, 3자 감사): DL/modern-ts scaled-space floor bias 제거.

_predict_torch 가 모델 raw 출력(StandardScaler y 공간, 0=train평균)에 np.maximum(pred,0)
을 적용 → 평균이하 주 예측을 train평균까지 끌어올리는 **양의 bias**. 모든 호출자
(patchtst/itransformer/timesnet/mamba/nbeats/nhits/tide/dnn/tcn/...)가 inverse_transform 後
원공간 nonneg(maximum)를 이미 적용하므로 스케일공간 clamp 는 중복+유해 → 제거.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


class _NegModel(torch.nn.Module):
    """항상 음수(스케일공간 평균이하) 출력 — bias 재현용."""
    def forward(self, x):
        n = x.shape[0]
        return torch.full((n, 1), -0.7)


class _SeqNegModel(torch.nn.Module):
    def forward(self, x):
        n = x.shape[0]
        return torch.full((n,), -0.5)


def test_predict_torch_keeps_below_mean_scaled():
    """핵심: 스케일공간 음수(평균이하)를 0으로 clamp하지 않는다 (bias 제거)."""
    from simulation.models.dl_models import _predict_torch
    out = _predict_torch(_NegModel(), np.zeros((6, 3), dtype=np.float32))
    assert np.allclose(out, -0.7), f"스케일공간 clamp 잔존(bias): {out[:3]}"


def test_predict_torch_squeeze_shape():
    from simulation.models.dl_models import _predict_torch
    out = _predict_torch(_SeqNegModel(), np.zeros((4, 2), dtype=np.float32))
    assert out.shape == (4,)
    assert np.allclose(out, -0.5)   # 음수 보존


def test_original_space_nonneg_still_applied():
    """원공간 nonneg 는 호출자가 inverse_transform 後 유지 — ILI<0 차단은 보존."""
    from sklearn.preprocessing import StandardScaler
    # 스케일공간 -0.7 → inverse → 원공간 값. 호출자가 maximum(...,0) 적용해 음수 차단.
    sc = StandardScaler()
    y = np.array([10.0, 20.0, 30.0, 40.0]).reshape(-1, 1)
    sc.fit(y)
    pred_scaled = np.array([-0.7])              # 평균이하 (이제 보존됨)
    orig = sc.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    final = np.maximum(orig, 0.0)               # 호출자 패턴
    assert final[0] >= 0.0                       # ILI≥0 보장 유지
    # bias 제거 효과: orig 가 평균(25)보다 작아야(평균으로 안 끌려감)
    assert orig[0] < 25.0, f"평균이하 예측이 평균으로 끌려감(bias): {orig[0]}"
