"""TiRex — xLSTM-based time-series foundation model (NX-AI, 2025).

웹 SOTA 커버리지 감사(G-265) 후 사용자 확정 add. TiRex(35M, xLSTM)는 2025년 한때 GIFT-Eval·Chronos-ZS
동시 1위, short-horizon 특화. **transformers 의존 없음(xLSTM)** → 메인 env(mlx-lm/ARIA) 충돌 0
(Chronos 가 퇴출된 바로 그 충돌을 회피). 실측: ILI zero-shot rolling 1-step r2=**0.944 (전 모델 최고)**
> TimesFM-2.5 0.939 > DLinear 0.935 > ARIMA 0.915.

왜 우리에 맞나
-------------
short-horizon 특화 = 우리 1-step 운영예측(phase 12)과 정렬. 35M 소형 + CPU/macOS 지원.
zero-shot foundation 이라 341주 소표본 한계를 사전학습으로 우회 (TimesFM 과 같은 각도, 다른 아키텍처).

인터페이스 (TimesFM/Chronos 와 동일)
----------------------------------
TimeSeriesForecaster (USES_FEATURES=False). fit_series=context 저장+모델 lazy-load,
forecast(steps)=직접 다단계 (TimeSeriesForecaster.predict 가 호출). 라이선스: tirex-ts(NX-AI) 오픈.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from simulation.models.base import ModelMeta, REGISTRY, TimeSeriesForecaster

log = logging.getLogger(__name__)

_HAS_TIREX = False
try:
    import tirex  # noqa: F401
    _HAS_TIREX = True
except ImportError:
    log.debug("tirex-ts 미설치 — TiRexForecaster 등록은 되나 fit 시 ImportError")


def _w_tirex_available() -> bool:
    return _HAS_TIREX


class TiRexForecaster(TimeSeriesForecaster):
    """TiRex (NX-AI xLSTM foundation) — zero-shot, transformers-free. TimesFM 과 동일 인터페이스.

    Caller responsibility: y ≥ 0. Performance: fit O(1)(context 저장)+load, forecast=xLSTM 추론.
    Side effects: 최초 fit 시 NX-AI/TiRex(35M) HF 다운로드(공개). HF_HUB_DISABLE_TELEMETRY 권장.
    """

    USES_FEATURES = False
    meta = ModelMeta(
        name="TiRex",
        category="dl",
        level=16,
        min_data=40,
        description="TiRex (NX-AI xLSTM 35M foundation, 2025). zero-shot, transformers-free → 메인 "
                    "env 네이티브. short-horizon SOTA — ILI rolling r2=0.944(전 모델 최고, G-265).",
        requires_gpu=False,
        dependencies=["tirex-ts"],
    )

    DEFAULT_REPO = "NX-AI/TiRex"

    def __init__(self, repo_id: str = DEFAULT_REPO, max_context: int = 512):
        super().__init__()
        self._repo_id = repo_id
        self._max_context = int(max_context)
        self._model = None
        self._context: Optional[np.ndarray] = None

    def fit_series(self, series: np.ndarray, **kwargs) -> "TiRexForecaster":
        if not _HAS_TIREX:
            raise ImportError("tirex-ts 미설치 — `uv pip install tirex-ts`")
        if self._model is None:
            from tirex import load_model
            log.info(f"  [TiRex] load_model: {self._repo_id}")
            self._model = load_model(self._repo_id, device="cpu")
        self._context = np.asarray(series, dtype=np.float32).ravel()
        self._fitted = True
        return self

    def forecast(self, steps: int, **kwargs) -> np.ndarray:
        if self._model is None or not self._fitted:
            raise RuntimeError("TiRex: fit_series() 먼저 호출")
        if self._context is None or len(self._context) == 0:
            raise ValueError("TiRex: context series 없음")
        import torch
        h = int(steps)
        ctx = torch.tensor(self._context[-self._max_context:], dtype=torch.float32).unsqueeze(0)
        try:
            _q, mean = self._model.forecast(context=ctx, prediction_length=h)
            pred = np.asarray(mean, dtype=np.float32).ravel()[:h]
        except Exception as e:
            log.error(f"  [TiRex] forecast 실패: {e}")
            raise
        if len(pred) < h:
            pred = np.concatenate([pred, np.full(h - len(pred), pred[-1] if len(pred) else 0.0,
                                                 dtype=np.float32)])
        return np.clip(pred[:h], 0.0, None).astype(np.float32)


REGISTRY.register(TiRexForecaster)
