"""DLinear — decomposition + single linear layer (Zeng et al., AAAI 2023).

"Are Transformers Effective for Time Series Forecasting?" 의 그 baseline. 시계열을 trend(이동평균)
+ seasonal(잔차)로 분해하고 각각에 **단일 선형층**을 적용 → 합. 파라미터가 극히 적어 소표본에서
가장 강건한 deep-TS 대표 baseline (9개 LTSF 벤치서 복잡 Transformer 를 큰 폭 능가, 'simple beats
complex' 패러다임의 대표). G-265 (2026-06-13, 웹 SOTA 커버리지 감사 후 사용자 확정 add).

왜 추가
------
우리 핵심 결론("소표본 341주에선 단순 모델이 deep 을 이긴다" — 실측 TimeMixer 0.366 ≪ ARIMA 0.915)을
**직접 입증하는 대표 baseline**. 파라미터 최소 → 과적합 위험 ≈ 0. ElasticNet/Theta 가 기능적으로
선형·분해를 partial 커버하나, 문헌 표준 DLinear 를 명시 보유하면 논거가 강화됨.

구현
----
TimeSeriesForecaster (USES_FEATURES=False, y 시계열만). channel-independent: 단변량 ILI 에 자연.
fit_series(y): lookback L 윈도우 → moving-avg 분해 → [trend, seasonal] 에 선형(lstsq) → 1-step.
forecast(steps): recursive 1-step (자기 예측 feed-back) — 다른 TS 모델과 동일 convention.
"""
from __future__ import annotations

import logging

import numpy as np

from simulation.models.base import ModelMeta, REGISTRY, TimeSeriesForecaster

log = logging.getLogger(__name__)


def _moving_avg(x: np.ndarray, k: int) -> np.ndarray:
    """이동평균 trend (kernel k, 양끝 edge-pad) — DLinear series_decomp 의 trend."""
    if k < 2 or len(x) < 2:
        return x.astype(float)
    k = min(k, len(x))
    if k % 2 == 0:
        k += 1
    pad = k // 2
    xp = np.concatenate([np.full(pad, x[0]), x.astype(float), np.full(pad, x[-1])])
    kern = np.ones(k) / k
    return np.convolve(xp, kern, mode="valid")[: len(x)]


class DLinearForecaster(TimeSeriesForecaster):
    """DLinear — series decomposition + single linear layer (Zeng 2023). 소표본 강건 baseline.

    Caller responsibility: y ≥ 0 (ILI rate). Performance: O(n·L) fit (lstsq), O(steps·L) forecast.
    """

    USES_FEATURES = False
    meta = ModelMeta(
        name="DLinear",
        category="dl",
        level=10,
        min_data=40,
        description="DLinear (Zeng AAAI 2023): trend/seasonal 분해 + 단일 선형층. 소표본 강건 — "
                    "'simple beats complex' 대표 baseline (G-265).",
        requires_gpu=False,
        dependencies=[],
    )

    def __init__(self, lookback: int = 26, kernel_size: int = 13):
        super().__init__()
        self._L = int(lookback)
        self._k = int(kernel_size)
        self._W_trend = None
        self._W_seas = None
        self._b = 0.0
        self._last = None

    def fit_series(self, series: np.ndarray, **kwargs) -> "DLinearForecaster":
        y = np.asarray(series, dtype=float).ravel()
        n = len(y)
        self._y_train_max = float(np.max(y)) if n else 100.0  # G-289 외삽 cap
        L = min(self._L, max(2, n // 3))     # 소표본 대비 lookback cap
        self._L = L
        if n <= L + 2:                        # 표본 부족 → 평균 fallback
            self._W_trend = np.zeros(L); self._W_seas = np.zeros(L)
            self._b = float(y.mean()); self._last = y[-L:]
            self._fitted = True
            return self
        # 샘플: window=y[t-L:t] → 분해 → [trend(L), seasonal(L), 1] → y[t]
        feats, tgts = [], []
        for t in range(L, n):
            w = y[t - L:t]
            tr = _moving_avg(w, self._k)
            se = w - tr
            feats.append(np.concatenate([tr, se, [1.0]]))
            tgts.append(y[t])
        A = np.asarray(feats); b = np.asarray(tgts)
        # ridge-안정화 lstsq (소표본 p=2L+1 ≈ n 대비 약한 정규화). errstate: 큰 값 trace overflow 경고 차단.
        with np.errstate(all="ignore"):
            G = A.T @ A
            lam = 1e-3 * np.trace(G) / A.shape[1]
            try:
                coef = np.linalg.solve(G + lam * np.eye(A.shape[1]), A.T @ b)
            except np.linalg.LinAlgError:
                # G-275 base layer: 특이행렬 fallback 도 safe_lstsq(rcond=1e-6 + ridge) — rcond=None 폭발 차단
                from simulation.models.safety import safe_lstsq
                coef = safe_lstsq(A, b)
        self._W_trend = coef[:L]; self._W_seas = coef[L:2 * L]; self._b = float(coef[-1])
        self._last = y[-L:].copy()
        self._fitted = True
        log.info(f"  [DLinear] fit 완료: n={n}, lookback={L}")
        return self

    def forecast(self, steps: int, **kwargs) -> np.ndarray:
        if not self._fitted or self._last is None:
            raise RuntimeError("DLinear: fit_series() 먼저 호출")
        hist = self._last.astype(float).copy()
        preds = []
        for _ in range(int(steps)):
            tr = _moving_avg(hist, self._k)
            se = hist - tr
            p = float(tr @ self._W_trend + se @ self._W_seas + self._b)
            preds.append(p)
            hist = np.roll(hist, -1); hist[-1] = p     # recursive 1-step
        from simulation.models.safety import apply_extrapolation_cap  # G-289
        _p = np.clip(np.asarray(preds, dtype=np.float32), 0.0, None)
        return apply_extrapolation_cap(_p, getattr(self, "_y_train_max", None)).astype(np.float32)

    def rolling_1step(self, y_observed: np.ndarray, **kwargs) -> np.ndarray:
        """G-327c: cheap rolling-origin 1-step — 학습 가중치를 **관측 슬라이딩 윈도**에 적용(refit 0).

        forecast 의 recursive self-feeding(hist[-1]=p → mean-revert→음수)을, 매 step 관측값으로 윈도를
        갱신해 대체. base rolling_1step(refit-per-step) 대비 NN/lstsq 재학습 없어 빠름.
        G-344 (2026-06-24): DLinear 는 identity(HIER_none) 라 ROLLING_EVAL_MODELS 로 이동 — baseline+R9
        양쪽 raw y_observed rolling(옛 BASELINE_ROLLING 단일원점 → R9 oof=inf 제외 버그 해소).

        Args:
            y_observed: (n_test,) 관측 hold-out y(raw). i 예측엔 y_observed[:i] 만 = leak-free 1-step.

        Returns:
            (n_test,) 1-step rolling 예측.

        Performance: O(n_test · L), 재학습 없음. Side effects: 없음(self 상태 불변).
        """
        if not self._fitted or self._last is None:
            return self.forecast(len(y_observed), **kwargs)
        yo = np.asarray(y_observed, dtype=float).ravel()
        hist = self._last.astype(float).copy()       # train 마지막 L-윈도
        preds = []
        for i in range(len(yo)):
            tr = _moving_avg(hist, self._k)
            se = hist - tr
            preds.append(float(tr @ self._W_trend + se @ self._W_seas + self._b))
            hist = np.roll(hist, -1); hist[-1] = float(yo[i])   # 관측값 슬라이딩(self-feeding 아님)
        from simulation.models.safety import apply_extrapolation_cap  # G-289
        _p = np.clip(np.asarray(preds, dtype=np.float32), 0.0, None)
        return apply_extrapolation_cap(_p, getattr(self, "_y_train_max", None)).astype(np.float32)


REGISTRY.register(DLinearForecaster)
