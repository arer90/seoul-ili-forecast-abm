"""TimesFM 2.5 wrapper — Google 의 foundation forecasting model (transformers-free).

목적
----
Chronos-2 의 **대체** foundation model (사용자 확정 2026-06-13). TimesFM 2.5 (200M, Google,
2026) 가 ILI 같은 weekly 시계열에 강함. **결정적 장점: transformers 의존이 전혀 없음**
(torch + huggingface_hub 만) → 메인 env(transformers 5.x + mlx-lm/ARIA) 와 충돌 0, 격리 venv 불필요.

왜 chronos 대신 (G-261, 2026-06-13)
-----------------------------------
chronos-forecasting 은 **모든 버전이 transformers<5 강제** (최신 2.2.2 도 동일) → 메인 env
(mlx-lm 가 transformers>=5 요구) 와 HARD 충돌, 작동 불가 → 격리 venv 가 유일 경로였음.
TimesFM 2.5 는 transformers 를 안 쓰므로 메인 env 네이티브. 실측 (ILI 341주, test=68):

    rolling 1-step (운영, phase 12):  TimesFM +0.939 (mae 3.31)  >  Chronos-2 +0.927 (mae 3.76)
    68-step direct (test-slab):       TimesFM −0.885            >  Chronos-2 −0.932 (둘 다 평균회귀)

→ 두 모드 모두 우위. foundation 모델의 가치 = 운영 rolling + 앙상블 다양성 (Chronos-2 와 동일 배치).

인터페이스 (ChronosForecaster 와 동일)
------------------------------------
    from simulation.models.timesfm_wrapper import TimesFMForecaster
    f = TimesFMForecaster()
    f.fit_series(y_train)          # zero-shot — 가중치 업데이트 없음, context 저장 + 모델 lazy-load
    pred = f.forecast(68)          # 68-step 직접 다단계 (TimeSeriesForecaster.predict 가 호출)

설치
----
    uv pip install "timesfm[torch]"     # transformers 안 건드림 (1 패키지만 추가)
    # checkpoint(google/timesfm-2.5-200m-pytorch ~200M) 는 첫 from_pretrained 시 자동 다운로드
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from simulation.models.base import ModelMeta, REGISTRY, TimeSeriesForecaster

log = logging.getLogger(__name__)

# 의존성 확인 플래그 (timesfm 미설치여도 모듈 import 는 성공 — registry 등록은 됨)
_HAS_TIMESFM = False
try:
    import timesfm  # noqa: F401
    _HAS_TIMESFM = True
except ImportError:
    log.debug("timesfm not installed — TimesFMForecaster 등록은 되나 fit 시 ImportError")


def _check_timesfm() -> None:
    """timesfm 패키지 확인 + 친절한 설치 안내."""
    if not _HAS_TIMESFM:
        raise ImportError(
            "timesfm 가 설치되지 않았습니다. 다음을 실행하세요:\n"
            "  uv pip install \"timesfm[torch]\"\n"
            "(transformers 를 건드리지 않습니다 — 메인 env 와 충돌 없음)"
        )


# ══════════════════════════════════════════════════════════
# TimesFMForecaster — Zero-shot foundation (Chronos-2 대체)
# ══════════════════════════════════════════════════════════

class TimesFMForecaster(TimeSeriesForecaster):
    """Google TimesFM 2.5 (200M) foundation model — zero-shot, transformers-free.

    Chronos-2 와 동일 인터페이스 (USES_FEATURES=False, y 시계열만 사용, X 무시).
    foundation 모델이므로 fit 은 가중치 업데이트가 아니라 context 저장 + 모델 lazy-load.

    Attributes:
        _model: TimesFM_2p5_200M_torch 인스턴스 (lazy, 첫 fit_series 에서 from_pretrained + compile)
        _context_series: fit 시 저장된 시계열 (예측 context)
        _last_quantiles: 마지막 forecast 의 quantile 배열 (PI 추적용)
    """

    USES_FEATURES = False  # y(ILI history)만 사용 — phase13 mc/feature probe 제외 (Chronos-2 와 동일)
    meta = ModelMeta(
        name="TimesFM-2.5",
        category="dl",
        level=16,
        min_data=52,
        description="Google TimesFM 2.5 (200M) 파운데이션 모델 (zero-shot). "
                    "transformers 의존 없음 → 메인 env 네이티브 (Chronos-2 대체, G-261).",
        requires_gpu=False,
        dependencies=["timesfm"],
    )

    DEFAULT_REPO = "google/timesfm-2.5-200m-pytorch"

    def __init__(self, repo_id: str = DEFAULT_REPO,
                 max_context: int = 1024, max_horizon: int = 256):
        """
        Args:
            repo_id: HuggingFace checkpoint (기본 timesfm-2.5-200m-pytorch).
            max_context: compile 시 최대 context 길이 (ILI 341주 < 1024 → 전 시계열 수용).
            max_horizon: compile 시 최대 예측 horizon (test-slab 68 + 여유 → 256).
        """
        super().__init__()
        self._repo_id = repo_id
        self._max_context = int(max_context)
        self._max_horizon = int(max_horizon)
        self._model = None
        self._context_series: Optional[np.ndarray] = None
        self._last_quantiles: Optional[np.ndarray] = None

    def fit_series(self, series: np.ndarray, **kwargs) -> "TimesFMForecaster":
        """시계열 저장 + 모델 lazy-load (zero-shot, 가중치 업데이트 없음).

        Args:
            series: (n_samples,) 1D ILI rate 시계열 (≥ min_data 권장).
            **kwargs: 무시됨 (foundation 모델 — HP 없음).

        Returns:
            self

        Side effects: 첫 호출 시 from_pretrained (checkpoint 캐시 다운로드) + compile.
        """
        _check_timesfm()

        if self._model is None:
            try:
                import timesfm
                log.info(f"  [TimesFM-2.5] from_pretrained: {self._repo_id}")
                m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self._repo_id)
                m.compile(timesfm.ForecastConfig(
                    max_context=self._max_context,
                    max_horizon=self._max_horizon,
                    normalize_inputs=True,
                    infer_is_positive=True,        # ILI ≥ 0
                    use_continuous_quantile_head=True,
                    fix_quantile_crossing=True,
                ))
                self._model = m
            except Exception as e:
                log.error(f"  [TimesFM-2.5] 모델 로드 실패: {e}")
                raise

        self._context_series = np.asarray(series, dtype=np.float32).ravel()
        self._fitted = True
        log.info(f"  [TimesFM-2.5] fit_series 완료: {len(self._context_series)} steps")
        return self

    def forecast(self, steps: int, **kwargs) -> np.ndarray:
        """n-step ahead 직접 다단계 예측 (point = median).

        TimeSeriesForecaster.predict(X_test) 가 forecast(steps=len(X_test)) 로 호출 →
        test-slab 전체를 단일 origin 에서 직접 예측 (Chronos-2 와 동일 convention).

        Args:
            steps: 예측 step 수 (≤ max_horizon; 초과 시 cap + 마지막값 pad).
            **kwargs: 무시됨.

        Returns:
            (steps,) float32 median point forecast.
        """
        if self._model is None or not self._fitted:
            raise RuntimeError("TimesFM-2.5: fit_series() 먼저 호출 필수")
        if self._context_series is None or len(self._context_series) == 0:
            raise ValueError("TimesFM-2.5: context series 없음")

        h = int(steps)
        h_req = min(h, self._max_horizon)        # compile horizon 초과 방지
        ctx = self._context_series[-self._max_context:]   # context 길이 cap

        try:
            point, quant = self._model.forecast(horizon=h_req, inputs=[ctx])
            pred = np.asarray(point[0], dtype=np.float32).ravel()[:h_req]
            # quantile 저장 (PI 추적; (h, n_quantiles)) — 실패해도 point 는 유지
            try:
                self._last_quantiles = np.asarray(quant[0], dtype=np.float32)
            except Exception:
                self._last_quantiles = None
        except Exception as e:
            log.error(f"  [TimesFM-2.5] forecast 실패: {e}")
            raise

        # 길이 보정 (h_req < h 인 경우 마지막값 pad)
        if len(pred) < h:
            pad = np.full(h - len(pred), pred[-1] if len(pred) else 0.0, dtype=np.float32)
            pred = np.concatenate([pred, pad])
        elif len(pred) > h:
            pred = pred[:h]
        return pred.astype(np.float32)


# ══════════════════════════════════════════════════════════
# 환경 검증 (CLI / 디버그용)
# ══════════════════════════════════════════════════════════

def check_timesfm_env() -> dict:
    """timesfm 설치 + device + checkpoint 캐시 검증.

    Returns:
        {"timesfm": bool, "msg": str, "device": str|None, "checkpoint": list[str]|None}
    """
    status = {"timesfm": _HAS_TIMESFM, "device": None, "checkpoint": None,
              "msg": "timesfm OK" if _HAS_TIMESFM else "timesfm 미설치 — `uv pip install \"timesfm[torch]\"`"}
    if _HAS_TIMESFM:
        try:
            import torch
            if torch.cuda.is_available():
                status["device"] = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                status["device"] = "mps"
            else:
                status["device"] = "cpu"
        except Exception:
            pass
        try:
            from pathlib import Path
            hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
            dirs = list(hf_cache.glob("models--google--timesfm-*")) if hf_cache.exists() else []
            status["checkpoint"] = [d.name for d in dirs]
        except Exception:
            pass
    return status


def main():
    print("=" * 60)
    print("  TimesFM 2.5 환경 검증 (Chronos-2 대체, transformers-free)")
    print("=" * 60)
    env = check_timesfm_env()
    print(f"\n[1] timesfm SDK: {'✓' if env['timesfm'] else '✗'}  ({env['msg']})")
    if env["timesfm"]:
        print(f"[2] Device: {env['device']}")
        print(f"[3] Checkpoint cache: {env['checkpoint'] or '(없음 — 첫 fit 시 다운로드)'}")
    print("=" * 60)


# ── registry 등록 (모듈 import 시 자동) ──────────────────────────────────────
REGISTRY.register(TimesFMForecaster)


if __name__ == "__main__":
    main()
