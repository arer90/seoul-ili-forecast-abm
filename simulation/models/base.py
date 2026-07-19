"""
simulation/models/base.py
=========================
모든 예측 모델의 공통 인터페이스(ABC) + 모델 등록 시스템.

설계 원칙:
  1. 모든 모델은 BaseForecaster를 상속
  2. fit(X_train, y_train) → self
  3. predict(X_test) → np.ndarray
  4. 모델 메타데이터: name, category, level, min_data
  5. ModelRegistry: 전체 모델 자동 등록 + 조건부 필터링

ILI rate(‰) 전용 -- sentinel_influenza 테이블 기반
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── Safety helpers (god-object decomposition sprint C3 partial, 2026-05-12) ──
# pick_device / device_str / _validate_shapes / sanitize_predictions 는
# simulation/models/safety.py 로 추출됨 (god-object 분해 1차). 모든 caller 의
# `from simulation.models.base import X` 가 동작하도록 re-export 만 유지.
# See: G-049 (portability), G-159 (sanitize), G-160/G-166 (_validate_shapes).
from simulation.models.safety import (
    _validate_shapes,
    device_str,
    pick_device,
    sanitize_predictions,
)


# ── 모델 메타데이터 ──

@dataclass
class ModelMeta:
    """모델 메타정보."""
    name: str
    category: str       # 런타임 coarse 분류 (B-taxonomy): ts / linear / tree / dl / epi / physics / meta
    level: int           # 복잡도 순서 (0~20)
    min_data: int        # 최소 필요 데이터 주수
    description: str = ""
    requires_gpu: bool = False
    dependencies: list[str] = field(default_factory=list)


# ── 추상 기반 클래스 ──

class BaseForecaster(ABC):
    """모든 예측 모델의 공통 인터페이스."""

    meta: ModelMeta  # 서브클래스에서 정의

    #: 이 모델이 X(engineered feature)를 실제로 사용하는가.
    #: 기본 True. False = y(target history)만 사용, X 무시 (예: TimesFM/TiRex foundation).
    #: 용도: R9(per_model_optimize) G-242 mc(none/vif/corr/pca) probe 는 X 를 거르는 비교이므로,
    #:   X 를 무시하는 모델엔 4 방법이 동일 입력→동일 예측 = 비교 무의미(중복 아님, irrelevance).
    #:   그런 모델은 probe 에서 제외(mc='none' fallback)해 불필요한 fit 을 회피.
    USES_FEATURES: bool = True

    def __init__(self):
        self._fitted = False

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> "BaseForecaster":
        """학습. X_train: (n_samples, n_features), y_train: (n_samples,)"""
        ...

    @abstractmethod
    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        """예측. 반환: (n_samples,), 음수 클리핑 포함."""
        ...

    def safe_fit(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "BaseForecaster":
        """fit() + 자동 _validate_shapes — fit() 직접 호출자의 single-point fix (Q17 D, G-160).

        12 caller (runner.py / R4 wfcv / R10 per_model_eval / R9 per_model_optimize _evaluate_config 등) 가
        `model.fit()` 직접 호출 시 G-166 _validate_shapes 적용 안 됨 → burn-time 위험.
        single-point fix: caller 가 `model.safe_fit()` 호출 시 자동 validation +
        명시적 ValueError (silent NaN 차단).

        D-4 (deep module): caller 는 `safe_fit(X_train, y_train, X_test=X_test)`
        만 호출 — validation logic 은 캡슐화. 점진 migration (기존 fit() 직접 호출자
        가 safe_fit 으로 변경하면 자동 보호).

        Args:
            X_train: (n, p) feature matrix.
            y_train: (n,) target.
            X_test: (m, p) optional — feature dim consistency check.
            **kwargs: fit() 에 그대로 전달 (val_predictions / val_actual / r2_floor 등).

        Returns:
            self (chain용).

        Raises:
            ValueError: shape mismatch (`_validate_shapes` 6 case 중 하나).
            기타: subclass.fit() 의 raise 그대로 전파.

        Performance: validation O(n) ≈ 50µs + fit() (모델별 다름).
        Side effects: subclass.fit() 의 side effects + log.error (validation fail 시).
        Caller responsibility:
            - X_test 명시 권장 (feature dim 검증 — caller 의 split mismatch 차단).
            - 기존 fit() 직접 호출자는 점진 migration (12 위치, G-166 후속).

        See: G-166 (shape validation), G-160 (caller bug burn-time 손실),
             Q17 D (BaseForecaster single-point fix), `_validate_shapes`.
        """
        try:
            _validate_shapes(X_train, y_train, X_test=X_test,
                             name=self.meta.name, min_n=self.meta.min_data)
        except ValueError as ve:
            log.error(f"[{self.meta.name}] safe_fit shape validation FAIL: {ve}")
            raise
        return self.fit(X_train, y_train, **kwargs)

    def fit_predict(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_test: np.ndarray,
        *,
        clip_nonneg: bool = False,  # G-180 P2: ILI 도메인 제약 (negative pred 차단)
        **kwargs,
    ) -> Optional[np.ndarray]:
        """학습 + 예측 (한 번에). 실패 시 None 반환.

        G-159 (2026-05-02): NaN/None/±inf prediction 자동 sanitize → 0.0.
        사용자 명시 "값이 없을 경우에만 0.0" → `nonneg=False` default
        (음수 prediction 도 정상 값으로 보존). ILI rate ≥ 0 도메인 제약은
        sanitize 책임 X — downstream (conformal PI, multi-criteria filter)
        에서 별도 처리.

        G-166 (2026-05-02): _validate_shapes() 강제 호출 — G-160 (X 235 vs y
        200 mismatch) 이 augment / feature_indices / WF-CV split 어디서 오든
        학습 0초 만에 ValueError 로 차단. 사용자 명시 "긴 시간 후 실패" 차단.

        G-180 P2 (2026-05-05): `clip_nonneg=True` flag — ILI rate ≥ 0 도메인 제약
        강제 (SARIMA 44/68 negative pred 차단). ARIMA family / pytorch-forecasting
        wrappers 가 차분 후 base level 복원 실패 시 negative 발생 → 도메인 위반.
        downstream multi-criteria filter 가 이미 reject 하지만, 정직성 위해 raw
        prediction 자체를 ≥0 으로 강제 가능.
        """
        # G-166: shape sanity — fail-fast 0초 (burn-time 1h+ 손실 차단)
        try:
            _validate_shapes(X_train, y_train, X_test=X_test,
                             name=self.meta.name, min_n=self.meta.min_data)
        except ValueError as ve:
            # ValueError 는 silent fail 안 함 — 명시적 raise (caller 가 잡아야)
            log.error(f"[{self.meta.name}] shape validation FAIL: {ve}")
            raise
        try:
            self.fit(X_train, y_train, **kwargs)
            pred = self.predict(X_test, **kwargs)
            # G-159: invalid sentinel (NaN/inf/-inf) 만 0.0, 음수 등 정상 값 보존.
            pred_clean = sanitize_predictions(pred)  # nonneg=False default
            # G-180 P2: ILI rate ≥ 0 도메인 제약 (opt-in)
            if clip_nonneg and pred_clean is not None:
                import numpy as _np_g180
                pred_clean = _np_g180.clip(pred_clean, 0.0, None)
            return pred_clean
        except Exception as e:
            log.warning(f"[{self.meta.name}] fit_predict 실패: {e}")
            return None
        finally:
            # 2026-05-28 사용자 명시 통합 cleanup (CUDA + MPS + GPU 모두):
            # "memory와 cache, gpu memory 등 초기화 및 정리를 할 수 있는 공간".
            # simulation/utils/memory_cleanup.cleanup_all() — 모든 backend 통합:
            #   1. Python GC (2 passes, PEP 442 cycle finalizer)
            #   2. CUDA empty_cache + cuBLAS clear + IPC collect (Linux/Windows)
            #   3. MPS empty_cache + synchronize (Mac M-series)
            #   4. Linux glibc malloc_trim(0) — heap arena 회수
            # libtorch_python.dylib segfault 회피 (MPS cache fragment).
            # G-161 trial-level callback 와 별개로 model-level 즉시 cleanup.
            try:
                from simulation.utils.memory_cleanup import cleanup_all
                cleanup_all()
            except Exception:
                pass

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def save(self, path: str) -> None:
        """학습된 모델을 파일로 저장.

        PyTorch 모델은 state_dict + scaler를 .pt로,
        sklearn 모델은 joblib로 저장.
        """
        import pickle
        from pathlib import Path

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # PyTorch 모델 감지 — _model 이 state_dict() 를 가지는 진짜 torch Module 일 때만.
        # fix: sklearn estimators (RidgeCV, ElasticNet, XGBoost 등) 도
        # `_model` 속성을 가지지만 state_dict() 가 없어 AttributeError → 저장 실패.
        # torch.nn.Module 여부로 분기 + 완전한 pickle fallback 로 커버.
        _is_torch_module = False
        if hasattr(self, "_model") and self._model is not None:
            try:
                import torch.nn as _tnn
                _is_torch_module = isinstance(self._model, _tnn.Module)
            except ImportError:
                _is_torch_module = False

        if _is_torch_module:
            import torch
            state = {
                "model_state_dict": self._model.state_dict(),
                "meta_name": self.meta.name,
                "fitted": self._fitted,
            }
            if hasattr(self, "_scaler_X") and self._scaler_X is not None:
                state["scaler_X"] = self._scaler_X
            if hasattr(self, "_scaler_y") and self._scaler_y is not None:
                state["scaler_y"] = self._scaler_y
            if hasattr(self, "_best_params"):
                state["best_params"] = self._best_params
            torch.save(state, str(save_path))
            # G-252b (2026-06-15): 0-byte/손상 .pt silent 저장 차단(DNN-Conformal.pt=0바이트 사건).
            # 저장 직후 size>0 + torch.load round-trip + 키 검증 → 실패 시 손상파일 unlink +
            # loud RuntimeError(per-model try/except 가 컨테인, run 무중단). 옛 silent pickle
            # fallback 제거(torch 모델은 round-trip 통과만 신뢰). weights_only=False=scaler 포함 필수.
            _sz = save_path.stat().st_size if save_path.exists() else 0
            _ok = _sz > 0
            if _ok:
                try:
                    _probe = torch.load(str(save_path), weights_only=False)
                    _ok = isinstance(_probe, dict) and "model_state_dict" in _probe
                except Exception:
                    _ok = False
            if not _ok:
                save_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"[{self.meta.name}] .pt 저장 무결성 실패 (size={_sz}B) — 손상파일 제거")
            log.info(f"[{self.meta.name}] 모델 저장 (torch state_dict, {_sz}B round-trip): {save_path}")
            return

        # Fallback: pickle (sklearn / statsmodels / epi 모델 전용)
        try:
            with open(save_path, "wb") as f:
                pickle.dump(self, f)
            log.info(f"[{self.meta.name}] 모델 저장 (pickle): {save_path}")
        except Exception as _pe:
            log.warning(f"[{self.meta.name}] 모델 저장 실패 (pickle도 실패): {_pe}")

    @classmethod
    def load(cls, path: str) -> "BaseForecaster":
        """저장된 모델 로드."""
        import pickle
        from pathlib import Path

        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"모델 파일 없음: {load_path}")

        # PyTorch .pt 파일 시도
        if load_path.suffix in (".pt", ".pth"):
            try:
                import torch
                state = torch.load(str(load_path), weights_only=False)
                instance = cls()
                if hasattr(instance, "_scaler_X") and "scaler_X" in state:
                    instance._scaler_X = state["scaler_X"]
                if hasattr(instance, "_scaler_y") and "scaler_y" in state:
                    instance._scaler_y = state["scaler_y"]
                if "best_params" in state:
                    instance._best_params = state["best_params"]
                instance._fitted = state.get("fitted", True)
                # model architecture must be rebuilt by subclass
                instance._state_dict = state["model_state_dict"]
                return instance
            except Exception:
                pass

        # Fallback: pickle
        with open(load_path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def smart_load(path: str):
        """G-178 (2026-05-05): torch.load + pickle.load 통합 utility.

        외부 verification / audit script 가 .pt 파일 형식 모를 때 사용.
        sklearn / scipy / statsmodels 모델 (pickle 직접 저장) 도 read 가능.

        Returns:
            로드된 객체 (dict, BaseForecaster instance, sklearn estimator, 등)
        """
        import pickle
        from pathlib import Path
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"파일 없음: {p}")
        if p.stat().st_size == 0:
            raise ValueError(f"EMPTY 파일 (G-179): {p}")

        # 1. torch.load 시도 (zipfile format)
        try:
            import torch
            return torch.load(str(p), map_location="cpu", weights_only=False)
        except RuntimeError as e:
            # "Invalid magic number" → pickle fallback
            if "magic" not in str(e).lower():
                raise

        # 2. pickle.load fallback
        with open(p, "rb") as f:
            return pickle.load(f)


# ── SARIMA 계열 전용 인터페이스 ──

class TimeSeriesForecaster(BaseForecaster):
    """
    SARIMA 등 시계열 모델 전용 인터페이스.

    X_train/X_test 대신 1-D 시계열(y)을 직접 사용.
    fit(series) → predict(steps) 패턴.
    """

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> "TimeSeriesForecaster":
        """시계열 모델: y_train만 사용, X_train은 무시."""
        self._train_series = np.asarray(y_train, dtype=float)   # G-321: rolling-origin fallback 용
        return self.fit_series(y_train, **kwargs)

    def predict(self, X_test: np.ndarray, y_observed=None, **kwargs) -> np.ndarray:
        """시계열 모델 예측.

        G-321 (2026-06-19, 사용자): ``y_observed`` 주면 **rolling-origin 1-step**(각 test 주 i 를
        관측된 과거 y_observed[:i] 로 1주 예측 = feature 모델의 predict(X_test)와 동일 task = 공정
        평가). 없으면 단일원점 ``forecast(len)``(legacy). 단일원점은 sequence 모델을 68주 외삽
        →mean-revert→불공정 음수로 만들어 feature 모델(실 lag 1-step)과 비교 불가였음(rolling A/B:
        ARIMA −0.89→+0.86, SARIMA −1.01→+0.86).
        """
        if y_observed is not None and len(y_observed) == len(X_test):
            return self.rolling_1step(np.asarray(y_observed, dtype=float), **kwargs)
        return self.forecast(steps=len(X_test), **kwargs)

    def rolling_1step(self, y_observed: np.ndarray, **kwargs) -> np.ndarray:
        """Rolling-origin 1-step 예측 — 각 test 주 i 를 관측된 과거(y_observed[:i])로 1주 예측.

        기본 = refit-per-step (fit_series 확장창 + forecast(1)) — 이제 **fallback 전용**(짧은/degenerate
        시계열). G-338 (2026-06-24, §8.6 symmetric-refit): 운영 rolling 모델은 전부 **fit-once + 관측-feed
        (B)** 로 통일 — 파라미터를 train 서 1회만 추정하고 매 origin 은 관측값을 고정-파라미터 모델에 흘림
        (배포 충실, Tashman 2000). 구현별: statsmodels(ARIMA/SARIMA/SARIMAX)=``append(refit=False)`` ·
        Theta=fixed α/seasonal/trend override · epi(PoissonAutoreg/hhh4/GLARMA/EpiEstim/Wallinga)=관측 lag
        feed(G-327) · foundation/pf=context-feed · feature 모델=fit-once batch. **per-origin 파라미터 재추정
        (A) 모델 0** — base refit-per-step 은 위 override 가 없는 경우의 안전망일 뿐.

        Args:
            y_observed: (n_test,) hold-out 관측 y. i 예측엔 y_observed[:i] 만 사용 = leak-free 1-step.

        Returns:
            (n_test,) 1-step rolling 예측.

        Side effects: refit-per-step 시 self 상태가 마지막 확장창으로 재적합됨(eval 후 재사용 안 함).
        """
        y_in = getattr(self, "_train_series", None)
        if y_in is None:
            return self.forecast(steps=len(y_observed), **kwargs)   # series 미저장 → legacy fallback
        y_in = np.asarray(y_in, dtype=float)
        preds = np.full(len(y_observed), np.nan, dtype=float)
        for i in range(len(y_observed)):
            hist = np.concatenate([y_in, y_observed[:i]]) if i > 0 else y_in
            try:
                self.fit_series(hist)
                preds[i] = float(np.asarray(self.forecast(1)).ravel()[0])
            except Exception:
                preds[i] = float(y_observed[i - 1]) if i > 0 else float(y_in[-1])   # persistence fallback
        return preds

    @abstractmethod
    def fit_series(self, series: np.ndarray, **kwargs) -> "TimeSeriesForecaster":
        """1-D 시계열 학습."""
        ...

    @abstractmethod
    def forecast(self, steps: int, **kwargs) -> np.ndarray:
        """n-step ahead 예측."""
        ...


# ── G-321: eval-time rolling-origin 1-step 대상(공정 평가) ────────────────
# META classic-ts = identity transform(y 변환 없음) → eval 시 raw y_observed 로 rolling 1-step 이
# 정확(model.fit 이 받은 y 공간 = raw). foundation(dlinear/timesfm/tirex)·pf 는 y-transform 받아
# raw y_observed 가 부정확 → 제외(scope: classic-ts, 사용자 2026-06-19). 단일원점 forecast(len) 은
# sequence 모델을 68주 외삽→mean-revert→불공정 음수로 만들어 feature 모델(1-step via lag)과 비교 불가.
ROLLING_EVAL_MODELS = frozenset({
    "ARIMA", "SARIMA", "SARIMAX", "Theta", "FluSight-Baseline",
    # G-327 (2026-06-20, 사용자: "rolling이면 된다며?"): epi self-feeder 4종 — predict(y_observed) 로
    #   관측 lag 사용(self-feeding 단일원점 누적 과소예측→음수 회피, 매주 1-step). 실측(R9 269-tr split):
    #   PoissonAutoreg −0.02→+0.76 · hhh4 −1.0→+0.91 · EpiEstim −0.86→+0.69 · Wallinga −1.0→+0.69.
    #   ⚠ hhh4 는 train-size fragility: 242-tr(baseline) 에선 AR 계수≈0→seasonal-only −3.56(rolling 무효),
    #   269-tr(R9) 에선 +0.91. baseline reference 만 음수, 챔피언 eval(R9)=정상.
    "PoissonAutoreg", "hhh4-equivalent", "EpiEstim", "Wallinga-Teunis",
    # G-327b (2026-06-20): GLARMA = observation-driven(Davis 2003) — 관측 y 로 pearson 잔차 매주 갱신
    #   (frozen resid → static 발산 G-319b cap 미봉 제거). rolling 이 개선(baseline −0.67→+0.01,
    #   R9 −0.84→−0.60)하나 구조적 약체 — 최종 disposition(유지/제거)은 R9-optimized 결과 의존.
    "GLARMA",
    # G-336 (2026-06-24): FusedEpi — TiRex base=raw foundation(내부 _tx=TiRex 출력, y-transform 없음),
    #   preproc 도 identity(y_mode=none) 선택 → raw y_observed rolling 정확. predict(y_observed) 지원.
    #   BASELINE_ROLLING→여기 이동: R9 도 rolling 평가(static R²−0.836 = USES_FEATURES↔rolling 충돌 → rolling 0.652).
    "FusedEpi",
    # G-344 (2026-06-24, 감사 P0-3): foundation 3종 — best_config transform=HIER_none(identity) 실측 →
    #   raw y_observed rolling 정확(FusedEpi와 동치). 옛 BASELINE_ROLLING(R9 단일원점)은 oof=inf 로
    #   챔피언 선택서 제외되던 버그(TiRex=hold-out test 1위인데 후보조차 안 됨). ROLLING_EVAL 이동 →
    #   _evaluate_config_hierarchical helper(supports_rolling_eval 게이트)가 R9 OOF 도 rolling → 유한.
    #   base rolling_1step=O(1) context-store 라 저비용. baseline 동작 불변(_sbr→_sre, 둘 다 raw).
    #   ★ N-HiTS 는 여기서 제거 → TRANSFORM_ROLLING: 실측 transform=HIER_individual(NON-identity)이라 raw
    #   y_observed 면 test R²−13.7 폭발. G-337 의 "N-HiTS=identity" 가정이 틀렸음(개별 변환 선택).
    "TiRex", "TimesFM-2.5", "DLinear",
})
# 미완(별도 캠페인): foundation(TimesFM/TiRex/DLinear)=base refit-per-step timeout(cheap override 필요)
#   + y-transform 공간(raw y_observed 부정확) · pf(N-BEATS/TiDE/N-HiTS)=_PfBase 편집 + 동일 transform 이슈.


def supports_rolling_eval(model) -> bool:
    """G-321: 이 모델이 eval 시 rolling-origin 1-step(raw y_observed) 대상인가.

    Args:
        model: 학습된 forecaster 인스턴스 (``model.meta.name`` 으로 판정).

    Returns:
        True = META classic-ts(ARIMA/SARIMA/SARIMAX/Theta/FluSight-Baseline) → ``predict(X, y_observed=
        y_raw)`` 로 rolling 1-step. False = 그 외(feature/foundation/pf) → 기존 ``predict(X)``.

    Caller responsibility: True 일 때만 raw y_observed 전달(identity transform 전제). 전달된 y_observed
    는 i 예측에 y_observed[:i] 만 쓰여 leak-free.
    """
    name = getattr(getattr(model, "meta", None), "name", None)
    return name in ROLLING_EVAL_MODELS


# ── G-327c (2026-06-20, 사용자 "baseline rolling만"): foundation/pf baseline 전용 rolling ──
# foundation(DLinear/TimesFM/TiRex)·pf(N-BEATS/N-HiTS/TiDE) = 단일원점 외삽→음수(DLinear −0.20, N-HiTS
# −1.16 collapse). baseline·external(raw, tt=none G-305) 에서만 rolling 1-step(관측 y) → 음수 회피.
# R9 는 ① transform-space(raw y_observed 부정확) ② 챔피언 무영향(이들 약체) 이라 단일원점 유지 —
# R9(_refit_and_predict_test)는 supports_rolling_eval 만 사용하므로 이 set 은 R9 에 영향 0.
#   foundation=TS-path: DLinear=cheap rolling_1step override(lstsq 재학습 회피), TimesFM/TiRex=base
#     rolling_1step(fit_series=O(1) context-store 라 refit-per-step 도 저비용).
#   pf=feature-path: _PfBase.predict(y_observed) 가 encoder target 을 관측값으로(placeholder 0.0 →
#     깊은 test collapse 회피, leak-free: pf 는 t 예측에 encoder[<t] 만 사용).
BASELINE_ROLLING_MODELS = frozenset()  # G-344: 전 멤버 migrated.
# foundation(DLinear/TimesFM-2.5/TiRex)=identity 라 ROLLING_EVAL(baseline+R9 양쪽 rolling, _sre 게이트) 로 이동
#   → R9 OOF 도 유한(옛 baseline-only 는 R9 단일원점 oof=inf 로 챔피언 제외 버그). N-HiTS/N-BEATS/TiDE=
#   transform-space 라 TRANSFORM_ROLLING(R9 transform-rolling, baseline=static diagnostic). 이 set 은 이제
#   비었지만 supports_baseline_rolling helper 는 보존(미래 baseline-only 모델 대비, runner _sre or _sbr).

# ── G-337/G-344 (2026-06-24): transform-space sequence(pf) 모델 — rolling 하되 TRANSFORMED y_observed 필요 ──
# N-BEATS(mcmc_robust)·TiDE(laplace)·N-HiTS(individual) = pf 인코더가 transform 공간이라 raw y_observed 주면
# 폭발(실측: N-BEATS −16, TiDE −9, N-HiTS −13.7). eval 이 transform(y_observed) 를 넘기면 공정 rolling.
# G-344 (감사 P0-2): N-HiTS 는 G-337 서 "identity"로 오판해 ROLLING_EVAL(raw)에 뒀다가 −13.7 폭발 → 여기로
#   정정(실측 transform=HIER_individual). identity 케이스도 transform(y)=y 라 이 경로가 안전 포함.
# R2 baseline=static diagnostic, 챔피언 R9(_refit_and_predict_test)=transform-rolling(공정).
TRANSFORM_ROLLING_MODELS = frozenset({"N-BEATS", "N-HiTS", "TiDE"})


def supports_transform_rolling(model) -> bool:
    """G-337: transform-space rolling — eval 이 transform(y_observed) 를 넘겨야 하는 모델."""
    name = getattr(getattr(model, "meta", None), "name", None)
    return name in TRANSFORM_ROLLING_MODELS


def supports_baseline_rolling(model) -> bool:
    """G-327c: baseline/external(raw) eval 에서만 rolling-origin 1-step 대상인가.

    supports_rolling_eval(baseline+R9 양쪽, identity/META 라 raw 정확) 과 분리 — 이건 baseline 만.
    foundation/pf 는 R9 transform-space 라 raw y_observed 부정확 → R9 단일원점 유지(챔피언 무영향).

    Args:
        model: forecaster 인스턴스 (``model.meta.name`` 으로 판정).

    Returns:
        True = baseline/external runner 가 raw y_observed 전달 → rolling 1-step.
    """
    name = getattr(getattr(model, "meta", None), "name", None)
    return name in BASELINE_ROLLING_MODELS


# ── 공통 X+y scaler helper (Sprint 1.5 R4+R5, 2026-05-26) ────────────────
def setup_xy_scalers(X: np.ndarray, y: np.ndarray):
    """Standard (X, y) StandardScaler pair — eliminates 10× duplicate `fit_transform`
    blocks across `dl_models.py` / `epi_models.py` / `negbin_glm.py` /
    `cqr_models.py` / `overseas_transfer.py`.

    Args:
        X: shape (n, p) training feature matrix
        y: shape (n,) or (n, 1) training target

    Returns:
        ``(sx, sy, X_scaled, y_scaled)`` — caller stores ``sx``, ``sy`` on
        ``self`` so ``predict()`` can apply ``sx.transform(X_new)`` + ``sy.
        inverse_transform(y_pred)`` at inference time without re-fitting.

    Performance: O(n·p) — two scalers, single pass each.
    Side effects: none (pure function).
    Caller responsibility:
        - keep returned ``sx`` and ``sy`` on ``self`` for inference replay.
        - ``y_scaled`` is 1-D ravel'd (the consumers all expect 1-D).
    """
    from sklearn.preprocessing import StandardScaler
    sx = StandardScaler()
    sy = StandardScaler()
    X_s = sx.fit_transform(np.asarray(X, dtype=np.float64))
    y_s = sy.fit_transform(
        np.asarray(y, dtype=np.float64).reshape(-1, 1)
    ).ravel()
    return sx, sy, X_s, y_s


# ── 모델 등록 시스템 ──

class ModelRegistry:
    """
    21개 모델을 등록·관리하는 중앙 레지스트리.

    사용법:
        registry = ModelRegistry()
        registry.register(SARIMAModel)
        registry.register(XGBoostModel)
        ...

        # 데이터 크기 기반 필터링
        available = registry.get_available(data_size=341, has_gpu=False)

        # 범주별 조회
        ts_models = registry.get_by_category("ts")
    """

    def __init__(self):
        self._models: dict[str, type[BaseForecaster]] = {}

    def register(self, model_cls: type[BaseForecaster]) -> None:
        """모델 클래스 등록."""
        name = model_cls.meta.name
        if name in self._models:
            log.warning(f"모델 '{name}' 중복 등록 -- 덮어쓰기")
        self._models[name] = model_cls

    def smoke_test_all(self, *, n_train: int = 240, n_val: int = 30,
                       n_test: int = 50, n_features: int = 20,
                       seed: int = 42, foundation_use_val_context: bool = True
                       ) -> dict:
        """G-181 (2026-05-05) — 영구 회귀 차단:
        모든 등록 모델이 synthetic data 로 fit/predict 작동하는지 검증.

        Foundation(TimesFM/TiRex)/-pf 류는 train+val context 사용 (R9 per_model_optimize align).
        catastrophic R²<0 발견 시 즉시 fail-fast — register 거부 또는 deprecate.

        Returns:
            {model_name: {status, r2, mape, time_sec, issues}}
        """
        import time
        from simulation.scripts.mini_tests.synthetic import make_synthetic
        from simulation.scripts.mini_tests.diagnose import diagnose_model

        data = make_synthetic(n_train=n_train, n_val=n_val,
                              n_test=n_test, n_features=n_features, seed=seed)
        results = {}
        for name, cls in self._models.items():
            t0 = time.time()
            try:
                r = diagnose_model(cls, data, model_name=name, timeout_sec=120)
                results[name] = {
                    'status': r['status'],
                    'r2': (r.get('metrics') or {}).get('r2'),
                    'mape': (r.get('metrics') or {}).get('mape'),
                    'time_sec': r.get('time_sec'),
                    'issues': r.get('issues', [])[:2],
                }
            except Exception as e:
                results[name] = {'status': 'EXCEPTION', 'issues': [str(e)[:120]],
                                  'time_sec': time.time() - t0}
        return results

    def get(self, name: str) -> Optional[type[BaseForecaster]]:
        """이름으로 모델 클래스 조회."""
        return self._models.get(name)

    def instantiate(self, name: str) -> Optional[BaseForecaster]:
        """이름으로 모델 인스턴스 생성."""
        cls = self._models.get(name)
        return cls() if cls else None

    def get_all(self) -> dict[str, type[BaseForecaster]]:
        """등록된 전체 모델."""
        return dict(self._models)

    def get_available(
        self,
        data_size: int = 0,
        has_gpu: bool = False,
        exclude_categories: list[str] = None,
    ) -> list[BaseForecaster]:
        """
        조건에 맞는 모델 인스턴스 목록 반환.

        Parameters:
            data_size: 가용 데이터 주수
            has_gpu: GPU 사용 가능 여부
            exclude_categories: 제외할 범주
        """
        exclude = set(exclude_categories or [])
        result = []

        for name, cls in sorted(self._models.items(), key=lambda x: x[1].meta.level):
            meta = cls.meta
            if meta.category in exclude:
                continue
            if meta.min_data > data_size:
                continue
            if meta.requires_gpu and not has_gpu:
                continue
            try:
                instance = cls()
                result.append(instance)
            except Exception as e:
                log.warning(f"[{name}] 인스턴스 생성 실패: {e}")

        return result

    def get_by_category(self, category: str) -> list[type[BaseForecaster]]:
        """범주별 모델 클래스 조회."""
        return [
            cls for cls in self._models.values()
            if cls.meta.category == category
        ]

    def summary(self) -> pd.DataFrame:
        """등록된 모델 요약 테이블."""
        rows = []
        for name, cls in sorted(self._models.items(), key=lambda x: x[1].meta.level):
            m = cls.meta
            rows.append({
                "name": m.name,
                "category": m.category,
                "level": m.level,
                "min_data": m.min_data,
                "gpu": m.requires_gpu,
                "description": m.description,
            })
        return pd.DataFrame(rows)


# ── 전역 레지스트리 ──
REGISTRY = ModelRegistry()
