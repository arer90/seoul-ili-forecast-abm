"""TabPFN v2 wrapper — tabular foundation model (in-context learning, 소표본 SOTA).

목적 (G-264, 2026-06-13, 사용자 확정)
-----------------------------------
TabPFN v2 (Hollmann et al., Nature 2025) = 소표본 tabular 회귀에 특화된 사전학습 foundation 모델.
in-context learning 으로 from-scratch 학습 없이 예측 → 341주 소표본(≈349 OOF)이 정확히 sweet-spot.
ILI 실측: hold-out r2=**0.917**(최우수), WF-CV r2=0.814 — incumbent(XGBoost 0.79·glum 0.878) 능가.

라이선스 (provenance 정직성)
---------------------------
가중치는 **Prior Labs License v1.1** (`priorlabs-1-1`) — 학술/연구 무료. 가중치 .ckpt 는 HuggingFace
`Prior-Labs/TabPFN-v2-reg` 공개(non-gated) repo 에 있어 표준 `huggingface_hub` 로 다운로드 가능.
본 wrapper 는 그 공개 가중치를 받아 TabPFN 의 **공식 `model_path` 인자**(저자 제공 offline 로딩
기능)로 로드 → PriorLabs 토큰-플로우 없이 동작. 사용자 결정(2026-06-13): 공개 가중치 + model_path.
논문에는 priorlabs-1-1 학술 사용을 명시.

성능/시간상 i.i.d. 주의
----------------------
TabPFN 은 행을 교환가능(i.i.d.)으로 가정 → 시계열 자기상관을 직접 모델링하지 않으나, lag feature 가
시간정보를 인코딩하므로 tabular 회귀로 작동(실측 확인). 논문 limitation 으로 명시 권장 (gemini 지적).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

from simulation.models.base import BaseForecaster, ModelMeta, REGISTRY

log = logging.getLogger(__name__)

# ── 프라이버시 (사용자 2026-06-13 "외부 유출 안 했으면, 기능만"): TabPFN 텔레메트리 완전 차단 ──
# tabpfn 은 기본적으로 usage 이벤트를 PostHog(eu.i.posthog.com)로 전송 + config 를 GCS 에서 download.
# 이 env 를 import 시점에 set → service.telemetry_enabled() 가 download_config() **전에** False 반환
# (tabpfn_common_utils/telemetry/core/service.py:80). 데이터(X/y) 아닌 메타데이터지만 외부 전송 0 보장.
# import tabpfn(=initialize_telemetry) 보다 먼저 set 되도록 모듈 top 에 위치. (setdefault: 명시 override 존중)
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")

_HF_REPO = "Prior-Labs/TabPFN-v2-reg"
_CKPT_NAME = "tabpfn-v2-regressor.ckpt"


def _w_tabpfn_available() -> bool:
    """tabpfn 패키지 설치 여부 (가드/테스트용)."""
    try:
        import tabpfn  # noqa: F401
        return True
    except ImportError:
        return False


def _load_tabpfn_token() -> bool:
    """TabPFN API key 를 env 또는 `simulation/data/api_key.txt`(여러 키 모음 — "TabPFN" 라벨 줄)서
    로드해 `TABPFN_TOKEN` in-process 설정 → 정식 라이선스 수락(provenance). 키는 노출/커밋 안 함
    (api_key.txt 는 .gitignore). cached 가중치면 런타임 네트워크 0(토큰 set 만 = 수락 선언).

    Returns: True (토큰 확보) / False (없음 → 공개 가중치 model_path 폴백).
    """
    if os.environ.get("TABPFN_TOKEN"):
        return True
    import re
    here = Path(__file__).resolve()
    for p in (Path("simulation/data/api_key.txt"),
              here.parents[2] / "simulation" / "data" / "api_key.txt"):
        if p.is_file():
            try:
                for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if re.search(r"tabpfn|prior.?labs", line, re.I):
                        runs = re.findall(r"[A-Za-z0-9_\-=.]{20,}", line)
                        if runs:
                            tok = max(runs, key=len)
                            tok.encode("latin-1")           # HTTP 헤더 호환 확인 (아니면 skip)
                            os.environ["TABPFN_TOKEN"] = tok
                            log.info("  [TabPFN] API key 로드(api_key.txt TabPFN 라벨) — 정식 라이선스")
                            return True
            except Exception:
                continue
    return False


def _ensure_weights() -> Optional[Path]:
    """공개 HF repo 가중치를 캐시에 확보(없으면 토큰 없이 다운로드) → local .ckpt 경로.

    Returns:
        캐시된 .ckpt Path (확보 성공) 또는 None (tabpfn 미설치/다운로드 실패 → 호출자가 model_path 생략).

    Side effects: 최초 1회 ~42MB 다운로드 (공개 repo, 토큰 불필요).
    """
    if not _w_tabpfn_available():
        return None
    try:
        from tabpfn.model_loading import get_cache_dir
        cache = get_cache_dir(); cache.mkdir(parents=True, exist_ok=True)
        dst = cache / _CKPT_NAME
        if not dst.exists():
            from huggingface_hub import hf_hub_download
            import shutil
            src = hf_hub_download(_HF_REPO, _CKPT_NAME)   # 공개 repo → 토큰 불필요
            shutil.copy(src, dst)
            log.info(f"  [TabPFN] 공개 가중치 확보: {dst} ({dst.stat().st_size // (1024*1024)}MB)")
        return dst
    except Exception as e:
        log.warning(f"  [TabPFN] 가중치 확보 실패: {e} → model_path 생략(정식 토큰 플로우로 폴백)")
        return None


class TabPFNForecaster(BaseForecaster):
    """TabPFN v2 tabular foundation model (in-context, zero-train) — 소표본 SOTA.

    USES_FEATURES=True (tabular X 사용). fit 은 가중치 학습이 아니라 context 적합(in-context).
    공개 가중치를 공식 model_path 로 로드(라이선스 = priorlabs-1-1 학술 무료, 논문 명시).

    Caller responsibility: X 는 ≤500 feature 권장(초과 시 ignore_pretraining_limits 로 허용),
        n ≤ 10k. Performance: fit O(1)(context 저장) + predict 시 in-context 추론(~수초/CPU).
    """

    USES_FEATURES = True
    meta = ModelMeta(
        name="TabPFN",
        category="dl",
        level=16,
        min_data=50,
        description="TabPFN v2 tabular foundation model (in-context learning, Nature 2025). "
                    "소표본 SOTA — ILI hold-out r2=0.917. 공개 가중치 + model_path (priorlabs-1-1, G-264).",
        requires_gpu=False,
        dependencies=["tabpfn"],
    )

    def __init__(self, n_estimators: int = 4, device: str = "cpu"):
        super().__init__()
        self._model = None
        self._n_est = int(n_estimators)
        self._device = device

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> "TabPFNForecaster":
        if not _w_tabpfn_available():
            raise ImportError("tabpfn 미설치 — `uv pip install tabpfn`")
        import warnings as _w
        _load_tabpfn_token()           # API key(api_key.txt) → TABPFN_TOKEN = 정식 라이선스 (import 전)
        from tabpfn import TabPFNRegressor
        ckpt = _ensure_weights()       # cached 공개 가중치 → model_path(런타임 네트워크 0). 토큰은 provenance.
        kw = {"device": self._device, "ignore_pretraining_limits": True,
              "n_estimators": self._n_est, "random_state": 42}
        if ckpt is not None:
            kw["model_path"] = str(ckpt)   # 공식 offline 로딩 (공개 가중치)
        # TabPFN 내부 preproc 가 benign matmul over/underflow 경고 발생 — 결과 유한·정상.
        # multi-day run 로그 오염 방지로 fit 중에만 억제 (caller 영향 0).
        with np.errstate(all="ignore"), _w.catch_warnings():
            _w.simplefilter("ignore")
            self._model = TabPFNRegressor(**kw).fit(
                np.asarray(X_train, float), np.asarray(y_train, float))
        self._fitted = True
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        if not self._fitted or self._model is None:
            raise RuntimeError("TabPFN: fit() 먼저 호출")
        import warnings as _w
        with np.errstate(all="ignore"), _w.catch_warnings():
            _w.simplefilter("ignore")
            return np.asarray(self._model.predict(np.asarray(X_test, float)), float)


# registry 등록 (모듈 import 시)
try:
    REGISTRY.register(TabPFNForecaster)
    log.info("[tabpfn_wrapper] TabPFNForecaster 등록됨")
except Exception as _e:
    log.debug(f"[tabpfn_wrapper] 등록 skip: {_e}")
