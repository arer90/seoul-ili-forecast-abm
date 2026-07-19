"""모델별 OMP/BLAS 스레드 동적 제어 (sklearn 표준).

배경
----
Mac-migration: macOS Apple Silicon 의 libomp pthread_mutex 충돌 회피를 위해
__main__.py 에서 OMP_NUM_THREADS=1 강제 (default). 그러나 KRR/SVR 같은
BLAS-heavy kernel solve 모델은 single-thread BLAS 에서 numerical precision
저하 → R² 0.7+ → 음수로 회귀.

해결
----
threadpoolctl (3.6+) 의 `threadpool_limits` 로 BLAS 스레드만 임시 증가.
XGBoost/LightGBM 의 libomp 는 영향 받지 않으므로 SIGSEGV 위험 없음.

사용 예시
--------
```python
from simulation.models._omp_context import blas_threads

class KRRForecaster(BaseForecaster):
 def fit(self, X_train, y_train):
 with blas_threads(2): # KRR fit 동안만 BLAS 2 thread
 self._model.fit(X_s, y_s) # threadpool_limits 안에서 안전
 return self
```
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

log = logging.getLogger(__name__)

# 기본값: macOS 에서는 BLAS-heavy 모델에 2-thread 권장
DEFAULT_BLAS_THREADS_KERNEL = 2


@contextmanager
def blas_threads(n: int = DEFAULT_BLAS_THREADS_KERNEL) -> Iterator[None]:
    """BLAS thread 수를 임시로 n 으로 설정. context 종료 시 자동 복원.

    Parameters
    ----------
    n : int
        BLAS pool 의 max threads. 1=완전 단일, 2=권장 default, 4+=대형 행렬.

    Notes
    -----
    - threadpoolctl 만 사용 (BLAS/OpenMP/MKL 모두 안전 제어).
    - libomp (XGBoost/LightGBM) 는 자체 OpenMP runtime 이라 영향 안 받음 →
      Mac SIGSEGV 위험 없이 KRR/SVR 등 sklearn 만 가속 가능.
    - threadpoolctl 미설치 시 no-op (warn).
    """
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        log.warning("[blas_threads] threadpoolctl 미설치 — no-op. "
                    "uv pip install threadpoolctl 권장.")
        yield
        return

    with threadpool_limits(limits=int(n), user_api="blas"):
        yield


@contextmanager
def all_threads(n: int) -> Iterator[None]:
    """BLAS + OpenMP 둘 다 n 으로 설정 (KRR + libgomp 모두 영향).

    경고: macOS 에서 n>1 + libomp 모델 (XGBoost) 같이 호출 시 SIGSEGV 위험.
    KRR/SVR 등 sklearn-only fit 에서만 사용.
    """
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        log.warning("[all_threads] threadpoolctl 미설치 — no-op.")
        yield
        return

    with threadpool_limits(limits=int(n)):  # user_api=None → 모든 pool
        yield


__all__ = ["blas_threads", "all_threads", "DEFAULT_BLAS_THREADS_KERNEL"]
