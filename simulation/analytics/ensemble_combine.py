"""Factorial ensemble combination (NNLS) — 2026-06-03.

per-model 격리 factorial 에서 ensemble(Ensemble-NNLS) 부활용 결합 로직. **순수 함수**(예측 배열만 입력)
— base 모델 적합/예측(val·test)은 caller(factorial_ensemble_runner) 책임.

NNLS = non-negative least squares: min ‖y_val − V·w‖² s.t. w ≥ 0 (V=base val 예측 행렬). 비음 가중치라
해석가능 + 음수 결합 없음. 정규화(Σw=1) 후 test 예측 결합. base 모델이 phase-13 per-model 격리로
따로 돌아 base-pred 배관이 끊긴 걸, val·test 예측을 모아 여기서 결합해 복원.
"""
from __future__ import annotations

import numpy as np

__all__ = ["nnls_ensemble"]


def nnls_ensemble(base_val_preds: dict, y_val, base_test_preds: dict):
    """NNLS 가중 앙상블 — base val 예측으로 비음 가중치 학습 → test 예측 결합.

    Args:
        base_val_preds: {model: val_pred 배열(len n_val)}. 가중치 학습용.
        y_val: 실제 val (len n_val).
        base_test_preds: {model: test_pred 배열(len n_test)}. 결합 대상.

    Returns:
        (ensemble_test_pred 배열(len n_test) | None, weights {model: w}).
        양쪽(val·test) 다 있는 model 만 사용. 사용 모델 0개면 (None, {}).
        NNLS 가중치 합 ≤ 0(전부 0) 이면 균등 가중 fallback.

    Side effects: none.
    Caller responsibility: base_val_preds/y_val 길이 일치, base_test_preds 길이 일치.
    """
    from scipy.optimize import nnls
    models = [m for m in base_val_preds
              if m in base_test_preds
              and base_val_preds[m] is not None and base_test_preds[m] is not None]
    if not models:
        return None, {}
    yv = np.asarray(y_val, dtype=np.float64).ravel()
    V = np.column_stack([np.asarray(base_val_preds[m], dtype=np.float64).ravel()[:len(yv)]
                         for m in models])
    # 길이 안전: val 예측이 yv 보다 짧으면 제외
    if V.shape[0] != yv.shape[0]:
        n = min(V.shape[0], yv.shape[0])
        V, yv = V[:n], yv[:n]
    try:
        w, _ = nnls(V, yv)
    except Exception:
        w = np.ones(len(models))
    if not np.isfinite(w).all() or w.sum() <= 0:
        w = np.ones(len(models))      # 전부 0/비유한 → 균등 가중
    w = w / w.sum()
    T = np.column_stack([np.asarray(base_test_preds[m], dtype=np.float64).ravel() for m in models])
    ens = T @ w
    return ens, {m: float(wi) for m, wi in zip(models, w)}
