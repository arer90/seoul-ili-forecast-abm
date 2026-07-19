"""Model safety helpers — extracted from simulation/models/base.py.

Phase C3 partial (2026-05-12): base.py was 646 lines with 75+ imports
(god-object). The 3 standalone safety helpers below are pure functions
with no model coupling — clean extraction target. After this split,
base.py focuses on BaseForecaster + ModelRegistry concerns.

Design (D-4 deep module, D-5 gray-box contract):
    Each helper has small interface (1-2 args + kwargs) + rich implementation
    (full sanity rule set, NaN-safe, documented Raises).

Public API:
    - pick_device(prefer)        → torch.device (cuda > mps > cpu)
    - device_str()               → str ("cuda"/"mps"/"cpu")
    - _validate_shapes(...)      → None (raises ValueError on mismatch)
    - sanitize_predictions(...)  → np.ndarray (NaN/inf → 0.0)

Performance: O(n) for sanitize_predictions / _validate_shapes (n=68 ≈ ~50µs).
            O(1) device probe (~5ms cold cuda import).
Side effects: none — pure functions.
Caller responsibility:
    - _validate_shapes: every fit/fit_predict entry (G-166 G-160 burn-time fix)
    - sanitize_predictions: every BaseForecaster.predict result (G-159)
    - pick_device: never hardcode torch.device("cuda") (G-049/portability)

References:
    - G-049 portability + #1 OS portability (pick_device)
    - G-159 (sanitize_predictions silent NaN block)
    - G-160 (X 235 vs y 200 burn-time 1h+; _validate_shapes fail-fast)
    - G-166 (_validate_shapes 모든 fit_predict 시작 강제)
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np


log = logging.getLogger(__name__)


# ── Device 선택 헬퍼 (cuda > mps > cpu) ──

def pick_device(prefer: str | None = None):
    """PyTorch device 선택.

    우선순위: env `MPH_DEVICE` > `MPH_FORCE_CPU=1` > `prefer` > cuda > mps > cpu.
    - `MPH_DEVICE` 에 "cuda" / "mps" / "cpu" 를 명시하면 가용성 체크 후 사용.
    - subprocess 워커에 device 를 전파할 때 이 env 로 전달.
    """
    import os
    import torch

    # SSOT 예외(2026-05-28): MPH_DEVICE / MPH_FORCE_CPU 의 canonical 은
    # GLOBAL.resources.device_override / force_cpu. 단, 본 device-pick 은 base.py 가
    # import 하는 foundational hot-path 라 config_global import 를 끌어오지 않고 직접 read
    # 유지 (default 동일 "" / False — divergence 없음, 단순 동일-env 재read).
    env = (os.environ.get("MPH_DEVICE") or "").strip().lower()
    if os.environ.get("MPH_FORCE_CPU") == "1":
        env = "cpu"

    def _available(name: str) -> bool:
        if name == "cuda":
            return torch.cuda.is_available()
        if name == "mps":
            return (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
                and torch.backends.mps.is_built()
            )
        if name == "cpu":
            return True
        return False

    for candidate in (env, prefer, "cuda", "mps", "cpu"):
        if candidate and _available(candidate):
            return torch.device(candidate)
    return torch.device("cpu")


def device_str() -> str:
    """`pick_device()` 결과의 문자열 표현 (e.g. "cuda", "mps", "cpu")."""
    return pick_device().type


# ─────────────────────────────────────────────────────────────────
# Shape validation helper (G-166, 2026-05-02)
# ─────────────────────────────────────────────────────────────────
# 사용자 요청 (2026-05-02): "긴 시간을 보냈는데 실패했다고 하거나 모델이 안 되거나
#   예측에서 shape가 안 맞아서 틀리거나 문제가 있는지 모르겠고"
# → G-160 (tree_models.py:86 IndexError "inconsistent samples [235, 200]") 의
#   진짜 root cause = X_train (n=235) vs y_train (n=200) mismatch 가 호출자에서
#   발생했는데 모든 fit() 함수가 shape assertion 부재 → trial burn-time 1h+ 후에야
#   IndexError 로 발견. "긴 시간 후 실패" 의 직접 원인.
# → 모든 BaseForecaster.fit_predict 시작에 _validate_shapes() 강제 호출 →
#   학습 0초 만에 명확한 ValueError("X_train n=235 != y_train n=200") 로 차단.

def _validate_shapes(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    X_test: Optional[np.ndarray] = None,
    name: str = "model",
    min_n: int = 1,
) -> None:
    """fit/fit_predict 호출 전 X / y 차원 / 샘플수 sanity check.

    G-166 (2026-05-02): G-160 의 burn-time 1h+ 손실 차단. shape mismatch 가
    augment / feature_indices / external Optuna / WF-CV split off-by-one 등 어떤
    호출자에서 일어나도 학습 0초 만에 명확한 ValueError 로 fail-fast.

    Args:
        X_train: (n, p) feature matrix. ndarray-like.
        y_train: (n,) target. ndarray-like.
        X_test:  (m, p) optional — 같은 p 갖는지 검사.
        name:    error message 에 모델 이름 명시.
        min_n:   최소 샘플 수 (default 1, fit 이 의미있게 동작하는 minimum).

    Returns:
        None — 검증만, 통과 시 silent.

    Raises:
        ValueError: 6 case 중 하나
          - X_train empty (X_arr.size == 0)
          - y_train empty (y_arr.size == 0)
          - inconsistent samples (n_X != n_y) — G-160 직접 차단
          - too few samples (n_X < min_n)
          - feature dim mismatch (X_train p != X_test p)
          - all-NaN/inf y_train (degenerate target)

    Performance: O(n) — finite mask scan. n=242 ≈ 50µs (negligible).
    Side effects: 없음 (pure validation).
    Caller responsibility: 모든 fit_predict 시작에 호출 (BaseForecaster 자동),
                          fit() 직접 호출 path (`_refit_and_predict_test`) 도 명시.

    See: G-166 (shape validation 강제), G-160 (burn-time 1h+ 손실 사건).
    """
    X_arr = np.asarray(X_train)
    y_arr = np.asarray(y_train)

    if X_arr.size == 0:
        raise ValueError(f"[{name}] X_train empty (shape={X_arr.shape})")
    if y_arr.size == 0:
        raise ValueError(f"[{name}] y_train empty (shape={y_arr.shape})")

    n_X = X_arr.shape[0]
    n_y = y_arr.shape[0]
    if n_X != n_y:
        raise ValueError(
            f"[{name}] inconsistent samples: X_train n={n_X} != y_train n={n_y}. "
            f"X.shape={X_arr.shape}, y.shape={y_arr.shape}. "
            "Caller bug — augment/feature_indices/WF-CV split mismatch (G-160)."
        )
    if n_X < min_n:
        raise ValueError(
            f"[{name}] too few samples: n={n_X} < min_n={min_n}. "
            "Caller likely passed a degenerate split."
        )

    # X dim consistency with X_test (if given)
    if X_test is not None:
        X_te = np.asarray(X_test)
        if X_te.size > 0 and X_arr.ndim >= 2 and X_te.ndim >= 2:
            p_train = X_arr.shape[1]
            p_test = X_te.shape[1]
            if p_train != p_test:
                raise ValueError(
                    f"[{name}] feature dim mismatch: X_train p={p_train} != "
                    f"X_test p={p_test}. Caller bug — feature_indices applied "
                    "inconsistently to train vs test."
                )

    # NaN-only y check (degenerate target)
    if np.all(~np.isfinite(y_arr.astype(np.float64, copy=False))):
        raise ValueError(
            f"[{name}] y_train all-NaN/inf (n={n_y}) — degenerate target."
        )


# ─────────────────────────────────────────────────────────────────
# Prediction sanitize helper (G-159, 2026-05-02)
# ─────────────────────────────────────────────────────────────────
# 사용자 요청 (2026-05-02): "prediction을 하는 부분에서는 python에서
#   nan이나 null 그리고 -inf나 inf에서는 0.0으로 할수 있게 만들어줘.
#   물론 값이 없을 경우에만."
# → invalid sentinel (NaN/None/+inf/-inf) 만 0.0 으로, 정상 값은 그대로.
# → 정상 값에 대한 음수 clipping (ILI rate ≥ 0) 은 별도 단계.

def sanitize_predictions(
    pred,
    *,
    nonneg: bool = False,
    fill_value: float = 0.0,
) -> np.ndarray:
    """예측값에서 NaN / None / ±inf 를 fill_value (default 0.0) 로 치환.

    G-159 (2026-05-02): 모든 BaseForecaster.predict() 결과는 이 함수를 거쳐야
    함. log1p inverse 발산 (G-146) / GAM singular matrix (G-159) / DL prediction
    overflow (G-153) 같은 numerical issue 가 발생해도 downstream (PI, WIS,
    multi-criteria filter) 가 NaN 으로 오염되지 않게 차단.

    설계 원칙 (사용자 명시 2026-05-02): "값이 없을 경우에만 0.0".
    - **invalid sentinel** (NaN / None / +inf / -inf) → fill_value
    - **정상 값 (음수 포함) 그대로 보존** ← `nonneg=False` default
    - ILI rate ≥ 0 같은 도메인 제약은 **별도 단계**로 적용 (sanitize 책임 X).

    Args:
        pred: array-like 또는 None.
              None / 빈 배열 → np.array([]) 반환.
        nonneg: False (default) — invalid sentinel 만 처리, 음수 보존.
                True — 음수도 0 clipping (도메인 제약 강제, ILI rate 등).
        fill_value: invalid 값 치환할 값 (default 0.0).

    Returns:
        np.ndarray (float64). NaN/None/±inf 는 fill_value 로,
        nonneg=True 일 때 음수도 0 으로 clipping.

    예:
        >>> sanitize_predictions([1.0, np.nan, -3.0, np.inf, -np.inf])
        array([ 1.,  0., -3.,  0.,  0.])  # default — 음수 보존
        >>> sanitize_predictions([1.0, np.nan, -3.0], nonneg=True)
        array([1., 0., 0.])  # 도메인 제약 강제

    Raises:
        절대 raise X — None / 빈 array / type 변환 실패 모두 graceful (빈 array 반환).

    Performance: O(n) — np.nan_to_num + optional clip. n=68 ≈ 5µs.
    Side effects: 없음 (pure function).
    Caller responsibility:
        - **invalid sentinel만** 처리 — 음수 prediction 의 도메인 제약은 별도 layer.
        - downstream (PI / WIS / multi-criteria filter) 에서 0.0 처리 가능 보장.

    See: G-159 (silent 100.0 sentinel root cause), G-169 (predictions 키 G-169 추가),
         `BaseForecaster.fit_predict` (자동 적용 위치).
    """
    if pred is None:
        return np.array([], dtype=float)
    arr = np.asarray(pred, dtype=float).ravel()
    if arr.size == 0:
        return arr
    # nan_to_num: NaN/+inf/-inf 를 fill_value (default 0) 로 치환.
    # posinf=fill_value, neginf=fill_value 로 명시 (default 는 finite extremes).
    arr = np.nan_to_num(arr, nan=fill_value, posinf=fill_value, neginf=fill_value)
    if nonneg:
        arr = np.maximum(arr, 0.0)
    return arr


def safe_lstsq(A, b, rcond: float = 1e-6, ridge_alpha: float = 1.0,
               max_coef: float = 1e6) -> np.ndarray:
    """수치적으로 robust 한 최소제곱 — ill-conditioned design 폭발 차단.

    표준 ``np.linalg.lstsq(A, b, rcond=None)`` 는 작은 특이값을 머신정밀도 기준으로만
    자르므로, near-singular design (collinear feature, cond ≫ 1e12) 에서 계수가
    |β| ~ 1e6+ 로 폭발 → downstream 예측이 발산한다 (G-275: BayesianMCMC OLS init
    rcond=None → 같은 lag 의 log1p/qbin/qnorm 동시선택 cond 7.6e16 → test r2 −4.35).
    이 헬퍼는 ① ``rcond`` 로 특이값 truncation 후 ② 그래도 |β| > max_coef 거나
    non-finite 면 ridge(Tikhonov) 로 fallback (특이행렬에도 무조건 안정).

    Args:
        A: design matrix (n, p). finite 가정 (caller 책임).
        b: target (n,) 또는 (n, k).
        rcond: 특이값 cutoff 비율 (default 1e-6; lstsq default=machine-eps 보다 공격적).
        ridge_alpha: fallback Tikhonov λ (default 1.0).
        max_coef: |β| 가 이 값을 넘으면 ridge fallback 발동 (default 1e6).

    Returns:
        coef: (p,) 또는 (p, k) — finite, bounded.

    Performance: O(n·p²). Side effects: 없음.
    Caller responsibility: A/b 에 NaN/inf 없어야 함 (여기서 처리 X).
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    try:
        coef, *_ = np.linalg.lstsq(A, b, rcond=rcond)
        if np.all(np.isfinite(coef)) and np.abs(coef).max() <= max_coef:
            return coef
    except np.linalg.LinAlgError:
        pass
    # Ridge (Tikhonov) fallback — 특이 A 에도 무조건 안정.
    p = int(A.shape[1])
    AtA = A.T @ A
    AtA.flat[:: p + 1] += ridge_alpha          # 대각에 λ 추가 (정규화)
    try:
        return np.linalg.solve(AtA, A.T @ b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A) @ b


def apply_extrapolation_cap(pred, y_train_max, mult: float = 1.5, floor: float = 100.0):
    """G-289 (2026-06-17, 3자 감사): 외삽 상한 cap — DL/modern-ts/graph/ensemble 다수가 0-floor
    만 있고 상한이 없어 outbreak 외삽 시 폭주(DNN/TCN/Mamba 는 이미 y_train_max×1.5 cap 보유).

    Args:
        pred: 원공간 예측 (이미 inverse_transform + nonneg 적용된 값).
        y_train_max: fit 의 max(y_train) (누수 0). None/≤0 이면 미적용(통과).
        mult: cap 배수 (1.5 = DNN/TCN 동형). floor: 최소 cap(작은 y_max 보호).
    Returns:
        np.ndarray — min(pred, max(y_train_max×mult, floor)).
    """
    p = np.asarray(pred, dtype=np.float64)
    if y_train_max is not None and float(y_train_max) > 0:
        return np.minimum(p, max(float(y_train_max) * mult, floor))
    return p


__all__ = [
    "pick_device",
    "device_str",
    "_validate_shapes",
    "sanitize_predictions",
    "safe_lstsq",
    "apply_extrapolation_cap",
]
