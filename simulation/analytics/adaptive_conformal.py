"""adaptive_conformal.py — model-agnostic adaptive conformal PI (G-365, 2026-06-26).

문제: R10 generic split-conformal PI 가 in-sample 잔차로 보정 → out-of-sample 정점 과소추정 →
전 모델 과소피복(중위 0.67). static conformal 은 분포이동(epidemic peak)에 대응 못 함.

해결: **adaptive conformal (Angelopoulos 2024 Conformal-PID / Gibbs-Candès 2021 ACI)** — rolling
1-step eval 서 *과거 관측* 으로 nonconformity 점수를 갱신해 구간을 동적 확장/축소. 분포이동 시
자동 확장 → 커버리지 유지. FusedEpi predict_quantiles 가 내부적으로 이 PID 로 0.926 달성 → 같은
로직을 **전 모델에 model-agnostic 하게** 적용(점예측 + 과거 관측만 필요).

leak-free: step i 구간은 과거 obs[0..i-1] 만 사용(운영 rolling 설정 동일, 채점 대상 y[i]·미래 미사용).

Performance: O(n·K) (K=PI level 수). Side effects: none.
Caller responsibility: y_observed 는 rolling eval 의 실제 관측(과거→현재 순서), test 채점 대상과 동일 슬랩.
"""
from __future__ import annotations

import numpy as np


def _pid_adjust(qlo, qhi, obs, init_scores, beta, target,
                window: int = 30, ki: float = 0.2, cap: float = np.inf):
    """Conformal-PID (P+I) 단일 level 구간 조정 (FusedEpi 미러, 검증 0.926).

    P(분위추적): 최근 window nonconformity 점수의 beta-분위. I(적분): coverage 오차 누적 →
    충격 후 drift 교정. 학습 파라미터 0, 스칼라 게인 ki 만.

    Args:
        qlo/qhi: (n,) base 분위 하/상한 (static conformal). obs: (n,) rolling 관측.
        init_scores: cal nonconformity 점수 시드(|residuals|). beta: P 분위레벨(=1-α).
        target: 목표 miscoverage(=α). window/ki/cap: rolling 윈도·I 게인·상한.
    Returns: (nlo, nhi) 조정된 (n,) 구간.
    """
    qlo = np.asarray(qlo, dtype=np.float64)
    qhi = np.asarray(qhi, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64).ravel()
    n = len(qlo)
    nlo = qlo.copy(); nhi = qhi.copy()
    buf = [float(s) for s in np.asarray(init_scores, dtype=np.float64).ravel()]
    integral = 0.0
    for i in range(n):
        q_p = max(0.0, float(np.quantile(buf[-window:], beta))) if buf else 0.0
        scale = max(q_p, 1.0)
        Q = max(0.0, q_p + ki * scale * integral)                  # P + I
        nlo[i] = max(0.0, qlo[i] - Q)
        nhi[i] = min(cap, qhi[i] + Q)
        miscov = 1.0 if (obs[i] < nlo[i] or obs[i] > nhi[i]) else 0.0
        integral = float(np.clip(integral + (miscov - target), -5.0, 5.0))   # I (windup clip)
        buf.append(float(max(qlo[i] - obs[i], obs[i] - qhi[i])))             # 점수 버퍼 갱신
    return nlo, nhi


def adaptive_conformal_bounds(pred, halfwidths: dict, residuals, y_observed,
                              alphas, window: int = 30, ki: float = 0.2,
                              cap: float | None = None) -> dict:
    """전 K=11 level 에 adaptive conformal 적용 → {alpha: (lo, hi)}.

    Args:
        pred: (n,) 점예측(rolling). halfwidths: {alpha: q} static conformal 반폭(k11_qs).
        residuals: leak-free in-sample/OOF 잔차(conformity 시드). y_observed: (n,) rolling 관측.
        alphas: K=11 FLUSIGHT_ALPHAS. cap: 상한(None→2*max(pred,obs)).
    Returns: {alpha: (lo_arr, hi_arr)} adaptive 구간.
    """
    pred = np.asarray(pred, dtype=np.float64).ravel()
    res = np.abs(np.asarray(residuals, dtype=np.float64).ravel())
    yo = np.asarray(y_observed, dtype=np.float64).ravel()
    if cap is None:
        cap = 2.0 * float(max(np.nanmax(pred) if pred.size else 0.0,
                              np.nanmax(yo) if yo.size else 0.0, 1.0))
    out = {}
    for a in alphas:
        q = halfwidths.get(a, halfwidths.get(float(a)))
        if q is None or not np.isfinite(q):
            continue
        qlo = pred - float(q)
        qhi = pred + float(q)
        nlo, nhi = _pid_adjust(qlo, qhi, yo, res, beta=1.0 - a, target=a,
                               window=window, ki=ki, cap=cap)
        out[a] = (nlo, nhi)
    return out


def online_conformal_bounds(pred, y_observed, alphas, init_residuals=None,
                            window: int = 40, ki: float = 0.2, cap: float | None = None) -> dict:
    """순수 online conformal (G-365c) — in-sample 잔차 불요, rolling 과거잔차로 base+PID.

    rolling/foundation 등 leak-free in-sample 잔차 없는 모델도 PI/WIS 산출 가능. 각 step i 의
    구간폭 = 과거 |잔차| 분위(window) + PID 적분. init_residuals(있으면) 로 cold-start 완화.
    leak-free: y[i] 는 구간 설정 후 append (과거만 사용).

    Args: pred:(n,) 점예측. y_observed:(n,) rolling 관측. alphas: K=11. init_residuals: 시드(opt).
    Returns: {alpha: (lo, hi)}.
    """
    pred = np.asarray(pred, dtype=np.float64).ravel()
    y = np.asarray(y_observed, dtype=np.float64).ravel()
    n = len(y)
    if cap is None:
        cap = 2.0 * float(max(np.nanmax(pred) if pred.size else 0.0,
                              np.nanmax(y) if y.size else 0.0, 1.0))
    seed = list(np.abs(np.asarray(init_residuals, dtype=np.float64).ravel())) if init_residuals is not None else []
    out = {}
    for a in alphas:
        buf = list(seed)
        integral = 0.0
        lo = np.empty(n); hi = np.empty(n)
        for i in range(n):
            if len(buf) >= 3:
                q = max(0.0, float(np.quantile(buf[-window:], 1.0 - a)))
            else:
                q = float(np.max(np.abs(buf))) if buf else 0.0
            Q = max(0.0, q + ki * max(q, 1.0) * integral)              # base(rolling) + PID
            lo[i] = max(0.0, pred[i] - Q); hi[i] = min(cap, pred[i] + Q)
            miscov = 1.0 if (y[i] < lo[i] or y[i] > hi[i]) else 0.0
            integral = float(np.clip(integral + (miscov - a), -5.0, 5.0))
            buf.append(float(abs(y[i] - pred[i])))                     # online 잔차(과거, leak-free)
        out[a] = (lo, hi)
    return out


def _interval_score(y, lo, hi, alpha):
    """단일 level interval score (Gneiting-Raftery): (hi-lo) + (2/α)(lo-y)1[y<lo] + (2/α)(y-hi)1[y>hi]."""
    y = np.asarray(y, dtype=np.float64); lo = np.asarray(lo, dtype=np.float64); hi = np.asarray(hi, dtype=np.float64)
    return (hi - lo) + (2.0 / alpha) * (lo - y) * (y < lo) + (2.0 / alpha) * (y - hi) * (y > hi)


def wis_from_bounds(y, bounds: dict, alphas, median=None) -> np.ndarray:
    """adaptive (lo,hi) per-level 구간서 per-point WIS (Bracher 2021).

    WIS = 1/(K+0.5) [0.5·|y-m| + Σ_k (α_k/2)·IS_{α_k}]. median 없으면 0.5 분위 구간 중점·또는 pred.
    Args: y:(n,). bounds:{α:(lo,hi)}. alphas. median:(n,) 중앙값(None→bounds 최소α 중점).
    Returns: (n,) WIS.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    ks = [a for a in alphas if a in bounds]
    if not ks:
        return np.full(len(y), np.nan)
    if median is None:
        a0 = max(ks)  # 가장 넓은 분위(=가장 작은 신뢰)의 중점 ≈ median proxy
        lo0, hi0 = bounds[a0]
        median = 0.5 * (np.asarray(lo0, dtype=np.float64) + np.asarray(hi0, dtype=np.float64))
    median = np.asarray(median, dtype=np.float64).ravel()
    acc = 0.5 * np.abs(y - median)
    for a in ks:
        lo, hi = bounds[a]
        acc = acc + (a / 2.0) * _interval_score(y, lo, hi, a)
    return acc / (len(ks) + 0.5)
