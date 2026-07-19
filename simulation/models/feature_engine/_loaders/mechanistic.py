"""mechanistic.py — ILI-only 기계적 feature (Rt·S/N·FoI), 인과적(per-week past-only).

NEW_MODEL_IDEAS의 SEIR 채널 — 단 사용자 원칙 "SEIR 없을 수 있으니 ILI만 가정":
full 25-gu metapop SEIR 실행 대신, **관측 ILI incidence만으로** 기계적 신호를 인과 도출.
→ 무거운 SEIR run 불필요(audit_retrain 무간섭·calibration 불필요) + **#16-C4 누수가드가
   인과 계산으로 자동 충족** (feature[t]는 incidence[:t+1]만 의존 → fold/train-end 무관하게 동일).

feature (각 주 t = past-only):
- ``rt``     : 재생산수 proxy (renewal/Cori-style 비율 I[t]/과거평균).
- ``s_frac`` : 감수성 분율 proxy = 1 - clip(trailing-window 누적발생 / N_eff). 계절 소진 반영.
- ``foi``    : force-of-infection proxy ≈ rt·(1-s_frac).

Caller responsibility: incidence ≥ 0. N_eff 고정(=train/full 무관, 누수 0). full RtEstimator
   (Cori 2013, rt_estimator.py)로 ``rt`` 교체 가능(더 정식). SEIR 채널은 *optional* — 모델은 ILI-only로도 동작.
Performance: O(T·window) 순수 numpy. Side effects: 없음.
"""
from __future__ import annotations

import numpy as np

_DEFAULT_N_EFF = 1.0e3   # 유효 모집단 스케일 (ILI rate 단위 — caller 조정 가능, 고정=누수0)
_DEFAULT_WINDOW = 52     # trailing 소진 창 (계절 reset 효과)


def causal_rt(incidence: np.ndarray, tau: int = 4, prior: float = 1.0) -> np.ndarray:
    """renewal-style 인과 Rt proxy: I[t] / (직전 tau주 평균). 각 t는 past만(t 미포함)."""
    inc = np.asarray(incidence, dtype=float)
    n = len(inc)
    rt = np.ones(n, dtype=float)
    for t in range(n):
        past = inc[max(0, t - tau):t]            # [lo, t) = 과거만 (current 미포함)
        denom = float(past.mean()) if past.size else prior
        rt[t] = inc[t] / denom if denom > 1e-9 else 1.0
    return np.clip(rt, 0.0, 10.0)


def susceptible_frac(incidence: np.ndarray, n_eff: float = _DEFAULT_N_EFF,
                     window: int = _DEFAULT_WINDOW) -> np.ndarray:
    """감수성 분율 proxy = 1 - clip(trailing-window 누적발생 / N_eff). trailing=인과·계절소진."""
    inc = np.asarray(incidence, dtype=float)
    n = len(inc)
    s = np.ones(n, dtype=float)
    for t in range(n):
        recent = float(inc[max(0, t - window + 1):t + 1].sum())   # trailing (current 포함=인과)
        s[t] = np.clip(1.0 - recent / float(n_eff), 0.0, 1.0)
    return s


def mechanistic_features(incidence: np.ndarray, n_eff: float = _DEFAULT_N_EFF,
                         tau: int = 4, window: int = _DEFAULT_WINDOW) -> np.ndarray:
    """ILI incidence → (T, 3) 기계적 feature [rt, s_frac, foi]. 각 주 인과(past-only).

    #16-C4 누수가드: feature[t]는 incidence[:t+1]만 의존 → 미래 데이터 유무가 과거 feature를
    바꾸지 않음(인과). 따라서 per-fold 재계산이든 full 계산이든 동일 = fold-future 누수 0.
    Returns: (T, 3) float, columns = [rt, s_frac, foi].
    """
    inc = np.asarray(incidence, dtype=float)
    rt = causal_rt(inc, tau=tau)
    s = susceptible_frac(inc, n_eff=n_eff, window=window)
    foi = rt * (1.0 - s)                          # force ≈ 전파력 × 소진(1-S/N)
    return np.column_stack([rt, s, foi])


MECHANISTIC_FEATURE_NAMES = ["mech_rt", "mech_s_frac", "mech_foi"]
