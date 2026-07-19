"""Ablation-ladder significance statistics (3자 검증 설계, 2026-06-02).

엄밀 component-attribution ablation (A0 baseline → A1 +tune → A2 +feature → A3 +AR) 의 통계 코어.
codex+Gemini 합의 fix 반영:
  • Harvey-Leybourne-Newbold (HLN 1997) 소표본 DM 보정 — n≈68 은 DM asymptotic regime 미만.
  • Holm-Bonferroni — 3개 순차 비교(ΔHP/ΔFeature/ΔAR) 다중검정 alpha 팽창 차단.
  • cumulative 사다리 deltas = 순서-의존 marginal effect (상호작용 존재 — 표에 명시).

심판 = held-out test slab (모든 arm 공통, 선택/튜닝에 미접촉 → 누수 0).
이 모듈은 **순수 통계**(예측 배열만 입력) — 예측 생성(A1 run/A3 AR)은 caller 책임.
"""
from __future__ import annotations

import numpy as np

__all__ = ["hln_dm_pvalue", "holm_correction", "ladder_deltas", "factorial_effects"]


def hln_dm_pvalue(loss_a, loss_b, *, h: int = 1) -> float:
    """Harvey-Leybourne-Newbold 소표본 보정 Diebold-Mariano 양측 p-value.

    H0: 두 예측의 기대손실 동일. loss_a < loss_b 면 a 가 더 정확. n 작을 때(<~100) 표준 DM 은
    과대확신 → HLN 보정 (DM* = DM·√[(n+1-2h+h(h-1)/n)/n], t_{n-1} 참조).

    Args:
        loss_a: arm A 의 per-point 손실 배열 (예: |y-pred| 또는 per-point WIS). 길이 n.
        loss_b: arm B 의 per-point 손실 배열. 길이 n. (a,b 같은 test point 정렬.)
        h: forecast horizon (h=1 이면 보정계수 √((n-1)/n)).

    Returns:
        양측 p-value [0,1]. 손실차가 0 분산이거나 n<3 이면 1.0 (구분 불가).

    Side effects: none.
    """
    from scipy import stats as _st
    d = np.asarray(loss_a, dtype=np.float64) - np.asarray(loss_b, dtype=np.float64)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 3:
        return 1.0
    dbar = float(np.mean(d))
    # h-step Newey-West 분산 (h=1 이면 단순 분산); 여기선 lag h-1 자기공분산 포함
    gamma0 = float(np.var(d, ddof=0))
    var = gamma0
    for k in range(1, h):
        if k < n:
            cov = float(np.mean((d[k:] - dbar) * (d[:-k] - dbar)))
            var += 2.0 * (1.0 - k / h) * cov
    if var <= 0:
        return 1.0
    dm = dbar / np.sqrt(var / n)
    # HLN 소표본 보정
    corr = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
    dm_hln = dm * corr
    p = 2.0 * (1.0 - _st.t.cdf(abs(dm_hln), df=n - 1))
    return float(min(max(p, 0.0), 1.0))


def holm_correction(pvalues) -> list[float]:
    """Holm-Bonferroni step-down 다중검정 보정 (순차 비교 alpha 팽창 차단).

    Args:
        pvalues: 원시 p-value 리스트 (예: [ΔHP, ΔFeature, ΔAR] 3개).
    Returns:
        같은 순서의 보정 p-value 리스트 (monotone 강제, [0,1] clip).
    """
    p = list(pvalues)
    m = len(p)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)          # step-down monotonicity
        adj[idx] = float(min(running, 1.0))
    return adj


def ladder_deltas(arms: dict, y_test, *, loss_fn=None, h: int = 1,
                  order=("A0", "A1", "A2", "A3")):
    """누적 ablation 사다리 한 단계씩의 손실차 + HLN-DM + Holm 보정.

    Args:
        arms: {arm_name: point_pred 배열(len n_test)}. 최소 order 의 인접쌍이 있어야 그 단계 산출.
        y_test: 실측 (len n_test).
        loss_fn: (y, pred)->per_point_loss 배열. None 이면 절대오차 |y-pred|.
        h: forecast horizon (HLN 보정).
        order: 사다리 순서 (인접쌍이 ΔHP/ΔFeature/ΔAR).

    Returns:
        {"steps": [{"from","to","label","mean_loss_from","mean_loss_to","delta","dm_p_raw","dm_p_holm","better"}],
         "n": n_test}. delta<0 = to 가 from 보다 정확(개선). dm_p_holm<0.05 = 유의.
    """
    y = np.asarray(y_test, dtype=np.float64)
    if loss_fn is None:
        loss_fn = lambda yy, pp: np.abs(np.asarray(yy, float) - np.asarray(pp, float))
    labels = {("A0", "A1"): "ΔHP/preproc", ("A1", "A2"): "ΔFeature", ("A2", "A3"): "ΔAR"}
    steps, raws = [], []
    for a, b in zip(order[:-1], order[1:]):
        if a not in arms or b not in arms:
            continue
        la, lb = loss_fn(y, arms[a]), loss_fn(y, arms[b])
        p_raw = hln_dm_pvalue(lb, la, h=h)   # b better than a?
        steps.append({"from": a, "to": b, "label": labels.get((a, b), f"{a}->{b}"),
                      "mean_loss_from": float(np.nanmean(la)), "mean_loss_to": float(np.nanmean(lb)),
                      "delta": float(np.nanmean(lb) - np.nanmean(la)), "dm_p_raw": p_raw})
        raws.append(p_raw)
    holm = holm_correction(raws)
    for s, hp in zip(steps, holm):
        s["dm_p_holm"] = hp
        s["better"] = "improve" if (s["delta"] < 0 and hp < 0.05) else (
            "worse" if (s["delta"] > 0 and hp < 0.05) else "ns")  # ns = 비유의(노이즈)
    return {"steps": steps, "n": int(np.isfinite(y).sum())}


def factorial_effects(cells, y_test, *, loss_fn=None, h: int = 1,
                      factors=("preproc", "hp", "feature")):
    """2^k 요인설계 주효과 + 2-way 상호작용 (per-cell 예측 → held-out 손실 contrast).

    누적 사다리(ladder_deltas)는 순서-의존 marginal 만 주지만, 요인설계는 각 요인의 **주효과**
    (다른 요인 평균) + **상호작용**(예: feature 효과가 HP 유무로 달라지나)을 분리. (사용자 요청 2026-06-02.)

    Args:
        cells: {(p,h,f): pred}. 키 = factors 순서의 0/1 튜플(완비 시 2^k cell). pred = test 점예측(len n).
        y_test: 실측 (len n). loss_fn: (y,pred)->per-point 손실(None=절대오차). h: horizon(HLN).
        factors: 요인 이름(키 튜플 순서 일치).

    Returns:
        {"main":[{"factor","effect","p_raw","p_holm","sig"}], "interactions":[{"pair","effect"}], "n"}.
        main.effect>0 = 그 요인 ON 이 손실 감소(개선); sig = p_holm<0.05. interaction.effect>0 = 시너지.

    Side effects: none. cells 불완비면 가능한 contrast 만.
    """
    import itertools
    y = np.asarray(y_test, dtype=np.float64)
    if loss_fn is None:
        loss_fn = lambda yy, pp: np.abs(np.asarray(yy, float) - np.asarray(pp, float))
    k = len(factors)
    losses = {key: loss_fn(y, pred) for key, pred in cells.items()}
    mean_loss = lambda key: float(np.nanmean(losses[key]))

    # 주효과: 요인 i 만 다른 (off,on) paired 쌍 pooled → HLN-DM
    main, raws = [], []
    for i, fac in enumerate(factors):
        on_c, off_c = [], []
        for key in cells:
            if key[i] == 1:
                off_key = key[:i] + (0,) + key[i + 1:]
                if off_key in cells:
                    on_c.append(losses[key]); off_c.append(losses[off_key])
        if not on_c:
            continue
        lon, loff = np.concatenate(on_c), np.concatenate(off_c)
        main.append({"factor": fac, "effect": float(np.nanmean(loff) - np.nanmean(lon)),
                     "p_raw": hln_dm_pvalue(lon, loff, h=h)})
        raws.append(main[-1]["p_raw"])
    for m, hp in zip(main, holm_correction(raws)):
        m["p_holm"] = hp
        m["sig"] = "yes" if (m["effect"] > 0 and hp < 0.05) else "no"

    # 2-way 상호작용 (descriptive): 요인 a 효과가 요인 b ON 일 때 vs OFF 일 때
    inter = []
    for ia, ib in itertools.combinations(range(k), 2):
        def a_effect(bval):
            ds = [mean_loss(key[:ia] + (0,) + key[ia + 1:]) - mean_loss(key)
                  for key in cells
                  if key[ia] == 1 and key[ib] == bval
                  and (key[:ia] + (0,) + key[ia + 1:]) in cells]
            return float(np.mean(ds)) if ds else float("nan")
        inter.append({"pair": f"{factors[ia]}:{factors[ib]}",
                      "effect": float(a_effect(1) - a_effect(0))})
    return {"main": main, "interactions": inter, "n": int(np.isfinite(y).sum())}
