#!/usr/bin/env python3
"""+1주 정확도 보정 — 잔차 bias 차감 + 예측수준 비례(relative) PI. (codex/gemini 합의 레버)

근거(horizon-reliability): +1주 bias +6.41(pred 과대) = 데이터로 보이는 계통오차, PI95 coverage
0.84(목표 0.95, 구간 너무 좁음). 둘 다 고칠 수 있는 systematic error.

레버 두 개(순수 함수, TDD):
  - recent_onestep_bias: 최근 k주 1-step rolling 잔차(pred−actual) 평균 → 차감(debias).
  - relative PI: 가산 q_hat 단일값 대신 예측수준 비례(pred·rel_q) 폭 → 피크서 넓어져 under-coverage
    교정. (relative-conformal, 과거 backtest 에서 coverage 38→66 입증된 그 방식)

compare_calibration(cutoff) = raw vs debiased vs +relPI 의 MAE·|bias|·coverage 비교(실증).
Read-only. Run: .venv/bin/python web/scripts/accuracy_calibration.py [CUTOFF]
"""
from __future__ import annotations

import datetime
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "web" / "scripts"))
# NOTE: 모듈 import 시 전역 logging 을 끄지 않는다(build_production_forecast 가 이 모듈을 import 해
# bias 보정에 쓰므로 — 끄면 그쪽 INFO 로그가 죽음). standalone 실행 시에만 main() 에서 끈다.
from build_production_forecast import _load_feature_matrix, _extract_basic_features, _gate_forecast, _surge_aware_bound  # noqa: E402
from simulation.models.epi_models import NegBinGLMForecaster  # noqa: E402


def _d(w):
    if hasattr(w, "date"):
        return w.date()
    return w if isinstance(w, datetime.date) else datetime.date.fromisoformat(str(w)[:10])


def _onestep(Xb, y, t):
    """t 주를 ≤t-1 학습 모델로 1-step 예측(gate+surge). (Xb[t]=실측 lag)"""
    m = NegBinGLMForecaster(topk=20); m.fit(Xb[:t], y[:t])
    raw = np.asarray(m.predict(Xb[t:t + 1]), float)
    p = float(_gate_forecast(raw, y[:t], fallback=float(y[t - 1]), k=3.0)["pred"][0])
    p, _s, _r = _surge_aware_bound(p, y[max(0, t - 4):t], float(np.nanmax(y[:t])))
    return p


def recent_onestep_bias(Xb, y, end_idx: int, k: int = 6) -> float:
    """최근 k주 1-step rolling 잔차(pred−actual) 평균 = 차감할 bias 추정.

    Args:
        Xb, y: BASIC feature·target.
        end_idx: 기준 인덱스(이 주까지 관측). 보정 대상은 end_idx+1.
        k: 최근 몇 주 잔차 평균.

    Returns:
        mean(pred − actual) over [end_idx-k+1 … end_idx]. 양수=모델이 최근 과대예측.

    Performance: k 회 NegBinGLM 적합.
    """
    res = []
    for t in range(max(1, end_idx - k + 1), end_idx + 1):
        if t < 30:
            continue
        res.append(_onestep(Xb, y, t) - y[t])
    return float(np.mean(res)) if res else 0.0


def compare_calibration(cutoff: datetime.date, k: int = 6, max_eval: int | None = None) -> dict:
    """raw vs debiased vs +relative-PI 의 MAE·|bias|·PI95 coverage 실증 비교.

    Returns:
        {n_eval, raw{mae,bias,cov}, debiased{mae,bias,cov}, debiased_relpi{cov}}.
    """
    X, y, fc, ws = _load_feature_matrix()
    X = np.asarray(X, float); y = np.asarray(y, float)
    Xb, _bc, _bi = _extract_basic_features(X, fc)
    dates = [_d(w) for w in ws]; n = len(y)
    c = max(i for i in range(n) if dates[i] <= cutoff)
    test = [i for i in range(c + 1, n)]
    if max_eval:
        test = test[:max_eval]

    raw_e, deb_e = [], []
    cov_add, cov_rel = [], []
    # relative 잔차(보정 전, val=학습구간 끝부분)로 rel_q 추정
    val_res = [( _onestep(Xb, y, t) - y[t]) / max(_onestep(Xb, y, t), 0.5) for t in range(c - 12, c) if t >= 30]
    rel_lo = float(np.quantile(val_res, 0.025)) if val_res else -0.5
    rel_hi = float(np.quantile(val_res, 0.975)) if val_res else 0.5
    add_res = [abs(_onestep(Xb, y, t) - y[t]) for t in range(c - 12, c) if t >= 30]
    q_add = float(np.quantile(add_res, 0.95)) if add_res else 10.0

    for t in test:
        p = _onestep(Xb, y, t)
        b = recent_onestep_bias(Xb, y, t - 1, k)         # t-1 까지로 bias 추정(누수 없음)
        pd = max(0.0, p - b)
        raw_e.append(p - y[t]); deb_e.append(pd - y[t])
        cov_add.append(1 if (pd - q_add) <= y[t] <= (pd + q_add) else 0)
        cov_rel.append(1 if max(0.0, pd * (1 + rel_lo)) <= y[t] <= pd * (1 + rel_hi) else 0)

    def m(e):
        e = np.array(e); return {"mae": round(float(np.mean(np.abs(e))), 2), "bias": round(float(np.mean(e)), 2)}
    return {"cutoff": str(dates[c]), "n_eval": len(test), "k": k,
            "raw": {**m(raw_e), "cov": round(float(np.mean(cov_add)), 2)},
            "debiased": {**m(deb_e), "cov_add": round(float(np.mean(cov_add)), 2)},
            "debiased_relpi_cov": round(float(np.mean(cov_rel)), 2),
            "rel_q": [round(rel_lo, 3), round(rel_hi, 3)], "q_add": round(q_add, 2)}


def main() -> None:
    logging.disable(logging.INFO)     # standalone 출력만 깔끔하게 (import 경로엔 영향 없음)
    cutoff = datetime.date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else datetime.date(2024, 9, 30)
    r = compare_calibration(cutoff)
    print(f"=== +1주 보정 실증 (cutoff {r['cutoff']}, 평가 {r['n_eval']}주, k={r['k']}) ===\n")
    print(f"  raw      : MAE {r['raw']['mae']}  bias {r['raw']['bias']:+}  PI95cov {r['raw']['cov']}")
    print(f"  debiased : MAE {r['debiased']['mae']}  bias {r['debiased']['bias']:+}  (잔차 bias 차감)")
    print(f"  +relative PI: coverage {r['debiased_relpi_cov']}  (가산 {r['debiased']['cov_add']} → 비례)")
    print(f"\n  결론: |bias| {abs(r['raw']['bias'])}→{abs(r['debiased']['bias'])}, "
          f"coverage {r['raw']['cov']}→{r['debiased_relpi_cov']} (목표 0.95)")


if __name__ == "__main__":
    main()
