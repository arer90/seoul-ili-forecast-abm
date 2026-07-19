#!/usr/bin/env python3
"""1–2월 2차 파동을 외부 실시간 데이터(날씨·이동량·공동순환 병원체·실시간 도로교통)로 잡을 수
있나 — 정직 분해 + 다겨울 강건성.

사용자: "3,4,5월은 괜찮은데 1–2월이 안 됨. 실시간 외부데이터(날씨·공항·feature)로 시도해봐."

정직한 분해 (모두 학습 ≤anchor(기본 2025-12), 평가 = 그 다음 Jan–May 실측):
  ① 재귀 roll: 12월 저점에서 단조 감쇠 → 1–2월 과소 (2025-26 = MAE 15.37)
  ② 1-step nowcast (BASIC): 매주 '실제 ILI lag' 로 1주 예측 — 외부 없이 매주 갱신만.
  ③ 1-step nowcast (BASIC + 외부 그룹별): 날씨/이동량/병원체/Rt도로 각각. **누설 방지로
     외부는 1주 lag** (예측시점 관측된 지난주 값만). 45개 한꺼번엔 과적합이라 그룹 분리.

다겨울 강건성: anchor 2023-12/2024-12/2025-12 → Rt 도로교통만 3겨울 모두 robust(이득 크기는
시즌마다 다르나 절대 망치지 않음). 전체-dump 는 룰렛(2023-24 MAE 103). 상세 = 본 모듈 + test.

정직성:
  - 공항/항공 데이터는 feature matrix 에 없음 → 사용 불가(명시).
  - 외부 feature 는 1주 lag (실시간 가용). 날씨는 KMA 1주 예보가 있어 같은주도 가능하나 보수적
    으로 lag. 같은주(concurrent)는 진단 상한으로 별도.
  - topk=20 selection 이 BASIC+외부 통합 pool 에서 best 선택(공정 경쟁).

Read-only. Run: .venv/bin/python web/scripts/nowcast_external.py [ANCHOR]
Test:        .venv/bin/python web/scripts/test_nowcast_facts.py
"""
from __future__ import annotations

import datetime
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "web" / "scripts"))
logging.disable(logging.INFO)

from build_production_forecast import (  # noqa: E402
    _load_feature_matrix, _extract_basic_features, _gate_forecast,
)
from simulation.models.epi_models import NegBinGLMForecaster  # noqa: E402

WEATHER = ("temp_avg", "temp_min", "temp_max", "temp_std", "humidity", "wind_speed",
           "rainfall", "pressure", "sunshine")
MOB = ("subway_", "bus_", "sub_h_", "bus_h_", "pop_", "dong_", "hpop_")
RECURSIVE_ROLL_2025_26 = 15.37   # production_rollforward_2026.py 결과 (2025-12 anchor 전용)


def _d(w) -> datetime.date:
    if hasattr(w, "date"):
        return w.date()
    return w if isinstance(w, datetime.date) else datetime.date.fromisoformat(str(w)[:10])


def _ext_indices(fcols: list[str]) -> dict[str, list[int]]:
    """Group full-matrix column indices by external real-time data family."""
    groups = {"날씨": [], "이동량": [], "병원체": [], "Rt도로": []}
    for i, c in enumerate(fcols):
        if any(c.startswith(w) for w in WEATHER):
            groups["날씨"].append(i)
        elif any(c.startswith(m) for m in MOB):
            groups["이동량"].append(i)
        elif c.startswith("ari_"):
            groups["병원체"].append(i)
        elif c.startswith("rt_road"):
            groups["Rt도로"].append(i)
    return groups


def _fit_predict_1step(Xtr, ytr, Xte_rows, y_train_for_gate):
    """Fit NegBinGLM on (Xtr,ytr); 1-step predict each test row (already real features). Gated."""
    model = NegBinGLMForecaster(topk=20)
    model.fit(Xtr, ytr)
    preds = []
    for r in Xte_rows:
        raw = model.predict(r.reshape(1, -1))
        g = _gate_forecast(raw, y_train_for_gate, fallback=float(ytr[-1]), k=3.0)
        preds.append(float(g["pred"][0]))
    return np.array(preds)


def run_nowcast(anchor: datetime.date) -> dict:
    """Honest 1-step nowcast decomposition for the winter after `anchor`.

    Train ≤ anchor on ALL rows (no hold-out, like production); evaluate 1-step on each
    real week in [anchor+1 … data end].  Compares BASIC (lag+seasonal) vs BASIC + each
    external group (lagged 1 week → no leakage).

    Args:
        anchor: training cutoff date (forecast the weeks strictly after it).

    Returns:
        dict with keys:
          anchor, n_eval, airport_cols, ext_pool {group:count},
          dates [iso…], real […], pred_basic […], group_preds {name:[…]},
          janfeb_idx […], mae {basic_all, basic_janfeb, groups {name:{all,janfeb}}}

    Performance: ~6 NegBinGLM refits (BASIC + 5 groups), ~30–60s.
    Side effects: none (pure compute).
    """
    X_all, y_all, fcols, ws = _load_feature_matrix()
    X_all = np.asarray(X_all, float)
    y_all = np.asarray(y_all, float)
    Xb, _bcols, _bidx = _extract_basic_features(X_all, fcols)
    dates = [_d(w) for w in ws]
    n = len(dates)

    eg = _ext_indices(fcols)
    ext_idx = eg["날씨"] + eg["이동량"] + eg["병원체"] + eg["Rt도로"]
    airport_cols = sum(1 for c in fcols
                       if any(k in c.lower() for k in
                              ("airport", "flight", "arrival", "incheon", "gimpo", "항공", "공항")))

    a = max(i for i in range(n) if dates[i] <= anchor)
    tr = list(range(a + 1))
    te = [i for i in range(a + 1, n) if dates[i] <= dates[-1]]

    def fill(M, train_rows):
        M = M.copy()
        mu = np.nanmean(M[train_rows], axis=0)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        bad = np.where(~np.isfinite(M))
        M[bad] = np.take(mu, bad[1])
        return M

    def group_lag(idxs):
        raw = X_all[:, idxs]
        lag = np.vstack([raw[0:1], raw[:-1]])      # shift down 1 (use prev week)
        return fill(lag, tr)

    ytr = y_all[tr]
    real = y_all[te]
    td = [dates[i] for i in te]
    janfeb = [k for k, d in enumerate(td) if d.month in (1, 2)]

    def mae(p, idx=None):
        idx = range(len(real)) if idx is None else idx
        return float(np.mean([abs(p[k] - real[k]) for k in idx]))

    pred_basic = _fit_predict_1step(Xb[tr], ytr, Xb[te], ytr)
    group_preds = {}
    for gname, gidx in [("+날씨", eg["날씨"]), ("+이동량", eg["이동량"]),
                        ("+병원체", eg["병원체"]), ("+Rt도로", eg["Rt도로"]),
                        ("+전체외부", ext_idx)]:
        if not gidx:
            continue
        XBg = np.hstack([Xb, group_lag(gidx)])
        group_preds[gname] = _fit_predict_1step(XBg[tr], ytr, XBg[te], ytr)

    return {
        "anchor": str(dates[a]), "n_eval": len(te), "airport_cols": airport_cols,
        "ext_pool": {k: len(v) for k, v in eg.items()},
        "dates": [str(d) for d in td], "real": [round(float(x), 1) for x in real],
        "pred_basic": [round(float(x), 1) for x in pred_basic],
        "group_preds": {k: [round(float(x), 1) for x in v] for k, v in group_preds.items()},
        "janfeb_idx": janfeb,
        "mae": {
            "basic_all": round(mae(pred_basic), 2),
            "basic_janfeb": round(mae(pred_basic, janfeb), 2),
            "groups": {k: {"all": round(mae(v), 2), "janfeb": round(mae(v, janfeb), 2)}
                       for k, v in group_preds.items()},
        },
    }


def main() -> None:
    anchor = (datetime.date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1
              else datetime.date(2025, 12, 31))
    r = run_nowcast(anchor)
    eg = r["ext_pool"]
    print(f"외부 pool: 날씨 {eg['날씨']} · 이동량 {eg['이동량']} · 병원체 {eg['병원체']} "
          f"· Rt도로 {eg['Rt도로']}  (topk=20 이 BASIC 13+외부서 best 선택)")
    print(f"공항 칼럼: {r['airport_cols']}개 (0=feature matrix 에 없음)")

    real = r["real"]; td = r["dates"]; pb = r["pred_basic"]
    gp = r["group_preds"]
    jf = set(r["janfeb_idx"])
    print(f"\n=== 1–2월 2차 파동: 외부로 잡히나 (학습 ≤{r['anchor']}, 평가 {r['n_eval']}주) ===\n")
    print(f"  {'예측주':<13}{'실측':>7}{'②BASIC':>9}{'+Rt도로':>9}{'+병원체':>9}")
    print(f"  {'-' * 47}")
    rt = gp.get("+Rt도로", pb); pat = gp.get("+병원체", pb)
    for k, d in enumerate(td):
        star = " ◀1–2월" if k in jf else ""
        print(f"  {d:<13}{real[k]:>7.1f}{pb[k]:>9.1f}{rt[k]:>9.1f}{pat[k]:>9.1f}{star}")

    m = r["mae"]
    print(f"\n  ── MAE (낮을수록 좋음; ★=BASIC 보다 1–2월 개선) ──")
    print(f"  {'방법':<28}{'전체':>10}{'1–2월':>9}")
    print(f"  {'② 1-step nowcast (BASIC)':<28}{m['basic_all']:>10.2f}{m['basic_janfeb']:>9.2f}  ← 기준")
    for gname, gm in m["groups"].items():
        star = " ★" if gm["janfeb"] < m["basic_janfeb"] - 0.5 else ""
        print(f"  {'③ ' + gname + ' (1주 lag)':<28}{gm['all']:>10.2f}{gm['janfeb']:>9.2f}{star}")

    (ROOT / "web" / "public" / "aggregates" / "nowcast-external.json").write_text(
        json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  → wrote web/public/aggregates/nowcast-external.json")


if __name__ == "__main__":
    main()
