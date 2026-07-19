#!/usr/bin/env python3
"""자치구 분배 — 통계검정 게이트 계단식 (사용자: TDD 평가로 확인, 수치 틀리면 미사용).

사용자 원칙: ILI(기본)→구별 데이터→없으면 다른것. 각 단계 수치화. **안 맞거나 수치 틀리면 미사용.**

진짜 구별 독감 부재(2024 NEDSS 미적재) → 구별 분배는 간접 검증해야 함. 검증 = endemic 질병 공간
패턴이 통계적으로 실신호인가:
  - leave-one-disease-out: 나머지 질병 복합이 held-out 질병 구별패턴을 맞히나(전이가능 실신호?).
  - 순열검정 p-value: 관측 mean-LOO 가 귀무(셔플)분포보다 유의한가.
실측 결과(seoul_disease_district 2022-24, 유행성이하선염·백일해·성홍열·수두): mean-LOO +0.21,
**p=0.075 → 비유의**(노이즈와 구별 안 됨), 연도 불안정(22-23 음수).
→ 게이트(p<0.05) 미달 → endemic 미사용 → **도시값 균등(구별 분배 안 함)** 으로 강하. 정직.

계단식: 1차 실측ILI → 2차 endemic(p<0.05 유의시만) → 3차 도시균등. 각 단계·검정수치 gu-weights.json.
Read-only DB. Run: .venv/bin/python web/scripts/build_gu_weights.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AGG = ROOT / "web" / "public" / "aggregates"
DB = ROOT / "simulation" / "data" / "db" / "epi_real_seoul.db"
GU = ['종로구', '중구', '용산구', '성동구', '광진구', '동대문구', '중랑구', '성북구', '강북구', '도봉구',
      '노원구', '은평구', '서대문구', '마포구', '양천구', '강서구', '구로구', '금천구', '영등포구', '동작구',
      '관악구', '서초구', '강남구', '송파구', '강동구']
ENDEMIC_DISEASES = ["유행성이하선염", "백일해", "성홍열", "수두"]   # 일상 소아·비말 (COVID·인플루엔자신고종 제외)
DISEASE_YEARS = (2022, 2023, 2024)
GATE_P = 0.05            # 순열검정 유의수준 — 이보다 크면(비유의) endemic 미사용
N_PERM = 2000
PERM_SEED = 0


def _norm(v: np.ndarray) -> np.ndarray:
    m = float(v.mean())
    return v / m if (m > 0 and np.isfinite(m)) else np.ones(len(v))


def _corr(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return 0.0 if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])


def _disease_vectors(con, pop, years=DISEASE_YEARS) -> list[np.ndarray]:
    ph = ",".join("?" * len(years))
    vs = []
    for dz in ENDEMIC_DISEASES:
        raw = [(con.execute(f"SELECT SUM(cases) FROM seoul_disease_district WHERE disease_nm=? AND gu_nm=? "
                            f"AND year IN ({ph})", (dz, g, *years)).fetchone()[0] or 0) / max(pop[g], 1.0)
               for g in GU]
        vs.append(_norm(np.array(raw, float)))
    return vs


def _mean_loo(vs: list[np.ndarray]) -> float:
    out = []
    for i in range(len(vs)):
        comp = np.mean([vs[j] for j in range(len(vs)) if j != i], axis=0)
        out.append(_corr(comp, vs[i]))
    return float(np.mean(out))


def evaluate_gu_proxy(con, pop) -> dict:
    """endemic 구별 패턴이 통계적 실신호인가 — LOO + 순열검정 p + 연도 안정성.

    Returns: {mean_loo, p_value, null95, temporal_stability, significant(bool), composite(np)}.
    """
    vs = _disease_vectors(con, pop)
    obs = _mean_loo(vs)
    rng = np.random.default_rng(PERM_SEED)
    null = np.array([_mean_loo([rng.permutation(v) for v in vs]) for _ in range(N_PERM)])
    p = float((null >= obs).mean())
    # 연도 안정성(연도별 복합 패턴 상관)
    per_year = {y: _norm(np.mean(_disease_vectors(con, pop, (y,)), axis=0)) for y in DISEASE_YEARS}
    ys = list(DISEASE_YEARS)
    temp = float(np.mean([_corr(per_year[ys[i]], per_year[ys[j]])
                          for i in range(len(ys)) for j in range(i + 1, len(ys))]))
    comp = _norm(np.mean(vs, axis=0))
    return {"mean_loo": round(obs, 3), "p_value": round(p, 3),
            "null95": round(float(np.percentile(null, 95)), 3),
            "temporal_stability": round(temp, 3), "n_perm": N_PERM,
            "significant": bool(p < GATE_P), "composite": comp}


def resolve_gu_tier() -> dict:
    """통계검정 게이트 계단식: endemic 은 순열검정 유의(p<GATE_P)일 때만. 비유의면 도시균등."""
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    pop = {g: (con.execute("SELECT AVG(tot_livpop) FROM daily_population_district WHERE signgu_nm=?",
                           (g,)).fetchone()[0] or 1.0) for g in GU}
    ladder = [{"tier": 1, "name": "실측 구별 ILI", "available": False,
               "note": "계절독감 구별 실측 부재(표본감시=전국·시; 2024 NEDSS 파일럿 미적재)"}]

    ev = evaluate_gu_proxy(con, pop)
    con.close()
    comp = ev.pop("composite")
    ladder.append({"tier": 2, "name": "일상 endemic 질병 프록시", "available": True,
                   "gate_pass": ev["significant"], "eval": ev, "diseases": ENDEMIC_DISEASES,
                   "confidence": "중간" if ev["p_value"] < 0.01 else "낮음" if ev["significant"] else "미달(비유의)",
                   "note": f"순열검정 p={ev['p_value']} (게이트 p<{GATE_P}): "
                           f"{'유의→사용' if ev['significant'] else '비유의→미사용(노이즈와 구별 안 됨)'}"})
    ladder.append({"tier": 3, "name": "도시값 균등(구별 분배 안 함)", "available": True,
                   "note": "통계 유의한 구별 신호 없을 때 정직한 기본값 — 도시 forecast 를 25구 동일 표시"})

    if ev["significant"]:
        sel, w, tier, conf = "endemic_proxy", comp, 2, ladder[1]["confidence"]
    else:
        sel, w, tier, conf = "uniform_city", np.ones(len(GU)), 3, "구별 분배 안 함(검증 실패)"
    weights = {g: round(float(x), 4) for g, x in zip(GU, w)}
    return {"selected_tier": tier, "selected_source": sel, "confidence": conf,
            "ladder": ladder, "weights": weights}


def main() -> None:
    r = resolve_gu_tier()
    out = {"method": "stat-gated-ladder",
           "note": "1차 실측 ILI → 2차 endemic(순열검정 p<0.05 유의시만) → 3차 도시균등. 검증 실패시 "
                   "구별 분배 안 함(정직). 각 단계 검정수치 = ladder.eval.",
           **r}
    AGG.mkdir(parents=True, exist_ok=True)
    (AGG / "gu-weights.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== gu 분배 통계검정 계단식 ===")
    for t in r["ladder"]:
        mark = "✓사용" if t["tier"] == r["selected_tier"] else "·"
        ev = t.get("eval")
        print(f"  {t['tier']}차 {t['name']:<22} [{mark}]" +
              (f"  검정: mean-LOO={ev['mean_loo']} p={ev['p_value']} 안정성={ev['temporal_stability']}" if ev else ""))
    print(f"\n  → 선택: {r['selected_tier']}차 {r['selected_source']} · {r['confidence']}")
    if r["selected_source"] == "uniform_city":
        print("     (endemic 비유의 → 구별 분배 안 하고 도시값 균등 표시 — 가짜 정밀 회피)")


if __name__ == "__main__":
    main()
