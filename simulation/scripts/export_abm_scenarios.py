#!/usr/bin/env python3
"""Export precomputed metapop SEIR-V-D scenario trajectories for the ABM map.

Runs a handful of Stage-5 scenarios (run_metapop_scenario → (T, 25, 6) state)
and writes per-gu-per-day infected fractions + summaries to
``web/public/aggregates/abm-scenarios.json`` so the /map3d/abm dashboard can
animate the epidemic across the 25 gu and feed ARIA the live sim state without a
round trip to the Python sim on every interaction.

Output schema::

    {
      "gu_names": string[25],          // 종로구 … 강동구 (init order)
      "days": int,                     // T
      "scenarios": {
        "<name>": {
          "label": str,
          "I_frac":  number[T][25],    // infected / N per gu per day
          "city_incidence": number[T], // new cases/day, city
          "peak_day": int,
          "attack_rate_pct": number[25],
          "city_attack_pct": number,
          "deaths": number,
          "epi_validity_ok": bool
        }, ...
      }
    }

Reproducible (pure compute, seed-fixed; ~1-3s per scenario). Run:
    python -m simulation.scripts.export_abm_scenarios
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INIT = ROOT / "web" / "public" / "aggregates" / "seir-metapop-init.json"
OUT = ROOT / "web" / "public" / "aggregates" / "abm-scenarios.json"

#: (scenario, UI label, 감염병예방법 legal basis) — the 4 the dashboard exposes.
#: Ties each intervention to its statute so the ABM stays consistent with the
#: ARIA KDCA-law wiki (build_aria_wiki.py).
SCENARIOS = [
    ("baseline", "기준 (개입 없음)", "개입 없음 — 표본감시만 (제16조)"),
    ("npi_lockdown", "봉쇄·거리두기", "제49조 집합 제한·금지 + 마스크 착용 명령"),
    ("school_closure", "학교 폐쇄", "제49조 휴교·휴원 (학교보건법 연계)"),
    ("vaccination_campaign", "백신 캠페인", "제24·25조 (필수·임시)예방접종 NIP"),
]
HORIZON = 250  # 모든 시나리오 피크가 윈도우 내에 위치하도록 (120d에서는 R0=1.4 epidemic이 단조증가)
SEED = 42


def _gu_names() -> list[str]:
    if INIT.exists():
        names = json.loads(INIT.read_text(encoding="utf-8")).get("district_names")
        if names and len(names) == 25:
            return names
    return [f"gu_{i:02d}" for i in range(25)]


def build() -> dict:
    from simulation.server.epimas_adapter import run_metapop_scenario
    gu_names = _gu_names()
    scenarios: dict[str, dict] = {}
    days = HORIZON + 1
    for name, label, legal_basis in SCENARIOS:
        r = run_metapop_scenario(name, horizon_days=HORIZON, seed=SEED)
        st = np.asarray(r.state, dtype=float)  # (T, 25, 6) = S,E,I,R,V,D
        days = st.shape[0]
        N = st[0].sum(axis=1)  # per-gu population (conserved)
        N_safe = np.where(N > 0, N, 1.0)
        infected = st[:, :, 2]  # I compartment
        i_frac = (infected / N_safe).round(6)
        # city_incidence: E→I daily flow (SimResult.incidence) — 역학적으로 올바른 신규 감염 지표
        city_incidence = np.asarray(r.incidence, dtype=float).sum(axis=1)
        # peak_day: incidence 기준 (I compartment argmax는 회복 지연으로 피크가 늦게 잡힘)
        peak_day = int(city_incidence.argmax())
        # attack rate: (E+I+R+D)/N — 실제 감염을 경험한 사람 비율
        # 구버그: (1 - S_final/N) 는 백신 접종자(S→V 이동)를 감염자로 오계산함
        # 수정: R+D+I+E (감염 경로를 통과했거나 통과 중인 사람) / N
        e_final = st[-1, :, 1]  # E
        i_final = st[-1, :, 2]  # I
        r_final = st[-1, :, 3]  # R
        d_final = st[-1, :, 5]  # D
        ever_infected = e_final + i_final + r_final + d_final  # (25,)
        attack = (ever_infected / N_safe * 100).round(2)
        ev = getattr(r, "epi_validity", None)
        ok = bool(ev.get("ok", True)) if isinstance(ev, dict) else bool(ev) if ev is not None else True
        scenarios[name] = {
            "label": label,
            "legal_basis": legal_basis,
            "I_frac": i_frac.tolist(),
            "city_incidence": np.round(city_incidence, 1).tolist(),
            "peak_day": peak_day,
            "attack_rate_pct": attack.tolist(),
            "city_attack_pct": round(float(ever_infected.sum() / N.sum() * 100), 2),
            "deaths": int(round(float(d_final.sum()))),
            "epi_validity_ok": ok,
        }
    return {"gu_names": gu_names, "days": days, "scenarios": scenarios}


def main() -> int:
    gj = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(gj, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(gj['scenarios'])} scenarios, "
          f"{gj['days']} days, {len(gj['gu_names'])} gu)")
    for nm, s in gj["scenarios"].items():
        print(f"  {nm:24} peak d{s['peak_day']:>3} · 발병 {s['city_attack_pct']:>5}% · "
              f"사망 {s['deaths']:>5} · gate {'ok' if s['epi_validity_ok'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
