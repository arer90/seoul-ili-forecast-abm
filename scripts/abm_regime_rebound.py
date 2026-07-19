"""Regime-dependent 행동 효과 — post-COVID rebound vs 정상 시즌 (Pillar 2, §4.16 정밀화).

사용자 결정(thesis-realignment): "행동 결합 necessary"를 **regime-dependent 결정적**으로 정밀화.
가설: 행동-on이 행동-off 대비 정점을 낮추는 효과가 **rebound(면역부채=고감수성) regime 에서 크고,
정상 시즌(고면역) regime 에선 작다** → 행동 결합은 *globally necessary 가 아니라 regime-dependently
decisive*. 이게 코드↔§4.5 긴장(behaviour-off 가 평시 best-fit 허용)을 정직하게 해소.

설계: 실 25-gu metapop(load_metapop_params) 기반. regime = initial_recovered(면역) 분율로 인코딩
  · rebound = 5% 면역(immunity debt)  · normal = 35% 면역(pre-COVID 통상).
각 regime × behaviour{off(α0)/on(α1.5 fatigue)} 의 city 정점. CI = R0 parametric ensemble
  {1.3..2.1} (⚠ run_coupled_abm 은 결정적이라 stochastic-seed 아님 — 정직: 파라미터 불확실성 band).
integrator = exp-Euler(기본). 산출 → results/abm_v1/regime_rebound.json.

용법: .venv/bin/python scripts/abm_regime_rebound.py [--smoke]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os

import numpy as np

from simulation.abm.behavioural import BehaviouralParams, run_coupled_abm
from simulation.sim.io import load_metapop_params
from simulation.sim.parameters import DiseaseParams

BEHAV_OFF = BehaviouralParams(alpha=0.0, kappa=0.0, tau=float("inf"))
BEHAV_ON = BehaviouralParams(alpha=1.5, kappa=0.8, tau=60.0, theta=0.15)  # post-COVID fatigue
IMMUNITY = {"rebound": 0.05, "normal": 0.35}          # initial_recovered 분율
R0_ENSEMBLE = (1.3, 1.5, 1.7, 1.9, 2.1)


def _regime_params(immune_frac: float, R0: float, days: int):
    base = load_metapop_params(disease=DiseaseParams(R0=float(R0)), days=days)
    pops = np.asarray(base.populations, dtype=float)
    rec = immune_frac * pops
    return dataclasses.replace(base, initial_recovered=rec)


def _peak_shift(params):
    off = run_coupled_abm(params, BEHAV_OFF)
    on = run_coupled_abm(params, BEHAV_ON)
    p_off = float(np.asarray(off.city_I()).max())
    p_on = float(np.asarray(on.city_I()).max())
    shift = 100.0 * (p_on - p_off) / p_off if p_off > 0 else 0.0
    comp_on = float(on.compliance.mean()) if hasattr(on, "compliance") else float("nan")
    return p_off, p_on, shift, comp_on


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--out", default="simulation/results/abm_v1/regime_rebound.json")
    args = ap.parse_args()

    regimes = {"rebound": 0.05} if args.smoke else IMMUNITY
    r0s = (1.5, 1.9) if args.smoke else R0_ENSEMBLE
    print(f"[regime] {list(regimes)} × R0{r0s} × behaviour{{off,on}}, days={args.days} (exp-Euler)")

    out = {}
    for regime, imm in regimes.items():
        shifts, rows = [], []
        for R0 in r0s:
            p_off, p_on, sh, comp = _peak_shift(_regime_params(imm, R0, args.days))
            shifts.append(sh)
            rows.append({"R0": R0, "peak_off": round(p_off, 1), "peak_on": round(p_on, 1),
                         "peak_shift_pct": round(sh, 2), "mean_compliance_on": round(comp, 4)})
            print(f"  {regime:8s} R0={R0}: peak_off={p_off:>10.0f} peak_on={p_on:>10.0f} "
                  f"shift={sh:+6.1f}%  comp_on={comp:.3f}")
        s = np.asarray(shifts, float)
        out[regime] = {
            "immune_frac": imm, "n_ensemble": len(s),
            "peak_shift_pct_mean": round(float(s.mean()), 2),
            "peak_shift_pct_ci95": [round(float(np.percentile(s, 2.5)), 2),
                                    round(float(np.percentile(s, 97.5)), 2)],
            "peak_shift_pct_min": round(float(s.min()), 2),
            "peak_shift_pct_max": round(float(s.max()), 2),
            "ensemble": rows,
        }

    # regime-dependence 판정
    verdict = None
    if "rebound" in out and "normal" in out:
        reb = abs(out["rebound"]["peak_shift_pct_mean"])
        nrm = abs(out["normal"]["peak_shift_pct_mean"])
        ratio = reb / nrm if nrm > 1e-9 else float("inf")
        verdict = {
            "rebound_effect_pct": out["rebound"]["peak_shift_pct_mean"],
            "normal_effect_pct": out["normal"]["peak_shift_pct_mean"],
            "ratio_rebound_over_normal": round(ratio, 2) if ratio != float("inf") else None,
            "regime_dependent_decisive": bool(reb > 2.0 * max(nrm, 1e-9) and reb > 5.0),
        }
        out["verdict"] = verdict
        print(f"\n[regime] rebound 효과={reb:.1f}% vs normal={nrm:.1f}%  비율={verdict['ratio_rebound_over_normal']}× "
              f"→ regime-dependent decisive = {verdict['regime_dependent_decisive']}")
        print("  (행동 결합은 globally necessary 가 아니라 rebound regime 에서 결정적 — §4.5 정밀화 근거)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"setup": "rebound(5% immune) vs normal(35% immune), R0 parametric CI, exp-Euler, behaviour off vs on(fatigue)",
               "behav_on": dataclasses.asdict(BEHAV_ON), **out},
              open(args.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
