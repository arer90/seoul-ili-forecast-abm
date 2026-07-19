"""SBI (neural posterior) calibration of behavioral params on real Seoul ILI.

외부평가 3차 권고: ABC rejection을 신경 사후추정(NPE)으로 격상. TDD(test_sbi_calibration)로
파이프라인을 toy에서 검증한 뒤, 같은 run_sbi를 ABM에 적용한다. 시뮬레이터 = simulate_response,
요약통계 = scale-invariant shape(전국 ILI vs 시뮬 prevalence 스케일 차 제거).

ABC(abc_posterior.json)와 비교 가능: 둘 다 약식별이면 P4(ILI만 부족) 헤드라인을 두 방법이
독립 확증. sbi 부재 시 ABC로 fallback.

Run:  .venv/bin/python -m simulation.scripts.sbi_posterior_calibration
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path("simulation/results/sbi_posterior.json")
PRIORS = {"alpha": (0.5, 3.5), "kappa": (0.05, 0.40),
          "tau": (40.0, 140.0), "theta": (0.03, 0.30)}


def _summary(traj) -> np.ndarray:
    """scale-invariant shape 통계 (peak 위치·정규화 rise/fall·peak/mean)."""
    t = np.asarray(traj, dtype=np.float64)
    t = t[np.isfinite(t)]
    if len(t) < 6 or t.max() <= 0:
        return np.array([np.nan] * 4)
    pk, n = int(np.argmax(t)), len(t)
    rise = (t[pk] - t[0]) / (t[pk] + 1e-9) / max(pk, 1)
    fall = (t[pk] - t[-1]) / (t[pk] + 1e-9) / max(n - pk, 1)
    return np.array([pk / n, rise * 52, fall * 52, t[pk] / (t.mean() + 1e-9)])


def main(n_sims: int = 400, season: int = 2023) -> int:
    from simulation.abm.realdata_identifiability import real_season_series
    from simulation.abm.sim_vs_observed import load_seoul_metapop, simulate_response
    try:
        from simulation.abm.sbi_calibration import run_sbi
    except ImportError:
        print("sbi 미설치 → ABC(abc_posterior_calibration)로 fallback 권장"); return 1

    names = list(PRIORS)
    lows = [PRIORS[k][0] for k in names]
    highs = [PRIORS[k][1] for k in names]
    mp = load_seoul_metapop(days=180)
    x_obs = _summary(real_season_series(season))
    print(f"SBI(NPE): {n_sims} sims, season {season}-{str(season+1)[2:]}, x_obs={np.round(x_obs,3)}")

    def simulator(theta):
        kw = {names[j]: float(theta[j]) for j in range(len(names))}
        try:
            return _summary(simulate_response(mp, kw)["prevalence"])
        except Exception:
            return np.array([np.nan] * 4)

    res = run_sbi(simulator, lows, highs, x_obs, n_sims=n_sims, n_posterior=2000, seed=42)
    report = {"method": "SBI (NPE, sbi pkg)", "season": f"{season}-{str(season+1)[2:]}",
              "n_sims": res["n_sims_used"], "params": {}}
    print(f"\nSBI posterior (n_sims={res['n_sims_used']}):")
    for i, k in enumerate(names):
        w = res["ci_width_vs_prior"][i]
        report["params"][k] = {"posterior_mean": res["posterior_mean"][i],
                               "ci95": res["ci95"][i], "ci_width_vs_prior": w,
                               "identifiable": bool(w < 0.6), "prior": list(PRIORS[k])}
        mark = "✓ 식별" if w < 0.6 else "✗ 약식별"
        print(f"  {k:6s} mean={res['posterior_mean'][i]:.3f} CI95={res['ci95'][i]} "
              f"width={w:.2f}×prior  {mark}")
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {OUT}  (ABC abc_posterior.json와 비교 가능)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
