"""Sobol global sensitivity of the behavioral ABM (외부평가 C-2, 2026-06-08).

비판: monotonicity/hysteresis 방향검사만 있고 global SA가 없다. 3차 외부평가: hand-rolled
Sobol은 분산 추정량 부호·교차항 오류 위험 → **검증된 SALib(Herman & Usher 2017, Saltelli
2010)로 교체**. SALib 없으면 manual Saltelli fallback(이식성 원칙 #1).

출력 = ABM 피크 유병률(`simulate_response`). 입력 = 행동 4파라미터 (α, κ, τ, θ).
Sᵢ = 단독 기여, STᵢ = 총효과, STᵢ−Sᵢ = 상호작용.

Run:  .venv/bin/python -m simulation.scripts.sobol_sensitivity
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path("simulation/results/sobol_sensitivity.json")
RANGES = {
    "alpha": (0.5, 3.5),
    "kappa": (0.05, 0.40),
    "tau":   (40.0, 140.0),
    "theta": (0.03, 0.30),
}


def _manual_saltelli(problem, N, f, rng):
    """SALib 부재 시 fallback (Saltelli 2010 / Jansen 추정량). 이식성 원칙 #1."""
    D = problem["num_vars"]
    lo = np.array([b[0] for b in problem["bounds"]])
    hi = np.array([b[1] for b in problem["bounds"]])
    sc = lambda u: lo + u * (hi - lo)
    A, B = sc(rng.random((N, D))), sc(rng.random((N, D)))
    yA = np.array([f(A[j]) for j in range(N)])
    yB = np.array([f(B[j]) for j in range(N)])
    var = float(np.var(np.concatenate([yA, yB])))
    S1, ST = {}, {}
    for i in range(D):
        ABi = A.copy(); ABi[:, i] = B[:, i]
        yABi = np.array([f(ABi[j]) for j in range(N)])
        S1[i] = max(float(np.mean(yB * (yABi - yA)) / var), 0.0) if var > 0 else 0.0
        ST[i] = max(float(0.5 * np.mean((yA - yABi) ** 2) / var), 0.0) if var > 0 else 0.0
    return {"S1": [S1[i] for i in range(D)], "ST": [ST[i] for i in range(D)],
            "library": "manual_saltelli_fallback", "total_variance": var}


def main(N: int = 64, seed: int = 42) -> int:
    from simulation.abm.sim_vs_observed import load_seoul_metapop, simulate_response
    mp = load_seoul_metapop(days=180)
    names = list(RANGES)
    problem = {"num_vars": len(names), "names": names,
               "bounds": [list(RANGES[k]) for k in names]}

    def f(row) -> float:
        kw = {names[j]: float(row[j]) for j in range(len(names))}
        try:
            return float(np.asarray(simulate_response(mp, kw)["prevalence"], float).max())
        except Exception:
            return float("nan")

    try:
        from SALib.sample.sobol import sample as saltelli_sample
        from SALib.analyze.sobol import analyze as sobol_analyze
        X = saltelli_sample(problem, N, calc_second_order=True, seed=seed)
        print(f"Sobol (SALib): N={N}, evals={len(X)}")
        Y = np.array([f(x) for x in X])
        # NaN 보호: 결측은 평균으로 (SALib는 finite 요구)
        if not np.all(np.isfinite(Y)):
            Y = np.nan_to_num(Y, nan=float(np.nanmean(Y)))
        Si = sobol_analyze(problem, Y, calc_second_order=True, seed=seed)
        S1, ST = [float(x) for x in Si["S1"]], [float(x) for x in Si["ST"]]
        lib = "SALib"
        var = float(np.var(Y))
    except ImportError:
        rng = np.random.default_rng(seed)
        man = _manual_saltelli(problem, N, f, rng)
        S1, ST, lib, var = man["S1"], man["ST"], man["library"], man["total_variance"]
        print(f"Sobol (fallback): N={N}")

    report = {"method": f"Sobol ({lib})", "N": N, "output": "ABM peak prevalence",
              "total_variance": round(var, 2), "indices": {}}
    print(f"\nSobol 지수 (lib={lib}, output 분산={var:.1f}):")
    for i, k in enumerate(names):
        s1, st = max(S1[i], 0.0), max(ST[i], 0.0)
        report["indices"][k] = {"S1_first_order": round(s1, 4), "ST_total": round(st, 4),
                                "interaction_ST_minus_S1": round(max(st - s1, 0.0), 4)}
        print(f"  {k:6s} S1={s1:.3f}  ST={st:.3f}  interaction={max(st-s1,0):.3f}")
    valid = {k: v for k, v in report["indices"].items()}
    dominant = max(valid, key=lambda k: valid[k]["ST_total"])
    report["dominant_parameter"] = dominant
    print(f"\n→ 출력 분산 지배 파라미터: {dominant}")
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
