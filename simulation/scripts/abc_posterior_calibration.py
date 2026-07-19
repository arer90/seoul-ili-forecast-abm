"""ABC posterior calibration of behavioral parameters (외부평가 C-1, 2026-06-08).

비판: 3-node grid floor → CV가 grid 인공물, 사후(posterior) 보정 부재. 현대 표준은
ABC/SBI. 이 스크립트는 **ABC rejection**으로 행동 파라미터 (α, τ, θ, κ)의 사후분포 +
**95% credible interval**을 실 Seoul ILI(전국 sentinel 대리, 한 시즌)에서 추정한다.

방법:
  1. prior에서 (α, τ, θ, κ) 다수 추출.
  2. 각 sample을 fit_agent_to_observed(1-point grid = sample)로 ABM 시뮬 → 실 ILI 대비
     R² (timing-faithful: max_shift=2 로 큰 shift shape-matching 차단 — 비판 C-4 동반대응).
  3. distance = 1 − R². 상위 accept_frac 채택 = posterior sample.
  4. posterior 평균 + 2.5/97.5 percentile credible interval 보고.

Grid CV(점추정) → posterior credible interval(불확실도)로 격상.

Run:  .venv/bin/python -m simulation.scripts.abc_posterior_calibration
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path("simulation/results/abc_posterior.json")
PRIORS = {                       # (lo, hi) uniform priors — 문헌 범위
    "alpha": (0.5, 3.5),         # risk sensitivity
    "tau":   (40.0, 140.0),      # fatigue time-constant (days)
    "theta": (0.03, 0.30),       # compliance threshold
    "kappa": (0.05, 0.40),       # reaction strength
}


def main(n_prior: int = 120, accept_frac: float = 0.2, n_agents: int = 500,
         season: int = 2023, seed: int = 42) -> int:
    from simulation.abm.realdata_identifiability import real_season_series
    from simulation.abm.sim_vs_observed import load_seoul_metapop
    from simulation.abm.validate_real import fit_agent_to_observed

    obs = real_season_series(season)
    if len(obs) < 20:
        print(f"season {season} 주차 부족 ({len(obs)})"); return 1
    mp = load_seoul_metapop(days=len(obs) * 7)
    rng = np.random.default_rng(seed)

    names = list(PRIORS)
    samples = {k: rng.uniform(*PRIORS[k], size=n_prior) for k in names}
    dist = np.full(n_prior, np.inf)
    print(f"ABC: {n_prior} prior samples, season {season}-{str(season+1)[2:]}, "
          f"n_agents={n_agents}, accept top {int(accept_frac*100)}%")
    for i in range(n_prior):
        try:
            fit = fit_agent_to_observed(
                obs, mp, n_agents=n_agents, seed=seed,
                alpha_grid=(float(samples["alpha"][i]),),
                kappa_grid=(float(samples["kappa"][i]),),
                tau_grid=(float(samples["tau"][i]),),
                theta_grid=(float(samples["theta"][i]),),
                beta_mult_grid=(1.0,), gamma_mult_grid=(1.0,), max_shift=2)
            r2 = float(fit.r2)
            dist[i] = 1.0 - r2 if np.isfinite(r2) else np.inf
        except Exception:
            dist[i] = np.inf
        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{n_prior} … best dist so far={np.nanmin(dist):.3f}")

    n_acc = max(int(n_prior * accept_frac), 5)
    keep = np.argsort(dist)[:n_acc]
    post = {k: samples[k][keep] for k in names}
    eps = float(dist[keep].max())

    report = {"method": "ABC rejection", "season": f"{season}-{str(season+1)[2:]}",
              "n_prior": n_prior, "n_accepted": n_acc, "epsilon_1_minus_r2": round(eps, 4),
              "n_agents": n_agents, "max_shift": 2, "posterior": {}}
    print(f"\nposterior (n={n_acc}, ε=1−R²≤{eps:.3f}):")
    for k in names:
        v = post[k]
        report["posterior"][k] = {
            "mean": round(float(v.mean()), 4),
            "median": round(float(np.median(v)), 4),
            "ci95": [round(float(np.percentile(v, 2.5)), 4),
                     round(float(np.percentile(v, 97.5)), 4)],
            "prior": list(PRIORS[k]),
        }
        ci = report["posterior"][k]["ci95"]
        # identifiable iff posterior CI is much narrower than the prior range
        width_ratio = (ci[1] - ci[0]) / (PRIORS[k][1] - PRIORS[k][0])
        report["posterior"][k]["ci_width_vs_prior"] = round(width_ratio, 3)
        report["posterior"][k]["identifiable"] = bool(width_ratio < 0.6)
        mark = "✓ 식별(CI<0.6×prior)" if width_ratio < 0.6 else "✗ 약식별(prior만큼 넓음)"
        print(f"  {k:6s} mean={report['posterior'][k]['mean']:.3f} "
              f"CI95={ci} width={width_ratio:.2f}×prior  {mark}")
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
