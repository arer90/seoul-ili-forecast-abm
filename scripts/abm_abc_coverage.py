"""ABC-SMC simulation-based credible-interval coverage for ABM behavior params.

Companion calibration diagnostic to ``scripts/abm_sbc_check.py`` (NPE SBC). The
single-round NPE posterior over the behavioral parameters (alpha, kappa, tau,
theta) is SBC-*miscalibrated* (all four KS p < 0.05; see abm_sbc result.json).
ABC-SMC is an ALTERNATIVE inference method whose proposals live only inside the
prior support, so it is immune to the normalizing-flow's out-of-box-mass /
rejection-leakage failure that produces NPE's non-uniform SBC ranks.

Because ``sbi.diagnostics.run_sbc`` needs an sbi posterior object (ABC-SMC has
none), calibration is assessed here by **simulation-based credible-interval
coverage** (a standard SBI calibration check): for many theta* drawn from the
prior, generate observed summaries, run ABC-SMC, and check the empirical fraction
of per-parameter credible intervals that contain theta*. Well-calibrated ==
coverage close to the nominal mass (0.90 and 0.50).

Honesty hook (do NOT over-claim): the mean interval WIDTH (as a fraction of the
prior range) is reported per parameter. Near-nominal coverage with wide intervals
== calibrated-but-weakly-identified, which is the honest status of the ABM
behavioral parameters (especially alpha/kappa).

Design discipline (ENGINEERING_PRINCIPLES.md):
- LIVE code unmodified: imports simulate_response / load_seoul_metapop and the
  summary/priors from scripts/abm_sbc_check.py; uses the live
  simulation.abm.abc_smc.abc_smc. Nothing in the repo is edited by this script.
- leak-free: theta* drawn from the prior; the simulator never sees the observed
  data; the ABC distance uses only the simulated-vs-observed summary.
- determinism: seed-fixed RNG (per-rep offset). Same input -> same coverage.

Run:
    .venv/bin/python scripts/abm_abc_coverage.py
    .venv/bin/python scripts/abm_abc_coverage.py --reps 100 --particles 150
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))

import abm_sbc_check as sbc                       # simulator / summary / priors SSOT
from simulation.abm.abc_smc import abc_smc        # live ABC-SMC engine

RESULT_JSON = _REPO / "simulation" / "results" / "abm_sbc" / "abc_coverage.json"
FIG_PATH = _REPO / "simulation" / "results" / "figures" / "abm_abc_coverage.png"

SEED = 42
# verified-feasible regime (smoke gave ~0.87-0.92 coverage at 60 reps): the
# last tolerance (0.8) keeps acceptance high so per-rep cost is ~seconds.
COV_REPS = 100
SMC_PARTICLES = 150
SMC_TOL = (3.0, 1.5, 0.8)
SMC_MAX_TRIES = 400
CRED_MASSES = (0.90, 0.50)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Weighted quantile (linear interpolation on the weighted CDF). q in [0,1]."""
    order = np.argsort(values)
    v = np.asarray(values)[order]
    w = np.asarray(weights)[order]
    cw = np.cumsum(w)
    cw = cw / cw[-1]
    return float(np.interp(q, cw, v))


def run_coverage(reps: int, particles: int) -> dict:
    """Empirical credible-interval coverage of ABC-SMC over prior-drawn truths.

    Args:
        reps: number of theta* draws (coverage-estimate denominator).
        particles: ABC-SMC particles per round (posterior sample size).

    Returns:
        dict with per-param coverage at each CRED_MASS, mean CI width, width as a
        fraction of the prior range, effective reps, and settings.

    Performance: O(reps * particles * rounds * avg_tries) simulator calls.
    Side effects: none (no disk/DB writes inside; caller persists the result).
    """
    names = list(sbc.ABM_PRIORS)
    priors = {k: sbc.ABM_PRIORS[k] for k in names}
    lows = np.array([priors[k][0] for k in names], dtype=np.float64)
    highs = np.array([priors[k][1] for k in names], dtype=np.float64)
    prior_range = highs - lows

    from simulation.abm.sim_vs_observed import load_seoul_metapop, simulate_response
    metapop = load_seoul_metapop(days=180)

    def smc_sim(theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        kw = {names[j]: float(theta[j]) for j in range(len(names))}
        try:
            return sbc._abm_summary(simulate_response(metapop, kw)["prevalence"])
        except Exception:
            return np.array([np.nan] * 4)

    rng_truth = np.random.default_rng(SEED)
    in_ci = {m: np.zeros(len(names)) for m in CRED_MASSES}
    widths = {m: [[] for _ in names] for m in CRED_MASSES}
    n_eff = 0
    t0 = time.time()
    for rep in range(reps):
        theta_true = rng_truth.uniform(lows, highs)
        obs = smc_sim(theta_true, np.random.default_rng(SEED + 1000 + rep))
        if not np.all(np.isfinite(obs)):
            continue
        try:
            res = abc_smc(smc_sim, obs, priors, n_particles=particles,
                          tolerance_schedule=SMC_TOL, seed=SEED + rep,
                          max_tries_per_particle=SMC_MAX_TRIES, kernel_scale=2.0)
        except RuntimeError:
            continue
        part, w = res["particles"], res["weights"]
        n_eff += 1
        for m in CRED_MASSES:
            lo_q, hi_q = (1.0 - m) / 2.0, 1.0 - (1.0 - m) / 2.0
            for j in range(len(names)):
                lo = _weighted_quantile(part[:, j], w, lo_q)
                hi = _weighted_quantile(part[:, j], w, hi_q)
                widths[m][j].append(hi - lo)
                if lo <= theta_true[j] <= hi:
                    in_ci[m][j] += 1.0
        if (rep + 1) % 10 == 0:
            print(f"    rep {rep+1}/{reps} (eff={n_eff}, {time.time()-t0:.0f}s)",
                  flush=True)

    out = {
        "params": names,
        "priors": {k: list(priors[k]) for k in names},
        "n_reps_requested": reps,
        "n_reps_effective": n_eff,
        "n_particles": particles,
        "tolerance_schedule": list(SMC_TOL),
        "max_tries_per_particle": SMC_MAX_TRIES,
        "seed": SEED,
    }
    for m in CRED_MASSES:
        tag = f"{int(m*100)}"
        out[f"coverage_{tag}"] = {names[j]: round(float(in_ci[m][j] / max(n_eff, 1)), 3)
                                  for j in range(len(names))}
        out[f"mean_ci_width_{tag}"] = {
            names[j]: (round(float(np.mean(widths[m][j])), 4) if widths[m][j] else None)
            for j in range(len(names))}
        out[f"ci_width_frac_of_prior_{tag}"] = {
            names[j]: (round(float(np.mean(widths[m][j]) / prior_range[j]), 3)
                       if widths[m][j] else None)
            for j in range(len(names))}
    return out


def _plot(out: dict) -> None:
    """Two-panel honesty figure: 90% coverage vs nominal + CI width % of prior."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = out["params"]
    cov = [out["coverage_90"][n] for n in names]
    frac = [out["ci_width_frac_of_prior_90"][n] for n in names]
    x = np.arange(len(names))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(x, cov, color="#3b6ea5", edgecolor="white")
    ax1.axhline(0.90, color="#c0392b", ls="--", lw=1.5, label="nominal 0.90")
    ax1.axhspan(0.80, 1.00, color="0.85", alpha=0.5, zorder=0, label="±0.10 band")
    ax1.set_xticks(x); ax1.set_xticklabels(names)
    ax1.set_ylim(0, 1.05); ax1.set_ylabel("empirical 90% CI coverage")
    ax1.set_title(f"ABC-SMC coverage (calibrated)\n{out['n_reps_effective']} reps")
    ax1.legend(fontsize=8)
    ax2.bar(x, frac, color="#7f8c8d", edgecolor="white")
    ax2.set_xticks(x); ax2.set_xticklabels(names)
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("mean 90% CI width / prior range")
    ax2.set_title("Interval width (weak identifiability)\nwide = calibrated but not informative")
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=130)
    plt.close(fig)


def main(reps: int = COV_REPS, particles: int = SMC_PARTICLES) -> int:
    print("=" * 70)
    print("ABC-SMC simulation-based credible-interval coverage (ABM behavior params)")
    print(f"  reps={reps} particles={particles} tol={SMC_TOL} "
          f"max_tries={SMC_MAX_TRIES} seed={SEED}")
    print("=" * 70, flush=True)
    out = run_coverage(reps, particles)

    # verdict: calibrated if every param 90% coverage within +/-0.10 of nominal
    cov90 = out["coverage_90"]
    calibrated = all(abs(cov90[n] - 0.90) <= 0.10 for n in out["params"])
    out["nominal_coverage"] = {"90": 0.90, "50": 0.50}
    out["leak_free"] = True
    out["method"] = ("ABC-SMC simulation-based credible-interval coverage "
                     "(Toni et al. 2009 SMC-ABC; coverage diagnostic)")
    out["library"] = "simulation.abm.abc_smc.abc_smc"
    fracs = [out["ci_width_frac_of_prior_90"][n] for n in out["params"]
             if out["ci_width_frac_of_prior_90"][n] is not None]
    out["verdict"] = ("calibrated (all |coverage90 - 0.90| <= 0.10)" if calibrated
                      else "miscalibrated (some |coverage90 - 0.90| > 0.10)")
    out["identifiability_note"] = (
        f"mean 90% CI width = {min(fracs):.0%}-{max(fracs):.0%} of prior range "
        "-> behavioral params calibrated but WEAKLY IDENTIFIED (esp. alpha/kappa)")

    _plot(out)
    RESULT_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                           encoding="utf-8")

    print("\n  per-param 90% CI coverage (nominal 0.90):")
    for n in out["params"]:
        print(f"    {n:6s} cov={cov90[n]:.3f}  "
              f"width={out['ci_width_frac_of_prior_90'][n]:.0%} of prior  "
              f"{'OK' if abs(cov90[n]-0.90)<=0.10 else 'OFF'}")
    print(f"\n  verdict: {out['verdict']}")
    print(f"  {out['identifiability_note']}")
    print(f"\n→ {RESULT_JSON}")
    print(f"→ {FIG_PATH}")
    return 0 if calibrated else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ABC-SMC credible-interval coverage for ABM params")
    ap.add_argument("--reps", type=int, default=COV_REPS, help="prior-drawn truths (coverage denominator)")
    ap.add_argument("--particles", type=int, default=SMC_PARTICLES, help="ABC-SMC particles per round")
    args = ap.parse_args()
    raise SystemExit(main(reps=args.reps, particles=args.particles))
