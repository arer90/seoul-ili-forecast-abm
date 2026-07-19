"""Stochastic ensemble + global sensitivity analysis for the agent-world ABM.

Addresses two external-review gaps (2026-06-06 methodological critique):

  1. **Stochastic ensemble** (Lee et al. 2015, JASSS 18(4):4) — the binomial
     tau-leap agent-world is genuinely stochastic, so a *single* seeded run is
     one draw, not the model's behaviour. `run_ensemble` runs it across many
     seeds and returns the output *distribution* (mean + percentile CI) plus a
     variance-stabilization curve to justify the replicate count.

  2. **Global variance-based sensitivity** (Marino et al. 2008, J Theor Biol
     254:178) — one-at-a-time (OAT) holds all-but-one fixed and is discouraged.
     `global_sensitivity` Latin-Hypercube-samples the full parameter space and
     computes Partial Rank Correlation Coefficients (PRCC) with significance,
     so every parameter is varied simultaneously over its range.

Dependency-free (numpy + scipy.stats.rankdata only — no SALib), reproducible
(seeded LHS + seeded runs). Smoke: `python -m simulation.abm.sensitivity`.
"""
from __future__ import annotations


import numpy as np

from .agent_kernel import run_agent_world

#: Baseline flu-like agent-world configuration (one stochastic draw ≈ 0.04 s).
BASE: dict = dict(
    N=8000, T_days=120, beta=0.9, sigma=0.5, gamma=0.25, delta=0.001, nu=0.002,
)

#: Global-SA parameter ranges (low, high) around the flu baseline. Disease
#: kinetics + the two behavioural means exposed by run_agent_world.
SA_RANGES: dict[str, tuple[float, float]] = {
    "beta": (0.5, 1.3),
    "sigma": (0.30, 0.70),
    "gamma": (0.15, 0.40),
    "delta": (0.0005, 0.0030),
    "nu": (0.0, 0.005),
    "theta_mean": (0.30, 0.70),
    "alpha_mean": (0.10, 0.60),
}


def _metrics(out: dict, n: int) -> dict[str, float]:
    """Epidemic summaries from one agent-world output (S/E/I/R/V/D daily)."""
    I = np.asarray(out["I"], float)
    np.asarray(out["S"], float)
    D = np.asarray(out["D"], float)
    E = np.asarray(out["E"], float)
    R = np.asarray(out["R"], float)
    return {
        "peak_day": float(np.argmax(I)),
        "peak_infected": float(I.max()),
        "attack_rate": float((E[-1] + I[-1] + R[-1] + D[-1]) / n) if n else 0.0,
        "deaths": float(D[-1]),
    }


# ── 1. Stochastic ensemble ──────────────────────────────────────────────────
def run_ensemble(n_seeds: int = 200, *, base: dict | None = None,
                 seed0: int = 0) -> dict:
    """Run the agent-world across `n_seeds` and summarize the output distribution.

    Args:
        n_seeds: number of stochastic replicates (each a distinct global_seed).
        base: agent-world kwargs (default `BASE`).
        seed0: first seed; seeds are seed0 .. seed0+n_seeds-1 (reproducible).

    Returns:
        ``{"n_seeds", "metrics": {name: {mean, sd, ci2.5, ci97.5, values}},
           "variance_stabilization": {"n": [...], "running_cv_attack": [...]}}``
        — variance_stabilization shows the running CV of attack_rate vs replicate
        count, so the reader can see where the estimate stabilizes (Lee 2015).

    Performance: ~`n_seeds` × 0.04 s. Side effects: none (pure compute).
    """
    base = {**BASE, **(base or {})}
    n = base["N"]
    rows: list[dict[str, float]] = []
    for s in range(seed0, seed0 + n_seeds):
        rows.append(_metrics(run_agent_world(**{**base, "global_seed": s}), n))
    names = list(rows[0].keys())
    arr = {k: np.array([r[k] for r in rows], float) for k in names}
    metrics = {
        k: {
            "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "ci2.5": float(np.percentile(v, 2.5)),
            "ci97.5": float(np.percentile(v, 97.5)),
            "values": v.round(4).tolist(),
        }
        for k, v in arr.items()
    }
    # variance stabilization: running CV of attack_rate as seeds accumulate
    a = arr["attack_rate"]
    ns = list(range(10, n_seeds + 1, max(1, n_seeds // 12)))
    running_cv = [
        float(a[:m].std(ddof=1) / a[:m].mean()) if a[:m].mean() else 0.0 for m in ns
    ]
    return {
        "n_seeds": n_seeds,
        "metrics": metrics,
        "variance_stabilization": {"n": ns, "running_cv_attack": running_cv},
    }


# ── 2. Global sensitivity (LHS + PRCC) ──────────────────────────────────────
def lhs(ranges: dict[str, tuple[float, float]], n: int, seed: int = 42) -> tuple[np.ndarray, list[str]]:
    """Latin Hypercube sample of `n` points over the named ranges.

    Returns (X of shape (n, p), param names). Each column is a stratified,
    randomly-permuted sample across its [low, high] interval.
    """
    rng = np.random.default_rng(seed)
    names = list(ranges)
    X = np.empty((n, len(names)))
    for j, name in enumerate(names):
        lo, hi = ranges[name]
        # stratified midpoints + jitter, then permute strata
        edges = (np.arange(n) + rng.random(n)) / n
        X[:, j] = lo + edges[rng.permutation(n)] * (hi - lo)
    return X, names


def prcc(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Partial Rank Correlation Coefficient of each input vs output (Marino 2008).

    Rank-transform, then PRCC_i = -C⁻¹[i,k] / sqrt(C⁻¹[i,i]·C⁻¹[k,k]) where C is
    the rank-correlation matrix of [inputs, output] and k is the output index.

    Returns (prcc per input, two-sided p-value per input). p from
    t = prcc·sqrt((n-2-p)/(1-prcc²)), df = n-2-p.
    """
    from scipy.stats import rankdata, t as t_dist

    n, p = X.shape
    R = np.column_stack([rankdata(X[:, j]) for j in range(p)] + [rankdata(y)])
    C = np.corrcoef(R, rowvar=False)
    Cinv = np.linalg.pinv(C)
    k = p
    pr = np.array([
        -Cinv[i, k] / np.sqrt(Cinv[i, i] * Cinv[k, k]) for i in range(p)
    ])
    pr = np.clip(pr, -0.999999, 0.999999)
    df = max(1, n - 2 - p)
    tval = pr * np.sqrt(df / (1 - pr**2))
    pval = 2 * t_dist.sf(np.abs(tval), df)
    return pr, pval


def global_sensitivity(n_samples: int = 500, *, ranges: dict | None = None,
                       base: dict | None = None, seed: int = 42) -> dict:
    """LHS over the parameter space → PRCC of each parameter on each metric.

    Args:
        n_samples: LHS sample size (every parameter varied simultaneously).
        ranges: param → (low, high) (default `SA_RANGES`).
        base: fixed agent-world kwargs (default `BASE`); sampled params override.
        seed: LHS + run seed.

    Returns:
        ``{"n_samples", "params", "prcc": {metric: {param: {prcc, p}}}}`` —
        |PRCC| ranks each parameter's monotone influence on the output, with
        significance. Performance: ~`n_samples` × 0.04 s.
    """
    ranges = ranges or SA_RANGES
    base = {**BASE, **(base or {})}
    n = base["N"]
    X, names = lhs(ranges, n_samples, seed)
    metric_names = ["peak_day", "peak_infected", "attack_rate", "deaths"]
    Y = {m: np.empty(n_samples) for m in metric_names}
    for i in range(n_samples):
        kw = {**base, **{names[j]: float(X[i, j]) for j in range(len(names))}, "global_seed": seed + i}
        m = _metrics(run_agent_world(**kw), n)
        for k in metric_names:
            Y[k][i] = m[k]
    out: dict[str, dict] = {}
    for m in metric_names:
        pr, pv = prcc(X, Y[m])
        out[m] = {names[j]: {"prcc": round(float(pr[j]), 3), "p": float(f"{pv[j]:.2e}")}
                  for j in range(len(names))}
    return {"n_samples": n_samples, "params": names, "prcc": out}


def _smoke() -> None:
    print("[ensemble] 80 seeds ...")
    ens = run_ensemble(n_seeds=80)
    for k, v in ens["metrics"].items():
        print(f"  {k:14} mean {v['mean']:.4g}  95% CI [{v['ci2.5']:.4g}, {v['ci97.5']:.4g}]")
    vs = ens["variance_stabilization"]
    print(f"  running CV(attack) n={vs['n'][0]}→{vs['n'][-1]}: "
          f"{vs['running_cv_attack'][0]:.3f} → {vs['running_cv_attack'][-1]:.3f}")
    print("[global-SA] LHS 300 × PRCC ...")
    sa = global_sensitivity(n_samples=300)
    for metric in ["attack_rate", "peak_day"]:
        ranked = sorted(sa["prcc"][metric].items(), key=lambda kv: -abs(kv[1]["prcc"]))
        print(f"  {metric}: " + ", ".join(f"{p}={d['prcc']:+.2f}" for p, d in ranked[:4]))


if __name__ == "__main__":
    _smoke()
