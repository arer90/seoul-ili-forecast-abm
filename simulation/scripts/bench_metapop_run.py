"""Bench MetapopSEIRVD.run across backends: numba default vs c batched.

Validates:
  * Speedup on full 300-day run with 6 scenarios (~Monte Carlo scale)
  * Numerical agreement (final state match within 1%)
  * Incidence agreement (trapezoid vs rectangle rule within 1%)

Output: stdout + simulation/results/bench_metapop_run.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def main():
    from simulation.sim.metapop_seirvd import MetapopSEIRVD
    from simulation.sim.parameters import MetapopParams, DiseaseParams

    # Synthetic baseline so bench is self-contained (no DB dep)
    # R0 = beta / gamma  →  0.35 / (1/3) = 1.05
    disease = DiseaseParams(
        R0=1.05, sigma=1.0/2.0, gamma=1.0/3.0, omega=1.0/120.0,
        VE=0.6, V_waning=1.0/180.0, ifr=1e-4, report_frac=1.0,
    )
    G = 25
    rng = np.random.default_rng(42)
    pops = rng.uniform(3e5, 5e5, size=G)
    mob = np.eye(G) * 0.7 + rng.uniform(0, 0.05, size=(G, G)) * (1 - np.eye(G))
    mob /= mob.sum(axis=1, keepdims=True)
    vax_rate = np.full(G, 0.001)
    initial_infected = (pops * 0.005).astype(np.float64)

    params = MetapopParams(
        disease=disease,
        district_names=[f"gu{i:02d}" for i in range(G)],
        populations=pops,
        mobility=mob,
        initial_infected=initial_infected,
        initial_recovered=None,
        initial_vaccinated=None,
        vaccination_rate=vax_rate,
        days=300,
        dt=0.25,
        seed=42,
    )

    def bench(label: str, backend: str, n_scenarios: int = 6) -> dict:
        model = MetapopSEIRVD(params)
        # warm-up (numba JIT compile, C library cache)
        for _ in range(2):
            _ = model.run(run_validator=False, backend=backend)
        trials = []
        final_state = None
        final_incidence_total = None
        for _ in range(3):
            t0 = time.perf_counter()
            for _ in range(n_scenarios):
                res = model.run(run_validator=False, backend=backend)
            trials.append(time.perf_counter() - t0)
            final_state = res.state[-1]
            final_incidence_total = float(res.incidence.sum())
        best = min(trials)
        per_scen_ms = best / n_scenarios * 1000
        print(f"  {label:<24s}  best-of-3: {best*1000:7.2f} ms  |  "
              f"{per_scen_ms:6.2f} ms/scenario  (N={n_scenarios})")
        return {
            "label": label, "backend": backend,
            "best_total_ms": best * 1000,
            "per_scen_ms": per_scen_ms,
            "n_scenarios": n_scenarios,
            "final_state_snapshot": final_state.ravel()[:12].tolist(),
            "total_incidence": final_incidence_total,
        }

    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  MetapopSEIRVD.run bench — 300-day × 6 scenarios × 3 trials     ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    numba_res = bench("numba (default)", "numba")
    c_res = bench("c (batched)", "c")

    # Compare
    print()
    print("━" * 70)
    speedup = numba_res["per_scen_ms"] / c_res["per_scen_ms"]
    print(f"C batched speedup: {speedup:.2f}× vs numba")
    print()
    a = np.array(numba_res["final_state_snapshot"])
    b = np.array(c_res["final_state_snapshot"])
    rel_state = float(np.max(np.abs(a - b) / (np.abs(a) + 1e-9)))
    inc_a = numba_res["total_incidence"]
    inc_b = c_res["total_incidence"]
    rel_inc = float(abs(inc_a - inc_b) / (abs(inc_a) + 1e-9))
    print(f"Numerical agreement:")
    print(f"  final state max rel diff: {rel_state*100:.4f}%  "
          f"({'✓ < 1%' if rel_state < 0.01 else '✗ > 1%'})")
    print(f"  total incidence rel diff: {rel_inc*100:.4f}%  "
          f"({'✓ < 1%' if rel_inc < 0.01 else '✗ > 1%'})  "
          f"(trapezoid vs rectangle rule)")

    out = {
        "config": {"G": G, "days": 300, "n_scenarios_per_trial": 6},
        "numba": numba_res, "c_batched": c_res,
        "speedup": speedup,
        "final_state_rel_diff": rel_state,
        "total_incidence_rel_diff": rel_inc,
    }
    from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
    _bench_out = get_results_dir() / "bench_metapop_run.json"
    _bench_out.write_text(json.dumps(out, indent=2))
    print()
    print(f"Wrote: {_bench_out}")


if __name__ == "__main__":
    main()
