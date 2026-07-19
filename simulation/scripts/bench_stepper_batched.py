"""Batched 4-way SEIR stepper benchmark: amortize per-call overhead.

Tests two workloads:
  (1) Single-step per call — the current Python day-loop pattern.
  (2) Batched 100-step per call — what C/Rust win at (ctypes ~5µs and PyO3
      ~1µs per boundary crossing become negligible when amortized over 100 steps).

Conclusion lives in simulation/results/bench_stepper_batched.json.
"""
from __future__ import annotations

import ctypes
import json
import time
from pathlib import Path

import numpy as np


def _synthetic(G: int = 25, seed: int = 42):
    rng = np.random.default_rng(seed)
    populations = rng.uniform(3.0e5, 5.0e5, size=G)
    mobility = np.eye(G) * 0.7 + rng.uniform(0, 0.05, size=(G, G)) * (1 - np.eye(G))
    mobility /= mobility.sum(axis=1, keepdims=True)
    state = np.zeros((G, 6), dtype=np.float64)
    state[:, 0] = populations * 0.99
    state[:, 1] = populations * 0.005
    state[:, 2] = populations * 0.005
    return state, populations, mobility


def _load_c_batch():
    """Load simulation/c/seir_core.dylib and bind rk4_step_batch_c."""
    import sys
    ext = {"darwin": "dylib", "linux": "so"}.get(sys.platform, "so")
    lib = ctypes.CDLL(f"simulation/c/seir_core.{ext}")
    DP = ctypes.POINTER(ctypes.c_double)
    fn = lib.rk4_step_batch_c
    fn.argtypes = [
        DP, DP, ctypes.c_int, ctypes.c_int, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        DP, DP, DP, DP,
    ]
    fn.restype = None
    return fn


def main():
    state, pops, mob = _synthetic(G=25)
    from simulation.sim.foi import effective_daytime_population
    day = effective_daytime_population(mob, pops)
    vax = np.full(25, 0.001, dtype=np.float64)

    p = dict(beta=0.35, sigma=0.5, gamma=1/3, omega=1/120,
             VE=0.6, V_waning=1/180, ifr=1e-4)
    dt = 0.25
    N_STEPS = 100       # each batch call does this many sub-steps
    N_CALLS = 100       # call count for averaging

    from simulation.sim.stepper import rk4_step_jit, rk4_step_c, rk4_step_rs
    import seir_core as rs_core
    c_batch = _load_c_batch()
    DP = ctypes.POINTER(ctypes.c_double)

    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print(f"║  Batched bench — {N_STEPS} sub-steps per call × {N_CALLS} calls × 5 trials  ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    # ---- Numba: N_STEPS single calls per "batch" ----
    def numba_batch():
        s = state.copy()
        for _ in range(N_STEPS):
            s = rk4_step_jit(s, dt, p["beta"], p["sigma"], p["gamma"], p["omega"],
                             p["VE"], p["V_waning"], p["ifr"], vax, pops, mob, day)
        return s

    # ---- C: single batched call of N_STEPS ----
    def c_batch_call():
        s_in = np.ascontiguousarray(state, dtype=np.float64)
        s_out = np.empty_like(s_in)
        c_batch(
            s_in.ctypes.data_as(DP), s_out.ctypes.data_as(DP),
            ctypes.c_int(25), ctypes.c_int(N_STEPS), ctypes.c_double(dt),
            ctypes.c_double(p["beta"]), ctypes.c_double(p["sigma"]),
            ctypes.c_double(p["gamma"]), ctypes.c_double(p["omega"]),
            ctypes.c_double(p["VE"]), ctypes.c_double(p["V_waning"]),
            ctypes.c_double(p["ifr"]),
            vax.ctypes.data_as(DP),
            np.ascontiguousarray(pops).ctypes.data_as(DP),
            np.ascontiguousarray(mob).ctypes.data_as(DP),
            np.ascontiguousarray(day).ctypes.data_as(DP),
        )
        return s_out

    # ---- C: unbatched (N_STEPS single calls — for comparison) ----
    def c_unbatched():
        s = state.copy()
        for _ in range(N_STEPS):
            s = rk4_step_c(s, dt, p["beta"], p["sigma"], p["gamma"], p["omega"],
                            p["VE"], p["V_waning"], p["ifr"], vax, pops, mob, day)
        return s

    # ---- Rust: batched PyO3 call ----
    def rust_batch():
        return rs_core.rk4_step_batch_rs(
            state, N_STEPS, dt,
            p["beta"], p["sigma"], p["gamma"], p["omega"],
            p["VE"], p["V_waning"], p["ifr"],
            vax, pops, mob, day,
        )

    # ---- Rust: unbatched (N_STEPS single calls) ----
    def rust_unbatched():
        s = state
        for _ in range(N_STEPS):
            s = rk4_step_rs(s, dt, p["beta"], p["sigma"], p["gamma"], p["omega"],
                             p["VE"], p["V_waning"], p["ifr"], vax, pops, mob, day)
        return s

    def bench(label, fn):
        for _ in range(3):
            fn()
        trials = []
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(N_CALLS):
                fn()
            trials.append(time.perf_counter() - t0)
        best = min(trials)
        per_step_us = best / (N_CALLS * N_STEPS) * 1e6
        per_call_ms = best / N_CALLS * 1000
        print(f"  {label:<32s}  {best*1000:8.3f} ms total  |  "
              f"{per_call_ms:7.3f} ms/call  |  {per_step_us:7.4f} µs/step")
        return {"label": label, "best_total_ms": best*1000,
                "per_call_ms": per_call_ms, "per_step_us": per_step_us}

    results = []
    print("── Numba (always unbatched — Python loop around single-step JIT) ──")
    results.append(bench("Numba single-step × N_STEPS", numba_batch))
    print()
    print("── C (via ctypes) ──")
    results.append(bench("C single-step × N_STEPS", c_unbatched))
    results.append(bench("C batched (one call)", c_batch_call))
    print()
    print("── Rust (via PyO3) ──")
    results.append(bench("Rust single-step × N_STEPS", rust_unbatched))
    results.append(bench("Rust batched (one call)", rust_batch))
    print()

    # Summary
    baseline = results[0]["per_step_us"]
    print("━" * 78)
    print(f"Per-step timing (lower is faster) vs Numba baseline ({baseline:.4f} µs/step):")
    print("━" * 78)
    for r in results:
        speedup = baseline / r["per_step_us"]
        mark = "✓" if speedup > 1.0 else " "
        print(f"  {mark} {r['label']:<32s}  {r['per_step_us']:7.4f} µs/step  "
              f"({speedup:5.2f}× vs Numba)")

    from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
    (get_results_dir() / "bench_stepper_batched.json").write_text(
        json.dumps({"N_STEPS": N_STEPS, "N_CALLS": N_CALLS, "results": results}, indent=2)
    )


if __name__ == "__main__":
    main()
