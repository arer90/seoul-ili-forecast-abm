"""Benchmark Numba-JIT rk4_step_jit vs pure-numpy rk4_step.

Usage:
    uv run python simulation/scripts/bench_stepper_numba.py [--steps N] [--G N]

Output: stdout table + simulation/results/bench_stepper_numba.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def _synthetic_params(G: int = 25, seed: int = 42):
    rng = np.random.default_rng(seed)
    # Realistic Seoul-like population around 400k per gu
    populations = rng.uniform(3.0e5, 5.0e5, size=G)
    # Row-stochastic mobility (diagonal-dominant; off-diag small)
    mobility = np.eye(G) * 0.7
    mobility += rng.uniform(0.0, 0.05, size=(G, G)) * (1.0 - np.eye(G))
    mobility /= mobility.sum(axis=1, keepdims=True)
    # Initial state: 99% S, 0.5% E, 0.5% I
    S0 = populations * 0.99
    E0 = populations * 0.005
    I0 = populations * 0.005
    state = np.zeros((G, 6), dtype=np.float64)
    state[:, 0] = S0
    state[:, 1] = E0
    state[:, 2] = I0
    return {
        "state": state,
        "beta": 0.35,
        "sigma": 1.0 / 2.0,  # 2-day incubation
        "gamma": 1.0 / 3.0,  # 3-day infectious
        "omega": 1.0 / 120.0,  # waning
        "VE": 0.6,
        "V_waning": 1.0 / 180.0,
        "ifr": 1e-4,
        "vax_rate": np.full(G, 0.001, dtype=np.float64),
        "populations": populations,
        "mobility": mobility,
    }


def _bench(label: str, fn, *, n_steps: int, warm: int = 3):
    # Warm-up (triggers JIT compile on the first call).
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        fn()
    elapsed = time.perf_counter() - t0
    per_step_us = elapsed / n_steps * 1e6
    print(f"  {label:<30s}  {elapsed*1000:7.2f} ms total  |  {per_step_us:7.3f} µs/step")
    return {"label": label, "total_ms": elapsed * 1000, "per_step_us": per_step_us}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000, help="rk4_step calls per run")
    ap.add_argument("--G", type=int, default=25, help="number of districts")
    from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
    ap.add_argument("--out", default=str(get_results_dir() / "bench_stepper_numba.json"))
    args = ap.parse_args()

    from simulation.sim.stepper import (
        HAS_NUMBA,
        rk4_step,
        rk4_step_jit,
    )
    from simulation.sim.foi import effective_daytime_population

    print(f"Benchmark: G={args.G}, steps={args.steps}, Numba available={HAS_NUMBA}")
    print("-" * 70)

    params = _synthetic_params(G=args.G)
    daytime_pop = effective_daytime_population(params["mobility"], params["populations"])
    params["daytime_pop"] = daytime_pop
    dt = 0.25

    results = []

    # Path 1: pure numpy fallback (force by rebuilding kwargs and bypassing JIT path).
    def numpy_path():
        from simulation.sim.stepper import seirvd_derivative
        s = params["state"].copy()
        pk = {k: v for k, v in params.items() if k != "state"}
        k1 = seirvd_derivative(s, **pk)
        k2 = seirvd_derivative(s + 0.5 * dt * k1, **pk)
        k3 = seirvd_derivative(s + 0.5 * dt * k2, **pk)
        k4 = seirvd_derivative(s + dt * k3, **pk)
        return np.clip(s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), 0.0, None)

    results.append(_bench("pure numpy rk4", numpy_path, n_steps=args.steps))

    # Path 2: Numba JIT rk4
    if HAS_NUMBA:
        def jit_path():
            return rk4_step_jit(
                params["state"], dt,
                params["beta"], params["sigma"], params["gamma"], params["omega"],
                params["VE"], params["V_waning"], params["ifr"],
                params["vax_rate"], params["populations"], params["mobility"],
                daytime_pop,
            )
        results.append(_bench("numba @njit rk4", jit_path, n_steps=args.steps))

        # Path 3: dispatcher (exercise end-to-end call via rk4_step)
        def dispatcher_path():
            return rk4_step(
                params["state"], dt,
                params_kwargs={k: v for k, v in params.items() if k != "state"},
            )
        results.append(_bench("dispatcher rk4_step", dispatcher_path, n_steps=args.steps))

    print("-" * 70)
    if len(results) >= 2:
        speedup = results[0]["per_step_us"] / results[1]["per_step_us"]
        print(f"Numba speedup over pure numpy: {speedup:.2f}×")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "config": {"G": args.G, "steps": args.steps, "HAS_NUMBA": HAS_NUMBA},
        "results": results,
    }, indent=2))
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
