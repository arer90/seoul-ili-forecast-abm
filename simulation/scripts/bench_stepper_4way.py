"""4-way SEIR stepper benchmark: pure numpy / Numba / C / Rust.

Usage:
    uv run python simulation/scripts/bench_stepper_4way.py [--steps N] [--G N]

Builds C if missing, assumes Rust already built via `maturin develop --release`.
Reports absolute µs/step and relative speedup against numpy baseline.

Output: stdout + simulation/results/bench_stepper_4way.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np


def _synthetic_params(G: int = 25, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    populations = rng.uniform(3.0e5, 5.0e5, size=G)
    mobility = np.eye(G) * 0.7
    mobility += rng.uniform(0.0, 0.05, size=(G, G)) * (1.0 - np.eye(G))
    mobility /= mobility.sum(axis=1, keepdims=True)
    state = np.zeros((G, 6), dtype=np.float64)
    state[:, 0] = populations * 0.99
    state[:, 1] = populations * 0.005
    state[:, 2] = populations * 0.005
    return {
        "state": state,
        "beta": 0.35, "sigma": 0.5, "gamma": 1.0 / 3.0,
        "omega": 1.0 / 120.0, "VE": 0.6, "V_waning": 1.0 / 180.0,
        "ifr": 1e-4,
        "vax_rate": np.full(G, 0.001, dtype=np.float64),
        "populations": populations,
        "mobility": mobility,
    }


def _bench(label: str, fn, *, n_steps: int, warm: int = 3) -> dict:
    # Warm-up (JIT compile, dyload cache).
    for _ in range(warm):
        fn()
    # Multiple trials, take min (standard bench practice — eliminates OS jitter).
    trials = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(n_steps):
            fn()
        trials.append(time.perf_counter() - t0)
    best = min(trials)
    per_step_us = best / n_steps * 1e6
    print(f"  {label:<28s}  best-of-5: {best*1000:8.3f} ms  |  "
          f"{per_step_us:8.4f} µs/step  (n={n_steps})")
    return {"label": label, "best_total_ms": best * 1000,
            "per_step_us": per_step_us, "n_steps": n_steps,
            "all_trials_ms": [t * 1000 for t in trials]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--G", type=int, default=25)
    from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
    ap.add_argument("--out", default=str(get_results_dir() / "bench_stepper_4way.json"))
    args = ap.parse_args()

    # Ensure C backend is built (idempotent).
    c_lib = Path("simulation/c/seir_core.dylib")
    if not c_lib.exists() and Path("simulation/c/seir_core.so").exists():
        pass  # Linux build
    elif not c_lib.exists():
        print("Building C backend...")
        subprocess.run(["bash", "simulation/c/build.sh"], check=True)

    from simulation.sim.stepper import (
        HAS_NUMBA, HAS_C_BACKEND, HAS_RUST_BACKEND,
        seirvd_derivative, rk4_step_jit, rk4_step_c, rk4_step_rs,
    )
    from simulation.sim.foi import effective_daytime_population

    print()
    print(f"╔══════════════════════════════════════════════════════════════════╗")
    print(f"║  4-way SEIR RK4 stepper bench  (G={args.G}, {args.steps} steps × 5 trials) ║")
    print(f"╠══════════════════════════════════════════════════════════════════╣")
    print(f"║  Numba: {HAS_NUMBA}  |  C: {HAS_C_BACKEND}  |  Rust: {HAS_RUST_BACKEND}                             ║")
    print(f"╚══════════════════════════════════════════════════════════════════╝")
    print()

    params = _synthetic_params(G=args.G)
    day = effective_daytime_population(params["mobility"], params["populations"])
    dt = 0.25

    # 1. pure numpy
    def numpy_path():
        s = params["state"]
        pk = {k: v for k, v in params.items() if k != "state"}
        pk["daytime_pop"] = day
        k1 = seirvd_derivative(s, **pk)
        k2 = seirvd_derivative(s + 0.5 * dt * k1, **pk)
        k3 = seirvd_derivative(s + 0.5 * dt * k2, **pk)
        k4 = seirvd_derivative(s + dt * k3, **pk)
        return np.clip(s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), 0.0, None)

    def numba_path():
        return rk4_step_jit(
            params["state"], dt,
            params["beta"], params["sigma"], params["gamma"], params["omega"],
            params["VE"], params["V_waning"], params["ifr"],
            params["vax_rate"], params["populations"], params["mobility"], day,
        )

    def c_path():
        return rk4_step_c(
            params["state"], dt,
            params["beta"], params["sigma"], params["gamma"], params["omega"],
            params["VE"], params["V_waning"], params["ifr"],
            params["vax_rate"], params["populations"], params["mobility"], day,
        )

    def rust_path():
        return rk4_step_rs(
            params["state"], dt,
            params["beta"], params["sigma"], params["gamma"], params["omega"],
            params["VE"], params["V_waning"], params["ifr"],
            params["vax_rate"], params["populations"], params["mobility"], day,
        )

    results = []
    results.append(_bench("pure numpy", numpy_path, n_steps=args.steps))
    if HAS_NUMBA:
        results.append(_bench("numba @njit", numba_path, n_steps=args.steps))
    if HAS_C_BACKEND:
        results.append(_bench("C (ctypes)", c_path, n_steps=args.steps))
    if HAS_RUST_BACKEND:
        results.append(_bench("Rust (pyo3)", rust_path, n_steps=args.steps))

    # Equivalence check — all backends should produce identical output to float precision.
    print()
    print("Equivalence check (should be < 1e-10):")
    ref = numpy_path()
    for res_fn, name in [(numba_path, "Numba"), (c_path, "C"), (rust_path, "Rust")]:
        got = res_fn()
        diff = float(np.abs(got - ref).max())
        marker = "✓" if diff < 1e-10 else "✗"
        print(f"  {marker} numpy vs {name:5s}: max |diff| = {diff:.3e}")

    # Summary
    baseline = results[0]["per_step_us"]
    print()
    print("━" * 70)
    print("SPEEDUP vs numpy baseline")
    print("━" * 70)
    for r in results:
        speedup = baseline / r["per_step_us"]
        print(f"  {r['label']:<28s}  {r['per_step_us']:8.4f} µs/step  "
              f"({speedup:6.2f}× vs numpy)")

    # Final ratios
    if len(results) >= 4:
        numba_us = results[1]["per_step_us"]
        c_us = results[2]["per_step_us"]
        rust_us = results[3]["per_step_us"]
        print()
        print("Head-to-head between native backends:")
        print(f"  C / Numba  = {c_us / numba_us:.2f}×  (Numba is {numba_us / c_us:.2f}× {'faster' if numba_us < c_us else 'slower'} than C)")
        print(f"  Rust / Numba = {rust_us / numba_us:.2f}×")
        print(f"  Rust / C = {rust_us / c_us:.2f}×")

    # Persist
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "config": {"G": args.G, "steps": args.steps,
                   "HAS_NUMBA": HAS_NUMBA, "HAS_C": HAS_C_BACKEND, "HAS_RUST": HAS_RUST_BACKEND},
        "results": results,
    }, indent=2))
    print()
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
