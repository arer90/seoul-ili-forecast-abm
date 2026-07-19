"""Benchmark Phase A Numba optimizations: before/after for 4 hot paths.

Usage:
    uv run python simulation/scripts/bench_phase_a_numba.py

Output: stdout table + simulation/results/bench_phase_a.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def _bench(label: str, fn, *, n_reps: int, warm: int = 3) -> dict:
    """Time fn() over n_reps calls, preceded by warm-up."""
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_reps):
        fn()
    elapsed = time.perf_counter() - t0
    per_call_us = elapsed / n_reps * 1e6
    per_call_ms = per_call_us / 1000.0
    print(f"  {label:<42s}  {elapsed*1000:8.2f} ms total  |  "
          f"{per_call_ms:9.4f} ms/call  ({n_reps} reps)")
    return {"label": label, "total_ms": elapsed * 1000,
            "per_call_ms": per_call_ms, "reps": n_reps}


def bench_bootstrap_pi(n_reps: int = 30) -> list[dict]:
    """Bench phase6 _bootstrap_pi (2000 boot iterations × percentile)."""
    print("━" * 70)
    print("1. _bootstrap_pi (phase6) — 2000 bootstrap iterations per call")
    print("━" * 70)
    from simulation.pipeline import intervals as p6

    # Fake test data: 345-week ILI series with noise
    rng = np.random.default_rng(42)
    y_true = rng.gamma(2.0, 1.5, size=345)
    y_pred = y_true + rng.normal(0, 0.3, size=345)

    results = []

    # Numba ON (current state)
    assert p6._HAS_NUMBA
    r1 = _bench("numba @njit (current)", lambda: p6._bootstrap_pi(y_true, y_pred, n_boot=2000), n_reps=n_reps)
    results.append(r1)

    # Pure numpy baseline (force fallback by monkey-patching)
    p6._HAS_NUMBA = False
    try:
        r2 = _bench("pure numpy (fallback)", lambda: p6._bootstrap_pi(y_true, y_pred, n_boot=2000), n_reps=max(3, n_reps // 10))
        results.append(r2)
    finally:
        p6._HAS_NUMBA = True

    speedup = r2["per_call_ms"] / r1["per_call_ms"]
    print(f"  → Numba speedup: {speedup:.2f}×")
    print()
    return results


def bench_jackknife_plus(n_reps: int = 100) -> list[dict]:
    """Bench conformal jackknife+ CV+ loop (n_test × n_cal order stats)."""
    print("━" * 70)
    print("2. jackknife_plus_interval (conformal) — n_cal × n_test order stats")
    print("━" * 70)
    from simulation.models import conformal as cf

    rng = np.random.default_rng(42)
    n_cal, n_test = 200, 40
    fold_preds_test = rng.normal(5.0, 1.0, size=(n_cal, n_test))
    residuals_cal = np.abs(rng.normal(0, 0.5, size=n_cal))

    results = []
    assert cf._HAS_NUMBA
    r1 = _bench("numba @njit (current)",
                lambda: cf.jackknife_plus_interval(fold_preds_test, residuals_cal, alpha=0.1),
                n_reps=n_reps)
    results.append(r1)

    cf._HAS_NUMBA = False
    try:
        r2 = _bench("pure numpy (fallback)",
                    lambda: cf.jackknife_plus_interval(fold_preds_test, residuals_cal, alpha=0.1),
                    n_reps=max(3, n_reps // 10))
        results.append(r2)
    finally:
        cf._HAS_NUMBA = True

    speedup = r2["per_call_ms"] / r1["per_call_ms"]
    print(f"  → Numba speedup: {speedup:.2f}×")
    print()
    return results


def bench_wavelet(n_reps: int = 200) -> list[dict]:
    """Bench feature_engine wavelet causal convolution at 3 scales."""
    print("━" * 70)
    print("3. _causal_convolve_ricker (wavelet features) — 345w × 3 scales")
    print("━" * 70)
    from simulation.models.feature_engine import transforms as tr

    # Need polars + DataFrame to call _add_wavelet_features directly
    import polars as pl
    rng = np.random.default_rng(42)
    df = pl.DataFrame({
        "idx": np.arange(345),
        "val": rng.gamma(2.0, 1.5, size=345).astype(np.float64),
    })

    results = []
    assert tr._HAS_NUMBA
    r1 = _bench("numba @njit (current)",
                lambda: tr._add_wavelet_features(df, "val", scales=[4, 8, 16]),
                n_reps=n_reps)
    results.append(r1)

    tr._HAS_NUMBA = False
    try:
        r2 = _bench("pure numpy (fallback)",
                    lambda: tr._add_wavelet_features(df, "val", scales=[4, 8, 16]),
                    n_reps=max(3, n_reps // 5))
        results.append(r2)
    finally:
        tr._HAS_NUMBA = True

    speedup = r2["per_call_ms"] / r1["per_call_ms"]
    print(f"  → Numba speedup: {speedup:.2f}×")
    print()
    return results


def bench_behavioural_ode(n_reps: int = 2000) -> list[dict]:
    """Bench ABM R/F/C Euler step (per-day update across 25 districts)."""
    print("━" * 70)
    print("4. _behavioural_step (ABM ODE) — R/F/C coupled update on 25 gu")
    print("━" * 70)
    from simulation.abm import behavioural as bh

    rng = np.random.default_rng(42)
    G = 25
    I_prev = rng.uniform(100, 1000, size=G).astype(np.float64)
    N_prev = rng.uniform(3e5, 5e5, size=G).astype(np.float64)
    R_prev = rng.uniform(0, 0.3, size=G).astype(np.float64)
    F_prev = rng.uniform(0, 0.2, size=G).astype(np.float64)
    C_prev = np.zeros(G, dtype=np.float64)
    alpha_, lambda_R, kappa, theta, delta, rho, strength, beta_0 = (
        2.0, 1/30, 0.3, 0.2, 0.05, 1/30, 0.5, 0.35,
    )

    results = []
    assert bh._HAS_NUMBA

    def numba_path():
        return bh._behavioural_step_jit(I_prev, N_prev, R_prev, F_prev, C_prev,
                                         alpha_, lambda_R, kappa, theta,
                                         delta, rho, strength, beta_0)

    def numpy_path():
        # Replicate the original logic without the JIT
        prev_ratio = I_prev / N_prev
        dR = alpha_ * prev_ratio - lambda_R * R_prev
        R_next = np.maximum(R_prev + dR, 0.0)
        compliant = ((R_next - kappa * F_prev) > theta).astype(float)
        dF = delta * C_prev - rho * F_prev
        F_next = np.maximum(F_prev + dF, 0.0)
        scale_district = (1.0 - strength * compliant) ** 2
        beta_eff = beta_0 * scale_district
        return R_next, F_next, compliant, beta_eff, float(beta_0 * scale_district.mean())

    r1 = _bench("numba @njit (current)", numba_path, n_reps=n_reps)
    results.append(r1)
    r2 = _bench("pure numpy (equivalent)", numpy_path, n_reps=n_reps)
    results.append(r2)

    speedup = r2["per_call_ms"] / r1["per_call_ms"]
    print(f"  → Numba speedup: {speedup:.2f}×")
    print()
    return results


def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Phase A Numba Benchmark — 4 hot paths, before vs after          ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    out = {
        "bootstrap_pi": bench_bootstrap_pi(n_reps=30),
        "jackknife_plus": bench_jackknife_plus(n_reps=100),
        "wavelet": bench_wavelet(n_reps=200),
        "behavioural_ode": bench_behavioural_ode(n_reps=2000),
    }

    from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
    out_path = get_results_dir() / "bench_phase_a.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote: {out_path}")

    # Print summary
    print()
    print("━" * 70)
    print("SUMMARY")
    print("━" * 70)
    for name, trials in out.items():
        if len(trials) >= 2:
            numba_ms = trials[0]["per_call_ms"]
            numpy_ms = trials[-1]["per_call_ms"]
            speedup = numpy_ms / numba_ms
            print(f"  {name:<25s}  Numba: {numba_ms:9.4f} ms  |  "
                  f"numpy: {numpy_ms:9.4f} ms  |  {speedup:5.2f}× speedup")


if __name__ == "__main__":
    main()
