"""simulation/scripts/bench_seir_python.py
=============================================================================
BENCH — Python Metapop SEIR-V-D (the "reference" implementation)

Fair-comparison harness vs the Rust/WASM build in _past/seir-wasm.

What we measure
---------------
Pure ``MetapopSEIRVD(params).run(interventions=[], run_validator=False)``
wall-clock time, over a **fixed** scenario that matches the Rust build's
parameter space 1:1 (same R0, γ, σ, dt, days, populations, mobility).

What we report
--------------
  - N=10 timed runs after 1 warm-up
  - median, p5, p95 (ms)
  - total CPU time + peak process RSS (for apples-to-apples vs WASM
    which runs in Node with a different baseline memory footprint)

Writes ``simulation/results/bench_seir_python.json`` that the meta
runner joins with ``bench_seir_wasm.json`` to produce one comparison
table.
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from pathlib import Path

import numpy as np
import psutil

from simulation.sim.metapop_seirvd import MetapopSEIRVD
from simulation.sim.parameters import DiseaseParams, MetapopParams


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = ROOT / "simulation" / "results" / "bench_seir_python.json"


# =========================================================================
# Fixed scenario — MUST match bench_seir_wasm.mjs exactly.
# =========================================================================

SEOUL_25GU_NAMES = [
    "Jongno", "Jung", "Yongsan", "Seongdong", "Gwangjin",
    "Dongdaemun", "Jungnang", "Seongbuk", "Gangbuk", "Dobong",
    "Nowon", "Eunpyeong", "Seodaemun", "Mapo", "Yangcheon",
    "Gangseo", "Guro", "Geumcheon", "Yeongdeungpo", "Dongjak",
    "Gwanak", "Seocho", "Gangnam", "Songpa", "Gangdong",
]

# Roughly proportional to 2024 KOSIS registered-resident counts (millions).
SEOUL_25GU_POPS = np.array([
    150_000, 125_000, 230_000, 290_000, 345_000,
    335_000, 390_000, 430_000, 300_000, 320_000,
    515_000, 470_000, 310_000, 370_000, 450_000,
    570_000, 395_000, 230_000, 370_000, 390_000,
    495_000, 410_000, 540_000, 660_000, 440_000,
], dtype=float)   # sum ≈ 9.56 M — Seoul-scale


def _build_mobility(n_gu: int, *, stay_home: float = 0.55) -> np.ndarray:
    """Row-stochastic mobility matrix.

    Each row i sums to 1: ``stay_home`` on diagonal, rest distributed
    uniformly across the other G-1 districts. Matches the Rust side
    which uses the same uniform fallback.
    """
    M = np.full((n_gu, n_gu), (1.0 - stay_home) / (n_gu - 1), dtype=float)
    np.fill_diagonal(M, stay_home)
    return M


def build_params(n_days: int = 365, dt: float = 0.25) -> MetapopParams:
    n_gu = len(SEOUL_25GU_POPS)

    # Disease: default DiseaseParams has R0=1.4, γ=1/3.5, σ=1/2, but with
    # waning (ω, V_waning) which Rust doesn't model. Zero them out for a
    # clean structural comparison.
    disease = DiseaseParams(
        R0=1.4,
        gamma=1.0 / 3.5,
        sigma=1.0 / 2.0,
        omega=0.0,            # no R→S waning (Rust side lacks this term)
        VE=0.50,
        V_waning=0.0,         # no V→S waning
        ifr=0.001,
        report_frac=0.10,
    )

    # Seed: 100 infected in Gangnam (index 22)
    initial_infected = np.zeros(n_gu, dtype=float)
    initial_infected[22] = 100.0

    return MetapopParams(
        disease=disease,
        populations=SEOUL_25GU_POPS,
        mobility=_build_mobility(n_gu),
        district_names=SEOUL_25GU_NAMES,
        initial_infected=initial_infected,
        vaccination_rate=0.0,      # no baseline vax
        days=n_days,
        dt=dt,
        seed=42,
    )


# =========================================================================
# Benchmark
# =========================================================================

def time_one_run(params: MetapopParams) -> tuple[float, float]:
    """Return (wall_ms, peak_sum_I_for_sanity)."""
    sim = MetapopSEIRVD(params)
    t0 = time.perf_counter_ns()
    result = sim.run(interventions=[], run_validator=False)
    t1 = time.perf_counter_ns()
    peak = float(result.city_total("I").max())
    return (t1 - t0) / 1e6, peak


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="timed runs")
    ap.add_argument("--warmup", type=int, default=1, help="warmup runs")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--dt", type=float, default=0.25)
    args = ap.parse_args()

    log.info(f"Building params — 25 gu, {args.days} days, dt={args.dt}")
    params = build_params(n_days=args.days, dt=args.dt)
    params.validate()

    log.info(f"Warm-up x {args.warmup}")
    for _ in range(args.warmup):
        time_one_run(params)

    log.info(f"Timed runs x {args.n}")
    wall_ms_list: list[float] = []
    peaks: list[float] = []
    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    peak_rss = rss_before
    for i in range(args.n):
        ms, peak = time_one_run(params)
        wall_ms_list.append(ms)
        peaks.append(peak)
        peak_rss = max(peak_rss, proc.memory_info().rss)
        log.info(f"  [{i+1:02d}/{args.n}]  wall={ms:.1f} ms  peak_I={peak:,.0f}")

    wall_ms_list.sort()
    report = {
        "impl": "python_numpy",
        "scenario": {
            "n_gu": len(SEOUL_25GU_POPS),
            "days": args.days,
            "dt": args.dt,
            "R0": 1.4,
            "gamma_days": 3.5,
            "sigma_days": 2.0,
            "initial_infected_total": 100.0,
            "initial_infected_gu": "Gangnam",
        },
        "n_runs": args.n,
        "n_warmup": args.warmup,
        "wall_ms": {
            "median": statistics.median(wall_ms_list),
            "p05": wall_ms_list[max(0, int(0.05 * args.n))],
            "p95": wall_ms_list[min(args.n - 1, int(0.95 * args.n))],
            "min": wall_ms_list[0],
            "max": wall_ms_list[-1],
            "all": wall_ms_list,
        },
        "sanity": {
            "peak_I_mean": statistics.mean(peaks),
            "peak_I_stdev": statistics.stdev(peaks) if len(peaks) > 1 else 0.0,
        },
        "memory_mb": {
            "rss_before": round(rss_before / 1024 / 1024, 1),
            "rss_peak": round(peak_rss / 1024 / 1024, 1),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2))

    med = report["wall_ms"]["median"]
    p05 = report["wall_ms"]["p05"]
    p95 = report["wall_ms"]["p95"]
    print("")
    print(f"=== Python SEIR-V-D bench  (n={args.n}, warmup={args.warmup}) ===")
    print(f"  wall_ms median : {med:.1f}  ({p05:.1f} - {p95:.1f})")
    print(f"  peak_I mean    : {report['sanity']['peak_I_mean']:,.0f}")
    print(f"  peak RSS       : {report['memory_mb']['rss_peak']:.1f} MB")
    print(f"  → {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
