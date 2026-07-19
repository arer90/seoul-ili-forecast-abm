"""simulation/scripts/bench_seir_python_ext.py
=============================================================================
EXTENDED BENCH — Python numpy SEIR. Adds:
  (1) Cold-start: time from fresh process to first run complete.
      Measured indirectly by importing simulation.sim *inside main()* and
      timing the first call separately from the warm-steady state.
  (2) Scaling: 90 / 180 / 365 / 730 day horizons.
  (3) Memory delta per run: tracemalloc + psutil RSS before / after loop.
  (4) N=30 timed runs for tighter p95.

Writes simulation/results/bench_seir_python_ext.json.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import statistics
import time
import tracemalloc
from pathlib import Path

import numpy as np
import psutil


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = ROOT / "simulation" / "results" / "bench_seir_python_ext.json"

SEOUL_25GU_NAMES = [
    "Jongno", "Jung", "Yongsan", "Seongdong", "Gwangjin",
    "Dongdaemun", "Jungnang", "Seongbuk", "Gangbuk", "Dobong",
    "Nowon", "Eunpyeong", "Seodaemun", "Mapo", "Yangcheon",
    "Gangseo", "Guro", "Geumcheon", "Yeongdeungpo", "Dongjak",
    "Gwanak", "Seocho", "Gangnam", "Songpa", "Gangdong",
]
SEOUL_25GU_POPS = np.array([
    150_000, 125_000, 230_000, 290_000, 345_000,
    335_000, 390_000, 430_000, 300_000, 320_000,
    515_000, 470_000, 310_000, 370_000, 450_000,
    570_000, 395_000, 230_000, 370_000, 390_000,
    495_000, 410_000, 540_000, 660_000, 440_000,
], dtype=float)


def _build_mobility(n_gu: int, stay_home: float = 0.55) -> np.ndarray:
    M = np.full((n_gu, n_gu), (1.0 - stay_home) / (n_gu - 1), dtype=float)
    np.fill_diagonal(M, stay_home)
    return M


def build_params(n_days: int):
    # Import inside function so main() can time the cold-start cost.
    from simulation.sim.parameters import DiseaseParams, MetapopParams
    n_gu = len(SEOUL_25GU_POPS)
    disease = DiseaseParams(
        R0=1.4, gamma=1.0 / 3.5, sigma=1.0 / 2.0,
        omega=0.0, VE=0.50, V_waning=0.0, ifr=0.001, report_frac=0.10,
    )
    initial_infected = np.zeros(n_gu, dtype=float)
    initial_infected[22] = 100.0
    return MetapopParams(
        disease=disease, populations=SEOUL_25GU_POPS,
        mobility=_build_mobility(n_gu), district_names=SEOUL_25GU_NAMES,
        initial_infected=initial_infected, vaccination_rate=0.0,
        days=n_days, dt=0.25, seed=42,
    )


def run_once(params):
    from simulation.sim.metapop_seirvd import MetapopSEIRVD
    t0 = time.perf_counter_ns()
    sim = MetapopSEIRVD(params)
    result = sim.run(interventions=[], run_validator=False)
    t1 = time.perf_counter_ns()
    peak = float(result.city_total("I").max())
    return (t1 - t0) / 1e6, peak


def bench_horizon(days: int, n: int) -> dict:
    """One complete bench (warmup + N timed) for a single horizon."""
    proc = psutil.Process()
    gc.collect()
    rss_before = proc.memory_info().rss

    # Cold measurement = time of the 1st call (no warmup)
    params = build_params(days)
    cold_ms, peak = run_once(params)

    # Warm measurements: N more runs after the cold.
    # tracemalloc would inflate timings ~3x, so we measure allocations
    # in a *separate* non-timed pass.
    wall = []
    peaks = [peak]
    peak_rss = rss_before
    for _ in range(n):
        ms, pk = run_once(params)
        wall.append(ms)
        peaks.append(pk)
        peak_rss = max(peak_rss, proc.memory_info().rss)

    # Separate allocation pass (one run only, not timed).
    tracemalloc.start()
    _ = run_once(params)
    tm_after = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    wall.sort()
    return {
        "days": days,
        "n_warm_runs": n,
        "cold_ms": cold_ms,
        "warm_median_ms": statistics.median(wall),
        "warm_p05_ms": wall[max(0, int(0.05 * n))],
        "warm_p95_ms": wall[min(n - 1, int(0.95 * n))],
        "warm_min_ms": wall[0],
        "warm_max_ms": wall[-1],
        "warm_stdev_ms": statistics.stdev(wall) if n > 1 else 0.0,
        "warm_all_ms": wall,
        "peak_I_mean": statistics.mean(peaks),
        "rss_before_mb": round(rss_before / 1024 / 1024, 1),
        "rss_peak_mb": round(peak_rss / 1024 / 1024, 1),
        "rss_delta_mb": round((peak_rss - rss_before) / 1024 / 1024, 1),
        "tracemalloc_peak_after_mb": round(tm_after / 1024 / 1024, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--horizons", type=str, default="90,180,365,730")
    args = ap.parse_args()

    # Process-level cold-start: time from "script entry" to first import done.
    # We re-import here so the import cost is counted as part of the cold-start
    # of the first horizon.
    t_entry = time.perf_counter_ns()
    import simulation.sim.parameters  # noqa
    import simulation.sim.metapop_seirvd  # noqa
    t_imports_done = time.perf_counter_ns()
    import_ms = (t_imports_done - t_entry) / 1e6

    horizons = [int(x) for x in args.horizons.split(",")]
    log.info(f"import time (numpy + simulation.sim.*): {import_ms:.1f} ms")
    log.info(f"horizons: {horizons}  n_warm={args.n}")

    results = []
    for H in horizons:
        log.info(f"--- horizon = {H} days ---")
        r = bench_horizon(H, args.n)
        log.info(f"  cold={r['cold_ms']:.1f} ms   warm median={r['warm_median_ms']:.1f} ms "
                 f"(p5={r['warm_p05_ms']:.1f} p95={r['warm_p95_ms']:.1f})   "
                 f"RSSΔ={r['rss_delta_mb']:+.1f} MB  peak_I={r['peak_I_mean']:,.0f}")
        results.append(r)

    report = {
        "impl": "python_numpy",
        "runtime_info": {
            "python": __import__("sys").version.split()[0],
            "numpy": np.__version__,
        },
        "cold_import_ms": import_ms,
        "horizons": results,
        "scenario_fixed": {
            "n_gu": 25, "dt": 0.25, "R0": 1.4, "gamma_days": 3.5,
            "sigma_days": 2.0, "initial_infected": 100, "seed_gu": "Gangnam",
            "stay_home": 0.55, "mobility": "uniform off-diagonal",
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2))
    print(f"-> {OUT_JSON}")


if __name__ == "__main__":
    main()
