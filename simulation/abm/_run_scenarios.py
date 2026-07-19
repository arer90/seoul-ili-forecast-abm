"""Runner entry for simulation.abm.scenarios (S1-S6 + spatial analytics).

Usage::

    python -m simulation.abm._run_scenarios \
        --out-dir simulation/results/abm_scenarios_v1 \
        --days 180 --seed-infected 1000
"""
from __future__ import annotations

import argparse
import logging
import sys

import numpy as np

from simulation.abm.scenarios import run_scenario_suite, write_artefacts
from simulation.sim.io import load_metapop_params
from simulation.sim.parameters import MetapopParams

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("abm_scenarios")


def main() -> int:
    ap = argparse.ArgumentParser()
    from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
    ap.add_argument("--out-dir", default=str(get_results_dir() / "abm_scenarios_v1"))
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--seed-infected", type=float, default=1000.0)
    args = ap.parse_args()

    params = load_metapop_params()
    G = int(params.populations.size)
    I0 = np.full(G, args.seed_infected, dtype=float)
    scen_params = MetapopParams(
        disease=params.disease,
        populations=params.populations,
        mobility=params.mobility,
        district_names=params.district_names,
        initial_infected=I0,
        initial_recovered=params.initial_recovered,
        initial_vaccinated=params.initial_vaccinated,
        vaccination_rate=params.vaccination_rate,
        days=args.days, dt=params.dt, seed=params.seed,
    )
    log.info("scenarios: G=%d days=%d I0/gu=%.0f", G, args.days, args.seed_infected)

    report = run_scenario_suite(scen_params)
    mob = np.asarray(scen_params.mobility, dtype=float)
    extras = write_artefacts(report, args.out_dir, mob)

    # Print compact summary
    print("\n=== S1-S6 policy comparison ===")
    for row in extras["policy_table"]:
        print(f"  {row['id']:3} {row['name']:40s}  "
              f"peak={row['peak_city_I']:>8.0f}  shift%={row['peak_shift_pct_vs_S1']:+.2f}  "
              f"compl={row['mean_compliance']:.3f}  attack={row['attack_rate_city']:.3f}")
    print("\n=== Spatial propagation ===")
    for row in extras["spatial"]:
        print(f"  {row['scenario_id']:3}  first={row['first_peak_district']:8s}  "
              f"last={row['last_peak_district']:8s}  "
              f"lag={row['mean_peak_lag_days']:.1f}d  "
              f"corr(centrality, peak_day)={row['centrality_peakday_correlation']:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
