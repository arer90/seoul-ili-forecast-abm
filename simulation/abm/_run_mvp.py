"""Standalone MVP runner for simulation.abm — used for thesis §4.16 evidence.

Run with::

    python -m simulation.abm._run_mvp

Writes artefacts under simulation/results/abm_v1/ and prints a compact
summary table suitable for §4.16 retrospective update.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from simulation.abm import (
    run_invariant_test,
    run_rebound_scenario,
)
from simulation.sim.io import load_metapop_params

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("abm_mvp")


def _load_params(
    days: int = 365,
    seed_infected: float = 1000.0,
) -> "simulation.sim.parameters.MetapopParams":
    params = load_metapop_params()
    if seed_infected > 0 and np.all(np.asarray(params.initial_infected) == 0):
        G = int(params.populations.size)
        seeded = np.full(G, seed_infected, dtype=float)
        params = params.__class__(
            disease=params.disease,
            populations=params.populations,
            mobility=params.mobility,
            district_names=params.district_names,
            initial_infected=seeded,
            initial_recovered=params.initial_recovered,
            initial_vaccinated=params.initial_vaccinated,
            vaccination_rate=params.vaccination_rate,
            days=days,
            dt=params.dt,
            seed=params.seed,
        )
    else:
        params = params.__class__(
            disease=params.disease,
            populations=params.populations,
            mobility=params.mobility,
            district_names=params.district_names,
            initial_infected=params.initial_infected,
            initial_recovered=params.initial_recovered,
            initial_vaccinated=params.initial_vaccinated,
            vaccination_rate=params.vaccination_rate,
            days=days,
            dt=params.dt,
            seed=params.seed,
        )
    return params


def main() -> int:
    ap = argparse.ArgumentParser()
    from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
    ap.add_argument("--out-dir", default=str(get_results_dir() / "abm_v1"))
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--seed-infected", type=float, default=1000.0)
    ap.add_argument("--invariant-tol", type=float, default=1e-6)
    ap.add_argument(
        "--rebound",
        action="store_true",
        default=True,
        help="Run the 2022-24 rebound comparison (default)",
    )
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    log.info("Loading metapop params from DB ...")
    params = _load_params(days=args.days, seed_infected=args.seed_infected)
    G = int(params.populations.size)
    log.info("G=%d districts, days=%d, initial_infected sum=%.0f",
             G, params.days, float(np.asarray(params.initial_infected).sum()))

    # ------------------------------------------------------------------
    # Step 1 -- alpha=0 invariant test
    # ------------------------------------------------------------------
    log.info("[1/3] alpha=0 invariant test (kernel-only <=> behaviour-off ABM)")
    invariant = run_invariant_test(params, tolerance=args.invariant_tol)
    log.info(
        "  -> passed=%s  rmse=%.3e  max_err=%.3e  compliance_off=%.3f",
        invariant["passed"], invariant["rmse"], invariant["max_abs_err"],
        invariant["abm_mean_compliance"],
    )
    with open(out / "invariant_test.json", "w", encoding="utf-8") as f:
        json.dump(invariant, f, indent=2)

    # ------------------------------------------------------------------
    # Step 2 -- 2022-24 rebound scenario
    # ------------------------------------------------------------------
    log.info("[2/3] S-rebound scenario (behaviour-off vs behaviour-on)")
    rebound = run_rebound_scenario(params)
    log.info(
        "  peak_off=%.0f @ day %d   peak_on=%.0f @ day %d   shift=%.1f%%",
        rebound["peak_off"], rebound["day_of_peak_off"],
        rebound["peak_on"], rebound["day_of_peak_on"],
        rebound["peak_shift_pct"],
    )
    log.info(
        "  mean_compliance_on=%.3f  mean_compliance_off=%.3f",
        rebound["mean_compliance_on"], rebound["mean_compliance_off"],
    )

    # persist a compact result file
    persist = {
        k: v for k, v in rebound.items()
        if k not in ("city_I_off", "city_I_on", "days", "district_names")
    }
    persist["n_days"] = len(rebound["city_I_off"])
    persist["trajectory_file"] = "trajectory.npz"
    with open(out / "rebound_summary.json", "w", encoding="utf-8") as f:
        json.dump(persist, f, indent=2)
    np.savez_compressed(
        out / "trajectory.npz",
        days=np.asarray(rebound["days"]),
        city_I_off=np.asarray(rebound["city_I_off"]),
        city_I_on=np.asarray(rebound["city_I_on"]),
        district_names=np.asarray(rebound["district_names"], dtype=object),
    )
    log.info("  wrote %s and %s",
             out / "rebound_summary.json", out / "trajectory.npz")

    # ------------------------------------------------------------------
    # Step 3 -- thesis-ready summary table (plain text)
    # ------------------------------------------------------------------
    log.info("[3/3] writing summary table for thesis §4.16 update")
    lines = [
        "# simulation.abm MVP -- §4.16 retrospective evidence",
        "",
        "## Invariant test (alpha=0 => behaviour-off <=> kernel-only)",
        f"  passed           : {invariant['passed']}",
        f"  RMSE             : {invariant['rmse']:.3e}",
        f"  max abs error    : {invariant['max_abs_err']:.3e}",
        f"  tolerance        : {invariant['tolerance']:.1e}",
        f"  off compliance   : {invariant['abm_mean_compliance']:.4f}  (must be 0)",
        "",
        "## S-rebound scenario",
        f"  behaviour-off peak I  : {rebound['peak_off']:.0f} at day {rebound['day_of_peak_off']}",
        f"  behaviour-on  peak I  : {rebound['peak_on']:.0f} at day {rebound['day_of_peak_on']}",
        f"  peak shift            : {rebound['peak_shift_pct']:+.1f}% (on vs off)",
        f"  on mean compliance    : {rebound['mean_compliance_on']:.3f}",
        f"  off mean compliance   : {rebound['mean_compliance_off']:.3f}",
        "",
        "## ABM parameter set (behaviour-on)",
    ]
    for k, v in rebound["behaviour_on_params"].items():
        lines.append(f"  {k:6}= {v}")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("  wrote %s", out / "summary.md")

    # return non-zero if invariant failed
    return 0 if invariant["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
