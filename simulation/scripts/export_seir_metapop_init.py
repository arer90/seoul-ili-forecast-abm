"""
Export the metapop SEIR-V-D initial structure as a static JSON that the
browser WASM demo can fetch at page load.

Writes ``web/public/aggregates/seir-metapop-init.json``::

    {
      "district_names": string[25],
      "populations":    number[25],           // residents per gu
      "mobility_flat":  number[625],          // 25x25 row-major, row-stochastic
      "n_gu": 25,
      "source": "commuter_matrix + daily_population_district (DB)",
      "generated_at": "2026-04-21T...Z"
    }

Why static instead of MCP at request-time?
  - pops + M change at most monthly (when import_external runs)
  - browser what-if slider fires every 27 ms — no reason to hit an API
  - one static JSON (~11 KB) loaded once, reused for every slider tick

Usage
-----
    .venv\\Scripts\\python.exe -m simulation.scripts.export_seir_metapop_init

Idempotent. Safe to re-run after any commuter / population refresh.
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import sys

import numpy as np

from simulation.database.config import SEOUL_GU_ORDERED
from simulation.sim.io import load_mobility_matrix, load_populations


WEB_AGGREGATES = (
    pathlib.Path(__file__).resolve().parents[2]
    / "web"
    / "public"
    / "aggregates"
)
OUT_PATH = WEB_AGGREGATES / "seir-metapop-init.json"


def main() -> int:
    districts = list(SEOUL_GU_ORDERED)
    pops = load_populations(districts)
    M = load_mobility_matrix(districts)

    # Sanity: row-stochastic within 1e-9.
    row_sums = M.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-9):
        print(
            f"[warn] mobility row sums not exactly 1 "
            f"(min={row_sums.min():.6f}, max={row_sums.max():.6f}); "
            f"WASM will still accept, simulator re-normalises internally",
            file=sys.stderr,
        )

    payload = {
        "district_names": districts,
        "populations": [float(x) for x in pops.tolist()],
        "mobility_flat": [float(x) for x in M.flatten().tolist()],
        "n_gu": len(districts),
        "source": "commuter_matrix + daily_population_district",
        "generated_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Compact JSON — browser fetch doesn't need pretty-print.
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))

    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"[ok] wrote {OUT_PATH} ({size_kb:.1f} KB, n_gu={len(districts)})")
    print(
        f"     pops min={pops.min():.0f} max={pops.max():.0f} sum={pops.sum():.0f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
