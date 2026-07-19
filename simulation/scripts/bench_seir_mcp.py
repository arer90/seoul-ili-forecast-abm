"""simulation/scripts/bench_seir_mcp.py
=============================================================================
BENCH — End-to-end scenario_run via the MCP stdio bridge.

This measures what the Next.js web app actually pays on the "simulate"
button: LLM/UI → POST /api/mcp/epi.scenario_run → MCP bridge JSON-RPC →
Python dispatcher → ``run_scenario`` → ``MetapopSEIRVD.run()`` → JSON
serialize → return.

We skip the Next.js layer (that's pure HTTP routing, adds ~1–5 ms on
localhost, ~30–80 ms on Vercel edge) and talk directly to the server's
tool handler.  The goal is to surface the *cost on top of the pure
numpy SEIR* — JSON encode/decode, compartment flattening, validator
attachment, etc.

Writes ``simulation/results/bench_seir_mcp.json``.
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from pathlib import Path

from simulation.server.mcp_epi import EpiMCPServer


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = ROOT / "simulation" / "results" / "bench_seir_mcp.json"


def time_one(server: EpiMCPServer, args: dict) -> tuple[float, dict]:
    t0 = time.perf_counter_ns()
    result = server._h_scenario_run(args)
    t1 = time.perf_counter_ns()
    # Simulate JSON serialization cost — this is what /api/mcp returns.
    s = json.dumps(result.content, default=str)
    t2 = time.perf_counter_ns()
    return (t1 - t0) / 1e6, {
        "sim_ms":   (t1 - t0) / 1e6,
        "jsonify_ms": (t2 - t1) / 1e6,
        "bytes":    len(s),
        "peak_I":   float(result.content.get("peak_I", 0.0)),
        "peak_day": int(result.content.get("peak_day", -1)),
        "days":     int(result.content.get("days", -1)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--days", type=int, default=200, help="scenario_run default is 200")
    ap.add_argument("--scenario", default="baseline")
    ap.add_argument("--use-db", action="store_true",
                    help="use real commuter + population from DB (slower)")
    args = ap.parse_args()

    call_args = {
        "scenario": args.scenario,
        "days": args.days,
        "seed_district": "Gangnam",
        "seed_infected": 100.0,
        "use_db": args.use_db,
    }

    server = EpiMCPServer()

    log.info(f"Warm-up x {args.warmup}")
    for _ in range(args.warmup):
        time_one(server, call_args)

    log.info(f"Timed runs x {args.n}  scenario={args.scenario!r}  days={args.days}")
    wall = []
    details = []
    for i in range(args.n):
        ms, d = time_one(server, call_args)
        wall.append(ms)
        details.append(d)
        log.info(f"  [{i+1:02d}/{args.n}]  total={ms:.1f} ms  sim={d['sim_ms']:.1f} ms  jsonify={d['jsonify_ms']:.2f} ms  bytes={d['bytes']:,}  peak_I={d['peak_I']:,.0f}")

    wall.sort()
    report = {
        "impl": "python_via_mcp_scenario_run",
        "scenario": {
            "scenario_id": args.scenario,
            "days": args.days,
            "use_db": args.use_db,
            "seed_district": "Gangnam",
            "seed_infected": 100.0,
        },
        "n_runs": args.n,
        "n_warmup": args.warmup,
        "wall_ms": {
            "median": statistics.median(wall),
            "p05":    wall[max(0, int(0.05 * args.n))],
            "p95":    wall[min(args.n - 1, int(0.95 * args.n))],
            "min":    wall[0],
            "max":    wall[-1],
            "all":    wall,
        },
        "jsonify_ms_mean": statistics.mean(d["jsonify_ms"] for d in details),
        "payload_bytes_mean": statistics.mean(d["bytes"] for d in details),
        "sanity": {
            "peak_I_mean":  statistics.mean(d["peak_I"]  for d in details),
            "peak_day_mode": statistics.mode([d["peak_day"] for d in details]),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2))

    med = report["wall_ms"]["median"]
    p05 = report["wall_ms"]["p05"]
    p95 = report["wall_ms"]["p95"]
    print("")
    print(f"=== MCP scenario_run bench  (n={args.n}, warmup={args.warmup}) ===")
    print(f"  total_ms  median : {med:.1f}  ({p05:.1f} - {p95:.1f})")
    print(f"  jsonify   mean   : {report['jsonify_ms_mean']:.2f} ms")
    print(f"  payload   mean   : {report['payload_bytes_mean']:,.0f} bytes")
    print(f"  peak_I    mean   : {report['sanity']['peak_I_mean']:,.0f}")
    print(f"  -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
