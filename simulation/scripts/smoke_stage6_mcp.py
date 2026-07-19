"""Stage 6a MCP server smoke — calls all 10 tools and logs a one-liner each.

Run:
    .venv\\Scripts\\python.exe -m simulation.scripts.smoke_stage6_mcp

Output: simulation/results/smoke_stage6_mcp.json  (+ console table)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from simulation.server.mcp_epi import EpiMCPServer

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "simulation" / "results" / "smoke_stage6_mcp.json"


def main() -> int:
    srv = EpiMCPServer()
    tools = srv.list_tools()
    wired_map = {t["name"]: bool(t.get("_meta", {}).get("wired")) for t in tools}

    # Smoke call plan — one lightweight invocation per tool. Weekly-incidence
    # tools get two calls each: seoul_city (city-aggregate — supported) and
    # a specific gu (documents the graceful-empty contract for gu-panel).
    calls = [
        ("epi.query_db", {"sql": "SELECT COUNT(*) AS n FROM epi.weekly_disease",
                           "limit": 1}),
        ("epi.rt_estimate", {"gu": "seoul_city", "window_weeks": 7,
                              "lookback_weeks": 52}),
        ("epi.outbreak_detect", {"gu": "seoul_city", "method": "EARS-C1",
                                  "lookback_weeks": 52, "z_threshold": 2.0}),
        ("epi.validity_check", {"predictions": [5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
                                 "params": {"R0": 1.4, "gamma": 0.285,
                                            "sigma": 0.5, "VE": 0.5, "ifr": 0.001}}),
        ("epi.scenario_run", {"scenario": "baseline", "days": 30, "use_db": False}),
        ("epi.forecast", {"gu": "seoul_city"}),
        ("epi.model_compare", {"models": ["NegBinGLM", "Ensemble-BMA"]}),
        ("epi.shap_features", {"top_n": 5}),
        ("epi.lead_time_analysis", {"model": "NegBinGLM"}),
        ("epi.literature_rag", {"query": "KDCA threshold"}),
    ]

    results = []
    for name, args in calls:
        t0 = time.perf_counter()
        r = srv.call_tool(name, args)
        ms = int((time.perf_counter() - t0) * 1000)
        c = r.content
        if isinstance(c, dict):
            status = c.get("status") or ("error" if "error" in c else "")
            preview = {k: c.get(k) for k in (
                "row_count", "n_points", "n_flagged", "n_weeks",
                "peak_I", "peak_day", "top_k", "in_mcs", "n_models",
                "message",
            ) if k in c}
        else:
            status = "text"
            preview = {"text": str(c)[:80]}
        results.append({
            "tool": name,
            "wired_declared": wired_map.get(name),
            "is_error": bool(r.is_error),
            "elapsed_ms": ms,
            "status": status,
            "preview": preview,
        })

    # Console summary
    print(f"\n=== Stage 6a MCP smoke ({len(results)} tools) ===")
    print(f"{'TOOL':32s}  {'WIRE':4s}  {'ERR':4s}  {'MS':>6s}  STATUS  PREVIEW")
    print("-" * 100)
    for r in results:
        wire = "[W]" if r["wired_declared"] else "[ ]"
        err = "ERR" if r["is_error"] else "ok"
        preview = ", ".join(f"{k}={v}" for k, v in r["preview"].items())[:60]
        print(f"{r['tool']:32s}  {wire:4s}  {err:4s}  {r['elapsed_ms']:>6d}  "
              f"{r['status']:<16s} {preview}")

    # Decide exit code — only real "is_error" fails the smoke
    n_err = sum(1 for r in results if r["is_error"])
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_stub = sum(1 for r in results if r["status"] == "not_available")
    n_insuf = sum(1 for r in results if r["status"] == "insufficient_data")
    print(f"\nSummary: {n_ok} ok / {n_insuf} insufficient_data / "
          f"{n_stub} not_available / {n_err} error")

    OUT.write_text(
        json.dumps({
            "results": results,
            "totals": {
                "n_tools": len(results),
                "n_ok": n_ok,
                "n_insufficient_data": n_insuf,
                "n_not_available": n_stub,
                "n_error": n_err,
            },
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nwrote {OUT}")

    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
