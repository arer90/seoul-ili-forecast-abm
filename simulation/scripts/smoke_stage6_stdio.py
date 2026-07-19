"""Stage 6a — MCP stdio JSON-RPC smoke.

Drives ``simulation.server.mcp_stdio.StdioServer`` in-memory with a
sequence of realistic JSON-RPC 2.0 messages and verifies each response
conforms to the protocol shape.

Run:
    .venv\\Scripts\\python.exe -m simulation.scripts.smoke_stage6_stdio

Output: simulation/results/smoke_stage6_stdio.json  (+ console table)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from simulation.server.mcp_stdio import StdioServer, _ShutdownSignal

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "simulation" / "results" / "smoke_stage6_stdio.json"


def _req(rid: int, method: str, params: dict | None = None) -> str:
    body: dict = {"jsonrpc": "2.0", "id": rid, "method": method}
    if params is not None:
        body["params"] = params
    return json.dumps(body, ensure_ascii=False)


def main() -> int:
    disp = StdioServer.create()
    lines = [
        ("initialize",
         _req(1, "initialize", {
             "clientInfo": {"name": "smoke_stage6_stdio", "version": "0.1"},
             "protocolVersion": "2024-11-05",
         })),
        ("notifications/initialized",
         json.dumps({"jsonrpc": "2.0",
                     "method": "notifications/initialized"})),
        ("ping",
         _req(2, "ping")),
        ("tools/list",
         _req(3, "tools/list")),
        ("tools/call epi.query_db",
         _req(4, "tools/call", {
             "name": "epi.query_db",
             "arguments": {"sql": "SELECT COUNT(*) AS n FROM epi.weekly_disease",
                            "limit": 1},
         })),
        ("tools/call epi.rt_estimate (seoul_city)",
         _req(5, "tools/call", {
             "name": "epi.rt_estimate",
             "arguments": {"gu": "seoul_city", "window_weeks": 7,
                            "lookback_weeks": 52},
         })),
        ("tools/call epi.scenario_run baseline",
         _req(6, "tools/call", {
             "name": "epi.scenario_run",
             "arguments": {"scenario": "baseline", "days": 30,
                            "use_db": False},
         })),
        ("tools/call unknown_tool (should error)",
         _req(7, "tools/call", {"name": "epi.nonexistent", "arguments": {}})),
        ("bad JSON",
         "this is not json"),
        ("shutdown",
         _req(8, "shutdown")),
    ]

    results = []
    shutdown_seen = False
    for label, raw in lines:
        t0 = time.perf_counter()
        try:
            resp = disp.handle_line(raw)
        except _ShutdownSignal:
            shutdown_seen = True
            resp = {"jsonrpc": "2.0", "id": 8, "result": "_shutdown_signal_raised"}
        ms = int((time.perf_counter() - t0) * 1000)

        shape_ok = True
        err_code = None
        preview = ""
        if resp is None:
            preview = "(notification — no response)"
        elif isinstance(resp, dict):
            if resp.get("jsonrpc") != "2.0":
                shape_ok = False
            if "result" in resp:
                r = resp["result"]
                if isinstance(r, dict):
                    if "tools" in r:
                        preview = f"tools={len(r['tools'])}"
                    elif "content" in r:
                        content = r.get("content", [])
                        is_err = r.get("isError")
                        meta = r.get("_meta", {})
                        t_first = content[0].get("text", "") if content else ""
                        t_first = t_first[:80].replace("\n", " ")
                        preview = f"isError={is_err} tool={meta.get('tool','?')} "
                        preview += f"first80={t_first!r}"
                    else:
                        preview = f"result_keys={list(r.keys())[:5]}"
                else:
                    preview = f"result={r!r}"[:80]
            elif "error" in resp:
                err_code = resp["error"]["code"]
                preview = f"code={err_code} msg={resp['error']['message'][:60]}"
            else:
                shape_ok = False
                preview = "(no result/error field)"
        else:
            shape_ok = False
            preview = f"(not a dict: {type(resp).__name__})"

        results.append({
            "label": label,
            "elapsed_ms": ms,
            "shape_ok": shape_ok,
            "err_code": err_code,
            "preview": preview,
            "is_notification": resp is None,
        })

    # Console output — normalize to ASCII so cp949 Windows consoles print it
    def _ascii(s: str) -> str:
        return s.encode("ascii", errors="replace").decode("ascii")

    print(f"\n=== Stage 6a STDIO smoke ({len(results)} messages) ===")
    print(f"{'LABEL':45s}  {'SHAPE':5s}  {'MS':>5s}  PREVIEW")
    print("-" * 120)
    for r in results:
        shape = "ok" if r["shape_ok"] else "BAD"
        print(f"{_ascii(r['label'])[:45]:45s}  {shape:5s}  "
              f"{r['elapsed_ms']:>5d}  {_ascii(r['preview'])}")

    # Contract summary
    ok = sum(1 for r in results if r["shape_ok"])
    print(f"\nShape-conformant responses: {ok}/{len(results)}")
    print(f"Shutdown signal received:   {shutdown_seen}")

    OUT.write_text(
        json.dumps({
            "results": results,
            "shutdown_signal_raised": shutdown_seen,
            "total_messages": len(results),
            "shape_ok_count": ok,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nwrote {OUT}")
    return 0 if ok == len(results) and shutdown_seen else 1


if __name__ == "__main__":
    raise SystemExit(main())
