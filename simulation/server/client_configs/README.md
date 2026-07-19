# MCP client configs — MPH epi-mcp

Drop-in configuration for hooking MCP-compatible LLM clients to the
Python `simulation.server.mcp_stdio` JSON-RPC server (Stage 6a).

| File | Target client | Transport |
|----------------------------|-----------------------------------|-----------------|
| `claude_desktop.json` | Claude Desktop (Anthropic) | stdio (native) |
| `cursor.json` | Cursor IDE | stdio (native) |
| `gemini.md` | Gemini / any HTTP LLM | HTTP bridge |

## Smoke-test the server

```powershell
# Dumps the 10-tool schema to stdout (does NOT block on stdin)
.venv\Scripts\python.exe -m simulation mcp-server --list-tools

# Run the stdio loop (blocks; Ctrl-C to exit)
.venv\Scripts\python.exe -m simulation mcp-server
```

A correctly-wired client should see 10 tools after `tools/list`:

1. `epi.weekly_incidence`
2. `epi.rt_snapshot`
3. `epi.shap_delta`
4. `epi.scenario_runs`
5. `epi.forecast_ensemble`
6. `epi.pi_diagnostics`
7. `epi.peak_onset_scatter`
8. `epi.regime_split_mape`
9. `epi.lead_time_analysis` ← Track B-1, wiring in progress
10. `epi.literature_rag` ← Track B-2, static-citation fallback

See `../mcp_epi.py` for the full tool schemas.

## Troubleshooting

| Symptom | Likely cause |
|------------------------------------------------------|--------------------------------------------------------------|
| Client reports "server did not respond" | Python path wrong — test with `--list-tools` from a shell |
| UnicodeDecodeError on Windows | Missing `PYTHONIOENCODING=utf-8` in `env` |
| "no such command: mcp-server" | `cwd` not set to repo root; `-m simulation` needs repo cwd |
| Tools list returns 8 not 10 | B-1 / B-2 not wired yet — check `docs/internal/stage_plan.md` |
