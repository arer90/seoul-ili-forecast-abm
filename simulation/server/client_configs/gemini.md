# Gemini / generic HTTP client wiring

Gemini does not yet speak native MCP stdio, so the MPH platform ships a
tiny Node HTTP bridge (`web/scripts/mcp-bridge.ts`) that fronts the Python
stdio server with a JSON-RPC-over-HTTP endpoint.

## Start the bridge

```powershell
# From the repo root
cd <REPO_PATH>\web
npx tsx scripts/mcp-bridge.ts
# → listens on http://127.0.0.1:8787 by default
```

Environment overrides:

| var | default | purpose |
|--------------------|-------------------------------------|--------------------------------|
| `MCP_BRIDGE_HOST` | `127.0.0.1` | bind host |
| `MCP_BRIDGE_PORT` | `8787` | bind port |
| `MCP_PYTHON` | `.venv/Scripts/python.exe` (win32) | python interpreter path |
| `MCP_ARTIFACTS_DIR`| unset | passes to `--artifacts-dir` |

The bridge spawns the Python server on first request and keeps it alive. Kill
with Ctrl-C; the child exits on SIGTERM within the read loop.

## JSON-RPC endpoints

```http
POST /rpc # proxies to the MCP stdio server (initialize / tools/list / tools/call)
POST /report # triggers epi.generate_report and streams the docx/pdf/pptx/xlsx back
```

## Example: call `epi.peak_onset_scatter` from Gemini / any HTTP client

```bash
curl -X POST http://127.0.0.1:8787/rpc \
 -H 'Content-Type: application/json' \
 -d '{
 "jsonrpc": "2.0",
 "id": 1,
 "method": "tools/call",
 "params": {
 "name": "epi.peak_onset_scatter",
 "arguments": {"top_k": 5}
 }
 }'
```

For Gemini agent wiring, register the bridge as an **OpenAPI tool** (not an
MCP tool) — point the Gemini function-call schema at the OpenAPI spec
published at `GET /openapi.json` (see `mcp-bridge.ts` for the live schema).

## Direct stdio invocation (no bridge)

If a client *does* speak MCP stdio natively (Claude Desktop, Cursor, Claude
Code) you don't need the HTTP bridge — use the JSON snippets in
`claude_desktop.json` / `cursor.json` in this folder instead.
