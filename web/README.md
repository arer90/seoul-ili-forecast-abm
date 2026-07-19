# Adaptive Behavior Simulators (ABS) — Web UI

> **Full thesis title**: 감염병 전파 양상에 따른 적응적 행동 반응 기반 다중 에이전트 시뮬레이션 연구 /
> *Multi-Agent Simulation of Adaptive Behavioral Responses to Infectious
> Disease Transmission Patterns.*
>
> **Short brand**: `적응행동 시뮬레이터` / `Adaptive Behavior Simulators` /
> `ABS`. The plural is intentional — the app runs five simulators
> side-by-side (Metapop SEIR-V-D in WASM, the 66-model forecaster, the
> What-if scenario player, the commuter-flow animator, and the LLM
> advisor).

Next.js 14 App Router UI for the metapopulation SEIR-V-D + multi-model
ILI forecasting project. This package is the front end of the
"LLM consultation layer" (internally still codenamed ARIA in file
paths and event names to keep the git history clean) — it talks to the
Python MCP server
(`simulation.server.mcp_stdio`, launched via
`python -m simulation mcp-server`) over a thin HTTP JSON-RPC bridge,
streams provider replies over SSE, and visualises the 25-gu choropleth
+ time track with six live Seoul open-data overlays on top.

## Folder layout

```
web/
├── app/ Next.js App Router
│ ├── api/
│ │ ├── chat/ solo + parallel SSE endpoints
│ │ ├── mcp/[tool]/ proxy to the Python bridge
│ │ └── report/ /report docx/pdf/pptx/xlsx dispatch
│ ├── layout.tsx root layout (Pretendard, dark theme)
│ └── page.tsx mounts <AppShell />
├── components/ React UI (MapPanel, ChatPanel, TimeTrack, …)
├── lib/ Hermes orchestrator, provider adapters,
│ MCP client, Turso, Upstash, validity
├── scripts/
│ ├── mcp-bridge.ts Node HTTP wrapper around python stdio MCP
│ ├── export-turso.py subset of epi_real_seoul.db → libSQL seed
│ └── build-static-aggregates.py pre-bake JSON for Vercel edge
├── public/seoul-gu.geojson stub; replace with real polygons
├── Dockerfile + docker-compose.yml local E2E
└── vercel.json deploy config
```

## Local E2E (docker compose)

```bash
# 1) populate .env at repo root with at least:
# ANTHROPIC_API_KEY=…
# OPENAI_API_KEY=…
# GOOGLE_API_KEY=…
# DEMO_TOKEN=changeme

# 2) start the stack
docker compose -f web/docker-compose.yml up --build

# 3) first time only — pull a small local model into the ollama volume
docker exec -it <ollama container id> ollama pull qwen2.5:14b

# 4) open http://localhost:3000
```

The MCP bridge lives at `http://bridge:8787` inside the compose
network, and `http://localhost:8787` on the host. Both Next.js and the
bridge hot-reload against the live `simulation/` source tree.

## Vercel deploy

ARIA is designed to run on Vercel Edge + Turso + Upstash.

1. **Turso** — create a DB, then seed it:
 ```bash
 .venv/bin/python web/scripts/export-turso.py --out web/scripts/turso_seed.sql
 turso db shell frame-d < web/scripts/turso_seed.sql
 ```
2. **Upstash Redis** — create a REST-enabled DB for rate limiting.
3. **MCP bridge host** — run `web/scripts/mcp-bridge.ts` somewhere that
 can spawn the Python venv and hold a long-lived TCP connection
 (Fly.io, Railway, a VM). Expose HTTPS at `MCP_BRIDGE_URL`.
4. **Static aggregates** — run
 `.venv/bin/python web/scripts/build-static-aggregates.py` before
 `vercel deploy`; it writes `public/seoul-gu.geojson` and
 `public/aggregates/*.json`.
5. **Env** — copy `.env.example` into the Vercel dashboard and set:
 - `NEXT_PUBLIC_HIDE_OLLAMA=1` (Ollama is unreachable from Edge)
 - `MCP_BRIDGE_URL=https://bridge.example.com`
 - `TURSO_URL`, `TURSO_TOKEN`
 (matches `lib/turso.ts` — NOT the `TURSO_AUTH_TOKEN` name shown
 in upstream libSQL docs)
 - `UPSTASH_URL`, `UPSTASH_TOKEN`
 (matches `lib/upstash.ts` — the bundled `@upstash/redis` will
 also read `UPSTASH_REDIS_REST_URL` / `_TOKEN` as a fallback,
 but prefer the short names for consistency)
 - `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`
 (at least one required — each is probed independently via
 `/api/providers`)
 - `DEMO_TOKEN` (a 32+ char random string; sets an httpOnly cookie)
6. `vercel deploy --prod` from the `web/` directory.

## MCP client wiring

For `Claude Desktop` / `Cursor` / Gemini (via HTTP bridge), ready-made
JSON snippets live in `../simulation/server/client_configs/`:

| File | Target |
| -------------------------------- | -------------------------------- |
| `claude_desktop.json` | Claude Desktop (native stdio) |
| `cursor.json` | Cursor IDE (native stdio) |
| `gemini.md` | Gemini / any HTTP client (bridge)|
| `README.md` | troubleshooting + tool inventory |

Smoke-test the server directly:

```powershell
# Dumps the tool schema without blocking — useful in CI
.venv\Scripts\python.exe -m simulation mcp-server --list-tools
```

## Demo rehearsal checklist

Before a live presentation:

1. On presenter laptop, `docker compose up` and confirm:
 - `/api/mcp/_list` returns 10 `epi.*` tools (8 wired in .6a;
 `epi.lead_time_analysis` + `epi.literature_rag` land in .6b)
 - `/api/chat` streams with each of Claude / Gemini / GPT / Ollama
 - `/report docx` downloads a valid Word document
2. Drive a full ≤ 3-minute drill:
 - Map: right-click a gu → "Ask forecast" → chat prefills.
 - Time track: drag the cursor → weekly Rt label updates.
 - Chip `시나리오 What-if` → `/report pdf`.
3. Mobile: open `http://<laptop-ip>:3000` on a phone, confirm:
 - Long-press on a gu opens the context menu.
 - Bottom sheet drags smoothly at 0 / 30 / 50 / 90 %.
4. Kill the bridge container; app should surface
 `mcp bridge unavailable` as a warning but still render.

## Keyboard shortcuts

| Key | Action |
| -------------------- | -------------------------------------- |
| `Enter` | send composer |
| `Shift + Enter` | newline in composer |
| `Esc` | abort an in-flight reply |

## Env reference

See `.env.example`. Secrets are never sent to the browser except
`NEXT_PUBLIC_HIDE_OLLAMA` (intentional).
