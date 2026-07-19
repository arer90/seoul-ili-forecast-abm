/**
 * MCP client — talks to the Python ``simulation.server`` stdio server
 * across an ndjson HTTP bridge (``scripts/mcp-bridge.ts``).
 *
 * Edge runtime compatible: uses ``fetch`` + streaming readers only, no
 * node net APIs. The bridge shape is:
 *
 *     POST /rpc   body = JSON-RPC 2.0 request   → body = JSON-RPC 2.0 response
 *     GET  /tools/list  (convenience cache endpoint)
 *
 * The bridge is in-process during local dev (Docker compose) and an
 * internal Vercel edge route in production.
 */

import type { ToolResult, ToolSpec } from "./providers/types";
import { withMcpCache } from "./mcp-cache";

const BRIDGE_URL =
  process.env.MCP_BRIDGE_URL ?? "http://localhost:8787";

export interface McpCallOptions {
  signal?: AbortSignal;
  /**
   * When set, the bridge will add this id to its audit trail so we can
   * correlate LLM turn → tool call → DB query.
   */
  requestId?: string;
  /**
   * Sprint 2026-05-06 quick-win #2: bypass the read-only tool cache
   * (Upstash KV) and always run a fresh MCP invocation. Default false —
   * read-only tools (epi.forecast, epi.rt_estimate, etc.) are cached
   * with TTL per `READ_ONLY_TOOL_TTL` in `mcp-cache.ts`.
   */
  noCache?: boolean;
}

async function rpc<T>(
  method: string,
  params: unknown,
  options: McpCallOptions = {},
): Promise<T> {
  const body = JSON.stringify({
    jsonrpc: "2.0",
    id: options.requestId ?? crypto.randomUUID(),
    method,
    params: params ?? {},
  });
  const res = await fetch(`${BRIDGE_URL}/rpc`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body,
    signal: options.signal,
    // Never cache — every call is a live tool invocation.
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`MCP bridge ${res.status}: ${await res.text()}`);
  }
  const envelope = (await res.json()) as {
    result?: T;
    error?: { code: number; message: string; data?: unknown };
  };
  if (envelope.error) {
    const err = new Error(
      `${envelope.error.code}: ${envelope.error.message}`,
    );
    (err as Error & { data?: unknown }).data = envelope.error.data;
    throw err;
  }
  return envelope.result as T;
}

export async function listTools(
  options: McpCallOptions = {},
): Promise<ToolSpec[]> {
  const { tools } = await rpc<{ tools: Array<ToolSpec & { _meta?: { wired?: boolean } }> }>(
    "tools/list",
    {},
    options,
  );
  // Flatten the `_meta.wired` attribute into the ToolSpec surface.
  return tools.map((t) => ({
    name: t.name,
    title: t.title,
    description: t.description,
    inputSchema: t.inputSchema,
    wired: t._meta?.wired ?? true,
  }));
}

export async function callTool(
  name: string,
  args: Record<string, unknown>,
  options: McpCallOptions = {},
): Promise<ToolResult> {
  // Sprint 2026-05-06 quick-win #2 (Codex optimization #2): wrap
  // read-only tools (epi.forecast / epi.rt_estimate / epi.shap_features
  // / epi.lead_time_analysis / epi.outbreak_detect / epi.literature_rag)
  // with Upstash KV TTL cache. Write/dynamic tools bypass cache.
  return withMcpCache(
    name,
    args,
    async () => {
      const raw = await rpc<{
        content: Array<{ type: string; text?: string }>;
        isError?: boolean;
        _meta?: Record<string, unknown>;
      }>("tools/call", { name, arguments: args }, options);

      const first = raw.content[0];
      let output: unknown = null;
      if (first && first.type === "text" && typeof first.text === "string") {
        try {
          output = JSON.parse(first.text);
        } catch {
          output = first.text;
        }
      }
      return {
        toolCallId: options.requestId ?? crypto.randomUUID(),
        output,
        isError: Boolean(raw.isError),
      };
    },
    { force: options.noCache },
  );
}
