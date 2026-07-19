/**
 * MCP read-only tool cache layer (sprint 2026-05-06 quick-win #2).
 *
 * Wraps ``callTool`` invocations of read-only MCP tools (epi.forecast,
 * epi.rt_estimate, etc.) with a TTL-based Upstash KV cache. Write or
 * dynamic tools (epi.scenario_run, epi.query_db, epi.model_compare,
 * epi.validity_check) bypass cache and always run live.
 *
 * Codex optimization review (2026-05-06) #2 — `cache:"no-store"` in
 * `mcp-client.ts:45` meant every chat turn re-ran the Python+SQLite
 * pipeline. Hot tools (e.g. multiple users asking "what's 강남구
 * forecast for next week") were paying full latency every time. With
 * Upstash KV TTL caching, repeated identical requests are served from
 * Edge KV instead of crossing the MCP bridge.
 *
 * Graceful degradation: if `UPSTASH_URL` / `UPSTASH_TOKEN` are missing
 * (`web/lib/upstash.ts:redis()` returns null), `cacheGetOrSet` falls
 * through and runs `build()` directly. So local dev works without
 * Upstash.
 */
import { cacheGetOrSet } from "./upstash";

/**
 * Read-only MCP tools and their cache TTL (seconds). Tools NOT in this
 * map bypass the cache (e.g. epi.scenario_run, epi.query_db,
 * epi.model_compare, epi.validity_check — these have variable args or
 * write semantics).
 */
export const READ_ONLY_TOOL_TTL: Record<string, number> = {
  // 1-minute TTL — fresh enough that a chat turn after the next weekly
  // sentinel push will see updated values, while back-to-back chat
  // questions reuse the result.
  "epi.forecast": 60,
  "epi.outbreak_detect": 60,
  // 5-minute TTL — Rt and SHAP recompute is expensive but the underlying
  // cohort moves slowly. Edge ISR for `web/app/api/overlays/live/route.ts`
  // already uses 300s; we mirror that here for MCP-layer hot reads.
  "epi.rt_estimate": 300,
  "epi.shap_features": 300,
  "epi.lead_time_analysis": 300,
  // 10-minute TTL — RAG corpus changes only when new PDFs land. The
  // research-grade default; bump down if you wire a "literature update"
  // event hook later.
  "epi.literature_rag": 600,
};

/** Stable JSON stringify (deterministic key order) for cache key hashing. */
function stableStringify(value: unknown): string {
  if (value === null || value === undefined) return "null";
  if (typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) {
    return "[" + value.map(stableStringify).join(",") + "]";
  }
  const obj = value as Record<string, unknown>;
  const keys = Object.keys(obj).sort();
  return (
    "{" +
    keys.map((k) => JSON.stringify(k) + ":" + stableStringify(obj[k])).join(",") +
    "}"
  );
}

/**
 * Cheap (non-cryptographic) hash for cache keys. djb2 — sufficient
 * because cache keys are namespaced by tool name and only conflict
 * locally; collisions degrade to a stale read at worst, not a security
 * issue.
 */
function djb2(s: string): string {
  let h = 5381;
  for (let i = 0; i < s.length; i++) {
    h = ((h * 33) ^ s.charCodeAt(i)) >>> 0;
  }
  return h.toString(36);
}

/** Compose the cache key from tool name and stable args hash. */
export function cacheKey(toolName: string, args: unknown): string {
  return `mcp:${toolName}:${djb2(stableStringify(args))}`;
}

/**
 * Wrap a build() that performs a single MCP `callTool` invocation. If
 * the tool is in `READ_ONLY_TOOL_TTL`, route through Upstash KV with
 * the configured TTL; otherwise call build() directly.
 */
export async function withMcpCache<T>(
  toolName: string,
  args: unknown,
  build: () => Promise<T>,
  options: { force?: boolean } = {},
): Promise<T> {
  if (options.force) return build();
  const ttl = READ_ONLY_TOOL_TTL[toolName];
  if (ttl === undefined) return build();
  return cacheGetOrSet(cacheKey(toolName, args), ttl, build);
}
