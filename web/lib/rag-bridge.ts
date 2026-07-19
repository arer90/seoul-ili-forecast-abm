/**
 * rag-bridge — runs the Python vector-RAG (simulation.server.rag) as a
 * short-lived subprocess and returns the top-k citation hits.
 *
 * Architecture
 * ------------
 * The Next.js app runs in Node (nodejs runtime — NOT edge) for the
 * chat-cli route.  Edge runtime cannot spawn processes, so this module
 * is intentionally Node-only and must only be imported by routes that
 * declare `export const runtime = "nodejs"`.
 *
 * Python path resolves using MCP_PYTHON (same var that mcp-bridge.ts
 * uses) so local and Docker dev are consistent.
 *
 * Deep-module contract
 * --------------------
 * Single public function: `ragQuery(query, k)` → `RagHit[]` | `null`.
 * Returns null (never throws) when:
 *   - Python / sentence-transformers / lancedb unavailable
 *   - vector index not yet built (falls back to static catalogue inside
 *     the Python layer via `epi.literature_rag` static fallback path —
 *     the stub path still returns structured hits)
 *   - subprocess exits non-zero or times out
 * Callers should inject the results as an optional grounding block and
 * continue normally when null.
 *
 * Performance: ~300-800 ms on first call (model cold-load from HF cache),
 * ~40-80 ms on subsequent calls because Python re-uses its own process
 * cache.  Each invocation is a fresh subprocess — there is no persistent
 * Python worker here (that is mcp-bridge.ts's job).  Acceptable for the
 * ARIA chat path where latency is dominated by the Claude CLI spawn.
 *
 * Side effects: none on the Node side.  Python may write a dense
 * embedding cache under `simulation/results/rag_index/` on first call.
 */
import { spawn } from "node:child_process";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Two levels up from web/lib/ → project root
const PROJECT_ROOT =
  process.env.MCP_CWD ??
  resolve(__dirname, "..", "..");

const PY_CMD =
  process.env.MCP_PYTHON ??
  (process.platform === "win32"
    ? ".venv\\Scripts\\python.exe"
    : ".venv/bin/python");

/** Shape of one citation hit returned by the Python layer. */
export interface RagHit {
  id?: string | null;
  title: string;
  abstract?: string | null;
  year?: number | null;
  tags?: string[];
  doi?: string | null;
  /** Cosine similarity in [0,1] */
  score: number;
  /** "vector_rag" | "static_fallback" | "pubmed_hybrid" */
  source?: string;
}

/** Payload shape from `simulation.server.rag.semantic_search` or `_h_literature_rag`. */
interface PyRagPayload {
  status: string;
  results: Array<{
    id?: string | null;
    title?: string | null;
    abstract?: string | null;
    year?: number | null;
    tags?: string | string[] | null;
    doi?: string | null;
    score?: number | null;
  }>;
}

/**
 * Query the vector RAG index (or its static fallback) for `query`.
 *
 * Args:
 *   query: natural-language question passed to `semantic_search`.
 *   k:     number of results to return (default 4).
 *   timeoutMs: hard subprocess kill timeout (default 8000 ms).
 *
 * Returns:
 *   Array of RagHit objects with title/abstract/score, or null on any
 *   error / unavailability.
 *
 * Raises: never — all errors are swallowed and null is returned.
 *
 * Caller responsibility: treat null as "RAG unavailable, continue
 * without grounding block".
 */
export async function ragQuery(
  query: string,
  k: number = 4,
  timeoutMs: number = 8_000,
): Promise<RagHit[] | null> {
  if (!query.trim()) return null;

  // Inline Python one-liner: import the served RAG layer and serialise
  // the result as JSON to stdout.  Uses the same rag.__init__.semantic_search
  // that mcp_epi._h_literature_rag prefers, with graceful static fallback
  // if lancedb/sentence-transformers are absent.
  const pyCode = `
import json, sys
q = sys.argv[1]
k = int(sys.argv[2])
try:
    from simulation.server.rag import semantic_search, rag_info, build_index
    info = rag_info()
    if not info.get("table_exists") and info.get("lancedb_available") and info.get("embedding_model_available"):
        build_index()
    hits = semantic_search(q, k=k)
    if hits:
        print(json.dumps({"status": "vector_rag", "results": hits}))
        sys.exit(0)
except Exception:
    pass
# static fallback via mcp tool handler
try:
    from simulation.server.mcp_epi import EpiMCPServer
    srv = EpiMCPServer()
    res = srv.call_tool("epi.literature_rag", {"query": q, "k": k})
    import json as _j
    payload = res.content if isinstance(res.content, dict) else _j.loads(res.content)
    print(json.dumps(payload))
except Exception as e:
    print(json.dumps({"status": "error", "results": [], "error": str(e)}))
`.trim();

  return new Promise<RagHit[] | null>((resolve) => {
    const child = spawn(
      PY_CMD,
      ["-c", pyCode, query, String(k)],
      {
        cwd: PROJECT_ROOT,
        stdio: ["ignore", "pipe", "pipe"],
        env: {
          ...process.env,
          PYTHONUNBUFFERED: "1",
          PYTHONIOENCODING: "utf-8",
          PYTHONUTF8: "1",
        },
      },
    );

    let stdout = "";
    let timedOut = false;

    const timer = setTimeout(() => {
      timedOut = true;
      child.kill("SIGKILL");
    }, timeoutMs);

    child.stdout.on("data", (chunk: Buffer) => {
      stdout += chunk.toString("utf-8");
    });

    child.on("close", (code) => {
      clearTimeout(timer);
      if (timedOut) {
        resolve(null);
        return;
      }
      const raw = stdout.trim();
      if (!raw) {
        resolve(null);
        return;
      }
      try {
        // Take the last JSON line (Python may emit warnings before)
        const lastLine = raw.split("\n").filter((l) => l.trim().startsWith("{")).pop();
        if (!lastLine) { resolve(null); return; }
        const payload = JSON.parse(lastLine) as PyRagPayload;
        if (!Array.isArray(payload.results) || payload.results.length === 0) {
          resolve(null);
          return;
        }
        const hits: RagHit[] = payload.results.map((r) => ({
          id: r.id ?? null,
          title: r.title ?? "(no title)",
          abstract: r.abstract ?? null,
          year: r.year != null ? Number(r.year) : null,
          tags: Array.isArray(r.tags)
            ? r.tags
            : typeof r.tags === "string"
            ? r.tags.split(",").map((t) => t.trim()).filter(Boolean)
            : [],
          doi: r.doi ?? null,
          score: typeof r.score === "number" ? r.score : 0,
          source: payload.status,
        }));
        resolve(hits);
      } catch {
        resolve(null);
      }
    });

    child.on("error", () => {
      clearTimeout(timer);
      resolve(null);
    });
  });
}

/**
 * Format a list of RAG hits as a grounding block string for injection
 * into the Claude prompt.
 *
 * Args:
 *   hits: result of ragQuery (null → returns empty string).
 *   source: label for the citation tier (default "벡터 RAG").
 *
 * Returns:
 *   Multi-line string ready to be appended to the system prompt, or ""
 *   when hits is null/empty.
 */
export function formatRagBlock(
  hits: RagHit[] | null,
  source: string = "벡터 RAG",
): string {
  if (!hits || hits.length === 0) return "";
  const lines = hits.map((h, i) => {
    const year = h.year ? ` (${h.year})` : "";
    const doi = h.doi ? ` DOI:${h.doi}` : "";
    const snippet = h.abstract ? `\n     ${h.abstract.slice(0, 200).replace(/\n/g, " ")}` : "";
    return `  [RAG-${i + 1}] ${h.title}${year}${doi} (score=${h.score.toFixed(3)})${snippet}`;
  });
  return (
    `\n\n[GraphRAG 근거 — ${source}] (아래를 1차 문헌 근거로; 인용 시 [RAG-N] 태그 사용):\n` +
    lines.join("\n")
  );
}
