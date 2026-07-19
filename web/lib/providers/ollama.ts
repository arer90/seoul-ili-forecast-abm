/**
 * Ollama adapter — local-only provider. Hidden in the UI when
 * NEXT_PUBLIC_HIDE_OLLAMA=1 (prod default).
 *
 * Ollama's tool-use surface is model-dependent; Qwen 2.5 and
 * Llama 3.1 Instruct both support OpenAI-compatible tool calls on
 * recent builds. We use the ``/api/chat`` endpoint with a streaming
 * body of ndjson chunks.
 */

import { callTool } from "../mcp-client";
import type {
  ChatMessage,
  CompletionRequest,
  ProviderAdapter,
  StreamEvent,
  ToolCall,
} from "./types";

// Fallback list used only when the daemon is unreachable or /api/tags
// returns junk. Real installs vary per machine — the live list comes
// from listOllamaModels() below, which /api/providers prefers.
const DEFAULT_MODELS = [
  "exaone3.5:7.8b",
  "qwen2.5:7b",
  "qwen2.5:1.5b",
] as const;

/**
 * Live query of /api/tags so the model dropdown matches what the user
 * actually has pulled. Returns ``null`` on any failure — the caller
 * should fall back to ``DEFAULT_MODELS``.
 *
 * Exported separately (rather than turning ``models()`` async) so the
 * synchronous ``ProviderAdapter`` contract stays intact; only
 * ``/api/providers`` needs the live list.
 */
export async function listOllamaModels(
  base?: string,
  timeoutMs = 500,
): Promise<string[] | null> {
  const url = base ?? process.env.OLLAMA_BASE_URL ?? "http://localhost:11434";
  try {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), timeoutMs);
    const r = await fetch(`${url}/api/tags`, { signal: ctl.signal });
    clearTimeout(t);
    if (!r.ok) return null;
    const body = (await r.json()) as {
      models?: Array<{ name?: string; model?: string }>;
    };
    const names = (body.models ?? [])
      .map((m) => m.name ?? m.model ?? "")
      .filter((s): s is string => s.length > 0);
    if (names.length === 0) return null;
    // Heuristic ranking: put Korean-capable names first, then by name
    // length (shorter tags are usually the "canonical" ones). The
    // caller controls the final order but this biases the picker
    // toward useful defaults.
    names.sort((a, b) => {
      const ka = a.toLowerCase().includes("exaone") ? -1 : 0;
      const kb = b.toLowerCase().includes("exaone") ? -1 : 0;
      if (ka !== kb) return ka - kb;
      return a.localeCompare(b);
    });
    return names;
  } catch {
    return null;
  }
}

export function createOllama(): ProviderAdapter {
  const base = process.env.OLLAMA_BASE_URL ?? "http://localhost:11434";
  const hideInUi = process.env.NEXT_PUBLIC_HIDE_OLLAMA === "1";

  return {
    id: "ollama",
    models: () => DEFAULT_MODELS,
    // Without an API key there's nothing to check synchronously, but we
    // do hide the chip when NEXT_PUBLIC_HIDE_OLLAMA=1 so the Vercel
    // prod deploy doesn't show a button that can't work.
    available: () => !hideInUi,
    async *stream(req: CompletionRequest): AsyncIterable<StreamEvent> {
      if (hideInUi) {
        yield {
          type: "error",
          message:
            "Ollama disabled in production. Run locally with NEXT_PUBLIC_HIDE_OLLAMA=0.",
        };
        return;
      }
      const tools = (req.tools ?? []).map((t) => ({
        type: "function" as const,
        function: {
          name: t.name.replace(/\./g, "_"),
          description: t.description,
          parameters: t.inputSchema,
        },
      }));

      let hops = 0;
      const maxHops = req.maxToolHops ?? 6;
      const history: ChatMessage[] = [...req.messages];
      // Some Ollama models (exaone3.5, phi3, gemma, …) don't implement
      // the function-calling surface and return HTTP 400 with
      // "does not support tools" when we attach a ``tools`` array.
      // When that happens we flip ``toolsDisabled=true`` for the rest
      // of this request, surface a one-shot status warning to the
      // user, and re-issue the call without the tools field. This
      // keeps the UX working — the model just can't use MCP tools.
      let toolsDisabled = false;

      while (hops <= maxHops) {
        hops += 1;
        const sendTools = !toolsDisabled && tools.length > 0;
        const res = await fetch(`${base}/api/chat`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            model: req.model,
            stream: true,
            options: { temperature: req.temperature ?? 0.3 },
            tools: sendTools ? tools : undefined,
            messages: history.map((m) => ({
              role: m.role,
              content: m.content,
            })),
          }),
          signal: req.signal,
        });
        if (!res.ok || !res.body) {
          const errText = await res.text();
          // Detect the tools-unsupported case and retry once without
          // tools. Match either the exact Ollama error string or a
          // looser substring for forward-compat.
          if (
            res.status === 400 &&
            sendTools &&
            /does not support tools/i.test(errText)
          ) {
            toolsDisabled = true;
            hops -= 1; // don't count the failed probe toward maxHops
            yield {
              type: "status",
              level: "warn",
              message: `${req.model} does not support tools — retrying without MCP tool access.`,
            };
            continue;
          }
          yield {
            type: "error",
            message: `ollama error ${res.status}: ${errText}`,
          };
          return;
        }

        const pendingToolUses: ToolCall[] = [];
        let assistantText = "";
        let done = false;

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buf = "";

        while (!done) {
          if (req.signal?.aborted) {
            yield { type: "done", reason: "aborted" };
            return;
          }
          const { value, done: streamDone } = await reader.read();
          if (streamDone) break;
          buf += decoder.decode(value, { stream: true });
          const lines = buf.split("\n");
          buf = lines.pop() ?? "";
          for (const line of lines) {
            if (!line.trim()) continue;
            let chunk: {
              message?: {
                content?: string;
                tool_calls?: Array<{
                  function?: { name: string; arguments: unknown };
                }>;
              };
              done?: boolean;
            };
            try {
              chunk = JSON.parse(line);
            } catch {
              continue;
            }
            const delta = chunk.message?.content;
            if (delta) {
              assistantText += delta;
              yield { type: "text", delta };
            }
            for (const tc of chunk.message?.tool_calls ?? []) {
              if (!tc.function) continue;
              pendingToolUses.push({
                id: crypto.randomUUID(),
                name: tc.function.name.replace(/_/g, "."),
                arguments:
                  typeof tc.function.arguments === "string"
                    ? safeParse(tc.function.arguments)
                    : ((tc.function.arguments as Record<string, unknown>) ??
                        {}),
              });
            }
            if (chunk.done) {
              done = true;
              break;
            }
          }
        }

        if (pendingToolUses.length === 0) {
          yield { type: "done", reason: "stop" };
          return;
        }

        history.push({ role: "assistant", content: assistantText });
        for (const call of pendingToolUses) {
          yield { type: "tool_call", call };
          const toolRes = await callTool(call.name, call.arguments, {
            signal: req.signal,
          });
          yield { type: "tool_result", result: toolRes };
          history.push({
            role: "user",
            content: `[tool_result:${call.name}] ${safeJSON(toolRes.output)}`,
          });
        }
      }
      yield {
        type: "status",
        level: "warn",
        message: `max tool hops (${maxHops}) reached`,
      };
      yield { type: "done", reason: "max_hops" };
    },
  };
}

function safeParse(s: string): Record<string, unknown> {
  try {
    return JSON.parse(s);
  } catch {
    return { _raw: s };
  }
}

function safeJSON(v: unknown): string {
  try {
    return JSON.stringify(v);
  } catch {
    return String(v);
  }
}
