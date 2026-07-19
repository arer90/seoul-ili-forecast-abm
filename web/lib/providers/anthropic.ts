/**
 * Anthropic adapter — Claude messages.stream + server-side tool use.
 *
 * We expose a minimal tool-use loop that translates between the MCP
 * tool spec (``lib/providers/types.ts#ToolSpec``) and Anthropic's
 * `tools` param, then streams text deltas as they arrive.
 *
 * The SDK is tree-shakeable enough to run on the Edge runtime.
 */

import Anthropic from "@anthropic-ai/sdk";

import { callTool } from "../mcp-client";
import type {
  ChatMessage,
  CompletionRequest,
  ProviderAdapter,
  StreamEvent,
  ToolCall,
} from "./types";

const DEFAULT_MODELS = [
  "claude-sonnet-4-6",
  "claude-opus-4-6",
  "claude-haiku-4-5-20251001",
] as const;

export function createAnthropic(): ProviderAdapter {
  const key = process.env.ANTHROPIC_API_KEY;
  // Optional Vercel AI Gateway routing — enable when expected concurrency
  // is high enough that automatic failover + semantic cache + observability
  // outweigh the gateway hop (~50-150 ms). Set AI_GATEWAY_URL to e.g.
  //   https://ai-gateway.vercel.sh/v1/anthropic
  // Leaving it empty keeps direct Anthropic SDK calls (lowest latency for
  // single-tenant prototype work). 30+ concurrent users → enable.
  const baseURL = process.env.AI_GATEWAY_URL || undefined;
  const client = key ? new Anthropic({ apiKey: key, baseURL }) : null;
  if (key && baseURL) {
    console.log(`[anthropic] routing through AI Gateway: ${baseURL}`);
  }

  return {
    id: "anthropic",
    models: () => DEFAULT_MODELS,
    available: () => client !== null,
    async *stream(req: CompletionRequest): AsyncIterable<StreamEvent> {
      if (!client) {
        yield { type: "error", message: "ANTHROPIC_API_KEY not set" };
        return;
      }
      const { system, conversation } = splitSystem(req.messages);
      // Anthropic's InputSchema is `{ type: "object", properties?: ...,
      // required?: string[] }`. Our MCP tool spec already emits that
      // shape; cast is only to satisfy the SDK's narrower literal type.
      //
      // Name mapping: Anthropic tool names must match /^[a-zA-Z0-9_-]+$/
      // so dots get replaced with underscores when sent. We CANNOT recover
      // the original by reversing all underscores back — `epi.lead_time_analysis`
      // → `epi_lead_time_analysis` would naively become `epi.lead.time.analysis`,
      // hitting "unknown tool" forever. Maintain an explicit map instead.
      const safeNameToOriginal = new Map<string, string>();
      const tools: Anthropic.Tool[] = (req.tools ?? []).map((t) => {
        const safeName = t.name.replace(/\./g, "_");
        safeNameToOriginal.set(safeName, t.name);
        return {
          name: safeName,
          description: t.description,
          input_schema: {
            type: "object" as const,
            ...(t.inputSchema as Record<string, unknown>),
          } as Anthropic.Tool.InputSchema,
        };
      });

      let hops = 0;
      // 6→3→5 (2026-05-07 evening). 3 hops freezes the UI when Claude
      // does schema-check → query A → query B (3 hops gone, no budget
      // left for the final synthesis text). User saw "✓ 결과 수신,
      // 분석 중..." stuck because the loop exited on max_hops with no
      // text in the final iteration. 5 hops covers the common chain
      // "schema → query 1 → query 2 → query 3 → answer" while staying
      // well under the Vercel Edge 25-s wall.
      const maxHops = req.maxToolHops ?? 5;
      const history: ChatMessage[] = [...conversation];

      // Prompt caching (2026-05-07): the system block is mostly fixed per
      // session (persona + DB schema hint + RULES, ~1.5 KB). Mark it with
      // cache_control:"ephemeral" so Anthropic caches it server-side. First
      // request pays normal cost; every subsequent request in the cache
      // window (~5 min) reads the cached prefix at ~10% input cost AND
      // ~3-5x faster TTFB. Tools list also benefits from caching.
      // Hoisted out of the while loop so the max-hops fallback below can
      // reuse the cached system block (saves another full-prompt input cost).
      const cachedSystem = system
        ? [{ type: "text" as const, text: system, cache_control: { type: "ephemeral" as const } }]
        : undefined;
      const cachedTools = tools.length
        ? tools.map((t, i) =>
            i === tools.length - 1
              ? { ...t, cache_control: { type: "ephemeral" as const } }
              : t,
          )
        : undefined;

      while (hops <= maxHops) {
        hops += 1;
        const stream = client.messages.stream({
          model: req.model,
          // 2048 → 4096 (2026-05-07 morning) — full schema-grounded answer.
          // 4096 → 8192 (2026-05-07 afternoon) — user wanted "deeper",
          // 4-section (역학/임상/모델/시뮬) responses with our project model
          // citation (PAPER_PRIMARY_11 etc.) hit 4k limit too often. Cost
          // doubles only on very long tail; cache_control on system block
          // keeps the input side cheap.
          max_tokens: 8192,
          temperature: req.temperature ?? 0.3,
          system: cachedSystem ?? system,
          tools: cachedTools as Anthropic.Tool[] | undefined,
          messages: history.map((m) => ({
            role: m.role === "assistant" ? "assistant" : "user",
            content: m.content,
          })),
        });

        const pendingToolUses: ToolCall[] = [];
        let assistantText = "";

        for await (const chunk of stream) {
          if (req.signal?.aborted) {
            yield { type: "done", reason: "aborted" };
            return;
          }
          if (chunk.type === "content_block_delta") {
            const delta = chunk.delta;
            if (delta.type === "text_delta") {
              assistantText += delta.text;
              yield { type: "text", delta: delta.text };
            } else if (delta.type === "input_json_delta") {
              // Accumulate tool input; SDK emits it in fragments.
              // The final tool_use block comes via stream.finalMessage.
            }
          }
        }

        const final = await stream.finalMessage();
        for (const block of final.content) {
          if (block.type === "tool_use") {
            pendingToolUses.push({
              id: block.id,
              // Reverse via the explicit map populated above. Falls back to
              // the safe name if Claude returns something we didn't send
              // (shouldn't happen, but graceful).
              name: safeNameToOriginal.get(block.name) ?? block.name,
              arguments: block.input as Record<string, unknown>,
            });
          }
        }

        if (pendingToolUses.length === 0) {
          yield { type: "done", reason: final.stop_reason ?? "stop" };
          return;
        }

        // Echo tool calls then execute them against MCP.
        history.push({ role: "assistant", content: assistantText });
        for (const call of pendingToolUses) {
          yield { type: "tool_call", call };
          const result = await callTool(call.name, call.arguments, {
            signal: req.signal,
          });
          yield { type: "tool_result", result };
          history.push({
            role: "user",
            content: `[tool_result:${call.name}] ${safeJSON(result.output)}`,
          });
        }
      }
      // Max-hops fallback (2026-05-07): if we exit the loop because Claude
      // kept calling tools without producing a final answer, do one more
      // pass WITHOUT tools so Claude is forced to write the synthesis based
      // on whatever data it has gathered. Without this, the UI freezes on
      // "✓ 결과 수신..." because the last hop emitted a tool_result event
      // but no follow-up text. Shorter max_tokens (2048) — we just need
      // a focused conclusion, not another deep dive.
      yield {
        type: "status",
        level: "warn",
        message: `max tool hops (${maxHops}) reached — forcing summary`,
      };
      try {
        const finalStream = client.messages.stream({
          model: req.model,
          max_tokens: 2048,
          temperature: req.temperature ?? 0.3,
          system: cachedSystem ?? system,
          // No tools — forces text-only response.
          messages: [
            ...history.map((m) => ({
              role: (m.role === "assistant" ? "assistant" : "user") as
                | "assistant"
                | "user",
              content: m.content,
            })),
            {
              role: "user" as const,
              content:
                "위 도구 결과들을 바탕으로 사용자 질문에 대한 **최종 답변**을 지금 작성하세요. 추가 도구 호출 X, narrative scaffold X. 메타 라인 + 4 섹션 (역학/임상/모델/시뮬) + 수치 + 출처 인용만.",
            },
          ],
        });
        for await (const chunk of finalStream) {
          if (req.signal?.aborted) break;
          if (
            chunk.type === "content_block_delta" &&
            chunk.delta.type === "text_delta"
          ) {
            yield { type: "text", delta: chunk.delta.text };
          }
        }
      } catch (e) {
        yield {
          type: "status",
          level: "warn",
          message: `summary fallback failed: ${(e as Error).message}`,
        };
      }
      yield { type: "done", reason: "max_hops" };
    },
  };
}

function splitSystem(
  messages: ChatMessage[],
): { system?: string; conversation: ChatMessage[] } {
  const systems = messages.filter((m) => m.role === "system");
  const conv = messages.filter((m) => m.role !== "system");
  return {
    system: systems.map((m) => m.content).join("\n\n") || undefined,
    conversation: conv,
  };
}

function safeJSON(v: unknown): string {
  try {
    return JSON.stringify(v);
  } catch {
    return String(v);
  }
}
