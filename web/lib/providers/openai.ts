/**
 * OpenAI adapter — chat.completions.stream + tool use.
 *
 * Maps MCP tools → OpenAI `tools` (function variant), runs a local
 * tool-use loop, and streams text deltas.
 */

import OpenAI from "openai";

import { callTool } from "../mcp-client";
import type {
  ChatMessage,
  CompletionRequest,
  ProviderAdapter,
  StreamEvent,
  ToolCall,
} from "./types";

const DEFAULT_MODELS = [
  "gpt-5",
  "gpt-5-mini",
  "gpt-4o",
] as const;

export function createOpenAI(): ProviderAdapter {
  const key = process.env.OPENAI_API_KEY;
  const client = key ? new OpenAI({ apiKey: key }) : null;

  return {
    id: "openai",
    models: () => DEFAULT_MODELS,
    available: () => client !== null,
    async *stream(req: CompletionRequest): AsyncIterable<StreamEvent> {
      if (!client) {
        yield { type: "error", message: "OPENAI_API_KEY not set" };
        return;
      }
      const tools = (req.tools ?? []).map((t) => ({
        type: "function" as const,
        function: {
          name: t.name.replace(/\./g, "_"),
          description: t.description,
          parameters: t.inputSchema as Record<string, unknown>,
        },
      }));

      let hops = 0;
      const maxHops = req.maxToolHops ?? 6;
      const history: ChatMessage[] = [...req.messages];

      while (hops <= maxHops) {
        hops += 1;
        const stream = await client.chat.completions.create({
          model: req.model,
          temperature: req.temperature ?? 0.3,
          tools: tools.length ? tools : undefined,
          stream: true,
          messages: history.map((m) => ({
            role: m.role as "system" | "user" | "assistant",
            content: m.content,
          })),
        });

        const pendingToolUses: ToolCall[] = [];
        const toolAccum: Map<
          number,
          { id: string; name: string; args: string }
        > = new Map();
        let assistantText = "";
        let finishReason: string | null = null;

        for await (const chunk of stream) {
          if (req.signal?.aborted) {
            yield { type: "done", reason: "aborted" };
            return;
          }
          const choice = chunk.choices?.[0];
          if (!choice) continue;
          const delta = choice.delta;
          if (delta?.content) {
            assistantText += delta.content;
            yield { type: "text", delta: delta.content };
          }
          for (const toolDelta of delta?.tool_calls ?? []) {
            const idx = toolDelta.index;
            const entry =
              toolAccum.get(idx) ?? { id: "", name: "", args: "" };
            if (toolDelta.id) entry.id = toolDelta.id;
            if (toolDelta.function?.name) entry.name = toolDelta.function.name;
            if (toolDelta.function?.arguments) {
              entry.args += toolDelta.function.arguments;
            }
            toolAccum.set(idx, entry);
          }
          if (choice.finish_reason) finishReason = choice.finish_reason;
        }

        for (const entry of toolAccum.values()) {
          let parsed: Record<string, unknown> = {};
          try {
            parsed = entry.args ? JSON.parse(entry.args) : {};
          } catch {
            parsed = { _raw: entry.args };
          }
          pendingToolUses.push({
            id: entry.id || crypto.randomUUID(),
            name: entry.name.replace(/_/g, "."),
            arguments: parsed,
          });
        }

        if (pendingToolUses.length === 0) {
          yield { type: "done", reason: finishReason ?? "stop" };
          return;
        }

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
      yield {
        type: "status",
        level: "warn",
        message: `max tool hops (${maxHops}) reached`,
      };
      yield { type: "done", reason: "max_hops" };
    },
  };
}

function safeJSON(v: unknown): string {
  try {
    return JSON.stringify(v);
  } catch {
    return String(v);
  }
}
