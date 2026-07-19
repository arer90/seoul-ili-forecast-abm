/**
 * Google adapter — Gemini streamGenerateContent + function calling.
 *
 * We use the official `@google/generative-ai` SDK. MCP tools are
 * re-shaped into `functionDeclarations`. The response comes back as a
 * chunk stream of `GenerateContentResponse`, each containing either
 * text parts or a `functionCall` part.
 */

import {
  type FunctionDeclaration,
  type FunctionDeclarationSchema,
  GoogleGenerativeAI,
  SchemaType,
} from "@google/generative-ai";

import { callTool } from "../mcp-client";
import type {
  ChatMessage,
  CompletionRequest,
  ProviderAdapter,
  StreamEvent,
  ToolCall,
} from "./types";

const DEFAULT_MODELS = [
  "gemini-2.5-pro",
  "gemini-2.5-flash",
  "gemini-2.0-pro",
] as const;

export function createGoogle(): ProviderAdapter {
  const key = process.env.GOOGLE_API_KEY;
  const client = key ? new GoogleGenerativeAI(key) : null;

  return {
    id: "google",
    models: () => DEFAULT_MODELS,
    available: () => client !== null,
    async *stream(req: CompletionRequest): AsyncIterable<StreamEvent> {
      if (!client) {
        yield { type: "error", message: "GOOGLE_API_KEY not set" };
        return;
      }

      const { system, conversation } = splitSystem(req.messages);
      // `toGeminiSchema` already populates `type: SchemaType.OBJECT` +
      // `properties` at runtime (see L145-L152), but its loose
      // `Record<string, unknown>` return type erases that — cast back to
      // FunctionDeclarationSchema for the SDK.
      const functionDeclarations: FunctionDeclaration[] = (req.tools ?? []).map((t) => ({
        name: t.name.replace(/\./g, "_"),
        description: t.description,
        parameters: toGeminiSchema(t.inputSchema) as unknown as FunctionDeclarationSchema,
      }));

      const model = client.getGenerativeModel({
        model: req.model,
        systemInstruction: system,
        tools: functionDeclarations.length
          ? [{ functionDeclarations }]
          : undefined,
      });

      let hops = 0;
      const maxHops = req.maxToolHops ?? 6;
      const history: ChatMessage[] = [...conversation];

      while (hops <= maxHops) {
        hops += 1;
        const chat = model.startChat({
          history: history.slice(0, -1).map((m) => ({
            role: m.role === "assistant" ? "model" : "user",
            parts: [{ text: m.content }],
          })),
          generationConfig: {
            temperature: req.temperature ?? 0.3,
          },
        });
        const lastUser = history[history.length - 1];
        const result = await chat.sendMessageStream(lastUser?.content ?? "");

        const pendingToolUses: ToolCall[] = [];
        let assistantText = "";

        for await (const chunk of result.stream) {
          if (req.signal?.aborted) {
            yield { type: "done", reason: "aborted" };
            return;
          }
          const text = chunk.text();
          if (text) {
            assistantText += text;
            yield { type: "text", delta: text };
          }
          const fnCalls = chunk.functionCalls();
          if (fnCalls) {
            for (const fc of fnCalls) {
              pendingToolUses.push({
                id: crypto.randomUUID(),
                name: fc.name.replace(/_/g, "."),
                arguments: (fc.args ?? {}) as Record<string, unknown>,
              });
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

/**
 * Convert a JSON Schema-like object into the Gemini SchemaType shape.
 * This is intentionally minimal — Gemini rejects `additionalProperties`
 * and some schema keywords that OpenAI/Anthropic happily accept.
 */
function toGeminiSchema(src: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  if (src.type === "object") {
    out.type = SchemaType.OBJECT;
    const props = (src.properties as Record<string, Record<string, unknown>>) ?? {};
    const cleanProps: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(props)) {
      cleanProps[k] = toGeminiSchema(v);
    }
    out.properties = cleanProps;
    if (src.required) out.required = src.required;
  } else if (src.type === "array") {
    out.type = SchemaType.ARRAY;
    out.items = toGeminiSchema(
      (src.items as Record<string, unknown>) ?? { type: "string" },
    );
  } else if (src.type === "string") {
    out.type = SchemaType.STRING;
    if (src.enum) out.enum = src.enum;
    if (src.description) out.description = src.description;
  } else if (src.type === "number" || src.type === "integer") {
    out.type = src.type === "integer" ? SchemaType.INTEGER : SchemaType.NUMBER;
    if (src.description) out.description = src.description;
  } else if (src.type === "boolean") {
    out.type = SchemaType.BOOLEAN;
  }
  return out;
}

function safeJSON(v: unknown): string {
  try {
    return JSON.stringify(v);
  } catch {
    return String(v);
  }
}
