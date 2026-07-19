/**
 * Provider-agnostic types for the Hermes gateway.
 *
 * Every provider adapter maps to this common shape so the orchestrator
 * in ``lib/hermes.ts`` can fan-out, relay, or synthesise without
 * caring about vendor-specific SDK shapes.
 */

export type ProviderId = "anthropic" | "google" | "openai" | "ollama";

export type ResponseMode = "solo" | "parallel" | "synthesis" | "relay";

export interface ChatMessage {
  /**
   * Normalised role. When relaying a reply across providers, we
   * prefix the visible text with `[Previous from X]` so the next
   * provider doesn't mistake it for its own history. See
   * ``lib/hermes.ts#injectRelayPrefix``.
   */
  role: "system" | "user" | "assistant";
  content: string;
  /** Optional tag we use for forensic logging only. */
  origin?: ProviderId;
}

export interface ToolCall {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
}

export interface ToolResult {
  toolCallId: string;
  /** JSON-serialisable output from the MCP tool. */
  output: unknown;
  /** When true, upstream reported an error. Surface to user. */
  isError?: boolean;
}

export interface ToolSpec {
  name: string;
  title?: string;
  description: string;
  inputSchema: Record<string, unknown>;
  /**
   * When the schema flag says the tool is not fully wired, providers
   * are still allowed to call it — they'll just get a graceful
   * `status="not_available"` payload back. We expose the flag on the
   * UI so users know what to expect.
   */
  wired: boolean;
}

export interface CompletionRequest {
  model: string;
  messages: ChatMessage[];
  temperature?: number;
  /**
   * Upstream tools. Every provider adapter translates this list into
   * its vendor format (OpenAI `tools`, Anthropic `tools`, Gemini
   * `functionDeclarations`).
   */
  tools?: ToolSpec[];
  /** Hard cap on tool-call loop depth; default 6. */
  maxToolHops?: number;
  /** Abort passed from the route handler. */
  signal?: AbortSignal;
}

export type StreamEvent =
  | { type: "text"; delta: string }
  | { type: "tool_call"; call: ToolCall }
  | { type: "tool_result"; result: ToolResult }
  | { type: "status"; message: string; level?: "info" | "warn" | "error" }
  | { type: "done"; reason?: string }
  | { type: "error"; message: string };

export interface ProviderAdapter {
  id: ProviderId;
  /**
   * Available model ids this adapter exposes to the picker. The first
   * entry is considered the default.
   */
  models(): readonly string[];
  /**
   * Whether this adapter should be shown in the UI — e.g. Ollama is
   * hidden in production via NEXT_PUBLIC_HIDE_OLLAMA.
   */
  available(): boolean;
  /**
   * Stream a completion as a series of ``StreamEvent``s. The adapter
   * is responsible for running its own tool-call loop (up to
   * ``maxToolHops``) by delegating each call to the shared MCP client.
   */
  stream(req: CompletionRequest): AsyncIterable<StreamEvent>;
}
