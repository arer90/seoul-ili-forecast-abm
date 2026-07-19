/**
 * POST /api/history/sessions/:id/messages — append one message
 *
 * Called by ChatPanel on both the user turn and each completed
 * assistant turn (parallel mode → multiple POSTs sharing the same
 * ``turn_idx``, one per provider).
 *
 * Identity: ``fd_uid`` from middleware. Ownership of the session is
 * enforced inside ``appendMessage``.
 *
 * Body
 *     {
 *       role: "user" | "assistant" | "tool",
 *       content: string,
 *       provider_id?: string,
 *       tool_calls?: unknown,   // JSON-serialisable, stringified server-side
 *       validity?: unknown,
 *       tokens_in?: number,
 *       tokens_out?: number,
 *       turn_idx?: number       // optional override for parallel siblings
 *     }
 *
 * Response
 *     { message: FdMessage }
 *
 * Also side-effects
 *   - bumps ``fd_sessions.updated_at`` (atomic with the INSERT)
 *   - sets ``fd_sessions.provider_id`` when the message has one
 *   - trims the oldest 100 messages when the per-session cap (1000)
 *     is exceeded
 */
import type { NextRequest } from "next/server";

import { fdUidOf, requireAuth, requireFdUid } from "@/lib/auth";
import {
  appendMessage,
  maybeAutoTitle,
  type AppendMessageInput,
  type FdMessageRole,
} from "@/lib/history-db";

export const runtime = "edge";

interface RouteCtx {
  params: { id: string };
}

function coerceRole(v: unknown): FdMessageRole | null {
  return v === "user" || v === "assistant" || v === "tool" ? v : null;
}

export async function POST(req: NextRequest, ctx: RouteCtx): Promise<Response> {
  const authFail = requireAuth(req);
  if (authFail) return authFail;
  const uidFail = requireFdUid(req);
  if (uidFail) return uidFail;
  const uid = fdUidOf(req) as string;
  const { id: sessionId } = ctx.params;
  if (!sessionId)
    return Response.json({ error: "missing session id" }, { status: 400 });

  let raw: Partial<AppendMessageInput> & { role?: unknown };
  try {
    raw = (await req.json()) as Partial<AppendMessageInput> & { role?: unknown };
  } catch {
    return Response.json({ error: "invalid JSON" }, { status: 400 });
  }

  const role = coerceRole(raw.role);
  if (!role) return Response.json({ error: "invalid role" }, { status: 400 });
  if (typeof raw.content !== "string")
    return Response.json({ error: "content must be string" }, { status: 400 });

  const input: AppendMessageInput = {
    role,
    content: raw.content,
    provider_id:
      typeof raw.provider_id === "string" ? raw.provider_id : undefined,
    tool_calls: raw.tool_calls,
    validity: raw.validity,
    tokens_in: typeof raw.tokens_in === "number" ? raw.tokens_in : undefined,
    tokens_out: typeof raw.tokens_out === "number" ? raw.tokens_out : undefined,
    turn_idx: typeof raw.turn_idx === "number" ? raw.turn_idx : undefined,
  };

  try {
    const message = await appendMessage(sessionId, uid, input);
    // Fire-and-forget: auto-title on the first user message. We await
    // it because edge runtime cancels pending work after the response
    // returns, so a truly detached promise would be killed. Cost is
    // 1-2 extra round-trips only on the first turn.
    let updatedSession = null;
    if (role === "user") {
      updatedSession = await maybeAutoTitle(sessionId, uid, input.content);
    }
    return Response.json(
      { message, session: updatedSession ?? undefined },
      { status: 201 },
    );
  } catch (e) {
    const msg = e instanceof Error ? e.message : "appendMessage failed";
    const status =
      msg.includes("not found") || msg.includes("not owned") ? 404 : 500;
    return Response.json({ error: msg }, { status });
  }
}
