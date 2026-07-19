/**
 * GET    /api/history/sessions/:id — fetch a session + all its messages
 * PATCH  /api/history/sessions/:id — update title / category / pin / archive
 * DELETE /api/history/sessions/:id — hard delete (cascades to messages)
 *
 * Identity: ``fd_uid`` from middleware. Ownership is enforced inside
 * ``history-db.ts`` — a mismatched uid looks like a 404 to the caller.
 *
 * Why hard-delete and not soft?
 *   Soft delete already lives at the ``archived`` flag. DELETE is the
 *   escape hatch for "I really don't want this history retained" and
 *   maps 1:1 to the user's mental model for trash. The CASCADE on
 *   ``fd_messages`` handles child rows.
 */
import type { NextRequest } from "next/server";

import { fdUidOf, requireAuth, requireFdUid } from "@/lib/auth";
import {
  deleteSession,
  getSession,
  patchSession,
  type SessionPatch,
} from "@/lib/history-db";

export const runtime = "edge";

interface RouteCtx {
  params: { id: string };
}

export async function GET(req: NextRequest, ctx: RouteCtx): Promise<Response> {
  const authFail = requireAuth(req);
  if (authFail) return authFail;
  const uidFail = requireFdUid(req);
  if (uidFail) return uidFail;
  const uid = fdUidOf(req) as string;
  const { id } = ctx.params;
  if (!id) return Response.json({ error: "missing id" }, { status: 400 });
  try {
    const result = await getSession(id, uid);
    if (!result) return Response.json({ error: "not found" }, { status: 404 });
    return Response.json(result);
  } catch (e) {
    return Response.json(
      { error: e instanceof Error ? e.message : "getSession failed" },
      { status: 500 },
    );
  }
}

export async function PATCH(req: NextRequest, ctx: RouteCtx): Promise<Response> {
  const authFail = requireAuth(req);
  if (authFail) return authFail;
  const uidFail = requireFdUid(req);
  if (uidFail) return uidFail;
  const uid = fdUidOf(req) as string;
  const { id } = ctx.params;
  if (!id) return Response.json({ error: "missing id" }, { status: 400 });

  let patch: SessionPatch = {};
  try {
    const raw = (await req.json()) as Partial<SessionPatch>;
    // Whitelist known fields; silently drop anything else so a
    // compromised client can't splat ``user_id`` or ``created_at``.
    if (typeof raw.title === "string") patch.title = raw.title;
    if ("category" in raw)
      patch.category =
        raw.category === null || typeof raw.category === "string"
          ? (raw.category as string | null)
          : undefined;
    if (typeof raw.provider_id === "string") patch.provider_id = raw.provider_id;
    if (raw.provider_id === null) patch.provider_id = null;
    if (typeof raw.mode === "string") patch.mode = raw.mode;
    if (raw.mode === null) patch.mode = null;
    if (typeof raw.pinned === "boolean") patch.pinned = raw.pinned;
    if (typeof raw.archived === "boolean") patch.archived = raw.archived;
  } catch {
    return Response.json({ error: "invalid JSON" }, { status: 400 });
  }

  try {
    const session = await patchSession(id, uid, patch);
    return Response.json({ session });
  } catch (e) {
    const msg = e instanceof Error ? e.message : "patchSession failed";
    const status = msg.includes("not found") || msg.includes("not owned")
      ? 404
      : 500;
    return Response.json({ error: msg }, { status });
  }
}

export async function DELETE(req: NextRequest, ctx: RouteCtx): Promise<Response> {
  const authFail = requireAuth(req);
  if (authFail) return authFail;
  const uidFail = requireFdUid(req);
  if (uidFail) return uidFail;
  const uid = fdUidOf(req) as string;
  const { id } = ctx.params;
  if (!id) return Response.json({ error: "missing id" }, { status: 400 });
  try {
    await deleteSession(id, uid);
    // Always 204 on success — deletion is idempotent, don't leak
    // existence via differentiated responses.
    return new Response(null, { status: 204 });
  } catch (e) {
    return Response.json(
      { error: e instanceof Error ? e.message : "deleteSession failed" },
      { status: 500 },
    );
  }
}
