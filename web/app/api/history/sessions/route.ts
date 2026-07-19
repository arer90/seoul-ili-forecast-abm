/**
 * GET  /api/history/sessions — list the caller's chat sessions
 * POST /api/history/sessions — create a new empty chat session
 *
 * Identity: ``fd_uid`` from middleware (header ``x-fd-uid``).
 *
 * GET query params
 *   - ``category`` (string | omitted)
 *        - ``none``  → sessions whose category IS NULL ("No category" bucket)
 *        - any other → exact category match
 *        - omitted   → all categories
 *   - ``archived`` (bool, default false) — ``1`` / ``true`` returns archived ones
 *   - ``limit``    (int, default 50, clamped to [1, 200])
 *
 * Response shape
 *     { sessions: FdSession[] }
 *
 * POST body
 *     { title?, category?, provider_id?, mode? }
 *
 * Response shape
 *     { session: FdSession }
 */
import type { NextRequest } from "next/server";

import { fdUidOf, requireAuth, requireFdUid } from "@/lib/auth";
import {
  createSession,
  listSessions,
  type CreateSessionInput,
} from "@/lib/history-db";

export const runtime = "edge";

export async function GET(req: NextRequest): Promise<Response> {
  const authFail = requireAuth(req);
  if (authFail) return authFail;
  const uidFail = requireFdUid(req);
  if (uidFail) return uidFail;
  const uid = fdUidOf(req) as string;

  const url = new URL(req.url);
  const categoryParam = url.searchParams.get("category");
  const archivedParam = url.searchParams.get("archived");
  const limitParam = url.searchParams.get("limit");

  let category: string | null | undefined = undefined;
  if (categoryParam != null) {
    category = categoryParam === "none" ? null : categoryParam;
  }
  const archived = archivedParam === "1" || archivedParam === "true";
  const limitN = limitParam ? Number(limitParam) : NaN;

  try {
    const sessions = await listSessions(uid, {
      category,
      archived,
      limit: Number.isFinite(limitN) ? limitN : undefined,
    });
    return Response.json({ sessions });
  } catch (e) {
    return Response.json(
      { error: e instanceof Error ? e.message : "listSessions failed" },
      { status: 500 },
    );
  }
}

export async function POST(req: NextRequest): Promise<Response> {
  const authFail = requireAuth(req);
  if (authFail) return authFail;
  const uidFail = requireFdUid(req);
  if (uidFail) return uidFail;
  const uid = fdUidOf(req) as string;

  // Empty body is allowed — callers can POST with no args to spin up
  // a default "New chat" session.
  let body: CreateSessionInput = {};
  try {
    const raw = (await req.json()) as Partial<CreateSessionInput>;
    body = {
      title: typeof raw.title === "string" ? raw.title : undefined,
      category:
        raw.category === null || typeof raw.category === "string"
          ? raw.category
          : undefined,
      provider_id:
        typeof raw.provider_id === "string" ? raw.provider_id : undefined,
      mode: typeof raw.mode === "string" ? raw.mode : undefined,
    };
  } catch {
    // ignore malformed JSON — treat as no body
  }

  try {
    const session = await createSession(uid, body);
    return Response.json({ session }, { status: 201 });
  } catch (e) {
    return Response.json(
      { error: e instanceof Error ? e.message : "createSession failed" },
      { status: 500 },
    );
  }
}
