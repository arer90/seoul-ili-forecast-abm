/**
 * GET   /api/history/me — return the caller's fd_user row (upserted if new)
 * PATCH /api/history/me — attach / update a display_name / label_email
 *
 * Identity: ``fd_uid`` from middleware.
 *
 * Privacy
 *   - GET is safe to call on every page load — it refreshes
 *     ``last_seen_at`` and creates the row idempotently on first hit.
 *   - PATCH is only triggered from the UserBadge modal, after the user
 *     actively enters a name/email. That's the consent moment for any
 *     non-technical identifier.
 *
 * Body (PATCH)
 *     { display_name?: string | null, label_email?: string | null }
 *
 * Response
 *     { user: FdUser }
 */
import type { NextRequest } from "next/server";

import { fdUidOf, requireAuth, requireFdUid } from "@/lib/auth";
import {
  updateUserLabel,
  upsertUser,
  type UserLabelPatch,
} from "@/lib/history-db";

export const runtime = "edge";

export async function GET(req: NextRequest): Promise<Response> {
  const authFail = requireAuth(req);
  if (authFail) return authFail;
  const uidFail = requireFdUid(req);
  if (uidFail) return uidFail;
  const uid = fdUidOf(req) as string;
  try {
    const user = await upsertUser(uid);
    return Response.json({ user });
  } catch (e) {
    return Response.json(
      { error: e instanceof Error ? e.message : "upsertUser failed" },
      { status: 500 },
    );
  }
}

export async function PATCH(req: NextRequest): Promise<Response> {
  const authFail = requireAuth(req);
  if (authFail) return authFail;
  const uidFail = requireFdUid(req);
  if (uidFail) return uidFail;
  const uid = fdUidOf(req) as string;

  let patch: UserLabelPatch = {};
  try {
    const raw = (await req.json()) as Partial<UserLabelPatch>;
    if ("display_name" in raw) {
      patch.display_name =
        raw.display_name === null || typeof raw.display_name === "string"
          ? (raw.display_name as string | null)
          : undefined;
    }
    if ("label_email" in raw) {
      patch.label_email =
        raw.label_email === null || typeof raw.label_email === "string"
          ? (raw.label_email as string | null)
          : undefined;
    }
  } catch {
    return Response.json({ error: "invalid JSON" }, { status: 400 });
  }

  try {
    const user = await updateUserLabel(uid, patch);
    return Response.json({ user });
  } catch (e) {
    return Response.json(
      { error: e instanceof Error ? e.message : "updateUserLabel failed" },
      { status: 500 },
    );
  }
}
