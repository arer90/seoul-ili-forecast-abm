/**
 * POST /api/history/export — stream the caller's full history as JSONL
 *
 * Returns a text/plain body of newline-delimited JSON. Each line is
 * either a session record (``{ type: "session", ... }``) or a message
 * record (``{ type: "message", ... }``). Sessions include both active
 * and archived; messages follow their parent session.
 *
 * The browser downloads via ``fetch`` → ``Blob`` → anchor click — no
 * Content-Disposition is needed for the demo, but we set one anyway
 * so users who open the endpoint directly get a friendly filename.
 *
 * POST (not GET) because the operation reads every row the user owns
 * and browsers are aggressive about prefetching GET URLs.
 */
import type { NextRequest } from "next/server";

import { fdUidOf, requireAuth, requireFdUid } from "@/lib/auth";
import { exportJsonl } from "@/lib/history-db";

export const runtime = "edge";

export async function POST(req: NextRequest): Promise<Response> {
  const authFail = requireAuth(req);
  if (authFail) return authFail;
  const uidFail = requireFdUid(req);
  if (uidFail) return uidFail;
  const uid = fdUidOf(req) as string;

  try {
    const body = await exportJsonl(uid);
    const ts = new Date().toISOString().replace(/[:.]/g, "-");
    return new Response(body, {
      status: 200,
      headers: {
        "content-type": "application/x-ndjson; charset=utf-8",
        "content-disposition": `attachment; filename="frame-d-history-${ts}.jsonl"`,
        "cache-control": "no-store",
      },
    });
  } catch (e) {
    return Response.json(
      { error: e instanceof Error ? e.message : "export failed" },
      { status: 500 },
    );
  }
}
