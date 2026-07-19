/**
 * GET /api/auth/check — probe whether the caller's session cookie is valid.
 *
 * Returns:
 *   200 {"ok": true,  "live": true,  "model": "claude-sonnet-4-6"}  when cookie matches DEMO_TOKEN
 *   200 {"ok": false, "live": false, "reason": "no_cookie"}         when cookie missing/expired/wrong
 *
 * The frontend calls this on page load (before any /api/chat traffic) so
 * the "⚡ Claude" / "🔌 Mock" status badge can render in the chat header
 * before the user has typed anything. Without this probe, the indicator
 * was stuck on `unknown` until after the first response — making it look
 * like Claude was disconnected when it wasn't.
 *
 * Always returns HTTP 200 (no 401) so a missing cookie isn't a network-
 * level error from the client's perspective; the JSON body carries the
 * decision. This keeps the probe cheap and never spams the browser
 * console with red 401s.
 */
import { NextResponse, type NextRequest } from "next/server";

import { isAuthed } from "@/lib/auth";

export const runtime = "edge";

export async function GET(req: NextRequest): Promise<Response> {
  const ok = isAuthed(req);
  const hasKey =
    typeof process.env.ANTHROPIC_API_KEY === "string" &&
    process.env.ANTHROPIC_API_KEY.length > 0;
  const publicDemo = process.env.PUBLIC_DEMO === "1";
  return NextResponse.json(
    ok
      ? {
          ok: true,
          live: hasKey,
          model: "claude-sonnet-4-6",
          mode: publicDemo ? "public" : "token",
          reason: hasKey
            ? publicDemo
              ? "public_demo"
              : "cookie_valid"
            : "no_api_key",
        }
      : { ok: false, live: false, mode: "token", reason: "no_cookie" },
    {
      status: 200,
      headers: {
        // Never cache — the answer depends on the caller's cookie and on
        // env state (DEMO_TOKEN/ANTHROPIC_API_KEY can be rotated without
        // a redeploy on Vercel).
        "cache-control": "no-store, max-age=0",
      },
    },
  );
}
