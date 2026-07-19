/**
 * POST /api/auth/exchange — demo-token → session cookie.
 *
 * Accepts `{ token: "…" }` in the body. If the token matches
 * `process.env.DEMO_TOKEN`, sets an httpOnly SameSite=Strict cookie
 * and returns 204. Never accepts the token via URL params — per our
 * prompt-injection defense, secrets never ride in query strings or
 * referer headers.
 */
import { NextResponse, type NextRequest } from "next/server";

import { demoToken, issueSessionCookie } from "@/lib/auth";

export const runtime = "edge";

export async function POST(req: NextRequest): Promise<Response> {
  const expected = demoToken();
  if (!expected) {
    // Local dev without a token — no-op 204.
    return new NextResponse(null, { status: 204 });
  }
  let body: { token?: string };
  try {
    body = (await req.json()) as { token?: string };
  } catch {
    return new NextResponse("invalid JSON", { status: 400 });
  }
  if (body.token !== expected) {
    return new NextResponse("unauthorized", { status: 401 });
  }
  const res = new NextResponse(null, { status: 204 });
  return issueSessionCookie(res);
}
