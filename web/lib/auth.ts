/**
 * Demo-token auth. The token is never exposed in URL parameters (prompt
 * injection risk). We accept it once via a POST to `/api/auth/exchange`
 * and set an httpOnly SameSite=Strict cookie for the session.
 *
 * This deliberately avoids any OAuth dance or user creation — per
 * ENGINEERING_PRINCIPLES.md "NEVER create accounts on the user's behalf".
 */
import { NextResponse, type NextRequest } from "next/server";

const COOKIE_NAME = "frame_d_session";

export function demoToken(): string | undefined {
  return process.env.DEMO_TOKEN;
}

/** Returns true when the request carries a valid session cookie.
 *
 * 2026-04-27 (Codex audit S1): fail-closed in production.
 *   Local dev (NODE_ENV != production): DEMO_TOKEN 미설정 시 통과 (편의).
 *   Production (Vercel/배포): DEMO_TOKEN 미설정 시 무조건 401 (security).
 *
 * 2026-05-08 (PUBLIC_DEMO escape hatch): set PUBLIC_DEMO=1 in Vercel env to
 *   allow ANY visitor to use Claude without a ?t=TOKEN link. Abuse is bounded
 *   by:
 *     1. Upstash sliding-window rate limit (60 req/min per IP, lib/upstash.ts:30)
 *     2. Anthropic monthly cap (Tier 1 = $100/mo by default — set in Anthropic console)
 *     3. Vercel Hobby/Pro per-deployment quotas
 *   Roll back by deleting the PUBLIC_DEMO env var (or setting it to "0") and
 *   redeploying. Token-gated mode (?t=TOKEN) still works in parallel —
 *   PUBLIC_DEMO is purely additive.
 */
export function isAuthed(req: NextRequest): boolean {
  // Public demo mode — open access, no cookie required.
  if (process.env.PUBLIC_DEMO === "1") return true;

  const t = demoToken();
  const isProd = process.env.NODE_ENV === "production";
  if (!t) {
    if (isProd) {
      // Fail-closed: production 에서 DEMO_TOKEN 없으면 거부
      console.error(
        "[auth] DEMO_TOKEN unset in production — refusing all requests",
      );
      return false;
    }
    return true; // local dev only
  }
  const cookie = req.cookies.get(COOKIE_NAME)?.value;
  return cookie === t;
}

export function requireAuth(req: NextRequest): NextResponse | null {
  return isAuthed(req)
    ? null
    : NextResponse.json({ error: "unauthorized" }, { status: 401 });
}

export function issueSessionCookie(res: NextResponse): NextResponse {
  const t = demoToken();
  if (!t) return res;
  // Cookie must be `secure` on Vercel prod (HTTPS) but MUST NOT be in
  // localhost/dev (HTTP) — the browser drops it silently otherwise,
  // leaving every /api/* at 401 with no visible clue. Gate on
  // NODE_ENV, matching the middleware's `fd_uid` policy.
  const isProd = process.env.NODE_ENV === "production";
  res.cookies.set(COOKIE_NAME, t, {
    httpOnly: true,
    sameSite: "strict",
    secure: isProd,
    path: "/",
    // 30 days — demo period spans the thesis defense (2026-05-08) and the
    // post-defense Q&A window. Previously 8h, which silently expired in the
    // middle of long sessions and made every fresh visitor see the mock
    // fallback (G-159 style symptom). The token itself is rotation-eligible
    // and will be revoked after the demo, so a longer cookie is acceptable.
    maxAge: 60 * 60 * 24 * 30,
  });
  return res;
}

// ── Per-browser identity (ARIA history) ──────────────────────────
//
// ``middleware.ts`` assigns every request a stable ``fd_uid`` UUID and
// forwards it to route handlers via the ``x-fd-uid`` header. The cookie
// itself is the source of truth, but route handlers should read the
// header because it's already validated (UUID-shaped) and guaranteed
// present — middleware issues one on the fly if the cookie was missing.
//
// Two users on different browsers get different uids. Two tabs on the
// same browser share an uid (and therefore a chat history). That's the
// intended privacy model for a demo: no accounts, no password, but you
// still see your own history back on reload.

const FD_UID_HEADER = "x-fd-uid";
const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/**
 * Return the caller's fd_uid, or ``null`` if middleware somehow
 * didn't stamp the request (should never happen on routes matched by
 * the middleware config). Routes that need an identity should treat
 * ``null`` as 401.
 */
export function fdUidOf(req: NextRequest): string | null {
  const h = req.headers.get(FD_UID_HEADER);
  if (h && UUID_RE.test(h)) return h.toLowerCase();
  // Fallback to the cookie directly — covers the (hypothetically
  // excluded) edge case where the route is hit before middleware for
  // some reason (e.g. a matcher change).
  const c = req.cookies.get("fd_uid")?.value;
  if (c && UUID_RE.test(c)) return c.toLowerCase();
  return null;
}

/**
 * Short-circuit 401 when the request has no ``fd_uid``. Returns
 * ``null`` on the happy path so callers can use the same pattern as
 * ``requireAuth``:
 *
 *     const authFail = requireAuth(req); if (authFail) return authFail;
 *     const uidFail  = requireFdUid(req); if (uidFail) return uidFail;
 *     const uid      = fdUidOf(req)!; // guaranteed non-null below
 */
export function requireFdUid(req: NextRequest): NextResponse | null {
  return fdUidOf(req) == null
    ? NextResponse.json(
        { error: "missing fd_uid — middleware did not run for this path" },
        { status: 401 },
      )
    : null;
}
