/**
 * ARIA middleware — attaches a stable per-browser identifier to every
 * request. That id is what every history row gets tagged with, so any
 * two devices sharing the same ``fd_uid`` share a chat history.
 *
 * Why a middleware and not a route-level helper?
 *   - history routes, chat routes, and static data routes all need the
 *     same id, so centralising avoids drift
 *   - cookie issuance must happen via ``Set-Cookie``, which route
 *     handlers can't safely do for GET requests that also stream
 *   - the generated UUID needs to be visible to downstream handlers
 *     via request headers on the very first hit (not one round-trip
 *     later), so we attach ``x-fd-uid`` to the forwarded request
 *
 * Privacy
 *   - ``fd_uid`` is a randomly-generated opaque UUID, not derived from
 *     any browser fingerprint or IP. It's a "technical" cookie in the
 *     GDPR sense (necessary for the service to work), so no consent
 *     banner is required for the id itself.
 *   - When the user later attaches a ``display_name`` / ``label_email``
 *     we DO need consent; that gate lives in the UserBadge modal.
 *
 * Scope
 *   - matcher excludes Next.js static assets, image optimisation, the
 *     map tiles, and the auth-exchange endpoint (that one has its own
 *     cookie-issuing semantics)
 */
import { NextResponse, type NextRequest } from "next/server";

const COOKIE = "fd_uid";
const HEADER = "x-fd-uid";
const ONE_YEAR_SEC = 60 * 60 * 24 * 365;

function isUuid(v: string | undefined): v is string {
  if (!v) return false;
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(v);
}

export function middleware(req: NextRequest) {
  const existing = req.cookies.get(COOKIE)?.value;
  const uid = isUuid(existing) ? existing : crypto.randomUUID();

  // Forward the id to the route handler via a header, so handlers can
  // read it synchronously without cookie parsing.
  const fwdHeaders = new Headers(req.headers);
  fwdHeaders.set(HEADER, uid);

  const res = NextResponse.next({ request: { headers: fwdHeaders } });

  // Set/refresh the cookie only when missing or malformed — the
  // refresh path is cheap and keeps the year-long maxAge rolling so
  // returning users don't get rotated out of their history.
  if (!isUuid(existing)) {
    res.cookies.set(COOKIE, uid, {
      httpOnly: true,
      sameSite: "lax", // needs to survive top-level nav from external links
      secure: process.env.NODE_ENV === "production",
      path: "/",
      maxAge: ONE_YEAR_SEC,
    });
  }

  // Sprint 2026-05-06 O9: Security headers (CSP / HSTS / X-Frame / Permissions).
  // CSP allows: self + Vercel Analytics + LLM provider APIs + map tiles +
  // Turso libsql + WASM (unsafe-eval needed for Rust seir_wasm).
  //
  // 2026-05-07: web_prototype prototype at /abs/* loads React + Leaflet +
  // Babel from unpkg, Pretendard font from jsdelivr, Seoul GeoJSON from
  // raw.githubusercontent, satellite tiles from arcgisonline, cloud overlay
  // from gibs.earthdata.nasa.gov. CSP widened so the prototype renders.
  const csp = [
    "default-src 'self'",
    "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://va.vercel-scripts.com https://unpkg.com",
    "style-src 'self' 'unsafe-inline' https://unpkg.com https://cdn.jsdelivr.net",
    "img-src 'self' data: blob: https://*.vercel-insights.com https://basemaps.cartocdn.com https://*.basemaps.cartocdn.com https://*.cartocdn.com https://*.tile.openstreetmap.org https://api.mapbox.com https://server.arcgisonline.com https://gibs.earthdata.nasa.gov https://unpkg.com",
    "font-src 'self' data: https://cdn.jsdelivr.net",
    "connect-src 'self' https://api.anthropic.com https://api.openai.com https://generativelanguage.googleapis.com https://ai-gateway.vercel.sh https://va.vercel-scripts.com https://*.turso.io wss://*.turso.io https://basemaps.cartocdn.com https://*.basemaps.cartocdn.com https://*.cartocdn.com https://api.mapbox.com https://raw.githubusercontent.com https://server.arcgisonline.com https://gibs.earthdata.nasa.gov",
    "worker-src 'self' blob:",
    "frame-ancestors 'none'",
    "base-uri 'self'",
    "form-action 'self'",
  ].join("; ");

  res.headers.set("Content-Security-Policy", csp);
  res.headers.set("X-Content-Type-Options", "nosniff");
  res.headers.set("X-Frame-Options", "DENY");
  res.headers.set("Referrer-Policy", "strict-origin-when-cross-origin");
  res.headers.set(
    "Permissions-Policy",
    "geolocation=(), camera=(), microphone=(), payment=()",
  );
  // HSTS — 1 year + subdomains + preload (Vercel HTTPS 강제, production only)
  if (process.env.NODE_ENV === "production") {
    res.headers.set(
      "Strict-Transport-Security",
      "max-age=31536000; includeSubDomains; preload",
    );
  }

  return res;
}

export const config = {
  // Run on every page and API route except Next.js build artefacts,
  // Leaflet tile proxy pass-through, and the demo-auth handshake.
  matcher: [
    "/((?!_next/static|_next/image|favicon.ico|api/auth/exchange).*)",
  ],
};
