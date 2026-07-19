/**
 * Shared rate-limit / spend-cap guard for every route that reaches an LLM.
 *
 * This exists as one helper rather than a per-route block because the first
 * version of the fail-closed check was written inline in /api/chat only, and
 * /api/chat/parallel — which reaches the same `runHermes` entry point and can
 * fan out to several providers per request — was left completely unguarded.
 *
 * Any new route that calls an LLM must call `llmRateGuard(req)` and return the
 * Response if one comes back.
 */
import type { NextRequest } from "next/server";

import { checkDailyGlobalCap, publicRatelimit, ratelimit } from "@/lib/upstash";

function clientIp(req: NextRequest): string {
  return (
    req.headers.get("x-forwarded-for")?.split(",")[0]?.trim() ??
    req.headers.get("x-real-ip") ??
    "unknown"
  );
}

/**
 * Returns a Response to short-circuit with, or null to proceed.
 *
 * Behaviour:
 *  - public mode (PUBLIC_DEMO=1) with no Upstash → 503, fail CLOSED. The auth
 *    gate is already off in this mode, so an absent limiter would leave an
 *    unauthenticated, unmetered proxy to a paid key. ALLOW_UNMETERED_PUBLIC_DEMO=1
 *    overrides for local development, with a warning.
 *  - per-IP limit exceeded → 429.
 *  - public mode daily global cap exceeded → 429 with retry-after.
 */
export async function llmRateGuard(req: NextRequest): Promise<Response | null> {
  const isPublic = process.env.PUBLIC_DEMO === "1";
  const rl = isPublic ? publicRatelimit() : ratelimit();

  if (isPublic && !rl) {
    if (process.env.ALLOW_UNMETERED_PUBLIC_DEMO !== "1") {
      console.error(
        "[rate-guard] PUBLIC_DEMO=1 but Upstash is not configured — refusing. " +
          "Set UPSTASH_URL and UPSTASH_TOKEN, or ALLOW_UNMETERED_PUBLIC_DEMO=1 for local dev.",
      );
      return new Response(
        JSON.stringify({
          error: "rate_limiter_unavailable",
          message:
            "공개 데모가 사용량 제한 없이 실행될 수 없습니다. 관리자에게 문의해 주세요.",
        }),
        { status: 503, headers: { "content-type": "application/json" } },
      );
    }
    console.warn(
      "[rate-guard] ALLOW_UNMETERED_PUBLIC_DEMO=1 — running with NO rate limit and NO daily cap.",
    );
  }

  if (rl) {
    const { success } = await rl.limit(clientIp(req));
    if (!success) {
      return new Response(JSON.stringify({ error: "rate_limited" }), {
        status: 429,
        headers: { "content-type": "application/json" },
      });
    }
  }

  if (isPublic) {
    const { allowed, count, cap } = await checkDailyGlobalCap();
    if (!allowed) {
      console.warn(
        `[rate-guard] daily global cap reached: ${count}/${cap} — refusing request`,
      );
      return new Response(
        JSON.stringify({
          error: "daily_cap_reached",
          message:
            "오늘의 공개 데모 사용량 한도를 초과했습니다. 내일 UTC 자정 이후 다시 시도해 주세요.",
          count,
          cap,
        }),
        {
          status: 429,
          headers: {
            "content-type": "application/json",
            "retry-after": "3600",
          },
        },
      );
    }
  }

  return null;
}
