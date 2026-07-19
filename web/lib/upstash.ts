/**
 * Upstash Redis (global edge) — session store + rate limit + query
 * cache. All three are optional: if env vars are missing the helpers
 * no-op gracefully so `npm run dev` works without an Upstash account.
 */
import { Ratelimit } from "@upstash/ratelimit";
import { Redis } from "@upstash/redis";

type Nullable<T> = T | null;

let _redis: Nullable<Redis> = null;
let _ratelimit: Nullable<Ratelimit> = null;
let _publicRatelimit: Nullable<Ratelimit> = null;

export function redis(): Nullable<Redis> {
  if (_redis) return _redis;
  const url = process.env.UPSTASH_URL;
  const token = process.env.UPSTASH_TOKEN;
  if (!url || !token) return null;
  _redis = new Redis({ url, token });
  return _redis;
}

/** 60 requests / minute / IP — token-gated mode (DEMO_TOKEN cookie present). */
export function ratelimit(): Nullable<Ratelimit> {
  if (_ratelimit) return _ratelimit;
  const r = redis();
  if (!r) return null;
  _ratelimit = new Ratelimit({
    redis: r,
    limiter: Ratelimit.slidingWindow(60, "1 m"),
    analytics: true,
    prefix: "frame-d",
  });
  return _ratelimit;
}

/** 15 requests / minute / IP — PUBLIC_DEMO mode (no auth gate). Tighter
 * than the token-mode limit because anyone can hit it. Override the per-min
 * count via `PUBLIC_RATE_PER_MIN` env var; default 15 covers a typical
 * conversational pace (one question every 4 sec) and still kicks bots out.
 */
export function publicRatelimit(): Nullable<Ratelimit> {
  if (_publicRatelimit) return _publicRatelimit;
  const r = redis();
  if (!r) return null;
  const perMin = Number.parseInt(
    process.env.PUBLIC_RATE_PER_MIN ?? "15",
    10,
  );
  _publicRatelimit = new Ratelimit({
    redis: r,
    limiter: Ratelimit.slidingWindow(
      Number.isFinite(perMin) && perMin > 0 ? perMin : 15,
      "1 m",
    ),
    analytics: true,
    prefix: "frame-d:public",
  });
  return _publicRatelimit;
}

/** Global daily request counter. Returns `{ count, allowed, cap }` —
 * caller is expected to short-circuit with HTTP 429 when `allowed === false`.
 *
 * Cap defaults to 2000 requests/day across the entire deployment (override
 * via `PUBLIC_DAILY_CAP` env). At a worst-case ~$0.12 per agent loop this
 * caps daily spend at ~$240 — well under most Anthropic Tier 1 monthly
 * limits ($100/mo default). Resets at UTC midnight via daily key rotation.
 *
 * Returns `{ allowed: true, count: -1, cap: -1 }` when Redis is unavailable.
 *
 * NOTE: the per-IP rate limit does NOT cover you in that case — `ratelimit()`
 * and `publicRatelimit()` depend on the same `redis()` call and return null
 * too, so both protections vanish together. `app/api/chat/route.ts` therefore
 * refuses public-mode requests outright when Upstash is unconfigured, which is
 * what keeps this fail-open path from ever being reached in production.
 */
export async function checkDailyGlobalCap(): Promise<{
  allowed: boolean;
  count: number;
  cap: number;
}> {
  const r = redis();
  const cap = Number.parseInt(
    process.env.PUBLIC_DAILY_CAP ?? "2000",
    10,
  );
  const effectiveCap = Number.isFinite(cap) && cap > 0 ? cap : 2000;
  if (!r) return { allowed: true, count: -1, cap: effectiveCap };

  // Use UTC date as key so the counter rotates at midnight UTC.
  // (Korean midnight = UTC 15:00; if you'd rather rotate at KST midnight,
  // add `+ (9*3600*1000)` to the Date and re-derive ISO date.)
  const day = new Date().toISOString().slice(0, 10); // YYYY-MM-DD
  const key = `frame-d:public:daily:${day}`;
  // INCR is atomic; first call returns 1, subsequent calls bump.
  const count = (await r.incr(key)) as number;
  // Set 26h expiry on first hit so the key auto-cleans (covers rotation).
  if (count === 1) {
    await r.expire(key, 60 * 60 * 26);
  }
  return { allowed: count <= effectiveCap, count, cap: effectiveCap };
}

/** Cache a JSON payload with a TTL in seconds. Safe to call w/o Redis. */
export async function cacheGetOrSet<T>(
  key: string,
  ttlSec: number,
  build: () => Promise<T>,
): Promise<T> {
  const r = redis();
  if (!r) return build();
  const hit = (await r.get(key)) as T | null;
  if (hit !== null && hit !== undefined) return hit;
  const fresh = await build();
  await r.set(key, fresh as unknown, { ex: ttlSec });
  return fresh;
}
