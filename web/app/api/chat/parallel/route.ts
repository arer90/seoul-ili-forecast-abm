/**
 * POST /api/chat/parallel — fan-out / synthesis / relay.
 * Body: HermesRequest shape. Response: SSE of tagged events.
 */
import type { NextRequest } from "next/server";

import { requireAuth } from "@/lib/auth";
import { runHermes, type HermesRequest } from "@/lib/hermes";
import { llmRateGuard } from "@/lib/rate-guard";
import { SSE_HEADERS, ndjsonToSSE } from "@/lib/util/sse";

export const runtime = "edge";

export async function POST(req: NextRequest): Promise<Response> {
  const authFail = requireAuth(req);
  if (authFail) return authFail;

  // Same guard as /api/chat — this route reaches the same LLM entry point and
  // fans out to several providers, so leaving it unguarded is strictly worse.
  const limited = await llmRateGuard(req);
  if (limited) return limited;

  let body: HermesRequest;
  try {
    body = (await req.json()) as HermesRequest;
  } catch {
    return new Response("invalid JSON", { status: 400 });
  }
  if (!body.providers || body.providers.length < 1) {
    return new Response("parallel mode needs at least 1 provider", {
      status: 400,
    });
  }
  body.signal = req.signal;
  const ndjson = runHermes(body);
  return new Response(ndjsonToSSE(ndjson), { headers: SSE_HEADERS });
}
