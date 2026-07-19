/**
 * POST /api/chat — solo mode shortcut + streaming.
 * Body: { provider, model, mode: "solo", messages, temperature? }
 *
 * Response: SSE stream of JSON events tagged with providerId.
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

  // Shared guard — same logic as /api/chat/parallel, so the two cannot drift.
  const limited = await llmRateGuard(req);
  if (limited) return limited;

  let body: Partial<HermesRequest> & { provider?: string; model?: string };
  try {
    body = (await req.json()) as Partial<HermesRequest> & {
      provider?: string;
      model?: string;
    };
  } catch {
    return new Response("invalid JSON", { status: 400 });
  }

  // Backcompat: if caller passed { provider, model, messages } use solo.
  const hermesReq: HermesRequest = {
    mode: body.mode ?? "solo",
    messages: body.messages ?? [],
    providers:
      body.providers ??
      (body.provider && body.model
        ? [
            {
              id: body.provider as HermesRequest["providers"][number]["id"],
              model: body.model,
            },
          ]
        : []),
    synthesiser: body.synthesiser,
    signal: req.signal,
  };
  if (hermesReq.providers.length === 0) {
    return new Response("no providers supplied", { status: 400 });
  }

  const ndjson = runHermes(hermesReq);
  return new Response(ndjsonToSSE(ndjson), { headers: SSE_HEADERS });
}
