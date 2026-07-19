/**
 * GET /api/providers — report which LLM adapters are actually usable,
 * plus the environment hints the client needs to pick a sane default.
 *
 * Response shape
 * --------------
 *     {
 *       providers: [
 *         { id, available, models, note? },
 *         ...
 *       ],
 *       environment: {
 *         mode: "local" | "cloud",
 *         recommended: ProviderId,
 *         fallbackOrder: ProviderId[],
 *         computedAt: number
 *       }
 *     }
 *
 * Every field is non-secret. Callers use ``recommended`` for the
 * ChatPanel initial selection and ``fallbackOrder`` for the future
 * 429/5xx chain in Hermes (Phase 2 / not yet wired — the field exists
 * now so client code can read it without a second round trip later).
 *
 * Caching happens inside ``detectEnvironment()`` (5 s TTL per isolate)
 * so hammering this endpoint during a page load doesn't stampede the
 * Ollama probe.
 */
import type { NextRequest } from "next/server";

import { requireAuth } from "@/lib/auth";
import { detectEnvironment } from "@/lib/environment";
import { listOllamaModels } from "@/lib/providers/ollama";
import { getProviders } from "@/lib/providers/registry";

export const runtime = "edge";

export async function GET(req: NextRequest): Promise<Response> {
  const authFail = requireAuth(req);
  if (authFail) return authFail;

  const env = await detectEnvironment();

  // We compose the per-provider row from two sources: the static
  // adapter (``models()``, ``available()`` based on key presence) plus
  // the already-probed availability from ``detectEnvironment()``. The
  // latter wins on conflicts — it knows about Ollama health, and its
  // ``available`` map is the one the client will cross-reference.
  const adapters = getProviders();
  // Ollama gets a live probe of /api/tags so the picker matches what
  // the user actually has pulled. Everything else keeps its static
  // ``models()`` because those lists are tied to provider SKUs, not
  // local install state.
  const ollamaLive = env.available.ollama ? await listOllamaModels() : null;

  const out: Array<{
    id: string;
    available: boolean;
    models: readonly string[];
    note?: string;
  }> = [];
  for (const p of adapters.values()) {
    const liveAvailable = env.available[p.id];
    const note = env.notes[p.id];
    const models =
      p.id === "ollama" && ollamaLive && ollamaLive.length > 0
        ? ollamaLive
        : p.models();
    out.push({
      id: p.id,
      available: liveAvailable,
      models,
      note,
    });
  }

  return Response.json({
    providers: out,
    environment: {
      mode: env.mode,
      recommended: env.recommended,
      fallbackOrder: env.fallbackOrder,
      computedAt: env.computedAt,
    },
  });
}
