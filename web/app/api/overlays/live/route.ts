/**
 * GET /api/overlays/live
 *
 * Returns an 8-layer ``LiveOverlaysResponse`` for the map picker:
 *
 *   ili_forecast  ILI 예측              (post_E v22.6 static aggregate)
 *   ili_live      ILI 실측              (Turso weekly_disease)
 *   ili_alert     유행 경보 (q70 breach)(Turso-derived)
 *   air           PM2.5                 (서울 열린데이터 광장)
 *   temp          기온                  (기상청 ASOS)
 *   humidity      습도                  (기상청 ASOS)
 *   er            응급실 과밀도          (NEDIS)
 *   metro         지하철 혼잡도          (서울교통공사 t-data)
 *
 * Every layer always renders something — providers that return ``null``
 * (missing env key) are backfilled from ``public/aggregates/live-
 * overlays.json`` or a deterministic synthetic generator so the map
 * never goes blank.
 *
 * Caching
 * -------
 * ``revalidate = 300`` — Vercel's ISR keeps the payload for 5 min per
 * edge region. That matches the upstream minimum cadence (air quality
 * refreshes hourly, weather hourly, ER every 15 min); fetching more
 * often burns rate-limit budget without producing new numbers.
 *
 * Auth
 * ----
 * We do NOT require ``DEMO_TOKEN`` here because the map is the first
 * thing anonymous visitors see. The response is non-secret aggregate
 * data — the same values that ship in ``public/aggregates/``.
 */
import type { NextRequest } from "next/server";

import { buildLiveOverlays } from "@/lib/live-overlays";

export const runtime = "edge";
export const revalidate = 300;

export async function GET(req: NextRequest): Promise<Response> {
  const origin = new URL(req.url).origin;
  try {
    const payload = await buildLiveOverlays(origin, req.signal);
    return Response.json(payload, {
      headers: {
        // Let Vercel + the browser share the same 5-minute window, with
        // ``stale-while-revalidate`` to avoid jank while the next
        // refresh is in flight.
        "cache-control":
          "public, max-age=60, s-maxage=300, stale-while-revalidate=600",
      },
    });
  } catch (e) {
    return Response.json(
      {
        error: e instanceof Error ? e.message : String(e),
        metrics: {},
        generated_at: new Date().toISOString(),
        ttl_seconds: 60,
      },
      { status: 502 },
    );
  }
}
