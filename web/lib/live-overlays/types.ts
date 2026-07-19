/**
 * Shared types for the ``/api/overlays/live`` Edge route and its
 * per-source providers in this folder.
 *
 * Design
 * ------
 * Each provider exposes a single async function:
 *
 *     export async function buildX(signal: AbortSignal): Promise<MetricPayload | null>
 *
 * - Return ``null`` when the source is unavailable (missing env key,
 *   HTTP failure, schema mismatch). The orchestrator then falls back
 *   to the cached demo aggregate so the map is never blank.
 * - Throw only for programmer errors. Network / auth failures MUST
 *   return ``null`` — they're expected.
 * - Providers are silent; all error diagnostics end up on the
 *   ``freshness`` + ``note`` fields so the UI can render them without
 *   blowing up a browser console.
 *
 * Why flat gu → value?
 *   Choropleth colouring only needs one scalar per gu. Keeping the
 *   structure boring means the map-render path doesn't grow a switch
 *   statement per new data source — the colormap + legend unit string
 *   is enough.
 */

import type { GuChoroplethRow } from "@/components/MapPanel";

/** Top-level metric identifier — 1:1 with the overlay picker value. */
export type LiveMetricId =
  | "ili_forecast" //  post_E v22.6 DL/epi ensemble (Turso weekly_disease)
  | "ili_live"     //  KDCA sentinel — whatever Turso has as "latest observed"
  | "ili_alert"    //  gu-level threshold breach flag (0/1 → colours)
  | "air"          //  Seoul PM2.5 realtime (µg/m³)
  | "temp"         //  KMA hourly surface temp (°C)
  | "humidity"     //  KMA hourly relative humidity (%)
  | "er"           //  NEDIS ER bed occupancy (% full)
  | "metro";       //  Seoul Metro peak-car crowding (% cars full)

/** Freshness tier rendered as 🔴 LIVE / 🟡 STALE / ⚫ OFFLINE. */
export type Freshness = "live" | "stale" | "offline";

export interface MetricPayload {
  id: LiveMetricId;
  /** Korean label shown in the picker + legend. */
  label_ko: string;
  /** English label. */
  label_en: string;
  /** Unit string for the tooltip (e.g. "µg/m³", "%", "°C"). */
  unit: string;
  /** One-word provenance hint (e.g. "turso", "seoul_openapi", "fallback"). */
  source: string;
  /** ISO timestamp of when the upstream data was produced/observed. */
  observed_at: string;
  /** Per-gu rows. Missing gu = transparent in the choropleth. */
  rows: GuChoroplethRow[];
  /** "live" / "stale" / "offline" badge colour. */
  freshness: Freshness;
  /** Optional one-line human hint (error message, fallback reason). */
  note?: string;
}

/** The full response from /api/overlays/live. */
export interface LiveOverlaysResponse {
  /** Keyed by metric id. Missing keys = server couldn't compute the
   *  layer at all AND didn't have a fallback file (very rare). */
  metrics: Partial<Record<LiveMetricId, MetricPayload>>;
  /** When the orchestrator assembled this response. */
  generated_at: string;
  /** How long the response is reliable for, in seconds. Lines up with
   *  the ISR ``revalidate`` — the client uses it to decide when to
   *  re-fetch without waiting for the route to expire. */
  ttl_seconds: number;
}

/**
 * Seoul's 25 gu in their canonical order. Used by providers that receive
 * data keyed by something other than gu_nm (e.g. station id, hospital
 * code) and need a translation table. Keep this in sync with
 * ``simulation.database.config.SEOUL_GU_NAMES``.
 */
export const SEOUL_GU = [
  "강남구", "강동구", "강북구", "강서구", "관악구",
  "광진구", "구로구", "금천구", "노원구", "도봉구",
  "동대문구", "동작구", "마포구", "서대문구", "서초구",
  "성동구", "성북구", "송파구", "양천구", "영등포구",
  "용산구", "은평구", "종로구", "중구", "중랑구",
] as const;

export type GuName = (typeof SEOUL_GU)[number];

/** Cheap classifier the providers use to stamp each payload. */
export function classifyFreshness(
  observedAt: string,
  liveWindowMin = 15,
  staleWindowMin = 60,
): Freshness {
  const obs = Date.parse(observedAt);
  if (!Number.isFinite(obs)) return "offline";
  const ageMin = (Date.now() - obs) / 60_000;
  if (ageMin <= liveWindowMin) return "live";
  if (ageMin <= staleWindowMin) return "stale";
  return "offline";
}

/** Timeout wrapper around fetch — all providers must cap their call so
 *  one dead upstream can't stall the whole /api/overlays/live response
 *  past the Edge function's 30 s wall. */
export async function tfetch(
  url: string,
  init: RequestInit & { timeoutMs?: number } = {},
): Promise<Response> {
  const { timeoutMs = 3000, signal: outer, ...rest } = init;
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeoutMs);
  // Chain an outer signal if the caller supplied one.
  if (outer) {
    if (outer.aborted) ctl.abort();
    else outer.addEventListener("abort", () => ctl.abort(), { once: true });
  }
  try {
    return await fetch(url, { ...rest, signal: ctl.signal });
  } finally {
    clearTimeout(timer);
  }
}
