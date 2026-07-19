/**
 * Seoul 열린데이터 광장 — ``RealtimeCityAir`` per-gu PM2.5 + PM10 + O3 etc.
 * We expose only PM2.5 because that's the layer the thesis cares about
 * (respiratory confounder for ILI). Adding PM10/O3 would be a 5-line
 * diff in the parser.
 *
 * Endpoint (as of 2025-04):
 *
 *     GET http://openAPI.seoul.go.kr:8088/${KEY}/json/RealtimeCityAir/1/25/
 *
 * Response shape (trimmed):
 *
 *     {
 *       "RealtimeCityAir": {
 *         "list_total_count": 25,
 *         "row": [
 *           { "MSRSTE_NM": "종로구", "PM25": 18, "IDEX_NM": "좋음", ... },
 *           ...
 *         ]
 *       }
 *     }
 *
 * MSRSTE_NM already matches our ``gu_nm`` exactly, so the mapping is
 * direct.
 */

import type { MetricPayload } from "./types";
import { SEOUL_GU, classifyFreshness, tfetch } from "./types";
import { fetchOpenMeteoAir } from "./open-meteo";

interface OpenApiRow {
  MSRSTE_NM?: string;
  PM25?: string | number;
  MSRDT?: string; // YYYYMMDDHHMM
}

interface OpenApiEnvelope {
  RealtimeCityAir?: {
    list_total_count?: number;
    row?: OpenApiRow[];
  };
  /** Errors come back at the top level as this shape. */
  RESULT?: { CODE?: string; MESSAGE?: string };
}

export async function buildAir(
  signal: AbortSignal,
): Promise<MetricPayload | null> {
  const key = process.env.SEOUL_OPENAPI_KEY;
  if (!key) return openMeteoFallback();

  const url =
    `http://openAPI.seoul.go.kr:8088/${encodeURIComponent(key)}` +
    `/json/RealtimeCityAir/1/25/`;

  let env: OpenApiEnvelope;
  try {
    const r = await tfetch(url, { signal, timeoutMs: 3500 });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    env = (await r.json()) as OpenApiEnvelope;
  } catch {
    return openMeteoFallback();
  }

  if (env.RESULT?.CODE && env.RESULT.CODE !== "INFO-000") {
    return openMeteoFallback();
  }
  const rows = env.RealtimeCityAir?.row ?? [];
  if (rows.length === 0) return openMeteoFallback();

  const seoulSet = new Set<string>(SEOUL_GU);
  const choro: MetricPayload["rows"] = [];
  let latestStamp = 0;
  for (const r of rows) {
    const gu = (r.MSRSTE_NM ?? "").trim();
    if (!gu || !seoulSet.has(gu)) continue;
    const pm = Number(r.PM25);
    if (!Number.isFinite(pm) || pm < 0) continue;
    choro.push({ gu_nm: gu, value: pm });
    // MSRDT comes as YYYYMMDDHHMM; use the max across rows so
    // observedAt reflects the freshest tick in the batch.
    const tsNum = Date.parse(parseStamp(r.MSRDT ?? ""));
    if (Number.isFinite(tsNum) && tsNum > latestStamp) latestStamp = tsNum;
  }
  if (choro.length === 0) return openMeteoFallback();

  const observedAt =
    latestStamp > 0 ? new Date(latestStamp).toISOString() : new Date().toISOString();
  return {
    id: "air",
    label_ko: "대기질 (PM2.5)",
    label_en: "Air (PM2.5)",
    unit: "µg/m³",
    source: "seoul_openapi",
    observed_at: observedAt,
    rows: choro,
    freshness: classifyFreshness(observedAt),
  };
}

function parseStamp(s: string): string {
  // "202604220930" -> "2026-04-22T09:30:00+09:00" (KST)
  if (!/^\d{12}$/.test(s)) return "";
  const y = s.slice(0, 4);
  const m = s.slice(4, 6);
  const d = s.slice(6, 8);
  const H = s.slice(8, 10);
  const M = s.slice(10, 12);
  return `${y}-${m}-${d}T${H}:${M}:00+09:00`;
}

async function openMeteoFallback(): Promise<MetricPayload | null> {
  const rows = await fetchOpenMeteoAir(SEOUL_GU);
  if (!rows) return null;
  const observedAt = new Date().toISOString();
  return {
    id: "air",
    label_ko: "대기질 (PM2.5)",
    label_en: "Air (PM2.5)",
    unit: "µg/m³",
    source: "open-meteo",
    observed_at: observedAt,
    rows: rows.map(({ gu_nm, pm25 }) => ({ gu_nm, value: pm25 })),
    freshness: classifyFreshness(observedAt),
    note: "Open-Meteo keyless fallback",
  };
}
