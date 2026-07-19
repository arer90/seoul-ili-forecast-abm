/**
 * KMA (기상청) ASOS 지상관측 → per-gu temperature + humidity.
 *
 * The Seoul ASOS network has roughly one station per 2-3 gu; we assign
 * each gu to its nearest station centroid via a small lookup table and
 * carry the scalar forward. This is the same trick the Python side uses
 * in ``simulation.collectors.weather_asos.collect_asos``.
 *
 * The API surface we target is the new apihub.kma.go.kr one:
 *
 *     GET https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php
 *         ?tm=YYYYMMDDHH&stn=0&help=0&authKey=${KMA_API_KEY}
 *
 * which returns a fixed-width text table with columns:
 *
 *     # YYMMDDHHMI   STN  WD WS GST_WD GST_WS GST_TM PA PS PT PR TA TD HM ...
 *
 * We only need STN, TA (temp °C), HM (relative humidity %).
 * Missing keys / HTTP failures fall back to keyless Open-Meteo. If
 * that also fails, ``null`` lets the route's static/synthetic fallback
 * fill the layer.
 */

import type { MetricPayload } from "./types";
import { SEOUL_GU, classifyFreshness, tfetch } from "./types";
import { fetchOpenMeteoWeather } from "./open-meteo";

// ── Seoul-area ASOS stations → nearest-gu mapping ────────────────────
// Source: KMA station registry (stn_id / stn_ko / lat / lon) cross-
// referenced with gu centroid distance. Only stations whose nearest
// gu is in Seoul are kept. If a gu has no station within 12 km, we
// fall back to the closest Seoul station (usually Jongno-gu ASOS).

interface StationMap {
  /** ASOS 3-digit station id. */
  stn: number;
  gu: string[];
}

const STN_ASSIGN: StationMap[] = [
  { stn: 108, gu: ["종로구", "중구", "용산구", "동대문구", "성북구", "서대문구", "마포구"] }, // 서울(종로)
  { stn: 109, gu: ["광진구", "성동구", "중랑구"] }, // 동대문 계열 → 대체 station
  { stn: 509, gu: ["강남구", "서초구"] },
  { stn: 510, gu: ["송파구", "강동구"] },
  { stn: 511, gu: ["양천구", "강서구", "영등포구"] },
  { stn: 512, gu: ["구로구", "금천구", "관악구", "동작구"] },
  { stn: 513, gu: ["은평구", "노원구", "도봉구", "강북구"] },
];

/** Flip the assignment so we can look up "gu → stn" in O(1). */
const GU_TO_STN = (() => {
  const m = new Map<string, number>();
  for (const { stn, gu } of STN_ASSIGN) {
    for (const g of gu) m.set(g, stn);
  }
  return m;
})();

interface StationReading {
  stn: number;
  ta: number; // °C
  hm: number; // 0-100
}

export async function buildTempAndHumidity(
  signal: AbortSignal,
): Promise<{ temp: MetricPayload | null; humidity: MetricPayload | null }> {
  const key = process.env.KMA_API_KEY;
  if (!key) return openMeteoFallback();

  // Hour stamp (UTC+9 Seoul) rounded down to the previous whole hour —
  // the ASOS endpoint only publishes top-of-the-hour observations.
  const now = new Date();
  now.setUTCMinutes(0, 0, 0);
  now.setUTCHours(now.getUTCHours() - 1); // last completed hour
  const tm = kstStamp(now);

  const url =
    `https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php?` +
    `tm=${tm}&stn=0&help=0&authKey=${encodeURIComponent(key)}`;

  let txt: string;
  try {
    const r = await tfetch(url, { signal, timeoutMs: 3500 });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    txt = await r.text();
  } catch {
    return openMeteoFallback();
  }

  const readings = parseAsos(txt);
  if (readings.length === 0) {
    return openMeteoFallback();
  }
  const byStn = new Map<number, StationReading>();
  for (const r of readings) byStn.set(r.stn, r);

  const tempRows: MetricPayload["rows"] = [];
  const humRows: MetricPayload["rows"] = [];
  for (const gu of SEOUL_GU) {
    const stn = GU_TO_STN.get(gu) ?? 108; // Seoul ASOS default
    const r = byStn.get(stn) ?? byStn.get(108);
    if (!r) continue;
    tempRows.push({ gu_nm: gu, value: r.ta });
    humRows.push({ gu_nm: gu, value: r.hm });
  }

  const observedAt = asIsoFromKst(now);
  const freshness = classifyFreshness(observedAt);

  return {
    temp: {
      id: "temp",
      label_ko: "기온",
      label_en: "Temperature",
      unit: "°C",
      source: "kma_asos",
      observed_at: observedAt,
      rows: tempRows,
      freshness,
    },
    humidity: {
      id: "humidity",
      label_ko: "습도",
      label_en: "Humidity",
      unit: "%",
      source: "kma_asos",
      observed_at: observedAt,
      rows: humRows,
      freshness,
    },
  };
}

/** Parse the ASOS fixed-width response. Header lines start with ``#``
 *  or are blank; data rows are space-separated with ``-99.0`` /
 *  ``-999`` as "missing" sentinels. */
function parseAsos(txt: string): StationReading[] {
  const out: StationReading[] = [];
  for (const line of txt.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const parts = trimmed.split(/\s+/);
    // Column layout per kma_sfctm2.php docs — STN is at idx 1 (after
    // YYMMDDHHMI), TA at idx 11, HM at idx 13 (0-indexed).
    if (parts.length < 14) continue;
    const stn = Number(parts[1]);
    const ta = Number(parts[11]);
    const hm = Number(parts[13]);
    if (!Number.isFinite(stn)) continue;
    if (!Number.isFinite(ta) || ta <= -90) continue;
    if (!Number.isFinite(hm) || hm < 0) continue;
    out.push({ stn, ta, hm });
  }
  return out;
}

/** YYYYMMDDHH in KST (UTC+9), which is the format the endpoint wants. */
function kstStamp(d: Date): string {
  const k = new Date(d.getTime() + 9 * 60 * 60 * 1000);
  const yyyy = k.getUTCFullYear().toString();
  const mm = (k.getUTCMonth() + 1).toString().padStart(2, "0");
  const dd = k.getUTCDate().toString().padStart(2, "0");
  const hh = k.getUTCHours().toString().padStart(2, "0");
  return `${yyyy}${mm}${dd}${hh}`;
}

/** Round-trip the KST stamp back to a UTC ISO string for the badge. */
function asIsoFromKst(d: Date): string {
  return d.toISOString();
}

async function openMeteoFallback(): Promise<{
  temp: MetricPayload | null;
  humidity: MetricPayload | null;
}> {
  const rows = await fetchOpenMeteoWeather(SEOUL_GU);
  if (!rows) return { temp: null, humidity: null };
  const observedAt = new Date().toISOString();
  const freshness = classifyFreshness(observedAt);
  return {
    temp: {
      id: "temp",
      label_ko: "기온",
      label_en: "Temperature",
      unit: "°C",
      source: "open-meteo",
      observed_at: observedAt,
      rows: rows.map(({ gu_nm, temp }) => ({ gu_nm, value: temp })),
      freshness,
      note: "Open-Meteo keyless fallback",
    },
    humidity: {
      id: "humidity",
      label_ko: "습도",
      label_en: "Humidity",
      unit: "%",
      source: "open-meteo",
      observed_at: observedAt,
      rows: rows.map(({ gu_nm, humidity }) => ({ gu_nm, value: humidity })),
      freshness,
      note: "Open-Meteo keyless fallback",
    },
  };
}
