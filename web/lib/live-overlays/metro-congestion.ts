/**
 * Seoul Metro (서울교통공사) 열린데이터광장 → per-gu subway daily ridership,
 * normalised to a 0-100 congestion-proxy score.
 *
 * Endpoint:
 *
 *     GET http://openapi.seoul.go.kr:8088/${KEY}/json/
 *         CardSubwayStatsNew/1/1000/${YYYYMMDD}
 *
 * Response fields per row:
 *   USE_YMD       — date (YYYYMMDD)
 *   SBWY_ROUT_LN_NM — line name (e.g. "2호선")
 *   SBWY_STNS_NM  — station name (official, may include parentheticals)
 *   GTON_TNOPE    — boarding count (승차)
 *   GTOFF_TNOPE   — alighting count (하차)
 *   REG_YMD       — registration date (data has a ~3-7 day lag)
 *
 * Congestion proxy
 * ----------------
 * This endpoint provides daily totals, not real-time congestion percent.
 * We treat (boarding + alighting) as a ridership score, normalise each
 * station by the city-wide maximum (서울역 ≈ 240k), then average per gu.
 * Scale: 0 = empty station, 100 = 서울역-level ridership.
 *
 * Endpoint notes
 * ---------------
 * - The :8088 port uses TLS 1.0 which macOS LibreSSL rejects. The Edge
 *   runtime (Vercel / Node.js) handles it fine. HTTP (not HTTPS) also
 *   works and avoids the TLS issue: use ``http://`` for this URL.
 * - Date lag: data for day D typically appears around D+3 to D+7.
 *   We scan backwards up to 10 days to find the most-recent available date.
 * - Key: uses METRO_API_KEY (= data.seoul.go.kr 열린데이터광장 key).
 *   SEOUL_OPENAPI_KEY is accepted too (same portal).
 *
 * Station → gu mapping
 * --------------------
 * STN_TO_GU covers 70+ high-ridership Seoul stations (>40k daily).
 * Stations outside this map are silently skipped (graceful degradation).
 * Names use the official portal strings, including parentheticals.
 */

import type { MetricPayload } from "./types";
import { SEOUL_GU, classifyFreshness, tfetch } from "./types";

/**
 * Station name → Seoul gu lookup.
 *
 * Names must match exactly the SBWY_STNS_NM values in the
 * CardSubwayStatsNew response (official Seoul Metro portal names).
 * Verified against the 2026-06-01 dataset (529 unique station names).
 *
 * NOTE: This is a static table — it does not need external API calls.
 * Stations outside Seoul (경기/인천 etc.) are intentionally omitted.
 */
const STN_TO_GU: Record<string, string> = {
  // 강남구
  강남: "강남구",
  선릉: "강남구",
  "삼성(무역센터)": "강남구",
  역삼: "강남구",
  압구정: "강남구",
  "강남구청": "강남구",
  신논현: "강남구",
  개포동: "강남구",
  // 강동구
  천호: "강동구",
  "천호(풍납토성)": "강동구",
  강동: "강동구",
  강일: "강동구",
  // 강북구
  "수유(강북구청)": "강북구",
  "수유역": "강북구",
  미아: "강북구",
  "미아(서울사이버대학)": "강북구",
  미아사거리: "강북구",
  // 강서구
  "마곡나루(서울식물원)": "강서구",
  발산: "강서구",
  화곡: "강서구",
  강서: "강서구",
  까치산: "강서구",
  "김포공항": "강서구",
  // 관악구
  신림: "관악구",
  "서울대입구(관악구청)": "관악구",
  "서울대입구": "관악구",
  "낙성대(강감찬)": "관악구",
  // 광진구
  건대입구: "광진구",
  "구의(광진구청)": "광진구",
  뚝섬: "광진구",
  // 구로구
  구로: "구로구",
  구로디지털단지: "구로구",
  신도림: "구로구",
  "대림(구로구청)": "구로구",
  // 금천구
  가산디지털단지: "금천구",
  // 노원구
  노원: "노원구",
  태릉입구: "노원구",
  "석계역": "노원구",
  석계: "노원구",
  // 도봉구
  창동: "도봉구",
  쌍문: "도봉구",
  // 동대문구
  경희대: "동대문구",
  "동대문역사문화공원(DDP)": "동대문구",
  동대문: "동대문구",
  "청량리(서울시립대입구)": "동대문구",
  청량리: "동대문구",
  회기: "동대문구",
  "회기역": "동대문구",
  "용두(동대문구청)": "동대문구",
  // 동작구
  사당: "동작구",
  노량진: "동작구",
  이수: "동작구",
  흑석: "동작구",
  // 마포구
  홍대입구: "마포구",
  합정: "마포구",
  공덕: "마포구",
  망원: "마포구",
  디지털미디어시티: "마포구",
  // 서대문구
  신촌: "서대문구",
  이대: "서대문구",
  // 서초구
  교대: "서초구",
  "교대(법원.검찰청)": "서초구",
  "양재(서초구청)": "서초구",
  "남부터미널(예술의전당)": "서초구",
  방배: "서초구",
  // 성동구
  왕십리: "성동구",
  "왕십리(성동구청)": "성동구",
  성수: "성동구",
  // 성북구
  "고려대역": "성북구",
  "성신여대입구(돈암)": "성북구",
  돈암: "성북구",
  "보문역": "성북구",
  // 송파구
  "잠실(송파구청)": "송파구",
  잠실새내: "송파구",
  잠실: "송파구",
  "석촌역": "송파구",
  석촌: "송파구",
  수서: "송파구",
  문정: "송파구",
  // 양천구
  목동: "양천구",
  "오목교(목동운동장앞)": "양천구",
  오목교: "양천구",
  신목동: "양천구",
  // 영등포구
  영등포: "영등포구",
  "영등포역": "영등포구",
  "영등포구청": "영등포구",
  여의도: "영등포구",
  당산: "영등포구",
  // 용산구
  용산: "용산구",
  이태원: "용산구",
  "녹사평(용산구청)": "용산구",
  삼각지: "용산구",
  "신용산": "용산구",
  // 은평구
  연신내: "은평구",
  불광: "은평구",
  응암: "은평구",
  // 종로구
  종각: "종로구",
  "광화문(세종문화회관)": "종로구",
  광화문: "종로구",
  종로3가: "종로구",
  안국: "종로구",
  경복궁: "종로구",
  "경복궁(정부서울청사)": "종로구",
  // 중구
  서울역: "중구",
  시청: "중구",
  명동: "중구",
  을지로입구: "중구",
  을지로3가: "중구",
  충무로: "중구",
  "회현(남대문시장)": "중구",
  동대입구: "중구",
  // 중랑구
  상봉: "중랑구",
  "군자(능동)": "중랑구",
};

interface CardSubwayRow {
  USE_YMD?: string;
  SBWY_ROUT_LN_NM?: string;
  SBWY_STNS_NM?: string;
  GTON_TNOPE?: string | number;
  GTOFF_TNOPE?: string | number;
  REG_YMD?: string;
}

interface CardSubwayEnvelope {
  CardSubwayStatsNew?: {
    list_total_count?: number;
    RESULT?: { CODE?: string; MESSAGE?: string };
    row?: CardSubwayRow[];
  };
  RESULT?: { CODE?: string; MESSAGE?: string };
}

/** Normalise ridership to 0–100 congestion proxy.
 * Reference max ≈ 250,000 (서울역 peak); use 300k as ceiling for headroom. */
const RIDERSHIP_SCALE = 300_000;

export async function buildMetro(
  signal: AbortSignal,
): Promise<MetricPayload | null> {
  const key = process.env.METRO_API_KEY ?? process.env.SEOUL_OPENAPI_KEY;
  if (!key) return null;

  // Find most-recent available date (data has a 3-7 day registration lag).
  const date = await findRecentDate(key, signal);
  if (!date) return offline("no recent data available (checked 10 days)");

  const url =
    `http://openapi.seoul.go.kr:8088/${encodeURIComponent(key)}/json/` +
    `CardSubwayStatsNew/1/1000/${date}`;

  let env: CardSubwayEnvelope;
  try {
    const r = await tfetch(url, { signal, timeoutMs: 6000 });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    env = (await r.json()) as CardSubwayEnvelope;
  } catch (e) {
    return offline(e instanceof Error ? e.message : String(e));
  }

  const topLevel = env.RESULT ?? env.CardSubwayStatsNew?.RESULT;
  if (topLevel?.CODE && topLevel.CODE !== "INFO-000") {
    return offline(`${topLevel.CODE}: ${topLevel.MESSAGE ?? ""}`);
  }

  const rows = env.CardSubwayStatsNew?.row ?? [];
  if (rows.length === 0) return offline(`no rows for date ${date}`);

  // Sum (boarding + alighting) per station across all lines.
  const byStation = new Map<string, number>();
  for (const r of rows) {
    const stn = String(r.SBWY_STNS_NM ?? "").trim();
    if (!stn) continue;
    const boarding = Number(r.GTON_TNOPE ?? 0);
    const alighting = Number(r.GTOFF_TNOPE ?? 0);
    const daily = (Number.isFinite(boarding) ? boarding : 0) +
                  (Number.isFinite(alighting) ? alighting : 0);
    byStation.set(stn, (byStation.get(stn) ?? 0) + daily);
  }

  // Aggregate station ridership → gu, then normalise to 0-100 proxy.
  const byGu = new Map<string, { num: number; den: number }>();
  for (const [stn, ridership] of byStation) {
    const gu = stationToGu(stn);
    if (!gu) continue;
    const score = Math.min(100, (ridership / RIDERSHIP_SCALE) * 100);
    const prev = byGu.get(gu) ?? { num: 0, den: 0 };
    prev.num += score;
    prev.den += 1;
    byGu.set(gu, prev);
  }

  const seoulSet = new Set<string>(SEOUL_GU);
  const out: MetricPayload["rows"] = [];
  for (const [gu, { num, den }] of byGu) {
    if (!seoulSet.has(gu) || den === 0) continue;
    out.push({ gu_nm: gu, value: Math.round((num / den) * 10) / 10 });
  }
  if (out.length === 0) return offline("no seoul gu matched");

  // Use the observation date from the data (not today — data lags 3-7 days).
  const observedAt = parseDataDate(date);
  return {
    id: "metro",
    label_ko: "지하철 혼잡도",
    label_en: "Subway ridership",
    unit: "ridership score",
    source: "seoul_metro_tdata",
    observed_at: observedAt,
    rows: out,
    freshness: classifyFreshness(observedAt, 10080, 20160), // 7d / 14d windows
    note: `일별 승하차 기준 (${date}); 실시간 혼잡도 아님`,
  };
}

/**
 * Scan backwards from today to find the most-recent date with available data.
 * Data typically has a 3-7 day registration lag.
 *
 * Args:
 *   key: Seoul OpenAPI key
 *   signal: AbortSignal from the route handler
 *
 * Returns:
 *   YYYYMMDD string if found within 10 days, null otherwise.
 */
async function findRecentDate(
  key: string,
  signal: AbortSignal,
): Promise<string | null> {
  const now = new Date();
  for (let daysAgo = 2; daysAgo <= 12; daysAgo++) {
    const d = new Date(now);
    d.setDate(d.getDate() - daysAgo);
    const dateStr = d.toISOString().slice(0, 10).replace(/-/g, "");
    const url =
      `http://openapi.seoul.go.kr:8088/${encodeURIComponent(key)}/json/` +
      `CardSubwayStatsNew/1/1/${dateStr}`;
    try {
      const r = await tfetch(url, { signal, timeoutMs: 3000 });
      if (!r.ok) continue;
      const env = (await r.json()) as CardSubwayEnvelope;
      const count = env.CardSubwayStatsNew?.list_total_count ?? 0;
      if (count > 0) return dateStr;
    } catch {
      // network error on this attempt — try next day
    }
  }
  return null;
}

/**
 * Convert YYYYMMDD string to ISO timestamp (midnight KST = UTC+9).
 *
 * Args:
 *   yyyymmdd: date string from CardSubwayStatsNew USE_YMD field
 *
 * Returns:
 *   ISO 8601 string (UTC)
 */
function parseDataDate(yyyymmdd: string): string {
  const y = yyyymmdd.slice(0, 4);
  const mo = yyyymmdd.slice(4, 6);
  const d = yyyymmdd.slice(6, 8);
  // KST midnight = UTC 15:00 of previous day
  return new Date(`${y}-${mo}-${d}T15:00:00Z`).toISOString();
}

/**
 * Look up the gu for a station name. Handles both exact and common alias
 * variants. Returns null for stations outside Seoul or not in the table.
 *
 * Args:
 *   stn: station name from SBWY_STNS_NM (may include parentheticals)
 *
 * Returns:
 *   gu name (e.g. "강남구") or null if unknown.
 */
export function stationToGu(stn: string): string | null {
  // 1. Exact match (covers official names including parentheticals).
  if (STN_TO_GU[stn]) return STN_TO_GU[stn];
  // 2. Try stripping parenthetical suffix: "잠실(송파구청)" → "잠실"
  const stripped = stn.replace(/\(.*?\)$/, "").trim();
  if (stripped !== stn && STN_TO_GU[stripped]) return STN_TO_GU[stripped];
  return null;
}

function offline(note: string): MetricPayload {
  return {
    id: "metro",
    label_ko: "지하철 혼잡도",
    label_en: "Subway crowding",
    unit: "ridership score",
    source: "fallback",
    observed_at: "1970-01-01T00:00:00Z",
    rows: [],
    freshness: "offline",
    note,
  };
}
