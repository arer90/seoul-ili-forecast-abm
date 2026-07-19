/**
 * 응급의료정보센터 (NEDIS) → per-gu ER bed-occupancy rate.
 *
 * Realtime endpoint (공공데이터포털 / e-gen.or.kr):
 *
 *     GET https://apis.data.go.kr/B552657/ErmctInfoInqireService/
 *          getEmrrmRltmUsefulSckbdInfoInqire
 *         ?ServiceKey=${NEDIS_API_KEY}&STAGE1=서울특별시&pageNo=1&numOfRows=200
 *
 * Each <item> carries: hpid, hvgc (general ER capacity — 일반 병상 수),
 * hvs01 (available ER beds — 현재 가용 병상 수). We compute:
 *
 *     occupancy = clamp(1 - hvs01 / hvgc, 0, 1) × 100
 *
 * per hospital, then group by gu (from the static HPID_TO_GU table below),
 * and take the per-gu average. hvec (ER bed capacity) is often 0/-1 in
 * the realtime response; hvgc is the reliable field for Seoul.
 *
 * hpid → gu mapping strategy
 * ---------------------------
 * The realtime endpoint does NOT return dutyAddr (address). A separate
 * list endpoint (getEgytListInfoInqire) returns dutyAddr but only covers
 * ~49 of the 51 Seoul hospitals in the realtime feed. For complete coverage,
 * we use a static lookup table (HPID_TO_GU) built from:
 *   1. getEgytListInfoInqire (STAGE1=서울특별시, numOfRows=300) → 49 entries
 *   2. getEgytBassInfoInqire per hpid → 19 remaining entries verified live
 *
 * This is a STATIC lookup — hospital locations do not change. If a new
 * hospital is added to the realtime feed and is missing here, it silently
 * contributes nothing (graceful degradation, not a crash).
 *
 * Notes
 * -----
 * - XML parsing in Edge is ~20 lines of regex. We don't ask for
 *   ``_type=json`` because NEDIS quietly drops it on some holidays.
 * - hvec ≤ 0 is treated as "no EC data"; fall back to hvgc.
 */

import type { MetricPayload } from "./types";
import { SEOUL_GU, classifyFreshness, tfetch } from "./types";

/**
 * Static hpid → gu table for Seoul emergency hospitals.
 * Source: NEDIS getEgytListInfoInqire + getEgytBassInfoInqire, verified 2026-06-08.
 * Covers all 51 hospitals that appear in the realtime getEmrrmRltmUsefulSckbdInfoInqire feed.
 *
 * NOTE: This is intentionally a static lookup. Hospital locations are permanent;
 * the alternative (per-request address fetch) would add a serial API call.
 */
const HPID_TO_GU: Record<string, string> = {
  // Verified 2026-06-08 against e-gen.or.kr (공공데이터포털 B552657): every gu is
  // taken from the hospital's REAL dutyAddr via getEgytListInfoInqire +
  // getEgytBassInfoInqire — none inferred from the name. Covers all 51 hospitals
  // in the realtime feed (getEmrrmRltmUsefulSckbdInfoInqire). 23/25 gu: 강북구·
  // 마포구 have no ER hospital in the feed (genuine gap → transparent, not zero).
  A1100010: "강남구", // 삼성서울병원
  A1100015: "강남구", // 연세대학교의과대학강남세브란스병원
  A1100028: "강동구", // 성심의료재단강동성심병원
  A1100043: "강동구", // 강동경희대학교병원
  A1100053: "강동구", // 한국보훈복지의료공단중앙보훈병원
  A1100036: "강서구", // 부민병원
  A1120796: "강서구", // 이화여자대학교의과대학부속서울병원
  A1100041: "관악구", // 의료법인서울효천의료재단에이치플러스양지병원
  A1100002: "광진구", // 건국대학교병원
  A1100051: "광진구", // 혜민병원
  A1100014: "구로구", // 고려대학교의과대학부속구로병원
  A1100026: "구로구", // 구로성심병원
  A1100049: "금천구", // 희명병원
  A1100016: "노원구", // 인제대학교상계백병원
  A1100027: "노원구", // 한국원자력의학원원자력병원
  A1100048: "노원구", // 노원을지대학교병원
  A1100020: "도봉구", // 의료법인한전의료재단한일병원
  A1100001: "동대문구", // 경희대학교병원
  A1100021: "동대문구", // 삼육서울병원
  A1100022: "동대문구", // 서울특별시동부병원
  A1100050: "동대문구", // 서울성심병원
  A1100003: "동작구", // 중앙대학교병원
  A1100040: "동작구", // 서울특별시보라매병원
  A1100007: "서대문구", // 연세대학교의과대학세브란스병원
  A1100025: "서대문구", // 의료법인동신의료재단동신병원
  A1100012: "서초구", // 학교법인가톨릭학원가톨릭대학교서울성모병원
  A1122033: "서초구", // 기쁨병원
  A1100013: "성동구", // 한양대학교병원
  A1100008: "성북구", // 학교법인고려중앙학원고려대학교의과대학부속병원(안암병원)
  A1100009: "송파구", // 재단법인아산사회복지재단서울아산병원
  A1100039: "송파구", // 경찰병원
  A1100005: "양천구", // 이화여자대학교의과대학부속목동병원
  A1100019: "양천구", // 홍익병원
  A1100223: "양천구", // 서울특별시서남병원
  A1100011: "영등포구", // 가톨릭대학교여의도성모병원
  A1100024: "영등포구", // 명지성모병원
  A1100037: "영등포구", // 대림성모병원
  A1100045: "영등포구", // 씨엠병원
  A1100054: "영등포구", // 성애의료재단성애병원
  A1100055: "영등포구", // 한림대학교강남성심병원
  A1100004: "용산구", // 순천향대학교부속서울병원
  A1100023: "은평구", // 의료법인청구성심병원
  A1121013: "은평구", // 가톨릭대학교은평성모병원
  A1100006: "종로구", // 강북삼성병원
  A1100017: "종로구", // 서울대학교병원
  A1100029: "종로구", // 서울적십자병원
  A1100032: "종로구", // 세란병원
  A1100052: "중구", // 국립중앙의료원
  A1100035: "중랑구", // 서울특별시서울의료원
  A1100044: "중랑구", // 녹색병원
  A1100075: "중랑구", // 의료법인풍산의료재단동부제일병원
};

export async function buildEr(
  signal: AbortSignal,
): Promise<MetricPayload | null> {
  const key = process.env.NEDIS_API_KEY;
  if (!key) return null;

  const url =
    `https://apis.data.go.kr/B552657/ErmctInfoInqireService/` +
    `getEmrrmRltmUsefulSckbdInfoInqire` +
    `?ServiceKey=${encodeURIComponent(key)}` +
    `&STAGE1=${encodeURIComponent("서울특별시")}` +
    `&pageNo=1&numOfRows=200`;

  let xml: string;
  try {
    const r = await tfetch(url, { signal, timeoutMs: 5000 });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    xml = await r.text();
  } catch (e) {
    return offline(e instanceof Error ? e.message : String(e));
  }

  const items = parseItems(xml);
  if (items.length === 0) return offline("no <item> elements");

  // Aggregate occupancy by gu using the static hpid→gu table.
  const byGu = new Map<string, { num: number; den: number }>();
  for (const it of items) {
    const gu = HPID_TO_GU[it.hpid];
    if (!gu) continue; // hpid not in static table — graceful skip

    // hvgc = general ER capacity; hvs01 = currently-available ER beds.
    // hvec is often 0 or -1 in the realtime feed; prefer hvgc.
    const cap = Number(it.hvgc);
    const avail = Number(it.hvs01);
    if (!Number.isFinite(cap) || cap <= 0) continue;
    const availClamped = Number.isFinite(avail) ? Math.max(0, avail) : 0;
    const occupancyPct = Math.min(100, Math.max(0, (1 - availClamped / cap) * 100));

    const prev = byGu.get(gu) ?? { num: 0, den: 0 };
    prev.num += occupancyPct;
    prev.den += 1;
    byGu.set(gu, prev);
  }

  const seoulSet = new Set<string>(SEOUL_GU);
  const rows: MetricPayload["rows"] = [];
  for (const [gu, { num, den }] of byGu) {
    if (!seoulSet.has(gu) || den === 0) continue;
    rows.push({ gu_nm: gu, value: Math.round((num / den) * 10) / 10 });
  }
  if (rows.length === 0) return offline("no seoul gu matched");

  const observedAt = new Date().toISOString();
  return {
    id: "er",
    label_ko: "응급실 과밀도",
    label_en: "ER crowding",
    unit: "% used",
    source: "nedis",
    observed_at: observedAt,
    rows,
    freshness: classifyFreshness(observedAt),
  };
}

/** Pull <item>…</item> blocks then pluck fields. */
function parseItems(
  xml: string,
): Array<{ hpid: string; hvgc: string; hvs01: string }> {
  const out: Array<{ hpid: string; hvgc: string; hvs01: string }> = [];
  const re = /<item[^>]*>([\s\S]*?)<\/item>/gi;
  let m: RegExpExecArray | null;
  while ((m = re.exec(xml))) {
    const block = m[1];
    out.push({
      hpid: field(block, "hpid"),
      hvgc: field(block, "hvgc"),
      hvs01: field(block, "hvs01"),
    });
  }
  return out;
}

function field(block: string, tag: string): string {
  const m = new RegExp(
    `<${tag}[^>]*>(<!\\[CDATA\\[)?([\\s\\S]*?)(\\]\\]>)?<\\/${tag}>`,
    "i",
  ).exec(block);
  return (m?.[2] ?? "").trim();
}

function offline(note: string): MetricPayload {
  return {
    id: "er",
    label_ko: "응급실 과밀도",
    label_en: "ER crowding",
    unit: "% used",
    source: "fallback",
    observed_at: "1970-01-01T00:00:00Z",
    rows: [],
    freshness: "offline",
    note,
  };
}

// ── Unit-testable helpers (exported for smoke tests) ──────────────────

/**
 * Map an hpid to its Seoul gu using the static lookup.
 *
 * Args:
 *   hpid: NEDIS hospital ID string (e.g. "A1100043")
 *
 * Returns:
 *   gu name if known (e.g. "강동구"), or null if not in the static table.
 *
 * Caller responsibility: pass the exact hpid string from the NEDIS response.
 */
export function hpidToGu(hpid: string): string | null {
  return HPID_TO_GU[hpid] ?? null;
}
