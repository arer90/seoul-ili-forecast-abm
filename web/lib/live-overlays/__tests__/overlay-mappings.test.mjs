/**
 * Smoke tests for hpid→gu and station→gu mapping helpers.
 * Run with: node --experimental-vm-modules lib/live-overlays/__tests__/overlay-mappings.test.mjs
 * (Or via npm test once a test runner is configured.)
 *
 * These tests verify correctness of the static lookup tables without
 * requiring any network calls or API keys.
 */

// ── Inline copies of the mapping logic (avoids TS compilation in CI) ──

const HPID_TO_GU = {
  A1100043: "강동구",   // 강동경희대학교병원
  A1100014: "구로구",   // 고려대구로병원
  A1100013: "성동구",   // 한양대병원
  A1100008: "성북구",   // 고려대안암병원
  A1100005: "양천구",   // 이화여자대목동병원
  A1100017: "종로구",   // 서울대병원
  A1100003: "동작구",   // 중앙대병원
  A1100009: "송파구",   // 서울아산병원
  A1100012: "서초구",   // 가톨릭서울성모병원
  A1100010: "강남구",   // 삼성서울병원
  A1100007: "서대문구", // 세브란스병원
  A1100002: "광진구",   // 건국대병원
  A1100004: "용산구",   // 순천향대서울병원
  A1121013: "은평구",   // 은평성모병원
  A1100006: "종로구",   // 강북삼성병원
  A1100001: "동대문구", // 경희대병원
  A1100040: "동작구",   // 보라매병원
  A1100035: "중랑구",   // 서울의료원
  A1100011: "영등포구", // 여의도성모병원
  A1100016: "노원구",   // 상계백병원
  A1100048: "노원구",   // 노원을지대병원
  A1100027: "노원구",   // 원자력병원
  A1100020: "도봉구",   // 한일병원
  A1100053: "강동구",   // 중앙보훈병원
  A1100028: "강동구",   // 강동성심병원
  A1100075: "중랑구",   // 동부제일병원
  A1100044: "중랑구",   // 녹색병원
  A1100052: "중구",     // 국립중앙의료원
  A1120796: "강서구",   // 이대서울병원
  A1100055: "영등포구", // 강남성심병원
  A1100049: "금천구",   // 희명병원
  A1100051: "광진구",   // 혜민병원
  A1100023: "은평구",   // 청구성심병원
  A1100019: "양천구",   // 홍익병원
  A1100041: "관악구",   // 에이치플러스양지병원
};

function hpidToGu(hpid) {
  return HPID_TO_GU[hpid] ?? null;
}

const STN_TO_GU_SUBSET = {
  "강남": "강남구",
  "잠실(송파구청)": "송파구",
  "잠실": "송파구",
  "서울역": "중구",
  "시청": "중구",
  "홍대입구": "마포구",
  "교대(법원.검찰청)": "서초구",
  "교대": "서초구",
  "서울대입구(관악구청)": "관악구",
  "가산디지털단지": "금천구",
};

function stationToGu(stn) {
  if (STN_TO_GU_SUBSET[stn]) return STN_TO_GU_SUBSET[stn];
  const stripped = stn.replace(/\(.*?\)$/, "").trim();
  if (stripped !== stn && STN_TO_GU_SUBSET[stripped]) return STN_TO_GU_SUBSET[stripped];
  return null;
}

// ── Test runner ────────────────────────────────────────────────────────

let pass = 0, fail = 0;
function assert(condition, label) {
  if (condition) { pass++; console.log(`  OK  ${label}`); }
  else           { fail++; console.error(`FAIL  ${label}`); }
}
function assertEqual(a, b, label) {
  assert(a === b, `${label}: expected="${b}" got="${a}"`);
}

// ── hpidToGu tests ────────────────────────────────────────────────────
console.log("\n[nedis-er] hpidToGu");

assertEqual(hpidToGu("A1100043"), "강동구", "강동경희대 → 강동구");
assertEqual(hpidToGu("A1100003"), "동작구", "중앙대병원 → 동작구");
assertEqual(hpidToGu("A1100009"), "송파구", "서울아산 → 송파구");
assertEqual(hpidToGu("A1100012"), "서초구", "서울성모 → 서초구");
assertEqual(hpidToGu("A1100010"), "강남구", "삼성서울 → 강남구");
assertEqual(hpidToGu("A1120796"), "강서구", "이대서울 → 강서구");
assertEqual(hpidToGu("A1100049"), "금천구", "희명병원 → 금천구");
assertEqual(hpidToGu("XXXUNKNOWN"), null,   "unknown hpid → null");
assertEqual(hpidToGu(""),           null,   "empty string → null");

// ── stationToGu tests ─────────────────────────────────────────────────
console.log("\n[metro-congestion] stationToGu");

// Exact matches
assertEqual(stationToGu("강남"),              "강남구",  "강남 exact");
assertEqual(stationToGu("서울역"),            "중구",    "서울역 exact");
assertEqual(stationToGu("홍대입구"),          "마포구",  "홍대입구 exact");
assertEqual(stationToGu("가산디지털단지"),    "금천구",  "가산디지털단지 exact");

// Parenthetical official names (real portal names)
assertEqual(stationToGu("잠실(송파구청)"),          "송파구", "잠실(송파구청) exact");
assertEqual(stationToGu("교대(법원.검찰청)"),       "서초구", "교대(법원.검찰청) exact");
assertEqual(stationToGu("서울대입구(관악구청)"),    "관악구", "서울대입구(관악구청) exact");

// Parenthetical strip fallback
assertEqual(stationToGu("잠실(송파구청new)"),       "송파구", "잠실(…) strip fallback");
assertEqual(stationToGu("교대(가나다)"),            "서초구", "교대(…) strip fallback");

// Unknown / out-of-Seoul
assertEqual(stationToGu("수원"),     null, "수원 (경기) → null");
assertEqual(stationToGu("UNKNOWN"),  null, "unknown → null");
assertEqual(stationToGu(""),         null, "empty → null");

// ── Result ─────────────────────────────────────────────────────────────
console.log(`\nResult: ${pass} passed, ${fail} failed`);
if (fail > 0) process.exit(1);
