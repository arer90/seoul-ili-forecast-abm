/**
 * Smoke tests for ili-forecast.json — verifies the production-refit-forecast JSON
 * has 25 Seoul gu entries and the expected champion model name.
 *
 * Run from the project root:
 *   node web/lib/live-overlays/__tests__/ili-forecast.test.mjs
 */

import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import { join, dirname } from "path";

const __dir = dirname(fileURLToPath(import.meta.url));
const FORECAST_PATH = join(
  __dir,
  "../../../../web/public/aggregates/ili-forecast.json",
);

const EXPECTED_GU_COUNT = 25;
// Champion from the 53-model run sorted by test_r2 (best-WIS proxy).
const EXPECTED_CHAMPION = "NegBinGLM";

const SEOUL_GU = [
  "강남구", "강동구", "강북구", "강서구", "관악구",
  "광진구", "구로구", "금천구", "노원구", "도봉구",
  "동대문구", "동작구", "마포구", "서대문구", "서초구",
  "성동구", "성북구", "송파구", "양천구", "영등포구",
  "용산구", "은평구", "종로구", "중구", "중랑구",
];

// ── Test runner ────────────────────────────────────────────────────────
let pass = 0, fail = 0;

function assert(condition, label) {
  if (condition) { pass++; console.log(`  OK  ${label}`); }
  else           { fail++; console.error(`FAIL  ${label}`); }
}

function assertEqual(a, b, label) {
  assert(a === b, `${label}: expected="${b}" got="${a}"`);
}

// ── Load JSON ─────────────────────────────────────────────────────────
let d;
try {
  d = JSON.parse(readFileSync(FORECAST_PATH, "utf-8"));
  console.log(`\n[ili-forecast] loaded ${FORECAST_PATH}`);
} catch (e) {
  console.error(`FAIL  could not load ili-forecast.json: ${e.message}`);
  process.exit(1);
}

// ── Top-level fields ──────────────────────────────────────────────────
console.log("\n[ili-forecast] top-level fields");

// source = 'production-refit-forecast' (build_production_forecast.py:725) — 운영 forecast
// (stable-median + conformal + surge-gate). 옛 'model-forecast' 라벨에서 정정(소스 drift fix).
assertEqual(d.source, "production-refit-forecast", "source == 'production-refit-forecast'");
assertEqual(d.model, EXPECTED_CHAMPION, `model == '${EXPECTED_CHAMPION}'`);
assert(typeof d.observed_at === "string" && d.observed_at.length >= 10, "observed_at is a date string");
assert(d.horizon_weeks === 1, "horizon_weeks == 1");
assert(typeof d.note === "string" && d.note.length > 0, "note is non-empty string");
assert(typeof d.gu === "object" && d.gu !== null, "gu is an object");

// ── Per-gu count and structure ────────────────────────────────────────
console.log("\n[ili-forecast] per-gu entries");

const guKeys = Object.keys(d.gu);
assert(guKeys.length === EXPECTED_GU_COUNT, `25 gu entries (got ${guKeys.length})`);

// Verify all canonical Seoul gu names are present
const missingGu = SEOUL_GU.filter((gu) => !(gu in d.gu));
assert(missingGu.length === 0, `all 25 Seoul gu present (missing: ${missingGu.join(", ")})`);

// Sample: each gu has ili, lo, hi as finite numbers
let structureOk = true;
for (const gu of SEOUL_GU) {
  const entry = d.gu[gu];
  if (
    typeof entry?.ili !== "number" || !isFinite(entry.ili) ||
    typeof entry?.lo  !== "number" || !isFinite(entry.lo)  ||
    typeof entry?.hi  !== "number" || !isFinite(entry.hi)
  ) {
    structureOk = false;
    console.error(`  gu "${gu}" has invalid ili/lo/hi: ${JSON.stringify(entry)}`);
  }
}
assert(structureOk, "every gu has finite {ili, lo, hi}");

// lo ≤ ili ≤ hi for all gus
let orderOk = true;
for (const gu of SEOUL_GU) {
  const { ili, lo, hi } = d.gu[gu] ?? {};
  if (lo > ili || ili > hi) {
    orderOk = false;
    console.error(`  gu "${gu}" lo/ili/hi order violated: lo=${lo} ili=${ili} hi=${hi}`);
  }
}
assert(orderOk, "lo ≤ ili ≤ hi for all gus");

// ili values are positive (ILI rate ≥ 0)
const negIli = SEOUL_GU.filter((gu) => (d.gu[gu]?.ili ?? -1) < 0);
assert(negIli.length === 0, `no negative ili values (found: ${negIli.join(", ")})`);

// ── Result ─────────────────────────────────────────────────────────────
console.log(`\nResult: ${pass} passed, ${fail} failed`);
if (fail > 0) process.exit(1);
