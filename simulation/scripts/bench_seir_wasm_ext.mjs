// simulation/scripts/bench_seir_wasm_ext.mjs
// ============================================================================
// EXTENDED BENCH — Rust/WASM SEIR. Matches bench_seir_python_ext.py:
//   (1) Cold-start: time from script entry → first run complete.
//   (2) Scaling: 90 / 180 / 365 / 730 day horizons.
//   (3) Memory delta per run (RSS before/after).
//   (4) N=30 timed warm runs.
// ============================================================================

import { performance } from "node:perf_hooks";
import { writeFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname  = dirname(__filename);
const ROOT       = resolve(__dirname, "..", "..");
const OUT_JSON   = resolve(ROOT, "simulation", "results", "bench_seir_wasm_ext.json");

// Cold-start: measure import cost too.
const t_entry = performance.now();
const mod = await import("../../_past/seir-wasm/pkg-node/seir_wasm.js");
const t_import_done = performance.now();
const importMs = t_import_done - t_entry;
const { DiseaseParams, Intervention, run_seir_metapop } = mod;

// ------------------------------------------------------------------ scenario
const SEOUL_25GU_POPS = new Float64Array([
  150_000, 125_000, 230_000, 290_000, 345_000,
  335_000, 390_000, 430_000, 300_000, 320_000,
  515_000, 470_000, 310_000, 370_000, 450_000,
  570_000, 395_000, 230_000, 370_000, 390_000,
  495_000, 410_000, 540_000, 660_000, 440_000,
]);

function buildMobility(nGu, stayHome = 0.55) {
  const off = (1.0 - stayHome) / (nGu - 1);
  const M = new Float64Array(nGu * nGu);
  for (let i = 0; i < nGu; i++) {
    for (let j = 0; j < nGu; j++) {
      M[i * nGu + j] = (i === j) ? stayHome : off;
    }
  }
  return M;
}

function runOnce(days, dt) {
  const params       = new DiseaseParams(1.4, 2.0, 3.5, 0.001, 0.0, 0.5);
  const intervention = new Intervention();
  const mobility     = buildMobility(SEOUL_25GU_POPS.length);

  const t0 = performance.now();
  const res = run_seir_metapop(
    SEOUL_25GU_POPS, 22, 100.0, mobility, params, intervention, days, dt,
  );
  const t1 = performance.now();

  const totalI = res.get_total_i();
  let peak = 0;
  for (let k = 0; k < totalI.length; k++) if (totalI[k] > peak) peak = totalI[k];

  res.free(); params.free(); intervention.free();
  return { wallMs: t1 - t0, peak };
}

function median(arr) {
  const s = arr.slice().sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}
function stdev(arr) {
  if (arr.length < 2) return 0;
  const m = arr.reduce((a, b) => a + b, 0) / arr.length;
  const v = arr.reduce((s, x) => s + (x - m) ** 2, 0) / (arr.length - 1);
  return Math.sqrt(v);
}

function benchHorizon(days, n) {
  if (globalThis.gc) globalThis.gc();
  const rssBefore = process.memoryUsage().rss;

  // Cold-ish: first call (after import) — not a true process-fresh cold
  // but still captures the first-invocation cost including any wasm-bindgen
  // lazy-init.
  const cold = runOnce(days, 0.25);

  const wall = [];
  const peaks = [cold.peak];
  let peakRss = rssBefore;
  for (let i = 0; i < n; i++) {
    const { wallMs, peak } = runOnce(days, 0.25);
    wall.push(wallMs);
    peaks.push(peak);
    const cur = process.memoryUsage().rss;
    if (cur > peakRss) peakRss = cur;
  }
  wall.sort((a, b) => a - b);
  return {
    days,
    n_warm_runs: n,
    cold_ms: cold.wallMs,
    warm_median_ms: median(wall),
    warm_p05_ms:    wall[Math.max(0, Math.floor(0.05 * n))],
    warm_p95_ms:    wall[Math.min(n - 1, Math.floor(0.95 * n))],
    warm_min_ms:    wall[0],
    warm_max_ms:    wall[wall.length - 1],
    warm_stdev_ms:  stdev(wall),
    warm_all_ms:    wall,
    peak_I_mean:    peaks.reduce((a, b) => a + b, 0) / peaks.length,
    rss_before_mb:  Math.round(rssBefore / 1024 / 1024 * 10) / 10,
    rss_peak_mb:    Math.round(peakRss   / 1024 / 1024 * 10) / 10,
    rss_delta_mb:   Math.round((peakRss - rssBefore) / 1024 / 1024 * 10) / 10,
  };
}

// ------------------------------------------------------------------ main
const args = { n: 30, horizons: "90,180,365,730" };
const argv = process.argv.slice(2);
for (let i = 0; i < argv.length; i++) {
  if (argv[i] === "--n")        args.n        = parseInt(argv[++i], 10);
  if (argv[i] === "--horizons") args.horizons = argv[++i];
}

const horizons = args.horizons.split(",").map((x) => parseInt(x, 10));
console.log(`import time (pkg-node/seir_wasm.js): ${importMs.toFixed(1)} ms`);
console.log(`horizons: ${horizons.join(",")}  n_warm=${args.n}`);

const results = [];
for (const H of horizons) {
  console.log(`--- horizon = ${H} days ---`);
  const r = benchHorizon(H, args.n);
  console.log(`  cold=${r.cold_ms.toFixed(1)} ms   warm median=${r.warm_median_ms.toFixed(1)} ms ` +
              `(p5=${r.warm_p05_ms.toFixed(1)} p95=${r.warm_p95_ms.toFixed(1)})   ` +
              `RSSd=${r.rss_delta_mb >= 0 ? "+" : ""}${r.rss_delta_mb.toFixed(1)} MB  peak_I=${r.peak_I_mean.toLocaleString()}`);
  results.push(r);
}

const report = {
  impl: "rust_wasm_nodejs",
  runtime_info: {
    node: process.version,
    wasm_target: "nodejs",
    wasm_opt: "disabled (wasm-pack 0.14 bundled wasm-opt too old for bulk-memory)",
  },
  cold_import_ms: importMs,
  horizons: results,
  scenario_fixed: {
    n_gu: 25, dt: 0.25, R0: 1.4, gamma_days: 3.5,
    sigma_days: 2.0, initial_infected: 100, seed_gu: "Gangnam",
    stay_home: 0.55, mobility: "uniform off-diagonal",
  },
};

mkdirSync(dirname(OUT_JSON), { recursive: true });
writeFileSync(OUT_JSON, JSON.stringify(report, null, 2));
console.log(`-> ${OUT_JSON}`);
