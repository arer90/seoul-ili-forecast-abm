// simulation/scripts/bench_seir_wasm.mjs
// ============================================================================
// BENCH — Rust/WASM Metapop SEIR runner (node target)
//
// Mirror of bench_seir_python.py.  The same Seoul 25-gu populations, the same
// row-stochastic mobility matrix (stay_home=0.55), the same disease params
// (R0=1.4, γ=1/3.5, σ=1/2, ifr=0.001, vax_rate=0, vax_eff=0.5), the same
// initial seed (100 infected in Gangnam = index 22), same dt=0.25, same 365
// day horizon.
//
// Run with:
//   node simulation/scripts/bench_seir_wasm.mjs --n 10 --warmup 1
//
// Writes simulation/results/bench_seir_wasm.json in a schema that matches
// bench_seir_python.json so the meta-runner can join them.
// ============================================================================

import { performance } from "node:perf_hooks";
import { writeFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import {
  DiseaseParams,
  Intervention,
  run_seir_metapop,
} from "../../_past/seir-wasm/pkg-node/seir_wasm.js";

// ------------------------------------------------------------------ paths
const __filename = fileURLToPath(import.meta.url);
const __dirname  = dirname(__filename);
const ROOT       = resolve(__dirname, "..", "..");
const OUT_JSON   = resolve(ROOT, "simulation", "results", "bench_seir_wasm.json");

// ------------------------------------------------------------------ CLI
function parseArgs() {
  const a = { n: 10, warmup: 1, days: 365, dt: 0.25 };
  const argv = process.argv.slice(2);
  for (let i = 0; i < argv.length; i++) {
    const k = argv[i];
    if (k === "--n")       a.n      = parseInt(argv[++i], 10);
    else if (k === "--warmup") a.warmup = parseInt(argv[++i], 10);
    else if (k === "--days")   a.days   = parseInt(argv[++i], 10);
    else if (k === "--dt")     a.dt     = parseFloat(argv[++i]);
  }
  return a;
}

// ------------------------------------------------------------------ scenario
const SEOUL_25GU_NAMES = [
  "Jongno", "Jung", "Yongsan", "Seongdong", "Gwangjin",
  "Dongdaemun", "Jungnang", "Seongbuk", "Gangbuk", "Dobong",
  "Nowon", "Eunpyeong", "Seodaemun", "Mapo", "Yangcheon",
  "Gangseo", "Guro", "Geumcheon", "Yeongdeungpo", "Dongjak",
  "Gwanak", "Seocho", "Gangnam", "Songpa", "Gangdong",
];

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

function buildParams() {
  // DiseaseParams(r0, incubation_days, infectious_days, cfr, vax_rate, vax_eff)
  // matches the Python DiseaseParams(R0=1.4, σ=1/2 → inc=2d, γ=1/3.5 → inf=3.5d,
  // ifr=0.001, vax_rate=0, VE=0.5).
  return new DiseaseParams(1.4, 2.0, 3.5, 0.001, 0.0, 0.5);
}

function buildIntervention() {
  // Default = no policy effect (matches interventions=[] on the Python side).
  return new Intervention();
}

// ------------------------------------------------------------------ bench
function timeOneRun(days, dt) {
  const params       = buildParams();
  const intervention = buildIntervention();
  const mobility     = buildMobility(SEOUL_25GU_POPS.length);

  const t0 = performance.now();
  const res = run_seir_metapop(
    SEOUL_25GU_POPS,
    22,                 // Gangnam index
    100.0,              // initial infected
    mobility,
    params,
    intervention,
    days,
    dt,
  );
  const t1 = performance.now();

  const totalI = res.get_total_i();
  let peak = 0;
  for (let k = 0; k < totalI.length; k++) if (totalI[k] > peak) peak = totalI[k];

  // Dispose native handles — wasm-bindgen doesn't GC them.
  res.free();
  params.free();
  intervention.free();

  return { wallMs: t1 - t0, peakI: peak };
}

// ------------------------------------------------------------------ main
function main() {
  const args = parseArgs();
  console.log(`Building params — 25 gu, ${args.days} days, dt=${args.dt}`);

  for (let i = 0; i < args.warmup; i++) timeOneRun(args.days, args.dt);
  console.log(`Warm-up x ${args.warmup} done`);

  const wallMsList = [];
  const peaks      = [];
  const rssBefore  = process.memoryUsage().rss;
  let peakRss      = rssBefore;

  for (let i = 0; i < args.n; i++) {
    const { wallMs, peakI } = timeOneRun(args.days, args.dt);
    wallMsList.push(wallMs);
    peaks.push(peakI);
    peakRss = Math.max(peakRss, process.memoryUsage().rss);
    console.log(
      `  [${String(i + 1).padStart(2, "0")}/${args.n}]  wall=${wallMs.toFixed(1)} ms  peak_I=${peakI.toLocaleString(undefined, { maximumFractionDigits: 0 })}`,
    );
  }

  wallMsList.sort((a, b) => a - b);
  const median = wallMsList[Math.floor(wallMsList.length / 2)];
  const p05    = wallMsList[Math.max(0, Math.floor(0.05 * args.n))];
  const p95    = wallMsList[Math.min(args.n - 1, Math.floor(0.95 * args.n))];
  const peakMean = peaks.reduce((a, b) => a + b, 0) / peaks.length;
  const peakVar  = peaks.length > 1
    ? peaks.reduce((s, v) => s + (v - peakMean) ** 2, 0) / (peaks.length - 1)
    : 0;
  const peakStd  = Math.sqrt(peakVar);

  const report = {
    impl: "rust_wasm_nodejs",
    runtime: `node ${process.version}`,
    scenario: {
      n_gu: SEOUL_25GU_POPS.length,
      days: args.days,
      dt: args.dt,
      R0: 1.4,
      gamma_days: 3.5,
      sigma_days: 2.0,
      initial_infected_total: 100.0,
      initial_infected_gu: "Gangnam",
    },
    n_runs: args.n,
    n_warmup: args.warmup,
    wall_ms: {
      median,
      p05,
      p95,
      min: wallMsList[0],
      max: wallMsList[wallMsList.length - 1],
      all: wallMsList,
    },
    sanity: {
      peak_I_mean:  peakMean,
      peak_I_stdev: peakStd,
    },
    memory_mb: {
      rss_before: Math.round(rssBefore / 1024 / 1024 * 10) / 10,
      rss_peak:   Math.round(peakRss   / 1024 / 1024 * 10) / 10,
    },
  };

  mkdirSync(dirname(OUT_JSON), { recursive: true });
  writeFileSync(OUT_JSON, JSON.stringify(report, null, 2));

  console.log("");
  console.log(`=== Rust/WASM SEIR-V-D bench  (n=${args.n}, warmup=${args.warmup}) ===`);
  console.log(`  wall_ms median : ${median.toFixed(1)}  (${p05.toFixed(1)} - ${p95.toFixed(1)})`);
  console.log(`  peak_I mean    : ${peakMean.toLocaleString(undefined, { maximumFractionDigits: 0 })}`);
  console.log(`  peak RSS       : ${(peakRss / 1024 / 1024).toFixed(1)} MB`);
  console.log(`  -> ${OUT_JSON}`);
}

main();
