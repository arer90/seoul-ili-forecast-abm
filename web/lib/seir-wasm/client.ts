/**
 * Browser-side Metapop SEIR-V-D runner, backed by the Rust/WASM build at
 * ``web/lib/seir-wasm/pkg``. The same binary we benchmark against the
 * Python simulator (see ``simulation/results/bench_seir_evaluation.md``)
 * — on desktop Chrome a 365-day / 25-gu run returns in ~27 ms, so the UI
 * can re-simulate on every slider tick without a network round-trip.
 *
 * Contract
 * --------
 * - Pure client module. Never call these helpers from Node / server
 *   components — ``WebAssembly.instantiateStreaming`` needs ``fetch``
 *   and the Edge runtime can't resolve ``/wasm/...`` asset URLs.
 * - ``initPromise`` is a module-scoped singleton so repeated callers
 *   share the same ``WebAssembly.Module``.
 * - ``runWhatIf`` always ``.free()``s its three wasm objects (params,
 *   intervention, result) before returning — the Float64Arrays we hand
 *   back are copies (``.slice()`` on the wasm-bindgen side), so the
 *   caller can keep them as long as they want.
 *
 * Numbers match ``simulation.sim.metapop_seirvd.MetapopSEIRVD`` to
 * within 0.007% on peak_I (floating-point roundoff only) as long as you
 * use the identical FOI form — which this binary does, post-Step-2.
 */

// The default export is the async ``__wbg_init`` function; the named
// exports are the public Rust surface. We alias the default so callers
// only see ``initSeirWasm``.
import __wbg_init, {
  DiseaseParams,
  Intervention,
  run_seir_metapop,
  type SimResult,
} from "./pkg/seir_wasm.js";

/** Static aggregate shape written by ``simulation.scripts.export_seir_metapop_init``. */
export interface SeirMetapopInit {
  district_names: string[];
  populations: number[];
  mobility_flat: number[];
  n_gu: number;
  source: string;
  generated_at: string;
}

/** Sliders the user controls in the what-if card. */
export interface WhatIfInputs {
  /** Basic reproduction number before interventions (1.0–4.0). */
  r0: number;
  /** Fraction of residents staying in home gu during daytime (0.3–0.95).
   *  Overrides the diagonal of the base mobility matrix — off-diagonals
   *  rescale proportionally so each row still sums to 1. */
  stayHome: number;
  /** Horizon in days. Typical: 90 / 180 / 365 / 730. */
  days: number;
  /** Seeded infectives dropped into ``seedGuIdx`` at t=0. */
  seedInfected: number;
  /** Index into ``district_names`` (0..24) where the epidemic starts. */
  seedGuIdx: number;
  /** Optional social-distancing effect (0..1) — 0 disables. */
  distancingEffect?: number;
  /** Intervention start day (only used when distancingEffect > 0). */
  npiStartDay?: number;
  /** Intervention end day. */
  npiEndDay?: number;
}

/** Summary series returned to the UI. All arrays are ``n_days+1`` long. */
export interface WhatIfOutputs {
  /** Sum of I(t) across all 25 gu. */
  totalI: Float64Array;
  /** Sum of S(t) — handy for attack-rate context. */
  totalS: Float64Array;
  /** Daily new infections aggregated to seoul-wide. */
  totalNew: Float64Array;
  /** Effective reproduction number per day. */
  reT: Float64Array;
  /** Elapsed ms from ``run_seir_metapop`` call to completion. */
  elapsedMs: number;
  /** Peak I value + the day it occurred. */
  peakI: number;
  peakDay: number;
  /**
   * Per-gu infectious-compartment time series, in the SAME order as
   * ``SeirMetapopInit.district_names``. Each entry is ``n_days+1`` long.
   * The map player uses these for the weekly choropleth animation.
   */
  perGuI: Float64Array[];
  /**
   * Per-gu *daily new infections* — more visually informative than I(t)
   * for animating spread because it isolates the wave front instead of
   * the cumulative infectious pool. Same shape as ``perGuI``.
   */
  perGuNew: Float64Array[];
}

// ── Singletons ─────────────────────────────────────────────────────────

let initPromise: Promise<void> | null = null;
let initCache: SeirMetapopInit | null = null;

/** Resolves the first time, then returns the cached promise for warm calls. */
export function initSeirWasm(): Promise<void> {
  if (initPromise) return initPromise;
  initPromise = (async () => {
    // Call with no args and let the wasm-bindgen glue resolve
    // ``new URL('seir_wasm_bg.wasm', import.meta.url)`` itself. Webpack
    // recognises that pattern, hashes the sibling ``seir_wasm_bg.wasm``,
    // and emits it under ``/_next/static/media/`` — so the request is
    // cache-busted per deploy automatically.
    //
    // A copy of the same binary also lives at ``/wasm/seir_wasm_bg.wasm``
    // (via ``web/public/wasm``) for stable-URL scenarios such as manual
    // prefetch or curl smoke tests.
    await __wbg_init();
  })();
  return initPromise;
}

/** Fetch + memoise the static (pops, M, district_names) blob. */
export async function loadMetapopInit(): Promise<SeirMetapopInit> {
  if (initCache) return initCache;
  const res = await fetch("/aggregates/seir-metapop-init.json", {
    cache: "force-cache",
  });
  if (!res.ok) {
    throw new Error(
      `seir-metapop-init.json fetch failed: HTTP ${res.status} ` +
        `${res.statusText}`,
    );
  }
  const json = (await res.json()) as SeirMetapopInit;
  if (
    !Array.isArray(json.district_names) ||
    !Array.isArray(json.populations) ||
    !Array.isArray(json.mobility_flat) ||
    json.district_names.length !== json.n_gu ||
    json.populations.length !== json.n_gu ||
    json.mobility_flat.length !== json.n_gu * json.n_gu
  ) {
    throw new Error("seir-metapop-init.json shape invalid");
  }
  initCache = json;
  return json;
}

// ── Mobility override ───────────────────────────────────────────────────

/**
 * Apply the stay-home slider on top of a row-stochastic base matrix.
 *
 * We keep the *shape* of the base commuter matrix (so Gangnam→Seocho is
 * still the biggest off-diagonal flow), but rescale so that each
 * diagonal entry equals ``stayHome`` and each row still sums to 1.
 *
 *   new_diag[i]    = stayHome
 *   new_off[i,j]   = (1 - stayHome) * base_off[i,j] / sum_j' base_off[i,j']
 *
 * When ``base_off`` sums to zero for some row (isolated gu), we spread
 * the remainder uniformly across the other 24 gu so the row-sum stays
 * exactly 1.
 */
function applyStayHome(
  baseFlat: number[],
  nGu: number,
  stayHome: number,
): Float64Array {
  const out = new Float64Array(nGu * nGu);
  for (let i = 0; i < nGu; i++) {
    let offSum = 0;
    for (let j = 0; j < nGu; j++) {
      if (i === j) continue;
      offSum += baseFlat[i * nGu + j] ?? 0;
    }
    const budget = Math.max(0, 1 - stayHome);
    for (let j = 0; j < nGu; j++) {
      if (i === j) {
        out[i * nGu + j] = stayHome;
      } else {
        const baseOff = baseFlat[i * nGu + j] ?? 0;
        if (offSum > 0) {
          out[i * nGu + j] = (baseOff / offSum) * budget;
        } else {
          out[i * nGu + j] = budget / (nGu - 1);
        }
      }
    }
  }
  return out;
}

// ── Primary entry ──────────────────────────────────────────────────────

/**
 * Run a single what-if simulation. Assumes ``initSeirWasm()`` + ``loadMetapopInit()``
 * have both resolved — typical call sequence:
 *
 *     await initSeirWasm();
 *     const init = await loadMetapopInit();
 *     const out = runWhatIf(init, { r0: 1.4, stayHome: 0.55, ... });
 */
export function runWhatIf(
  init: SeirMetapopInit,
  inputs: WhatIfInputs,
): WhatIfOutputs {
  const {
    r0,
    stayHome,
    days,
    seedInfected,
    seedGuIdx,
    distancingEffect = 0,
    npiStartDay = 0,
    npiEndDay = 0,
  } = inputs;

  const populations = Float64Array.from(init.populations);
  const mobility = applyStayHome(init.mobility_flat, init.n_gu, stayHome);

  // Flu-like defaults aligned with DEFAULT_FLU_PARAMS (see
  // simulation/sim/parameters.py).
  const params = new DiseaseParams(
    r0,
    2.0, // incubation_days → σ = 0.5
    3.5, // infectious_days → γ ≈ 0.2857
    0.001, // cfr
    0.0, // vax_rate  (handled by scenarios, not what-if slider)
    0.5, // vax_eff
  );
  const intervention = new Intervention();
  if (distancingEffect > 0) {
    intervention.distancing_effect = distancingEffect;
    intervention.start_day = npiStartDay;
    intervention.end_day = npiEndDay;
  }

  const clampedSeedGu = Math.max(
    0,
    Math.min(init.n_gu - 1, Math.floor(seedGuIdx)),
  );

  let result: SimResult | null = null;
  const t0 = performance.now();
  try {
    result = run_seir_metapop(
      populations,
      clampedSeedGu,
      seedInfected,
      mobility,
      params,
      intervention,
      days,
      0.25,
    );
    const totalI = result.get_total_i();
    const totalS = result.get_total_s();
    const totalNew = result.get_total_new();
    const reT = result.get_re_t();

    // Pull per-gu series up-front while ``result`` is still live. Each
    // call returns a fresh ``Float64Array`` (``.slice()`` on the wasm
    // side) so we can safely hold them after ``result.free()`` below.
    const perGuI: Float64Array[] = new Array(init.n_gu);
    const perGuNew: Float64Array[] = new Array(init.n_gu);
    for (let g = 0; g < init.n_gu; g++) {
      perGuI[g] = result.get_i(g);
      perGuNew[g] = result.get_daily_new(g);
    }

    let peakI = 0;
    let peakDay = 0;
    for (let t = 0; t < totalI.length; t++) {
      if (totalI[t] > peakI) {
        peakI = totalI[t];
        peakDay = t;
      }
    }

    return {
      totalI,
      totalS,
      totalNew,
      reT,
      elapsedMs: performance.now() - t0,
      peakI,
      peakDay,
      perGuI,
      perGuNew,
    };
  } finally {
    // Order matters: SimResult keeps a pointer into the arena; free it
    // before the dependency params/intervention.
    result?.free();
    intervention.free();
    params.free();
  }
}
