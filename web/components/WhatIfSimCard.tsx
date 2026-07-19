/**
 * WhatIfSimCard — client-only interactive Metapop SEIR-V-D slider.
 *
 * Why it exists
 * -------------
 * The MCP ``epi.scenario_run`` path goes: browser → Edge fn → Python MCP
 * → ODE (~200 ms warm, dominated by DB load + JSON). That is fast
 * enough for canonical scenarios, but blocks audience-pace slider
 * interaction.
 *
 * This card calls the Rust/WASM build of the same ODE *inside the
 * browser*. On desktop Chrome the median for a 365-day × 25-gu run is
 * ~27 ms, so it's viable to re-simulate on every slider tick without a
 * network round-trip. See ``simulation/results/bench_seir_evaluation.md``
 * for the full benchmark methodology and numbers.
 *
 * Design
 * ------
 * - Initialisation (WASM module + static init JSON) happens exactly
 *   once on mount. Until both resolve, we show a soft skeleton instead
 *   of a broken chart.
 * - Slider changes set React state immediately (for UI feedback) and
 *   schedule a simulation on the next animation frame. Rerunning in
 *   rAF rather than synchronously lets React paint the thumb movement
 *   before the 25–30 ms WASM call.
 * - The chart is hand-rolled SVG — the project doesn't pull Chart.js
 *   or D3, and we only need a single trace + peak annotation.
 * - All wasm-bindgen objects are freed inside ``runWhatIf`` — no
 *   long-lived references leak out of the module.
 *
 * Numbers match Python
 * --------------------
 * After the Step 2 FOI rewrite + mass-conservation fix, the WASM and
 * Python simulators agree on peak_I to within 0.007% (274,789 vs
 * 274,771). That's the numerical-noise floor, not a bug. So the
 * client-side what-if curves are *the same science* as the MCP
 * authoritative curves — they just render faster.
 */
"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import {
  initSeirWasm,
  loadMetapopInit,
  runWhatIf,
  type SeirMetapopInit,
  type WhatIfInputs,
  type WhatIfOutputs,
} from "@/lib/seir-wasm/client";
import { Button } from "./ui/button";

// Native <select> for the two dropdowns — the project's ``Select`` UI
// wraps in its own <label>, which nests poorly with our slider-style
// <label> layouts. Styling is inline to match ``ui/select.tsx`` dark
// theme so the look is consistent.
const selectCx =
  "h-7 rounded-md border border-slate-700 bg-slate-900 px-2 " +
  "text-xs text-slate-100 outline-none " +
  "focus:border-sky-400 focus:ring-1 focus:ring-sky-400/70 " +
  "disabled:opacity-50";

const HORIZONS = [90, 180, 365, 730] as const;

/** Slider default values, tuned to match the Python/WASM bench baseline. */
const DEFAULTS: WhatIfInputs = {
  r0: 1.4,
  stayHome: 0.55,
  days: 365,
  seedInfected: 100,
  seedGuIdx: 22, // "강남구" in SEOUL_GU_ORDERED (index 22)
  distancingEffect: 0,
  npiStartDay: 30,
  npiEndDay: 120,
};

interface RangeProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
  hint?: string;
}

function Range({
  label,
  value,
  min,
  max,
  step,
  onChange,
  format = (v) => v.toString(),
  hint,
}: RangeProps) {
  return (
    <label className="flex flex-col gap-1 text-xs text-slate-300">
      <div className="flex items-baseline justify-between">
        <span className="font-medium text-slate-200">{label}</span>
        <span className="font-mono text-sky-300">{format(value)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="accent-sky-400"
      />
      {hint ? <span className="text-[10px] text-slate-500">{hint}</span> : null}
    </label>
  );
}

interface TraceChartProps {
  totalI: Float64Array;
  peakI: number;
  peakDay: number;
  days: number;
}

/**
 * Tiny SVG line chart. 240 × 96 viewport, fully responsive via
 * ``preserveAspectRatio="none"`` — sized by the parent div.
 */
function TraceChart({ totalI, peakI, peakDay, days }: TraceChartProps) {
  const pathD = useMemo(() => {
    if (totalI.length < 2 || peakI <= 0) return "";
    const maxY = peakI * 1.05;
    const parts: string[] = [];
    for (let i = 0; i < totalI.length; i++) {
      const x = (i / (totalI.length - 1)) * 240;
      const y = 96 - (totalI[i] / maxY) * 96;
      parts.push(`${i === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`);
    }
    return parts.join(" ");
  }, [totalI, peakI]);

  const peakPx = totalI.length > 1 ? (peakDay / (totalI.length - 1)) * 240 : 0;

  return (
    <div className="flex flex-col gap-0.5">
      <div className="relative h-24 w-full overflow-hidden rounded-md border border-slate-800 bg-slate-950/80">
        <svg
          viewBox="0 0 240 96"
          preserveAspectRatio="none"
          className="h-full w-full"
        >
          {/* Baseline */}
          <line
            x1={0}
            x2={240}
            y1={95.5}
            y2={95.5}
            stroke="rgb(51 65 85)"
            strokeWidth={0.5}
          />
          {/* Peak vertical */}
          {peakI > 0 ? (
            <line
              x1={peakPx}
              x2={peakPx}
              y1={0}
              y2={96}
              stroke="rgb(217 70 70 / 0.35)"
              strokeDasharray="2 3"
              strokeWidth={0.6}
            />
          ) : null}
          {/* Infected trace */}
          {pathD ? (
            <path
              d={pathD}
              fill="none"
              stroke="rgb(56 189 248)"
              strokeWidth={1.2}
              vectorEffect="non-scaling-stroke"
            />
          ) : null}
        </svg>
      </div>
      <div className="flex items-center justify-between px-0.5 text-[10px] text-slate-500">
        <span>day 0</span>
        <span className="text-red-300/80">
          peak day {peakDay} · I = {Math.round(peakI).toLocaleString()}
        </span>
        <span>day {days}</span>
      </div>
    </div>
  );
}

export function WhatIfSimCard() {
  const [ready, setReady] = useState(false);
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [init, setInit] = useState<SeirMetapopInit | null>(null);
  const [inputs, setInputs] = useState<WhatIfInputs>(DEFAULTS);
  const [out, setOut] = useState<WhatIfOutputs | null>(null);
  const frameRef = useRef<number | null>(null);

  // One-time: load WASM + static init JSON in parallel.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [, initJson] = await Promise.all([
          initSeirWasm(),
          loadMetapopInit(),
        ]);
        if (cancelled) return;
        setInit(initJson);
        setReady(true);
      } catch (err) {
        if (cancelled) return;
        setLoadErr(err instanceof Error ? err.message : String(err));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Re-run the simulator whenever any slider changes — but defer to the
  // next animation frame so the input thumb paints before the 27 ms ODE.
  useEffect(() => {
    if (!ready || !init) return;
    if (frameRef.current != null) {
      cancelAnimationFrame(frameRef.current);
    }
    frameRef.current = requestAnimationFrame(() => {
      try {
        setOut(runWhatIf(init, inputs));
      } catch (err) {
        setLoadErr(err instanceof Error ? err.message : String(err));
      }
    });
    return () => {
      if (frameRef.current != null) {
        cancelAnimationFrame(frameRef.current);
        frameRef.current = null;
      }
    };
  }, [ready, init, inputs]);

  const totalPop = init
    ? init.populations.reduce((a, b) => a + b, 0)
    : 0;
  const attackRatePct =
    out && totalPop > 0
      ? (1 - out.totalS[out.totalS.length - 1] / totalPop) * 100
      : null;

  if (loadErr) {
    return (
      <div className="rounded-md border border-red-900/60 bg-red-950/40 p-2 text-xs text-red-200">
        WASM load failed: {loadErr}
      </div>
    );
  }

  return (
    <section
      aria-label="Interactive what-if simulator"
      className="flex flex-col gap-2 rounded-md border border-slate-800 bg-slate-950/60 p-2 text-xs"
    >
      <header className="flex items-center justify-between">
        <h3 className="text-xs font-semibold text-slate-100">
          What-If (browser WASM)
        </h3>
        <div className="text-[10px] text-slate-500">
          {ready
            ? out
              ? `ran in ${out.elapsedMs.toFixed(1)} ms`
              : "ready"
            : "loading wasm…"}
        </div>
      </header>

      {out ? (
        <TraceChart
          totalI={out.totalI}
          peakI={out.peakI}
          peakDay={out.peakDay}
          days={inputs.days}
        />
      ) : (
        <div className="h-24 animate-pulse rounded-md border border-slate-800 bg-slate-900/60" />
      )}

      <div className="grid grid-cols-2 gap-3">
        <Range
          label="R₀"
          value={inputs.r0}
          min={0.8}
          max={3.0}
          step={0.05}
          onChange={(v) => setInputs((s) => ({ ...s, r0: v }))}
          format={(v) => v.toFixed(2)}
          hint="basic reproduction number"
        />
        <Range
          label="Stay-home fraction"
          value={inputs.stayHome}
          min={0.30}
          max={0.95}
          step={0.01}
          onChange={(v) => setInputs((s) => ({ ...s, stayHome: v }))}
          format={(v) => `${(v * 100).toFixed(0)}%`}
          hint="diagonal of commuter matrix"
        />
        <Range
          label="Seed infected"
          value={inputs.seedInfected}
          min={1}
          max={1000}
          step={1}
          onChange={(v) => setInputs((s) => ({ ...s, seedInfected: v }))}
          format={(v) => v.toString()}
          hint="initial cases at t=0"
        />
        <label className="flex flex-col gap-1 text-slate-300">
          <div className="flex items-baseline justify-between">
            <span className="font-medium text-slate-200">Horizon</span>
            <span className="font-mono text-sky-300">{inputs.days} d</span>
          </div>
          <select
            value={inputs.days}
            onChange={(e) =>
              setInputs((s) => ({ ...s, days: Number(e.target.value) }))
            }
            className={selectCx}
          >
            {HORIZONS.map((d) => (
              <option key={d} value={d}>
                {d} days
              </option>
            ))}
          </select>
          <span className="text-[10px] text-slate-500">
            longer runs still finish under 60 ms
          </span>
        </label>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <label className="flex flex-col gap-1 text-slate-300">
          <div className="flex items-baseline justify-between">
            <span className="font-medium text-slate-200">Seed district</span>
            <span className="font-mono text-sky-300">
              {init?.district_names[inputs.seedGuIdx] ?? "—"}
            </span>
          </div>
          <select
            value={inputs.seedGuIdx}
            onChange={(e) =>
              setInputs((s) => ({ ...s, seedGuIdx: Number(e.target.value) }))
            }
            className={selectCx}
            disabled={!init}
          >
            {(init?.district_names ?? []).map((name, idx) => (
              <option key={name} value={idx}>
                {name}
              </option>
            ))}
          </select>
        </label>
        <Range
          label="NPI distancing"
          value={inputs.distancingEffect ?? 0}
          min={0}
          max={0.6}
          step={0.02}
          onChange={(v) => setInputs((s) => ({ ...s, distancingEffect: v }))}
          format={(v) => (v === 0 ? "off" : `${(v * 100).toFixed(0)}%`)}
          hint={
            inputs.distancingEffect && inputs.distancingEffect > 0
              ? `day ${inputs.npiStartDay}–${inputs.npiEndDay}`
              : "distancing-off baseline"
          }
        />
      </div>

      <div className="flex flex-wrap items-center gap-3 text-[10px] text-slate-400">
        <span>
          Seoul pop ≈ {(totalPop / 1_000_000).toFixed(2)} M
        </span>
        {attackRatePct != null ? (
          <span>
            Attack rate ≈{" "}
            <span className="font-mono text-sky-300">
              {attackRatePct.toFixed(2)}%
            </span>
          </span>
        ) : null}
        <Button
          size="sm"
          variant="outline"
          onClick={() => setInputs(DEFAULTS)}
          className="ml-auto"
        >
          Reset
        </Button>
      </div>
    </section>
  );
}

export default WhatIfSimCard;
