/**
 * TransmissionPlayer — animates the WASM SEIR-V-D output across the 25
 * Seoul gu on the map.
 *
 * Why this exists
 *   Showing "전파가 일어나고 있다" is hard to do with a static choropleth.
 *   The evaluation side of the thesis has the 53-model forecast + the
 *   simulation benchmark, but the demo UI previously rendered only a
 *   single snapshot. This component turns the Metapop simulation into a
 *   time-resolved animation so a viewer watches the wave spread from
 *   the seeded gu outward through the commuter matrix.
 *
 * How it works
 *   1. Reuses the browser WASM runner (``runWhatIf``) so the numbers are
 *      exactly the same as the What-if card — no new science, just a
 *      new rendering of the same simulation output.
 *   2. Downsamples the daily output to one frame per 7 days so the
 *      week-based slider / TimeTrack cursor stays in sync.
 *   3. Calls ``onFrame(weekIdx, guRows)`` on each tick. Parent decides
 *      what to do — in AppShell we update both the week cursor AND the
 *      choropleth ``guRows`` so the map, the TimeTrack, and this player
 *      are all pinned to the same week.
 *   4. Play / Pause / Reset / Speed × speed (0.5/1/2/4). All controls
 *      are keyboard-reachable (Tab) + touch-sized (≥ 32 px).
 *
 * Scope limits
 *   - This is a demo visualisation. For the paper, cite the authoritative
 *     ``simulation/sim/metapop_seirvd.py`` run; the WASM matches to within
 *     0.007% (see bench_seir_evaluation.md).
 *   - We don't persist the simulation output anywhere — a fresh run
 *     happens on mount, and again only when ``seedGuIdx`` / ``r0`` change.
 */
"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  initSeirWasm,
  loadMetapopInit,
  runWhatIf,
  type SeirMetapopInit,
  type WhatIfOutputs,
} from "@/lib/seir-wasm/client";
import { useT } from "@/lib/i18n";

import { HelpIcon } from "./ui/HelpIcon";
import type { GuChoroplethRow } from "./MapPanel";

// Type-only import — the runtime module is loaded lazily inside
// ``onExportGif`` via ``await import("gif.js")`` so the ~50 KB worker
// shim never enters the critical-path bundle. The class itself doubles
// as the instance type (TS convention for class imports).
import type GIFType from "gif.js";

const SPEEDS = [0.5, 1, 2, 4] as const;

/**
 * Per-tick summary emitted to the parent. Previously just the week
 * index + per-gu values; now also carries the fields the MapPanel's
 * HUD (Day X / Y · W N · 피크 W P · 누적 ·  활성 구) needs, plus a
 * short diagnostic so downstream (chat auto-context, B2) can form a
 * one-line prompt without reaching back into the WASM output.
 */
export interface PlayerFrame {
  /** 0-based week index into ``frames``. */
  weekIdx: number;
  /** Day number (weekIdx * 7), for the HUD's "Day 84 / 180" readout. */
  dayIdx: number;
  /** Total simulated days — denominator for the HUD counter. */
  totalDays: number;
  /** Total weeks in the simulation (== frames.length). */
  nFrames: number;
  /** Choropleth rows for THIS week, in the SEIR init gu order. */
  guRows: GuChoroplethRow[];
  /** Cumulative new infections across all gu up through this week. */
  cumulative: number;
  /** Number of gu with ≥ 1 new infection in THIS week alone. */
  activeGuCount: number;
  /** Week index of the Seoul-total peak (from the full simulation). */
  peakWeek: number;
  /** Top 3 gu names by this week's new-infection count (HUD + chat). */
  topGus: string[];
  /** Is the player currently running? Drives the HUD's red/blue dot. */
  playing: boolean;
}

export interface TransmissionPlayerProps {
  /** Called whenever the playhead moves — parent wires to the map + cursor. */
  onFrame?: (frame: PlayerFrame) => void;
  /** Week labels so the playback label can show "2025-W15" style text. */
  weekLabels?: string[];
  /** When provided, align the animation with the Week slider: if the
   *  user drags the slider, we follow; if the user presses Play, we
   *  advance from here. */
  cursorWeek?: string;
  /** Gu index the epidemic seeds from. Exposed so AppShell can let the
   *  user "click a gu, press play, watch it spread from there". */
  seedGuIdx?: number;
  /** Initial R0 for the simulation. */
  r0?: number;
}

export function TransmissionPlayer({
  onFrame,
  weekLabels = [],
  cursorWeek,
  seedGuIdx = 22,
  r0 = 1.4,
}: TransmissionPlayerProps) {
  const { t } = useT();
  const [init, setInit] = useState<SeirMetapopInit | null>(null);
  const [sim, setSim] = useState<WhatIfOutputs | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<(typeof SPEEDS)[number]>(1);
  const [weekIdx, setWeekIdx] = useState(0);

  // Mount-time WASM load + initial simulation. Re-runs when r0 /
  // seedGuIdx change so the "seed district" story is live — if the user
  // wants to watch it spread from a different gu, they pick it on the
  // map, the parent passes the new index, we re-simulate.
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const [, initJson] = await Promise.all([
          initSeirWasm(),
          loadMetapopInit(),
        ]);
        if (cancelled) return;
        setInit(initJson);
        const out = runWhatIf(initJson, {
          r0,
          stayHome: 0.55,
          days: 180,
          seedInfected: 100,
          seedGuIdx,
          distancingEffect: 0,
          npiStartDay: 0,
          npiEndDay: 0,
        });
        if (cancelled) return;
        setSim(out);
      } catch (e) {
        if (!cancelled) setErr(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [r0, seedGuIdx]);

  // Build a [week][gu] matrix of daily-new values, one frame per 7 days.
  // ``perGuNew`` (daily new infections) animates the wave front better
  // than the cumulative infectious pool — cases per week is what public-
  // health briefings use anyway.
  const frames = useMemo<number[][]>(() => {
    if (!sim) return [];
    const nDays = sim.perGuNew[0]?.length ?? 0;
    const nWeeks = Math.max(1, Math.floor(nDays / 7));
    const out: number[][] = [];
    for (let w = 0; w < nWeeks; w++) {
      const row: number[] = [];
      const t0 = w * 7;
      const t1 = Math.min(nDays, t0 + 7);
      for (let g = 0; g < sim.perGuNew.length; g++) {
        let s = 0;
        const arr = sim.perGuNew[g];
        for (let t = t0; t < t1; t++) s += arr[t] ?? 0;
        row.push(s);
      }
      out.push(row);
    }
    return out;
  }, [sim]);

  const nFrames = frames.length;

  // When the parent week cursor moves AND we're not actively playing,
  // snap our internal weekIdx to match. That way dragging the TimeTrack
  // slider rewinds the player too.
  useEffect(() => {
    if (playing) return;
    if (!cursorWeek || !weekLabels.length) return;
    const idx = weekLabels.indexOf(cursorWeek);
    if (idx >= 0 && idx < nFrames) setWeekIdx(idx);
  }, [cursorWeek, weekLabels, playing, nFrames]);

  const peakWeek = useMemo(() => {
    if (!frames.length) return -1;
    let best = 0;
    let bestSum = 0;
    for (let w = 0; w < frames.length; w++) {
      const s = frames[w].reduce((a, b) => a + b, 0);
      if (s > bestSum) {
        bestSum = s;
        best = w;
      }
    }
    return best;
  }, [frames]);

  // Emit frames to the parent. Deferred to useEffect so we don't call
  // setState of the parent during our own render.
  useEffect(() => {
    if (!init || !frames[weekIdx]) return;
    const row = frames[weekIdx];
    const guRows: GuChoroplethRow[] = init.district_names.map((gu, g) => ({
      gu_nm: gu,
      value: row[g] ?? 0,
    }));
    // HUD-side derived stats. We compute everything here rather than
    // in the parent so the player is the single source of truth — the
    // HUD numbers never drift out of sync with the colored choropleth.
    let cumulative = 0;
    for (let w = 0; w <= weekIdx; w++) {
      const r = frames[w];
      for (let g = 0; g < r.length; g++) cumulative += r[g] ?? 0;
    }
    const activeGuCount = row.filter((v) => v >= 1).length;
    const indexed = row
      .map((v, g) => ({ gu: init.district_names[g] ?? `gu${g}`, v }))
      .sort((a, b) => b.v - a.v);
    const topGus = indexed.slice(0, 3).filter((e) => e.v >= 1).map((e) => e.gu);
    const totalDays = nFrames * 7;
    const dayIdx = Math.min(totalDays, (weekIdx + 1) * 7);
    onFrame?.({
      weekIdx,
      dayIdx,
      totalDays,
      nFrames,
      guRows,
      cumulative,
      activeGuCount,
      peakWeek,
      topGus,
      playing,
    });
    // ``playing`` is in the deps so a Play/Pause toggle re-emits with the
    // same weekIdx — the MapPanel HUD's dot colour then flips immediately
    // instead of waiting for the next tick (which would never come while
    // paused). The extra emission is cheap (one frame) and idempotent for
    // the rest of the HUD state.
  }, [weekIdx, frames, init, onFrame, nFrames, peakWeek, playing]);

  // Animation loop. Uses setInterval instead of rAF because the cadence
  // we want is "one frame per ~800/speed ms", not "every vsync".
  useEffect(() => {
    if (!playing || nFrames === 0) return;
    const period = Math.max(50, Math.floor(800 / speed));
    const id = window.setInterval(() => {
      setWeekIdx((i) => {
        const next = i + 1;
        if (next >= nFrames) {
          // Stop at the end instead of looping — the user almost always
          // wants to see "and now it levels off" rather than an infinite
          // spinner. Press Play again to replay from start if desired.
          setPlaying(false);
          return nFrames - 1;
        }
        return next;
      });
    }, period);
    return () => window.clearInterval(id);
  }, [playing, speed, nFrames]);

  // peakWeek is defined earlier (above the onFrame effect) so the
  // effect can pass it through without hitting a TDZ reference.

  const onPlayPause = useCallback(() => {
    if (!nFrames) return;
    // If at end, pressing Play restarts from week 0 — usability > purity.
    if (weekIdx >= nFrames - 1 && !playing) {
      setWeekIdx(0);
    }
    setPlaying((p) => !p);
  }, [nFrames, weekIdx, playing]);

  // ── GIF export (Stage 6 hybrid-option step 1) ────────────────────
  //
  // The thesis demo needs a static artefact for slides / the paper /
  // README. Live Leaflet playback is great in front of a computer but
  // useless in a PDF. This button walks the simulation once, snapshots
  // the map (SVG choropleth + commuter-edge pulses + HUD) per week,
  // encodes the frames into an animated GIF, and offers it as a
  // download.
  //
  // Tile tiles are *excluded* from the snapshot because (a) OSM-tainted
  // canvases throw on toBlob (CORS), and (b) the result is cleaner as a
  // publication figure anyway — dark background + gu silhouette only.
  //
  // Dynamic imports keep ``gif.js`` + ``html-to-image`` (~50 KB each)
  // off the critical path — they only load when the user actually
  // presses the button.
  const [exporting, setExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState(0);
  const [exportErr, setExportErr] = useState<string | null>(null);
  const exportAbortRef = useRef<{ cancel: boolean }>({ cancel: false });

  const onExportGif = useCallback(async () => {
    if (!sim || !init || !frames.length || exporting) return;
    setExportErr(null);
    // Need a mounted MapPanel. We stamp `data-map-root="true"` on the
    // panel's outer div — query-selecting for it avoids plumbing a ref
    // all the way from AppShell.
    const mapEl = document.querySelector<HTMLElement>(
      '[data-map-root="true"]',
    );
    if (!mapEl) {
      setExportErr("map not mounted");
      return;
    }
    setPlaying(false);
    setExporting(true);
    setExportProgress(0);
    exportAbortRef.current = { cancel: false };
    const abort = exportAbortRef.current;

    try {
      const [{ default: GIF }, { toPng }] = await Promise.all([
        import("gif.js"),
        import("html-to-image"),
      ]);

      const rect = mapEl.getBoundingClientRect();
      // Cap the GIF at 900 × 900 — larger maps waste bytes and slow
      // down encoding without adding detail at the 25-gu resolution.
      const ratio = Math.min(900 / rect.width, 900 / rect.height, 1);
      const width = Math.max(240, Math.round(rect.width * ratio));
      const height = Math.max(240, Math.round(rect.height * ratio));

      // ``gif`` is declared ``const`` so TS keeps its non-null type
      // across the ``await`` points inside the frame loop — we don't
      // need a separate nullable handle.
      const gif: GIFType = new GIF({
        workers: 2,
        quality: 10,
        workerScript: "/gif.worker.js",
        width,
        height,
        background: "#0b0f14",
        transparent: null,
      });

      const originalWeekIdx = weekIdx;
      for (let w = 0; w < frames.length; w++) {
        if (abort.cancel) break;
        setWeekIdx(w);
        // Two rAFs + a short settle delay so React flushes the state
        // change, the map-layer effect runs setStyle() on every gu, and
        // the commuter-edge animation ticks its dash offset. 80 ms is
        // the empirical sweet spot — below that the choropleth sometimes
        // still shows the previous week's colours.
        await new Promise<void>((resolve) => {
          requestAnimationFrame(() =>
            requestAnimationFrame(() => {
              setTimeout(resolve, 80);
            }),
          );
        });

        const dataUrl = await toPng(mapEl, {
          cacheBust: false,
          pixelRatio: 1,
          width,
          height,
          canvasWidth: width,
          canvasHeight: height,
          backgroundColor: "#0b0f14",
          filter: (node) => {
            if (node.nodeType !== 1) return true;
            const el = node as Element;
            const cls =
              typeof el.className === "string" ? el.className : "";
            // Exclude OSM tile layer (CORS-tainted → toBlob throws) and
            // the freshness-stamp controls that only clutter the figure.
            if (cls.includes("leaflet-tile-pane")) return false;
            if (cls.includes("leaflet-control-attribution")) return false;
            return true;
          },
        });

        const img = document.createElement("img");
        img.src = dataUrl;
        await new Promise<void>((resolve, reject) => {
          img.onload = () => resolve();
          img.onerror = () => reject(new Error("frame image decode failed"));
        });
        gif.addFrame(img, { delay: 400 });
        setExportProgress(w + 1);
      }

      if (abort.cancel) {
        setExporting(false);
        setExportProgress(0);
        setWeekIdx(originalWeekIdx);
        return;
      }

      const blob: Blob = await new Promise((resolve) => {
        gif.on("finished", (b) => resolve(b));
        gif.render();
      });

      // Restore previous week after encoding so the user doesn't jump
      // to the end of the season.
      setWeekIdx(originalWeekIdx);

      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `abs-${new Date()
        .toISOString()
        .slice(0, 10)}-seed${seedGuIdx}-r0${r0}.gif`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      // Revoke on next tick so the download survives Chrome's download
      // manager read. Not strictly necessary but avoids a leak under
      // repeated clicks.
      setTimeout(() => URL.revokeObjectURL(url), 5000);
    } catch (e) {
      setExportErr(e instanceof Error ? e.message : String(e));
    } finally {
      setExporting(false);
      setExportProgress(0);
    }
  }, [sim, init, frames, exporting, weekIdx, seedGuIdx, r0]);

  const onCancelExport = useCallback(() => {
    exportAbortRef.current.cancel = true;
  }, []);

  const onReset = useCallback(() => {
    setPlaying(false);
    setWeekIdx(0);
  }, []);

  const onJumpPeak = useCallback(() => {
    if (peakWeek < 0) return;
    setPlaying(false);
    setWeekIdx(peakWeek);
  }, [peakWeek]);

  const seedLabel = init?.district_names?.[seedGuIdx] ?? "—";

  return (
    <div className="flex flex-wrap items-center gap-1.5 rounded-md border border-slate-800 bg-slate-900/60 px-2 py-1 text-[11px] text-slate-300">
      <span className="flex items-center gap-1 text-slate-400">
        <span aria-hidden="true">🦠</span>
        <span>{t("transmissionLabel")}</span>
        <HelpIcon
          label={t("helpTransmission")}
          content={t("helpTransmission")}
          side="bottom"
        />
      </span>
      <button
        type="button"
        onClick={onPlayPause}
        disabled={!sim || !!err}
        className={[
          "inline-flex items-center gap-1 rounded-md border px-2 py-0.5",
          "border-slate-700 bg-slate-800 text-slate-100",
          "hover:border-sky-500/60 hover:bg-slate-700",
          "disabled:cursor-not-allowed disabled:opacity-60",
        ].join(" ")}
        aria-pressed={playing}
        title={playing ? t("transmissionPause") : t("transmissionPlay")}
      >
        <span aria-hidden="true">{playing ? "⏸" : "▶"}</span>
        <span>{playing ? t("transmissionPause") : t("transmissionPlay")}</span>
      </button>
      <button
        type="button"
        onClick={onReset}
        disabled={!sim}
        className="rounded-md border border-slate-700 bg-slate-800 px-2 py-0.5 text-slate-200 hover:border-sky-500/60 hover:bg-slate-700 disabled:opacity-60"
        title={t("transmissionReset")}
      >
        <span aria-hidden="true">⟲</span>
      </button>
      <button
        type="button"
        onClick={onJumpPeak}
        disabled={!sim || peakWeek < 0}
        className="rounded-md border border-slate-700 bg-slate-800 px-2 py-0.5 text-slate-200 hover:border-sky-500/60 hover:bg-slate-700 disabled:opacity-60"
        title={t("transmissionJumpPeak")}
      >
        <span aria-hidden="true">⇡</span>{" "}
        {peakWeek >= 0 ? `W${peakWeek}` : ""}
      </button>
      <label className="flex items-center gap-1">
        <span className="text-slate-500">{t("transmissionSpeed")}</span>
        <select
          value={speed}
          onChange={(e) => setSpeed(Number(e.target.value) as (typeof SPEEDS)[number])}
          className="rounded border border-slate-700 bg-slate-950 px-1 py-0.5 text-slate-200 focus:border-sky-500 focus:outline-none"
        >
          {SPEEDS.map((s) => (
            <option key={s} value={s}>
              {s}×
            </option>
          ))}
        </select>
      </label>
      {/* GIF export — hybrid-option step 1. Separate button so the
          "just play" buttons stay one click. Progress bar replaces the
          button label while encoding so the user sees it's working. */}
      {exporting ? (
        <span
          className="inline-flex items-center gap-1 rounded-md border border-emerald-700 bg-emerald-900/40 px-2 py-0.5 text-[11px] text-emerald-100"
          role="status"
          aria-live="polite"
        >
          <span aria-hidden="true">●</span>
          <span className="tabular-nums">
            {exportProgress}/{nFrames}
          </span>
          <button
            type="button"
            onClick={onCancelExport}
            className="ml-1 rounded border border-emerald-600 bg-emerald-800/60 px-1 text-[10px] text-emerald-50 hover:bg-emerald-700"
            title={t("exportGifCancel")}
          >
            ✕
          </button>
        </span>
      ) : (
        <button
          type="button"
          onClick={onExportGif}
          disabled={!sim || !!err}
          className="rounded-md border border-slate-700 bg-slate-800 px-2 py-0.5 text-slate-200 hover:border-emerald-500/60 hover:bg-slate-700 disabled:opacity-60"
          title={t("exportGifHint")}
        >
          <span aria-hidden="true">📸</span> {t("exportGifLabel")}
        </button>
      )}
      <span className="ml-auto flex items-center gap-2 text-[10px] text-slate-500">
        <span>
          {t("transmissionSeed")}: <span className="text-slate-300">{seedLabel}</span>
        </span>
        {sim ? (
          <span>
            W{weekIdx + 1}/{nFrames} · R₀={r0}
          </span>
        ) : err ? (
          <span className="text-red-300">{err}</span>
        ) : (
          <span>{t("transmissionLoading")}</span>
        )}
        {exportErr ? (
          <span className="text-red-300">export: {exportErr}</span>
        ) : null}
      </span>
    </div>
  );
}
