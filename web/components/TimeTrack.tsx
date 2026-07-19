/**
 * TimeTrack — a 3–5 row time-axis strip that sits above the map and
 * chat panels on desktop, and below the map on mobile.
 *
 * Row layout (per stage plan):
 *   1. Week cursor slider (controls the `week` selected on the map)
 *   2. Observed ILI mini line chart (rolled up across Seoul)
 *   3. Forecast median + 95% PI band
 *   4. NPI / holiday / school-vacation event pips
 *   5. Rt trajectory with 1.0 reference line
 *
 * Each row label carries a small `?` help icon (see
 * components/ui/HelpIcon.tsx) that hover-reveals or tap-pins a short
 * explanation, so demo viewers unfamiliar with Rt / PI / ISO weeks get
 * the context without cluttering the chart.
 *
 * Everything is inline SVG so we don't need a charting library for
 * such a compact strip. Data flows in via props; AppShell owns the
 * store.
 */
"use client";

import { useMemo } from "react";

import { useT } from "@/lib/i18n";
import { HelpIcon } from "./ui/HelpIcon";
import { LastUpdated } from "./ui/LastUpdated";

export interface TimeTrackPoint {
  week: string;    // ISO week label, e.g. "2026-W15"
  observed?: number | null;
  forecast?: number | null;
  pi_lo?: number | null;
  pi_hi?: number | null;
  rt?: number | null;
}

export interface TimeTrackEvent {
  week: string;
  label: string;
  kind: "npi" | "holiday" | "vacation" | "vaccination";
}

export interface TimeTrackProps {
  data: TimeTrackPoint[];
  events?: TimeTrackEvent[];
  cursorWeek?: string;
  onCursorChange?: (week: string) => void;
  /** Optional ISO timestamp indicating when `data` was last refreshed.
   *  Surfaced via a <LastUpdated> badge next to the Week row. */
  dataGeneratedAt?: string | null;
  /** Human-readable week label for the "latest DB week" — shown next to
   *  the freshness badge so viewers see both the wall-clock age AND the
   *  calendar week of the last observation. */
  latestDataWeek?: string | null;
}

const EVENT_COLOR: Record<TimeTrackEvent["kind"], string> = {
  npi: "#f97316",
  holiday: "#a78bfa",
  vacation: "#22d3ee",
  vaccination: "#34d399",
};

export function TimeTrack({
  data,
  events = [],
  cursorWeek,
  onCursorChange,
  dataGeneratedAt,
  latestDataWeek,
}: TimeTrackProps) {
  const { t } = useT();
  const w = 640;
  const h = 28;
  const n = data.length;

  const xs = useMemo(() => {
    if (n < 2) return new Array(n).fill(0);
    const step = w / (n - 1);
    return data.map((_, i) => i * step);
  }, [n, data]);

  const { obsPath, fcPath, piPath, rtPath, obsMax, rtYMap } = useMemo(() => {
    const obs = data.map((d) => d.observed).filter((v): v is number => v != null);
    const fc = data.map((d) => d.forecast).filter((v): v is number => v != null);
    const obsMax = Math.max(1, ...obs, ...fc);
    const obsY = (v: number | null | undefined) =>
      v == null ? null : h - (v / obsMax) * (h - 2) - 1;
    const pathFrom = (getter: (d: TimeTrackPoint) => number | null | undefined) => {
      let s = "";
      data.forEach((d, i) => {
        const y = obsY(getter(d));
        if (y == null) return;
        s += (s ? " L " : "M ") + `${xs[i].toFixed(1)} ${y.toFixed(1)}`;
      });
      return s;
    };
    const obsPath = pathFrom((d) => d.observed);
    const fcPath = pathFrom((d) => d.forecast);

    // PI band as a closed polygon — hi forward then lo backward.
    let piPath = "";
    const hiPts: string[] = [];
    const loPts: string[] = [];
    data.forEach((d, i) => {
      if (d.pi_hi != null) hiPts.push(`${xs[i].toFixed(1)} ${obsY(d.pi_hi)!.toFixed(1)}`);
      if (d.pi_lo != null) loPts.unshift(`${xs[i].toFixed(1)} ${obsY(d.pi_lo)!.toFixed(1)}`);
    });
    if (hiPts.length && loPts.length) {
      piPath = `M ${hiPts.join(" L ")} L ${loPts.join(" L ")} Z`;
    }

    // Rt: separate axis, 0..3 clipped, with 1.0 midline.
    const rtY = (v: number | null | undefined) =>
      v == null ? null : h - Math.min(Math.max(v, 0), 3) / 3 * (h - 2) - 1;
    const rtYMap = data.map((d) => rtY(d.rt));
    let rtPath = "";
    data.forEach((d, i) => {
      const y = rtYMap[i];
      if (y == null) return;
      rtPath += (rtPath ? " L " : "M ") + `${xs[i].toFixed(1)} ${y.toFixed(1)}`;
    });

    return { obsPath, fcPath, piPath, rtPath, obsMax, rtYMap };
  }, [data, h, xs]);

  const cursorIdx = cursorWeek
    ? data.findIndex((d) => d.week === cursorWeek)
    : -1;
  const cursorX = cursorIdx >= 0 ? xs[cursorIdx] : null;

  return (
    <div className="flex flex-col gap-1 rounded-md border border-slate-800 bg-slate-900/60 p-2 text-[10px] text-slate-400">
      {/* Row 1 — week cursor */}
      <div className="flex items-center gap-2">
        <span className="flex w-16 shrink-0 items-center gap-1">
          <span>Week</span>
          <HelpIcon label={t("helpWeek")} content={t("helpWeek")} side="bottom" />
        </span>
        <input
          type="range"
          min={0}
          max={Math.max(0, n - 1)}
          value={Math.max(0, cursorIdx)}
          onChange={(e) => {
            const idx = Number(e.target.value);
            if (!Number.isFinite(idx)) return;
            const d = data[idx];
            if (d && onCursorChange) onCursorChange(d.week);
          }}
          className="flex-1 accent-sky-400"
          aria-label="Week cursor"
        />
        <span className="w-20 shrink-0 text-right tabular-nums text-slate-200">
          {cursorWeek ?? data[data.length - 1]?.week ?? "—"}
        </span>
      </div>

      {/* Freshness badge — always rendered when we know `generated_at`
          or the latest DB week. Keeps the "is this live?" question
          answerable at a glance. */}
      {(dataGeneratedAt || latestDataWeek) ? (
        <div className="flex justify-end pr-1">
          <LastUpdated at={dataGeneratedAt ?? undefined} weekLabel={latestDataWeek ?? undefined} />
        </div>
      ) : null}

      {/* Row 2 — Observed ILI */}
      <Row title={`ILI obs (max ${obsMax.toFixed(1)})`} help={t("helpIli")}>
        <svg viewBox={`0 0 ${w} ${h}`} className="h-6 w-full" preserveAspectRatio="none">
          {piPath ? (
            <path d={piPath} fill="#38bdf8" fillOpacity={0.15} />
          ) : null}
          {fcPath ? (
            <path d={fcPath} stroke="#38bdf8" strokeWidth={1} fill="none" strokeDasharray="3 2" />
          ) : null}
          {obsPath ? (
            <path d={obsPath} stroke="#e2e8f0" strokeWidth={1.2} fill="none" />
          ) : null}
          {cursorX != null ? (
            <line x1={cursorX} x2={cursorX} y1={0} y2={h} stroke="#facc15" strokeWidth={0.5} />
          ) : null}
        </svg>
      </Row>

      {/* Row 3 — NPI / holiday / vacation / vaccination event pips */}
      <Row title="Events" help={t("helpEvents")}>
        <svg viewBox={`0 0 ${w} ${h}`} className="h-4 w-full" preserveAspectRatio="none">
          {events.map((ev, i) => {
            const idx = data.findIndex((d) => d.week === ev.week);
            if (idx < 0) return null;
            return (
              <g key={`${ev.week}-${i}`}>
                <circle
                  cx={xs[idx]}
                  cy={h / 2}
                  r={2.4}
                  fill={EVENT_COLOR[ev.kind]}
                  opacity={0.85}
                >
                  <title>{`${ev.kind}: ${ev.label} @ ${ev.week}`}</title>
                </circle>
              </g>
            );
          })}
        </svg>
      </Row>

      {/* Row 4 — Rt trajectory with 1.0 midline */}
      <Row title="Rt (0–3)" help={t("helpRt")}>
        <svg viewBox={`0 0 ${w} ${h}`} className="h-6 w-full" preserveAspectRatio="none">
          <line
            x1={0}
            x2={w}
            y1={h - (1 / 3) * (h - 2) - 1}
            y2={h - (1 / 3) * (h - 2) - 1}
            stroke="#475569"
            strokeDasharray="2 3"
            strokeWidth={0.5}
          />
          {rtPath ? (
            <path d={rtPath} stroke="#a78bfa" strokeWidth={1.2} fill="none" />
          ) : null}
          {cursorX != null ? (
            <line x1={cursorX} x2={cursorX} y1={0} y2={h} stroke="#facc15" strokeWidth={0.5} />
          ) : null}
        </svg>
      </Row>
    </div>
  );
}

function Row({
  title,
  help,
  children,
}: {
  title: string;
  help?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="flex w-16 shrink-0 items-center gap-1 truncate">
        <span className="truncate">{title}</span>
        {help ? <HelpIcon label={help} content={help} side="bottom" /> : null}
      </span>
      <div className="flex-1">{children}</div>
    </div>
  );
}
