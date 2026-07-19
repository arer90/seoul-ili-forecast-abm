/**
 * LastUpdated — a compact "updated X minutes ago" badge.
 *
 * Two sources feed it:
 *
 *   1. Static aggregates (``live-overlays.json``, ``trained-models.json``)
 *      carry a ``generated_at`` ISO string — baked at `uv run python -m
 *      simulation.scripts.build_live_overlays` time.
 *   2. Dynamic DB timestamp — the chat stack queries ``epi.query_db`` for
 *      ``MAX(year, week_no) FROM weekly_disease`` but that's async, so
 *      this component accepts either a raw ISO string, a "Yr-Wk" label,
 *      or both.
 *
 * Accepts a ``stale`` threshold in hours (default 48) — beyond that the
 * badge shades amber and shows a small "⚠" so demos don't
 * inadvertently pitch stale data.
 *
 * Intentionally does NOT fetch on its own: the caller already fetched
 * the aggregate and owns the timestamp. Having a single component
 * author all fetches would conflate freshness across metrics.
 */
"use client";

import { useEffect, useMemo, useState } from "react";

import { useT } from "@/lib/i18n";

export interface LastUpdatedProps {
  /** ISO-8601 timestamp string (e.g. "2026-04-21T23:06:21Z"). */
  at?: string | null;
  /** Optional human label (e.g. "W17 · 2026"). Shown in parens. */
  weekLabel?: string | null;
  /** Hours beyond which to shade amber. Default 48. */
  staleHours?: number;
  className?: string;
}

function fmtRelative(ms: number, locale: "ko" | "en"): string {
  const s = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(s / 60);
  const h = Math.floor(m / 60);
  const d = Math.floor(h / 24);
  if (locale === "ko") {
    if (s < 45) return "방금 전";
    if (m < 2) return "1분 전";
    if (m < 60) return `${m}분 전`;
    if (h < 2) return "1시간 전";
    if (h < 24) return `${h}시간 전`;
    if (d < 2) return "어제";
    return `${d}일 전`;
  }
  if (s < 45) return "just now";
  if (m < 2) return "1 min ago";
  if (m < 60) return `${m} min ago`;
  if (h < 2) return "1 hr ago";
  if (h < 24) return `${h} hr ago`;
  if (d < 2) return "yesterday";
  return `${d} days ago`;
}

function fmtAbsolute(iso: string, locale: "ko" | "en"): string {
  const d = new Date(iso);
  if (!Number.isFinite(d.getTime())) return iso;
  const yyyy = d.getUTCFullYear();
  const mm = String(d.getUTCMonth() + 1).padStart(2, "0");
  const dd = String(d.getUTCDate()).padStart(2, "0");
  const hh = String(d.getUTCHours()).padStart(2, "0");
  const mi = String(d.getUTCMinutes()).padStart(2, "0");
  return locale === "ko"
    ? `${yyyy}-${mm}-${dd} ${hh}:${mi} UTC`
    : `${yyyy}-${mm}-${dd} ${hh}:${mi} UTC`;
}

export function LastUpdated({
  at,
  weekLabel,
  staleHours = 48,
  className = "",
}: LastUpdatedProps) {
  const { locale, t } = useT();
  const [now, setNow] = useState<number>(() => Date.now());

  // Re-tick every 60 s so "5 min ago" actually advances. Cheap — a
  // single setInterval for the whole app since the component is tiny.
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 60_000);
    return () => window.clearInterval(id);
  }, []);

  const parsed = useMemo(() => {
    if (!at) return null;
    const ms = Date.parse(at);
    return Number.isFinite(ms) ? ms : null;
  }, [at]);

  if (!parsed && !weekLabel) return null;

  const ageMs = parsed != null ? now - parsed : 0;
  const stale = parsed != null && ageMs > staleHours * 3600 * 1000;

  const relative = parsed != null ? fmtRelative(ageMs, locale) : null;
  const absolute = parsed != null ? fmtAbsolute(at!, locale) : null;

  return (
    <span
      className={[
        "inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[10px]",
        stale
          ? "border-amber-600/60 bg-amber-900/30 text-amber-200"
          : "border-slate-700 bg-slate-900/60 text-slate-400",
        className,
      ].join(" ")}
      title={absolute ?? undefined}
      aria-label={`${t("lastUpdated")}: ${absolute ?? weekLabel ?? ""}`}
    >
      {stale ? <span aria-hidden="true">⚠</span> : null}
      <span className="font-medium text-slate-300">{t("lastUpdated")}</span>
      {relative ? <span>{relative}</span> : null}
      {weekLabel ? (
        <span className="text-slate-500">· {weekLabel}</span>
      ) : null}
    </span>
  );
}
