/**
 * ShapDelta — "this week vs last week" top-10 SHAP diff bar.
 *
 * Data shape is the same as returned by ``epi.shap_topk`` with
 * ``compare_to="prev_week"``. Positive delta means the feature got
 * more important; negative means less. We clamp the bar to ±1 relative
 * to |max| so the chart is visually balanced even if a single feature
 * dominates.
 */
"use client";

export interface ShapRow {
  feature: string;
  value_now: number;
  value_prev: number;
  delta: number;
}

export interface ShapDeltaProps {
  rows: ShapRow[];
  title?: string;
}

export function ShapDelta({
  rows,
  title = "SHAP delta — this week vs last",
}: ShapDeltaProps) {
  if (!rows.length)
    return (
      <div className="rounded-md border border-slate-800 bg-slate-950/60 p-3 text-xs text-slate-500">
        No SHAP delta yet. Ask the assistant for “SHAP delta” to populate.
      </div>
    );
  const maxAbs = Math.max(...rows.map((r) => Math.abs(r.delta)), 1e-9);

  return (
    <div className="rounded-md border border-slate-800 bg-slate-950/60 p-3">
      <div className="mb-2 text-xs font-medium text-slate-300">{title}</div>
      <ol className="flex flex-col gap-1 text-[11px] font-mono">
        {rows.slice(0, 10).map((r) => {
          const pct = (Math.abs(r.delta) / maxAbs) * 100;
          const pos = r.delta >= 0;
          return (
            <li key={r.feature} className="grid grid-cols-[10rem_1fr_4rem] items-center gap-2">
              <span className="truncate text-slate-300" title={r.feature}>
                {r.feature}
              </span>
              <span className="relative h-2 rounded bg-slate-800">
                <span
                  className={[
                    "absolute top-0 h-2 rounded",
                    pos ? "left-1/2 bg-emerald-500/80" : "right-1/2 bg-rose-500/80",
                  ].join(" ")}
                  style={{ width: `${pct / 2}%` }}
                />
                <span className="absolute left-1/2 top-0 h-2 w-px bg-slate-600" />
              </span>
              <span
                className={[
                  "text-right tabular-nums",
                  pos ? "text-emerald-300" : "text-rose-300",
                ].join(" ")}
              >
                {pos ? "+" : ""}
                {r.delta.toFixed(3)}
              </span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
