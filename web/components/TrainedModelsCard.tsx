/**
 * TrainedModelsCard — read `public/aggregates/trained-models.json` and
 * render the top-10-by-WIS (default) with a "view all 53" toggle.
 *
 * Surfaces the fact that the app is backed by a real 53-model registry
 * (post_E v22.6), which was previously invisible to the user — one of
 * the "학습된 모델이 없는데?" complaints.
 */
"use client";

import { useEffect, useState } from "react";

import { useT } from "@/lib/i18n";
import { HelpIcon } from "./ui/HelpIcon";
import { LastUpdated } from "./ui/LastUpdated";

interface ModelRow {
  rank: number;
  name: string;
  r2: number;
  rmse: number;
  wis: number;
  crps: number;
  cov95: number;
  mape: number;
  family: string;
}

interface Payload {
  version: string;
  timestamp: string;
  total_models: number;
  source: string;
  metric_hint: string;
  top: ModelRow[];
  all: ModelRow[];
}

const FAMILY_COLOR: Record<string, string> = {
  classical: "bg-slate-700/60 text-slate-200",
  epi_bayes: "bg-emerald-700/40 text-emerald-100",
  tree_ml: "bg-amber-700/40 text-amber-100",
  svm_krr: "bg-rose-700/40 text-rose-100",
  dl_tabular: "bg-indigo-700/40 text-indigo-100",
  seq_dl: "bg-sky-700/40 text-sky-100",
  graph: "bg-fuchsia-700/40 text-fuchsia-100",
  foundation: "bg-teal-700/40 text-teal-100",
  ensemble: "bg-violet-700/40 text-violet-100",
  other: "bg-slate-700/40 text-slate-300",
};

export function TrainedModelsCard() {
  const { t, locale } = useT();
  const [data, setData] = useState<Payload | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [showAll, setShowAll] = useState(false);

  useEffect(() => {
    const ctl = new AbortController();
    void (async () => {
      try {
        const r = await fetch("/aggregates/trained-models.json", {
          signal: ctl.signal,
          cache: "force-cache",
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        setData((await r.json()) as Payload);
      } catch (e) {
        if ((e as Error).name === "AbortError") return;
        setErr(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => ctl.abort();
  }, []);

  if (err) {
    return (
      <div className="rounded-md border border-slate-800 bg-slate-900/40 p-2 text-[11px] text-slate-400">
        {t("trainedModels")}: {err}
      </div>
    );
  }
  if (!data) {
    return (
      <div className="rounded-md border border-slate-800 bg-slate-900/40 p-2 text-[11px] text-slate-500">
        {t("trainedModels")}…
      </div>
    );
  }

  const rows = showAll ? data.all : data.top;

  return (
    <div className="rounded-md border border-slate-800 bg-slate-900/40 p-2">
      <div className="mb-1.5 flex items-baseline justify-between gap-2">
        <div>
          <div className="flex items-center gap-1 text-[12px] font-medium text-slate-100">
            <span>{t("trainedModels")}</span>
            <HelpIcon
              label={t("helpTrainedModels")}
              content={t("helpTrainedModels")}
              side="bottom"
            />
            {data.timestamp ? (
              <LastUpdated at={data.timestamp} className="ml-1.5" />
            ) : null}
          </div>
          <div className="text-[10px] text-slate-500">
            {t("trainedModelsHint", { n: data.total_models })}
          </div>
        </div>
        <button
          type="button"
          onClick={() => setShowAll((v) => !v)}
          className="rounded border border-slate-700 px-1.5 py-0.5 text-[10px] text-slate-300 hover:bg-slate-800"
        >
          {showAll
            ? t("hideAllModels")
            : t("viewAllModels", { n: data.total_models })}
        </button>
      </div>
      <div
        className={[
          "grid grid-cols-1 gap-1 sm:grid-cols-2",
          showAll ? "max-h-56 overflow-y-auto pr-1" : "",
        ].join(" ")}
      >
        {rows.map((r) => (
          <div
            key={r.name}
            className="flex items-center justify-between gap-1 rounded border border-slate-800 bg-slate-950/40 px-1.5 py-1"
            title={`rank=${r.rank} family=${r.family}\nR²=${r.r2} RMSE=${r.rmse}\nWIS=${r.wis} CRPS=${r.crps} cov95=${r.cov95} MAPE=${r.mape}%`}
          >
            <span className="flex min-w-0 items-center gap-1">
              <span className="w-5 shrink-0 text-right text-[10px] text-slate-500">
                {r.rank}
              </span>
              <span
                className={[
                  "shrink-0 rounded px-1 py-[1px] text-[9px] uppercase tracking-wide",
                  FAMILY_COLOR[r.family] ?? FAMILY_COLOR.other,
                ].join(" ")}
              >
                {r.family.replace("_", " ")}
              </span>
              <span className="truncate text-[11px] text-slate-100">
                {r.name}
              </span>
            </span>
            <span className="shrink-0 font-mono text-[10px] text-slate-400">
              {locale === "ko" ? "WIS" : "WIS"} {r.wis.toFixed(2)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
