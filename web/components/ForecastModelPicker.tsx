/**
 * ForecastModelPicker — dropdown for selecting which of the 53 trained
 * forecasting models the chat should *emphasise* in its replies.
 *
 * This is a UX affordance, not a routing decision. The actual forecast
 * numbers always come from the MCP ``epi.forecast`` tool, which in v22.6
 * ships a frozen tournament ensemble. What this picker does:
 *
 *   1. Reads ``/aggregates/trained-models.json`` so the list matches the
 *      real 53-model registry (NegBinGLM / ElasticNet / ... / GE-DNN).
 *   2. Passes the selected model id to ChatPanel, which stuffs it into
 *      a lightweight system message prefix ("User prefers model X, use
 *      it as the reference when possible"). LLMs that call
 *      ``epi.forecast`` can pass the id along (schema supports a
 *      ``model_name`` override); LLMs that don't simply mention the
 *      model in their explanation.
 *
 * Value ``"__auto__"`` means "let the backend decide" (ensemble /
 * recommended). That's the default so the picker is additive, not
 * required.
 */
"use client";

import { useEffect, useState } from "react";

import { useT } from "@/lib/i18n";
import { Select } from "./ui/select";

export const FORECAST_MODEL_AUTO = "__auto__";

interface ModelRow {
  rank: number;
  name: string;
  wis: number;
  family: string;
}

interface Payload {
  total_models: number;
  all: ModelRow[];
}

export interface ForecastModelPickerProps {
  value: string;
  onChange: (modelId: string) => void;
  className?: string;
}

export function ForecastModelPicker({
  value,
  onChange,
  className,
}: ForecastModelPickerProps) {
  const { t } = useT();
  const [models, setModels] = useState<ModelRow[] | null>(null);

  useEffect(() => {
    const ctl = new AbortController();
    void (async () => {
      try {
        const r = await fetch("/aggregates/trained-models.json", {
          signal: ctl.signal,
          cache: "force-cache",
        });
        if (!r.ok) return;
        const body = (await r.json()) as Payload;
        setModels(body.all);
      } catch {
        /* silent — the auto option still works */
      }
    })();
    return () => ctl.abort();
  }, []);

  return (
    <div className={className}>
      <Select
        label={t("forecastModelLabel")}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-56"
        title={t("forecastModelHint")}
      >
        <option value={FORECAST_MODEL_AUTO}>{t("forecastModelAuto")}</option>
        {models?.map((m) => (
          <option key={m.name} value={m.name}>
            #{m.rank} · {m.name} (WIS {m.wis.toFixed(2)})
          </option>
        ))}
      </Select>
    </div>
  );
}
