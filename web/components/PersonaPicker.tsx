/**
 * PersonaPicker — consultation-mode dropdown (general / epi / model /
 * simulation / clinical). Sits next to ProviderPicker and
 * ForecastModelPicker in the chat header.
 *
 * This is a FRAMING control, not a model-routing control. The picked
 * persona prepends a short system prompt (see ``lib/personas.ts``);
 * the LLM still runs the same provider and calls the same MCP tools.
 * Keeping it as a plain <Select> visually signals that parity.
 *
 * Why it exists
 *   The MPH paper reviewer (``model-advisor``) warned that adding
 *   personas as "new models" would let the LLM double-dip into the
 *   53-model OOF ranking when citing numbers. By labelling this
 *   explicitly as a consultation mode and restricting the persona
 *   prompts to PAPER_PRIMARY_11 + Ensemble-NNLS, we keep the paper's
 *   evaluation axis clean while still giving the user a useful
 *   framing lever.
 */
"use client";

import { useT, type Locale } from "@/lib/i18n";
import { PERSONAS, type PersonaDef } from "@/lib/personas";

import { Select } from "./ui/select";

function personaLabel(p: PersonaDef, locale: Locale): string {
  return locale === "ko" ? p.label_ko : p.label_en;
}

function personaTagline(p: PersonaDef, locale: Locale): string {
  return locale === "ko" ? p.tagline_ko : p.tagline_en;
}

export interface PersonaPickerProps {
  value: string;
  onChange: (personaId: string) => void;
  className?: string;
}

export function PersonaPicker({
  value,
  onChange,
  className,
}: PersonaPickerProps) {
  const { t, locale } = useT();
  const active = PERSONAS.find((p) => p.id === value);
  return (
    <div className={className}>
      <Select
        label={t("personaLabel")}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-40"
        title={active ? personaTagline(active, locale) : t("personaHint")}
      >
        {PERSONAS.map((p) => (
          <option key={p.id} value={p.id}>
            {personaLabel(p, locale)}
          </option>
        ))}
      </Select>
    </div>
  );
}
