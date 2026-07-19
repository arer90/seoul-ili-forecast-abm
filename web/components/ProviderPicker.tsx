/**
 * ProviderPicker — checkboxes for the 4 LLM providers.
 *
 * The Ollama entry is hidden when ``NEXT_PUBLIC_HIDE_OLLAMA=1``; the
 * deployed Vercel demo can't run a local model so we remove it from
 * the UI rather than leave a disabled checkbox.
 */
"use client";

import { PROVIDER_LABELS } from "@/lib/constants";
import type { ProviderId } from "@/lib/providers/types";

const PROVIDER_COLOUR: Record<ProviderId, string> = {
  anthropic: "bg-orange-500",
  google: "bg-blue-500",
  openai: "bg-emerald-500",
  ollama: "bg-violet-500",
};

export interface ProviderPickerProps {
  selected: ProviderId[];
  onChange: (next: ProviderId[]) => void;
  /** IDs we were told not to render (e.g. Ollama in prod). */
  hidden?: ProviderId[];
  /** Single-select mode for solo. */
  single?: boolean;
  /** Per-provider availability — disables the chip when false. */
  availability?: Partial<Record<ProviderId, boolean>>;
}

const ALL: ProviderId[] = ["anthropic", "google", "openai", "ollama"];

export function ProviderPicker({
  selected,
  onChange,
  hidden = [],
  single = false,
  availability,
}: ProviderPickerProps) {
  const visible = ALL.filter((id) => !hidden.includes(id));
  const isAvail = (id: ProviderId) =>
    availability ? availability[id] !== false : true;

  const toggle = (id: ProviderId) => {
    if (!isAvail(id)) return;
    if (single) {
      onChange([id]);
      return;
    }
    if (selected.includes(id)) {
      onChange(selected.filter((x) => x !== id));
    } else {
      onChange([...selected, id]);
    }
  };

  return (
    <div
      role={single ? "radiogroup" : "group"}
      aria-label="Model providers"
      className="flex flex-wrap gap-2"
    >
      {visible.map((id) => {
        const active = selected.includes(id);
        const available = isAvail(id);
        return (
          <button
            key={id}
            role={single ? "radio" : "checkbox"}
            aria-checked={active}
            aria-disabled={!available}
            disabled={!available}
            onClick={() => toggle(id)}
            title={
              !available
                ? id === "ollama"
                  ? "Ollama not reachable at OLLAMA_BASE_URL"
                  : `${PROVIDER_LABELS[id] ?? id} — API key missing`
                : undefined
            }
            className={[
              "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1",
              "text-xs font-medium transition-colors",
              !available
                ? "cursor-not-allowed border-slate-800 bg-slate-900/40 text-slate-500 line-through"
                : active
                  ? "border-sky-400 bg-sky-500/10 text-sky-200"
                  : "border-slate-700 bg-slate-900 text-slate-300 hover:bg-slate-800",
            ].join(" ")}
          >
            <span
              aria-hidden
              className={[
                "h-2 w-2 rounded-full",
                PROVIDER_COLOUR[id],
                !available ? "opacity-30" : active ? "opacity-100" : "opacity-50",
              ].join(" ")}
            />
            {PROVIDER_LABELS[id] ?? id}
          </button>
        );
      })}
    </div>
  );
}
