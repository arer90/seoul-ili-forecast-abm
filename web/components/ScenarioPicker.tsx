/**
 * ScenarioPicker — selects one of the six canonical Metapop SEIR-V-D
 * scenarios. Clicking "Run" fires ``epi.scenario_run`` directly via
 * the MCP proxy (no LLM round-trip), because the ODE is deterministic
 * and we don't want to pay for provider tokens on a pure data pull.
 */
"use client";

import { useState } from "react";

import { SCENARIOS } from "@/lib/constants";
import type { ScenarioName } from "@/lib/constants";
import { Button } from "./ui/button";
import { Select } from "./ui/select";

const SCENARIO_LABELS: Record<ScenarioName, string> = {
  baseline: "Baseline",
  npi_lockdown: "NPI — lockdown",
  vaccination_campaign: "Vaccination campaign",
  antiviral_prophylaxis: "Antiviral prophylaxis",
  combined_response: "Combined response",
  sensitivity_strain_mismatch: "Sensitivity — strain mismatch",
};

export interface ScenarioPickerProps {
  value: ScenarioName;
  onChange: (v: ScenarioName) => void;
  onRun: (name: ScenarioName) => Promise<void> | void;
  running?: boolean;
}

export function ScenarioPicker({
  value,
  onChange,
  onRun,
  running = false,
}: ScenarioPickerProps) {
  const [busy, setBusy] = useState(false);
  const disabled = busy || running;

  return (
    <div className="flex items-end gap-2">
      <Select
        label="Scenario"
        value={value}
        onChange={(e) => onChange(e.target.value as ScenarioName)}
        disabled={disabled}
      >
        {SCENARIOS.map((s) => (
          <option key={s} value={s}>
            {SCENARIO_LABELS[s]}
          </option>
        ))}
      </Select>
      <Button
        size="sm"
        onClick={async () => {
          if (disabled) return;
          setBusy(true);
          try {
            await onRun(value);
          } finally {
            setBusy(false);
          }
        }}
        disabled={disabled}
      >
        {disabled ? "Running…" : "Run"}
      </Button>
    </div>
  );
}
