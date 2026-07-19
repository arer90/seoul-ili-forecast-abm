/**
 * ModelPicker — a grouped <optgroup> select for the "forecast model"
 * axis in a reply. The 50 models come from the runner's registry; we
 * group them by broad family so the picker doesn't become a flat wall
 * of 50 names.
 *
 * This is the *forecasting* model picker (i.e. which of the 50 ILI
 * models the LLM should reference). Provider model choice lives in
 * the ChatPanel next to the provider picker.
 */
"use client";

import { Select } from "./ui/select";

export interface ModelGroup {
  label: string;
  items: Array<{ value: string; label: string }>;
}

/**
 * Static fallback groups. The UI can also pull this list at runtime
 * from `epi.list_models` — but having a static seed lets the picker
 * render before the MCP round-trip completes.
 */
export const DEFAULT_MODEL_GROUPS: ModelGroup[] = [
  {
    label: "Classical",
    items: [
      { value: "LinearRegression", label: "Linear Regression" },
      { value: "Ridge", label: "Ridge" },
    ],
  },
  {
    label: "Epi / Bayesian",
    items: [
      { value: "SEIR_vanilla", label: "SEIR (vanilla)" },
      { value: "SEIR_rt_kalman", label: "SEIR + Rt Kalman" },
      { value: "BayesianStructural", label: "Bayesian Structural" },
      { value: "StateSpace_ARIMAX", label: "State-space ARIMAX" },
    ],
  },
  {
    label: "Machine Learning",
    items: [
      { value: "RandomForest", label: "Random Forest" },
      { value: "GradientBoosting", label: "Gradient Boosting" },
      { value: "XGBoost", label: "XGBoost" },
      { value: "LightGBM", label: "LightGBM" },
      { value: "CatBoost", label: "CatBoost" },
    ],
  },
  {
    label: "Deep Learning",
    items: [
      { value: "TabularDNN", label: "TabularDNN (attention + FM)" },
      { value: "TabularDNN_Lite", label: "TabularDNN-Lite (MLP)" },
      { value: "DNN_Conformal", label: "DNN + Conformal PI" },
      { value: "LSTM_Attn", label: "LSTM + Attention" },
      { value: "TCN", label: "TCN" },
      { value: "iTransformer", label: "iTransformer" },
      { value: "TimesNet", label: "TimesNet" },
    ],
  },
  {
    label: "Graph / Spatial",
    items: [
      { value: "GE_DNN", label: "GE-DNN (commuter GCN)" },
      { value: "HeatGNN", label: "HeatGNN" },
    ],
  },
  {
    label: "Ensemble",
    items: [
      { value: "Ensemble_OOF_Softmax", label: "Ensemble — OOF R² softmax" },
      { value: "Ensemble_Caruana", label: "Ensemble — Caruana forward" },
      { value: "Ensemble_Tournament", label: "Ensemble — 3-stage tournament" },
    ],
  },
];

export interface ModelPickerProps {
  value: string;
  onChange: (v: string) => void;
  groups?: ModelGroup[];
  label?: string;
}

export function ModelPicker({
  value,
  onChange,
  groups = DEFAULT_MODEL_GROUPS,
  label = "Forecasting model",
}: ModelPickerProps) {
  return (
    <Select
      label={label}
      value={value}
      onChange={(e) => onChange(e.target.value)}
    >
      {groups.map((g) => (
        <optgroup key={g.label} label={g.label}>
          {g.items.map((m) => (
            <option key={m.value} value={m.value}>
              {m.label}
            </option>
          ))}
        </optgroup>
      ))}
    </Select>
  );
}
