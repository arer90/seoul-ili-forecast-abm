/**
 * Shared client/server constants. Keep small + free of secrets; env
 * reads happen in `lib/auth.ts`, `lib/turso.ts` etc.
 */

export const SEOUL_GU_ORDERED: readonly string[] = [
  "종로구", "중구", "용산구", "성동구", "광진구",
  "동대문구", "중랑구", "성북구", "강북구", "도봉구",
  "노원구", "은평구", "서대문구", "마포구", "양천구",
  "강서구", "구로구", "금천구", "영등포구", "동작구",
  "관악구", "서초구", "강남구", "송파구", "강동구",
];

export const SCENARIOS = [
  "baseline",
  "npi_lockdown",
  "vaccination_campaign",
  "antiviral_prophylaxis",
  "combined_response",
  "sensitivity_strain_mismatch",
] as const;

export type ScenarioName = (typeof SCENARIOS)[number];

export const RESPONSE_MODES = ["solo", "parallel", "synthesis", "relay"] as const;

/**
 * Display labels for the UI — kept alongside the raw names so we can
 * localise without dragging translations into components.
 */
export const MODE_LABELS: Record<string, string> = {
  solo: "Solo",
  parallel: "Parallel",
  synthesis: "Synthesis",
  relay: "Relay",
};

export const PROVIDER_LABELS: Record<string, string> = {
  anthropic: "Claude",
  google: "Gemini",
  openai: "GPT",
  ollama: "Local",
};
