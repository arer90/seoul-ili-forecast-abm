/**
 * /api/sim/scenarios — Sprint 2026-05-06 Phase D.1.
 *
 * 14 scenarios list + metadata (paper §5.6 6 base + 8 extended). UI 의
 * scenario selector 가 호출.
 */
import { NextResponse } from "next/server";

// 14 scenarios — mirror simulation/sim/scenarios.py + scenarios_extended.py
// (paper §5.6, sprint 2026-05-06 register_extended_scenarios 후)
const SCENARIOS = [
  // Base 6 (sim/scenarios.py)
  {
    id: "baseline",
    name_ko: "기본 (개입 없음)",
    name_en: "Baseline (no intervention)",
    category: "base",
    paper_section: "§5.6 base",
    description: "No interventions, default flu biology. Reference scenario.",
  },
  {
    id: "npi_lockdown",
    name_ko: "NPI 도시봉쇄 (β -40%)",
    name_en: "NPI lockdown (β -40%)",
    category: "intervention",
    paper_section: "§4.8",
    description: "40% β reduction (distancing + masks), days 21-63.",
  },
  {
    id: "vaccination_campaign",
    name_ko: "예방접종 캠페인",
    name_en: "Vaccination campaign",
    category: "intervention",
    paper_section: "§4.5",
    description: "0.5%/day S→V draw, weeks 6-18.",
  },
  {
    id: "antiviral_prophylaxis",
    name_ko: "항바이러스 조기처방 (γ 2x)",
    name_en: "Antiviral early treatment (γ 2x)",
    category: "intervention",
    paper_section: "§3 antiviral",
    description: "γ doubled, weeks 4-10. Models early oseltamivir/baloxavir.",
  },
  {
    id: "combined_response",
    name_ko: "복합 대응 (NPI + 백신 + 항바이러스)",
    name_en: "Combined response",
    category: "intervention",
    paper_section: "§5.6",
    description: "Staggered NPI + vaccination + antiviral.",
  },
  {
    id: "sensitivity_strain_mismatch",
    name_ko: "변이 mismatch (VE 0.20)",
    name_en: "Strain mismatch (VE 0.20)",
    category: "sensitivity",
    paper_section: "§sensitivity",
    description: "Vaccine effectiveness set to 0.20 (poor strain match year).",
  },
  // Extended 8 (sim/scenarios_extended.py, sprint 2026-04-30)
  {
    id: "school_closure",
    name_ko: "학교 휴교",
    name_en: "School closure",
    category: "intervention",
    paper_section: "§4.8 school",
    description: "School closure (target gu, weeks 4-12).",
  },
  {
    id: "hospital_surge",
    name_ko: "병원 부담 surge",
    name_en: "Hospital surge",
    category: "outcome",
    paper_section: "§5.7",
    description: "Hospital ICU beds exceed capacity, triage trigger.",
  },
  {
    id: "partial_compliance",
    name_ko: "부분 NPI 준수율 (50-70%)",
    name_en: "Partial NPI compliance",
    category: "sensitivity",
    paper_section: "§sensitivity",
    description: "Real-world distancing compliance 50-70%.",
  },
  {
    id: "reactive_intervention",
    name_ko: "반응형 개입 (threshold trigger)",
    name_en: "Reactive intervention (threshold)",
    category: "intervention",
    paper_section: "§5.6",
    description: "NPI activates when ILI rate > KDCA threshold (8.6/1000).",
  },
  {
    id: "delayed_response",
    name_ko: "지연 대응 (2주 lag)",
    name_en: "Delayed response (2-week lag)",
    category: "sensitivity",
    paper_section: "§sensitivity",
    description: "NPI rolls out 2 weeks after KDCA threshold breach.",
  },
  {
    id: "vaccine_uptake_low",
    name_ko: "낮은 접종률",
    name_en: "Low vaccine uptake",
    category: "sensitivity",
    paper_section: "§4.5",
    description: "Vaccine coverage 25% (2024-25 actual: ~55-60%).",
  },
  {
    id: "subtype_a_h1n1_pdm09",
    name_ko: "Subtype A/H1N1 pdm09 우세",
    name_en: "Subtype A/H1N1 pdm09 dominance",
    category: "subtype",
    paper_section: "§2.1",
    description: "A/H1N1pdm09 strain dominance (R0 1.4-1.7).",
  },
  {
    id: "subtype_a_h3n2",
    name_ko: "Subtype A/H3N2 우세",
    name_en: "Subtype A/H3N2 dominance",
    category: "subtype",
    paper_section: "§2.1",
    description: "A/H3N2 strain dominance (R0 1.5-1.9, drift-prone).",
  },
];

export const revalidate = 86400; // 1 day cache

export async function GET() {
  return NextResponse.json({
    scenarios: SCENARIOS,
    count: SCENARIOS.length,
    categories: ["base", "intervention", "sensitivity", "outcome", "subtype"],
    source: "simulation/sim/scenarios.py + scenarios_extended.py (sprint 2026-05-06)",
  });
}
