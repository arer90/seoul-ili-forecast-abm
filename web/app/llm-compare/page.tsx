/**
 * /llm-compare — LLM 비교 dashboard (sprint 2026-05-06).
 *
 * Standalone page. Open via direct URL (`http://localhost:3000/llm-compare`)
 * or `<a target="_blank">` from the main app — fully isolated from the
 * chat / map components on `/`.
 *
 * Data source: `simulation/results/llm_compare/report.json` (produced by
 * `simulation.llm_compare.runner`). Visualises 20 golden items (5
 * scenarios × 4 personas, KO/EN) × 7 pillar rubric.
 */
import { LLMCompareDashboard } from "@/components/LLMCompareDashboard";

export const metadata = {
  title: "LLM Compare — 보건역학/시뮬 평가 | MPH-Seoul",
  description:
    "Multi-LLM benchmark dashboard for the MPH-Seoul ARIA consultation layer.",
};

export default function LLMComparePage() {
  return (
    <main className="min-h-screen p-6 md:p-10 bg-slate-50 text-slate-900">
      <header className="mb-8 max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold tracking-tight">
          LLM Comparison Dashboard
        </h1>
        <p className="mt-2 text-base text-slate-600">
          보건역학 + 시뮬 질문 (20 golden items: S1 triage / S2 vaccination /
          S3 antiviral / S4 district alert / S5 SEIR-V-D 해석) × 7 pillar
          rubric. paper PART B (TRIPOD-LLM Stage 7) 평가 결과 시각화.
        </p>
        <p className="mt-1 text-xs text-slate-500">
          데이터: <code>simulation/results/llm_compare/report.json</code> ·
          생성: <code>python -m simulation.llm_compare.runner</code>
        </p>
      </header>
      <div className="max-w-6xl mx-auto">
        <LLMCompareDashboard />
      </div>
    </main>
  );
}
