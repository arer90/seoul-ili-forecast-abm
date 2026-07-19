/**
 * Legacy AppShell — preserved at /legacy for A/B compare or rollback.
 *
 * The root URL "/" now serves the web_prototype prototype (see app/page.tsx
 * for the redirect). This route keeps the existing chat workspace alive so
 * we can compare side-by-side or revert by swapping the redirect target.
 */
import dynamic from "next/dynamic";

const AppShell = dynamic(() => import("@/components/AppShell"), {
  ssr: false,
  loading: () => (
    <div className="flex min-h-svh items-center justify-center text-sm text-slate-400">
      Loading workspace…
    </div>
  ),
});

export default function LegacyPage() {
  return <AppShell />;
}
