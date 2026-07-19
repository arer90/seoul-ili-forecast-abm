/**
 * Route-level loading UI — shown while the App Router streams initial
 * data for /. Kept minimal because AppShell has its own loader.
 */
export default function Loading() {
  return (
    <div
      role="status"
      aria-live="polite"
      className="flex min-h-svh items-center justify-center bg-[var(--bg)] text-sm text-slate-400"
    >
      <div className="flex items-center gap-3">
        <span
          className="h-2 w-2 animate-pulse rounded-full bg-sky-400"
          aria-hidden
        />
        ABS · 적응행동 시뮬레이터 로딩 중…
      </div>
    </div>
  );
}
