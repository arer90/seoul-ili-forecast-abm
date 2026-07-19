/**
 * Root page — serves the web_prototype ABS prototype.
 *
 * 2026-05-07: User decision — the current UI for Vercel deployment is the
 * mobile-first web_prototype prototype (verified iPhone 17 Pro / iOS 26.4).
 * Static assets live in /public/abs/ (copied from /web_prototype/), and we
 * server-redirect "/" → "/abs/index.html" so the prototype owns the root URL.
 *
 * The previous AppShell (existing legacy chat workspace) is preserved at
 * /legacy via app/legacy/page.tsx so the user can A/B compare or roll back
 * with a single file edit if needed.
 *
 * API routes (/api/*), provider adapters, Turso wiring, MCP — all unchanged
 * and continue to work as the BACKEND for whichever frontend is mounted.
 */
import { redirect } from "next/navigation";

interface PageProps {
  // Next.js 14 App Router passes searchParams to server components.
  searchParams: Record<string, string | string[] | undefined>;
}

export default function Page({ searchParams }: PageProps) {
  // Preserve the ?t=<DEMO_TOKEN> param (and any others — model, v) through
  // the redirect so bootAuth() in /abs/app.jsx can exchange it for a
  // session cookie. Previous version dropped them, leaving every visitor
  // unauthed → /api/chat 401 → mockReply.
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(searchParams)) {
    if (v == null) continue;
    if (Array.isArray(v)) v.forEach((x) => qs.append(k, x));
    else qs.set(k, v);
  }
  const tail = qs.toString();
  redirect(tail ? `/abs/index.html?${tail}` : "/abs/index.html");
}
