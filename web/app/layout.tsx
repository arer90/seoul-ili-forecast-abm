/**
 * Root layout — wraps the app in the Pretendard font and global dark
 * theme. The shell is a single-page SPA so we don't need nested layouts.
 */
import type { Metadata, Viewport } from "next";
import { Analytics } from "@vercel/analytics/next";

import "./globals.css";

export const metadata: Metadata = {
  // Short brand in the browser tab; full paper title in the description.
  title: "ABS · 적응행동 시뮬레이터 — 감염병 전파 실시간 대시보드",
  description:
    "Adaptive Behavior Simulators (ABS) — Multi-Agent Simulation of " +
    "Adaptive Behavioral Responses to Infectious Disease Transmission " +
    "Patterns. Metapopulation SEIR-V-D + 53-model ILI forecasting + " +
    "LLM consultation layer grounded in MCP tools, with live Seoul " +
    "open-data overlays (PM2.5 / weather / ER crowding / subway).",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  // The layout uses dynamic panel heights and the iOS URL bar would
  // otherwise eat part of the map. 100svh avoids that.
  viewportFit: "cover",
  themeColor: "#0b0f14",
};

/**
 * Sprint 2026-05-06 (사용자 명시): PWA install prompt 제거. 단순 web app —
 * "앱 설치" 권유 / 홈 화면 추가 / standalone window 모두 비활성. browser tab
 * 으로만 접근. 기존 등록된 service worker 는 unregister script 으로 정리.
 */
function ServiceWorkerUnregister() {
  return (
    <script
      dangerouslySetInnerHTML={{
        __html: `
          if ('serviceWorker' in navigator) {
            navigator.serviceWorker.getRegistrations()
              .then((regs) => regs.forEach((r) => r.unregister()))
              .catch(() => {});
          }
        `,
      }}
    />
  );
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ko" suppressHydrationWarning>
      <body className="min-h-svh antialiased">
        {children}
        <ServiceWorkerUnregister />
        {/* Sprint 2026-05-06 O2: Vercel Analytics (Web Vitals + page views) */}
        <Analytics />
      </body>
    </html>
  );
}
