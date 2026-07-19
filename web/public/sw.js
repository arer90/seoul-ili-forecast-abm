// Service Worker — Sprint 2026-05-06 (사용자 명시 PWA install 제거).
// 이전 sprint 의 cache 정리 + 자체 unregister.
// 즉 기존 사용자 browser 에 등록된 sw 가 이 빈 sw 로 교체 → cache 정리 후
// 다음 page reload 시 layout.tsx 의 ServiceWorkerUnregister 가 완전 제거.

self.addEventListener("install", () => {
  // Skip waiting so this minimal sw activates immediately
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      // Clear all old caches
      const keys = await caches.keys();
      await Promise.all(keys.map((k) => caches.delete(k)));
      // Take control of all open clients so they unregister via layout.tsx
      await self.clients.claim();
      // Self-unregister
      await self.registration.unregister();
    })(),
  );
});

// Pass-through fetch — no cache, no offline fallback.
self.addEventListener("fetch", () => {});
