/**
 * NewChatButton — sprint 2026-05-06 (#1 사용자 critique "새 대화 안 만들어져"):
 * 이전엔 lazy create (closeSession 만 → empty composer, 첫 message 시 session
 * 생성) — 사용자 expectation 과 mismatch 였음 (sidebar 에 entry 안 보여
 * "안 됐다" 인식). 이제 **eager create** — 클릭 즉시 newSession 호출, sidebar
 * 에 entry 즉시 표시.
 *
 * Turso 가 unreachable (store.disabled) 시 newSession 가 fail — fallback 으로
 * closeSession 호출 (empty composer 만 — local dev mode 작동 보장).
 */
"use client";

import { useT } from "@/lib/i18n";
import { useSessionStore } from "@/lib/use-session-store";

import { Button } from "./ui/button";

export function NewChatButton() {
  const { t } = useT();
  const { closeSession, newSession, disabled } = useSessionStore();

  const onClick = async () => {
    if (disabled) {
      // Turso unreachable — local dev mode. Just empty the composer.
      closeSession();
      return;
    }
    try {
      await newSession({});
    } catch {
      // Persistence failed — still empty composer so user can type.
      closeSession();
    }
  };

  return (
    <Button
      size="sm"
      variant="default"
      className="w-full"
      onClick={onClick}
      title={t("newChat")}
    >
      <span>{t("newChat")}</span>
    </Button>
  );
}
