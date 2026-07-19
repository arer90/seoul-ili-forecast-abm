/**
 * POST /api/chat-cli — stream a locally authenticated Claude CLI prompt.
 *
 * Body: { messages: [{ role, content }], context?: { ...simulation state },
 *         sessionId?: string }
 * Response: the same tagged JSON SSE events consumed by AriaSimPanel.
 *
 * Memory architecture
 * -------------------
 * Short-term (within session):
 *   When `sessionId` is supplied the last N turns are loaded from
 *   fd_messages (Turso) and prepended to the conversation so ARIA
 *   remembers what was said earlier in the same session.  The user
 *   turn and completed assistant reply are persisted after streaming.
 *
 * Long-term (cross-session):
 *   Per-user facts from `aria_user_memory` (Turso) are injected as a
 *   [장기기억] block in the system prompt.  Key facts extracted from
 *   the new user message are upserted after each turn.
 *
 * RAG grounding (TASK 1):
 *   Before spawning Claude the user query is passed to the Python
 *   vector-RAG (simulation.server.rag.semantic_search → LanceDB +
 *   sentence-transformers) via rag-bridge.ts.  Results are injected as
 *   a [GraphRAG 근거] block in the prompt.  Falls back gracefully to the
 *   static citation catalogue (inside Python) when the vector index is
 *   not yet built.  When Python is entirely unavailable the block is
 *   omitted and Claude continues with KDCA wiki + digest grounding.
 */
import { spawn } from "node:child_process";
import type { NextRequest } from "next/server";

import ariaWiki from "@/lib/aria-wiki.json";
import { requireAuth } from "@/lib/auth";
import { SSE_HEADERS } from "@/lib/util/sse";
import { ragQuery, formatRagBlock } from "@/lib/rag-bridge";
import {
  getMemoryBlock,
  upsertMemory,
  extractMemoryFacts,
} from "@/lib/user-memory";
import { fdUidOf } from "@/lib/auth";
import { appendMessage, listMessages } from "@/lib/history-db";

export const runtime = "nodejs";

type PromptMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

const W = ariaWiki as {
  law: {
    name: string;
    authority: string;
    classification: string;
    provisions: string[];
    sources: string[];
    caveat: string;
  };
  kdca_data: {
    influenza?: {
      legal_grade: string;
      transmission: string;
      vaccine_available: boolean;
    };
    sentinel_ili_latest?: {
      season: number;
      week: string;
      age_group: string;
      ili_rate: number;
      unit: string;
    };
  };
};

const KDCA_WIKI =
  "\n\n[법령·KDCA wiki] — 한국 감염병 법령 + 질병관리청 사실 (이 블록을 1차 근거로; 조문·등급 환각 금지):\n" +
  `· 법: ${W.law.name}, 소관 ${W.law.authority}.\n` +
  `· 분류: ${W.law.classification}\n` +
  W.law.provisions.map((p) => `· ${p}`).join("\n") +
  (W.kdca_data.influenza
    ? `\n· KDCA disease_master: 인플루엔자=${W.kdca_data.influenza.legal_grade}, ${W.kdca_data.influenza.transmission}, 백신 ${W.kdca_data.influenza.vaccine_available ? "있음" : "없음"}.`
    : "") +
  (W.kdca_data.sentinel_ili_latest
    ? `\n· 최근 표본감시 ILI: ${W.kdca_data.sentinel_ili_latest.season}절기 ${W.kdca_data.sentinel_ili_latest.week} ${W.kdca_data.sentinel_ili_latest.age_group} ${W.kdca_data.sentinel_ili_latest.ili_rate} (${W.kdca_data.sentinel_ili_latest.unit}).`
    : "") +
  `\n· 출처: ${W.law.sources.join("; ")}. ${W.law.caveat}`;

const SYSTEM_GROUNDING =
  "You are the ARIA consultant for the Seoul flu forecasting project. " +
  "Citation policy for numeric / district / disease / date claims — try " +
  "tiers in order, never refuse just because the top tier is missing:\n" +
  "  1. MCP tools when available (epi.query_db, epi.forecast, " +
  "epi.rt_estimate, epi.scenario_run, epi.validity_check, " +
  "epi.literature_rag) — cite [tool:<name>].\n" +
  "  1b. [법령·KDCA wiki] block (아래 첨부) — 한국 감염병예방법 + 질병관리청 " +
  "사실(법정등급·표본감시·신고·방역조치·예방접종). 한국 법령 / 법정감염병 / 신고 / " +
  "방역 / 예방접종 관련 주장은 이 블록을 우선 인용 — [law:감염병예방법] 또는 " +
  "[data:KDCA]. 조문 번호나 감염병 등급을 임의로 지어내지 말 것.\n" +
  "  2. The [실데이터 digest] block in the user-supplied system prompt " +
  "(real numbers baked from epi_real_seoul.db at deploy time) — " +
  "cite [tool:data_digest].\n" +
  "  3. The [현재 컨텍스트] block (sim day / selected gu / scenario) — " +
  "cite [tool:sim_context].\n" +
  "  4. Established epi knowledge (KDCA / WHO / peer-reviewed " +
  "meta-analyses, Claude training cutoff) with explicit caveat — " +
  "cite [기존 문헌].\n" +
  "Always state which tier the evidence came from. For simulation / " +
  "prediction asks where MCP is unavailable, do hand-calculation from " +
  "digest + SEIR knowledge and prepend '(추정 — 실 모델 미연동)'.\n" +
  "수식·지표 정책: ① 모든 공식은 LaTeX로 출력 — 인라인 $...$, 블록 $$...$$ " +
  "(예: $$\\mathrm{WIS}=\\sum_{\\alpha}\\left[\\tfrac{\\alpha}{2}(u_\\alpha-l_\\alpha)+(l_\\alpha-y)\\mathbb{1}(y<l_\\alpha)+(y-u_\\alpha)\\mathbb{1}(y>u_\\alpha)\\right]$$). " +
  "평문 Σ/𝟙 기호 나열 금지. ② 공식을 제시하면 반드시 각 기호·항의 의미를 한국어로 풀어 해석하고" +
  "(예: y=관측값, l_α·u_α=예측구간 하/상한, 𝟙(·)=지시함수), 값의 방향(낮을수록/높을수록 좋음)과 " +
  "실무적 함의를 1–2문장 덧붙일 것. 공식만 단독으로 출력 금지.\n\n" +
  "커스텀 agent 생성 정책 — 사용자가 특정 프로필의 agent를 지도에 추가 요청할 때:\n" +
  "  형식: 답변 텍스트 내에 [CREATE_AGENT]{\"age\":<number>,\"comorbidity\":[<조건...>],\"gu\":\"<구이름>\"} 토큰을 삽입.\n" +
  "  comorbidity 허용 값: \"obesity\"(비만), \"diabetes\"(당뇨병), \"hypertension\"(고혈압), " +
  "\"hypercholesterolemia\"(고지혈증) — 이 4가지만 (KNHANES 4대 만성질환).\n" +
  "  gu: 서울 25개 자치구 중 하나 (예: \"강남구\", \"서초구\", \"노원구\").\n" +
  "  예: 사용자 '70세 당뇨 강남구 agent 만들어' → [CREATE_AGENT]{\"age\":70,\"comorbidity\":[\"diabetes\"],\"gu\":\"강남구\"}\n" +
  "  agent 생성 후 반드시 해당 comorbidity의 인플루엔자 중증화 위험도를 설명할 것:\n" +
  "    - 당뇨병: 상대위험도 RR≈1.75 (Allard 2010 Diabetes Care; Mertz 2013 BMJ)\n" +
  "    - 비만(BMI≥25): RR≈1.45 (Mertz 2013 BMJ 메타분석)\n" +
  "    - 고혈압: RR≈1.25 (Wang 2021 기반)\n" +
  "    - 고지혈증: RR≈1.15\n" +
  "    - 2개 이상 동반: 승산 곱, 최대 3.0 cap (복합부담 고위험군)\n" +
  "  이 agent는 가설 What-if 시나리오 (개인 감시 아님, KNHANES 유병률 모델 유도) 임을 명시." +
  KDCA_WIKI;

function isPromptMessage(value: unknown): value is PromptMessage {
  if (!value || typeof value !== "object") return false;
  const message = value as Record<string, unknown>;
  return (
    (message.role === "system" ||
      message.role === "user" ||
      message.role === "assistant") &&
    typeof message.content === "string"
  );
}

function buildPrompt(
  messages: PromptMessage[],
  context: unknown,
  ragBlock: string,
  memoryBlock: string,
  historyBlock: string,
): string {
  const roleLabels: Record<PromptMessage["role"], string> = {
    system: "System",
    user: "User",
    assistant: "Assistant",
  };
  const contextBlock =
    context === undefined ? "제공되지 않음" : JSON.stringify(context, null, 2);
  const conversation = messages
    .map((message) => `${roleLabels[message.role]}: ${message.content}`)
    .join("\n");

  return (
    `${SYSTEM_GROUNDING}` +
    memoryBlock +
    ragBlock +
    `\n\n현재 시뮬레이션 컨텍스트:\n${contextBlock}\n\n` +
    historyBlock +
    `${conversation}\n`
  );
}

function streamClaude(prompt: string, signal: AbortSignal): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  let child: ReturnType<typeof spawn> | null = null;
  let cancelled = false;

  return new ReadableStream({
    start(controller) {
      const proc = spawn("claude", ["-p", prompt], {
        stdio: ["ignore", "pipe", "pipe"] as const,
      });
      child = proc;
      const stdoutDecoder = new TextDecoder();
      const stderrDecoder = new TextDecoder();
      let stderr = "";
      let closed = false;
      let timedOut = false;

      const write = (event: Record<string, unknown>) => {
        if (!closed && !cancelled) {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`));
        }
      };
      const finish = () => {
        if (closed) return;
        closed = true;
        clearTimeout(timeout);
        signal.removeEventListener("abort", abort);
        if (!cancelled) {
          controller.enqueue(encoder.encode("event: done\ndata: {}\n\n"));
          controller.close();
        }
      };
      const abort = () => {
        cancelled = true;
        child?.kill("SIGTERM");
      };
      const timeout = setTimeout(() => {
        timedOut = true;
        child?.kill("SIGKILL");
      }, CHAT_TIMEOUT_MS);

      if (signal.aborted) abort();
      else signal.addEventListener("abort", abort, { once: true });

      proc.stdout.on("data", (chunk: Buffer) => {
        const delta = stdoutDecoder.decode(chunk, { stream: true });
        if (delta) write({ type: "text", delta, providerId: "claude-cli" });
      });
      proc.stderr.on("data", (chunk: Buffer) => {
        stderr += stderrDecoder.decode(chunk, { stream: true });
        if (stderr.length > 16_000) stderr = stderr.slice(-16_000);
      });
      proc.on("error", (error) => {
        write({ type: "error", message: error.message, providerId: "claude-cli" });
        finish();
      });
      proc.on("close", (code, closeSignal) => {
        const trailing = stdoutDecoder.decode();
        if (trailing) {
          write({ type: "text", delta: trailing, providerId: "claude-cli" });
        }
        stderr += stderrDecoder.decode();
        if (timedOut) {
          // ★ C1 (Codex/Gemini): emit a VISIBLE text delta first, so the user sees a
          //   timeout message instead of an empty response. The error event alone was
          //   never rendered (client accumulator only reads delta/content/text/token).
          const secs = Math.round(CHAT_TIMEOUT_MS / 1000);
          write({
            type: "text",
            delta: `\n\n⏱️ 응답이 ${secs}초를 초과해 중단되었습니다. 질문 범위를 좁히거나 더 구체적으로 물어봐 주세요. (서버 계산 한도)`,
            providerId: "claude-cli",
          });
          write({
            type: "error",
            message: `claude CLI timed out after ${secs} seconds`,
            providerId: "claude-cli",
          });
        } else if (code !== 0) {
          write({
            type: "error",
            message:
              stderr.trim() ||
              `claude CLI exited with code ${code ?? "null"} (${closeSignal ?? "no signal"})`,
            providerId: "claude-cli",
          });
        }
        finish();
      });
    },
    cancel() {
      cancelled = true;
      child?.kill("SIGTERM");
    },
  });
}

/** Hard timeout for the claude CLI subprocess (ms). Configurable via env (C1). */
const CHAT_TIMEOUT_MS = Number(process.env.CHAT_TIMEOUT_MS) || 60_000;

/** Max recent turns to include as short-term session context. */
const MAX_HISTORY_TURNS = 10;

export async function POST(req: NextRequest): Promise<Response> {
  const authFail = requireAuth(req);
  if (authFail) return authFail;

  let body: { messages?: unknown; context?: unknown; sessionId?: unknown };
  try {
    body = (await req.json()) as {
      messages?: unknown;
      context?: unknown;
      sessionId?: unknown;
    };
  } catch {
    return new Response("invalid JSON", { status: 400 });
  }

  if (!Array.isArray(body.messages) || !body.messages.every(isPromptMessage)) {
    return new Response("messages must be an array of { role, content }", {
      status: 400,
    });
  }

  const messages = body.messages as PromptMessage[];
  const sessionId =
    typeof body.sessionId === "string" ? body.sessionId : undefined;
  const uid = fdUidOf(req) ?? undefined;

  // ── 1. Extract user query for RAG + memory ───────────────────────────
  const lastUserMsg =
    [...messages].reverse().find((m) => m.role === "user")?.content ?? "";

  // ── 2. RAG grounding (parallel with memory fetch) ────────────────────
  const [ragHits, memoryBlock] = await Promise.all([
    ragQuery(lastUserMsg, 4, 6_000).catch(() => null),
    uid ? getMemoryBlock(uid) : Promise.resolve(""),
  ]);
  const ragBlock = formatRagBlock(ragHits);

  // ── 3. Short-term: load recent session history ────────────────────────
  let historyBlock = "";
  if (sessionId && uid) {
    try {
      const past = await listMessages(sessionId);
      const recent = past.slice(-MAX_HISTORY_TURNS * 2); // user+assistant pairs
      if (recent.length > 0) {
        const lines = recent.map((m) => {
          const label =
            m.role === "user"
              ? "User"
              : m.role === "assistant"
              ? "Assistant"
              : "Tool";
          return `${label}: ${m.content}`;
        });
        historyBlock =
          `[이전 대화 (단기기억 — 이 세션)]:\n${lines.join("\n")}\n\n`;
      }
    } catch {
      // history unavailable — continue without it
    }
  }

  // ── 4. Build prompt + stream ──────────────────────────────────────────
  const prompt = buildPrompt(
    messages,
    body.context,
    ragBlock,
    memoryBlock,
    historyBlock,
  );

  // ── 5. Collect full answer for memory + history persistence ──────────
  //    We tee the stream: SSE events go to the client unchanged; we also
  //    reconstruct the full text to persist after streaming completes.
  const encoder = new TextEncoder();
  let fullAnswer = "";

  const sourceStream = streamClaude(prompt, req.signal);
  const transformedStream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const reader = sourceStream.getReader();
      const dec = new TextDecoder();
      try {
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          // Parse delta from each SSE chunk to accumulate the answer.
          const text = dec.decode(value, { stream: true });
          for (const line of text.split("\n")) {
            const m = line.match(/^data:\s*(.+)$/);
            if (m) {
              try {
                const ev = JSON.parse(m[1]) as Record<string, unknown>;
                const piece =
                  ev.delta ?? ev.content ?? ev.text ?? ev.token;
                if (typeof piece === "string") fullAnswer += piece;
              } catch {
                // keepalive or non-JSON
              }
            }
          }
          controller.enqueue(value);
        }
        // Stream done — persist history + memory (fire-and-forget).
        if (sessionId && uid && lastUserMsg) {
          persistTurnAsync(
            sessionId,
            uid,
            lastUserMsg,
            fullAnswer,
          ).catch(() => {});
        }
        if (uid && lastUserMsg) {
          persistMemoryAsync(uid, lastUserMsg).catch(() => {});
        }
      } catch (e) {
        controller.error(e);
        return;
      }
      controller.close();
    },
    cancel() {
      // propagate cancel upstream
      sourceStream.cancel().catch(() => {});
    },
  });

  return new Response(transformedStream, { headers: SSE_HEADERS });
}

/** Persist one user+assistant turn to fd_messages. Best-effort. */
async function persistTurnAsync(
  sessionId: string,
  uid: string,
  userContent: string,
  assistantContent: string,
): Promise<void> {
  if (!process.env.TURSO_URL) return;
  try {
    // user turn
    await appendMessage(sessionId, uid, {
      role: "user",
      content: userContent,
      provider_id: "claude-cli",
    });
    // assistant turn — only if we got a non-empty answer
    if (assistantContent.trim()) {
      await appendMessage(sessionId, uid, {
        role: "assistant",
        content: assistantContent.trim(),
        provider_id: "claude-cli",
      });
    }
  } catch {
    // best-effort — never crash the streaming response
  }
}

/** Extract key facts from user message and upsert into user memory. */
async function persistMemoryAsync(
  uid: string,
  userContent: string,
): Promise<void> {
  const facts = extractMemoryFacts(userContent);
  for (const { key, value } of facts) {
    await upsertMemory(uid, key, value).catch(() => {});
  }
}
