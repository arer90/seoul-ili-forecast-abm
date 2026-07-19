/**
 * SSE helpers — converting a ``ReadableStream<Uint8Array>`` of ndjson
 * lines into SSE frames and back. Used by the chat route to stream
 * ``TaggedEvent`` objects to the browser.
 */

export function ndjsonToSSE(
  source: ReadableStream<Uint8Array>,
): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();
  return new ReadableStream({
    async start(controller) {
      const reader = source.getReader();
      let buf = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.trim()) continue;
          controller.enqueue(encoder.encode(`data: ${line}\n\n`));
        }
      }
      if (buf.trim()) {
        controller.enqueue(encoder.encode(`data: ${buf}\n\n`));
      }
      controller.enqueue(encoder.encode(`event: done\ndata: {}\n\n`));
      controller.close();
    },
  });
}

export const SSE_HEADERS: Record<string, string> = {
  "content-type": "text/event-stream; charset=utf-8",
  "cache-control": "no-cache, no-transform",
  "connection": "keep-alive",
  "x-accel-buffering": "no",
};

/** Client-side consumer — yields parsed event objects. */
export async function* readSSE(
  resp: Response,
): AsyncGenerator<Record<string, unknown>> {
  if (!resp.body) return;
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const chunks = buf.split("\n\n");
    buf = chunks.pop() ?? "";
    for (const c of chunks) {
      const dataLine = c
        .split("\n")
        .find((l) => l.startsWith("data:"))
        ?.slice(5)
        .trim();
      if (!dataLine) continue;
      try {
        yield JSON.parse(dataLine) as Record<string, unknown>;
      } catch {
        // ignore malformed frames
      }
    }
  }
}
