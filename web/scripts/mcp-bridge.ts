/**
 * mcp-bridge — tiny HTTP JSON-RPC gateway in front of the Python MCP
 * server.
 *
 * The MCP server speaks stdio MCP; the Next.js Edge routes can't spawn
 * Python or hold a long-lived stdio pipe. So this Node process:
 *
 *   1. spawns `python -m simulation mcp-server` once
 *   2. tracks the request/response id correlation
 *   3. exposes `POST /rpc` that accepts JSON-RPC 2.0 requests from the
 *      Next.js `lib/mcp-client.ts`
 *   4. exposes `POST /report` that triggers the docx/pdf/pptx/xlsx
 *      skill path via `epi.generate_report`
 *
 * This file is intentionally dependency-free (no Express) so `tsx`
 * can run it straight. `node >= 20` required.
 */
import { spawn } from "node:child_process";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { createInterface } from "node:readline";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

interface RpcRequest {
  jsonrpc: "2.0";
  id: string | number;
  method: string;
  params?: unknown;
}

interface RpcResponse {
  jsonrpc: "2.0";
  id: string | number;
  result?: unknown;
  error?: { code: number; message: string; data?: unknown };
}

const PORT = Number(process.env.MCP_BRIDGE_PORT ?? 8787);
const HOST = process.env.MCP_BRIDGE_HOST ?? "127.0.0.1";

// Operator-precedence fix: `??` binds looser than `?:`, so the original
// form `env.MCP_PYTHON ?? platform === "win32" ? A : B` parsed as
// `(env.MCP_PYTHON ?? (platform === "win32")) ? A : B` — meaning MCP_PYTHON
// was ignored on Windows. Parenthesise the ternary branch.
const PY_CMD = process.env.MCP_PYTHON ?? (process.platform === "win32"
  ? ".venv\\Scripts\\python.exe"
  : ".venv/bin/python");
const PY_ARGS = (process.env.MCP_PY_ARGS ?? "-m simulation mcp-server").split(/\s+/);

// --- spawn the Python MCP server -----------------------------------------

// Resolve the project root (two levels up from web/scripts/). Python has
// to be launched from the project root because the MCP server is invoked
// via `python -m simulation`, which requires `simulation/` to be on the
// current working directory (not on PYTHONPATH automatically). Without
// this, `npx tsx web/scripts/mcp-bridge.ts` started from the `web/`
// folder would die with `No module named simulation`.
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const PROJECT_ROOT = process.env.MCP_CWD ?? resolve(__dirname, "..", "..");

console.log(`[mcp-bridge] cwd: ${PROJECT_ROOT}`);
console.log(`[mcp-bridge] spawning: ${PY_CMD} ${PY_ARGS.join(" ")}`);
const child = spawn(PY_CMD, PY_ARGS, {
  cwd: PROJECT_ROOT,
  stdio: ["pipe", "pipe", "inherit"],
  env: {
    ...process.env,
    PYTHONUNBUFFERED: "1",
    // Force UTF-8 on both stdio *and* pipe I/O so Hangul arguments
    // survive the Windows default cp949 code page. Without these,
    // `gu="강남구"` arrives at Python as "占쏙옙占쏙옙占쏙옙".
    PYTHONIOENCODING: "utf-8",
    PYTHONUTF8: "1",
  },
});
child.on("exit", (code) => {
  console.error(`[mcp-bridge] python exited with code ${code}`);
  process.exit(code ?? 1);
});

const pending = new Map<
  string | number,
  (msg: RpcResponse) => void
>();

const rl = createInterface({ input: child.stdout! });
rl.on("line", (line) => {
  if (!line.trim()) return;
  let msg: RpcResponse;
  try {
    msg = JSON.parse(line) as RpcResponse;
  } catch {
    return;
  }
  const pend = pending.get(msg.id);
  if (pend) {
    pending.delete(msg.id);
    pend(msg);
  }
});

/** Forward a JSON-RPC request to the child, await the response. */
function callRpc(req: RpcRequest, timeoutMs = 60_000): Promise<RpcResponse> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      pending.delete(req.id);
      reject(new Error(`rpc timeout: ${req.method}`));
    }, timeoutMs);
    pending.set(req.id, (msg) => {
      clearTimeout(timer);
      resolve(msg);
    });
    child.stdin!.write(JSON.stringify(req) + "\n");
  });
}

// --- HTTP endpoints ------------------------------------------------------

async function readBody(req: IncomingMessage): Promise<string> {
  const chunks: Buffer[] = [];
  for await (const c of req) chunks.push(c as Buffer);
  return Buffer.concat(chunks).toString("utf-8");
}

function sendJson(res: ServerResponse, status: number, body: unknown) {
  res.writeHead(status, { "content-type": "application/json" });
  res.end(JSON.stringify(body));
}

const server = createServer(async (req, res) => {
  if (req.method === "GET" && req.url === "/health") {
    sendJson(res, 200, { ok: true, pid: child.pid });
    return;
  }

  if (req.method === "POST" && req.url === "/rpc") {
    try {
      const body = JSON.parse(await readBody(req)) as RpcRequest;
      const msg = await callRpc(body);
      sendJson(res, 200, msg);
    } catch (e) {
      sendJson(res, 500, { error: e instanceof Error ? e.message : String(e) });
    }
    return;
  }

  if (req.method === "POST" && req.url === "/report") {
    try {
      const body = JSON.parse(await readBody(req)) as {
        format: "docx" | "pdf" | "pptx" | "xlsx";
        pinned: unknown;
        context: unknown;
      };
      const msg = await callRpc(
        {
          jsonrpc: "2.0",
          id: `report-${Date.now()}`,
          method: "tools/call",
          params: {
            name: "epi.generate_report",
            arguments: body,
          },
        },
        180_000,
      );
      sendJson(res, 200, msg.result ?? msg);
    } catch (e) {
      sendJson(res, 500, { error: e instanceof Error ? e.message : String(e) });
    }
    return;
  }

  res.writeHead(404, { "content-type": "text/plain" });
  res.end("not found");
});

server.listen(PORT, HOST, () => {
  console.log(`[mcp-bridge] http://${HOST}:${PORT}`);
});

function shutdown() {
  console.log("[mcp-bridge] shutting down");
  server.close();
  if (!child.killed) child.kill("SIGTERM");
  setTimeout(() => process.exit(0), 500);
}
process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
