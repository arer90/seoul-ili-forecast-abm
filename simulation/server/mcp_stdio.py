"""
simulation.server.mcp_stdio
===========================
JSON-RPC 2.0 stdio transport for the ARIA MCP server.

Protocol
--------
Follows the Model Context Protocol (MCP) 1.0 subset we need:

- ``initialize`` → {protocolVersion, capabilities, serverInfo}
- ``initialized`` notification — no-op
- ``tools/list`` → {tools: [...]}
- ``tools/call`` → {content, isError, _meta}
- ``ping`` → {}
- ``shutdown`` → cleanly exits the read loop

Framing: **newline-delimited JSON** (ndjson). One request per line,
one response per line. This is simpler than the Content-Length frames
used by some MCP SDKs and works fine for stdin/stdout piping from
Node.js / Python clients.

Robustness
----------
- Parse errors → JSON-RPC ``error.code = -32700``
- Unknown method → ``-32601``
- Handler exceptions → ``-32603`` with ``data.traceback``
- Notifications (no ``id``) silently swallow return values
- EOF on stdin → graceful exit
- Log stream stays on stderr so it never pollutes the ndjson channel
"""
from __future__ import annotations

import json
import logging
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, IO, Optional

from .mcp_epi import EpiMCPServer


log = logging.getLogger(__name__)


# ── JSON-RPC error codes ───────────────────────────────────────────────
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


# ── MCP version we claim to speak ──────────────────────────────────────
MCP_PROTOCOL_VERSION = "2024-11-05"  # current stable draft


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════
def _ok(rid: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def _err(rid: Any, code: int, message: str, data: Any = None) -> dict:
    err: dict = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": rid, "error": err}


def _write_line(stream: IO[str], obj: dict) -> None:
    line = json.dumps(obj, ensure_ascii=False)
    stream.write(line + "\n")
    stream.flush()


# ══════════════════════════════════════════════════════════════════════
# Dispatcher
# ══════════════════════════════════════════════════════════════════════
@dataclass
class StdioServer:
    """JSON-RPC 2.0 dispatcher wrapping an :class:`EpiMCPServer`.

    ``handle_line`` is pure (string → string), which keeps the loop
    trivial to unit-test in-memory without touching stdio.
    """
    server: EpiMCPServer
    methods: dict[str, Callable[[dict], Any]]
    _initialized: bool = False

    @classmethod
    def create(cls, server: Optional[EpiMCPServer] = None) -> "StdioServer":
        srv = server or EpiMCPServer()
        s = cls(server=srv, methods={})
        s.methods = {
            "initialize": s._m_initialize,
            "initialized": s._m_initialized_notif,
            "notifications/initialized": s._m_initialized_notif,
            "ping": s._m_ping,
            "shutdown": s._m_shutdown,
            "tools/list": s._m_tools_list,
            "tools/call": s._m_tools_call,
        }
        return s

    # ── Protocol handlers ─────────────────────────────────────────────
    def _m_initialize(self, params: dict) -> dict:
        # We accept any client protocolVersion; clients that care can
        # compare the number we return.
        client = params.get("clientInfo", {}) if isinstance(params, dict) else {}
        log.info(
            "[mcp] initialize from client name=%s version=%s",
            client.get("name"), client.get("version"),
        )
        self._initialized = True
        return {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {
                "tools": {"listChanged": False},
                # No resources/prompts/sampling yet.
            },
            "serverInfo": {
                "name": self.server.SERVER_NAME,
                "version": self.server.SERVER_VERSION,
            },
        }

    def _m_initialized_notif(self, params: dict) -> None:
        # MCP client's "ready" ping. Nothing to do.
        return None

    def _m_ping(self, params: dict) -> dict:
        return {}

    def _m_shutdown(self, params: dict) -> dict:
        # Signal the read loop (set by caller via exception below).
        raise _ShutdownSignal()

    def _m_tools_list(self, params: dict) -> dict:
        return {"tools": self.server.list_tools()}

    def _m_tools_call(self, params: dict) -> dict:
        if not isinstance(params, dict):
            raise _RpcError(INVALID_PARAMS, "params must be an object")
        name = params.get("name")
        if not isinstance(name, str):
            raise _RpcError(INVALID_PARAMS, "params.name must be a string")
        args = params.get("arguments") or {}
        if not isinstance(args, dict):
            raise _RpcError(INVALID_PARAMS, "params.arguments must be an object")
        result = self.server.call_tool(name, args)
        return result.to_mcp()

    # ── Line-level dispatch ───────────────────────────────────────────
    def handle_line(self, line: str) -> Optional[dict]:
        """Parse one JSON-RPC request, return the response dict (or None
        for notifications)."""
        line = line.strip()
        if not line:
            return None
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            return _err(None, PARSE_ERROR, f"invalid JSON: {e}")

        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
            return _err(msg.get("id") if isinstance(msg, dict) else None,
                        INVALID_REQUEST, "malformed JSON-RPC 2.0 envelope")

        method = msg.get("method")
        rid = msg.get("id")  # may be None for notifications
        params = msg.get("params") or {}

        handler = self.methods.get(method)
        if handler is None:
            if rid is None:  # notification; nothing to reply
                return None
            return _err(rid, METHOD_NOT_FOUND,
                        f"unknown method: {method!r}",
                        {"known_methods": sorted(self.methods)})

        try:
            result = handler(params)
        except _ShutdownSignal:
            raise  # let the loop catch it
        except _RpcError as e:
            if rid is None:
                return None
            return _err(rid, e.code, e.message, e.data)
        except Exception as e:
            log.exception("[mcp] handler %r crashed", method)
            if rid is None:
                return None
            return _err(
                rid, INTERNAL_ERROR, f"{type(e).__name__}: {e}",
                {"traceback": traceback.format_exc()},
            )

        if rid is None:
            # Notification: result discarded by protocol.
            return None
        return _ok(rid, result)


class _RpcError(Exception):
    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


class _ShutdownSignal(Exception):
    """Raised from the shutdown handler to break the read loop cleanly."""


# ══════════════════════════════════════════════════════════════════════
# Read loop
# ══════════════════════════════════════════════════════════════════════
def run_stdio_server(
    *,
    stdin: Optional[IO[str]] = None,
    stdout: Optional[IO[str]] = None,
    artifacts_dir: Optional[Path] = None,
) -> int:
    """Block on stdin, dispatch JSON-RPC messages, write to stdout.

    Exit codes:
      * 0 — clean shutdown (EOF or explicit ``shutdown`` request)
      * 2 — fatal transport error

    The log stream is redirected to stderr with a fixed prefix so the
    ndjson channel on stdout is never polluted.
    """
    _configure_stderr_logging()

    in_ = stdin if stdin is not None else sys.stdin
    out = stdout if stdout is not None else sys.stdout

    dispatcher = StdioServer.create(
        EpiMCPServer(artifacts_dir=artifacts_dir) if artifacts_dir else None
    )
    log.info(
        "[mcp] epi-mcp stdio server starting (%d tools, artifacts=%s)",
        len(dispatcher.server.list_tools()),
        dispatcher.server.artifacts_dir,
    )

    try:
        for raw in in_:
            try:
                response = dispatcher.handle_line(raw)
            except _ShutdownSignal:
                log.info("[mcp] shutdown requested; closing loop")
                break
            if response is not None:
                _write_line(out, response)
    except BrokenPipeError:
        # Client hung up on us. Not an error — just exit quietly.
        return 0
    except KeyboardInterrupt:  # pragma: no cover
        return 0
    except Exception:
        log.exception("[mcp] fatal transport error")
        return 2

    return 0


def _configure_stderr_logging() -> None:
    """Attach a stderr StreamHandler if the root logger has none.

    We deliberately do *not* call ``logging.basicConfig`` because the
    CLI entry point already owns log configuration. This is a minimal
    safety net for the case where the stdio server is run directly
    from ``python -m simulation.server.mcp_stdio``.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    h = logging.StreamHandler(stream=sys.stderr)
    h.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    ))
    root.addHandler(h)
    root.setLevel(logging.INFO)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(run_stdio_server())
