"""
simulation.server
=================
ARIA (Stage 6) — LLM Consultation Layer server package.

Public surface kept small and dependency-light so importing
``simulation.server`` never drags in heavy ML deps (torch, duckdb).

- :class:`EpiMCPServer` — pure-Python MCP-style server exposing 10 epi
  tools over ``list_tools()`` / ``call_tool(name, args)``.
- :data:`TOOL_SPECS` — declarative schema list (MCP ``tools/list`` shape).
- :class:`CallResult` — wrapper yielded by :meth:`EpiMCPServer.call_tool`.
- :func:`validate_read_only` — SQL guard used by ``epi.query_db``.
- :func:`run_stdio_server` — JSON-RPC 2.0 ndjson stdio transport for the
  MCP protocol. Used by the ``python -m simulation mcp-server`` CLI.

See ``docs/internal/stage_plan.md`` → Stage 6a for the wiring plan.
"""
from __future__ import annotations

from .mcp_epi import (
    CallResult,
    EpiMCPServer,
    TOOL_BY_NAME,
    TOOL_SPECS,
    ToolSpec,
)
from .mcp_stdio import run_stdio_server
from .sql_guard import (
    ALLOWED_LEADING_KEYWORDS,
    FORBIDDEN_KEYWORDS,
    GuardResult,
    SqlGuardError,
    validate_read_only,
)


__all__ = [
    # MCP server surface
    "EpiMCPServer",
    "TOOL_SPECS",
    "TOOL_BY_NAME",
    "ToolSpec",
    "CallResult",
    # Stdio transport
    "run_stdio_server",
    # SQL guard
    "validate_read_only",
    "GuardResult",
    "SqlGuardError",
    "ALLOWED_LEADING_KEYWORDS",
    "FORBIDDEN_KEYWORDS",
]
