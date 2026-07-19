"""MCP metrics emitter — sprint 2026-05-06 top3 ROI #1 (Codex optimization).

Lightweight in-memory counter + structured stderr log. No external
dependencies (no prometheus_client / opentelemetry). Production-ready
hook for Prometheus / Datadog / Grafana via stderr log scraping or a
future ``/metrics`` HTTP endpoint exposed by the MCP bridge.

Why not prometheus_client?
--------------------------
Adding prometheus_client (or opentelemetry) makes the simulation server
import-heavier and harder to bundle for Vercel / Cloudflare Workers Edge
runtime. The structured stderr emit can be picked up by:

- Vercel Logs → Datadog forwarder
- Docker stdout/stderr → Loki / Promtail
- systemd journald → Grafana Cloud

If full Prometheus pull-based scraping is required later, a thin
``/metrics`` endpoint can read ``McpMetrics.snapshot()`` and format it
as Prometheus exposition. That is future work (Stage 7+).

Public API
----------
- ``get_metrics()`` — singleton accessor
- ``McpMetrics.record_call(tool, duration_ms, is_error=False)`` — emit
- ``McpMetrics.snapshot()`` — read-only counter + histogram snapshot
  (``calls``, ``errors``, ``p50_ms``, ``p95_ms``, ``p99_ms`` per tool)
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from threading import Lock
from typing import Optional

log = logging.getLogger("simulation.server.metrics")


class McpMetrics:
    """Thread-safe in-memory counter + rolling-window histogram (last 100
    latencies per tool, sufficient for short-window p95/p99 estimate)."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._calls: dict[str, int] = defaultdict(int)
        self._errors: dict[str, int] = defaultdict(int)
        self._duration_ms: dict[str, list[int]] = defaultdict(list)
        self._first_call_ts: float = time.time()

    def record_call(
        self, tool: str, duration_ms: int, is_error: bool = False,
    ) -> None:
        """Record one MCP tool call."""
        with self._lock:
            self._calls[tool] += 1
            if is_error:
                self._errors[tool] += 1
            self._duration_ms[tool].append(int(duration_ms))
            # Rolling window — keep last 100 latencies per tool
            if len(self._duration_ms[tool]) > 100:
                self._duration_ms[tool] = self._duration_ms[tool][-100:]
        # Structured stderr emit — log scraping path
        log.info(
            "mcp_metric tool=%s duration_ms=%d is_error=%s",
            tool, duration_ms, is_error,
        )

    def snapshot(self) -> dict:
        """Read-only counter + histogram snapshot."""
        # Defer numpy import to keep module lightweight on cold-start
        try:
            import numpy as np
        except ImportError:
            np = None  # type: ignore[assignment]

        with self._lock:
            tools = sorted(set(self._calls.keys()))
            out: dict = {
                "tools": tools,
                "calls": dict(self._calls),
                "errors": dict(self._errors),
                "uptime_sec": int(time.time() - self._first_call_ts),
                "p50_ms": {}, "p95_ms": {}, "p99_ms": {},
            }
            for tool in tools:
                latencies = self._duration_ms.get(tool, [])
                if not latencies:
                    continue
                if np is not None:
                    arr = np.asarray(latencies)
                    out["p50_ms"][tool] = float(np.percentile(arr, 50))
                    out["p95_ms"][tool] = float(np.percentile(arr, 95))
                    out["p99_ms"][tool] = float(np.percentile(arr, 99))
                else:
                    # numpy-free fallback
                    sorted_lat = sorted(latencies)
                    n = len(sorted_lat)
                    out["p50_ms"][tool] = float(sorted_lat[n // 2])
                    out["p95_ms"][tool] = float(sorted_lat[min(n - 1, int(n * 0.95))])
                    out["p99_ms"][tool] = float(sorted_lat[min(n - 1, int(n * 0.99))])
        return out


_METRICS: Optional[McpMetrics] = None


def get_metrics() -> McpMetrics:
    """Return the global singleton metrics emitter."""
    global _METRICS
    if _METRICS is None:
        _METRICS = McpMetrics()
    return _METRICS
