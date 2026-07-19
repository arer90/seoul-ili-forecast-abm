"""Lightweight cache layer: Redis (if available) → in-memory dict fallback.

Usage:
    from simulation.cache import get_cache
    cache = get_cache()
    cache.set("forecast:2026-W17", json_payload, ttl=3600)
    hit = cache.get("forecast:2026-W17")

Configuration:
    REDIS_URL env var (default: redis://localhost:6379/0)
    Falls back to process-local dict if Redis unavailable.

Use cases in this project:
    * MCP session store (ARIA UI user sessions)
    * forecast TTL cache (recent N-week predictions per gu, 1h TTL)
    * Rt nowcast latest-value cache
    * rate-limiting counters (external API quotas)
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

log = logging.getLogger(__name__)

_DEFAULT_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")


class _InMemoryCache:
    """Process-local dict cache with TTL — used when Redis unavailable."""
    def __init__(self):
        self._store: dict[str, tuple[Any, float]] = {}

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires = entry
        if expires > 0 and time.time() > expires:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl: int = 0) -> None:
        expires = time.time() + ttl if ttl > 0 else 0
        self._store[key] = (value, expires)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def backend(self) -> str:
        return "in-memory"

    def ping(self) -> bool:
        return True


class _RedisCache:
    """Thin redis-py wrapper. JSON-encoded values, seconds TTL."""
    def __init__(self, client):
        self._r = client

    def get(self, key: str) -> Optional[Any]:
        try:
            raw = self._r.get(key)
            return json.loads(raw) if raw else None
        except Exception as e:
            log.debug(f"redis.get({key}) failed: {e}")
            return None

    def set(self, key: str, value: Any, ttl: int = 0) -> None:
        try:
            payload = json.dumps(value, default=str)
            if ttl > 0:
                self._r.setex(key, ttl, payload)
            else:
                self._r.set(key, payload)
        except Exception as e:
            log.debug(f"redis.set({key}) failed: {e}")

    def delete(self, key: str) -> None:
        try:
            self._r.delete(key)
        except Exception as e:
            log.debug(f"redis.delete({key}) failed: {e}")

    def backend(self) -> str:
        return "redis"

    def ping(self) -> bool:
        try:
            return bool(self._r.ping())
        except Exception:
            return False


_cache_singleton: Optional[Any] = None


def get_cache(url: Optional[str] = None):
    """Return the cache singleton. Connects to Redis on first call; falls
    back to in-memory dict if Redis server is not reachable.

    Lazy/idempotent: safe to call many times.
    """
    global _cache_singleton
    if _cache_singleton is not None:
        return _cache_singleton

    url = url or _DEFAULT_URL
    try:
        import redis
        client = redis.from_url(url, socket_connect_timeout=1, socket_timeout=2)
        # Ping to force connection check
        if client.ping():
            _cache_singleton = _RedisCache(client)
            log.info(f"cache: redis @ {url}")
            return _cache_singleton
    except ImportError:
        log.info("cache: redis-py not installed, using in-memory fallback")
    except Exception as e:
        log.info(f"cache: redis unreachable ({e}), using in-memory fallback")

    _cache_singleton = _InMemoryCache()
    return _cache_singleton


def cache_info() -> dict:
    """Diagnostic."""
    c = get_cache()
    return {
        "backend": c.backend(),
        "ping": c.ping(),
        "redis_url": _DEFAULT_URL,
    }
