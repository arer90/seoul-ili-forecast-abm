"""
OS-aware Memory Management
===========================
Windows: subprocess.Popen 격리
Linux/Mac: ProcessPoolExecutor
공통: psutil 기반 가용 메모리 감시 + 강제 GC
"""
import gc
import os
import sys
import platform
import logging
import numpy as np

log = logging.getLogger(__name__)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class MemoryGuard:
    """메모리 감시 + GC + OOM 방지."""

    def __init__(self, min_free_mb: int = 800, use_float32: bool = True):
        self.min_free_mb = min_free_mb
        self.use_float32 = use_float32
        self.is_windows = platform.system() == "Windows"
        self._gc_count = 0

    def check_and_gc(self, context: str = "") -> bool:
        """가용 메모리 확인, 부족하면 GC 실행. True=안전, False=위험."""
        if not HAS_PSUTIL:
            gc.collect()
            return True
        mem = psutil.virtual_memory()
        free_mb = mem.available / (1024 * 1024)
        if free_mb < self.min_free_mb:
            log.warning(f"  [메모리 경고] {context}: {free_mb:.0f}MB 가용 -- 강제 GC 실행")
            gc.collect()
            self._gc_count += 1
            mem2 = psutil.virtual_memory()
            free_mb2 = mem2.available / (1024 * 1024)
            if free_mb2 < self.min_free_mb:
                log.error(f"  [메모리 위험] GC 후에도 {free_mb2:.0f}MB < {self.min_free_mb}MB")
                return False
            log.info(f"  [메모리 회복] GC 후 {free_mb2:.0f}MB 가용")
        return True

    def convert_float32(self, X):
        """float64 → float32 변환 (메모리 50% 절감)."""
        if self.use_float32 and hasattr(X, 'dtype') and X.dtype == np.float64:
            return X.astype(np.float32)
        return X

    def get_free_mb(self) -> float:
        if not HAS_PSUTIL:
            return 9999.0
        return psutil.virtual_memory().available / (1024 * 1024)

    # (2026-05-29 G-236 후속 제거) should_use_subprocess: dead 중복 게이트 —
    #   live path 미호출(MemoryGuard 는 check_and_gc 만 사용). 격리 판단의 단일 출처는
    #   runner._should_use_subprocess + runner._SUBPROCESS_CATEGORIES. 재도입 시 그쪽 사용.

    @property
    def gc_count(self) -> int:
        return self._gc_count
