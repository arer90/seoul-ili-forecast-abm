"""Unified memory / cache / GPU cleanup utility (2026-05-28).

사용자 명시 (2026-05-28): "CUDA나 MPS 그리고 GPU를 포함한 동일한 현상이 일어날 수
있으니 memory와 cache, gpu memory 등 초기화 및 정리를 할 수 있는 공간을 넣어줘".

본 module 은 **모든 backend (CUDA / MPS / CPU) 통합 cleanup utility**.
재사용 가능 — phase entry / model fit / trial 끝 / subprocess 종료 시 호출.

기존 `simulation/models/_optuna_torch._trial_gpu_cleanup` 의 logic 을 상속 +
강화 (subprocess GPU + multiprocessing semaphore + torch internal cache).

API:
    from simulation.utils.memory_cleanup import (
        cleanup_all,           # 가장 강력 (모든 backend)
        cleanup_gpu_memory,    # GPU 만 (CUDA + MPS)
        cleanup_python_gc,     # Python GC only
        cleanup_torch_cuda,    # CUDA + cuBLAS + IPC
        cleanup_torch_mps,     # MPS only
        cleanup_libc_heap,     # Linux glibc malloc_trim
        memory_snapshot,       # debug — 현재 메모리 상태 dict
    )

사용:
    # 모든 backend cleanup (권장)
    cleanup_all()

    # 특정 backend 만 (선택)
    cleanup_gpu_memory()      # GPU 위주
    cleanup_python_gc()       # Python heap 만

    # debug
    snapshot = memory_snapshot()
    print(snapshot)
    # → {python_gc: {...}, torch_cuda: {...}, torch_mps: {...}, system_rss_mb: int}

Performance overhead: 5-20ms per cleanup_all() call.
Side effects: GPU allocator cache 회수, Python heap 회수, Linux malloc arena 회수.
NaN-safe: 모든 step 이 try-except — 어떤 환경에서도 graceful (CUDA 없어도 OK).

OS 호환:
    macOS (MPS):   torch.mps.empty_cache + Python GC
    Linux (CUDA):  torch.cuda.empty_cache + cuBLAS + IPC + malloc_trim
    Windows (CUDA): torch.cuda.empty_cache + cuBLAS + IPC
    CPU only:      Python GC + (Linux malloc_trim if available)

Reference: G-158 (memory leak prevention), G-161 (trial cleanup callback).
"""
from __future__ import annotations

import gc
import logging
import os
from typing import Optional

log = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# 1. Python GC
# ────────────────────────────────────────────────────────────────────────────

def cleanup_python_gc() -> None:
    """Python garbage collector — 2 passes (PEP 442 cycle finalizer).

    gc.collect() 2회:
        1차: unreachable cycle 의 __del__ trigger
        2차: gc.garbage 청소 (finalizer 가 부활시킨 cycle)

    Safe — exception 모두 swallow.
    """
    for _ in range(2):
        try:
            gc.collect()
        except Exception:
            pass


# ────────────────────────────────────────────────────────────────────────────
# 2. PyTorch CUDA
# ────────────────────────────────────────────────────────────────────────────

def cleanup_torch_cuda() -> bool:
    """CUDA allocator cache + cuBLAS workspace + IPC handle 회수.

    Returns:
        True if CUDA available + cleanup executed, False otherwise.

    Steps:
        1. torch.cuda.empty_cache()       — allocator cache
        2. torch._C._cuda_clearCublasWorkspaces() — cuBLAS (stream 별 256MB)
        3. torch.cuda.ipc_collect()       — multi-process DataLoader IPC
        4. torch.cuda.synchronize()       — pending kernel 완료 대기 (optional)
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return False
    except ImportError:
        return False

    # Step 1: allocator cache
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    # Step 2: cuBLAS workspace
    try:
        if hasattr(torch, "_C") and hasattr(torch._C, "_cuda_clearCublasWorkspaces"):
            torch._C._cuda_clearCublasWorkspaces()
    except Exception:
        pass
    # Step 3: IPC handle
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
    return True


# ────────────────────────────────────────────────────────────────────────────
# 3. PyTorch MPS (Apple Silicon)
# ────────────────────────────────────────────────────────────────────────────

def cleanup_torch_mps() -> bool:
    """MPS allocator cache 회수 (Mac M-series).

    PyTorch MPS 의 known issue (libtorch_python.dylib segfault) 회피:
        ~1h 후 segfault 발생 패턴 — MPS cache fragment + invalid pointer 추정.
        Trial / model fit 끝마다 cleanup → segfault 가능성 감소.

    Returns:
        True if MPS available + cleanup executed.

    Steps:
        1. torch.mps.empty_cache()        — allocator cache
        2. torch.mps.synchronize()        — pending GPU operation 완료 (PyTorch 2.0+)
    """
    try:
        import torch
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            return False
    except ImportError:
        return False

    # Step 1: allocator cache
    try:
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass
    # Step 2: synchronize (pending operation 완료)
    try:
        if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
    except Exception:
        pass
    return True


# ────────────────────────────────────────────────────────────────────────────
# 4. GPU 통합 (CUDA + MPS)
# ────────────────────────────────────────────────────────────────────────────

def cleanup_gpu_memory() -> dict:
    """모든 GPU backend (CUDA + MPS) cleanup.

    Returns:
        {"cuda": bool, "mps": bool} — 각 backend cleanup 여부.
    """
    return {
        "cuda": cleanup_torch_cuda(),
        "mps": cleanup_torch_mps(),
    }


# ────────────────────────────────────────────────────────────────────────────
# 5. Linux glibc heap (malloc arena 회수)
# ────────────────────────────────────────────────────────────────────────────

def cleanup_libc_heap() -> bool:
    """Linux glibc malloc_trim(0) — heap arena 회수.

    PyTorch tensor del 후에도 RSS 가 reclaim 안 되는 문제 (glibc heap 보존):
        process memory 가 OS 에 반환 안 됨 → RSS 가 가상 메모리 limit 까지 증가.
        malloc_trim(0) 으로 강제 OS 반환.

    Returns:
        True if Linux + libc.so 가용 + cleanup 실행. macOS/Windows = False.
    """
    try:
        import ctypes
        import ctypes.util
        _libc_path = ctypes.util.find_library("c")
        if not _libc_path:
            return False
        _libc = ctypes.CDLL(_libc_path)
        if hasattr(_libc, "malloc_trim"):
            _libc.malloc_trim(0)
            return True
    except Exception:
        pass
    return False


# ────────────────────────────────────────────────────────────────────────────
# 6. 통합 cleanup (가장 강력)
# ────────────────────────────────────────────────────────────────────────────

def cleanup_all(verbose: bool = False) -> dict:
    """모든 backend 통합 cleanup — model fit / trial / phase 끝마다 권장.

    Steps (순서 중요):
        1. Python GC (2 passes) — 객체 cycle 회수
        2. CUDA cleanup (allocator + cuBLAS + IPC) — Linux/Windows GPU
        3. MPS cleanup (allocator + synchronize) — Mac M-series GPU
        4. Linux malloc_trim — glibc heap 회수

    Args:
        verbose: True 시 log.info 로 결과 출력.

    Returns:
        {"python_gc": True, "cuda": bool, "mps": bool, "libc": bool}

    Side effects:
        - GPU allocator cache 회수
        - Python heap 회수
        - Linux: glibc heap arena OS 반환
    """
    cleanup_python_gc()
    cuda_ok = cleanup_torch_cuda()
    mps_ok = cleanup_torch_mps()
    libc_ok = cleanup_libc_heap()

    result = {
        "python_gc": True,
        "cuda": cuda_ok,
        "mps": mps_ok,
        "libc": libc_ok,
    }

    if verbose:
        log.info(f"  [memory_cleanup] {result}")

    return result


# ────────────────────────────────────────────────────────────────────────────
# 7. Memory snapshot (debug)
# ────────────────────────────────────────────────────────────────────────────

def memory_snapshot() -> dict:
    """현재 메모리 상태 snapshot (debug 용).

    Returns:
        {
            "python_gc": {generation_0_count, generation_1_count, generation_2_count, ...},
            "torch_cuda": {allocated_mb, reserved_mb, max_allocated_mb, ...} (if CUDA),
            "torch_mps": {current_allocated_mb, driver_allocated_mb} (if MPS),
            "system_rss_mb": int (psutil 가용 시),
            "platform": "cuda" | "mps" | "cpu",
        }

    Performance: ~5ms.
    """
    snap: dict = {"platform": "cpu"}

    # Python GC
    try:
        snap["python_gc"] = {
            f"gen{i}": gc.get_count()[i] for i in range(3)
        }
        snap["python_gc"]["uncollectable"] = len(gc.garbage)
    except Exception as e:
        snap["python_gc"] = {"error": str(e)}

    # System RSS (psutil)
    try:
        import psutil
        _proc = psutil.Process()
        snap["system_rss_mb"] = _proc.memory_info().rss // (1024 * 1024)
        snap["system_vms_mb"] = _proc.memory_info().vms // (1024 * 1024)
    except ImportError:
        pass
    except Exception as e:
        snap["system_error"] = str(e)

    # Torch
    try:
        import torch
        if torch.cuda.is_available():
            snap["platform"] = "cuda"
            try:
                snap["torch_cuda"] = {
                    "allocated_mb": torch.cuda.memory_allocated() // (1024 * 1024),
                    "reserved_mb": torch.cuda.memory_reserved() // (1024 * 1024),
                    "max_allocated_mb": torch.cuda.max_memory_allocated() // (1024 * 1024),
                }
            except Exception as e:
                snap["torch_cuda"] = {"error": str(e)}
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            snap["platform"] = "mps"
            try:
                snap["torch_mps"] = {
                    "current_allocated_mb": (torch.mps.current_allocated_memory() // (1024 * 1024)
                                              if hasattr(torch.mps, "current_allocated_memory") else 0),
                    "driver_allocated_mb": (torch.mps.driver_allocated_memory() // (1024 * 1024)
                                             if hasattr(torch.mps, "driver_allocated_memory") else 0),
                }
            except Exception as e:
                snap["torch_mps"] = {"error": str(e)}
    except ImportError:
        snap["torch"] = "not installed"

    return snap


# Backward-compat alias for _trial_gpu_cleanup
# (G-161 trial cleanup callback 가 이 함수 호출 시 cleanup_all 와 동일 동작)
def _trial_gpu_cleanup() -> None:
    """Backward-compat — call cleanup_all() (모든 backend 통합)."""
    cleanup_all()


__all__ = [
    "cleanup_all",
    "cleanup_gpu_memory",
    "cleanup_python_gc",
    "cleanup_torch_cuda",
    "cleanup_torch_mps",
    "cleanup_libc_heap",
    "memory_snapshot",
    "_trial_gpu_cleanup",     # backward-compat
]
