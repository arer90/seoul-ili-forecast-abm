"""Phase 별 자원/시간 tracking — 2026-05-28 사용자 명시 design.

사용자 명시 (2026-05-28): "사용하는 자원과 시간기록을 다 result 에 하는거지?".
모든 phase result dict 에 elapsed_sec / peak_rss_mb / mean_cpu_pct / peak_gpu_memory_mb
첨부.

사용:
    from simulation.utils.resource_tracker import ResourceTracker

    with ResourceTracker(phase_id="phase12") as rt:
        # phase work...
        result = run_per_model_optimize(...)
    result.update(rt.to_dict())   # → result["resource_tracker"] = {...}

또는 decorator (next sprint):
    @track_resources("phase12")
    def run_per_model_optimize(...): ...

측정:
    - elapsed_sec: float — wall-clock
    - peak_rss_mb: int — peak Resident Set Size (process memory)
    - mean_cpu_pct: float — CPU 사용률 (sampled 평균, %)
    - peak_gpu_memory_mb: int — PyTorch GPU/MPS peak allocated (MB)
    - platform: "cuda" | "mps" | "cpu"
    - start_at / end_at: ISO timestamp

Performance overhead: <1% (sampler thread 0.5s interval).
Side effects: 없음 (read-only system metrics).

OS 호환: macOS (MPS) / Linux (CUDA) / Windows (CPU).
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

log = logging.getLogger(__name__)

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
    log.warning("psutil not available — ResourceTracker 의 CPU/메모리 측정 제한")

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _get_platform() -> str:
    if not _HAS_TORCH:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ResourceTracker:
    """Phase 별 자원/시간 추적 ContextManager.

    Args:
        phase_id: phase 이름 (예: "phase12", "phase1.5b", "phase_evaluator").
        sample_interval_sec: CPU/RSS sampling 주기 (default 0.5s).

    Attributes (after __exit__):
        elapsed_sec: float
        start_at / end_at: ISO timestamp
        peak_rss_mb: int (psutil 없으면 0)
        mean_cpu_pct: float (psutil 없으면 0)
        peak_gpu_memory_mb: int (torch / CUDA / MPS)
        platform: "cuda" | "mps" | "cpu"
    """

    def __init__(self, phase_id: str, sample_interval_sec: float = 0.5):
        self.phase_id = phase_id
        self.sample_interval = max(0.1, sample_interval_sec)
        self._proc = psutil.Process() if _HAS_PSUTIL else None
        self._sampler_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._cpu_samples: list[float] = []
        self._rss_samples: list[int] = []

        # Filled in __exit__
        self.elapsed_sec: float = 0.0
        self.start_at: float = 0.0
        self.end_at: float = 0.0
        self.start_iso: str = ""
        self.end_iso: str = ""
        self.peak_rss_mb: int = 0
        self.mean_cpu_pct: float = 0.0
        self.peak_gpu_memory_mb: int = 0
        self.platform: str = _get_platform()

    def _sampler_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._proc is not None:
                try:
                    self._cpu_samples.append(self._proc.cpu_percent(interval=0))
                    self._rss_samples.append(self._proc.memory_info().rss)
                except Exception:
                    pass
            self._stop_event.wait(self.sample_interval)

    def __enter__(self) -> "ResourceTracker":
        self.start_at = time.time()
        self.start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
        if self._proc is not None:
            try:
                self._proc.cpu_percent(interval=0)  # baseline (first call returns 0.0)
            except Exception:
                pass
        # Reset GPU peak counters
        if _HAS_TORCH and torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        # Start sampler thread
        self._sampler_thread = threading.Thread(target=self._sampler_loop, daemon=True)
        self._sampler_thread.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._stop_event.set()
        if self._sampler_thread is not None:
            self._sampler_thread.join(timeout=2.0)
        self.end_at = time.time()
        self.end_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.elapsed_sec = self.end_at - self.start_at

        if self._cpu_samples:
            self.mean_cpu_pct = float(sum(self._cpu_samples) / len(self._cpu_samples))
        if self._rss_samples:
            self.peak_rss_mb = max(self._rss_samples) // (1024 * 1024)

        # GPU peak
        if _HAS_TORCH and torch.cuda.is_available():
            try:
                self.peak_gpu_memory_mb = torch.cuda.max_memory_allocated() // (1024 * 1024)
            except Exception:
                pass
        # MPS: PyTorch 2.x 는 peak memory API 없음 (current_allocated_memory 만)
        elif _HAS_TORCH and self.platform == "mps":
            try:
                self.peak_gpu_memory_mb = torch.mps.current_allocated_memory() // (1024 * 1024)
            except Exception:
                pass

    def to_dict(self) -> dict:
        """Result dict 형식 — phase result 에 merge 가능."""
        return {
            "resource_tracker": {
                "phase_id": self.phase_id,
                "elapsed_sec": round(self.elapsed_sec, 2),
                "start_at": self.start_iso,
                "end_at": self.end_iso,
                "peak_rss_mb": self.peak_rss_mb,
                "mean_cpu_pct": round(self.mean_cpu_pct, 1),
                "peak_gpu_memory_mb": self.peak_gpu_memory_mb,
                "platform": self.platform,
                "_n_samples": len(self._cpu_samples),
            }
        }


def track_resources(phase_id: str):
    """Decorator wrapping function with ResourceTracker.

    Use:
        @track_resources("phase12")
        def run_per_model_optimize(...) -> dict:
            ...

    Returned result dict 에 자동 'resource_tracker' key 추가.
    원본 함수가 dict 가 아닌 경우 dict 로 wrap.
    """
    def decorator(fn):
        def wrapper(*args, **kwargs):
            with ResourceTracker(phase_id) as rt:
                result = fn(*args, **kwargs)
            if isinstance(result, dict):
                result.update(rt.to_dict())
            else:
                # Wrap non-dict result
                result = {"return": result, **rt.to_dict()}
            return result
        wrapper.__wrapped__ = fn
        wrapper.__name__ = fn.__name__
        return wrapper
    return decorator


__all__ = ["ResourceTracker", "track_resources"]
