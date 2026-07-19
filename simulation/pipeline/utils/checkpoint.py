"""
Phase Checkpoint Manager
=========================
각 Phase 완료 후 중간 결과를 JSON으로 저장.
--resume-from-phase N 으로 특정 Phase부터 재개 가능.
"""
import json
import time
import logging
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


def _make_serializable(obj):
    """numpy/pandas 타입을 JSON 직렬화 가능하도록 변환."""
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    return obj


class CheckpointManager:
    """R/P phase별 체크포인트 저장/로딩.

    phase 식별자는 R/P 라벨(예: "R9") — 옛 매직 숫자(13) 폐기. 라벨↔순서는
    :mod:`simulation.pipeline.phases` 레지스트리(SSOT)가 결정. 파일명 = ``checkpoint_<label>.json``.
    """

    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.checkpoint_dir = self.save_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._timings: Dict[str, float] = {}

    def save(self, phase: str, data: dict, label: str = ""):
        """Phase 결과를 JSON으로 저장. phase = R/P 라벨(예: 'R9')."""
        fname = f"checkpoint_{phase}.json"
        path = self.checkpoint_dir / fname
        payload = {
            "phase": phase,
            "label": label,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data": _make_serializable(data),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        log.info(f"  ✓ {phase} 체크포인트 저장: {path}")

    def load(self, phase: str) -> Optional[dict]:
        """Phase 체크포인트 로딩. 없으면 None."""
        fname = f"checkpoint_{phase}.json"
        path = self.checkpoint_dir / fname
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        log.info(f"  ✓ {phase} 체크포인트 로딩: {path}")
        return payload.get("data")

    def phase_exists(self, phase: str) -> bool:
        fname = f"checkpoint_{phase}.json"
        return (self.checkpoint_dir / fname).exists()

    def start_timer(self, phase: str):
        self._timings[phase] = time.time()

    def stop_timer(self, phase: str) -> float:
        elapsed = time.time() - self._timings.get(phase, time.time())
        return elapsed

    def get_last_completed_phase(self) -> Optional[str]:
        """가장 진행된(파이프라인 순서상 최대) 완료 phase 라벨. 없으면 None."""
        from simulation.pipeline import phases as _ph
        done = []
        for f in self.checkpoint_dir.glob("checkpoint_*.json"):
            lbl = f.stem.replace("checkpoint_", "")
            if _ph.is_known(lbl):
                done.append(lbl)
        return max(done, key=_ph.order) if done else None
