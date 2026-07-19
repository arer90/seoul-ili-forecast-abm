"""
simulation/models/modern_ts/_base.py
====================================
Shared utilities and imports for modern time-series models.

최신 시계열 모델 (2024-2026 SOTA):
 - N-BEATS (Neural Basis Expansion) -- Level 11
 - N-HiTS (Hierarchical Interpolation) -- Level 12
 - PatchTST (Patch Time Series Transformer) -- Level 15 [실험적]
 - iTransformer (Inverted Transformer) -- Level 16 [실험적]
 - Mamba (Selective State Space) -- Level 17 [실험적]
 - TimesNet (2D Temporal Variation) -- Level 18 [실험적]
 - TiDE (Time series Dense Encoder) -- Level 19 [실험적]

341주 소표본 제약:
 - N-BEATS/N-HiTS: 200+ 샘플로 충분, hidden=64로 경량화
 - PatchTST/iTransformer/Mamba/TimesNet/TiDE: 1000+ 권장이나 실험적 포함
 → 강한 정규화(dropout=0.3, weight_decay=5e-4)로 과적합 방지

변경 이력:
 - (2026-03-30): 초기 구현. 6개 최신 모델 추가.
 - (2026-04-11): Refactored into modular package structure.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

from simulation.models.base import BaseForecaster, ModelMeta, REGISTRY

log = logging.getLogger(__name__)

__all__ = [
    "BaseForecaster",
    "ModelMeta",
    "REGISTRY",
    "log",
    "logging",
    "math",
    "Optional",
    "np",
]
