"""
simulation/models/modern_ts/
============================
Modern time-series forecasting models (2024-2026 SOTA).

Package structure:
 - nbeats.py: N-BEATS (Level 11)
 - nhits.py: N-HiTS (Level 12)
 - patchtst.py: PatchTST (Level 15)
 - itransformer.py: iTransformer (Level 16)
 - mamba.py: Mamba/S4 (Level 17)
 - timesnet.py: TimesNet (Level 18)
 - tide.py: TiDE (Level 29)
 - conformal.py: Conformal Prediction wrapper

모든 Forecaster 클래스는 backward compatibility를 위해 여기서 re-export됨.

변경 이력:
 - (2026-03-30): Initial implementation (6 modern models + TiDE)
 - (2026-04-11): Refactored into modular package structure
"""

from __future__ import annotations

# Re-export all Forecaster classes for backward compatibility
# from simulation.models.modern_ts import NBEATSForecaster
from simulation.models.modern_ts.nbeats import NBEATSForecaster
from simulation.models.modern_ts.nhits import NHiTSForecaster
from simulation.models.modern_ts.patchtst import PatchTSTForecaster
from simulation.models.modern_ts.itransformer import iTransformerForecaster
from simulation.models.modern_ts.mamba import MambaForecaster
from simulation.models.modern_ts.timesnet import TimesNetForecaster
from simulation.models.modern_ts.tide import TiDEForecaster
from simulation.models.modern_ts.conformal import ConformalPredictionWrapper

# : pytorch_forecasting 1.7.0 reference implementations (Tier 1+2)
# 기존 custom 과 병행하여 재현성 A/B 벤치마크 제공.
try:
    from simulation.models.modern_ts.pf_models import (
        PfTFTForecaster,
        PfNBeatsForecaster,
        PfNHiTSForecaster,
        PfTiDEForecaster,
        PfRNNForecaster,
        PfDeepARForecaster,
    )
    _PF_MODELS_AVAILABLE = True
except Exception as _pf_import_err:
    import logging as _logging
    _logging.getLogger(__name__).warning(
        f"[modern_ts] pf_models import 실패 (pytorch_forecasting 미설치?): {_pf_import_err}"
    )
    _PF_MODELS_AVAILABLE = False

# Also expose base utilities for model developers
from simulation.models.base import BaseForecaster, ModelMeta, REGISTRY

__all__ = [
    # Custom Forecaster classes
    "NBEATSForecaster",
    "NHiTSForecaster",
    "PatchTSTForecaster",
    "iTransformerForecaster",
    "MambaForecaster",
    "TimesNetForecaster",
    "TiDEForecaster",
    # pf 1.7.0 reference implementations 
    "PfTFTForecaster",
    "PfNBeatsForecaster",
    "PfNHiTSForecaster",
    "PfTiDEForecaster",
    "PfRNNForecaster",
    "PfDeepARForecaster",
    # Utilities
    "ConformalPredictionWrapper",
    "BaseForecaster",
    "ModelMeta",
    "REGISTRY",
]

# Register all forecasters with the registry
# 2026-05-12 (사용자 명시): -pf 정책 적용 — 중복 base 등록 차단.
# pf_models.py 의 wrapper 가 TFT/N-BEATS/N-HiTS/TiDE/RNN/DeepAR 이름 점유.
# Custom torch impl (NBEATSForecaster, NHiTSForecaster, TiDEForecaster) 은
# class 정의 보존 + REGISTRY 등록 차단.
try:
    # REGISTRY.register(NBEATSForecaster)  # 2026-05-12: PfNBeats 가 "N-BEATS" 등록
    # REGISTRY.register(NHiTSForecaster)   # 2026-05-12: PfNHiTS 가 "N-HiTS"
    REGISTRY.register(PatchTSTForecaster)   # base 유지 (pf 없음)
    REGISTRY.register(iTransformerForecaster)  # base 유지
    REGISTRY.register(MambaForecaster)       # base 유지
    REGISTRY.register(TimesNetForecaster)    # base 유지
    # REGISTRY.register(TiDEForecaster)    # 2026-05-12: PfTiDE 가 "TiDE"
    if _PF_MODELS_AVAILABLE:
        REGISTRY.register(PfTFTForecaster)      # name="TFT"
        REGISTRY.register(PfNBeatsForecaster)   # name="N-BEATS"
        REGISTRY.register(PfNHiTSForecaster)    # name="N-HiTS"
        REGISTRY.register(PfTiDEForecaster)     # name="TiDE"
        REGISTRY.register(PfRNNForecaster)      # name="RNN"
        REGISTRY.register(PfDeepARForecaster)   # name="DeepAR"
except Exception as e:
    import logging
    log = logging.getLogger(__name__)
    log.warning(f"[modern_ts] REGISTRY registration failed: {e}")
