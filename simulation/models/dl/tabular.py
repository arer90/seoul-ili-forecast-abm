"""Tabular DNN family forecasters (Sprint β Item 6 minimal — re-export from dl_models.py).

2 classes: TabularDNNForecaster (negative-control per registry.py),
TabularDNNLiteForecaster (parsimony-first DL Tier A).

Class definitions live in `simulation/models/dl_models.py`; this module
re-exports for per-architecture caller paths. ChampionArtifact pickle ABI
preserved.
"""
from simulation.models.dl_models import (
    TabularDNNForecaster,
    TabularDNNLiteForecaster,
)

__all__ = ["TabularDNNForecaster", "TabularDNNLiteForecaster"]
