"""TCN family forecasters (Sprint β Item 6 minimal — re-export from dl_models.py).

2 classes: TCNForecaster (vanilla temporal CNN), OptunaTCNForecaster
(50-trial HPO with MedianPruner).

Class definitions live in `simulation/models/dl_models.py`; this module
re-exports for per-architecture caller paths. ChampionArtifact pickle ABI
preserved.
"""
from simulation.models.dl_models import (
    TCNForecaster,
    OptunaTCNForecaster,
)

__all__ = ["TCNForecaster", "OptunaTCNForecaster"]
