"""DNN family forecasters (Sprint β Item 6 minimal — re-export from dl_models.py).

3 classes: DNNForecaster (vanilla MLP), OptunaDNNForecaster (50-trial HPO),
TinyMLPForecaster (S2-3 sanity-floor baseline, 321→32→16→1).

Class definitions live in `simulation/models/dl_models.py`; this module
re-exports for per-architecture caller paths. ChampionArtifact pickle ABI is
preserved because the canonical module path stays `simulation.models.dl_models`.
"""
from simulation.models.dl_models import (
    DNNForecaster,
    OptunaDNNForecaster,
    TinyMLPForecaster,
)

__all__ = ["DNNForecaster", "OptunaDNNForecaster", "TinyMLPForecaster"]
