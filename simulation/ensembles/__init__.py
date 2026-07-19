"""
simulation.ensembles
====================
3-stage Tournament Ensemble orchestration (§5.2.5, RECOMMENDED_PIPELINE.md).

This package is a THIN ORCHESTRATION LAYER — it does NOT reimplement the
ensemble algorithms. It wraps the existing `simulation.models.ensemble`
classes (InverseRMSE, Stacking, Blending, BMA, NNLS, TemporalWeight,
Diversity, SelectiveBMA, ResidualCorrected, AdaptiveWeight).

3-Stage Tournament:
 Stage A'-1 (intra-category rank):
 Within each category (ts / linear / tree / dl / epi), rank models
 by OOF R² and take top-K.
 Stage A'-2 (Caruana 2004 forward stepwise):
 Across the 5 category winners (+ PAPER_PRIMARY), iteratively
 select models maximizing OOF R² on ensemble predictions.
 Stage A'-3 (meta-ensemble competition):
 Pit multiple ensemble strategies (InverseRMSE, NNLS, Stacking,
 BMA) against each other on OOF predictions; pick the one with
 highest OOF R² + lowest CRPS.
"""
from .tournament import (
    TournamentResult,
    TournamentOrchestrator,
    intra_category_rank,
)
from .caruana import (
    caruana_forward_stepwise,
    CaruanaResult,
)
from .meta_compete import (
    compete_meta_ensembles,
    META_ENSEMBLE_CLASSES,
    MetaCompetitionResult,
)

__all__ = [
    "TournamentResult", "TournamentOrchestrator", "intra_category_rank",
    "caruana_forward_stepwise", "CaruanaResult",
    "compete_meta_ensembles", "META_ENSEMBLE_CLASSES", "MetaCompetitionResult",
]
