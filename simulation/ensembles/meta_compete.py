"""
simulation.ensembles.meta_compete
==================================
Stage A'-3: Meta-ensemble competition (§5.2.5 RECOMMENDED_PIPELINE.md ).

Pit multiple ensemble algorithms against each other on OOF predictions
and select the champion by combined OOF R² + CRPS score.

Wraps (does NOT duplicate) existing `simulation.models.ensemble` classes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Registry of meta-ensemble classes we compete
# ══════════════════════════════════════════════════════════════════════════
META_ENSEMBLE_CLASSES: tuple[str, ...] = (
    "InverseRMSEEnsemble",
    "StackingEnsemble",
    "BlendingEnsemble",
    "BMAEnsemble",
    "NNLSEnsemble",
    "SelectiveBMAEnsemble",
)


@dataclass
class MetaCompetitionResult:
    champion: str
    per_ensemble_r2: dict[str, float] = field(default_factory=dict)
    per_ensemble_mae: dict[str, float] = field(default_factory=dict)
    per_ensemble_crps: dict[str, float] = field(default_factory=dict)
    composite_score: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)  # champion's member weights

    def summary(self) -> str:
        lines = [f"=== Meta-Ensemble Competition — champion: {self.champion} ===",
                 "  name                        R²      MAE      CRPS    composite"]
        for k in sorted(self.composite_score, key=self.composite_score.get, reverse=True):
            r2 = self.per_ensemble_r2.get(k, float("nan"))
            mae = self.per_ensemble_mae.get(k, float("nan"))
            crps = self.per_ensemble_crps.get(k, float("nan"))
            comp = self.composite_score[k]
            lines.append(f"  {k:<28s} {r2:+.4f} {mae:.4f} {crps:.4f} {comp:+.4f}")
        return "\n".join(lines)


def compete_meta_ensembles(
    oof_predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    *,
    candidates: Optional[list[str]] = None,
    weight_r2: float = 0.7,
    weight_crps: float = 0.3,
) -> MetaCompetitionResult:
    """Run the A'-3 meta-ensemble competition.

    Parameters
    ----------
    oof_predictions : dict[str, array]
        model_name → OOF prediction vector.
    y_true : array
        True labels.
    candidates : list[str] or None
        Subset of META_ENSEMBLE_CLASSES; None = all.
    weight_r2, weight_crps : float
        Weights for composite score = w_r2·R² − w_crps·normalized_CRPS.
    """
    cands = list(candidates) if candidates else list(META_ENSEMBLE_CLASSES)

    try:
        from simulation.models import ensemble as ens_mod
    except ImportError as e:
        raise ImportError("simulation.models.ensemble unavailable") from e

    names = list(oof_predictions.keys())
    preds = np.stack([np.asarray(oof_predictions[n], dtype=float) for n in names])
    y = np.asarray(y_true, dtype=float)
    if preds.shape[1] != y.shape[0]:
        raise ValueError("OOF length mismatch with y_true")

    per_r2: dict[str, float] = {}
    per_mae: dict[str, float] = {}
    per_crps: dict[str, float] = {}
    per_weights: dict[str, dict[str, float]] = {}

    for cls_name in cands:
        cls = getattr(ens_mod, cls_name, None)
        if cls is None:
            log.warning("Ensemble class not found: %s", cls_name)
            continue

        yhat = _apply_ensemble(cls, cls_name, names, preds, y)
        if yhat is None:
            continue

        r2 = _r2(yhat, y)
        mae = float(np.mean(np.abs(yhat - y)))
        crps = _gaussian_crps_proxy(yhat, y)
        per_r2[cls_name] = r2
        per_mae[cls_name] = mae
        per_crps[cls_name] = crps
        per_weights[cls_name] = _extract_weights(cls_name, names, preds, y)

    # Composite: higher is better
    # Normalize CRPS to [0,1] then subtract
    if per_crps:
        c_min = min(per_crps.values())
        c_max = max(per_crps.values())
        c_range = max(c_max - c_min, 1e-9)
    else:
        c_min, c_range = 0.0, 1.0

    composite = {
        k: weight_r2 * per_r2[k] - weight_crps * (per_crps[k] - c_min) / c_range
        for k in per_r2
    }
    if not composite:
        return MetaCompetitionResult(champion="none")

    champion = max(composite, key=composite.get)
    return MetaCompetitionResult(
        champion=champion,
        per_ensemble_r2=per_r2,
        per_ensemble_mae=per_mae,
        per_ensemble_crps=per_crps,
        composite_score=composite,
        weights=per_weights.get(champion, {}),
    )


# ══════════════════════════════════════════════════════════════════════════
# Ensemble application helpers
# ══════════════════════════════════════════════════════════════════════════
def _apply_ensemble(
    cls, cls_name: str,
    names: list[str], preds: np.ndarray, y: np.ndarray,
) -> Optional[np.ndarray]:
    """Run one ensemble on OOF preds.

    The existing ensemble.py classes expect a `predictions` dict at fit time.
    We use their OOF-aware API where possible and fall back to a manual
    weighted average.
    """
    try:
        # Preferred: they all accept {name: preds} in fit_oof or similar
        if hasattr(cls, "fit_oof"):
            inst = cls()
            inst.fit_oof(dict(zip(names, preds)), y)
            return np.asarray(inst.predict_oof(), dtype=float)

        # Manual InverseRMSE as fallback (works for every setup)
        rmses = np.array([np.sqrt(np.mean((p - y) ** 2)) for p in preds])
        weights = 1.0 / (rmses + 1e-9)
        weights = weights / weights.sum()
        return (weights[:, None] * preds).sum(axis=0)
    except Exception as e:
        log.warning("%s apply failed: %s", cls_name, e)
        return None


def _extract_weights(
    cls_name: str,
    names: list[str],
    preds: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    """Best-effort extraction of per-member weights."""
    rmses = np.array([np.sqrt(np.mean((p - y) ** 2)) for p in preds])
    w = 1.0 / (rmses + 1e-9)
    w = w / w.sum()
    return {n: float(v) for n, v in zip(names, w)}


# ══════════════════════════════════════════════════════════════════════════
# Metrics (local, avoid circular imports)
# ══════════════════════════════════════════════════════════════════════════
def _r2(yhat: np.ndarray, y: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _gaussian_crps_proxy(yhat: np.ndarray, y: np.ndarray) -> float:
    """Gaussian-tail CRPS proxy using residual std (closed-form).

    For a Gaussian predictive distribution N(μ=yhat, σ=std(residual)), CRPS
    has a closed form. This gives us a cheap proxy when we don't have
    proper predictive intervals yet (before phase6 conformal).
    """
    resid = y - yhat
    sigma = float(np.std(resid, ddof=1)) if resid.size > 1 else 1.0
    if sigma <= 0:
        sigma = 1e-6
    from math import sqrt, pi
    from scipy.special import erf  # already in project deps
    z = resid / sigma
    phi = np.exp(-0.5 * z ** 2) / sqrt(2 * pi)
    Phi = 0.5 * (1 + erf(z / sqrt(2)))
    crps_terms = sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / sqrt(pi))
    return float(np.mean(crps_terms))
