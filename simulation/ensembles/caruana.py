"""
simulation.ensembles.caruana
============================
Caruana et al. 2004 forward stepwise selection with replacement.

Reference:
    Caruana, R., Niculescu-Mizil, A., Crew, G., & Ksikes, A. (2004).
    "Ensemble selection from libraries of models." ICML.

Algorithm:
    1. Start with empty ensemble ω.
    2. For each step t ∈ [1, T]:
         - For each candidate model m in the library L:
             compute R²_oof( ensemble(ω ∪ {m}) )  # with replacement
         - Pick m* = argmax of above and add to ω (may duplicate).
    3. Return the model weights = count of each model in ω / T.

Properties:
    * Selection with replacement prevents over-weighting rare-variance models.
    * Avoids normalization constraints on weights → robust to heterogeneous
      prediction scales.
    * Empirically matches NNLS/stacking with less risk of overfitting.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class CaruanaResult:
    selected_models: list[str]                   # order of selection (w/ duplicates)
    model_weights: dict[str, float]              # final normalized weights
    r2_trajectory: list[float] = field(default_factory=list)   # R² after each pick
    best_r2: float = -np.inf
    n_steps: int = 0

    def summary(self) -> str:
        pairs = sorted(self.model_weights.items(), key=lambda kv: -kv[1])
        lines = [
            f"Caruana forward stepwise — {self.n_steps} picks, best OOF R²={self.best_r2:.4f}",
        ]
        for name, w in pairs[:10]:
            lines.append(f"  {name:<35s} weight={w:.3f}")
        return "\n".join(lines)


def caruana_forward_stepwise(
    oof_predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    *,
    n_steps: int = 50,
    init_size: int = 1,
    metric: str = "r2",
    random_state: Optional[int] = None,
) -> CaruanaResult:
    """Select ensemble members via forward stepwise with replacement.

    Parameters
    ----------
    oof_predictions : dict[str, array]
        model_name → OOF prediction vector (all same length, same order).
    y_true : array
        True labels, same length as each prediction vector.
    n_steps : int
        Number of picks (typical: 50-100).
    init_size : int
        Warm-start ensemble with top-K best singleton models (default 1).
    metric : str
        "r2" (default), "neg_mse", "neg_mae".
    random_state : int
        Tie-breaking seed.
    """
    if not oof_predictions:
        raise ValueError("oof_predictions is empty")

    names = list(oof_predictions.keys())
    preds = np.stack([np.asarray(oof_predictions[n], dtype=float) for n in names])
    y = np.asarray(y_true, dtype=float)
    if preds.shape[1] != y.shape[0]:
        raise ValueError(
            f"OOF pred length {preds.shape[1]} != y_true length {y.shape[0]}"
        )

    rng = np.random.default_rng(random_state)
    score_fn = _make_score(metric)

    # ── Warm-start: pick top-K singletons ──────────────────────────────
    singleton_scores = np.array([score_fn(p, y) for p in preds])
    init_idx = np.argsort(-singleton_scores)[:init_size]
    picks: list[int] = list(int(i) for i in init_idx)
    trajectory: list[float] = []

    # Running sum (for incremental mean)
    running = preds[picks].sum(axis=0)
    counter = Counter(picks)

    best_score = score_fn(running / len(picks), y)
    trajectory.append(best_score)

    # ── Forward stepwise with replacement ─────────────────────────────
    for step in range(len(picks), n_steps):
        # For each candidate model, compute score of adding it (with replacement)
        best_idx = -1
        best_step_score = -np.inf
        for idx in range(preds.shape[0]):
            candidate_sum = running + preds[idx]
            candidate_mean = candidate_sum / (len(picks) + 1)
            s = score_fn(candidate_mean, y)
            if s > best_step_score or (
                np.isclose(s, best_step_score) and rng.random() < 0.5
            ):
                best_step_score = s
                best_idx = idx
        if best_idx < 0:
            break
        picks.append(best_idx)
        counter[best_idx] += 1
        running = running + preds[best_idx]
        trajectory.append(best_step_score)
        best_score = max(best_score, best_step_score)

    # ── Normalize weights ─────────────────────────────────────────────
    weights = {names[i]: counter[i] / len(picks) for i in counter}

    return CaruanaResult(
        selected_models=[names[i] for i in picks],
        model_weights=weights,
        r2_trajectory=trajectory,
        best_r2=float(best_score),
        n_steps=len(picks),
    )


# ══════════════════════════════════════════════════════════════════════════
# Metric factory
# ══════════════════════════════════════════════════════════════════════════
def _make_score(metric: str):
    if metric == "r2":
        return _r2
    if metric == "neg_mse":
        return lambda yhat, y: -float(np.mean((yhat - y) ** 2))
    if metric == "neg_mae":
        return lambda yhat, y: -float(np.mean(np.abs(yhat - y)))
    raise ValueError(f"Unknown metric: {metric}")


def _r2(yhat: np.ndarray, y: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return 0.0
    return 1.0 - ss_res / ss_tot
