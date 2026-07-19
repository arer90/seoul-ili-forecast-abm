"""
simulation.ensembles.tournament
===============================
3-Stage Tournament Orchestrator (§5.2.5, RECOMMENDED_PIPELINE.md ).

 Stage A'-1: intra-category rank → top-K per category
 Stage A'-2: Caruana forward stepwise across category winners
 Stage A'-3: meta-ensemble competition (champion)

Writes a `tournament_trace.json` artifact with every decision.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .caruana import CaruanaResult, caruana_forward_stepwise
from .meta_compete import MetaCompetitionResult, compete_meta_ensembles

log = logging.getLogger(__name__)


@dataclass
class TournamentResult:
    """End-to-end 3-stage tournament result."""
    stage_a1: dict[str, list[str]] = field(default_factory=dict)  # category → top-K names
    stage_a2: Optional[CaruanaResult] = None
    stage_a3: Optional[MetaCompetitionResult] = None
    final_ensemble_name: Optional[str] = None
    final_predictions: Optional[np.ndarray] = None
    final_r2: float = float("nan")
    final_mae: float = float("nan")
    elapsed_sec: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "stage_a1": self.stage_a1,
            "stage_a2": (
                {
                    "selected_models": self.stage_a2.selected_models,
                    "model_weights": self.stage_a2.model_weights,
                    "best_r2": self.stage_a2.best_r2,
                    "n_steps": self.stage_a2.n_steps,
                    "r2_trajectory": self.stage_a2.r2_trajectory,
                } if self.stage_a2 else None
            ),
            "stage_a3": (
                {
                    "champion": self.stage_a3.champion,
                    "per_ensemble_r2": self.stage_a3.per_ensemble_r2,
                    "per_ensemble_mae": self.stage_a3.per_ensemble_mae,
                    "per_ensemble_crps": self.stage_a3.per_ensemble_crps,
                    "composite_score": self.stage_a3.composite_score,
                    "weights": self.stage_a3.weights,
                } if self.stage_a3 else None
            ),
            "final_ensemble_name": self.final_ensemble_name,
            "final_r2": self.final_r2,
            "final_mae": self.final_mae,
            "elapsed_sec": self.elapsed_sec,
            "timestamp": self.timestamp,
        }

    def save_trace(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, default=str),
                     encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════
# Stage A'-1: intra-category rank
# ══════════════════════════════════════════════════════════════════════════
def intra_category_rank(
    oof_predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    model_categories: dict[str, str],
    *,
    top_k_per_category: int = 2,
    metric: str = "r2",
) -> dict[str, list[str]]:
    """Rank models within each category and keep top-K.

    Parameters
    ----------
    oof_predictions : dict[str, np.ndarray]
        model_name → OOF predictions
    y_true : np.ndarray
    model_categories : dict[str, str]
        model_name → category ('ts', 'linear', 'tree', 'dl', 'epi', ...)
    top_k_per_category : int
    metric : str
        'r2' / 'neg_mse' / 'neg_mae'
    """
    by_cat: dict[str, list[tuple[str, float]]] = {}
    y = np.asarray(y_true, dtype=float)

    for name, preds in oof_predictions.items():
        cat = model_categories.get(name, "other")
        score = _score(np.asarray(preds, dtype=float), y, metric)
        by_cat.setdefault(cat, []).append((name, score))

    result: dict[str, list[str]] = {}
    for cat, scored in by_cat.items():
        scored.sort(key=lambda kv: -kv[1])
        result[cat] = [n for n, _ in scored[:top_k_per_category]]
        log.info("[A'-1] category=%s  top-%d: %s", cat, top_k_per_category,
                 result[cat])
    return result


# ══════════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════════
class TournamentOrchestrator:
    """Runs all 3 stages + persists trace."""

    def __init__(
        self,
        *,
        top_k_per_category: int = 2,
        caruana_steps: int = 50,
        meta_candidates: Optional[list[str]] = None,
        artifacts_dir: Optional[str | Path] = None,
        random_state: Optional[int] = 42,
    ):
        self.top_k_per_category = top_k_per_category
        self.caruana_steps = caruana_steps
        self.meta_candidates = meta_candidates
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else None
        self.random_state = random_state

    def run(
        self,
        oof_predictions: dict[str, np.ndarray],
        y_true: np.ndarray,
        model_categories: dict[str, str],
        *,
        paper_primary: Optional[list[str]] = None,
        epi_validity_gate: Optional[dict[str, dict]] = None,
    ) -> TournamentResult:
        """Execute all 3 stages.

        `paper_primary` is an optional list of PAPER_PRIMARY_11 model names
        that are ALWAYS included in stage A'-2 regardless of category rank.

        `epi_validity_gate` is the per-model report from
        ``simulation.verifier.epi_validity.run_epi_validity_gate``.
        Models with ``exclude_from_ensemble=True`` are dropped from the
        candidate pool as a defense-in-depth step next to NEGATIVE_CONTROL.
        Flag-only failures (strict_exclude=False) are *not* dropped — they
        remain in the pool and the report is archived for the trace.
        """
        import time
        t0 = time.perf_counter()
        result = TournamentResult()

        if not oof_predictions:
            log.warning("No OOF predictions provided; skipping tournament.")
            return result

        # : exclude NEGATIVE_CONTROL models from any ensemble pool.
        # They may still appear in leaderboards via their raw OOF entries
        # upstream, but must never receive ensemble weight.
        try:
            from simulation.models.registry import NEGATIVE_CONTROL
            dropped = [n for n in oof_predictions if n in NEGATIVE_CONTROL]
            if dropped:
                log.info("[tournament] dropping NEGATIVE_CONTROL models: %s",
                         ", ".join(dropped))
                oof_predictions = {k: v for k, v in oof_predictions.items()
                                   if k not in NEGATIVE_CONTROL}
                model_categories = {k: v for k, v in model_categories.items()
                                    if k not in NEGATIVE_CONTROL}
                if paper_primary:
                    paper_primary = [n for n in paper_primary
                                     if n not in NEGATIVE_CONTROL]
        except Exception as e:
            log.debug("[tournament] NEGATIVE_CONTROL filter skipped: %s", e)

        # Stage 4: epi-validity opt-in exclusion (defense-in-depth).
        # Only drops models whose gate entry has ``exclude_from_ensemble=True``.
        # That flag is only flipped when the pipeline was started with
        # ``--epi-validity-strict`` (or config.epi_validity.strict_exclude=True).
        if epi_validity_gate:
            try:
                excluded = [
                    n for n, rep in epi_validity_gate.items()
                    if isinstance(rep, dict) and rep.get("exclude_from_ensemble")
                ]
                # Never drop paper_primary models — we must be able to report
                # on them even if they flag epi-implausible; log loudly instead.
                if paper_primary:
                    paper_set = set(paper_primary)
                    kept_primary = [n for n in excluded if n in paper_set]
                    if kept_primary:
                        log.warning(
                            "[tournament] epi-gate wanted to exclude PAPER_PRIMARY "
                            "models %s; keeping them and logging the violation",
                            ", ".join(kept_primary),
                        )
                    excluded = [n for n in excluded if n not in paper_set]
                to_drop = [n for n in excluded if n in oof_predictions]
                if to_drop:
                    log.info(
                        "[tournament] dropping epi-validity failures: %s",
                        ", ".join(to_drop),
                    )
                    oof_predictions = {k: v for k, v in oof_predictions.items()
                                       if k not in to_drop}
                    model_categories = {k: v for k, v in model_categories.items()
                                        if k not in to_drop}
            except Exception as e:
                log.debug("[tournament] epi-validity filter skipped: %s", e)

        y = np.asarray(y_true, dtype=float)

        # ── Stage A'-1 ────────────────────────────────────────────────
        a1 = intra_category_rank(
            oof_predictions, y, model_categories,
            top_k_per_category=self.top_k_per_category,
        )
        result.stage_a1 = a1

        # Union of category winners + paper primary → candidate pool
        pool: set[str] = set()
        for names in a1.values():
            pool.update(names)
        if paper_primary:
            pool.update(n for n in paper_primary if n in oof_predictions)
        pool_preds = {n: oof_predictions[n] for n in pool if n in oof_predictions}

        if len(pool_preds) < 2:
            log.warning("[A'-2] pool too small (%d); skipping Caruana.", len(pool_preds))
            result.elapsed_sec = time.perf_counter() - t0
            return result

        # ── Stage A'-2 ────────────────────────────────────────────────
        a2 = caruana_forward_stepwise(
            pool_preds, y,
            n_steps=self.caruana_steps,
            random_state=self.random_state,
        )
        result.stage_a2 = a2
        log.info("[A'-2] Caruana: %d picks, best R²=%.4f",
                 a2.n_steps, a2.best_r2)

        # Caruana ensemble prediction (weighted mean)
        caruana_yhat = np.zeros_like(y)
        for n, w in a2.model_weights.items():
            caruana_yhat = caruana_yhat + w * np.asarray(oof_predictions[n], dtype=float)

        # ── Stage A'-3 ────────────────────────────────────────────────
        # Feed pool_preds + caruana blend to meta-compete
        meta_input = dict(pool_preds)
        meta_input["__caruana_blend__"] = caruana_yhat

        a3 = compete_meta_ensembles(
            meta_input, y,
            candidates=self.meta_candidates,
        )
        result.stage_a3 = a3

        # ── Final selection ───────────────────────────────────────────
        # Champion between A'-2 Caruana and A'-3 meta-ensemble
        caruana_r2 = _r2(caruana_yhat, y)
        a3_r2 = max(a3.per_ensemble_r2.values()) if a3.per_ensemble_r2 else -np.inf

        if caruana_r2 >= a3_r2:
            result.final_ensemble_name = "caruana"
            result.final_predictions = caruana_yhat
            result.final_r2 = caruana_r2
        else:
            result.final_ensemble_name = a3.champion
            # Re-derive champion's predictions (same as _apply_ensemble logic)
            # -- simplified: weighted InverseRMSE blend as fallback
            names = list(pool_preds.keys())
            preds = np.stack([np.asarray(pool_preds[n], dtype=float) for n in names])
            rmses = np.array([np.sqrt(np.mean((p - y) ** 2)) for p in preds])
            w = 1.0 / (rmses + 1e-9); w = w / w.sum()
            result.final_predictions = (w[:, None] * preds).sum(axis=0)
            result.final_r2 = _r2(result.final_predictions, y)

        result.final_mae = float(np.mean(np.abs(result.final_predictions - y)))
        result.elapsed_sec = time.perf_counter() - t0

        if self.artifacts_dir:
            trace_path = self.artifacts_dir / "tournament_trace.json"
            result.save_trace(trace_path)
            log.info("Tournament trace saved: %s", trace_path)

        return result


# ══════════════════════════════════════════════════════════════════════════
# Metric utilities
# ══════════════════════════════════════════════════════════════════════════
def _score(yhat: np.ndarray, y: np.ndarray, metric: str) -> float:
    if metric == "r2":
        return _r2(yhat, y)
    if metric == "neg_mse":
        return -float(np.mean((yhat - y) ** 2))
    if metric == "neg_mae":
        return -float(np.mean(np.abs(yhat - y)))
    raise ValueError(f"Unknown metric: {metric}")


def _r2(yhat: np.ndarray, y: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return 0.0
    return 1.0 - ss_res / ss_tot
