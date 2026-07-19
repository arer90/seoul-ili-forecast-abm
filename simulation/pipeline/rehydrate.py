"""Rebuild the in-memory ``all_results`` dict from what a finished run left on disk.

The pipeline passes results between phases in a single ``all_results`` dict that
lives only for the duration of one process. Every launch starts it empty, so a
run resumed at R10 reaches the evaluation phases with nothing to evaluate, and a
targeted re-run (``--models A,B``) overwrites the phase checkpoints of the full
run it was supposed to supplement. Both are the same missing piece: nothing ever
reads the checkpoints back.

This module is that piece. It reads the phase checkpoints and the per-model
artifacts and reconstructs the dict under the keys the consuming phases expect.

**What can and cannot be recovered.** The checkpoints were written as a progress
log, not as a resume source, and two of them are lossy by construction:

  ``R2`` (baseline)  stores only ``{"baseline_n_models": N}`` — the model
                     predictions it produced are not in the file at all.
  ``R4`` (wfcv)      stores a subset, not the full walk-forward result.

So a rehydrated ``all_results`` is honest but incomplete: R5-R11 and P1 come
back whole, R2 and R4 do not. Callers are told which keys are missing rather
than being handed a dict that looks complete. R9 is reconstructed from
``per_model_optimal/*.json`` — the per-model files R9 writes as it goes — which
survive a targeted re-run even when the R9 checkpoint itself is overwritten.

Typical use::

    state = rehydrate_all_results(Path("simulation/results"))
    if "baseline" in state.missing:
        ...  # champion selection would see only part of the pool
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# Checkpoint label -> the all_results key the consuming phases read.
#
# These are NOT phases.name_of(): the runner uses "dm_tests" for R6 (whose
# semantic name is dm_test) and "prediction_intervals" for R7 (intervals), so
# the mapping has to be explicit. Every entry below is paired with the
# `all_results[...] = ` assignment in runner.py that it mirrors.
LABEL_TO_KEY: dict[str, str] = {
    "R5": "diagnostics",
    "R6": "dm_tests",
    "R7": "prediction_intervals",
    "R8": "scoring",
    "R9": "per_model_optimize",
    "R10": "per_model_eval",
    "R11": "feature_importance",
    "R12": "comprehensive_eval",
    "P1": "real_eval",
}

# Written as a progress log only — the payload does not contain enough to
# reconstruct what the phase produced.
LOSSY_LABELS: dict[str, str] = {
    "R1": "stores a filtered subset of phase1; the runner recomputes R1 anyway",
    "R2": "stores only baseline_n_models — the model predictions are absent",
    "R4": "stores a subset of the walk-forward result",
}


@dataclass
class RehydratedState:
    """``all_results`` rebuilt from disk, plus what could not be rebuilt.

    Attributes:
        results: the dict to hand to downstream phases.
        recovered: all_results keys that were restored, with their source.
        missing: keys a full run would have that this one cannot supply.
    """

    results: dict = field(default_factory=dict)
    recovered: dict[str, str] = field(default_factory=dict)
    missing: dict[str, str] = field(default_factory=dict)

    def __contains__(self, key: str) -> bool:
        return key in self.results


def _load_checkpoint(ckpt_dir: Path, label: str):
    p = ckpt_dir / f"checkpoint_{label}.json"
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        log.warning(f"  [rehydrate] {label} checkpoint unreadable: {e}")
        return None
    return payload.get("data")


def _rebuild_r9(results_dir: Path):
    """Reconstruct R9 from the per-model files it writes as it goes.

    Preferred over checkpoint_R9.json because a ``--models``-filtered re-run
    overwrites that checkpoint with just the filtered subset, while leaving
    every other model's per_model_optimal/<MODEL>.json intact.
    """
    d = results_dir / "per_model_optimal"
    if not d.is_dir():
        return None
    configs = {}
    for f in sorted(d.glob("*.json")):
        try:
            configs[f.stem] = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
    return {"per_model_configs": configs} if configs else None


def rehydrate_all_results(results_dir: Path) -> RehydratedState:
    """Rebuild ``all_results`` from a results directory.

    Args:
        results_dir: the run's output root, e.g. ``simulation/results`` —
            the directory holding ``checkpoints/`` and ``per_model_optimal/``.

    Returns:
        RehydratedState. ``.results`` is safe to hand to a consuming phase;
        ``.missing`` names the keys that could not be rebuilt and why, so a
        caller can refuse to proceed rather than silently evaluate a partial
        pool.

    Performance: O(number of checkpoint + per-model files); pure file reads,
        no network, no database. Typically well under a second.
    Side effects: none — reads only.
    Caller responsibility: check ``.missing`` before treating the result as a
        complete run. A phase that ranks models must not run on a partial pool
        without saying so.
    """
    results_dir = Path(results_dir)
    ckpt_dir = results_dir / "checkpoints"
    state = RehydratedState()

    for label, key in LABEL_TO_KEY.items():
        data = None
        source = ""
        if label == "R9":
            data = _rebuild_r9(results_dir)
            if data:
                source = f"per_model_optimal/ ({len(data['per_model_configs'])} models)"
        if data is None:
            data = _load_checkpoint(ckpt_dir, label)
            source = f"checkpoint_{label}.json"
        if data is None:
            state.missing[key] = f"{label}: no checkpoint on disk"
            continue
        # A phase that skipped produced nothing; carrying it forward as if it
        # held results is how an empty evaluation gets reported as a success.
        if isinstance(data, dict) and data.get("skipped"):
            state.missing[key] = f"{label}: skipped ({data.get('reason', 'no reason')})"
            continue
        state.results[key] = data
        state.recovered[key] = source

    # R10's consumers read the metrics CSV off disk rather than from the dict,
    # so point them at it even when the R10 checkpoint itself was overwritten.
    eval_dir = results_dir / "per_model_eval"
    csv_path = eval_dir / "per_model_metrics.csv"
    if csv_path.exists():
        node = state.results.get("per_model_eval")
        if not isinstance(node, dict):
            node = {}
            state.missing.pop("per_model_eval", None)
        node.setdefault("metrics_csv", str(csv_path))
        # The cross-phase Borda ranking needs the OOF top-10 that R10 recorded.
        # It lives in ranking.json, which a targeted re-run does not touch.
        rank_path = eval_dir / "ranking.json"
        if "ranking_top10" not in node and rank_path.exists():
            try:
                top = json.loads(rank_path.read_text(encoding="utf-8")).get("top10_by_oof_wis")
            except (json.JSONDecodeError, OSError):
                top = None
            if isinstance(top, list) and top:
                node["ranking_top10"] = [
                    e.get("model") if isinstance(e, dict) else e for e in top
                ]
        state.results["per_model_eval"] = node
        state.recovered.setdefault("per_model_eval", str(csv_path.name))

    for label, why in LOSSY_LABELS.items():
        key = {"R1": "phase1", "R2": "baseline", "R4": "wfcv"}[label]
        if key not in state.results:
            state.missing[key] = f"{label}: {why}"

    log.info(f"  [rehydrate] recovered {len(state.recovered)}, missing {len(state.missing)}")
    return state
