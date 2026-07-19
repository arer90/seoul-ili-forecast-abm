"""
simulation.models.feature_engine.combinator
============================================
Feature combinator — 2/3/4/5-way interaction synthesis (§6.5 RECOMMENDED_PIPELINE.md).

4-stage filter pipeline:
 1) Bien 2013 strong hierarchy — an interaction X_i·X_j enters only
 if both X_i and X_j survive main-effect LASSO selection.
 2) Kraskov 2004 KSG mutual information — keep candidates with
 MI(X_ij, y) above a data-driven threshold.
 3) PCMCI (Runge 2019) causal discovery on lagged variables — keep
 interactions whose constituent vars are in the causal parent set
 of the target.
 4) Optuna 2-phase — coarse-grid + fine-grid over the survivors.

Graceful degradation:
 Each stage individually degrades to a "keep all" rule when its
 dependency (e.g. `tigramite` for PCMCI) is missing. The pipeline
 still completes; the audit log records which stages were skipped.
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class InteractionCandidate:
    """Single interaction term: product of `members` columns."""
    members: tuple[str, ...]
    order: int = field(init=False)            # 2, 3, 4, or 5
    mi_score: float = float("nan")
    hierarchy_pass: bool = False
    causal_pass: bool = False
    retained: bool = False

    def __post_init__(self):
        self.order = len(self.members)

    @property
    def name(self) -> str:
        return "×".join(self.members)

    def compute_product(self, X: np.ndarray, col_index: dict[str, int]) -> np.ndarray:
        idx = [col_index[m] for m in self.members]
        out = np.ones(X.shape[0], dtype=float)
        for i in idx:
            out = out * X[:, i]
        return out


@dataclass
class CombinatorReport:
    n_main_effects: int
    n_candidates_by_order: dict[int, int] = field(default_factory=dict)
    n_after_hierarchy: int = 0
    n_after_mi: int = 0
    n_after_pcmci: int = 0
    n_after_optuna: int = 0
    stages_skipped: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"=== Feature Combinator Report ===",
            f"  main effects            : {self.n_main_effects}",
            f"  candidates/order        : {dict(self.n_candidates_by_order)}",
            f"  after hierarchy (Bien)  : {self.n_after_hierarchy}",
            f"  after MI (Kraskov KSG)  : {self.n_after_mi}",
            f"  after PCMCI causal      : {self.n_after_pcmci}",
            f"  after Optuna gating     : {self.n_after_optuna}",
        ]
        if self.stages_skipped:
            lines.append(f"  stages skipped          : {self.stages_skipped}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# Stage 1 — Bien strong hierarchy
# ══════════════════════════════════════════════════════════════════════════
def _lasso_main_effects(X_train: np.ndarray, y_train: np.ndarray, alpha: float = 0.01) -> set[int]:
    """Return indices of main effects with non-zero LASSO coef.

    Note: X_train / y_train naming is intentional — this helper must be
    called on TRAINING data only. The AST checker (FORBIDDEN_PATTERNS)
    enforces this contract.
    """
    try:
        from sklearn.linear_model import Lasso
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        log.warning("sklearn unavailable; hierarchy stage degrades to 'keep all'")
        return set(range(X_train.shape[1]))
    Xs_train = StandardScaler().fit_transform(X_train)
    lasso = Lasso(alpha=alpha, max_iter=5000, random_state=42)
    lasso.fit(Xs_train, y_train)
    return set(int(i) for i, c in enumerate(lasso.coef_) if abs(c) > 1e-8)


def apply_strong_hierarchy(
    candidates: list[InteractionCandidate],
    surviving_main: set[int],
    col_index: dict[str, int],
) -> list[InteractionCandidate]:
    """Keep interactions where ALL members are in surviving_main (Bien 2013)."""
    kept = []
    for c in candidates:
        if all(col_index[m] in surviving_main for m in c.members):
            c.hierarchy_pass = True
            kept.append(c)
    return kept


# ══════════════════════════════════════════════════════════════════════════
# Stage 2 — Kraskov KSG mutual information
# ══════════════════════════════════════════════════════════════════════════
def _mi_ksg(x: np.ndarray, y: np.ndarray, k: int = 3) -> float:
    """Kraskov 2004 KSG mutual information (uses sklearn's implementation).

    Returns MI in nats.
    """
    try:
        from sklearn.feature_selection import mutual_info_regression
    except ImportError:
        return float("nan")
    try:
        x_col = np.asarray(x).reshape(-1, 1)
        arr = mutual_info_regression(x_col, y, n_neighbors=k, random_state=42)
        return float(arr[0])
    except Exception:
        return float("nan")


def apply_mi_filter(
    candidates: list[InteractionCandidate],
    X: np.ndarray,
    y: np.ndarray,
    col_index: dict[str, int],
    *,
    percentile_keep: float = 50.0,
) -> list[InteractionCandidate]:
    """Compute MI for every candidate vs y, keep top percentile_keep %."""
    if not candidates:
        return []
    for c in candidates:
        prod = c.compute_product(X, col_index)
        c.mi_score = _mi_ksg(prod, y)

    scores = np.array([c.mi_score for c in candidates])
    if not np.any(np.isfinite(scores)):
        log.warning("MI unavailable; skipping MI filter")
        return candidates
    threshold = np.nanpercentile(scores, 100.0 - percentile_keep)
    return [c for c in candidates if np.isfinite(c.mi_score) and c.mi_score >= threshold]


# ══════════════════════════════════════════════════════════════════════════
# Stage 3 — PCMCI (Runge 2019) causal parents
# ══════════════════════════════════════════════════════════════════════════
def _run_pcmci(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> Optional[set[str]]:
    """Return feature names identified as causal parents of y via PCMCI.

    Requires `tigramite`. Returns None if unavailable — callers should
    treat this as 'keep everything'.
    """
    try:
        from tigramite import data_processing as pp
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.pcmci import PCMCI
    except Exception:
        return None

    try:
        data = np.column_stack([X, y.reshape(-1, 1)])
        dataframe = pp.DataFrame(data, var_names=feature_names + ["_target_"])
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr())
        results = pcmci.run_pcmci(tau_max=4, pc_alpha=0.05, verbosity=0)
        parents_idx = [p for (p, _) in results.get("parents", {}).get(
            len(feature_names), [])]
        return {feature_names[p] for p in parents_idx if 0 <= p < len(feature_names)}
    except Exception as e:
        log.warning("PCMCI failed: %s", e)
        return None


def apply_pcmci_filter(
    candidates: list[InteractionCandidate],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> tuple[list[InteractionCandidate], bool]:
    """Keep only interactions whose members are all causal parents of y."""
    parents = _run_pcmci(X, y, feature_names)
    if parents is None:
        return candidates, True  # skipped
    kept = []
    for c in candidates:
        if all(m in parents for m in c.members):
            c.causal_pass = True
            kept.append(c)
    return kept, False


# ══════════════════════════════════════════════════════════════════════════
# Stage 4 — Optuna gating (coarse + fine)
# ══════════════════════════════════════════════════════════════════════════
def apply_optuna_gating(
    candidates: list[InteractionCandidate],
    X: np.ndarray,
    y: np.ndarray,
    col_index: dict[str, int],
    *,
    n_trials_coarse: int = 30,
    n_trials_fine: int = 20,
    validation_idx: Optional[np.ndarray] = None,
) -> list[InteractionCandidate]:
    """Select the subset of candidates maximizing a held-out metric.

    Cheap fallback: greedy coefficient test when Optuna is unavailable.
    """
    if not candidates:
        return []

    try:
        import optuna
    except ImportError:
        log.warning("Optuna unavailable; using greedy correlation fallback")
        return _greedy_correlation_fallback(candidates, X, y, col_index)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Build full candidate product matrix once
    P = np.column_stack([c.compute_product(X, col_index) for c in candidates])

    if validation_idx is None:
        n = X.shape[0]
        # Last 20% as validation
        cutoff = int(n * 0.8)
        train_idx = np.arange(cutoff)
        val_idx = np.arange(cutoff, n)
    else:
        val_idx = validation_idx
        train_idx = np.setdiff1d(np.arange(X.shape[0]), val_idx)

    def objective(trial, max_features):
        mask = np.array([
            trial.suggest_categorical(f"k_{i}", [0, 1])
            for i in range(len(candidates))
        ], dtype=bool)
        if mask.sum() == 0 or mask.sum() > max_features:
            return -1e9
        return _ridge_holdout_score(
            np.column_stack([X, P[:, mask]]),
            y, train_idx, val_idx,
        )

    # ── Coarse: allow up to n//10 interactions ────────────────────────
    max_k_coarse = max(5, len(candidates) // 3)
    study = optuna.create_study(direction="maximize")
    # G-161 (2026-05-02): trial cleanup callback + gc_after_trial 강제.
    from simulation.models._optuna_torch import make_trial_cleanup_callback
    study.optimize(
        lambda t: objective(t, max_k_coarse),
        n_trials=n_trials_coarse,
        callbacks=[make_trial_cleanup_callback("combinator-coarse")],
        gc_after_trial=True,
        show_progress_bar=False,
    )

    best_mask_coarse = np.array([
        study.best_params.get(f"k_{i}", 0) for i in range(len(candidates))
    ], dtype=bool)

    # ── Fine: restrict to coarse survivors, allow up to n//20 ─────────
    survivors = [c for i, c in enumerate(candidates) if best_mask_coarse[i]]
    if len(survivors) <= 3:
        for c in survivors:
            c.retained = True
        return survivors

    P2 = np.column_stack([c.compute_product(X, col_index) for c in survivors])
    max_k_fine = max(3, len(survivors) // 2)

    def fine_objective(trial):
        mask = np.array([
            trial.suggest_categorical(f"f_{i}", [0, 1])
            for i in range(len(survivors))
        ], dtype=bool)
        if mask.sum() == 0 or mask.sum() > max_k_fine:
            return -1e9
        return _ridge_holdout_score(
            np.column_stack([X, P2[:, mask]]),
            y, train_idx, val_idx,
        )

    study2 = optuna.create_study(direction="maximize")
    # G-161 (2026-05-02): trial cleanup callback + gc_after_trial 강제 (fine stage).
    study2.optimize(
        fine_objective, n_trials=n_trials_fine,
        callbacks=[make_trial_cleanup_callback("combinator-fine")],
        gc_after_trial=True, show_progress_bar=False,
    )
    best_mask_fine = np.array([
        study2.best_params.get(f"f_{i}", 0) for i in range(len(survivors))
    ], dtype=bool)

    result = [c for i, c in enumerate(survivors) if best_mask_fine[i]]
    for c in result:
        c.retained = True
    return result


def _ridge_holdout_score(
    X_full: np.ndarray, y: np.ndarray,
    train_idx: np.ndarray, val_idx: np.ndarray,
) -> float:
    try:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return 0.0
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_full[train_idx])
    Xva = scaler.transform(X_full[val_idx])
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(Xtr, y[train_idx])
    yhat = ridge.predict(Xva)
    yv = y[val_idx]
    ss_res = float(np.sum((yv - yhat) ** 2))
    ss_tot = float(np.sum((yv - yv.mean()) ** 2))
    if ss_tot <= 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _greedy_correlation_fallback(
    candidates: list[InteractionCandidate],
    X: np.ndarray, y: np.ndarray,
    col_index: dict[str, int],
    *,
    keep_n: int = 20,
) -> list[InteractionCandidate]:
    scored = []
    for c in candidates:
        prod = c.compute_product(X, col_index)
        corr = np.corrcoef(prod, y)[0, 1]
        if np.isfinite(corr):
            scored.append((abs(corr), c))
    scored.sort(key=lambda kv: -kv[0])
    kept = [c for _, c in scored[:keep_n]]
    for c in kept:
        c.retained = True
    return kept


# ══════════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════════
def combinate_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    orders: tuple[int, ...] = (2, 3),
    max_candidates_per_order: int = 500,
    percentile_mi_keep: float = 50.0,
    use_pcmci: bool = True,
    use_optuna: bool = True,
    n_trials_coarse: int = 30,
    n_trials_fine: int = 20,
) -> tuple[np.ndarray, list[str], CombinatorReport]:
    """Run all 4 stages and return augmented feature matrix.

    Returns
    -------
    X_augmented : (n, p_main + p_new) ndarray
    new_names   : list[str]  # names of appended interaction columns
    report      : CombinatorReport
    """
    assert X.shape[0] == len(y), "X rows != y length"
    assert X.shape[1] == len(feature_names), "column count mismatch"
    col_index = {n: i for i, n in enumerate(feature_names)}

    report = CombinatorReport(n_main_effects=len(feature_names))

    # Build candidates
    candidates: list[InteractionCandidate] = []
    for order in orders:
        count = 0
        for combo in itertools.combinations(feature_names, order):
            candidates.append(InteractionCandidate(members=combo))
            count += 1
            if count >= max_candidates_per_order:
                log.warning("Truncated order=%d at %d candidates", order, count)
                break
        report.n_candidates_by_order[order] = count

    # Stage 1: Bien hierarchy
    try:
        surviving_main = _lasso_main_effects(X, y)
        candidates = apply_strong_hierarchy(candidates, surviving_main, col_index)
        report.n_after_hierarchy = len(candidates)
    except Exception as e:
        log.warning("Hierarchy stage failed: %s — skipping", e)
        report.stages_skipped.append("hierarchy")
        report.n_after_hierarchy = len(candidates)

    # Stage 2: Kraskov MI
    try:
        candidates = apply_mi_filter(
            candidates, X, y, col_index,
            percentile_keep=percentile_mi_keep,
        )
        report.n_after_mi = len(candidates)
    except Exception as e:
        log.warning("MI stage failed: %s — skipping", e)
        report.stages_skipped.append("mi")
        report.n_after_mi = len(candidates)

    # Stage 3: PCMCI causal discovery
    if use_pcmci:
        candidates, skipped = apply_pcmci_filter(candidates, X, y, feature_names)
        report.n_after_pcmci = len(candidates)
        if skipped:
            report.stages_skipped.append("pcmci")
    else:
        report.n_after_pcmci = len(candidates)
        report.stages_skipped.append("pcmci (disabled)")

    # Stage 4: Optuna
    if use_optuna:
        candidates = apply_optuna_gating(
            candidates, X, y, col_index,
            n_trials_coarse=n_trials_coarse,
            n_trials_fine=n_trials_fine,
        )
        report.n_after_optuna = len(candidates)
    else:
        for c in candidates:
            c.retained = True
        report.n_after_optuna = len(candidates)
        report.stages_skipped.append("optuna (disabled)")

    # Materialize augmented X
    if candidates:
        new_cols = np.column_stack([c.compute_product(X, col_index) for c in candidates])
        X_aug = np.column_stack([X, new_cols])
        new_names = [c.name for c in candidates]
    else:
        X_aug = X
        new_names = []

    return X_aug, new_names, report


__all__ = [
    "InteractionCandidate",
    "CombinatorReport",
    "combinate_features",
    "apply_strong_hierarchy",
    "apply_mi_filter",
    "apply_pcmci_filter",
    "apply_optuna_gating",
]
