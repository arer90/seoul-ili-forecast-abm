"""Hansen-Lunde-Nason Model Confidence Set (audit Stage 2.1, Task #16).

The Model Confidence Set (MCS) is a set of models that contains the best model
with a given level of confidence. Used as an alternative to single-champion
selection in R6 (dm_test) / audit_problem_models.

Reference:
    Hansen PR, Lunde A, Nason JM (2011)
    "The Model Confidence Set"
    Econometrica 79(2):453-497. doi:10.3982/ECTA5771
    Original quote: "A MCS is a set of models that is constructed such that it
    will contain the best model with a given level of confidence. The MCS is in
    this sense analogous to a confidence interval for a parameter."

Audit context (TRIPOD+AI 2024):
    53 model 동시 비교에서 pairwise DM 만 사용하면 transitivity 위반 + 다중성
    인플레이션. MCS_{90} membership 으로 "동등 best" set 보고 — champion 단일
    선택 회피.

Self-implementation:
    arch package 가 부재 (현 env). simplified self-implement — stationary
    bootstrap (Politis-Romano 1994 doi:10.1080/01621459.1994.10476870) +
    iterative elimination (Hansen et al. 2011 Algorithm 2.1):
        1. Compute loss matrix L (n_obs × n_model) — squared error / pinball / WIS.
        2. Compute pairwise t-statistics t_{ij} = mean(d_{ij}) / std_boot(d_{ij}).
        3. Iteratively eliminate models with the largest t_{i, max} until
           p-value > alpha (max t-statistic test).

D-5 gray-box contract:
    - O(M^2 * B) where M = n_model, B = n_bootstrap.
    - For M=52, B=1000 → ~2.7M ops, ~10-30s.
    - Stationary block bootstrap (block_size from Politis-White rule).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = [
    "compute_mcs",
    "mcs_pvalues",
    "stationary_bootstrap_indices",
]


def stationary_bootstrap_indices(
    n: int,
    block_size: int = 8,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Politis-Romano (1994) stationary block bootstrap indices.

    Args:
        n: original sample length.
        block_size: expected geometric block length.
        rng: numpy Generator (None → default seed 42).

    Returns:
        np.ndarray (n,) of int indices.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    p = 1.0 / max(int(block_size), 1)
    idx = np.zeros(n, dtype=np.int64)
    i = 0
    while i < n:
        start = int(rng.integers(0, n))
        block_len = max(1, int(rng.geometric(p)))
        for j in range(block_len):
            if i + j >= n:
                break
            idx[i + j] = (start + j) % n
        i += block_len
    return idx


def compute_mcs(
    loss_matrix: np.ndarray,
    model_names: list[str],
    *,
    alpha: float = 0.10,
    n_bootstrap: int = 1000,
    block_size: int = 8,
    seed: int = 42,
) -> dict:
    """Hansen-Lunde-Nason MCS via t-max iterative elimination.

    Implements simplified MCS algorithm (Hansen et al. 2011 Algorithm 2.1):
        1. Compute per-pair loss differential d_{ij} = L_i - L_j
        2. Bootstrap pairwise t-statistics t_{ij,b} (stationary block bootstrap)
        3. Max-statistic test: T_max = max_i t_{i, max}
        4. p-value = P_b(T_max,boot >= T_max,obs)
        5. Eliminate worst model if p < alpha, else stop

    Args:
        loss_matrix: (n_obs, n_model) — per-observation loss per model
                     (squared error, pinball loss, WIS, etc. — lower is better).
        model_names: list[str] — order matches columns of loss_matrix.
        alpha: 1 - confidence level. alpha=0.10 → MCS_{90}.
        n_bootstrap: 1000 default.
        block_size: stationary bootstrap mean block length (8 default).
        seed: 42 default.

    Returns:
        dict {
            "mcs_members": list[str],     # MCS_{1-alpha} member names
            "eliminated": list[(str, float)],  # (name, p_value at elimination)
            "alpha": float,
            "n_models": int,
            "n_obs": int,
            "n_bootstrap": int,
            "block_size": int,
            "reference": "Hansen, Lunde & Nason (2011) Econometrica 79(2):453-497",
        }

    Raises:
        절대 raise X (NaN-safe).

    Performance: O(M^2 * B * n_obs).
    Side effects: 없음 (pure).
    Caller responsibility:
        - loss_matrix shape (n_obs, n_model) with no NaN columns
        - Higher loss = worse (e.g. squared error, WIS — not r2)
        - Use stationary block bootstrap for time-series correlation
    """
    out = {
        "mcs_members": list(model_names),
        "eliminated": [],
        "alpha": alpha,
        "n_models": len(model_names),
        "n_obs": 0,
        "n_bootstrap": n_bootstrap,
        "block_size": block_size,
        "reference": "Hansen, Lunde & Nason (2011) Econometrica 79(2):453-497",
    }
    if loss_matrix is None or len(model_names) < 2:
        return out
    L = np.asarray(loss_matrix, dtype=np.float64)
    if L.ndim != 2 or L.shape[1] != len(model_names):
        return out
    n_obs, n_model = L.shape
    out["n_obs"] = n_obs
    if n_obs < block_size * 2:
        return out  # insufficient data

    # Drop columns with all-NaN
    valid_mask = ~np.all(np.isnan(L), axis=0)
    if not valid_mask.all():
        L = L[:, valid_mask]
        model_names = [m for i, m in enumerate(model_names) if valid_mask[i]]
        if len(model_names) < 2:
            out["mcs_members"] = list(model_names)
            return out

    rng = np.random.default_rng(seed)

    # Pre-compute bootstrap indices (reuse across iterations for stability)
    boot_indices = np.array([
        stationary_bootstrap_indices(n_obs, block_size, rng)
        for _ in range(n_bootstrap)
    ])  # (n_bootstrap, n_obs)

    current_idx = list(range(L.shape[1]))
    current_names = list(model_names)
    eliminated = []

    while len(current_idx) > 1:
        L_curr = L[:, current_idx]  # (n_obs, k)
        k = L_curr.shape[1]

        # Mean per model
        means = np.nanmean(L_curr, axis=0)  # (k,)

        # Pairwise differential matrix: d_{ij} = L_i - L_j  (k×k×n_obs is too big)
        # We use t_i = max_j (mean_i - mean_j) / sd_boot(mean_i - mean_j)
        # where sd_boot is the bootstrap std of (mean_i - mean_j)

        # Bootstrap mean per model
        boot_means = np.full((n_bootstrap, k), np.nan, dtype=np.float64)
        for b in range(n_bootstrap):
            idx = boot_indices[b]
            boot_means[b] = np.nanmean(L_curr[idx], axis=0)

        # Per-model t-statistic: t_i = max_j (means_i - means_j) / std_boot(means_i - means_j)
        # Higher t_i = worse model (loss higher than peers)
        t_stats = np.full(k, -np.inf, dtype=np.float64)
        for i in range(k):
            t_max_i = -np.inf
            for j in range(k):
                if i == j:
                    continue
                # observed difference
                d_obs = means[i] - means[j]
                # bootstrap std of difference
                d_boot = boot_means[:, i] - boot_means[:, j]
                d_boot = d_boot[np.isfinite(d_boot)]
                if len(d_boot) < 10:
                    continue
                sd = float(np.std(d_boot, ddof=1))
                if sd < 1e-12:
                    continue
                t_ij = float(d_obs / sd)
                if t_ij > t_max_i:
                    t_max_i = t_ij
            t_stats[i] = t_max_i

        # Test statistic = max(t_i)
        finite_t = t_stats[np.isfinite(t_stats)]
        if len(finite_t) == 0:
            break
        t_max_obs = float(np.max(finite_t))

        # Bootstrap distribution of T_max under null (equal loss)
        # Use centered bootstrap: re-center bootstrap differences to 0
        t_max_boot = np.empty(n_bootstrap, dtype=np.float64)
        for b in range(n_bootstrap):
            # Re-compute t-stats with boot_means[b] as "observed"
            t_b = -np.inf
            for i in range(k):
                for j in range(k):
                    if i == j:
                        continue
                    # centered: subtract observed mean diff
                    d_obs_ij = means[i] - means[j]
                    d_boot_b_ij = boot_means[b, i] - boot_means[b, j]
                    d_diff = d_boot_b_ij - d_obs_ij  # null = 0
                    # std across bootstrap
                    d_full = boot_means[:, i] - boot_means[:, j]
                    sd = float(np.std(d_full[np.isfinite(d_full)], ddof=1))
                    if sd < 1e-12:
                        continue
                    t_ij = float(d_diff / sd)
                    if t_ij > t_b:
                        t_b = t_ij
            t_max_boot[b] = t_b

        # p-value
        p_value = float(np.mean(t_max_boot >= t_max_obs))

        # If p > alpha, stop (cannot reject null of equal predictive ability)
        if p_value > alpha:
            break

        # Else eliminate model with largest t (worst)
        worst_local = int(np.argmax(t_stats))
        worst_name = current_names[worst_local]
        eliminated.append((worst_name, p_value))
        current_idx.pop(worst_local)
        current_names.pop(worst_local)

    out["mcs_members"] = list(current_names)
    out["eliminated"] = eliminated
    return out


def mcs_pvalues(
    loss_matrix: np.ndarray,
    model_names: list[str],
    *,
    n_bootstrap: int = 1000,
    block_size: int = 8,
    seed: int = 42,
) -> dict:
    """MCS p-value per model — probability that model is in MCS at level alpha.

    Useful for ranking — models with higher MCS p-value are "more in MCS".
    Implementation: incremental MCS computation (decreasing alpha).
    """
    out = {n: 0.0 for n in model_names}
    # Quick approximation: run compute_mcs at several alpha levels
    for alpha in [0.50, 0.25, 0.10, 0.05]:
        res = compute_mcs(loss_matrix, model_names, alpha=alpha,
                          n_bootstrap=n_bootstrap, block_size=block_size, seed=seed)
        for m in res["mcs_members"]:
            # Higher confidence MCS membership = higher p-value
            out[m] = max(out[m], 1.0 - alpha)
    return {
        "mcs_pvalues": out,
        "method": "incremental_alpha_scan",
        "reference": "Hansen, Lunde & Nason (2011) Econometrica 79(2):453-497",
    }
