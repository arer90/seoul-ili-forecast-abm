"""Multi-seed bootstrap stability (audit Stage 3.1).

Multi-seed reproducibility framework. Default = single seed (42); env
`MPH_MULTI_SEED_RUN=1` enables ≥5 seed parallel runs for stability analysis.

Audit context (TRIPOD+AI 2024):
    단일 seed=42 의 best HP / champion 이 anecdote 일 수 있음 — TPE sampler 의
    stochastic nature. ≥5 seed bootstrap 의 stability score (champion 동일 비율,
    HP inter-seed variance) 필요.

⚠ Seed list correction (external audit 2026-05-27):
    SEED_LIST_DEFAULT = [13, 42, 137, 1729, 31415] 의 *명명* 가 종래 "Bergstra
    primes" 였으나 두 가지 부정확이 있음:
      (i) Bergstra & Bengio (2012, JMLR 13:281-305) 원논문에는 prime seed list
          권고 X — Mersenne Twister 의 3-seed 만 권고 (Section 2.4 neural-net experiments).
      (ii) 42 = 2×3×7, 1729 = 7×13×19 (Hardy-Ramanujan), 31415 = 5×61×103 모두
          *합성수* (only 13, 137 are prime).
    → seed list 자체는 유지 (학습 reproducibility 보호), 명명만 "arbitrary values
      spanning a wide magnitude range" 로 변경. paper Methods 에:
      "We chose 5 seeds spanning a wide range of magnitudes (13, 42, 137, 1729, 31415)
       to reduce common-mode pseudo-random correlations; specific values were
       arbitrary and not based on number-theoretic properties."

Reference:
    - Bergstra J, Bengio Y (2012) "Random search for hyper-parameter optimization"
      JMLR 13:281-305 (multi-seed reproducibility motivation; NB: prime list 권고
      *없음* — Mersenne Twister 3-seed 만 권고)
    - Bouthillier X et al. (2021) "Accounting for Variance in Machine Learning
      Benchmarks" MLSys 2021. arXiv:2103.03098 (seed diversity 권고; specific
      list 없음)
    - Picard D (2021) "Torch.manual_seed(3407) is all you need" arXiv:2109.08203
      (single-seed reporting 의 위험성)

D-5 gray-box contract:
    - O(N_seed × T_train_one_model) — env-gated to avoid 5× training time
    - seed_manifest: sampler + model fit + numpy + torch + CUDA seed traced
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from simulation.config_global import GLOBAL  # SSOT (2026-05-28)

__all__ = [
    "SEED_LIST_DEFAULT",
    "lock_global_seeds",
    "multi_seed_enabled",
    "compute_champion_stability",
    "compute_hp_inter_seed_variance",
    "build_seed_manifest",
]

#: Default seed list — arbitrary values spanning wide magnitude range (~10^1 to ~10^4).
#: NB: not all primes — 42, 1729, 31415 are composite (external audit 2026-05-27).
#: Bergstra & Bengio (2012) 권고는 multi-seed reporting 자체 — specific list 없음.
#: SSOT (2026-05-28): config_global.training.seed_list 참조 (단일 source + env override).
SEED_LIST_DEFAULT: list[int] = list(GLOBAL.training.seed_list)


def multi_seed_enabled() -> bool:
    """Env `MPH_MULTI_SEED_RUN=1` enables multi-seed training mode."""
    return GLOBAL.training.multi_seed_run


def get_seed_list() -> list[int]:
    """Returns seed list (env override `MPH_SEED_LIST=13,42,137,...` honored via GLOBAL)."""
    return list(GLOBAL.training.seed_list)


def lock_global_seeds(seed: int = 42, *, log: bool = False) -> dict:
    """Locks all RNG layers: numpy + torch (CPU+CUDA+MPS) + python random.

    Returns:
        dict — seed manifest (which layers were seeded).
    """
    manifest = {"requested_seed": seed, "layers": []}

    import random
    random.seed(seed)
    manifest["layers"].append("python.random")

    np.random.seed(seed)
    manifest["layers"].append("numpy.random")

    try:
        import torch
        torch.manual_seed(seed)
        manifest["layers"].append("torch.cpu")
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            manifest["layers"].extend(["torch.cuda", "cudnn.deterministic"])
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.manual_seed(seed) if hasattr(torch.mps, "manual_seed") else None
            manifest["layers"].append("torch.mps")
    except ImportError:
        pass

    # XGBoost, LightGBM, CatBoost 의 seed 는 model HP space 에서 명시 (per-model)
    # — XGBClassifier(random_state=seed), LGBMRegressor(random_state=seed)

    if log:
        import logging
        logging.getLogger(__name__).info(f"[seed] locked: {manifest}")
    return manifest


def compute_champion_stability(
    champions_per_seed: dict[int, str],
) -> dict:
    """Champion stability score across seeds.

    Args:
        champions_per_seed: {seed: champion_model_name}

    Returns:
        dict {
            "n_seeds": int,
            "unique_champions": list[str],
            "modal_champion": str,
            "stability_score": float (0-1, fraction of seeds with modal champion),
            "is_stable_50pct": bool,  # audit caveat — ≥50% same → reproducible
        }
    """
    out = {
        "n_seeds": len(champions_per_seed),
        "unique_champions": [],
        "modal_champion": "",
        "stability_score": 0.0,
        "is_stable_50pct": False,
    }
    if not champions_per_seed:
        return out
    from collections import Counter
    counts = Counter(champions_per_seed.values())
    modal, modal_count = counts.most_common(1)[0]
    stability = modal_count / len(champions_per_seed)
    out["unique_champions"] = list(counts.keys())
    out["modal_champion"] = modal
    out["stability_score"] = float(stability)
    out["is_stable_50pct"] = bool(stability >= 0.5)
    return out


def compute_hp_inter_seed_variance(
    best_hp_per_seed: dict[int, dict[str, float]],
    *,
    keys: Optional[list[str]] = None,
) -> dict:
    """Best HP inter-seed variance per HP key.

    Args:
        best_hp_per_seed: {seed: {hp_name: hp_value, ...}}
        keys: HP keys to track. None = union of all keys.

    Returns:
        dict {
            "<hp_name>": {"mean": float, "std": float, "cv": float, "n_seeds": int}
            ...
        }
    """
    out = {}
    if not best_hp_per_seed:
        return out
    if keys is None:
        all_keys = set()
        for d in best_hp_per_seed.values():
            all_keys.update(d.keys())
        keys = sorted(all_keys)
    for k in keys:
        vals = [d.get(k) for d in best_hp_per_seed.values() if d.get(k) is not None]
        finite_vals = [v for v in vals if isinstance(v, (int, float)) and np.isfinite(v)]
        if len(finite_vals) < 2:
            out[k] = {"mean": float("nan"), "std": float("nan"),
                       "cv": float("nan"), "n_seeds": len(finite_vals)}
            continue
        arr = np.array(finite_vals, dtype=np.float64)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1))
        cv = float(std / abs(mean)) if abs(mean) > 1e-9 else float("nan")  # coefficient of variation
        out[k] = {"mean": mean, "std": std, "cv": cv, "n_seeds": len(finite_vals)}
    return out


def build_seed_manifest(
    seed: int,
    *,
    sampler_seed: Optional[int] = None,
    model_seed: Optional[int] = None,
    cv_seed: Optional[int] = None,
) -> dict:
    """Reproducibility manifest — track all seed layers for paper reporting.

    Returns:
        dict with sampler/model/cv/global seeds + sub-RNG seeds.
    """
    return {
        "global_seed": seed,
        "sampler_seed": sampler_seed if sampler_seed is not None else seed,
        "model_seed": model_seed if model_seed is not None else seed,
        "cv_seed": cv_seed if cv_seed is not None else seed,
        "numpy_seed": seed,
        "torch_cpu_seed": seed,
        "torch_cuda_seed": seed,
        "python_random_seed": seed,
        "reference": "Bouthillier et al. (2021) arXiv:2103.03098",
    }
