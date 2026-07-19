"""Simulation-based inference (SBI / neural posterior) for behavioral params.

외부평가 3차 권고: ABC rejection을 **신경 사후추정(SNPE/NPE, sbi 패키지)**으로 격상해
약식별을 더 정밀히 정량화(Cranmer/Brehmer/Louppe 2020; Tejero-Cantero et al. 2020).

이 모듈은 *시뮬레이터-비종속* `run_sbi`만 제공한다(deep module). 행동 ABM 적용은
``scripts/sbi_posterior_calibration.py``. TDD(``test_sbi_calibration``)는 toy Gaussian
시뮬레이터로 파이프라인이 known param을 복원하는지 먼저 검증한 뒤 ABM에 적용한다.

sbi 부재 시 ImportError — 호출 측에서 ABC(`abc_posterior_calibration`)로 fallback.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def run_sbi(simulator: Callable[[np.ndarray], np.ndarray],
            lows, highs, x_obs, *, n_sims: int = 600, n_posterior: int = 2000,
            seed: int = 42) -> dict:
    """SNPE/NPE posterior for params θ given observed summary stats x_obs.

    Args:
        simulator: callable(theta_row 1-D np) → x_row 1-D np (summary statistics).
        lows, highs: per-parameter uniform-prior bounds (length D).
        x_obs: observed summary statistics (1-D np, length = x_dim).
        n_sims: prior simulations to train the neural posterior.
        n_posterior: posterior samples to draw at x_obs.
        seed: RNG seed (reproducibility).

    Returns:
        ``{samples (n_posterior, D) np, posterior_mean, ci95, ci_width_vs_prior,
        n_sims_used, library}``. ci95[i] = [2.5%, 97.5%] credible interval.

    Raises:
        ImportError: if ``sbi`` is not installed (caller may fall back to ABC).

    Performance: O(n_sims) simulator calls + 1 NPE train. Side effects: none.
    """
    import torch
    from sbi.inference import NPE
    from sbi.utils import BoxUniform

    torch.manual_seed(seed)
    np.random.seed(seed)
    lows = np.asarray(lows, dtype=np.float64)
    highs = np.asarray(highs, dtype=np.float64)
    D = len(lows)
    prior = BoxUniform(low=torch.tensor(lows, dtype=torch.float32),
                       high=torch.tensor(highs, dtype=torch.float32))
    inference = NPE(prior=prior)
    theta = prior.sample((n_sims,))
    x_rows = np.array([np.asarray(simulator(t.numpy()), dtype=np.float64) for t in theta])
    x = torch.tensor(x_rows, dtype=torch.float32)
    finite = torch.isfinite(x).all(dim=1) & torch.isfinite(theta).all(dim=1)
    theta, x = theta[finite], x[finite]
    if int(finite.sum()) < 20:
        raise RuntimeError(f"too few finite simulations ({int(finite.sum())})")
    inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior()
    xo = torch.tensor(np.asarray(x_obs, dtype=np.float64), dtype=torch.float32)
    samples = posterior.sample((n_posterior,), x=xo).numpy()

    out = {"samples": samples, "n_sims_used": int(finite.sum()), "library": "sbi.NPE",
           "posterior_mean": [], "ci95": [], "ci_width_vs_prior": []}
    for i in range(D):
        s = samples[:, i]
        ci = [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]
        out["posterior_mean"].append(round(float(s.mean()), 4))
        out["ci95"].append([round(ci[0], 4), round(ci[1], 4)])
        out["ci_width_vs_prior"].append(round((ci[1] - ci[0]) / (highs[i] - lows[i]), 3))
    return out
