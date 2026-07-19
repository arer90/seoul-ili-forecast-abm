"""G-280 (2026-06-16, 사용자): preproc Optuna n_startup 비율-기반 → transform 커버리지.

사건: default n_startup=10 + plateau-stop(25) → log1p 가 random 탐색서 거의 안 뽑혀(실측
53모델 중 1개만 log1p 선택) TPE 가 exploit 못 함. 비율(MPH_OPTUNA_STARTUP_RATIO=0.4)로
올려 STABLE transform pool 전체가 충분히 sampled 되게 보장(TPE 수렴=절대개수 와 별개 = 커버리지).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

optuna = pytest.importorskip("optuna")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _n_startup(n_trials, ratio=0.4, floor=15):
    """_inline_optuna_3stage.py 의 비율 공식 (회귀 가드)."""
    v = max(floor, int(ratio * n_trials))
    return min(v, max(1, n_trials - 1))


def test_startup_formula():
    assert _n_startup(100) == 40            # 0.4 × 100
    assert _n_startup(50) == 20             # 0.4 × 50
    assert _n_startup(20) == 15             # floor 15
    assert _n_startup(8) == 7               # n_trials-1 cap
    assert _n_startup(100, ratio=0.5) == 50


def test_proportion_startup_covers_all_stable_transforms():
    """비율 startup 이면 STABLE Y-transform 5개 전부 sampled (log1p 포함)."""
    os.environ["MPH_STABLE_TRANSFORMS"] = "1"
    from simulation.pipeline.preproc_optuna_hierarchical import (
        suggest_y_preproc, STABLE_Y_TRANSFORMS)
    y = np.abs(np.random.RandomState(0).randn(200) * 5 + 20)
    seen = set()

    def obj(t):
        _, _, st = suggest_y_preproc(t, y.copy())
        if st.get("y_mode") == "individual":
            seen.add(st.get("y_individual"))
        return 0.0

    n_trials = 60
    s = optuna.create_study(
        sampler=optuna.samplers.TPESampler(n_startup_trials=_n_startup(n_trials), seed=1))
    s.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    # log1p 가 반드시 시도됨 + 5개 중 ≥4개 커버
    assert "log1p" in seen, f"log1p 미탐색 (커버리지 실패): {seen}"
    assert len(seen & set(STABLE_Y_TRANSFORMS)) >= 4, f"커버리지 부족: {seen}"


def test_inline_applies_env_ratio():
    """_inline_optuna_3stage 가 비율 env 를 읽는다 (배선 확인)."""
    import inspect
    from simulation.pipeline import _inline_optuna_3stage as m
    src = inspect.getsource(m)
    assert "MPH_OPTUNA_STARTUP_RATIO" in src
    assert "n_startup_trials=_n_startup" in src


def _patience(n_trials, n_startup, configured=25):
    """The plateau-patience formula at _inline_optuna_3stage.py:971."""
    return max(8, min(configured, n_trials - n_startup - 2))


def test_plateau_patience_leaves_room_after_startup():
    """Plateau-stop must be able to fire, and must not eat the random startup.

    2026-07-19: this asserted the source text contained ``"_n_startup + 5"``.
    That expression was deliberately removed by G-329f — it made patience LARGER
    than the post-startup budget (60 trials, 30 startup, patience 35 > 30), so
    ``plateau.stop()`` could never fire and the early-stop was dead code. The
    replacement caps patience BELOW the post-startup budget. Asserting the old
    string made the fix look like a regression, so assert the property both
    formulas were reaching for instead.
    """
    for n_trials in (20, 40, 60, 100, 200):
        for ratio in (0.3, 0.4, 0.5):
            n_startup = _n_startup(n_trials, ratio=ratio)
            pat = _patience(n_trials, n_startup)
            post_startup = n_trials - n_startup
            assert pat >= 8, (n_trials, ratio, pat)
            if post_startup > 10:
                assert pat <= post_startup, (
                    f"n_trials={n_trials} ratio={ratio}: patience {pat} exceeds the "
                    f"{post_startup} trials left after startup — plateau-stop can "
                    f"never fire (the G-329f dead-code bug)"
                )
