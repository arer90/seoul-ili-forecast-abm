"""Optuna trial budget helper .

Per-model Optuna trial count resolution for both in-process and subprocess
execution contexts. Runner sets MPH_OPTUNA_TRIALS_JSON (JSON-encoded
{model_name: int}) before spawning subprocesses; model classes call
``get_trials(name, default)`` to look up their budget.

Why env-var: subprocess workers (_run_model_in_subprocess) inherit parent
env, but cannot access the parent's Python objects. Env is the simplest
IPC channel for a small config dict.
"""

from __future__ import annotations

import json
import os
from typing import Dict

from simulation.config_global import GLOBAL  # SSOT (2026-05-28)

_ENV_KEY = "MPH_OPTUNA_TRIALS_JSON"
_CACHE: Dict[str, int] | None = None


def _load_budget() -> Dict[str, int]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    raw = os.environ.get(_ENV_KEY, "")
    if not raw:
        _CACHE = {}
        return _CACHE
    try:
        _CACHE = json.loads(raw)
    except Exception:
        _CACHE = {}
    return _CACHE


def get_trials(model_name: str, default: int = 20) -> int:
    """Per-model Optuna trial count. Falls back to `default` when unset.

    2026-05-28 사용자 명시 "HP trial default 동일 반영": default 50 → 20.
    Source-of-truth = MPH_HP_OPTUNA_TRIALS env (또는 MPH_OPTUNA_TRIALS_JSON model 별).

    Models with model-specific default (current sprint):
      - XGBoost / LightGBM / CatBoost: 20 (이미 일치, _optuna_budget default)
      - SVR-RBF: 25 → 20 통일 권장 (caller default override 가능)
      - GP-RBF-Periodic: 20 (일치)
      - DNN-Optuna / TCN-Optuna: class N_TRIALS (별도 통일 필요)
      - GE-DNN/GAT/TFT/PyG: 0 (opt-in, MPH_OPTUNA_TRIALS_JSON 에서 명시 시 활성)
    """
    # MPH_HP_OPTUNA_TRIALS env override (사용자 명시 2026-05-28 통일)
    _hp_trials = GLOBAL.optuna.hp_trials_default  # 0 = unset sentinel
    if _hp_trials > 0:
        default = max(5, _hp_trials)

    budget = _load_budget()
    val = budget.get(model_name)
    if val is None:
        return default
    try:
        return max(5, int(val))
    except Exception:
        return default


def set_budget(per_model_trials: Dict[str, int]) -> None:
    """Called by runner.py before subprocess spawn. Idempotent."""
    global _CACHE
    os.environ[_ENV_KEY] = json.dumps(per_model_trials)
    _CACHE = dict(per_model_trials)
