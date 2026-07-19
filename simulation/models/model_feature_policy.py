"""
simulation.models.model_feature_policy
========================================
— 모델별 feature subset policy (spatial / mobility 변수 제외 정책).

배경 (session_v22.4.6_errors_and_fixes.md §1-E10):
 Metapop-SEIR 와 Bayesian-SEIR forecasting R² ≈ -8 로 REGISTRY 격리.
 원인은 구조적이었지만 (SEIR ODE 가 계절 ILI 를 못 맞춤), 다른 epi/physics
 모델 (NegBinGLM, PoissonAutoreg, GP-RBF) 역시 spatial coordinate
 (위경도, 지하철/버스 승객수) feature 가 섞이면 over-fit 위험이 있다.

 사용자 요청: "공간좌표나 지하철, 버스 같은 변수들이 있어서 문제가 있어서
 성능을 안 좋게 만들었다면 제외해야할까? 물리기반과 베이지안 문제점에서 말이야."

정책:
 FULL — 모든 feature (default for ML/DL models)
 MINIMAL — ILI lag + climate 만 (epi, physics 모델)
 NO_SPATIAL — spatial/mobility 제외 (bayesian)

사용:
 from simulation.models.model_feature_policy import resolve_feature_subset
 X_sub, feat_names_sub = resolve_feature_subset(X, feature_names, "MINIMAL")

시점: 정책만 정의. 적용은 phase6_wfcv._resolve_model_features_strict
에서 모델 meta 의 feature_policy 필드를 읽어 호출할 예정 (별도 PR).
"""
from __future__ import annotations

import logging
import re
from typing import Iterable, Literal

import numpy as np

log = logging.getLogger(__name__)


PolicyName = Literal["FULL", "MINIMAL", "NO_SPATIAL"]


# ══════════════════════════════════════════════════════════════════════════
# Feature category regex — 이름 기반 분류
# ══════════════════════════════════════════════════════════════════════════
# feature_engine 의 loader 들이 생성하는 feature 이름 패턴 기준:
#  - rolling_*, lag_*, ma_* : time-series endogenous
#  - ta_avg, ta_min, rn_day, hm_avg, ws_avg : weather
#  - subway_*, bus_* , pub_trans_* : mobility (HIRA 승객수)
#  - gu_lat, gu_lon, dist_* : spatial coordinates
#  - vax_*, vaccine_* : vaccination
#  - holiday_*, school_* : calendar
#  - google_*, trends_* : digital surveillance
#
# "MINIMAL" 은 ILI-lag + climate 만. "NO_SPATIAL" 은 lat/lon/subway/bus 만 제외.

_PATTERN_SPATIAL = re.compile(r"(gu_lat|gu_lon|dist_|centroid_|coord_)", re.I)
_PATTERN_MOBILITY = re.compile(r"(subway_|bus_|pub_trans_|transit_|ridership_|station_)", re.I)
_PATTERN_ILI_LAG = re.compile(r"(lag_\d+|rolling_\d+|ma_\d+|y_lag|target_lag)", re.I)
_PATTERN_CLIMATE = re.compile(r"(ta_avg|ta_min|ta_max|rn_day|hm_avg|hm_min|ws_avg|pa_avg|icsr)", re.I)
_PATTERN_ILI_ENDO = re.compile(r"(ili_|case_|weekly_case|new_case|notification_)", re.I)


def _match_category(name: str) -> str:
    """Feature name → category label."""
    if _PATTERN_SPATIAL.search(name):
        return "spatial"
    if _PATTERN_MOBILITY.search(name):
        return "mobility"
    if _PATTERN_ILI_LAG.search(name) or _PATTERN_ILI_ENDO.search(name):
        return "ili"
    if _PATTERN_CLIMATE.search(name):
        return "climate"
    return "other"


def resolve_feature_subset(
    X: np.ndarray,
    feature_names: Iterable[str],
    policy: PolicyName = "FULL",
) -> tuple[np.ndarray, list[str], list[int]]:
    """Apply feature subset policy to (X, feature_names).

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    feature_names : iterable of str, length n_features
    policy : 'FULL' | 'MINIMAL' | 'NO_SPATIAL'

    Returns
    -------
    X_sub : np.ndarray subset columns applied
    names_sub : list[str] matching column names
    idx : list[int] selected column indices (for reproducibility / PI alignment)
    """
    names = list(feature_names)
    if X.shape[1] != len(names):
        raise ValueError(
            f"X.shape[1]={X.shape[1]} != len(feature_names)={len(names)}"
        )

    if policy == "FULL":
        idx = list(range(len(names)))
    elif policy == "MINIMAL":
        idx = [i for i, n in enumerate(names) if _match_category(n) in ("ili", "climate")]
        if not idx:
            log.warning(
                "  [feature_policy] MINIMAL 요청했으나 ILI/climate 매칭 0 — FULL fallback"
            )
            idx = list(range(len(names)))
    elif policy == "NO_SPATIAL":
        idx = [
            i for i, n in enumerate(names)
            if _match_category(n) not in ("spatial", "mobility")
        ]
    else:
        raise ValueError(f"unknown policy: {policy}")

    names_sub = [names[i] for i in idx]
    X_sub = X[:, idx]

    # 로깅: 어떤 feature 가 제외되었는지 요약
    dropped = [names[i] for i in range(len(names)) if i not in set(idx)]
    cat_counts: dict[str, int] = {}
    for n in dropped:
        cat_counts.setdefault(_match_category(n), 0)
        cat_counts[_match_category(n)] += 1
    log.info(
        f"  [feature_policy={policy}] kept {len(idx)}/{len(names)} "
        f"(dropped by category: {cat_counts or 'none'})"
    )
    return X_sub, names_sub, idx


# ══════════════════════════════════════════════════════════════════════════
# Per-model policy lookup (meta.feature_policy 이 없으면 기본값)
# ══════════════════════════════════════════════════════════════════════════
# 이 매핑은 소스-of-truth 성격. model.meta.feature_policy 는 추후 ModelMeta
# 확장 시 추가할 예정이지만, 그 전에는 여기 lookup 으로 동작한다.

MODEL_POLICY: dict[str, PolicyName] = {
    # ── Epi / physics — spatial 제외 (hhh4/SEIR 계보) ──
    "NegBinGLM":        "NO_SPATIAL",
    "PoissonAutoreg":   "NO_SPATIAL",
    "GAM-Spline":       "NO_SPATIAL",
    "GP-RBF-Periodic":  "NO_SPATIAL",
    "BayesianMCMC":     "NO_SPATIAL",
    "BayesianRidge":    "NO_SPATIAL",
    # ── Physics / PINN — ILI+climate 만 ──
    "PINN-SEIR":        "MINIMAL",
    "MP-PINN":          "MINIMAL",
    # ── ML/DL — full spectrum OK ──
    # (미지정 = "FULL")
}


def policy_for_model(name: str) -> PolicyName:
    """모델 이름 → 적용할 policy."""
    return MODEL_POLICY.get(name, "FULL")
