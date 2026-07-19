"""Per-feature data-driven preprocessor (2026-04-29).

사용자 통찰:
  "각 변수들이 역할별로 따로 따로 나와있는데 근거가 있어?
   내 생각에는 테스트로 전체를 말해서 해야하지 않을까?"

  → 그룹별 매핑 (textbook 추측) 대신 각 feature 의 실제 분포로 결정
  → 모델별로도 다른 처리 필요 (사용자 두 번째 통찰)

설계:
  Tier 1 (per-feature): 각 column 의 실제 분포 stats (skew, neg, zero, ...)
                         → 데이터 기반 transform 권장
  Tier 2 (per-model):  모델 카테고리 (tree/linear/kernel/dl/glm/...)
                         → 적합한 처리 lookup
  Tier 3 (결합):        두 정보로 각 column 의 transform 자동 결정

학술 근거:
  · Box-Cox (1964): 적정 transform 은 데이터 분포에 의존
  · Anscombe (1948), Freeman-Tukey (1950): Poisson VST
  · Hastie ESL §3.4: linear 모델 scale 정규화
  · Schölkopf (2001): kernel distance 정규화
  · McCullagh-Nelder (1989): GLM link function 이 transform 내장
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# Tier 1 — Per-feature 분포 측정
# ════════════════════════════════════════════════════════════════
def profile_feature(x: np.ndarray) -> dict:
    """한 column 의 분포 stats."""
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    x_clean = x[finite]
    if len(x_clean) < 3:
        return {"error": "insufficient"}

    n_unique = int(len(np.unique(x_clean)))
    if n_unique == 1:
        return {"is_constant": True}

    mean = float(np.mean(x_clean))
    std = float(np.std(x_clean))
    if std < 1e-12:
        return {"is_constant": True}

    return {
        "is_constant": False,
        "is_binary": n_unique == 2,
        "min": float(np.min(x_clean)),
        "max": float(np.max(x_clean)),
        "mean": mean,
        "median": float(np.median(x_clean)),
        "std": std,
        "skew": float(np.mean(((x_clean - mean) / std) ** 3)),
        "kurt": float(np.mean(((x_clean - mean) / std) ** 4) - 3.0),
        "neg_ratio": float(np.mean(x_clean < 0)),
        "zero_ratio": float(np.mean(np.abs(x_clean) < 1e-9)),
        "is_bounded_unit": bool(np.min(x_clean) >= -0.001 and np.max(x_clean) <= 1.001),
        "is_bounded_pm1": bool(np.min(x_clean) >= -1.001 and np.max(x_clean) <= 1.001),
        "n_unique": n_unique,
    }


# ════════════════════════════════════════════════════════════════
# Tier 2 — Model category lookup
# ════════════════════════════════════════════════════════════════
_MODEL_CATEGORY = {
    # Tree (split-based, monotonic invariant)
    "LightGBM": "tree", "XGBoost": "tree", "RandomForest": "tree",
    "GradientBoosting": "tree", "ExtraTrees": "tree",
    # Linear
    "ElasticNet": "linear", "Ridge": "linear", "Lasso": "linear",
    "SVR-Linear": "linear", "KRR": "linear", "BayesianRidge": "linear",
    # Kernel
    "SVR-RBF": "kernel", "GP-RBF-Periodic": "kernel",
    # DL
    "DNN": "dl", "DNN-Optuna": "dl", "TabularDNN-Lite": "dl",
    "TFT": "dl", "PatchTST": "dl",
    "iTransformer": "dl", "TimesNet": "dl", "Mamba": "dl",
    "TCN": "dl", "TCN-Optuna": "dl",
    "N-BEATS": "dl", "N-HiTS": "dl",
    "TiDE": "dl",
    "DeepAR": "dl", "RNN": "dl",
    "GCN": "dl", "GAT": "dl",
    # Bayesian GLM (link 내장)
    "NegBinGLM": "glm", "NegBinGLM-V7": "glm",
    "PoissonAutoreg": "glm", "BayesianMCMC": "glm",
    # GAM
    "GAM-Spline": "gam",
    # TS (자체 boxcox)
    "ARIMA": "ts", "SARIMA": "ts", "SARIMAX": "ts",
    # Mechanistic (X 안 받음)
    "Bayesian-SEIR": "mechanistic", "Metapop-SEIR": "mechanistic",
    "SEIRForcedForecaster": "mechanistic",   # class name compat (legacy)
    "SEIR-V2-Forced": "mechanistic",          # registry name (Phase B)
    "PINN-Lite": "mechanistic", "MP-PINN": "mechanistic",
    "Rt-Augmented": "mechanistic",
    # Foundation
    # G-261 (2026-06-13): Chronos 전 변형 제거 — Chronos retire. 대체 = TimesFM-2.5 + TiRex.
    "TimesFM-2.5": "foundation", "TiRex": "foundation",
    "OverseasTransfer": "foundation",
    # Ensemble (passthrough preprocessing + y-transform none)
    # 66model T7_Ensemble 10개 기준 (사용자 명시 2026-05-25):
    # 제거: Ensemble-TopTierStacking, FluSight-Ensemble, Phase-Adaptive, Ensemble-Stacking
    "Ensemble-NNLS": "ensemble", "Ensemble-NNLS-Filtered": "ensemble",
    "Ensemble-BMA": "ensemble", "Ensemble-SelectiveBMA": "ensemble",
    "Ensemble-InvRMSE": "ensemble", "Ensemble-Temporal": "ensemble",
    "Ensemble-Diversity": "ensemble", "Ensemble-Adaptive": "ensemble",
    "Ensemble-Blending": "ensemble", "Ensemble-ResidualAR": "ensemble",
    # DL (missing entries)
    "TabularDNN": "dl", "DNN-Conformal": "dl",
    # CQR (Romano 2019 NeurIPS)
    "CQR-LightGBM": "tree", "CQR-GBR": "tree", "CQR-QuantReg": "linear",
}


def get_model_category(model_name: str) -> str:
    return _MODEL_CATEGORY.get(model_name, "default")


# ════════════════════════════════════════════════════════════════
# Tier 3 — Combined: per-feature stats + model category → transform
# ════════════════════════════════════════════════════════════════
def recommend_transform(stats: dict, model_category: str) -> str:
    """각 feature 의 (분포 stats, 모델 카테고리) → transform 결정.

    학술 근거:
      · Box-Cox: positive + skewed
      · Yeo-Johnson: mixed-sign + skewed
      · Anscombe: Poisson mean>1
      · Freeman-Tukey: Poisson low-mean
      · Tree (Breiman): monotonic 무관 → none
      · Linear/Kernel/DL: scale 정규화 필수
      · GLM (McCullagh-Nelder): Y transform 금지 (link 내장)

    Returns: transform name (str)
    """
    if "error" in stats or stats.get("is_constant"):
        return "drop"
    if stats.get("is_binary"):
        return "passthrough"
    if stats.get("is_bounded_pm1") and abs(stats.get("mean", 0)) < 0.1:
        return "passthrough"   # cyclic-like

    # Tree: monotonic transform 무관 → robust (사용자 선택 A)
    # 이론: split 은 raw value OK 지만 numerical precision 위해 light scaling 권장
    # ESL Friedman §10.7: Tree-based 도 outlier 정규화 도움
    # 1e15 같은 극단적 scale 의 features (인구이동) 가 있을 때 안전
    if model_category == "tree":
        return "robust"

    # Mechanistic / Ensemble / TS: X 안 받거나 자체 처리
    if model_category in ("mechanistic", "ensemble", "ts"):
        return "none"

    # Foundation: 자체 representation
    if model_category == "foundation":
        return "none"

    # GLM: Y transform 금지 (link 내장)
    # 단 X 는 light scaling 가능 (기능별 결정)
    # → X 는 일반 규칙 따라 결정 (Y 만 별도 처리)

    # 일반 (Linear/Kernel/DL/GAM/GLM-X): 분포 기반
    skew = stats.get("skew", 0.0)
    neg = stats.get("neg_ratio", 0.0)
    zero = stats.get("zero_ratio", 0.0)
    mean = stats.get("mean", 0.0)
    is_unit = stats.get("is_bounded_unit", False)

    # bounded [0, 1] proportion
    if is_unit:
        if model_category in ("kernel", "dl"):
            return "standard"   # kernel/DL 표준화
        return "arcsine_sqrt"

    # mixed-sign + skewed
    if neg > 0.05:
        if abs(skew) > 1.5:
            return "yeo_johnson"
        return "standard"   # neg + symmetric

    # all positive, sparse
    if zero > 0.3:
        if mean > 1.0:
            return "anscombe"
        return "freeman_tukey"

    # all positive, strongly right-skewed
    if skew > 2.5:
        return "log1p"

    # all positive, moderate skew
    if skew > 1.0:
        return "sqrt"

    # all positive, near-normal
    if model_category in ("kernel", "dl", "linear"):
        return "standard"
    return "robust"


def get_y_transform(y: np.ndarray, model_category: str) -> str:
    """Y target 의 transform 결정.

    GLM (link 내장) → identity 강제 (double-log 회피)
    ARIMA (자체 boxcox) → identity
    Mechanistic → identity
    Tree/Linear/Kernel/DL/GAM → 분포 기반 (log1p / identity)
    """
    if model_category in ("glm", "ts", "mechanistic", "ensemble"):
        return "identity"

    y_clean = y[np.isfinite(y)]
    if len(y_clean) < 3:
        return "identity"
    if np.any(y_clean < 0):
        return "identity"   # 음수 있으면 log1p 안 됨
    skew = float(np.mean(((y_clean - y_clean.mean()) / (y_clean.std() + 1e-12)) ** 3))
    return "log1p" if skew > 1.0 else "identity"


# ════════════════════════════════════════════════════════════════
# Transformer instantiation helpers
# ────────────────────────────────────────────────────────────────
# Sprint 1.5 R7 (2026-05-26): 7 VST helpers 통합 →
#   simulation/models/_vst_primitives.py (single source of truth).
# 이전: 본 파일이 각 7개 def 로컬 정의 (~52줄). grouped_preprocessor 와 byte-identical.
# 현재: import alias. caller 17곳 (호출 코드 변경 X, 이름 동일).
# `_arcsinh_t` default scale=10.0 (per_feature legacy 호환 — _vst_primitives 도 동일).
# ════════════════════════════════════════════════════════════════
from simulation.models._vst_primitives import (
    _log1p_t,
    _sqrt_t,
    _anscombe_t,
    _freeman_tukey_t,
    _arcsinh_t,
    _arcsine_sqrt_t,
    _yeo_johnson_t,
)


def _build_op(name: str, model_category: str = "default"):
    """Transform name → instance. Tree 모델이면 (분포 무관) passthrough."""
    if name in ("drop", "none", "passthrough"):
        return "passthrough"
    if name == "log1p":
        return Pipeline([("log1p", _log1p_t()),
                         ("scale", _scale_for(model_category))])
    if name == "sqrt":
        return Pipeline([("sqrt", _sqrt_t()),
                         ("scale", _scale_for(model_category))])
    if name == "anscombe":
        return Pipeline([("anscombe", _anscombe_t()),
                         ("scale", _scale_for(model_category))])
    if name == "freeman_tukey":
        return Pipeline([("freeman_tukey", _freeman_tukey_t()),
                         ("scale", _scale_for(model_category))])
    if name == "arcsinh":
        return Pipeline([("arcsinh", _arcsinh_t()),
                         ("scale", _scale_for(model_category))])
    if name == "arcsine_sqrt":
        return _arcsine_sqrt_t()
    if name == "yeo_johnson":
        return _yeo_johnson_t()
    if name == "standard":
        return StandardScaler()
    if name == "robust":
        return RobustScaler()
    return RobustScaler()   # default


def _scale_for(model_category: str):
    """모델 카테고리에 적합한 scaler."""
    if model_category in ("tree", "mechanistic", "ts", "ensemble", "foundation"):
        return FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x,
                                     validate=False)
    if model_category in ("kernel", "dl"):
        return StandardScaler()
    return RobustScaler()


# ════════════════════════════════════════════════════════════════
# Main entry — build per-feature ColumnTransformer
# ════════════════════════════════════════════════════════════════
def build_per_feature_preprocessor(
    feature_cols: list[str],
    X_train: np.ndarray,
    model_name: str = "default",
    return_decisions: bool = False,
) -> object:
    """각 feature 의 분포 + model category → 자동 transform → ColumnTransformer.

    사용자 통찰 반영:
      ① 그룹 가정 X (per-feature)
      ② 데이터 기반 (실제 분포 측정)
      ③ 모델별 적합 (tree/linear/kernel/dl/glm/ts/mechanistic 분기)

    Args:
        feature_cols: 피처 이름 리스트
        X_train: train data (분포 측정용)
        model_name: 모델 이름 (categori 결정)
        return_decisions: True 시 (preprocessor, decisions_dict) 반환

    Returns:
        ColumnTransformer (or with decisions tuple)
    """
    model_category = get_model_category(model_name)

    transformers = []
    decisions = {}
    for i, col in enumerate(feature_cols):
        stats = profile_feature(X_train[:, i])
        rec = recommend_transform(stats, model_category)
        decisions[col] = {
            "transform": rec,
            "skew": stats.get("skew"),
            "neg_ratio": stats.get("neg_ratio"),
            "zero_ratio": stats.get("zero_ratio"),
            "is_constant": stats.get("is_constant", False),
        }
        if rec == "drop":
            continue   # constant feature 자동 제외
        op = _build_op(rec, model_category)
        transformers.append((f"f{i}", op, [i]))

    if not transformers:
        # fallback — 모든 features 가 constant 인 극단적 경우
        from sklearn.preprocessing import RobustScaler as _RS
        return (_RS() if not return_decisions else (_RS(), decisions))

    ct = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0,
        verbose_feature_names_out=False,
    )

    if return_decisions:
        return ct, decisions
    return ct


def summarize_decisions(decisions: dict) -> dict:
    """decision dict → category 별 카운트."""
    from collections import Counter
    counter = Counter(v["transform"] for v in decisions.values())
    return dict(counter)


# ════════════════════════════════════════════════════════════════
# Per-feature Optuna with combo (2026-04-29 신규)
# ────────────────────────────────────────────────────────────────
# 사용자 통찰: "여기서 1개만 사용하는거야? 조합은?"
#
# 기존 (deterministic recommend_transform): 1개 transform 선택
# 신규: (log_op, scale_op) 조합 분리 search
#
# 핵심:
#   1. 분포 stats → log_op menu 결정 (constant→drop, binary→passthrough,
#                                       skew → [log1p, sqrt, yeo_johnson, ...])
#   2. model_category → scale_op menu 결정 (Tree → [robust, passthrough],
#                                            Linear/DL → [robust, standard, ...])
#   3. Optuna 가 두 menu 안에서 각각 search → 진짜 조합
# ════════════════════════════════════════════════════════════════

def _log_menu_for_stats(stats: dict) -> list[str]:
    """분포 stats → log/transform op 후보 menu (좁힘)."""
    if stats.get("is_constant"):
        return ["drop"]
    if stats.get("is_binary"):
        return ["none"]
    if stats.get("is_bounded_pm1") and abs(stats.get("mean", 0)) < 0.1:
        return ["none"]   # cyclic
    if stats.get("is_bounded_unit"):
        return ["arcsine_sqrt", "logit", "none"]   # proportion

    skew = stats.get("skew", 0.0)
    neg = stats.get("neg_ratio", 0.0)
    zero = stats.get("zero_ratio", 0.0)
    mean = stats.get("mean", 0.0)

    # mixed-sign + skewed
    if neg > 0.05:
        if abs(skew) > 1.5:
            return ["yeo_johnson", "arcsinh", "none"]
        return ["arcsinh", "yeo_johnson", "none"]

    # all positive, sparse zero
    if zero > 0.3:
        if mean > 1.0:
            return ["anscombe", "freeman_tukey", "log1p", "sqrt", "none"]
        return ["freeman_tukey", "anscombe", "log1p", "sqrt"]

    # all positive, strongly right-skewed
    if skew > 2.5:
        return ["log1p", "sqrt", "yeo_johnson", "none"]
    if skew > 1.0:
        return ["sqrt", "log1p", "yeo_johnson", "none"]

    # near-normal
    return ["none"]   # 정규에 가까움 — log 불필요


def _scale_menu_for_model(model_category: str) -> list[str]:
    """모델 category → scale op 후보 menu."""
    if model_category == "tree":
        return ["robust", "passthrough"]   # split-based, monotonic 무관
    if model_category == "linear":
        return ["robust", "standard", "passthrough"]
    if model_category == "kernel":
        return ["standard", "robust"]   # kernel distance 표준화
    if model_category == "dl":
        return ["standard", "robust"]   # gradient flow
    if model_category == "glm":
        return ["robust", "passthrough"]   # link 내장 — light X scaling
    if model_category == "gam":
        return ["robust", "passthrough"]
    if model_category in ("ts", "mechanistic", "ensemble", "foundation"):
        return ["passthrough"]
    return ["robust", "passthrough"]   # default


def suggest_per_feature_preproc_combo(
    trial,
    col: str,
    stats: dict,
    model_category: str,
) -> tuple[str, str]:
    """각 feature 의 (log_op, scale_op) 조합을 Optuna 가 search.

    Search space 축소 전략:
      · 분포 stats 로 log_op menu 좁힘 (1-5 candidates)
      · model_category 로 scale_op menu 좁힘 (1-3 candidates)
      · Constant/binary/cyclic features: 0 keys (deterministic)
      · 그 외: 1-2 keys/feature

    Returns:
        (log_op_name, scale_op_name)
    """
    log_menu = _log_menu_for_stats(stats)
    scale_menu = _scale_menu_for_model(model_category)

    # log op suggest
    if len(log_menu) == 1:
        log_op = log_menu[0]
    else:
        log_op = trial.suggest_categorical(f"prep_{col}_log", log_menu)

    # scale op suggest (drop 인 경우 skip)
    if log_op == "drop":
        return "drop", "drop"
    if len(scale_menu) == 1:
        scale_op = scale_menu[0]
    else:
        scale_op = trial.suggest_categorical(f"prep_{col}_scale", scale_menu)

    return log_op, scale_op


def _build_combo_op(log_op: str, scale_op: str):
    """(log_op, scale_op) 조합 → sklearn Pipeline / Transformer."""
    from sklearn.pipeline import Pipeline as _Pipeline
    if log_op == "drop":
        return None   # caller drops the column

    steps = []
    # Log/transform step
    if log_op == "log1p":
        steps.append(("log1p", _log1p_t()))
    elif log_op == "sqrt":
        steps.append(("sqrt", _sqrt_t()))
    elif log_op == "anscombe":
        steps.append(("anscombe", _anscombe_t()))
    elif log_op == "freeman_tukey":
        steps.append(("freeman_tukey", _freeman_tukey_t()))
    elif log_op == "arcsinh":
        steps.append(("arcsinh", _arcsinh_t()))
    elif log_op == "arcsine_sqrt":
        steps.append(("arcsine_sqrt", _arcsine_sqrt_t()))
    elif log_op == "yeo_johnson":
        steps.append(("yeo", _yeo_johnson_t()))
    # logit, none → no log step

    # Scale step
    if scale_op == "robust":
        steps.append(("robust", RobustScaler()))
    elif scale_op == "standard":
        steps.append(("standard", StandardScaler()))
    # passthrough, none → no scale step

    if not steps:
        return "passthrough"
    if len(steps) == 1:
        return steps[0][1]
    return _Pipeline(steps)


def build_per_feature_preprocessor_optuna(
    feature_cols: list[str],
    X_train: np.ndarray,
    model_name: str,
    trial,
    return_decisions: bool = False,
) -> object:
    """Optuna 가 각 feature 의 (log_op, scale_op) 조합 search → ColumnTransformer.

    사용자 통찰 반영:
      ① 그룹 X (per-feature)
      ② 1개 X (조합 — log + scale 분리)
      ③ Optuna search (deterministic X)
      ④ 모델별 다름 (model_category)

    Args:
        feature_cols: 피처 이름 list
        X_train: train data (분포 측정)
        model_name: 모델 이름
        trial: Optuna trial
        return_decisions: True 시 (preprocessor, decisions) 반환

    Returns:
        ColumnTransformer (or with decisions tuple)
    """
    model_category = get_model_category(model_name)

    transformers = []
    decisions = {}
    for i, col in enumerate(feature_cols):
        stats = profile_feature(X_train[:, i])
        log_op, scale_op = suggest_per_feature_preproc_combo(
            trial, col, stats, model_category
        )
        decisions[col] = {
            "log_op": log_op,
            "scale_op": scale_op,
            "skew": stats.get("skew"),
            "neg_ratio": stats.get("neg_ratio"),
            "zero_ratio": stats.get("zero_ratio"),
            "is_constant": stats.get("is_constant", False),
        }
        if log_op == "drop":
            continue   # constant feature 자동 제외
        op = _build_combo_op(log_op, scale_op)
        if op is None:
            continue
        transformers.append((f"f{i}", op, [i]))

    if not transformers:
        from sklearn.preprocessing import RobustScaler as _RS
        ct = _RS()
    else:
        ct = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            sparse_threshold=0,
            verbose_feature_names_out=False,
        )

    if return_decisions:
        return ct, decisions
    return ct


__all__ = [
    "profile_feature",
    "get_model_category",
    "recommend_transform",
    "get_y_transform",
    "build_per_feature_preprocessor",
    "build_per_feature_preprocessor_optuna",   # 2026-04-29 신규
    "suggest_per_feature_preproc_combo",        # 2026-04-29 신규
    "summarize_decisions",
]
