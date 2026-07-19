"""Group-aware feature preprocessing (ColumnTransformer).

각 feature 는 분포 특성이 다르다 (cyclic 은 [-1,1], counts 는 sparse skewed,
weather 는 outlier 있음, search trend 는 heavy-tail 등). 단일 글로벌 scaler
(StandardScaler / RobustScaler) 를 모두에 적용하면 정보 손실 + 분포 가정 위반.

이 모듈은 feature 이름 패턴으로 그룹을 분류하고, 그룹별 적절한 preprocessing
을 적용하는 sklearn ColumnTransformer 를 만든다.

[2026-04-28 추가] R9 _evaluate_config 의 글로벌 scaler 대안.
환경변수 ``MPH_GROUPED_PREPROC=1`` 시 활성화.
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
# Feature group classification
# ════════════════════════════════════════════════════════════════
def classify_feature(name: str) -> str:
    """Feature 이름 패턴으로 분포 특성 그룹 분류.

    [2026-04-28 v2] 'other' 117 → ~10 까지 축소.

    Returns one of:
      cyclic, spectral, discrete, lag_ili, rmean, weather, fcst_weather,
      disease_count, mobility, mobility_rt, search_trend, vaccine, binary,
      composite, advanced (new derived), health_resource, claims,
      epi_indicator, other
    """
    cl = name.lower()
    # 1) 가장 좁은 매칭부터 (priority order)
    if any(x in cl for x in ("_imf", "hilbert_amp", "hilbert_phase", "hilbert_freq",
                              "perment", "spec_ent", "fft_slope", "hjorth_",
                              "takens_", "rqa_", "catch22_", "quantum_",
                              "stl_trend", "stl_seasonal", "stl_resid",
                              "savgol_", "hampel_", "sax_", "paa_")):
        return "advanced"
    if "sin_" in cl or "cos_" in cl or "season_" in cl:
        return "cyclic"
    if "wavelet" in cl or "fft" in cl or "spectral" in cl:
        return "spectral"
    if "qbin" in cl or "qnorm" in cl or "_bit" in cl:
        return "discrete"
    # 2) ILI 파생 (rmean before lag_ili — rmean 도 ili_ 시작 가능)
    if "ili_rate_lag" in cl or "ili_lag" in cl or "ili_rate_l" in cl:
        return "lag_ili"
    if "rmean" in cl or "rstd" in cl or "rmin" in cl or "rmax" in cl:
        return "rmean"
    if cl.startswith("ili_") or cl.startswith("ari_"):
        # ili_diff*, ili_log1p, etc. — ILI 파생 (rmean 으로 묶음)
        return "rmean"
    # 3) 기상 (관측 + 예보)
    if any(p in cl for p in ("temp_", "ta_", "humidity", "rainfall", "rn_",
                              "pressure", "wind", "sunshine", "cold_", "humid_")):
        return "weather"
    if cl.startswith("fcst_"):
        return "fcst_weather"
    # 4) 질병 카운트
    if cl.startswith("dis_") or "disease" in cl or cl.startswith("sari_") \
            or cl.startswith("hfmd") or cl.startswith("enterovirus"):
        return "disease_count"
    # 5) 인구이동 — 실시간 (rt_*) vs 통상
    if cl.startswith("rt_"):
        return "mobility_rt"
    if any(p in cl for p in ("pop_", "subway_", "bus_", "commute", "inflow",
                              "hourly_pop", "_metro", "_traffic", "sub_",
                              "dong_", "hotspot_", "emp_")):
        return "mobility"
    # 6) 검색트렌드
    if cl.startswith("gt_"):
        return "search_trend"
    # 7) 백신
    if "vax" in cl or "vacc" in cl:
        return "vaccine"
    # 8) 의료자원 (병상/의사 — 천천히 변화하는 stock)
    if cl.startswith("hospital_"):
        return "health_resource"
    # 9) HIRA claims (청구)
    if cl.startswith("hira_"):
        return "claims"
    # 10) Mortality rate (epi indicator)
    if cl.startswith("mr_"):
        return "epi_indicator"
    # 11) Binary flag (covid_era_indicator 등 *_indicator 도)
    if any(p in cl for p in ("closure", "above_thr", "consec_rise", "school_",
                              "_flag", "is_", "binary", "_indicator", "_era")):
        return "binary"
    # 12) Composite interaction
    if "comp_" in cl or "_x_" in cl:
        return "composite"
    return "other"


# ════════════════════════════════════════════════════════════════
# Per-group preprocessor recipes
# ────────────────────────────────────────────────────────────────
# Sprint 1.5 R7 (2026-05-26): 6 VST helpers 통합 →
#   simulation/models/_vst_primitives.py (single source of truth).
# 이전: 본 파일이 각 def 로컬 정의 (~130줄, anscombe/freeman_tukey 의 자세한 docstring 포함).
# 현재: import alias. caller 9곳 (호출 코드 변경 X, 이름 동일).
# Reference 주석은 _vst_primitives 의 docstring 으로 이동.
# `_arcsinh_transformer` default scale=10.0 — 본 파일의 caller (L307/314/323/329)
# 는 모두 explicit `scale=100.0` / `scale=10.0` 으로 호출하므로 default 변경 영향 X.
# ════════════════════════════════════════════════════════════════
from simulation.models._vst_primitives import (
    _log1p_transformer,
    _sqrt_transformer,
    _yeo_johnson_safe,
    _anscombe_transformer,
    _freeman_tukey_transformer,
    _arcsinh_transformer,
)


def _winsorize_transformer(p_low: float = 0.01, p_high: float = 0.99):
    """Winsorization: clip at quantiles [p_low, p_high].

    NOTE: stateful — saves train-set quantiles. fit_transform on train, transform on test.
    sklearn FunctionTransformer 는 stateless 이므로 별도 클래스 필요.
    """
    from sklearn.base import BaseEstimator, TransformerMixin

    class _Winsorizer(BaseEstimator, TransformerMixin):
        def __init__(self, p_low_=p_low, p_high_=p_high):
            self.p_low_ = p_low_
            self.p_high_ = p_high_

        def fit(self, X, y=None):
            Xa = np.asarray(X, dtype=np.float64)
            self.lo_ = np.nanquantile(Xa, self.p_low_, axis=0)
            self.hi_ = np.nanquantile(Xa, self.p_high_, axis=0)
            return self

        def transform(self, X):
            Xa = np.asarray(X, dtype=np.float64).copy()
            Xa = np.minimum(Xa, self.hi_)
            Xa = np.maximum(Xa, self.lo_)
            return Xa

    return _Winsorizer()


def _logit_transformer(eps: float = 1e-3):
    """Logit transform for proportions ∈ (0, 1).

    log(p / (1-p)). Bounded p in [eps, 1-eps] for numerical safety.
    """
    return FunctionTransformer(
        func=lambda x, e=eps: np.log(
            np.clip(np.asarray(x, dtype=np.float64), e, 1 - e)
            / (1.0 - np.clip(np.asarray(x, dtype=np.float64), e, 1 - e))
        ),
        inverse_func=lambda x: 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64))),
        validate=False,
    )


# Sprint 1.5 R7 (2026-05-26): _arcsine_sqrt_transformer 도 _vst_primitives 로 이전 →
# 이 파일은 stateful (_winsorize_transformer, _logit_transformer, _boxcox_offset_transformer)
# + grouped-specific recipe builder (build_grouped_preprocessor) 만 유지.
from simulation.models._vst_primitives import _arcsine_sqrt_transformer  # noqa: F401


def _boxcox_offset_transformer(offset: float = 0.5):
    """Box-Cox (positive only): (x^λ - 1) / λ. Add offset to handle zeros.

    Auto-selects λ via maximum likelihood (PowerTransformer with method='box-cox').
    NOTE: fails on negatives. Use yeo_johnson for mixed-sign.
    """
    from sklearn.preprocessing import PowerTransformer
    # Box-Cox 는 strictly positive 필요 → wrapper 가 +offset 후 적용
    pt = PowerTransformer(method="box-cox", standardize=True)
    pre = FunctionTransformer(
        func=lambda x, off=offset: np.maximum(np.asarray(x, dtype=np.float64), 0.0) + off,
        inverse_func=lambda x, off=offset: np.maximum(np.asarray(x, dtype=np.float64) - off, 0.0),
        validate=False,
    )
    return Pipeline([("offset", pre), ("boxcox", pt)])


def _make_recipe(group: str) -> object:
    """Group → preprocessor 매핑.

    각 그룹의 분포 특성을 고려한 11가지 변환의 조합:
      • log1p           → 양수 skewed (counts, rates) variance 안정화
      • sqrt            → Poisson-like counts (단순)
      • Anscombe        → 2*sqrt(x + 3/8) — Poisson VST optimal (mean > 1)
      • Freeman-Tukey   → sqrt(x) + sqrt(x+1) — low-mean Poisson VST
      • arcsinh (asinh) → mixed-sign + heavy-tail, λ-free
      • logit / arcsine → proportions ∈ [0, 1]
      • RobustScaler    → outlier 안전 (median/IQR)
      • StandardScaler  → 정규분포 가정 가능 시 (mean/std)
      • yeo_johnson     → 음수 포함 mixed-sign + skewed (auto-fit λ)
      • QuantileTransformer → heavy-tail → normal 강제 매핑
      • Winsorization   → outlier 에 cap [p1, p99]
      • Box-Cox (offset)→ strictly positive, auto-fit λ
      • passthrough     → 이미 적절히 normalized
    """
    if group == "lag_ili":
        # ILI lags = 핵심 AR 신호 (rate, non-neg, skewed).
        # log1p → variance 안정 + RobustScaler (test 분포 shift 안전).
        return Pipeline([
            ("log1p", _log1p_transformer()),
            ("robust", RobustScaler()),
        ])
    if group == "disease_count":
        # 매우 sparse Poisson-like counts (zero 많음).
        # Freeman-Tukey VST 가 low-mean Poisson 에 가장 안정.
        return Pipeline([
            ("freeman_tukey", _freeman_tukey_transformer()),
            ("winsor", _winsorize_transformer(0.01, 0.99)),
            ("robust", RobustScaler()),
        ])
    if group == "mobility":
        # 인구이동: scale 차이 매우 큼 (수십~수천만). log → winsor → standardize.
        return Pipeline([
            ("log1p", _log1p_transformer()),
            ("winsor", _winsorize_transformer(0.01, 0.99)),
            ("standard", StandardScaler()),
        ])
    if group == "mobility_rt":
        # 실시간 mobility (rt_*): 빠른 변동 + outlier 자주 발생.
        # arcsinh 가 large dynamic range + 부호 보존에 적합.
        return Pipeline([
            ("arcsinh", _arcsinh_transformer(scale=100.0)),
            ("winsor", _winsorize_transformer(0.01, 0.99)),
            ("robust", RobustScaler()),
        ])
    if group == "search_trend":
        # Google Trends 0~100, heavy-tail. arcsinh + Quantile-Normal 강력.
        return Pipeline([
            ("arcsinh", _arcsinh_transformer(scale=10.0)),
            ("quantile", QuantileTransformer(
                n_quantiles=100, output_distribution="normal", random_state=42
            )),
        ])
    if group == "weather":
        # 기온 (음수), 강수 (sparse), 습도 ([0,100]) — mixed.
        # arcsinh 가 음수 처리 + λ-free 안정.
        return Pipeline([
            ("arcsinh", _arcsinh_transformer(scale=10.0)),
            ("standard", StandardScaler()),
        ])
    if group == "fcst_weather":
        # KMA 단기예보 — 관측 weather 와 같은 분포.
        return Pipeline([
            ("arcsinh", _arcsinh_transformer(scale=10.0)),
            ("standard", StandardScaler()),
        ])
    if group == "spectral":
        # wavelet/FFT coef: 음수 포함 mixed-sign, heavy-tail.
        # yeo_johnson 으로 auto-λ 정규화 후 RobustScaler.
        return Pipeline([
            ("yeo", _yeo_johnson_safe()),
            ("robust", RobustScaler()),
        ])
    if group == "advanced":
        # entropy/IMF/Hilbert phase ∈ [-π, π], freq, complexity (mixed).
        return Pipeline([
            ("yeo", _yeo_johnson_safe()),
            ("robust", RobustScaler()),
        ])
    if group == "composite":
        # interaction features (temp × ili 등): scale 차이 큼.
        return Pipeline([
            ("winsor", _winsorize_transformer(0.01, 0.99)),
            ("robust", RobustScaler()),
        ])
    if group == "health_resource":
        # 병상/의사 수 — 천천히 변하는 stock, scale 큼.
        return Pipeline([
            ("log1p", _log1p_transformer()),
            ("standard", StandardScaler()),
        ])
    if group == "claims":
        # HIRA 청구 — count 류, sparse zero 가능.
        return Pipeline([
            ("anscombe", _anscombe_transformer()),
            ("robust", RobustScaler()),
        ])
    if group == "epi_indicator":
        # mortality rate (mr_*) — 비율, [0, 1] 가능.
        return Pipeline([
            ("log1p", _log1p_transformer()),
            ("robust", RobustScaler()),
        ])
    if group == "cyclic":
        # sin/cos ∈ [-1, 1] → passthrough.
        return "passthrough"
    if group == "vaccine":
        # 백신 coverage ∈ [0, 1] — proportion. arcsine sqrt VST.
        return _arcsine_sqrt_transformer()
    if group == "binary":
        # 0/1 flag → passthrough.
        return "passthrough"
    if group == "discrete":
        # qbin/qnorm/bit → 이미 normalized → passthrough.
        return "passthrough"
    if group == "rmean":
        # rolling mean of ili_rate (smooth, non-neg).
        return Pipeline([
            ("log1p", _log1p_transformer()),
            ("robust", RobustScaler()),
        ])
    # default fallback — 무엇이든 robust 가 안전한 default
    return RobustScaler()


# ════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════
def build_grouped_preprocessor(
    feature_cols: list[str],
    return_groups: bool = False,
) -> object:
    """feature_cols 를 그룹 분류 → 그룹별 preprocessor → ColumnTransformer.

    Args:
        feature_cols: 피처 이름 리스트 (X_train 의 column 순서).
        return_groups: True 시 (preprocessor, group_summary) 튜플 반환.

    Returns:
        sklearn ColumnTransformer (또는 with summary tuple).
    """
    by_group: dict[str, list[int]] = {}
    for i, c in enumerate(feature_cols):
        g = classify_feature(c)
        by_group.setdefault(g, []).append(i)

    transformers = []
    summary = {}
    # 그룹 우선순위 (특수 → 일반)
    priority = [
        "lag_ili", "cyclic", "weather", "fcst_weather",
        "disease_count", "mobility", "mobility_rt", "search_trend",
        "vaccine", "binary", "spectral", "discrete", "rmean",
        "composite", "advanced", "health_resource", "claims",
        "epi_indicator", "other",
    ]
    for g in priority:
        if g not in by_group:
            continue
        idx = by_group[g]
        recipe = _make_recipe(g)
        transformers.append((g, recipe, idx))
        summary[g] = len(idx)

    ct = ColumnTransformer(
        transformers=transformers,
        remainder="drop",        # 분류 안 된 경우 (있으면 안 됨, "other" 에 fallback)
        sparse_threshold=0,
        verbose_feature_names_out=False,
    )

    if return_groups:
        return ct, summary
    return ct


def summarize_groups(feature_cols: list[str]) -> dict:
    """feature group 분류 결과 dict 반환 (디버그/로깅용)."""
    summary = {}
    for c in feature_cols:
        g = classify_feature(c)
        summary.setdefault(g, []).append(c)
    return summary


# ════════════════════════════════════════════════════════════════
# Optuna 화된 그룹별 전처리 (2026-04-29 추가)
# ────────────────────────────────────────────────────────────────
# 기존 _make_recipe(group): group → fixed Pipeline 1:1
# 신규 _suggest_recipe(group, trial): group → Optuna 가 (log_op, scale_op)
#                                      조합을 trial 마다 search
#
# 활성화: MPH_PREPROC_OPTUNA=1 + R9 가 trial 객체 전달 시
# ════════════════════════════════════════════════════════════════

# 그룹별 후보 전처리 옵션 (각 그룹 안에서 Optuna 가 선택)
_GROUP_OPTIONS = {
    "lag_ili": {
        "log_op":   ["log1p", "sqrt", "none"],
        "scale_op": ["robust", "standard", "none"],
    },
    "rmean": {
        "log_op":   ["log1p", "sqrt", "none"],
        "scale_op": ["robust", "standard", "none"],
    },
    "disease_count": {
        "log_op":   ["freeman_tukey", "anscombe", "log1p", "sqrt"],
        "winsor":   [True, False],
        "scale_op": ["robust", "standard", "none"],
    },
    "mobility": {
        "log_op":   ["log1p", "sqrt", "arcsinh", "none"],
        "winsor":   [True, False],
        "scale_op": ["standard", "robust", "none"],
    },
    "mobility_rt": {
        "log_op":   ["arcsinh", "log1p", "none"],
        "winsor":   [True, False],
        "scale_op": ["robust", "standard", "none"],
    },
    "weather": {
        "log_op":   ["arcsinh", "yeo_johnson", "none"],
        "scale_op": ["standard", "robust", "none"],
    },
    "fcst_weather": {
        "log_op":   ["arcsinh", "yeo_johnson", "none"],
        "scale_op": ["standard", "robust", "none"],
    },
    "search_trend": {
        "log_op":   ["arcsinh", "log1p", "none"],
        "scale_op": ["quantile_normal", "robust", "standard"],
    },
    "vaccine": {
        "log_op":   ["arcsine_sqrt", "logit", "none"],
        "scale_op": ["none", "standard"],
    },
    "claims": {
        "log_op":   ["anscombe", "log1p", "freeman_tukey", "sqrt"],
        "scale_op": ["robust", "standard", "none"],
    },
    "health_resource": {
        "log_op":   ["log1p", "sqrt", "none"],
        "scale_op": ["standard", "robust"],
    },
    "epi_indicator": {
        "log_op":   ["log1p", "logit", "arcsine_sqrt", "none"],
        "scale_op": ["robust", "standard", "none"],
    },
    "spectral": {
        "log_op":   ["yeo_johnson", "arcsinh", "none"],
        "scale_op": ["robust", "standard"],
    },
    "advanced": {
        "log_op":   ["yeo_johnson", "arcsinh", "none"],
        "scale_op": ["robust", "standard"],
    },
    "composite": {
        "winsor":   [True, False],
        "scale_op": ["robust", "standard", "none"],
    },
    "cyclic":   {"scale_op": ["none"]},   # passthrough only
    "vaccine":  {"log_op": ["arcsine_sqrt", "logit", "none"], "scale_op": ["none"]},
    "binary":   {"scale_op": ["none"]},
    "discrete": {"scale_op": ["none"]},
    "other":    {"scale_op": ["robust", "standard"]},
}


def _build_op(name: str):
    """이름 → transformer instance."""
    if name == "log1p":
        return _log1p_transformer()
    if name == "sqrt":
        return _sqrt_transformer()
    if name == "anscombe":
        return _anscombe_transformer()
    if name == "freeman_tukey":
        return _freeman_tukey_transformer()
    if name == "arcsinh":
        return _arcsinh_transformer(scale=10.0)
    if name == "arcsine_sqrt":
        return _arcsine_sqrt_transformer()
    if name == "logit":
        return _logit_transformer()
    if name == "yeo_johnson":
        return _yeo_johnson_safe()
    if name == "robust":
        return RobustScaler()
    if name == "standard":
        return StandardScaler()
    if name == "quantile_normal":
        return QuantileTransformer(n_quantiles=100, output_distribution="normal",
                                     random_state=42)
    return None


def _suggest_recipe(group: str, trial, group_idx: int):
    """그룹별 Optuna trial 에서 (log_op, [winsor], scale_op) suggest → Pipeline.

    group_idx: 같은 trial 안에서 여러 그룹의 prefix 충돌 방지 (각 그룹 unique key).
    """
    options = _GROUP_OPTIONS.get(group, {"scale_op": ["robust", "standard", "none"]})
    pre = group  # prefix
    steps = []

    # 1. Log/transform op
    if "log_op" in options:
        log_choice = trial.suggest_categorical(
            f"prep_{pre}_log", options["log_op"]
        )
        if log_choice != "none":
            op = _build_op(log_choice)
            if op is not None:
                steps.append((f"{log_choice}", op))

    # 2. Winsorization (옵션, 일부 그룹만)
    if "winsor" in options:
        if trial.suggest_categorical(f"prep_{pre}_winsor", options["winsor"]):
            steps.append(("winsor", _winsorize_transformer(0.01, 0.99)))

    # 3. Scale op
    if "scale_op" in options:
        scale_choice = trial.suggest_categorical(
            f"prep_{pre}_scale", options["scale_op"]
        )
        if scale_choice != "none":
            op = _build_op(scale_choice)
            if op is not None:
                steps.append((f"{scale_choice}", op))

    if not steps:
        return "passthrough"
    if len(steps) == 1:
        return steps[0][1]
    return Pipeline(steps)


def build_grouped_preprocessor_optuna(
    feature_cols: list[str],
    trial,
) -> object:
    """Optuna trial 에서 그룹별 preprocessing suggest → ColumnTransformer.

    각 그룹의 (log_op, [winsor], scale_op) 를 Optuna 가 trial 마다 search.
    R9 _evaluate_config 가 MPH_PREPROC_OPTUNA=1 시 사용.

    Args:
        feature_cols: 피처 이름 리스트
        trial: Optuna trial 객체

    Returns:
        ColumnTransformer (각 그룹마다 trial-suggested preprocessing)
    """
    by_group: dict[str, list[int]] = {}
    for i, c in enumerate(feature_cols):
        g = classify_feature(c)
        by_group.setdefault(g, []).append(i)

    transformers = []
    priority = [
        "lag_ili", "cyclic", "weather", "fcst_weather",
        "disease_count", "mobility", "mobility_rt", "search_trend",
        "vaccine", "binary", "spectral", "discrete", "rmean",
        "composite", "advanced", "health_resource", "claims",
        "epi_indicator", "other",
    ]
    for gi, g in enumerate(priority):
        if g not in by_group:
            continue
        idx = by_group[g]
        recipe = _suggest_recipe(g, trial, gi)
        transformers.append((g, recipe, idx))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0,
        verbose_feature_names_out=False,
    )


__all__ = [
    "classify_feature",
    "build_grouped_preprocessor",
    "build_grouped_preprocessor_optuna",   # 2026-04-29 신규
    "summarize_groups",
]
