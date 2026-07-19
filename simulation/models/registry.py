"""
simulation.models.registry
===========================
PAPER_PRIMARY_11 freeze management + live registry coverage verification.

Responsibilities:
 1. Verify the live REGISTRY coverage against CATEGORY_MODELS (the SSOT)
 via `verify_registry_coverage()` — source of truth for active/registered
 counts (no hardcoded catalog; counts are measured at import time).
 2. Mark PAPER_PRIMARY models (publication backbone) and freeze their
 source-file hashes so later commits cannot silently change which models
 produced the published numbers. (Currently empty — refrozen post-training,
 registry.py PAPER_PRIMARY_11 declaration.)
 3. Persist everything in the `model_registry` table (defined in
 schema.py) with a `frozen_at` timestamp on PAPER_PRIMARY rows.

This module is INFORMATIONAL + INTEGRITY layer. It does not instantiate
models — that remains `simulation.models.base.ModelRegistry`.

refresh (2026-04-19):
 The prior list had 5/11 stale entries ("Ridge", "Chronos", "SEIR-V-D",
 "PINN-SEIR", "TournamentEnsemble") that never reached REGISTRY —
 re-verified the full 63-model registry (after runner._import_all_models)
 and rebuilt the paper-primary set from names that actually resolve.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# PAPER_PRIMARY_11 declaration (§5.2, RECOMMENDED_PIPELINE.md )
# ══════════════════════════════════════════════════════════════════════════
#: 11 models that appear in the paper's main results tables.
#: Each entry: (model_name, source_file_relative_to_simulation_root).
#:
#: Every name here has been cross-checked against the live REGISTRY
#: populated by runner.MultiModelRunner._import_all_models(). Names that
#: resolve to REGISTRY entries whose meta.category / source_file match the
#: declared tuple are the only legal paper-primary candidates.
#:
#: TabularDNN → NEGATIVE_CONTROL (bench R²=0.7185 < TinyMLP 0.8166).
#: replaced the five stale names (Ridge/Chronos/SEIR-V-D/PINN-SEIR/
#: TournamentEnsemble) with shipped equivalents (ElasticNet / Chronos-2 /
#: NegBinGLM + BayesianMCMC + PINN-Lite / Ensemble-Stacking).
# 2026-05-12 (사용자 명시): "그냥 없애. paper primary도 다시 만들어야해" +
#   "지금 제외하고 다 학습하고 나서 만들꺼야"
# → 빈 tuple. 학습 후 사용자가 새 PaperPRIMARY 정의.
# 이전 11개 중 PINN-Lite (mech 제거됨), TabularDNN-Lite (dl-tabular 단순화로
# 제거됨) 가 invalid → 전체 재구성 필요.
PAPER_PRIMARY_11: tuple[tuple[str, str], ...] = ()

#: — 정책 변경 (2026-04-26 사용자 지적):
#: 이전엔 TabularDNN 을 "over-param negative-control" 로 격하시켰으나,
#: DNN 계열은 forecasting 에서 필수 모델군이고 (TFT/PatchTST 가 paper-11 에
#: 포함되어 있는데 원본 TabularDNN 만 격하시킨 건 비일관). over-param 은
#: regularization / dropout / weight-decay 로 해결할 hyperparameter 문제이지
#: 영구 격하 사유가 아니다.
#:
#: 따라서 NEGATIVE_CONTROL 을 빈 set 으로 만들어 모든 모델이 동등하게
#: ensemble 후보 + R9 grid search 에 들어가게 한다. TabularDNN 은
#: EXTRA_MODELS["DL_variants"] 에 그대로 등록되어 정식 extra 로 평가됨.
NEGATIVE_CONTROL: frozenset[str] = frozenset()  # 빈 set — 격하 모델 없음


# ══════════════════════════════════════════════════════════════════════════
# EXTRA_MODELS — paper-11 외 의 모든 평가 가능한 모델 (post_E_eval 의 55개)
# ══════════════════════════════════════════════════════════════════════════
#:
#: post_E_eval 이 평가한 66 모델 중 paper-11 을 뺀 55 개. 카테고리별로
#: 정리해서 ranking / visualize 에서 명시적 라벨 부여 가능.
#:
#: 사용처:
#:   - `tier_of(model_name)` → "paper" / "extra" / "unknown"
#:   - champion_log.json 의 각 entry 메타에 tier 자동 라벨링
#:   - visualize 의 ranking 표에 tier column 추가 (paper-11 + extras 분리)
#:   - R9 SLOW_MODELS 가 paper-11 을 우회하지 않도록 차단
EXTRA_MODELS: dict[str, list[str]] = {
    # ─────────────────────────────────────────────────────────────────────
    # 2026-05-26 Sprint D (사용자 명시 "MERGE-drop 모델 다 없애버려"): MERGE-drop
    # 5 모델 (GradientBoosting / Ensemble-SelectiveBMA / Ensemble-Temporal /
    # Renewal / Ensemble-Blending) registry trace 완전 제거.
    # 이전 REMOVED 10 (deregister target; OverseasTransfer 는 유지 — active+phase18, 2026-06-01):
    #   GE-DNN, MP-PINN, PINN-Lite, SEIR-V2-Forced, Phase-Adaptive,
    #   FluSight-Ensemble, Ensemble-Stacking, Ensemble-TopTierStacking,
    #   Ensemble-Blending, TinyMLP
    # ─────────────────────────────────────────────────────────────────────
    "DL_variants":       [
        "DNN", "DNN-Optuna", "DNN-Conformal",
        "TabularDNN",  # over-param negative control (66model idx 29)
        "TimesNet", "Mamba", "iTransformer",
        "TiDE", "N-BEATS", "N-HiTS",
        "RNN", "DeepAR",
        "TFT",
        "TCN", "TCN-Optuna",
        "GAT", "GCN",
    ],
    "ensemble_variants": [
        # Active in CATEGORY_MODELS["ensemble"]:
        "Ensemble-NNLS", "Ensemble-NNLS-Filtered",
        "Ensemble-BMA", "Ensemble-InvRMSE", "Ensemble-Diversity",
        "Ensemble-Adaptive", "Ensemble-ResidualAR",
    ],
    "cqr_variants": [
        # G-262 (2026-06-13): CQR-GBR 는 active 에서 감축(DEFER) — CQR-LightGBM 과 GBM 중복.
        # 여기(EXTRA_MODELS = tier 라벨)에는 남겨 "extra" tier 로 정확히 라벨링.
        "CQR-LightGBM", "CQR-GBR", "CQR-QuantReg",
    ],
    "epi_variants": [
        "BayesianRidge", "GAM-Spline",
        "NegBinGLM-V7", "PoissonAutoreg",
    ],
    "foundation_variants": [
        # G-261 (2026-06-13): Chronos 전 변형(Chronos-2-FT/-FT-Real/-MultiCountry) 완전 retire —
        #   transformers<5 ⊥ mlx-lm(ARIA). active foundation = TimesFM-2.5 + TiRex + OverseasTransfer
        #   (CATEGORY_MODELS["foundation"]). 등록되는 extra 변형 없음.
    ],
    "tree_variants": [
        "LightGBM", "RandomForest", "CatBoost",
    ],
    "linear_variants": [
        "KRR", "SVR-Linear", "SVR-RBF",
    ],
    "ts_variants": [
        "SARIMAX", "Theta",  # Theta added Sprint S3 2026-05-26
    ],
    "baselines": [
        "ar1", "persistence", "climatology",
    ],
}

#: Flat helper — set of every name in EXTRA_MODELS
EXTRA_MODELS_FLAT: frozenset[str] = frozenset(
    nm for grp in EXTRA_MODELS.values() for nm in grp
)

#: Combined: paper + extras (전체 등록 후보 ranking 에 사용)
ALL_TIERED_MODELS: dict[str, list[str]] = {
    "paper_primary": [nm for nm, _ in PAPER_PRIMARY_11],
    **EXTRA_MODELS,
}


def tier_of(model_name: str) -> str:
    """Classify a model name into one of 4 tiers.

    Returns:
        "paper"     — model is in PAPER_PRIMARY_11 (publication backbone)
        "extra"     — model is in EXTRA_MODELS (ablation / variant)
        "negative"  — model is in NEGATIVE_CONTROL (excluded from ensembles)
        "unknown"   — name not registered (custom user model, typo, …)
    """
    paper_names = {nm for nm, _ in PAPER_PRIMARY_11}
    if model_name in paper_names:
        return "paper"
    if model_name in NEGATIVE_CONTROL:
        # NEGATIVE_CONTROL takes precedence for the (rare) overlap
        # (TabularDNN appears in EXTRA_MODELS["DL_variants"] but is also
        # tagged negative — semantics: "extra-negative", reported but never
        # in operational ensembles).
        return "negative"
    if model_name in EXTRA_MODELS_FLAT:
        return "extra"
    return "unknown"


def category_of(model_name: str) -> str:
    """Return the EXTRA_MODELS sub-category, or 'paper' for paper-primary."""
    if model_name in {nm for nm, _ in PAPER_PRIMARY_11}:
        return "paper"
    for cat, names in EXTRA_MODELS.items():
        if model_name in names:
            return cat
    return "unknown"


def list_by_tier(tier: str) -> list[str]:
    """Return all model names of a given tier.

    Args:
      tier ∈ {"paper", "extra", "negative", "all"}
    """
    if tier == "paper":
        return [nm for nm, _ in PAPER_PRIMARY_11]
    if tier == "extra":
        return sorted(EXTRA_MODELS_FLAT)
    if tier == "negative":
        return sorted(NEGATIVE_CONTROL)
    if tier == "all":
        return ([nm for nm, _ in PAPER_PRIMARY_11] +
                sorted(EXTRA_MODELS_FLAT))
    return []

# (2026-05-29 retired) DL_TIER_A / REGISTRY_CATALOG / REGISTRY_43_CATEGORIES 제거.
#   DL_TIER_A: 비기능 — stale test 만 참조 (deregistered TinyMLP 잔재 포함).
#   REGISTRY_CATALOG/REGISTRY_43_CATEGORIES: 하드코딩 count snapshot (stale, 합 68 ≠ live).
#   live count 의 단일 출처 = `verify_registry_coverage()` (force_import 후 REGISTRY 실측).


# ══════════════════════════════════════════════════════════════════════════
# Registry snapshot
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class ModelSnapshot:
    model_name: str
    category: str
    level: int
    min_data: int
    is_paper_primary: bool
    is_registered: bool
    requires_gpu: bool
    source_file: Optional[str]
    source_sha256: Optional[str]
    description: str
    registered_at: str
    frozen_at: Optional[str] = None

    def to_row(self) -> dict:
        return {
            "model_name": self.model_name,
            "category": self.category,
            "level": self.level,
            "min_data": self.min_data,
            "is_paper_primary": 1 if self.is_paper_primary else 0,
            "is_registered": 1 if self.is_registered else 0,
            "requires_gpu": 1 if self.requires_gpu else 0,
            "source_file": self.source_file,
            "source_sha256": self.source_sha256,
            "registered_at": self.registered_at,
            "frozen_at": self.frozen_at,
            "description": self.description,
        }


# ══════════════════════════════════════════════════════════════════════════
# Hash utilities
# ══════════════════════════════════════════════════════════════════════════
def sha256_file(path: Path | str) -> Optional[str]:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _simulation_root() -> Path:
    """Return the path to simulation/ directory."""
    return Path(__file__).resolve().parent.parent


# ══════════════════════════════════════════════════════════════════════════
# Snapshot builder
# ══════════════════════════════════════════════════════════════════════════
def build_snapshot(
    *,
    mark_paper_primary: bool = True,
    freeze_paper_primary: bool = False,
) -> list[ModelSnapshot]:
    """Walk the in-process ModelRegistry and build a snapshot.

    Parameters
    ----------
    mark_paper_primary : bool
        If True, set is_paper_primary=1 for PAPER_PRIMARY_11 names.
    freeze_paper_primary : bool
        If True, set frozen_at=now() for PAPER_PRIMARY_11 names.
        Only use this when you're committing to the paper's model set.
    """
    from simulation.models.base import REGISTRY

    now_iso = datetime.now().isoformat()
    paper_names = {name for name, _ in PAPER_PRIMARY_11}
    paper_sources = {name: src for name, src in PAPER_PRIMARY_11}

    snapshots: list[ModelSnapshot] = []
    for name, cls in REGISTRY.get_all().items():
        meta = cls.meta
        is_paper = mark_paper_primary and (name in paper_names)
        source_rel = paper_sources.get(name)
        source_path = (_simulation_root() / source_rel) if source_rel else None
        sha = sha256_file(source_path) if source_path else None

        snapshots.append(ModelSnapshot(
            model_name=name,
            category=meta.category,
            level=meta.level,
            min_data=meta.min_data,
            is_paper_primary=is_paper,
            is_registered=True,
            requires_gpu=meta.requires_gpu,
            source_file=str(source_rel) if source_rel else None,
            source_sha256=sha,
            description=meta.description,
            registered_at=now_iso,
            frozen_at=now_iso if (is_paper and freeze_paper_primary) else None,
        ))

    # Also record PAPER_PRIMARY members that aren't yet registered
    registered_names = {s.model_name for s in snapshots}
    for name, src in PAPER_PRIMARY_11:
        if name in registered_names:
            continue
        source_path = _simulation_root() / src
        snapshots.append(ModelSnapshot(
            model_name=name,
            category="pending",
            level=99,
            min_data=0,
            is_paper_primary=True,
            is_registered=False,
            requires_gpu=False,
            source_file=src,
            source_sha256=sha256_file(source_path),
            description="PAPER_PRIMARY placeholder (not yet registered in REGISTRY)",
            registered_at=now_iso,
            frozen_at=now_iso if freeze_paper_primary else None,
        ))

    return snapshots


# ══════════════════════════════════════════════════════════════════════════
# Persistence
# ══════════════════════════════════════════════════════════════════════════
def persist_snapshot(
    snapshots: list[ModelSnapshot],
    *,
    replace: bool = True,
) -> int:
    """Write snapshots into model_registry table. Returns rows inserted."""
    try:
        from simulation.database import insert_rows
    except ImportError as e:
        log.error("DB unavailable: %s", e)
        return 0

    rows = [s.to_row() for s in snapshots]
    mode = "REPLACE" if replace else "IGNORE"
    return insert_rows("model_registry", rows, on_conflict=mode)


def load_snapshots() -> list[dict]:
    """Read all model_registry rows."""
    try:
        from simulation.database import query
    except ImportError:
        return []
    return query(
        "SELECT * FROM model_registry ORDER BY is_paper_primary DESC, "
        "category, level"
    )


def verify_paper_primary_frozen() -> dict:
    """Check whether all PAPER_PRIMARY_11 source files still match their
    frozen SHA-256 hashes. Returns diff report."""
    from simulation.database import query
    rows = query(
        "SELECT model_name, source_file, source_sha256, frozen_at "
        "FROM model_registry WHERE is_paper_primary=1 AND frozen_at IS NOT NULL"
    )
    if not rows:
        return {"ok": True, "message": "no frozen primaries"}

    sim_root = _simulation_root()
    mismatches: list[dict] = []
    for r in rows:
        src = r["source_file"]
        if not src:
            continue
        current = sha256_file(sim_root / src)
        if current != r["source_sha256"]:
            mismatches.append({
                "model_name": r["model_name"],
                "source_file": src,
                "frozen_sha": r["source_sha256"],
                "current_sha": current,
                "frozen_at": r["frozen_at"],
            })
    return {
        "ok": not mismatches,
        "n_frozen": len(rows),
        "mismatches": mismatches,
    }


# ══════════════════════════════════════════════════════════════════════════
# G-161 (2026-05-02): REGISTRY coverage check — CatBoost 같은 누락 영구 차단
# ══════════════════════════════════════════════════════════════════════════
# 사용자 우려 (2026-05-02): "REGISTRY 없거나 실패하거나 모델이 안 되는지"
# train_by_category.sh 의 카테고리 list 와 REGISTRY 의 등록된 모델이 어긋나면
# 학습 시 silent 누락 (Cat 1 의 4/5 결과 처럼). preflight 로 자동 차단.

# train_by_category.sh:get_models() 의 카테고리 list (source-of-truth)
# G-169 (2026-05-03): Bayesian-SEIR / Metapop-SEIR 는 forecasting 격리 (R²=-8),
# 시뮬레이션 전용 — `bayesian_seir.py:743` + `metapop_seir.py:624` 의 REGISTRY.register
# 의도적 주석. 카테고리 list 에서도 제거 → "missing" false-positive 차단.
CATEGORY_MODELS: dict[str, list[str]] = {
    # ─────────────────────────────────────────────────────────────────────
    # 2026-05-26 Sprint: 78 → 50 active prune (Codex + Gemini + user decisions).
    #   실제 착지: user KEEP override 후 **53 active** (OverseasTransfer 포함 — 사용자 확정 2026-06-01;
    #   prune 의 REMOVE 11 중 OverseasTransfer 는 결국 유지 → 실제 REMOVE 10).
    #   ★ 활성 수의 단일 SSOT = verify_registry_coverage() (아래 주석 숫자는 참고용, 하드코딩 신뢰 X).
    #
    # ── 모델 그룹핑 세 종류 (혼동 방지 — 같은 모델이 grouping마다 다른 라벨) ──────────
    #   (a) CATEGORY_MODELS 의 12 family = train-by-category + 논문 reporting (가장 fine; 이 dict).
    #   (b) _scenarios.ALL_MODELS 의 7 coarse(DL/ML/STAT/EPI/FOUNDATION/ENSEMBLE/CQR) = SCENARIOS preset.
    #   (c) 각 모델 meta.category(7: tree/linear/dl/epi/ts/meta/physics) = 등록·내부 로직(gate).
    #       예: feature-guard resolve_feature_path 가 meta.category=='dl' 판정 → gate 는 (c) 만 신뢰.
    #   셋은 목적이 달라 라벨이 어긋남(예: KRR = (a)"kernel" / (b)"ML" / (c)"linear"). 정상 — 통일 대상 아님.
    #   ★ 단 셋 다 동일 모델 집합(현재 53, OverseasTransfer 포함) — 합은 항상 일치해야 함.
    #
    # α-blend cleanup (user explicit):
    #   - dl_anchored.py / post_anchor.py / apply_anchor_fallback.py / package_c/
    #     모두 simulation/_archive/anchor_deprecated_20260526/ 로 archive (옛 파일/경로명)
    #   - MPH_PI_AUGMENT_LO/HI env vars 영구 비활성 (augment_factor=0 hardcoded)
    #   - 4 α-blend DNN result JSONs archive
    #
    # REMOVE 10 (deregister + class archive elsewhere):
    #   GE-DNN, MP-PINN, PINN-Lite, SEIR-V2-Forced, Phase-Adaptive,
    #   FluSight-Ensemble, Ensemble-Stacking, Ensemble-TopTierStacking,
    #   Ensemble-Blending, TinyMLP
    #   (OverseasTransfer 는 REMOVE 취소 — active foundation 유지 + phase18 사용, 사용자 확정 2026-06-01)
    #
    # MERGE → drop from CATEGORY_MODELS 5 (class kept registered for audit trail):
    #   Ensemble-SelectiveBMA (→ Ensemble-BMA threshold config),
    #   Ensemble-Temporal (→ Ensemble-InvRMSE, R²=0.870 cluster identical),
    #   GradientBoosting (→ XGBoost optimized superset),
    #   Renewal (→ EpiEstim, Rt family Cori 2013 more rigorous),
    #   TabularDNN-Lite (→ TabularDNN same architecture parsimony variant)
    #
    # DEFER (2026-05-26 snapshot, moved to DEFER_MODELS — explicit flag only):
    #   BayesianMCMC, Chronos-2-FT-Real, Chronos-MultiCountry,
    #   CoxPH, DeepAR, GP-RBF-Periodic, PROPHET, RNN, Rt-Augmented, TFT, TSIR
    #   (Chronos-2-FT 제외 — 2026-05-26 active foundation 승격 "상급" zero-shot→few-shot fine-tuned.)
    #   ※ 2026-06-13 (G-261): Chronos-2-FT-Real / Chronos-MultiCountry 는 DEFER 에서도 제거
    #     (Chronos 전 변형 retire — transformers<5 ⊥ mlx-lm). 현 DEFER_MODELS 는 8 개.
    #
    # KEEP for comparison (user override of Codex MERGE):
    #   DNN, TCN, CQR-GBR, NegBinGLM, EARS-C1, EARS-C2, GAT, GCN,
    #   CQR-QuantReg (vs CQR-LightGBM base-model comparison),
    #   N-BEATS (vs N-HiTS Oreshkin 2020 vs Challu 2023 architectures),
    #   NegBinGLM-V7 (F10 documented ablation case)
    # ─────────────────────────────────────────────────────────────────────
    # G-272 (2026-06-14, 사용자): CatBoost 제외(active 54→53) — 핵심장점(네이티브 범주형 처리)이
    #   이 프로젝트(전 feature 수치형 lag/계절성)서 무용 + XGBoost/LightGBM/RandomForest 와 중복(GBM/tree 공간)
    #   + 최느림(full 398 feat·mc=none, 3.5h vs ~2h) + baseline R²=0.683 중위권(RF 0.775·XGB 0.720 하위).
    #   클래스는 등록 유지(DEFER_MODELS) — --include-only 로 재활성 가능.
    "tree": ["XGBoost", "LightGBM", "RandomForest"],
    # G-263 (2026-06-13): NegBinGLM-Glum 추가 — glum elastic-net 진짜 NB-GLM(full-pool L1/L2 shrinkage).
    #   V7(hard top-K)·NegBinGLM(V6 RidgeCV salvage) 와 다른 접근. 실측 ILI test r2=0.878 + peak 도달
    #   (소표본 SOTA 서베이서 유일하게 incumbent 능가한 후보 — TimeMixer/CQR-CatBoost/ExtraTrees 전부 패배).
    # G-319f (2026-06-19): NegBinGLM-V7 DEFER — statsmodels true-NB 가 ILI-rate·소표본(p=309,n=234)서
    #   train R²=−0.16 미수렴 → 항상 V6 salvage fallback(=NegBinGLM 과 byte-identical, MD5 확인). 진짜
    #   NB 역할은 NegBinGLM-Glum(G-319c threadpool_limits 로 회복, test R²=0.882)이 담당 → V7=중복.
    # G-347 (2026-06-25, 사용자): NegBinGLM-Glum DEFER — NegBinGLM(V6, test R²0.904)의 열등 중복
    #   (Glum R²0.800<0.904, OOF 3.25>1.62). NB 해석 역할은 NegBinGLM 이 담당.
    "linear": ["ElasticNet", "BayesianRidge", "NegBinGLM", "PoissonAutoreg"],
    "kernel": ["KRR", "SVR-Linear", "SVR-RBF"],
    "other": ["GAM-Spline", "BayesianMCMC"],  # GP-RBF-Periodic → DEFER (high cost); BayesianMCMC 승격 2026-06-11 (baseline r2=0.746 #16/66 — deferred 중 유일 경쟁력, high-cost MCMC 감수)
    # G-323 (2026-06-19, 사용자): EARS-C1/C2/C3 DEFER — aberration detector(상수 임계 출력)지
    #   point-forecaster 아님 → baseline R² 구조적 음수(고칠 수 없음). 탐지 역할은 ABM/eval 에 유지.
    # G-347 (2026-06-25, 사용자): GLARMA DEFER — test R²=−1.20 음수(구조약체). 관측-구동 GLARMA(Davis 2003)는
    #   소표본 ILI 외삽서 발산. count/renewal 역할은 EpiEstim/hhh4/Wallinga 가 담당.
    "epi-extended": ["EpiEstim", "hhh4-equivalent", "Wallinga-Teunis"],
    # FluSight-Baseline (Round 4 audit G1, 2026-05-27) — Mathis et al. 2024 (Nat Commun
    # 15:6289, PMID 39060259) `simplets::quantile_baseline` Python port. Required as the
    # *comparator baseline* for relative WIS metric (paper Methods).  Added pre-launch
    # to avoid 60-100h ensemble cascade re-training.  See simulation/models/flusight_baseline.py.
    "ts": ["ARIMA", "SARIMA", "SARIMAX", "Theta", "FluSight-Baseline"],  # Theta = M3 baseline (S3 2026-05-26); FluSight = G1 (R4 2026-05-27)
    # G-262 (2026-06-13, 사용자 확정): DNN-Optuna 감축 — phase-13 이 모든 모델을 Optuna 튜닝하므로
    #   "DNN + 자체 nested Optuna" 는 phase-13-튜닝 DNN 과 중첩 중복 (신경망이라 병목). DNN(base)·
    #   DNN-Conformal(PI)·TabularDNN(attention arch) 은 진짜 구분 → 유지. DNN-Optuna 는 등록 유지(DEFER).
    # G-264 (2026-06-13, 사용자 확정): TabPFN 추가 — TabPFN v2 tabular foundation(in-context, Nature 2025).
    #   소표본 SOTA: ILI hold-out r2=0.917(최우수)·WF-CV 0.814 > incumbent. 공개 가중치+model_path
    #   (priorlabs-1-1 학술 무료). glum 과 함께 SOTA 서베이서 incumbent 능가한 2번째 add.
    # 2026-06-23 (사용자 "포함 — 51로 재시작"): SeirCount-TabPFN 활성 승격 —
    #   TabPFN(in-context) + NegBin count-native head. meta.category=dl → dl-tabular(TabPFN 계열).
    # G-347 (2026-06-25, 사용자): TabularDNN DEFER — test R²=−0.80 음수(구조약체, nf=12 floor). DNN/DNN-Conformal
    #   유지(구분되는 deep). (N-HiTS/N-BEATS/TiDE/GCN/GAT 는 비교분석 목적 유지 — 사용자 명시.)
    "dl-tabular": ["DNN", "DNN-Conformal", "TabPFN", "SeirCount-TabPFN"],
    # modern-ts: removed N-BEATS-only superset (N-HiTS kept), TFT defer.
    # G-262: TCN-Optuna 감축 (DNN-Optuna 와 동일 — phase-13 이 TCN 을 튜닝하므로 nested 중복, 신경망 병목).
    #   TCN(base) 유지, TCN-Optuna 는 등록 유지(DEFER).
    # N-BEATS kept (user override — vs N-HiTS comparison).
    # G-265 (2026-06-13, 웹 SOTA 감사 후 사용자 확정): DLinear(Zeng AAAI 2023) 추가 — 분해+단일선형층,
    #   소표본 강건 'simple beats complex' 대표 baseline (우리 결론 직접 입증, 과적합 위험 ≈0).
    "modern-ts": ["PatchTST", "iTransformer", "Mamba", "TimesNet",
                  "N-BEATS", "N-HiTS", "TiDE",
                  "TCN", "DLinear"],
    # G-262: CQR-GBR 감축 — sklearn GBR quantile = CQR-LightGBM(LightGBM quantile) 과 같은 GBM 계열 중복.
    #   CQR-LightGBM(GBM) + CQR-QuantReg(선형) 으로 method 다양성 유지, CQR-GBR 은 등록 유지(DEFER).
    "cqr": ["CQR-LightGBM", "CQR-QuantReg"],
    # graph: GE-DNN REMOVED (user explicit — not in paper). GAT + GCN 둘 다 (2026-06-03 재포함):
    #   GAT 느림 원인 = per-sample Python 루프(attention 아님!) → 배치그래프 1-call 로 고침
    #   (0eb851d, ~4-5× ↑·1.5GB·valid). GCN = BatchNorm 3D 버그 수정(4664db3). 둘 다 정상.
    "graph": ["GAT", "GCN"],
    # foundation: TimesFM-2.5 (Google, zero-shot) — Chronos-2/Chronos-2-FT 의 **대체** (G-261,
    #   사용자 확정 2026-06-13). chronos 는 모든 버전이 transformers<5 강제 → 메인 env(mlx-lm 가
    #   transformers>=5 요구, ARIA) 와 HARD 충돌, 작동 불가. TimesFM 2.5 는 transformers 의존이 없어
    #   메인 env 네이티브 + 격리 불필요. 실측 우위(rolling 0.939>0.927, 68-step −0.885>−0.932).
    #   chronos 전 변형은 registry 등록·wrapper 파일·force_import 모두 제거 (G-261 완료, 2026-06-13).
    #   필요 시 .venv_chronos standalone 으로만 사용 — 메인 코드베이스에서는 완전 retire.
    # OverseasTransfer 유지 (사용자 확정 2026-06-01) — active 패널 + phase 18(overseas 별도 CLI) 둘 다 사용.
    # G-265 (2026-06-13, 웹 SOTA 감사 후 사용자 확정): TiRex(NX-AI xLSTM 35M, 2025) 추가 — zero-shot
    #   foundation, transformers-free(Chronos 충돌 회피). ILI rolling r2=0.944(전 모델 최고)>TimesFM 0.939.
    # 2026-06-23 (사용자 "포함 — 51로 재시작"): FusedEpi 활성 승격 —
    #   TiRex+TabPFN+NegBin+conformal(PID) 융합, meta.category=foundation, BASELINE_ROLLING.
    "foundation": ["TimesFM-2.5", "OverseasTransfer", "TiRex", "FusedEpi"],
    # ensemble: 6 kept (REMOVE Blending, MERGE SelectiveBMA→BMA + Temporal→InvRMSE).
    # G-359 (2026-06-25, 사용자): 약체 ensemble 3 deprecate (rel-WIS>0.8, FusedEpi 0.455 대비 열세) —
    #   Adaptive 1.434(FluSight baseline 보다 나쁨)·ResidualAR 0.869·BMA 0.832. 유지 4개는 모두 <0.72.
    #   ensemble 은 base 집계 meta(oof=inf 설계상) — "결합해도 최고 단일 못 이김" 비교군으로 4개 유지.
    "ensemble": ["Ensemble-NNLS", "Ensemble-NNLS-Filtered",
                 "Ensemble-InvRMSE", "Ensemble-Diversity"],
}

# ─────────────────────────────────────────────────────────────────────────
# DEFER_MODELS — 2026-05-26 (Codex + user prune)
# Registered models excluded from default training due to high compute cost
# or low marginal value. Re-enabled via explicit `--include-only` flag.
# ─────────────────────────────────────────────────────────────────────────
DEFER_MODELS: list[str] = [
    # 2026-05-26 user override: Chronos-2-FT 제거 (CATEGORY foundation 으로 이동).
    # 2026-06-11: BayesianMCMC 제거 — active 승격(baseline r2=0.746 #16/66, 사용자 결정).
    # 2026-06-13 (G-261): Chronos 전 변형 retire — transformers<5 ⊥ mlx-lm(ARIA), 작동 불가.
    #   wrapper 파일·등록·force_import 전부 제거 (TimesFM-2.5 + TiRex 가 대체).
    # 2026-06-14 (G-272): CatBoost 제외 — 범주형 장점 무용(전 feature 수치형)+GBM 중복+최느림+중위권.
    "CatBoost",              # G-272: categorical 장점 무용 + XGB/LGB/RF 중복 + 3.5h 최느림
    "CoxPH",                 # event/hazard, not core point forecast
    "DeepAR",                # no current result under non-pf name
    "GP-RBF-Periodic",       # high-cost GP, lower marginal vs GAM/NegBinGLM
    "PROPHET",               # heavy dep, lower marginal vs GAM/SARIMAX
    "RNN",                   # generic recurrent, no current result
    "Rt-Augmented",          # R²=0.65 self-limited
    "TFT",                   # very high Lightning cost
    "TSIR",                  # susceptible reconstruction, not central to ILI
    # 2026-06-15 (per-model 감사 ww3jbm73o): orphan 4 — registered 인데 active CATEGORY 도
    #   DEFER 도 아니어서 manifest 무결성 위반(registered 66 = active 53 + DEFER 9 + orphan 4).
    #   G-261/262 감축(TimesFM 대체 시 중복 제거)의 audit-trail. DEFER 추가는 학습 behavior
    #   영향 0 — 학습 제외 게이트는 CATEGORY_MODELS 멤버십이지 DEFER 가 아님.
    "CQR-GBR",               # G-262: CQR-LightGBM(GBM) 과 GBM 계열 중복
    # G-319f (2026-06-19): NegBinGLM-V7 — true-NB 데이터한계 미수렴 → V6 fallback(중복). Glum 이 진짜 NB.
    "NegBinGLM-V7",
    "DNN-Optuna",            # G-262: DNN 과 중복(Optuna HP 변형)
    "TCN-Optuna",            # G-262: TCN 과 중복
    "TabularDNN-Lite",       # TabularDNN 과 중복(축소판)
    # G-347 (2026-06-25, 사용자): active 라인업서 deprecate (51→48). 클래스 등록 유지(audit/--include-only 재활성).
    "NegBinGLM-Glum",        # G-347: NegBinGLM(V6) 열등 중복 (R²0.800<0.904)
    "GLARMA",                # G-347: test R²=−1.20 음수 구조약체
    "TabularDNN",            # G-347: test R²=−0.80 음수 구조약체 (nf=12 floor)
    # G-359 (2026-06-25, 사용자): 약체 ensemble 3 deprecate (48→45). 클래스 등록 유지(비교군 재활성 가능).
    "Ensemble-Adaptive",     # G-359: rel-WIS 1.434 (FluSight baseline 보다 나쁨)
    "Ensemble-BMA",          # G-359: rel-WIS 0.832 (FusedEpi 0.455 대비 열세)
    "Ensemble-ResidualAR",   # G-359: rel-WIS 0.869 (열세)
]


def verify_registry_coverage(
    *, force_import: bool = True
) -> dict:
    """`CATEGORY_MODELS` (train_by_category.sh source-of-truth) vs REGISTRY 동기화 검증 (G-161, D-4).

    누락 모델 자동 검출 — CatBoost 같은 silent skip 영구 차단.
    `scripts/preflight_check.sh` 가 학습 시작 전 호출 → missing 시 warn.

    Args:
        force_import: True (default) — `runner._import_all_models` 와 동등한
                      19 모듈 sweep 강제 (REGISTRY 채우기). False 시 caller 가
                      이미 REGISTRY 채워뒀다고 가정.

    Returns:
        dict:
          - total_expected (int): `CATEGORY_MODELS` 의 unique 모델 합계 (현재 53 — live SSOT, 하드코딩 X; G-272 CatBoost 제외)
          - total_registered (int): `REGISTRY.get_all()` 실제 등록 수 (현재 66 = active 53 + DEFER 13;
            G-261 으로 Chronos 3종 등록 제거 → 69→66, 2026-06-15 orphan 4 DEFER 편입)
          - missing (list[tuple[str, str]]): `[(category, model_name)]` — 카테고리 list
                                              에는 있지만 REGISTRY 부재 (CatBoost 같은 사건)
          - extra (list[str]): REGISTRY 에는 있지만 카테고리 list 부재 (test 모델 등)
          - ok (bool): missing 가 0 이면 True

    Raises:
        절대 raise X — import 실패는 log.debug 로 표시 후 계속.

    Side effects:
        - force_import=True 시 19 모듈 import (≤500ms).
        - log.debug: import 실패 모듈 명시.

    Performance: O(1) (REGISTRY = dict). force_import 시 ≤500ms.

    Caller responsibility:
        - missing 발견 시 사용자 alert (preflight 의 log_warn).
        - 신규 카테고리 추가 시 `CATEGORY_MODELS` 에 등록 후 검증.

    Example:
        >>> r = verify_registry_coverage()
        >>> r["ok"]
        True
        >>> len(r["missing"])
        0
        >>> r["total_expected"]
        54
        >>> r["total_registered"]  # extras 12 = DEFER 8 + DNN-Optuna/TCN-Optuna/CQR-GBR/TabularDNN-Lite (G-261 Chronos 3종 제거 후)
        66

    See: G-161 (CatBoost 누락 영구 차단), G-169 (DNN-Conformer / NNLS-Filtered 추가),
         `CATEGORY_MODELS` (train_by_category.sh:get_models 동기화).
    """
    if force_import:
        # runner._import_all_models 와 동일한 sweep — 모든 모델 모듈 import.
        # G-161: 이전 5 모듈만 import → modern_ts/α-blend/graph/mech/foundation/ensemble
        # 누락 → 45 false-positive missing 보고. 이제 19 모듈 sweep.
        for mod in [
            "simulation.models.ts_models",
            "simulation.models.linear_models",
            "simulation.models.tree_models",
            "simulation.models.dl_models",
            "simulation.models.modern_ts",
            "simulation.models.tft_wrapper",
            "simulation.models.ensemble",
            "simulation.models.pinn_model",
            "simulation.models.rt_estimator",
            "simulation.models.bayesian_seir",
            "simulation.models.conformal",
            "simulation.models.timesfm_wrapper",   # TimesFM-2.5 (Chronos-2 대체, G-261)
            "simulation.models.tabpfn_wrapper",     # TabPFN v2 tabular foundation (G-264)
            "simulation.models.dlinear",            # DLinear (Zeng 2023, G-265)
            "simulation.models.tirex_wrapper",      # TiRex xLSTM foundation (G-265)
            "simulation.models.metapop_seir",
            "simulation.models.phase_ensemble",
            "simulation.models.foundation_model",
            "simulation.models.overseas_transfer",
            "simulation.models.epi_models",
            "simulation.models.graph_models",
            # G-231 (2026-05-22): dl_anchored 제거 — α-blend DNN 모델 폐기
            # G-261 (2026-06-13): chronos_wrapper / chronos_finetune_real 제거 — Chronos retire.
            "simulation.models.negbin_glm",
            "simulation.models.seir_forced",
            # 2026-05-26 (Sprint 1 Cleanup): epi-extended 11 + cqr 3 + graph_pyg
            # — verify_registry_coverage 누락 fix.  CATEGORY_MODELS 에는 있었으나
            # module 자체가 import 안 돼서 REGISTRY 등록 안 됨 → preflight WARN 16개.
            "simulation.models.cox_models",          # CoxPH
            "simulation.models.epiestim_models",     # EpiEstim
            "simulation.models.hhh4_models",         # hhh4-equivalent
            "simulation.models.wallinga_teunis",     # Wallinga-Teunis
            "simulation.models.glarma_models",       # GLARMA
            "simulation.models.tsir_models",         # TSIR
            "simulation.models.prophet_models",      # PROPHET
            "simulation.models.ears_models",         # EARS-C1/C2/C3
            "simulation.models.cqr_models",          # CQR-LightGBM/GBR/QuantReg
            "simulation.models.graph_models_pyg",    # GCN
            "simulation.models.flusight_baseline",   # FluSight-Baseline (G1, R4 2026-05-27)
            "simulation.models.fused_epi",           # FusedEpi (2026-06-23 신규 융합모델 — 자동등록 fire용)
            "simulation.models.seir_count",          # SeirCount-TabPFN (2026-06-23 경량 count-native, 성공)
        ]:
            try:
                __import__(mod)
            except ImportError as e:
                log.debug(f"verify_registry_coverage: {mod} skip: {e}")

    from simulation.models.base import REGISTRY as _REG
    registered = set(_REG.get_all().keys())
    expected = set()
    for cat, models in CATEGORY_MODELS.items():
        expected.update(models)

    missing = []
    for cat, models in CATEGORY_MODELS.items():
        for m in models:
            if m not in registered:
                missing.append((cat, m))

    extra = sorted(registered - expected)

    return {
        "total_expected": len(expected),
        "total_registered": len(registered),
        "missing": missing,
        "extra": extra,
        "ok": len(missing) == 0,
    }


__all__ = [
    "PAPER_PRIMARY_11",
    "NEGATIVE_CONTROL",
    "CATEGORY_MODELS",
    "ModelSnapshot",
    "build_snapshot",
    "persist_snapshot",
    "load_snapshots",
    "verify_paper_primary_frozen",
    "verify_registry_coverage",
    "sha256_file",
]


# ══════════════════════════════════════════════════════════════════════════
# CLI — Phase C6 (2026-05-12): scripts/train_by_category.sh sources from here
# ══════════════════════════════════════════════════════════════════════════
# 이전: bash 의 get_models() 가 CATEGORY_MODELS 와 중복 → silent drift risk
# (G-161 CatBoost 누락 + tree_by_category.sh:13 의 BASH 주석이 stale).
# 정정: bash 가 이 CLI 호출 → 단일 source-of-truth (CATEGORY_MODELS dict).

def _main_cli(argv: Optional[list[str]] = None) -> int:
    """CLI for scripts/train_by_category.sh: print models per category.

    Args:
        argv: ``--list-categories`` (print categories in canonical order, one per line)
              ``--get-models <category>`` (print comma-separated model names)
              ``--all`` (print ``<category>=<csv-models>`` lines for parsing)
              ``--verify`` (run verify_registry_coverage + report)

    Returns:
        Exit code: 0 PASS / 1 unknown category or verify drift / 2 bad args.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="simulation.models.registry",
        description="Single source-of-truth for category→models mapping.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list-categories", action="store_true")
    group.add_argument("--get-models", metavar="CATEGORY", type=str)
    group.add_argument("--all", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)

    if args.list_categories:
        # Canonical order matches train_by_category.sh CATS_ORDER (2026-05-02).
        # G-231 (2026-05-22): α-blend/"mech" 카테고리 제거. "dl-seq" 제거 (modern-ts 통합).
        order = ["tree", "linear", "kernel", "other", "ts", "dl-tabular",
                 "modern-ts", "cqr", "graph",
                 "foundation", "ensemble", "epi-extended"]
        for cat in order:
            if cat in CATEGORY_MODELS:
                print(cat)
        return 0

    if args.get_models is not None:
        models = CATEGORY_MODELS.get(args.get_models)
        if models is None:
            print(f"Unknown category: {args.get_models}", flush=True)
            return 1
        print(",".join(models))
        return 0

    if args.all:
        for cat, models in CATEGORY_MODELS.items():
            print(f"{cat}={','.join(models)}")
        return 0

    if args.verify:
        result = verify_registry_coverage()
        if result["ok"]:
            print(
                f"OK total_expected={result['total_expected']} "
                f"total_registered={result['total_registered']}"
            )
            return 0
        miss = ", ".join(f"[{c}]{m}" for c, m in result["missing"][:10])
        print(
            f"DRIFT missing n={len(result['missing'])} ({miss}); "
            f"extra n={len(result['extra'])}"
        )
        return 1

    return 2


if __name__ == "__main__":
    raise SystemExit(_main_cli())
