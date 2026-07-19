"""
ChampionArtifact — full inference bundle for a trained model.
============================================================

Saving just `model.pt` is insufficient: at inference time we need to apply
the *same* (transform → scaler → model → inverse-transform) pipeline that
produced the training-time fit. Forgetting any one of these leaks training
information *or* corrupts predictions:

  • Re-fitting the scaler on inference X uses inference statistics (lossy,
    distorts predictions when seasons differ).
  • Box–Cox without the saved λ collapses to log1p (different mean/variance).
  • Yeo–Johnson without the fitted `PowerTransformer` is silently a no-op.
  • Feature subset chosen by Optuna at train time must be applied at
    inference time in the same order.

Therefore R9 (per_model_optimize) wraps every champion candidate as:

    ChampionArtifact(
        model=<sklearn / torch / xgboost object>,
        scaler=<fitted StandardScaler/RobustScaler/QuantileTransformer | None>,
        transform_name="boxcox" | "yeo_johnson" | "log1p" | "identity",
        transform_state={"lambda": λ}      # boxcox
                       | {"power_transformer": PT}  # yeo-johnson
                       | {},                          # log1p / identity
        feature_indices=[2, 5, 11, ...] | None,
        feature_cols=["temp_avg", "ili_lag1", ...],
        config={"transform": ..., "scaler": ..., "n_features": ..., ...},
        meta={"trained_at": "...", "test_wis": 3.42, "test_mae": 4.12, ...},
    )

The artifact is the pickle that lands in ``models/<Name>.pt``. Both
``ChampionLog.propose`` and ``inference.predict_with_champion``
operate on this bundle.

Backward-compat: if a legacy ``.pt`` contains only a bare model object,
:func:`load_artifact` wraps it in a default-config artifact and emits a
WARN — the prediction will fall back to "no scaler, identity transform,
all features", which matches the old (broken) behaviour rather than
crashing.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class ChampionArtifact:
    """Self-contained inference bundle. All fields needed to reproduce the
    exact training-time prediction pipeline on new X.

    G-232/G-234 (2026-05-24): multicollinearity filter replay.
    When R9 (per_model_optimize) applied a VIF/corr/PCA filter before training,
    the saved ``feature_indices`` are relative to the *filtered* X, NOT the
    original full feature matrix.  ``mc_method`` + ``mc_state`` allow inference
    (Pinf) to replay the same filter so that ``predict(X_full)`` works correctly:

        apply_features(X_full):
          1. If mc_method != "none": apply mc filter (VIF/corr keep cols,
             or PCA scaler+transform) → X_mc
          2. Apply feature_indices on X_mc → X_sub
    """
    model: Any
    transform_name: str = "identity"
    transform_state: dict = field(default_factory=dict)
    scaler: Optional[Any] = None       # fitted sklearn transformer or None
    feature_indices: Optional[list[int]] = None
    feature_cols: Optional[list[str]] = None
    config: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    # G-232/G-234: multicollinearity filter state for inference replay
    mc_method: str = "none"            # "none" | "vif" | "corr" | "pca"
    mc_state: Optional[Any] = None     # list[int] (vif/corr) | {"scaler","pca"} (pca) | None
    # G-233 (2026-05-30): hierarchical preproc fitted state. When set, the y-inverse at
    # inference replays via apply_y_preproc_inverse_only(.., hier_y_state) — covering every
    # mode (none/individual/group-chain/categorical) and every METRIC/CATEGORICAL transform,
    # which the by-name transform_name path below cannot. None ⇒ flat by-name path (unchanged).
    hier_y_state: Optional[dict] = None

    # ── pipeline application ──────────────────────────────────────────
    def apply_features(self, X: np.ndarray) -> np.ndarray:
        """Subset columns the same way as at train time.

        G-232/G-234: if mc_method is set, first replay the multicollinearity
        filter (using mc_state) before applying feature_indices. This ensures
        inference (Pinf) with full X_inference produces correct predictions
        even when R9 (per_model_optimize) was trained on a mc-filtered feature subset.

        Performance: O(n × p) where n = rows, p = original feature count.
        Side effects: none (pure transform, no mutation of self).
        Caller responsibility: X must have the same column order as training X.
        """
        # Step 1: replay multicollinearity filter if used during training
        mc = (self.mc_method or "none").lower()
        if mc != "none" and self.mc_state is not None:
            try:
                if mc in ("vif", "corr"):
                    # mc_state = list[int] of kept column indices (absolute)
                    X = X[:, self.mc_state]
                elif mc == "pca":
                    # mc_state = {"scaler": fitted_scaler, "pca": fitted_pca}
                    _sc  = self.mc_state.get("scaler")
                    _pca = self.mc_state.get("pca")
                    if _sc is not None and _pca is not None:
                        X = _pca.transform(_sc.transform(
                            np.asarray(X, dtype=np.float64)))
                    else:
                        log.warning("  [artifact] PCA mc_state missing scaler/pca "
                                    "— skipping mc filter (predictions may be off)")
            except Exception as e:
                log.warning(f"  [artifact] mc filter replay ({mc}) failed: {e} "
                             f"— skipping (predictions may be off)")
        # Step 2: apply feature_indices on mc-filtered (or full) X
        if self.feature_indices is None:
            return X
        try:
            return X[:, self.feature_indices]
        except Exception as e:
            log.warning(f"  [artifact] feature_indices apply failed: {e} "
                         f"(returning X unchanged)")
            return X

    def apply_scaler(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted scaler.transform if any."""
        if self.scaler is None:
            return X
        try:
            return self.scaler.transform(X)
        except Exception as e:
            log.warning(f"  [artifact] scaler.transform failed: {e} "
                         f"(returning X unchanged — predictions will be off)")
            return X

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Run model.predict on transformed X, returns predictions in
        TRANSFORMED target space (still need inverse_transform_target)."""
        return np.asarray(self.model.predict(X), dtype=np.float64)

    def inverse_transform_target(self, y_pred_t: np.ndarray) -> np.ndarray:
        """Map transformed-space predictions back to original ILI rate space."""
        yp = np.asarray(y_pred_t, dtype=np.float64)
        # G-233 (2026-05-30): hierarchical preproc → replay its full fitted state (picklable).
        # getattr() so legacy artifacts pickled before this field load without AttributeError.
        hys = getattr(self, "hier_y_state", None)
        if hys:
            from simulation.pipeline.preproc_optuna_hierarchical import (
                apply_y_preproc_inverse_only,
            )
            return np.asarray(apply_y_preproc_inverse_only(yp, hys), dtype=np.float64)
        name = (self.transform_name or "identity").lower()
        st = self.transform_state or {}
        if name == "identity":
            return yp
        if name == "log1p":
            return np.expm1(np.clip(yp, -50, 50))
        if name == "boxcox":
            lam = st.get("lambda")
            if lam is None:
                # Lambda missing → fall back to log1p (legacy behaviour with WARN)
                log.warning("  [artifact] boxcox lambda missing — using log1p "
                             "fallback (predictions may be biased)")
                return np.expm1(np.clip(yp, -50, 50))
            if abs(lam) < 1e-6:
                return np.exp(yp)
            base = np.maximum(yp * lam + 1.0, 1e-8)
            return np.power(base, 1.0 / lam)
        if name == "yeo_johnson":
            pt = st.get("power_transformer")
            if pt is None:
                log.warning("  [artifact] yeo_johnson PowerTransformer missing — "
                             "returning yp unchanged")
                return yp
            try:
                return pt.inverse_transform(yp.reshape(-1, 1)).ravel()
            except Exception as e:
                log.warning(f"  [artifact] PowerTransformer.inverse_transform "
                             f"failed: {e} — returning yp")
                return yp
        return yp

    def predict(self, X_full: np.ndarray) -> np.ndarray:
        """End-to-end: feature_subset → scale → model → inverse_target_transform → ILI≥0 floor."""
        X1 = self.apply_features(X_full)
        X2 = self.apply_scaler(X1)
        y_t = self.predict_raw(X2)
        # G-303 (2026-06-17, 검증 적발): ILI≥0 도메인 floor in ORIGINAL units. G-298 가 트리/SVR/
        #   ElasticNet/KRR/BayesianRidge 의 transformed-space clamp 를 제거(trough 편향 수정)했는데,
        #   median-centered transform(mcmc_robust/laplace)의 affine inverse 는 하한이 없어 .pt 추론
        #   경로(phase-17/web)서 음수 ILI 가 샐 수 있었다. phase-13 4-site floor 와 동형으로 여기서 floor.
        return np.maximum(np.asarray(self.inverse_transform_target(y_t), dtype=np.float64), 0.0)

    # ── persistence ───────────────────────────────────────────────────
    def to_pickle_bytes(self) -> bytes:
        # G-304 (2026-06-17, 스모크 적발): cloudpickle 우선. GAT·OverseasTransfer 등은 nn.Module 을
        #   함수-내부 LOCAL 클래스로 정의(graph_models._build_gat_model.<locals>.GraphAttentionDNN,
        #   overseas_transfer._build_finetuning_model.<locals>.TransferModel)라 표준 pickle.dumps 가
        #   "Can't get local object" 로 실패 → ChampionArtifact 저장 실패("champion-log failed") →
        #   .pt 가 preproc 없는 base.py torch-dict 로 남아 추론서 identity+no-scaler(예측≠학습).
        #   cloudpickle 은 local 클래스를 by-value 직렬화 → 저장 성공. load_artifact 의 pickle.loads
        #   는 cloudpickle bytes 도 로드 가능(검증됨). cloudpickle 부재 시 표준 pickle fallback.
        try:
            import cloudpickle
            return cloudpickle.dumps(self)
        except Exception:
            return pickle.dumps(self)

    def summary(self) -> dict:
        """JSON-friendly summary (no model bytes, no scaler internals)."""
        return {
            "transform_name": self.transform_name,
            "transform_state_keys": list((self.transform_state or {}).keys()),
            "scaler_class": self.scaler.__class__.__name__ if self.scaler else None,
            "n_features_used": (len(self.feature_indices)
                                  if self.feature_indices is not None
                                  else (len(self.feature_cols) if self.feature_cols else None)),
            "feature_indices_first10": (
                list(self.feature_indices[:10]) if self.feature_indices else None
            ),
            "config": dict(self.config or {}),
            "meta": dict(self.meta or {}),
        }


def make_artifact(
    *,
    model: Any,
    transform_name: str,
    transform_inv_obj: Any = None,   # boxcox lambda *or* fitted PowerTransformer
    fitted_scaler: Optional[Any] = None,
    feature_indices: Optional[list[int]] = None,
    feature_cols: Optional[list[str]] = None,
    config: Optional[dict] = None,
    meta: Optional[dict] = None,
    model_name: Optional[str] = None,
    # G-232/G-234 (2026-05-25): multicollinearity filter replay
    mc_method: str = "none",          # "none"|"vif"|"corr"|"pca"
    mc_state: Optional[Any] = None,   # list[int] (vif/corr) | {"scaler","pca"} (pca)
    hier_y_state: Optional[dict] = None,  # G-233: hierarchical preproc state for inference
) -> ChampionArtifact:
    """Assemble a ChampionArtifact, normalizing transform_state by name.

    `transform_inv_obj`:
      • boxcox       → λ (float)
      • yeo_johnson  → fitted sklearn PowerTransformer
      • log1p / identity → ignored

    The artifact's ``meta`` dict is auto-tagged with:
      • ``saved_at``     — UTC timestamp
      • ``tier``         — "paper" / "extra" / "negative" / "unknown"
                            (resolved via registry.tier_of)
      • ``category``     — paper or extra sub-category
                            (resolved via registry.category_of)
    so downstream visualize / predict-real / champion_log can group results
    by tier without importing the registry separately.
    """
    name = (transform_name or "identity").lower()
    st: dict = {}
    if name == "boxcox" and transform_inv_obj is not None:
        try:
            st["lambda"] = float(transform_inv_obj)
        except Exception:
            log.warning("  [artifact] boxcox lambda not coercible to float; "
                         "leaving transform_state empty")
    elif name == "yeo_johnson" and transform_inv_obj is not None:
        st["power_transformer"] = transform_inv_obj

    base_meta = {"saved_at":
                 datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    # Auto-tier label
    if model_name:
        try:
            from simulation.models.registry import tier_of, category_of
            base_meta["tier"]     = tier_of(model_name)
            base_meta["category"] = category_of(model_name)
            base_meta["model_name"] = model_name
        except Exception as e:
            log.debug(f"  [artifact] tier resolution failed: {e}")
            base_meta["tier"]     = "unknown"
            base_meta["category"] = "unknown"
    base_meta.update(meta or {})

    return ChampionArtifact(
        model=model,
        transform_name=name,
        transform_state=st,
        scaler=fitted_scaler,
        feature_indices=list(feature_indices) if feature_indices is not None else None,
        feature_cols=list(feature_cols) if feature_cols is not None else None,
        config=dict(config or {}),
        meta=base_meta,
        mc_method=(mc_method or "none").lower(),
        mc_state=mc_state,
        hier_y_state=hier_y_state,
    )


def load_artifact(pt_path) -> Optional[ChampionArtifact]:
    """Load a `.pt` file. Resolution order:

      1. ``pickle.loads(bytes)`` — standard ChampionArtifact / sklearn pickle
      2. ``torch.load(file, weights_only=False)`` — torch.save'd nn.Module
         (tensor pickles use a "persistent_load" hook plain pickle can't)

    Bare-model objects (legacy from R2 baseline) get wrapped in a default
    identity-transform / no-scaler artifact and a WARN is logged.
    Returns None on missing or unparseable file.
    """
    from pathlib import Path
    p = Path(pt_path)
    if not p.exists():
        log.warning(f"  [artifact] not found: {p}")
        return None

    obj = None
    err_pickle: Optional[str] = None
    err_torch: Optional[str] = None

    # 1. plain pickle
    try:
        obj = pickle.loads(p.read_bytes())
    except Exception as e:
        err_pickle = str(e)[:120]

    # 2. torch.load fallback (lazy import — torch may not be installed)
    if obj is None:
        try:
            import torch
            # Pre-import model modules so torch unpickler can find their
            # classes via fully-qualified name lookup.
            for _m in ("dl_models", "graph_models", "physics_models",
                        "pinn_model", "seir_forced", "foundation_models",
                        "time_series_dl"):
                try:
                    __import__(f"simulation.models.{_m}")
                except Exception:
                    pass
            try:
                obj = torch.load(p, weights_only=False, map_location="cpu")
            except TypeError:
                # weights_only kwarg only on torch ≥ 2.0
                obj = torch.load(p, map_location="cpu")
        except Exception as e:
            err_torch = str(e)[:120]

    if obj is None:
        log.error(f"  [artifact] load failed for {p}: "
                   f"pickle={err_pickle}; torch={err_torch}")
        return None

    if isinstance(obj, ChampionArtifact):
        return obj

    # Legacy: bare model / nn.Module / state_dict — wrap in default artifact
    log.warning(f"  [artifact] {p.name}: legacy bare-model pickle. "
                 f"Inference will use identity transform + no scaler "
                 f"(predictions may differ from training-time pipeline).")
    return ChampionArtifact(
        model=obj,
        transform_name="identity",
        transform_state={},
        scaler=None,
        feature_indices=None,
        feature_cols=None,
        config={"legacy": True},
        meta={"loaded_from": str(p), "wrapped_legacy_at":
              datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
    )


__all__ = ["ChampionArtifact", "make_artifact", "load_artifact"]
