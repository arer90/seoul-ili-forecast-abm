"""
R11: Comprehensive SHAP + Feature Importance — ALL model families.
=======================================================================
Every promoted champion (.pt) is explained, regardless of family:

  • Universal backbone — **permutation importance** on ``artifact.predict(X)``.
    Works for ANY model (tree, linear/GLM, kernel, **deep-learning**, classical
    TS) in the ORIGINAL feature space, because it only needs ``.predict``.
    Nothing is left unexplained.

  • Native SHAP (richer values + beeswarm), dispatched by the raw model family
    extracted from the forecaster wrapper:
        tree   → shap.TreeExplainer
        linear → shap.LinearExplainer
        dl     → shap.GradientExplainer / DeepExplainer  (torch nn.Module)
        kernel / other → shap.KernelExplainer  (small background)
    Each native explainer is best-effort; on any failure the model still has
    permutation importance.

Artifacts written under ``<save_dir>/shap/``:
    <model>/importance.csv        permutation (+ native) per-feature importance
    <model>/shap_values.npy       native SHAP values (when available)
    <model>/beeswarm.png, bar.png native SHAP figures (when available)
    _summary.json                 per-model status + top features
    REPORT.md                     human-readable summary

R11 is degrade-and-continue: a per-model failure logs and proceeds.
"""
import json
import logging
import time
import numpy as np
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# ── feature importance helpers ────────────────────────────────────────────
def _compute_mi_importance(X_train, y_train, feature_cols, max_features=50):
    """Mutual Information feature importance (model-free baseline).

    G-121: sklearn MI rejects NaN — sanitize first.
    """
    from sklearn.feature_selection import mutual_info_regression
    from .sanitize import sanitize_numpy
    X_clean = sanitize_numpy(X_train, feature_cols, label="MI")
    y_clean = sanitize_numpy(y_train, label="MI target")
    mi = mutual_info_regression(X_clean, y_clean, random_state=42, n_neighbors=5)
    ranked = sorted(zip(feature_cols, mi), key=lambda x: -x[1])
    return ranked[:max_features], dict(ranked)


def _permutation_importance(predict_fn, X, y, feature_cols, n_repeats=3, seed=42,
                            time_budget=90.0):
    """Model-agnostic permutation importance using a bare ``predict`` callable.

    Importance of feature j = mean increase in MSE when column j is shuffled.
    Works for EVERY family (the universal backbone) because it only calls
    ``predict_fn(X)``. Models that ignore covariates (e.g. univariate ARIMA)
    correctly score ~0 for every feature.

    Args:
        predict_fn: callable X(n,p)->y_hat(n,). Wrapped artifact.predict.
        X: (n, p) feature matrix in ORIGINAL space.
        y: (n,) ground truth.
        feature_cols: length-p names.
        n_repeats: shuffles per feature (averaged).
        time_budget: soft wall-clock cap in seconds (G-360). If the *measured*
            base-predict time × p × n_repeats would exceed this, fall back to
            n_repeats=1 over the top-K highest-variance features (K = budget /
            per-call time). Fast models (ms predict) are unaffected; slow-predict
            high-dim models (TabPFN/foundation, p=348) are bounded.

    Returns:
        list[(feature, importance)] sorted desc, or [] if predict fails.

    Performance: O(min(p, budget/dt) · n_repeats) predict calls, ≤ ~time_budget s
        wall-clock for the perturbation loop. Side effects: none.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rng = np.random.default_rng(seed)
    try:
        _t0 = time.perf_counter()
        base = np.asarray(predict_fn(X), dtype=np.float64)
        dt = time.perf_counter() - _t0          # one predict's wall-clock
    except Exception as e:
        log.debug(f"    [perm] base predict failed: {e}")
        return []
    mask = np.isfinite(base) & np.isfinite(y)
    if mask.sum() < 3:
        return []
    base_err = float(np.mean((base[mask] - y[mask]) ** 2))
    p = X.shape[1]
    # G-360 (2026-06-25): predict-TIME budget. permutation = O(p·n_repeats) predict;
    #   slow-predict(TabPFN ~5s) × high-dim(348) = 1044 predict 가 R11 을 1.5h+ 정체
    #   (SeirCount-TabPFN). dt 측정 → est > budget 이면 n_repeats=1 + 분산 top-K feature
    #   (predict 0회 proxy). 빠른 모델(dt≈ms)은 est≪budget 라 영향 0 = full 3-repeat.
    feat_idx = list(range(p))
    if dt > 0 and dt * p * n_repeats > time_budget:
        n_repeats = 1
        max_feats = max(5, int(time_budget / dt))
        if p > max_feats:
            var = np.nan_to_num(np.nanvar(X, axis=0))
            feat_idx = sorted(int(i) for i in np.argsort(-var)[:max_feats])
            log.info(f"    [perm] predict {dt:.2f}s/call × p={p} → budget cap "
                     f"(G-360): n_repeats=1, top-{max_feats} var-feat")
    imps = np.zeros(p, dtype=np.float64)
    for j in feat_idx:
        errs = []
        for _ in range(n_repeats):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            try:
                pred = np.asarray(predict_fn(Xp), dtype=np.float64)
            except Exception:
                pred = base  # treat unpredictable perturbation as no-change
            m = np.isfinite(pred) & np.isfinite(y)
            if m.sum() >= 3:
                errs.append(float(np.mean((pred[m] - y[m]) ** 2)))
        imps[j] = (np.mean(errs) - base_err) if errs else 0.0
    ranked = sorted(zip(feature_cols, imps.tolist()), key=lambda t: -t[1])
    return ranked


# ── raw-model extraction + family detection ───────────────────────────────
def _extract_raw_model(forecaster):
    """Reach into a forecaster wrapper for the underlying raw estimator.

    Champions are saved as forecaster wrappers (e.g. ``NegBinGLMForecaster``)
    that expose ``.predict`` but not ``feature_importances_``/``coef_``/torch
    parameters at the top level. The raw model lives under a private attr that
    varies per family; try the common ones.

    Returns:
        the raw estimator (sklearn / xgboost / torch.nn.Module / statsmodels) or
        the forecaster itself if no inner model is found.
    """
    if forecaster is None:
        return None
    # direct hit — already a raw model
    if (hasattr(forecaster, "feature_importances_") or hasattr(forecaster, "coef_")
            or hasattr(forecaster, "parameters")):
        return forecaster
    for attr in ("_model", "model", "_estimator", "estimator_", "regressor_",
                 "_regressor", "_net", "net", "_booster", "booster_", "_results"):
        inner = getattr(forecaster, attr, None)
        if inner is not None and not isinstance(inner, (int, float, str, bool, dict, list)):
            return inner
    # DL: list of seed models
    seq = getattr(forecaster, "_models", None)
    if isinstance(seq, (list, tuple)) and seq:
        return seq[0]
    return forecaster


def _detect_family(raw) -> str:
    """Classify a raw estimator → explainer family.

    Returns one of: 'tree' | 'linear' | 'dl' | 'kernel' | 'other'.
    """
    if raw is None:
        return "other"
    cls = type(raw).__name__.lower()
    mod = type(raw).__module__.lower()
    # deep learning: torch module
    if hasattr(raw, "parameters") and ("torch" in mod or hasattr(raw, "forward")):
        return "dl"
    # tree ensembles
    if hasattr(raw, "feature_importances_") or any(
            t in cls for t in ("forest", "boosting", "xgb", "lgbm", "lightgbm",
                               "catboost", "extratrees", "tree")):
        return "tree"
    # kernel methods (check before generic linear — SVR has coef_ only when linear)
    if any(t in cls for t in ("kernelridge", "svr", "svc", "gaussianprocess")):
        return "kernel"
    # linear / GLM (sklearn coef_ or statsmodels results)
    if hasattr(raw, "coef_") or "glm" in cls or "linear" in cls or "statsmodels" in mod \
            or any(t in cls for t in ("ridge", "lasso", "elasticnet", "bayesian",
                                      "poisson", "negbin", "ols", "glmresults")):
        return "linear"
    return "other"


# ── native SHAP per family ─────────────────────────────────────────────────
def _native_shap(raw, family, X_bg, X_sample, feature_names):
    """Compute native SHAP values for a raw model in the model-input space.

    Args:
        raw: extracted raw estimator.
        family: from _detect_family.
        X_bg: (nb, q) background sample (model-input space).
        X_sample: (ns, q) sample to explain.
        feature_names: length-q names for the model-input columns.

    Returns:
        (ranked[(feat, mean|shap|)], shap_values ndarray|None). ([], None) on failure.
    """
    try:
        import shap
    except Exception:
        return [], None
    try:
        sv = None
        if family == "tree":
            sv = shap.TreeExplainer(raw).shap_values(X_sample)
        elif family == "linear":
            sv = shap.LinearExplainer(raw, X_bg).shap_values(X_sample)
        elif family == "dl":
            import torch
            raw.eval()
            bg = torch.as_tensor(np.asarray(X_bg, dtype=np.float32))
            xs = torch.as_tensor(np.asarray(X_sample, dtype=np.float32))
            try:
                sv = shap.DeepExplainer(raw, bg).shap_values(xs)
            except Exception as _de:
                log.debug(f"    [native dl] DeepExplainer fail ({_de}); GradientExplainer")
                sv = shap.GradientExplainer(raw, bg).shap_values(xs)
        else:  # kernel / other → model-agnostic KernelExplainer on a tiny background
            f = raw.predict if hasattr(raw, "predict") else raw
            # G-348 (2026-06-25): 고차원(p>100) kernel SHAP = KernelExplainer 의 underdetermined 가중회귀로
            #   pathologically 느림(SVR-RBF 348-feat·X_sample 100행 → 40분+ 정체 실측, R11 SHAP 1-2h 블록).
            #   permutation importance(이미 perm=✓ 별도 계산)가 model-agnostic 대체라 native kernel SHAP 생략.
            if np.asarray(X_sample).shape[1] > 100:
                # G-348b (2026-06-25): log.debug → log.info — skip 분기는 정상 작동(검증: SVR-RBF 348-feat
                #   0.9초 [],None 반환)인데 log.debug 가 WARNING 레벨서 침묵 → "안 먹는 듯" 오귀속. INFO 로 가시화.
                log.info(f"    [native kernel] p={np.asarray(X_sample).shape[1]}>100 → KernelExplainer 생략(G-348, perm 대체)")
                return [], None
            bg = shap.kmeans(X_bg, min(10, len(X_bg))) if len(X_bg) > 10 else X_bg
            Xs = np.asarray(X_sample)[:min(30, len(X_sample))]   # 행 cap (SHAP 평균 안정 + 속도 bound)
            sv = shap.KernelExplainer(f, bg).shap_values(Xs, nsamples=100, silent=True)
        if isinstance(sv, list):          # some explainers return a list (one per output)
            sv = sv[0]
        sv = np.asarray(sv, dtype=np.float64)
        if sv.ndim == 3 and sv.shape[-1] == 1:
            sv = sv[..., 0]               # (n, p, 1) single-output torch/DL → (n, p)
        if sv.ndim == 1:
            sv = sv.reshape(1, -1)
        imp = np.abs(sv).mean(axis=0)
        names = feature_names[:len(imp)]
        ranked = sorted(zip(names, imp.tolist()), key=lambda t: -t[1])
        return ranked, sv
    except Exception as e:
        log.debug(f"    [native {family}] SHAP failed: {e}")
        return [], None


def _permutation_shap_fallback(predict_fn, X, feature_cols, perm_ranked, k=30, max_rows=30,
                               time_budget=120.0):
    """Model-agnostic per-sample SHAP via ``shap.PermutationExplainer`` (G-362).

    Fallback for when the family-specific native explainer fails or is skipped:
    deep ``DeepExplainer``/``GradientExplainer`` fail (custom nets), GAM
    ``LinearExplainer`` fail (non-sklearn-linear), high-dim kernel
    ``KernelExplainer`` skipped (G-348). Runs in the ORIGINAL feature space
    (``artifact.predict``) so it aligns with the permutation-importance column.
    For ``p > k`` it attributes only the top-k features (by the already-computed
    permutation importance) with the rest held at the background median → bounds
    cost to ``~2·k+1`` predict calls regardless of ``p`` (e.g. SVR-RBF 348-feat).

    Args:
        predict_fn: ``artifact.predict`` — callable X(n,p)->y_hat(n,), original space.
        X: (n, p) original-space feature matrix.
        feature_cols: length-p names.
        perm_ranked: permutation-importance ranking ``[(feat, imp), …]`` (top-k pick).
        k: max features to attribute when p>k (high-dim cap).
        max_rows: rows of X to explain.

    Returns:
        ``(ranked[(feat, mean|shap|)], sv(n,p))`` or ``([], None)`` on failure.

    Performance: ``~2·min(p,k)+1`` predict calls. Side effects: none.
    """
    try:
        import shap
    except Exception:
        return [], None
    try:
        X = np.asarray(X, dtype=np.float64)
        p = X.shape[1]
        Xs = X[:min(max_rows, len(X))]
        # G-362b (2026-06-25): predict-TIME 가드 (G-360 정신) — PermutationExplainer 는
        #   ~2·k+1 predict 를 부르는데, ultra-slow predict(TabPFN ~5-30s/call, SeirCount)면
        #   bounded eval 수여도 wall-clock 폭발(R11 1/30서 25분+ 정체 실측). base predict dt 측정 →
        #   dt·(2k+1) > budget 면 k 축소; k<3(=TabPFN류)면 fallback skip(perm importance 유지,
        #   per-sample SHAP 비현실). 빠른 모델(dt≈ms)은 영향 0.
        try:
            _t0 = time.perf_counter()
            predict_fn(Xs)
            dt = time.perf_counter() - _t0
        except Exception:
            return [], None
        k_eff = min(k, p)
        if dt > 0 and dt * (2 * k_eff + 1) > time_budget:
            k_eff = int((time_budget / dt - 1) / 2)
            if k_eff < 3:
                log.info(f"    [perm-shap G-362] predict {dt:.1f}s/call × p={p} → fallback skip "
                         f"(perm importance 유지; TabPFN류 per-sample SHAP 비현실)")
                return [], None
        if p <= k_eff:
            sv = np.asarray(shap.PermutationExplainer(predict_fn, X)(
                Xs, max_evals=2 * p + 1).values, dtype=np.float64)
        else:
            # high-dim: attribute top-k_eff by perm importance (fallback: variance), rest fixed
            if perm_ranked:
                idx = [feature_cols.index(f) for f, _ in perm_ranked[:k_eff]
                       if f in feature_cols][:k_eff]
            else:
                idx = [int(i) for i in np.argsort(
                    -np.nan_to_num(np.nanvar(X, axis=0)))[:k_eff]]
            if not idx:
                return [], None
            bg_med = np.nan_to_num(np.nanmedian(X, axis=0))

            def f_sub(z_sub):
                z2 = np.atleast_2d(np.asarray(z_sub, dtype=np.float64))
                full = np.tile(bg_med, (len(z2), 1))
                full[:, idx] = z2
                return np.asarray(predict_fn(full), dtype=np.float64).ravel()

            sv_sub = np.asarray(shap.PermutationExplainer(f_sub, X[:, idx])(
                Xs[:, idx], max_evals=2 * len(idx) + 1).values, dtype=np.float64)
            sv = np.zeros((sv_sub.shape[0], p), dtype=np.float64)
            sv[:, idx] = sv_sub
        if sv.ndim == 1:
            sv = sv.reshape(1, -1)
        imp = np.abs(sv).mean(axis=0)
        names = list(feature_cols)[:len(imp)]
        ranked = sorted(zip(names, imp.tolist()), key=lambda t: -t[1])
        return ranked, sv
    except Exception as e:
        log.debug(f"    [perm-shap G-362] failed: {e}")
        return [], None


def _write_figures(sv, X_sample, feature_names, out_dir, model_name):
    """Beeswarm + bar SHAP figures (best-effort; needs shap + matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap
        names = list(feature_names[:sv.shape[1]])
        for kind, fn in (("bar", lambda: shap.summary_plot(
                                sv, X_sample[:, :sv.shape[1]], feature_names=names,
                                plot_type="bar", show=False)),
                         ("beeswarm", lambda: shap.summary_plot(
                                sv, X_sample[:, :sv.shape[1]], feature_names=names,
                                show=False))):
            try:
                plt.figure()
                fn()
                plt.tight_layout()
                plt.savefig(out_dir / f"{kind}.png", dpi=120, bbox_inches="tight")
                plt.close("all")
            except Exception as _fe:
                log.debug(f"    [fig {kind}] {model_name}: {_fe}")
                plt.close("all")
    except Exception as e:
        log.debug(f"    [figures] {model_name}: {e}")


# ── per-model orchestration ────────────────────────────────────────────────
def _four_axis_explanation(name, sv, feature_names, family, predictions=None) -> dict:
    """Organize SHAP values into the FOUR explainable-AI axes the thesis reports
    (사용자 명시: "feature, input, output, model 에 대해서 정리").

    From the (n_sample × n_feature) SHAP matrix (the additive identity
    ``prediction ≈ base + Σ_j φ_j``), derive:
      * FEATURE — which inputs matter GLOBALLY (mean|φ| ranking + signed direction);
      * INPUT  — per-instance attribution for representative rows (which inputs
                 drove the highest / lowest / median prediction);
      * OUTPUT — the prediction DECOMPOSITION (base value, range, fraction of the
                 attribution carried by the top-3 features);
      * MODEL  — family-level behavior (explainer family, dominant feature, its
                 sign-consistency ⇒ monotone-ish vs interaction-driven).

    Args:
        name: champion model id. sv: SHAP values (n, p) or (n, p, 1). feature_names:
        length-p. family: detected explainer family. predictions: optional (n,)
        model output aligned to ``sv`` rows (enables INPUT/OUTPUT axes).

    Returns:
        ``{feature_axis, input_axis, output_axis, model_axis}`` — JSON-safe. Never
        raises (a degraded axis returns its reason).

    Performance: O(n·p). Side effects: none.
    """
    sv = np.asarray(sv, dtype=np.float64)
    if sv.ndim == 3:                              # (n, p, 1) squeeze gotcha
        sv = sv[..., 0]
    if sv.ndim != 2 or sv.size == 0:
        return {"error": f"unexpected SHAP shape {sv.shape}"}
    n, p = sv.shape
    names = list(feature_names)[:p] + [f"f{i}" for i in range(len(feature_names), p)]
    abs_mean = np.abs(sv).mean(axis=0)
    order = list(np.argsort(abs_mean)[::-1])
    preds = None if predictions is None else np.asarray(predictions, dtype=np.float64).ravel()
    if preds is not None and len(preds) != n:
        preds = None

    # FEATURE axis — global importance + signed direction
    feature_axis = [{"feature": names[i], "mean_abs_shap": round(float(abs_mean[i]), 6),
                     "mean_shap": round(float(sv[:, i].mean()), 6),
                     "direction": "raises" if sv[:, i].mean() >= 0 else "lowers"}
                    for i in order[:min(15, p)]]

    # OUTPUT axis — prediction = base + Σφ
    row_sum = sv.sum(axis=1)
    total_abs = float(np.abs(sv).sum()) + 1e-12
    top3_abs = float(np.abs(sv[:, order[:3]]).sum())
    output_axis = {"identity": "prediction ≈ base_value + Σ_j shap_j",
                   "base_value": round(float(np.mean(preds) - np.mean(row_sum)), 4)
                   if preds is not None else None,
                   "prediction_mean": round(float(np.mean(preds)), 4) if preds is not None else None,
                   "prediction_range": [round(float(np.min(preds)), 4),
                                        round(float(np.max(preds)), 4)] if preds is not None else None,
                   "fraction_explained_by_top3": round(top3_abs / total_abs, 4)}

    # INPUT axis — per-instance attribution for representative rows
    def _row(i):
        contrib = sorted(((names[j], float(sv[i, j])) for j in range(p)),
                         key=lambda t: abs(t[1]), reverse=True)[:3]
        return {"row": int(i),
                "prediction": round(float(preds[i]), 4) if preds is not None else None,
                "top_drivers": [{"feature": f, "shap": round(v, 4)} for f, v in contrib]}
    if preds is not None and n >= 3:
        srt = np.argsort(preds)
        input_axis = {"highest_prediction": _row(int(srt[-1])),
                      "lowest_prediction": _row(int(srt[0])),
                      "median_prediction": _row(int(srt[n // 2]))}
    else:
        input_axis = {"representative": _row(0)}

    # MODEL axis — family-level behavior
    dom = order[0]
    consistency = float((np.sign(sv[:, dom]) == np.sign(sv[:, dom].mean() or 1.0)).mean())
    model_axis = {"family": family, "n_features_explained": int(p), "n_samples": int(n),
                  "dominant_feature": names[dom],
                  "dominant_feature_sign_consistency": round(consistency, 3),
                  "global_mean_abs_shap": round(float(abs_mean.mean()), 6),
                  "behavior": ("monotone-ish in dominant feature" if consistency > 0.8
                               else "context-dependent (feature interactions)")}
    return {"feature_axis": feature_axis, "input_axis": input_axis,
            "output_axis": output_axis, "model_axis": model_axis}


def _measured(ranking) -> bool:
    """True only if a ranking carries at least one non-zero attribution.

    ``_permutation_importance`` starts from ``np.zeros(p)`` and always returns one
    entry per feature, so a run in which nothing could be measured — the model is
    invariant to the perturbation, every scored attempt raised, or the budget cap
    left most features untouched — still returns a FULL, non-empty list of
    ``(feature, 0.0)`` pairs. Testing that list for truthiness therefore reports
    "measured" for a result that measured nothing, and a stable sort over an
    all-zero vector then emits the original column order as if it were a ranking.

    That is how 7 of 41 models shipped an all-zero ``shap_values.npy`` while
    ``_summary.json`` counted them among ``n_with_native_shap`` and ``REPORT.md``
    listed ``temp_avg, temp_min, humidity, wind_speed`` — columns 0-3 — as their
    top drivers. A reader could not tell "measured, genuinely irrelevant" from
    "never measured".
    """
    if not ranking:
        return False
    try:
        return any(float(v) != 0.0 and np.isfinite(float(v)) for _, v in ranking)
    except (TypeError, ValueError):
        return False


def _measured_array(sv) -> bool:
    """Same rule for a SHAP value matrix: an all-zero matrix explains nothing."""
    if sv is None:
        return False
    a = np.asarray(sv, dtype=np.float64)
    return bool(a.size) and bool(np.any(np.isfinite(a) & (a != 0.0)))


def _explain_one(name, artifact, X_full, y, feature_cols, out_root):
    """Explain a single champion: universal permutation + native SHAP + write."""
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    status = {"model": name, "permutation": False, "native": False, "family": "other"}

    # 1) Universal permutation importance (original feature space) — every family.
    # G-360: X-무시 foundation(USES_FEATURES=False: TiRex/TimesFM-2.5/DLinear/TiRex-LoRA)은
    #   X shuffle 해도 예측 불변 → permutation 무의미 + foundation predict 가 느려 R11 정체.
    #   skip + 명시(가짜 all-zero importance 보다 정직). slow high-dim(TabPFN 등)은 Part B budget cap.
    # ChampionArtifact wrapper 는 USES_FEATURES 를 노출 안 함 → inner forecaster(art.model)서 읽음.
    if not getattr(getattr(artifact, "model", None), "USES_FEATURES", True):
        status["permutation_note"] = ("X-independent foundation (USES_FEATURES=False) — "
                                      "forecasts from ILI history only; feature permutation N/A")
        perm = []
    else:
        perm = _permutation_importance(artifact.predict, X_full, y, feature_cols)
    if _measured(perm):
        status["permutation"] = True
        status["top_permutation"] = [f for f, _ in perm[:5]]
    elif perm:
        status["permutation_note"] = (
            "permutation ran but every attribution came back 0.0 — the model is "
            "invariant to the perturbation or the scored predictions failed; "
            "NOT counted as measured")

    # 2) Native SHAP (model-input space) — best effort per family.
    native, sv = [], None
    try:
        raw = _extract_raw_model(getattr(artifact, "model", None))
        family = _detect_family(raw)
        status["family"] = family
        # reconstruct the model-input X the raw estimator actually sees
        Xmi = artifact.apply_scaler(artifact.apply_features(np.asarray(X_full, dtype=np.float64)))
        # model-input feature names (after feature_indices); fall back to generic
        fi = getattr(artifact, "feature_indices", None)
        if fi is not None and len(fi) == Xmi.shape[1]:
            names = [feature_cols[i] if i < len(feature_cols) else f"f{i}" for i in fi]
        elif Xmi.shape[1] == len(feature_cols):
            names = list(feature_cols)
        else:
            names = [f"f{i}" for i in range(Xmi.shape[1])]
        if family != "other" or True:  # KernelExplainer covers 'other' too
            bg = Xmi[:min(50, len(Xmi))]
            xs = Xmi[:min(100, len(Xmi))]
            native, sv = _native_shap(raw, family, bg, xs, names)
            if native and not _measured(native):
                status["native_note"] = (
                    "native SHAP returned an all-zero attribution — not counted "
                    "as measured, and shap_values.npy is not written")
                native, sv = [], None
            if _measured(native):
                status["native"] = True
                status["top_native"] = [f for f, _ in native[:5]]
                if _measured_array(sv):
                    np.save(out_dir / "shap_values.npy", sv)
                    _write_figures(sv, xs, names, out_dir, name)
                    # 4-axis XAI organization (feature/input/output/model)
                    try:
                        preds = np.asarray(
                            artifact.predict(np.asarray(X_full, dtype=np.float64)[:len(xs)]),
                            dtype=np.float64).ravel()
                    except Exception:
                        preds = None
                    axes = _four_axis_explanation(name, sv, names, family, preds)
                    import json as _json
                    (out_dir / "xai_explanation.json").write_text(
                        _json.dumps(axes, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")
                    status["xai_axes"] = ["feature", "input", "output", "model"]
                    if "model_axis" in axes:
                        status["dominant_feature"] = axes["model_axis"].get("dominant_feature")
    except Exception as e:
        log.debug(f"    [{name}] native path failed: {e}")

    # 2b) G-362: native SHAP 실패/skip 시 model-agnostic PermutationExplainer fallback (원본 공간).
    #   대상: deep(DeepExplainer/GradientExplainer fail)·GAM(LinearExplainer fail)·high-dim kernel
    #   (KernelExplainer skip, G-348). X 사용 모델만(perm 있음); X-무시 foundation(perm=[])은 제외.
    #   고차원은 top-K subset 로 bound(SVR-RBF 348→top-30, ~61 predict).
    if not status["native"] and _measured(perm):
        nat2, sv2 = _permutation_shap_fallback(artifact.predict, X_full, list(feature_cols), perm)
        if nat2 and not _measured(nat2):
            status["native_note"] = (
                "G-362 permutation-SHAP fallback returned an all-zero attribution "
                "— not counted as measured")
            nat2, sv2 = [], None
        if _measured(nat2):
            native = nat2                       # importance.csv 의 dict(native) 가 이걸 소비
            status["native"] = True
            status["native_method"] = "permutation_shap (G-362 fallback)"
            status["top_native"] = [f for f, _ in nat2[:5]]
            if _measured_array(sv2):
                np.save(out_dir / "shap_values.npy", sv2)
                try:
                    _write_figures(sv2, np.asarray(X_full, dtype=np.float64),
                                   list(feature_cols), out_dir, name)
                except Exception as _fe:
                    log.debug(f"    [{name}] G-362 figure fail: {_fe}")

    # 3) Write per-feature importance CSV (permutation ∪ native).
    import csv as _csv
    perm_map = dict(perm)
    nat_map = dict(native)
    feats = list(dict.fromkeys(list(perm_map) + list(nat_map)))
    with (out_dir / "importance.csv").open("w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["feature", "permutation_importance", "native_shap_importance"])
        for f in feats:
            w.writerow([f, round(perm_map.get(f, float("nan")), 6),
                        round(nat_map.get(f, float("nan")), 6)])
    status["importance_csv"] = str(out_dir / "importance.csv")
    return status


from simulation.utils.resource_tracker import track_resources


def _active_model_names() -> list:
    """Flatten registry.CATEGORY_MODELS (the active-lineup SSOT) into a flat model-name list."""
    from simulation.models.registry import CATEGORY_MODELS
    names: list = []
    for v in CATEGORY_MODELS.values():
        names.extend(v if isinstance(v, (list, tuple, set)) else [v])
    return names


def _select_champion_pts(model_dir: Path, active_models) -> list:
    """Select ONE champion .pt per active model — not every ``*.pt`` (G-310, 2026-06-18).

    The champion-challenger writes versioned artifacts per model (``<name>.pt`` eval champion,
    ``<name>_deploy.pt`` production refit, ``<name>_attempt_vN_<ts>.pt`` challenger history).
    Previously SHAP globbed ``*.pt`` and explained ALL of them (231 incl. retired models +
    cross-run accumulation) instead of the active lineup's finals. This returns, per active
    model, its eval champion (plain ``<name>.pt``) if present else its newest ``<name>_*.pt``
    artifact. Models with no artifact (failed this run) are skipped; retired / cross-run /
    non-active .pt are never returned.

    Args:
        model_dir: directory holding champion ``.pt`` files.
        active_models: iterable of active model names (registry.CATEGORY_MODELS SSOT).

    Returns:
        list[(name, Path)] — one entry per active model that has an artifact, name-sorted.

    Performance: O(len(active)) directory globs. Side effects: none (read-only stat).
    """
    if not model_dir.exists():
        return []
    selected: list = []
    for name in sorted(set(active_models)):
        plain = model_dir / f"{name}.pt"
        if plain.exists():
            selected.append((name, plain))
            continue
        alts = sorted(model_dir.glob(f"{name}_*.pt"), key=lambda p: p.stat().st_mtime)
        if alts:
            selected.append((name, alts[-1]))
    return selected


@track_resources("shap")
def run_shap(X_all, y_all, feature_cols, config,
             model_dir: Optional[Path] = None) -> dict:
    """R11: comprehensive SHAP + importance across ALL champion families.

    Args:
        X_all: (n, p) full feature matrix (original space).
        y_all: (n,) target.
        feature_cols: length-p feature names.
        config: pipeline config (provides get_model_dir / get_save_dir).
        model_dir: override champion .pt directory (default config.get_model_dir()).

    Returns:
        {"feature_importance": {mi_ranking, per_model: {...}, summary_path},
         "elapsed": float}.

    Side effects: writes <save_dir>/shap/ artifacts. Never raises
    (degrade-and-continue) — per-model failures are logged and recorded.
    """
    from .utils.logging_util import phase_banner, fmt_time
    phase_banner("R11", "SHAP + Feature Importance (all families)")
    t0 = time.time()

    from simulation.pipeline.data import compute_split_indices
    from simulation.utils.model_artifact import load_artifact
    n_train, n_val, n_test = compute_split_indices(len(y_all), config)
    X_all = np.asarray(X_all, dtype=np.float64)
    y_all = np.asarray(y_all, dtype=np.float64)
    # explain on the held-out test slab (generalization importance)
    test_start = n_train + n_val
    X_eval, y_eval = X_all[test_start:], y_all[test_start:]
    if len(y_eval) < 5:                      # tiny data → fall back to train tail
        X_eval, y_eval = X_all[:n_train], y_all[:n_train]

    results: dict = {}
    # MI baseline (model-free)
    mi_ranked, _ = _compute_mi_importance(X_all[:n_train], y_all[:n_train], feature_cols)
    results["mi_ranking"] = [{"feature": f, "score": round(float(s), 6)} for f, s in mi_ranked]
    log.info(f"  MI Top-5: {[f for f, _ in mi_ranked[:5]]}")

    model_dir = Path(model_dir) if model_dir else config.get_model_dir()
    out_root = config.get_save_dir() / "shap"
    out_root.mkdir(parents=True, exist_ok=True)

    # G-310 (2026-06-18): explain ONLY each active model's champion (one per model), NOT every
    # *.pt — the champion-challenger + un-archived models/ leaves retired + cross-run challenger
    # history (was 231 .pt → SHAP explained all). Filter to CATEGORY_MODELS finals.
    _n_all = len(list(model_dir.glob("*.pt"))) if model_dir.exists() else 0
    champion_pts = _select_champion_pts(model_dir, _active_model_names())
    log.info(f"  explaining {len(champion_pts)} active champion(s) from {model_dir} "
             f"(filtered from {_n_all} .pt on disk)")
    per_model: dict = {}
    for name, pt in champion_pts:
        try:
            artifact = load_artifact(pt)
            st = _explain_one(name, artifact, X_eval, y_eval, list(feature_cols), out_root)
            per_model[name] = st
            log.info(f"  [{name}] family={st['family']} "
                     f"perm={'✓' if st['permutation'] else '·'} "
                     f"native={'✓' if st['native'] else '·'}")
        except Exception as e:
            log.warning(f"  [{name}] explain failed: {type(e).__name__}: {e}")
            per_model[name] = {"model": name, "error": f"{type(e).__name__}: {e}"}

    results["per_model"] = per_model
    n_perm = sum(1 for v in per_model.values() if v.get("permutation"))
    n_native = sum(1 for v in per_model.values() if v.get("native"))

    # summary + report
    summary = {
        "n_models": len(per_model),
        "n_with_permutation": n_perm,
        "n_with_native_shap": n_native,
        "families": {n: v.get("family") for n, v in per_model.items()},
        "mi_top10": [f for f, _ in mi_ranked[:10]],
    }
    (out_root / "_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    md = ["# R11 — SHAP + Feature Importance (all families)", "",
          f"- Models explained: **{len(per_model)}** "
          f"(permutation {n_perm}, native SHAP {n_native})",
          f"- Eval slab: n={len(y_eval)} (held-out test)",
          "", "## Per-model", "",
          "| model | family | permutation | native SHAP | top features |",
          "|-------|--------|-------------|-------------|--------------|"]
    for n, v in per_model.items():
        top = ", ".join((v.get("top_native") or v.get("top_permutation") or [])[:4])
        md.append(f"| {n} | {v.get('family', '?')} | "
                  f"{'✓' if v.get('permutation') else '·'} | "
                  f"{'✓' if v.get('native') else '·'} | {top} |")
    (out_root / "REPORT.md").write_text("\n".join(md))
    results["summary_path"] = str(out_root / "_summary.json")

    elapsed = time.time() - t0
    log.info(f"  ✓ R11 shap complete [{fmt_time(elapsed)}] — "
             f"{n_perm}/{len(per_model)} permutation, {n_native}/{len(per_model)} native")
    return {"feature_importance": results, "elapsed": elapsed}


# back-compat alias (2026-06-02 semantic rename)
run_phase15 = run_shap
