"""Multicollinearity filter (mc, R9 per_model_optimize 내부 단계) — 4 methods (D/A/B/C 비교 가능).

D-1 사용자 명시 (2026-05-22): "코드에 a,b,c,d를 다 해줘. 그러면 구별이 될 것 같아.
phase A를 전체에" → 4 method 모두 구현 + env var 로 선택 + 결과 비교 가능.

## Methods

- **D = "none"**: passthrough (baseline, 현 상태 — Stage 2 Optuna parsimony 만)
- **A = "vif"**: VIF iterative drop (threshold=10, Belsley/Kuh/Welsch 1980)
- **B = "corr"**: Pairwise |corr| > 0.9 + MI tie-break (Dormann 2013 / Guyon 2003)
- **C = "pca"**: PCA orthogonalize (variance retained = 95%, Jolliffe 2002)

## Selection
env var ``MPH_MULTICOLLINEARITY`` ∈ {"none", "vif", "corr", "pca"} (default "none").

## Output contract (D-5 gray-box)
모든 method 가 같은 tuple 반환:
    (X_train_f, X_val_f, X_test_f, kept_indices_or_pca_info, metadata)
- X_*_f: shape preserved (rows × n_kept_features), method=pca 시는 components
- kept_indices: list[int] (method=vif/corr 시 원본 column index), None (none/pca)
- metadata: dict — method, n_kept, n_dropped, runtime_s, drop_reasons (top 10)

## Side effects
- None (pure function, idempotent with seed=42)
- 모든 method 가 train 만 fit, val/test 는 transform (no leakage)
"""
from __future__ import annotations
import time
import os
import json
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, List
from simulation.config_global import GLOBAL  # SSOT (2026-05-28)

import logging
log = logging.getLogger(__name__)

# Method registry
_VALID_METHODS = {"none", "vif", "corr", "pca"}

# Cutoffs (literature-grounded; env var override 가능)
def _get_vif_threshold() -> float:
    return GLOBAL.training.vif_threshold

def _get_corr_threshold() -> float:
    return GLOBAL.training.corr_threshold

def _get_pca_variance() -> float:
    return GLOBAL.training.pca_variance

def _get_vif_max_iter() -> int:
    return GLOBAL.training.vif_max_iter


# ── VIF keep-index cache (2026-06-03) ───────────────────────────────────────
#: VIF keep_idx 는 X_train (+threshold/max_iter) 에만 의존 — 모델과 무관.
#: R9(per_model_optimize) G-242 per-model mc probe 가 _method_vif 를 동일 X_train 으로 ~53× 재호출
#: (53 models × {2 OOF folds + 1 insample} ≈ 159 calls, distinct X_train 은 ~3개) →
#: ~156 회 redundant O(iter·p²·n) (~68s/회 ≈ 2h 낭비). X_train 지문으로 keep_idx memoize.
#: 결정적이라 cached 결과 = fresh 와 bit-identical (재현성 #5). 프로세스 수명 한정(run 마다 fresh).
_VIF_KEEP_CACHE: "dict[str, tuple[list, list]]" = {}
_VIF_CACHE_MAX = 64


# ── VIF L2 disk cache (2026-06-11, G-251) ───────────────────────────────────
# G-236/G-249 isolated the R9(per_model_optimize) per-model probe into one subprocess PER MODEL.
# That broke the in-memory _VIF_KEEP_CACHE's cross-model dedup (each child starts
# with an empty dict) → VIF recomputed 49×2 = 98× instead of 2-3× → +6h. VIF
# keep_idx depends ONLY on X_train (model-independent), so persist it to disk keyed
# by the same fingerprint: the first child computes a fold's VIF, every later child
# (different process) reads it back. Identical results, ~98× → ~3× compute.
def _vif_disk_dir() -> Path:
    """Cross-process L2 cache dir for VIF keep-indices. Honors MPH_OUTPUT_ROOT (G-150)."""
    root = os.environ.get("MPH_OUTPUT_ROOT", "")
    base = Path(root) / "cache" if root else Path(__file__).resolve().parent.parent / "cache"
    d = base / "vif_keep"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _vif_disk_load(fp: str) -> "Optional[tuple[list, list]]":
    """Return (keep_idx, drop_log) from the L2 disk cache, or None on miss/error."""
    try:
        p = _vif_disk_dir() / f"{fp}.json"
        if p.exists():
            obj = json.loads(p.read_text(encoding="utf-8"))
            return (list(obj["keep_idx"]), [tuple(t) for t in obj["drop_log"]])
    except Exception:
        pass
    return None


def _vif_disk_save(fp: str, keep_idx: list, drop_log: list) -> None:
    """Atomically persist (keep_idx, drop_log) to the L2 disk cache (best-effort)."""
    try:
        p = _vif_disk_dir() / f"{fp}.json"
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"keep_idx": [int(i) for i in keep_idx],
                                   "drop_log": [[str(c), float(v)] for c, v in drop_log]}),
                       encoding="utf-8")
        os.replace(tmp, p)            # atomic on POSIX
    except Exception:
        pass


def _vif_fingerprint(X_train: np.ndarray, threshold: float, max_iter: int) -> str:
    """Content fingerprint of the VIF inputs (X_train + cutoffs).

    Args:
        X_train: train feature matrix (n × p). 내용 전체가 키.
        threshold / max_iter: VIF 컷오프 (keep_idx 에 영향 → 키에 포함).

    Returns:
        sha1 hex. O(n·p) 해시 (~1ms / 349×400) ≪ VIF O(iter·p²·n) (~68s) — 큰 skip 위한 싼 키.
    """
    import hashlib
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(X_train, dtype=np.float64).tobytes())
    h.update(f"|{X_train.shape}|{threshold}|{max_iter}".encode())
    return h.hexdigest()


# ════════════════════════════════════════════════════════════════
# Method D: none (passthrough baseline)
# ════════════════════════════════════════════════════════════════
def _method_none(X_train, X_val, X_test, y_train, feature_cols):
    """Method D: passthrough. Baseline 으로 사용 (Stage 2 Optuna parsimony 만)."""
    t0 = time.time()
    return (
        X_train, X_val, X_test, None,
        {
            "method": "none",
            "n_kept": X_train.shape[1],
            "n_dropped": 0,
            "runtime_s": round(time.time() - t0, 4),
            "drop_reasons": [],
            "description": "passthrough — Stage 2 Optuna parsimony 만 사용",
        },
    )


# ════════════════════════════════════════════════════════════════
# Method A: VIF iterative drop
# ════════════════════════════════════════════════════════════════
def _method_vif(X_train, X_val, X_test, y_train, feature_cols):
    """Method A: VIF > 10 iterative drop.

    Train 만 fit (variance + collinearity 계산). Val/test 는 같은 column index drop.
    """
    t0 = time.time()
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from sklearn.preprocessing import StandardScaler

    threshold = _get_vif_threshold()
    max_iter = _get_vif_max_iter()

    # ── cache hit: 동일 X_train (G-242 probe 재호출) → O(iter·p²·n) 루프 skip ──
    _fp = _vif_fingerprint(X_train, threshold, max_iter)
    _hit = _VIF_KEEP_CACHE.get(_fp)
    if _hit is None:                       # L1 (memory) miss → L2 (disk, cross-process G-251)
        _hit = _vif_disk_load(_fp)
        if _hit is not None:
            _VIF_KEEP_CACHE[_fp] = _hit
    if _hit is not None:
        keep_idx, drop_log = list(_hit[0]), list(_hit[1])
        elapsed = time.time() - t0
        log.info(f"  [phase0.5/vif] {X_train.shape[1]} → {len(keep_idx)} "
                 f"(dropped {len(drop_log)}, cached {elapsed:.2f}s)")
        return (
            X_train[:, keep_idx], X_val[:, keep_idx], X_test[:, keep_idx], keep_idx,
            {
                "method": "vif",
                "threshold": threshold,
                "n_kept": len(keep_idx),
                "n_dropped": len(drop_log),
                "runtime_s": round(elapsed, 4),
                "drop_reasons": [{"col": c, "vif": float(v)} for c, v in drop_log[:10]],
                "kept_indices": keep_idx,
                "description": f"VIF > {threshold} iterative drop (Belsley/Kuh/Welsch 1980)",
                "cached": True,
            },
        )

    # Train 만 standardize (VIF 는 scale 민감)
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)

    keep_idx = list(range(X_train.shape[1]))
    drop_log: List[Tuple[str, float]] = []

    for it in range(max_iter):
        if len(keep_idx) < 2:
            break
        X_sub = X_train_std[:, keep_idx]
        try:
            vifs = np.array([variance_inflation_factor(X_sub, i)
                              for i in range(X_sub.shape[1])])
        except Exception as e:
            log.warning(f"  [phase0.5/vif] iter {it} VIF 실패: {e}")
            break
        vifs = np.where(np.isfinite(vifs), vifs, 1e6)
        max_vif = float(vifs.max())
        if max_vif <= threshold:
            break
        drop_pos = int(np.argmax(vifs))
        drop_col_idx = keep_idx[drop_pos]
        col_name = (feature_cols[drop_col_idx] if feature_cols
                    else f"col_{drop_col_idx}")
        drop_log.append((col_name, max_vif))
        keep_idx.pop(drop_pos)

    # store keep_idx for identical-X_train re-calls (G-242 probe dedup; FIFO cap)
    if len(_VIF_KEEP_CACHE) >= _VIF_CACHE_MAX:
        _VIF_KEEP_CACHE.pop(next(iter(_VIF_KEEP_CACHE)))
    _VIF_KEEP_CACHE[_fp] = (list(keep_idx), list(drop_log))
    _vif_disk_save(_fp, keep_idx, drop_log)   # L2 cross-process persist (G-251)

    # Apply same keep_idx to train/val/test
    X_train_f = X_train[:, keep_idx]
    X_val_f = X_val[:, keep_idx]
    X_test_f = X_test[:, keep_idx]

    elapsed = time.time() - t0
    log.info(f"  [phase0.5/vif] {X_train.shape[1]} → {len(keep_idx)} "
             f"(dropped {len(drop_log)}, {elapsed:.1f}s)")

    return (
        X_train_f, X_val_f, X_test_f, keep_idx,
        {
            "method": "vif",
            "threshold": threshold,
            "n_kept": len(keep_idx),
            "n_dropped": len(drop_log),
            "runtime_s": round(elapsed, 4),
            "drop_reasons": [{"col": c, "vif": float(v)} for c, v in drop_log[:10]],
            "kept_indices": keep_idx,
            "description": f"VIF > {threshold} iterative drop (Belsley/Kuh/Welsch 1980)",
        },
    )


# ════════════════════════════════════════════════════════════════
# Method B: Pairwise |corr| + MI tie-break
# ════════════════════════════════════════════════════════════════
def _method_corr_mi(X_train, X_val, X_test, y_train, feature_cols):
    """Method B: |corr| > 0.9 인 쌍에서 MI 낮은 쪽 drop.

    Train 만 fit. MI seed=42 고정 (재현성).
    """
    t0 = time.time()
    from sklearn.feature_selection import mutual_info_regression

    threshold = _get_corr_threshold()

    # Corr matrix on train only
    corr_mat = np.corrcoef(X_train.T)
    corr_mat = np.where(np.isfinite(corr_mat), corr_mat, 0.0)
    abs_corr = np.abs(corr_mat)
    np.fill_diagonal(abs_corr, 0.0)

    # MI for tie-break (train only)
    mi = mutual_info_regression(X_train, y_train, random_state=42, n_neighbors=3)

    keep_idx = list(range(X_train.shape[1]))
    drop_log: List[Tuple[str, float, float, str, float]] = []

    while True:
        sub_corr = abs_corr[np.ix_(keep_idx, keep_idx)]
        if sub_corr.size == 0 or sub_corr.max() <= threshold:
            break
        i_sub, j_sub = np.unravel_index(int(sub_corr.argmax()), sub_corr.shape)
        if i_sub == j_sub:
            break
        i, j = keep_idx[i_sub], keep_idx[j_sub]
        # MI 낮은 쪽 drop
        if mi[i] < mi[j]:
            drop, keep = i, j
        else:
            drop, keep = j, i
        col_drop = feature_cols[drop] if feature_cols else f"col_{drop}"
        col_keep = feature_cols[keep] if feature_cols else f"col_{keep}"
        drop_log.append((col_drop, float(sub_corr.max()),
                          float(mi[drop]), col_keep, float(mi[keep])))
        keep_idx.remove(drop)

    X_train_f = X_train[:, keep_idx]
    X_val_f = X_val[:, keep_idx]
    X_test_f = X_test[:, keep_idx]

    elapsed = time.time() - t0
    log.info(f"  [phase0.5/corr] {X_train.shape[1]} → {len(keep_idx)} "
             f"(dropped {len(drop_log)}, {elapsed:.2f}s)")

    return (
        X_train_f, X_val_f, X_test_f, keep_idx,
        {
            "method": "corr",
            "threshold": threshold,
            "n_kept": len(keep_idx),
            "n_dropped": len(drop_log),
            "runtime_s": round(elapsed, 4),
            "drop_reasons": [
                {"drop": c, "corr": cr, "mi_drop": md,
                 "kept": k, "mi_kept": mk}
                for c, cr, md, k, mk in drop_log[:10]
            ],
            "kept_indices": keep_idx,
            "description": f"|corr| > {threshold} + MI tie-break (Dormann 2013, Guyon 2003)",
        },
    )


# ════════════════════════════════════════════════════════════════
# Method C: PCA orthogonalize
# ════════════════════════════════════════════════════════════════
def _method_pca(X_train, X_val, X_test, y_train, feature_cols):
    """Method C: PCA(n_components=0.95 variance) orthogonalize.

    Train 만 fit. Val/test 는 같은 PCA basis 로 transform.
    주의: 결과 X 는 PC1~PCN (column 이름 사라짐) — SHAP/PDP 해석 불가.
    """
    t0 = time.time()
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    variance = _get_pca_variance()

    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_val_std = sc.transform(X_val)
    X_test_std = sc.transform(X_test)

    pca = PCA(n_components=variance, random_state=42)
    X_train_pca = pca.fit_transform(X_train_std)
    X_val_pca = pca.transform(X_val_std)
    X_test_pca = pca.transform(X_test_std)

    n_comp = int(pca.n_components_)
    total_var = float(pca.explained_variance_ratio_.sum())

    elapsed = time.time() - t0
    log.info(f"  [phase0.5/pca] {X_train.shape[1]} features → "
             f"{n_comp} components ({total_var*100:.1f}% variance, {elapsed:.2f}s)")

    return (
        X_train_pca, X_val_pca, X_test_pca,
        {"scaler": sc, "pca": pca},   # Pinf(inference, 구 phase14) replay 용
        {
            "method": "pca",
            "variance_retained": variance,
            "n_kept": n_comp,
            "n_dropped": X_train.shape[1] - n_comp,
            "runtime_s": round(elapsed, 4),
            "drop_reasons": [
                {"PC": i + 1, "var_ratio": float(pca.explained_variance_ratio_[i])}
                for i in range(min(10, n_comp))
            ],
            "top5_var_sum": float(pca.explained_variance_ratio_[:5].sum()),
            "description": f"PCA {variance*100:.0f}% variance retained (Jolliffe 2002)",
        },
    )


# ════════════════════════════════════════════════════════════════
# Public API: selector
# ════════════════════════════════════════════════════════════════
_METHOD_FNS = {
    "none": _method_none,
    "vif": _method_vif,
    "corr": _method_corr_mi,
    "pca": _method_pca,
}


def apply_multicollinearity_filter(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    feature_cols: Optional[List[str]] = None,
    method: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, object, Dict]:
    """Multicollinearity filter selector — env var 우선, fallback 'none'.

    Args:
        X_train: (n_train, n_features) train slab
        X_val: (n_val, n_features) val slab (transform only)
        X_test: (n_test, n_features) test slab (transform only)
        y_train: (n_train,) target — MI tie-break (method=corr) 용
        feature_cols: feature 이름 list (drop reason 로깅용)
        method: ∈ {"none", "vif", "corr", "pca"} — None 이면 env var 사용

    Returns:
        (X_train_f, X_val_f, X_test_f, transformer_state, metadata)
        - method=none: transformer_state=None
        - method=vif/corr: transformer_state=list[int] (kept column indices)
        - method=pca: transformer_state={"scaler": ..., "pca": ...}

    Raises:
        ValueError: method 잘못된 경우

    Performance: vif ~25s, corr ~0.25s, pca ~0.01s for (200, 250) input
    Side effects: None (pure)
    """
    if method is None:
        from simulation.config_global import GLOBAL as _GCFG  # SSOT (2026-05-28)
        method = _GCFG.training.multicollinearity
    if method not in _VALID_METHODS:
        raise ValueError(f"Unknown method '{method}'. Valid: {_VALID_METHODS}")

    # Shape sanity
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1], \
        f"feature count mismatch: train={X_train.shape[1]}, val={X_val.shape[1]}, test={X_test.shape[1]}"
    assert X_train.shape[0] == y_train.shape[0], \
        f"X_train ({X_train.shape[0]}) vs y_train ({y_train.shape[0]}) row mismatch"

    fn = _METHOD_FNS[method]
    return fn(X_train, X_val, X_test, y_train, feature_cols)


__all__ = ["apply_multicollinearity_filter"]
