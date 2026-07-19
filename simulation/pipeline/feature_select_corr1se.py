"""Per-model feature 선택 — STABILITY selection(|corr| 재표본 빈도) + nested size-path guard.

(파일명 `corr1se` 는 historical — 옛 |corr| 1-SE size-search 의 흔적. **폐기**: n=242서 per-k OOF
 size-search 는 selection-optimism 으로 신뢰 불가(1-SE 과소적합 / argmin 과적합, 2026-06-01 실측)
 → STABILITY 재표본 빈도로 교체. 옛 derive_k_bounds/pick_k_1se/select_features_corr1se/fixed_epv/
 top_k_indices 는 2026-06-01 제거(codex+Gemini 청소).)

LIVE 인터페이스 (phase13 가 사용):
  - select_features_stability(X_pool, y_pool, ...) : Meinshausen-Bühlmann 재표본 빈도 선택
      (B subsample × 점수 상위 inner_k 빈도 ≥ π). n-adaptive 점수: 작은 n=|corr|(global) /
       n≥epv×p=model importance(per-model). train pool/fold only(누수 0), 결정론(seed).
  - build_nested_size_path / select_size_path_1se : binary{subset,full} 대신 π ladder nested 사다리 +
       Breiman 1-SE/parsimony per-model 선택 (MPH_FEAT_PATH=nested).
  - resolve_feature_path : deep-NN(meta.category=='dl') → binary(작은-fold OOF 불신뢰), 나머지 → nested.
  - feature_guard_keep : subset vs full margin guard (parsimony default).
  - forward_select / backward_select : wrapper — bake-off 비교 전용(production 미사용, 과적합 실증).
  - make_model_importance_fn : massive n 의 model-based stability importance_fn.

Deep module: 작은 인터페이스 + 순수 함수(모델 없이 TDD 가능). ranking=|corr(feature, y)|, 누수 0.
"""
from __future__ import annotations

import numpy as np


def _abs_corr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """|Pearson corr(X[:,j], y)| per feature. 분산 0 feature → 0."""
    y = np.asarray(y, float).ravel()
    p = X.shape[1]
    out = np.zeros(p, float)
    ys = np.std(y)
    if ys <= 1e-12:
        return out
    for j in range(p):
        xj = X[:, j]
        if np.std(xj) > 1e-9:
            c = np.corrcoef(xj, y)[0, 1]
            out[j] = abs(c) if np.isfinite(c) else 0.0
    return out


def select_features_stability(X_pool, y_pool, *, B: int = 50, pi: float = 0.6,
                              epv_ratio: int = 20, seed: int = 42, min_keep: int = 1,
                              importance_fn=None, model_based_min_n: int | None = None,
                              feature_names: list | None = None,
                              mandatory: set | None = None) -> dict:
    """STABILITY SELECTION (Meinshausen & Bühlmann 2010, JRSS-B) — feature optimization 기본.

    재표본 빈도 기반: B번 n/2 subsample 추출 → 각 subsample 의 점수 상위 inner_k 를
    "선택" → 선택 빈도 ≥ π 인 feature 만 keep. 한 OOF split 에 과적합하지 않음(robust) + n 커질수록
    빈도 추정 정밀 → 일반화 강화(scalable). 7-way bake-off + codex/gemini 1위.

    n-adaptive 점수 (C, 사용자 결정 2026-06-01 — "data 작다고 무시 말고 massive 대비"):
        n_pool <  threshold  : score = |corr(feature, y)|        (filter, model-agnostic·global)
        n_pool >= threshold  : score = importance_fn(X_sub,y_sub) (per-model, model-based)
        threshold = model_based_min_n or epv_ratio × p_eff       (Harrell EPV — 도출, 하드코드 n 아님)
    작은 n 에선 model-based wrapper 가 과적합(7-way bake-off 실증) → robust |corr|; n 이 EPV×p 를
    넘으면(massive) per-model importance 가 reliable + 다변량/모델 구조 포착 → 자동 전환.

    n-adaptive size (전부 도출, 하드코드 size 없음):
        subsample 크기 = n_pool // 2          · inner_k = max(1, n_pool // epv_ratio)  (20:1 EPV)
        출력 size      = 빈도 ≥ π feature 수 (데이터가 결정 = dynamic)

    Args:
        X_pool, y_pool: train pool only (누수-safe). y_pool 은 caller 가 원하는 target 공간(log1p 등).
        B: 재표본 수 (Monte-Carlo 정밀도; tuning 아님).
        pi: 안정 임계 빈도 (Meinshausen-Bühlmann 표준 0.6~0.75). 주: 본 함수는 |corr| screen +
            시계열 데이터라 M-B 의 FDR/PFER bound 미적용(다변량·교환가능 가정 위배) —
            "resampling-stabilized marginal screen" 으로 해석(논문 methods 와 일치).
        epv_ratio: inner_k = n//epv_ratio + threshold = epv×p_eff (events-per-variable, Harrell).
        seed: 결정론. min_keep: 빈도 미달 시 최소 보장(상위 빈도).
        importance_fn: optional callable(X_sub, y_sub) -> per-feature 점수(길이 p). None 이거나
            n < threshold 면 미사용(|corr|). massive 에서만 per-model model-based 활성.
            subsample 별 실패(예외/NaN/shape mismatch) → 그 subsample 만 |corr| fallback (crash 0).
        model_based_min_n: model-based 전환 하한. None → epv_ratio × p_eff 도출.
            env MPH_STABILITY_MB_MIN_N 로 override 가능(테스트/강제).

    Returns:
        {selected_indices, stability(빈도 배열), inner_k, pi, B, n_pool,
         mode("corr"|"model_based"), model_based_min_n, p_eff}.
    """
    import os as _os_st
    X = np.asarray(X_pool, float); y = np.asarray(y_pool, float).ravel()
    n, p = X.shape
    p_eff = int(np.sum(np.std(X, axis=0) > 1e-12))                 # non-constant 후보 수
    inner_k = max(1, n // max(1, int(epv_ratio)))
    sub_n = max(4, n // 2)
    # threshold = EPV(events-per-variable, Harrell) × 후보 feature 수 — n 하드코드 아님, 데이터가 결정.
    if model_based_min_n is None:
        model_based_min_n = int(epv_ratio) * max(1, p_eff)
    _env_min = _os_st.environ.get("MPH_STABILITY_MB_MIN_N")
    if _env_min:
        try:
            model_based_min_n = int(_env_min)
        except ValueError:
            pass
    use_model_based = importance_fn is not None and n >= int(model_based_min_n)
    mode = "model_based" if use_model_based else "corr"
    rng = np.random.default_rng(seed)
    freq = np.zeros(p, float)
    for _ in range(int(B)):
        ix = rng.choice(n, size=min(sub_n, n), replace=False)
        if use_model_based:
            try:
                sc = np.asarray(importance_fn(X[ix], y[ix]), float).ravel()
                if sc.shape[0] != p or not np.all(np.isfinite(sc)):
                    sc = _abs_corr(X[ix], y[ix])                  # robust per-subsample fallback
                else:
                    sc = np.abs(sc)                               # importance 크기 (coef 부호 무시)
            except Exception:
                sc = _abs_corr(X[ix], y[ix])
        else:
            sc = _abs_corr(X[ix], y[ix])
        for j in np.argsort(sc)[::-1][:inner_k]:
            freq[j] += 1
    freq /= max(1, int(B))
    sel = [int(j) for j in range(p) if freq[j] >= pi]
    if len(sel) < min_keep:                       # 빈도 미달 → 상위-빈도 min_keep 보장
        sel = sorted(int(j) for j in np.argsort(freq)[::-1][:min_keep])
    # FORCE-INCLUDE mandatory features (e.g. target autoregressive lags ili_rate_lag1-4) by NAME,
    # regardless of |corr| frequency. A pure-|corr| screen can drop the AR backbone (weather/ARI
    # have higher marginal corr), leaving a forecaster with NO recent-ILI signal → negative
    # hold-out R² collapse (verified by v2 retrain recovery). Only force columns present in this
    # X (feature_names aligned to X by caller, post-mc) and non-constant. p_eff·EPV·sel order unchanged.
    n_forced = 0
    if feature_names is not None and mandatory:
        forced = [j for j, nm in enumerate(feature_names)
                  if j < p and nm in mandatory and float(np.std(X[:, j])) > 1e-12]
        new = [j for j in forced if j not in set(sel)]
        if new:
            sel = sorted(set(sel) | set(new))
            n_forced = len(new)
    return {"selected_indices": sorted(sel), "stability": freq, "inner_k": inner_k,
            "pi": pi, "B": int(B), "n_pool": n, "mode": mode, "n_forced_mandatory": n_forced,
            "model_based_min_n": int(model_based_min_n), "p_eff": p_eff}


def derive_min_keep_from_stability(selection_meta: dict, p: int) -> int:
    """Data-derived minimum candidate size from STABILITY's EPV inner-k rule."""
    import os as _os
    p = max(1, int(p))
    override = _os.environ.get("MPH_FEAT_MIN_KEEP") or _os.environ.get("MPH_FEAT_FLOOR")
    if override:
        try:
            return min(p, max(1, int(override)))
        except ValueError:
            pass
    inner_k = int(selection_meta.get("inner_k", 1) or 1)
    forced = int(selection_meta.get("n_forced_mandatory", 0) or 0)
    p_eff = int(selection_meta.get("p_eff", p) or p)
    return min(p, max(1, min(p_eff, max(inner_k, forced))))


def build_nested_size_path(stability_freq, p, *, pi_levels=(0.8, 0.6, 0.4), min_keep=1):
    """ORDERED NESTED size-path of candidate feature-sets from stability frequencies.

    For the per-model 1-SE/parsimony guard (codex+Gemini 2026-06-01): instead of the binary
    {STABILITY subset, full} choice, give each model a small NESTED ladder of subset sizes plus
    full, and let a 1-SE/parsimony rule pick. NESTED (higher π ⊂ lower π ⊂ full) constrains the
    search space → low select-on-OOF optimism at small n; an UNORDERED method menu would overfit
    (forward/backward/binary — bake-off-proven). Sizes are data-driven (emergent from π), not
    hardcoded. Reuses the `stability` frequency vector already computed by select_features_stability
    (no re-resampling).

    Args:
        stability_freq: (p,) per-feature selection frequency (select_features_stability["stability"]).
        p: total feature count (full set = range(p)).
        pi_levels: descending π thresholds → smallest→larger nested subsets (default 0.8/0.6/0.4).
        min_keep: minimum features per candidate (top-frequency fallback if a π level is empty).
    Returns:
        list of sorted index-lists, ascending by size, DEDUPLICATED, full set last. Each is a
        subset of the next (nested). Always ≥1 candidate (full). Caller scores each by OOF-WIS.
    """
    freq = np.asarray(stability_freq, float).ravel()
    p = int(p)
    cands: list[list[int]] = []
    seen: set[tuple] = set()
    for pi in sorted({float(x) for x in pi_levels}, reverse=True):   # high π first = smallest set
        idx = [int(j) for j in range(p) if j < freq.shape[0] and freq[j] >= pi]
        if len(idx) < int(min_keep):                                 # empty level → top-freq fallback
            idx = sorted(int(j) for j in np.argsort(freq)[::-1][:max(1, int(min_keep))])
        key = tuple(sorted(idx))
        if key and key not in seen:
            seen.add(key); cands.append(sorted(idx))
    full = list(range(p))
    if tuple(full) not in seen:
        cands.append(full)
    cands.sort(key=len)                                              # ascending size (nested-preserving)
    return cands


def select_size_path_1se(oof_means, sizes, *, fold_scores=None, margin=0.02, se_mult=1.0):
    """Most-parsimonious candidate within 1-SE (or margin) of the best OOF-WIS (Breiman 1-SE).

    Among candidates whose mean OOF-WIS is within max(se_mult × SE_best, margin × best) of the
    minimum, choose the SMALLEST (fewest features). Controls select-the-min-of-K optimism at
    small n while removing the binary {subset, full} extreme. SE is computed from the BEST
    candidate's per-fold scores when given (true 1-SE); else margin-only parsimony.

    Args:
        oof_means: per-candidate mean OOF-WIS (lower=better); inf/nan tolerated.
        sizes: per-candidate feature count (same order as oof_means).
        fold_scores: optional per-candidate list of per-fold WIS lists (for true 1-SE SE_best).
        margin: relative margin floor (default 0.02 = MPH_FEAT_MARGIN).
        se_mult: SE multiplier (1.0 = classic 1-SE) when fold_scores given.
    Returns:
        int index (into oof_means) of the chosen candidate (smallest within threshold; argmin if none).
    """
    means = np.asarray(oof_means, float).ravel()
    sizes = np.asarray(sizes, float).ravel()
    finite = np.isfinite(means)
    if not np.any(finite):
        return 0
    best = int(np.argmin(np.where(finite, means, np.inf)))
    best_mean = float(means[best])
    se = 0.0
    if fold_scores is not None:
        try:
            fb = np.asarray(fold_scores[best], float).ravel()
            fb = fb[np.isfinite(fb)]
            if fb.size >= 2:
                se = float(np.std(fb, ddof=1) / np.sqrt(fb.size))
        except Exception:
            se = 0.0
    thr = best_mean + max(float(se_mult) * se, abs(float(margin)) * abs(best_mean))
    eligible = [i for i in range(means.shape[0]) if finite[i] and means[i] <= thr + 1e-12]
    if not eligible:
        return best
    return min(eligible, key=lambda i: (sizes[i], means[i]))


# deep-NN families with unreliable small-fold OOF → binary (stability-anchored) not nested.
# 2026-06-02: TimesNet/TiDE/N-HiTS/iTransformer 등 누락 보강 (이름-힌트 backstop).
_DL_NAME_HINTS = ("DNN", "TCN", "TFT", "DeepAR", "N-BEATS", "NBEATS", "NBeats", "NHiTS",
                  "N-HiTS", "PatchTST", "TimesNet", "TiDE", "iTransformer", "Transformer",
                  "LSTM", "GRU", "Mamba", "TimesFM", "TimeFM", "TiRex", "Informer",
                  "Autoformer", "GAT", "GCN")

# deep-NN/foundation families (CATEGORY_MODELS) — robust DL signal (G-250). meta.category 가
# unpopulated 라 family 멤버십이 1차 신호 (2026-06-02 broken-category fix; TimesNet nested 3h 정체 사건).
_DL_FAMILIES = frozenset({"dl-tabular", "modern-ts", "graph", "foundation"})


def _is_dl_family(model_name):
    """model_name 이 deep-NN/foundation family(CATEGORY_MODELS)에 속하면 True.

    meta.category 가 비어있을 때(전 모델 '') family-aware guard 가 무력화되는 회귀 방지 —
    family 멤버십을 직접 조회. lazy import (순환 회피). 미등록/조회실패 → False (보수적 nested 허용).
    """
    if not model_name:
        return False
    try:
        from simulation.models.registry import CATEGORY_MODELS
    except Exception:
        return False
    return any(model_name in members
               for fam, members in CATEGORY_MODELS.items() if fam in _DL_FAMILIES)


def resolve_feature_path(env_path, *, category="", model_name=""):
    """Resolve the per-model feature-guard path, applying the deep-NN family override.

    Deep-NN families (dl-tabular + modern-ts + graph + foundation, G-250) have UNRELIABLE
    small-fold OOF at n≈40/fold: the net underfits so OOF ≫ test (observed: TabularDNN binary
    OOF 6.79 > test 4.74). `nested` SELECTS among sizes BY that unreliable OOF → misfires
    (DNN 실측 nested k=12 test 손해). `binary` ANCHORS on the stability frequency (|corr|-based,
    not OOF-selected) and only sanity-checks vs full → robust. So dl-family uses binary even
    when the env requests nested. (codex+Gemini "regularization-as-selection" + DNN 실측 2026-06-01.)

    DL 판정 = 3 신호 OR: ① category=='dl' ② 이름-힌트 ③ **CATEGORY_MODELS family 멤버십**(_is_dl_family).
    ③ 가 1차 — meta.category 가 전 모델 '' 라 ①은 사실상 죽어있고 ②는 일부 이름 누락
    (2026-06-02: TimesNet/TiDE/N-HiTS 가 nested 로 빠져 phase 13 3h 정체 사건). ③ 으로 근본 차단.

    Args:
        env_path: MPH_FEAT_PATH value ('binary' | 'nested'); anything but 'nested' → 'binary'.
        category: model meta.category (e.g. 'dl', 'tree', 'linear', 'epi'). 보조 신호(현재 unpopulated).
        model_name: model name — 이름-힌트 + family 조회 키.
    Returns:
        'binary' or 'nested'.
    """
    path = (env_path or "binary").strip().lower()
    if path != "nested":
        return "binary"
    cat = (category or "").strip().lower()
    name = model_name or ""
    # G-329e (2026-06-20, 3AI H-1/F-1): DNN-Conformal 은 이름과 달리 closed-form RidgeCV(deep-NN 아님,
    #   conformal.py:632) → small-fold OOF 불신뢰 없음. dl-family binary misroute 시 full-pool(52) 유지
    #   = 과소예측. closed-form allow-list 로 DL 라우팅 제외하고 nested parsimony 적용.
    import os as _os_fp
    _nested_ok = {s.strip() for s in _os_fp.environ.get(
        "MPH_NESTED_LINEAR_MODELS", "DNN-Conformal").split(",") if s.strip()}
    if name in _nested_ok:
        return "nested"
    if (cat == "dl"
            or any(h.lower() in name.lower() for h in _DL_NAME_HINTS)
            or _is_dl_family(name)):
        return "binary"           # deep-NN: small-fold OOF unreliable → stability-anchored binary
    return "nested"


def make_model_importance_fn(factory_fn):
    """적용 모델 기반 per-feature importance 콜백 생성 (model-based stability 의 importance_fn).

    select_features_stability(..., importance_fn=make_model_importance_fn(factory_fn)) 로 주입.
    massive n (n ≥ epv×p) 에서만 subsample 별 호출됨 — 작은 n 은 |corr| 라 미호출(비용 0).
    각 subsample 에 모델을 fit 후 importance 추출 (우선순위):
        1) underlying _model.feature_importances_  (트리·부스팅)
        2) |_model.coef_|                          (선형)
        3) permutation importance (predict 기반)   (DNN·kernel 등 모델-불문)
    어느 단계든 실패/미지원/shape 불일치 → 길이-0 반환 → stability 가 해당 subsample |corr| fallback.

    Args:
        factory_fn: callable() -> BaseForecaster (fresh instance; .fit/.predict, optional ._model).
    Returns:
        callable(X_sub, y_sub) -> np.ndarray (길이 p importance, 또는 길이-0 = fallback 신호).

    Performance: subsample 당 모델 1 fit (+ permutation 경로면 p predict). massive 에서만 활성.
    Side effects: 없음 (각 호출이 독립 fresh 모델, 외부 상태 미변경).
    """
    def _imp(X_sub, y_sub):
        X_sub = np.asarray(X_sub, float); y_sub = np.asarray(y_sub, float).ravel()
        p = X_sub.shape[1]
        try:
            fc = factory_fn()
            fc.fit(X_sub, y_sub)
        except Exception:
            return np.zeros(0)                    # fit 실패 → |corr| fallback
        mdl = getattr(fc, "_model", None)
        if mdl is not None:
            for attr, take_abs in (("feature_importances_", False), ("coef_", True)):
                if hasattr(mdl, attr):
                    try:
                        v = np.asarray(getattr(mdl, attr), float).ravel()
                    except Exception:
                        continue
                    if take_abs:
                        v = np.abs(v)
                    if v.shape[0] == p and np.all(np.isfinite(v)):
                        return v
        try:                                       # permutation importance (모델-불문)
            base = np.asarray(fc.predict(X_sub), float).ravel()
            base_mse = float(np.mean((y_sub - base) ** 2))
            rng = np.random.default_rng(0)
            imp = np.zeros(p, float)
            for j in range(p):
                Xp = X_sub.copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                pm = np.asarray(fc.predict(Xp), float).ravel()
                imp[j] = float(np.mean((y_sub - pm) ** 2)) - base_mse
            return np.clip(imp, 0.0, None)
        except Exception:
            return np.zeros(0)
    return _imp


def feature_guard_keep(oof_full: float, oof_sel: float, rel_margin: float = 0.02,
                       prefer_subset: bool = False) -> bool:
    """feature 선택(subset) vs full feature 유지 결정 — Stage-2 margin-guard.

    사용자 명시(2026-06-01): "각 단계는 이전 단계 대비 개선 보장." + 후속(parsimony): "개선 안 되면
    full(399) 로 복원하는 게 불편 — subset 을 기본 유지하고 싶다." 두 모드:

    prefer_subset=False (STRICT 개선 보장): subset 이 full 대비 OOF-WIS 를 rel_margin 이상 **개선**할
        때만 subset, 아니면 full. (개선 보장; 단 동등시 full 로 복원.)
    prefer_subset=True (PARSIMONY 우선, default in phase13): subset 을 **기본 유지**, subset 이 full
        대비 rel_margin 이상 **명백히 나쁠 때만** full 복원. → 동등/약간나쁨이면 subset (간결·해석성↑,
        full 399 안 쏟아짐). "보장"은 유지(많이 나빠지면 full). Part2 실측: subset 9 ≈ full 399 정확도.

    Args:
        oof_full: full feature OOF-WIS (lower=better). oof_sel: STABILITY subset OOF-WIS.
        rel_margin: 상대 임계 (default 0.02 = 2%). prefer_subset: parsimony 모드 여부.
    Returns:
        True → subset 유지, False → full 복원. 비교 불가(비유한) → False (full, 안전한 완전 set).
    """
    if not (np.isfinite(oof_full) and np.isfinite(oof_sel)):
        return False
    if prefer_subset:
        return float(oof_sel) <= float(oof_full) * (1.0 + float(rel_margin))   # 명백히 나쁘지 않으면 subset
    return float(oof_sel) <= float(oof_full) * (1.0 - float(rel_margin))        # 개선시만 subset


def forward_select(score_fn, candidates, *, k_cap=None, tol: float = 1e-9) -> list:
    """Greedy FORWARD wrapper (쌓기): 빈 집합 → score 가장 낮추는 feature 1개씩 추가.

    개선(score 감소)이 없거나 k_cap 도달 시 정지. model-based: score_fn 이 모델 OOF 를 씀.
    Args:
        score_fn: callable(tuple(sorted idx)) -> float (낮을수록 좋음).
        candidates: 후보 feature index 리스트.
        k_cap: 최대 선택 수 (None = len(candidates)). n-adaptive cap 은 caller 가 전달.
    Returns: 선택된 index (정렬).
    """
    cand = list(candidates); selected: list = []
    cap = k_cap if k_cap is not None else len(cand)
    cur = score_fn(tuple())
    while len(selected) < cap:
        rem = [c for c in cand if c not in selected]
        if not rem:
            break
        s, c = min((score_fn(tuple(sorted(selected + [c]))), c) for c in rem)
        if s < cur - tol:                      # 개선될 때만 추가
            selected.append(c); cur = s
        else:
            break                              # 더 추가해도 개선 없음 → 정지
    return sorted(selected)


def backward_select(score_fn, candidates, *, k_min: int = 1, tol: float = 1e-9) -> list:
    """Greedy BACKWARD wrapper (줄이기): 전체 후보 → 제거해도 안 나빠지는 feature 1개씩 제거.

    제거가 score 를 높이면(나빠지면) 정지, 또는 k_min 도달. model-based reductive.
    "찾아서 줄여나가는" 형식. Args/Returns: forward 와 동일 (k_min = 하한).
    """
    selected = list(candidates)
    if len(selected) <= k_min:
        return sorted(selected)
    cur = score_fn(tuple(sorted(selected)))
    while len(selected) > k_min:
        s, c = min((score_fn(tuple(sorted(x for x in selected if x != c))), c) for c in selected)
        if s <= cur + tol:                     # 제거가 안 나빠지면(또는 개선) 제거
            selected.remove(c); cur = s
        else:
            break                              # 제거하면 나빠짐 → 정지
    return sorted(selected)
