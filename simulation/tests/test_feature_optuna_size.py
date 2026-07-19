"""feature-optuna 메커니즘 TDD — "feature optuna를 어떻게 하나" 를 코드로 못박음.

설계 확정: feature-optuna = **|corr(y)| top-k 의 SIZE k 를 작은 그리드에서 OOF-WIS 로 탐색**
(binary mask 2^k 아님). margin-guard: 기본 k 대비 ≥margin 상대개선 시에만 deviate.

이 파일은 두 층을 검증:
  1. 선택 로직 `feature_optuna_size(scores_oof, default_k, margin)` (순수, 합성 — 빠름)
  2. 실데이터 OOF 채점 (ElasticNet — OpenMP-safe 선형) → OOF-WIS(k) 곡선 + 고른 k 출력

run: KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 .venv/bin/python -m pytest <this> -q -s
"""
import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore")
CAND_KS = [8, 12, 15, 20, 25]
DEFAULT_K = 12
MARGIN = 0.02


def feature_optuna_size(scores_oof: dict, default_k: int = DEFAULT_K, margin: float = MARGIN) -> int:
    """OOF-WIS(k) 점수에서 feature-subset SIZE 선택 (margin-guarded).

    Args:
        scores_oof: {k: OOF-WIS} (낮을수록 좋음). k = |corr| top-k 의 크기.
        default_k: parsimony 기준 크기 (이만큼 안 좋으면 default 유지).
        margin: default 대비 최소 상대 개선 (이상이어야 deviate).
    Returns:
        선택된 k (binary mask 아님 — 크기 1개).
    """
    if not scores_oof:
        return default_k
    finite = {k: v for k, v in scores_oof.items() if np.isfinite(v)}
    if not finite:
        return default_k
    base_k = default_k if default_k in finite else min(finite, key=finite.get)
    best_k = min(finite, key=finite.get)
    base = finite[base_k]
    improve = (base - finite[best_k]) / abs(base) if base not in (0, None) else 0.0
    return best_k if improve >= margin else base_k


# ── 1. 선택 로직 (합성, 빠름) ──────────────────────────────────────────
def test_picks_best_when_clearly_better():
    # k=20 이 0.50 으로 default(12)=0.80 보다 37% 개선 → deviate
    assert feature_optuna_size({8: 1.0, 12: 0.80, 20: 0.50}, 12, 0.02) == 20


def test_margin_guard_keeps_default_when_marginal():
    # k=15 가 0.79 로 default(12)=0.80 보다 1.25% (<2%) → default 유지 (overfit 가드)
    assert feature_optuna_size({8: 0.81, 12: 0.80, 15: 0.79}, 12, 0.02) == 12


def test_search_space_is_size_not_mask():
    # 후보가 SIZE 정수들 (작은 그리드) — 2^k binary mask 가 아님
    assert all(isinstance(k, int) for k in CAND_KS)
    assert len(CAND_KS) <= 8, "size 그리드는 작아야 (binary mask 폭발 방지)"


def test_deterministic():
    s = {8: 0.9, 12: 0.7, 15: 0.72}
    assert feature_optuna_size(s, 12, 0.02) == feature_optuna_size(dict(s), 12, 0.02)


def test_nan_scores_fall_back_to_default():
    assert feature_optuna_size({8: float("nan"), 12: float("nan")}, 12, 0.02) == 12


# ── 2. 실데이터 OOF 채점 (ElasticNet, OpenMP-safe) — 메커니즘 시연 ──────
def _oof_wis_at_k(fac, Pp, yp, ylog, inv, corr_order, k, n_folds=3):
    from simulation.tests._real_data_prep import _ewis
    idx = sorted(corr_order[:k]); n = len(yp); fs = n // (n_folds + 1); ws = []
    for f in range(1, n_folds + 1):
        etr = f * fs; eva = (f + 1) * fs if f < n_folds else n
        if eva - etr < 4:
            continue
        try:
            m = fac(); m.fit(Pp[:etr][:, idx], ylog[:etr])
            pe = inv(m.predict(Pp[etr:eva][:, idx])); rs = yp[:etr] - inv(m.predict(Pp[:etr][:, idx]))
            w = float(np.median(_ewis(yp[etr:eva], pe, rs)))
            if np.isfinite(w):
                ws.append(w)
        except Exception:
            continue
    return float(np.median(ws)) if ws else float("inf")


def test_real_oof_scoring_elasticnet(capsys):
    """ElasticNet: 각 k 의 OOF-WIS 를 실제 계산 → feature_optuna_size 가 k 선택. 메커니즘 시연."""
    from simulation.tests._real_data_prep import _prep_full
    from simulation.models.registry import verify_registry_coverage
    verify_registry_coverage(force_import=True)
    from simulation.models.base import REGISTRY
    Pp, Pt, yp, yt, ylog, inv, cols = _prep_full()
    yt_ = ylog.ravel()
    sc = np.array([abs(np.corrcoef(Pp[:, j], yt_)[0, 1]) if np.std(Pp[:, j]) > 1e-9 else 0.0
                   for j in range(Pp.shape[1])]); sc[~np.isfinite(sc)] = 0.0
    corr_order = list(np.argsort(sc)[::-1])
    fac = (lambda: REGISTRY.get("ElasticNet")())
    scores = {k: _oof_wis_at_k(fac, Pp, yp, ylog, inv, corr_order, k) for k in CAND_KS}
    picked = feature_optuna_size(scores, DEFAULT_K, MARGIN)
    print("\n  [feature-optuna 시연: ElasticNet]")
    for k in CAND_KS:
        print(f"    k={k:>3}: OOF-WIS={scores[k]:.4f}{'  ← argmin' if scores[k]==min(scores.values()) else ''}"
              f"{'  ← picked' if k==picked else ''}")
    print(f"    → feature-optuna 선택 k={picked} (default={DEFAULT_K}, margin={MARGIN})")
    assert picked in CAND_KS
    assert all(np.isfinite(v) for v in scores.values()), "OOF-WIS 계산 실패"
