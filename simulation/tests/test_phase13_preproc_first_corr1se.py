"""B1 wiring 검증 — phase13 가 preproc-first + STABILITY feature optimization 을 쓰는지.

(1) MPH_PREPROC_FIRST default=1 + select_features_stability + log1p 배선 (소스).
    (7-way bake-off + codex/gemini 1위 = STABILITY; size-search/binary/forward 능가, robust+scalable.)
(2) 실데이터 통합: select_features_stability(X, log1p y) → 유효 dynamic 선택 (n-adaptive inner_k).

run: KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 .venv/bin/python -m pytest <this> -q -s
"""
import inspect

import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore")


def test_preproc_first_default_on_and_wired():
    import simulation.pipeline.per_model_optimize as m
    src = inspect.getsource(m.optimize_one_model)
    assert 'MPH_PREPROC_FIRST", "1"' in src, "preproc-first 가 default ON(=1) 이어야"
    assert "select_features_stability" in src, "STABILITY feature-opt 가 배선돼야"
    assert "make_model_importance_fn" in src, "n-adaptive model-based importance_fn 가 배선돼야"
    assert "feature_guard_keep" in src, "Stage-2 margin-guard(이전 대비 개선 보장) 가 배선돼야"
    assert "np.log1p" in src, "feature 랭킹은 log1p(y) 공간 (count-like ILI)"
    assert "select_features_fixed_epv" not in src, "fixed-EPV 는 stability 로 교체돼야"


def test_stability_integration_real_data():
    from simulation.tests._real_data_prep import _prep_full
    from simulation.pipeline.feature_select_corr1se import select_features_stability

    Pp, Pt, yp, yt, ylog, inv, cols = _prep_full()
    _ylog = np.log1p(np.clip(yp, 0, None))
    called = {"n": 0}
    def _spy_imp(Xs, ys):                         # phase13 가 넘기는 importance_fn 모사
        called["n"] += 1
        return np.zeros(Xs.shape[1])
    sel = select_features_stability(Pp, _ylog, B=40, pi=0.6, epv_ratio=20, seed=42,
                                    importance_fn=_spy_imp)
    n = sel["n_pool"]
    print(f"\n  stability: selected={len(sel['selected_indices'])}  inner_k={sel['inner_k']}  "
          f"pi={sel['pi']}  B={sel['B']}  n_pool={n}  mode={sel['mode']}  "
          f"mb_min_n={sel['model_based_min_n']}")
    # n-adaptive (C): 실데이터 n (≈242) < epv×p → corr 모드, model importance 미호출 (현 동작 보존)
    assert sel["mode"] == "corr", f"실데이터 n 은 corr 모드여야 (< epv×p): mode={sel['mode']}"
    assert called["n"] == 0, "작은 n 에서 model importance 호출되면 안 됨 (비용 0, regression 0)"
    assert sel["model_based_min_n"] == 20 * sel["p_eff"], "threshold 가 epv×p_eff 도출 아님"
    assert sel["inner_k"] == max(1, n // 20), "inner_k 가 n//20 (derived) 와 불일치"
    assert 1 <= len(sel["selected_indices"]) <= Pp.shape[1], "선택 수 범위 밖"
    assert sel["selected_indices"] == sorted(sel["selected_indices"])
    # dynamic: 빈도 배열 길이 = feature 수
    assert len(sel["stability"]) == Pp.shape[1]
