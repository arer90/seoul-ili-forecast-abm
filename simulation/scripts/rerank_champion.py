"""Champion RE-RANK (post-hoc, no re-train) — G-339 LEAK-FREE, hold-out test 미사용.

Background (2026-06-19): live champion = argmin(OOF-WIS) = SVR-RBF won train-pool WF-CV by a
NOISE margin (1.590 vs 1.619) but was WORST of the top group on hold-out (R²=0.865). Under flu-
season distribution shift, OOF-argmin overfits. G-318 (옛) fixed this by picking the hold-out
test-best WITHIN an OOF top-8 shortlist — but that REUSES the test set for a 1-of-8 selection =
winner's curse 재유입 (외부 reviewer #1, 2026-06-24; Cawley & Talbot 2010; Varma & Simon 2006).

G-339 (current) removes test from selection entirely:
  1. OOF 1-SE band  = models within max(SE, 2% margin) of the best OOF-WIS (Breiman 1984 1-SE,
                      top-K cap) = statistically-tied cluster (leak-free).
  2. Tiebreaker     = fold 안정성(_oof_fold_cv = OOF 분포이동 견고성 proxy) → parsimony
                      (n_features) → OOF-WIS. **NO hold-out test.**
  3. Diagnostic     = hold-out test WIS ranking + Diebold-Mariano are DISPLAYED for transparency
                      (did the leak-free champion also do well on the unseen season?) but do NOT
                      drive selection. select_champion_holdout_best = test-best (병기 진단).
  4. Champion       = select_champion_g318 (G-339 leak-free SSOT). There is exactly one champion;
                      NegBinGLM is reported alongside it as the interpretable epidemiological
                      count model (canonical NB for ILI), not as a second champion.

Usage: python -m simulation.scripts.rerank_champion
"""
import glob
import json
import math
import os

import numpy as np

# G-318 SSOT: selection 로직은 파이프라인(per_model_eval)에 단일 정의 — 이 post-hoc CLI 도 동일 함수
# 사용(파이프라인 in-line 챔피언과 100% 일치 보장).
from simulation.pipeline.per_model_eval import (
    select_champion_g318, select_champion_holdout_best, CHAMPION_SHORTLIST_K,
)

D = "simulation/results/per_model_optimal"
SHORTLIST_K = CHAMPION_SHORTLIST_K


def _load():
    # G-351 (2026-06-25, 감사 latent): deprecated(DEFER_MODELS) 모델은 후보 pool 제외 — 디스크에
    #   옛 run JSON 이 남아있어도 active 라인업만 재선정/진단 표에 노출(챔피언 결과 불변, 진단 정직).
    try:
        from simulation.models.registry import DEFER_MODELS as _DEFER
        _defer = set(_DEFER)
    except Exception:
        _defer = set()
    rows = []
    for f in sorted(glob.glob(os.path.join(D, "*.json"))):
        n = os.path.basename(f)[:-5]
        if n == "summary" or n in _defer:
            continue
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        vm = d.get("val_metrics") or {}
        tm = d.get("test_metrics") or {}
        oof = vm.get("oof_wis")
        if not isinstance(oof, (int, float)) or not math.isfinite(oof):
            continue
        rows.append({
            "model": n,
            "oof_wis": float(oof),
            # G-322b/P0-4: 파이프라인 select_champion_g318 과 byte-동일 입력 — fold 벡터·n_features 누락 시
            #   tiebreaker degenerate(inf,inf,oof)로 OOF-argmin fallback → run 과 다른 챔피언(SSOT 분열).
            "oof_wis_folds": vm.get("oof_wis_folds"),
            "n_features": (d.get("best_config") or {}).get("n_features"),
            "test_wis": tm.get("wis"),
            "test_r2": tm.get("r2"),
            "mape": tm.get("mape"),
            "picp95": tm.get("pi95_coverage"),
            "pred": d.get("refit_test_predictions"),
        })
    return rows


# G-322b (2026-06-24): 하드코딩 15 = 버그. test slab 은 series 끝에서 'real 구간' 길이만큼 앞인데
#   real 길이가 run 마다 다름(354주 run = 16; 하드코딩 15 면 1주 어긋나 모든 R² −0.29). 아래는 최후
#   fallback 만 — 실제 offset 은 _resolve_real_horizon 이 split + stored-R² 자가보정으로 유도.
REAL_HORIZON = 15  # fallback only (ranking.json / 자가보정 둘 다 실패 시)


def _resolve_real_horizon(y, n_test, *, calib_pred=None, calib_r2=None):
    """test slab offset(REAL_HORIZON) 을 split 에서 유도 — 하드코딩 금지 (G-322b).

    1순위: ranking.json test_window_idx → ``len(y) - test_window_idx[1] - 1`` (R10 과 동일 split 원천).
    자가보정: calib 모델의 stored test R² 를 **정확 재현**하는 offset 을 13..20 에서 탐색(split 후보
    우선) — 데이터/정렬 변화에도 robust. 실패 시 split 후보, 그것도 없으면 REAL_HORIZON(15) fallback.

    Args:
        y: 전체 series (load_kr_sentinel_ili). n_test: hold-out test 주 수(68).
        calib_pred: 보정용 모델 refit_test_predictions. calib_r2: 그 모델 stored test_metrics.r2.
    Returns:
        int — test slab = ``y[-(n_test+H):-H]``.
    """
    import os
    cand = None
    rk = "simulation/results/per_model_eval/ranking.json"
    if os.path.exists(rk):
        try:
            twi = (json.load(open(rk, encoding="utf-8")).get("test_window_idx") or [])
            if len(twi) == 2 and isinstance(twi[1], (int, float)):
                cand = int(len(y) - int(twi[1]) - 1)
        except Exception:
            cand = None
    # 자가보정: calib 모델 stored R² 를 정확 재현하는 offset (split 후보 우선)
    if calib_pred is not None and isinstance(calib_r2, (int, float)) and np.isfinite(calib_r2):
        p = np.asarray(calib_pred, dtype=float)
        order = ([cand] if cand else []) + [h for h in range(13, 21) if h != cand]
        for H in order:
            if H is None or H <= 0 or len(y) < n_test + H:
                continue
            yt = y[-(n_test + H):-H]
            if len(yt) != n_test:
                continue
            denom = float(((yt - yt.mean()) ** 2).sum())
            if denom <= 0:
                continue
            r2 = 1.0 - float(((yt - p[-n_test:]) ** 2).sum()) / denom
            if abs(r2 - float(calib_r2)) < 1e-3:
                return int(H)
    return int(cand) if cand is not None else REAL_HORIZON


def _y_test(n_test, *, calib_pred=None, calib_r2=None):
    """Reconstruct the hold-out TEST target = the n_test weeks BEFORE the trailing real slab.

    test slab = ``y_full[-(n_test+H):-H]``, H=REAL_HORIZON(test 뒤 real 구간 길이) — **split 에서
    유도**(_resolve_real_horizon), 하드코딩 아님. 354주 run = H=16(stored R² 정확 재현 검증).

    Args:
        n_test: hold-out test 주 수(68).
        calib_pred/calib_r2: offset 자가보정용 (한 모델 pred + stored R²) — 없으면 split-only.

    Returns:
        np.ndarray length n_test, 또는 None(series 짧음/실패).

    Verified: H=16 이 FusedEpi(0.936)/TabPFN(0.861)/ARIMA(0.924)/SeirCount(0.932) stored R² 를
    0.0000 diff 재현 (tests/test_rerank_y_test.py). 옛 하드코딩 15 는 1주 어긋나 R² −0.29.
    """
    try:
        from simulation.database import safe_connect
        from simulation.pipeline.true_ili_cohort import load_kr_sentinel_ili
        con = safe_connect("simulation/data/db/epi_real_seoul.db")
        rows = load_kr_sentinel_ili(con)
        con.close()
        y = np.asarray([v for _, _, v in rows], dtype=float)
        H = _resolve_real_horizon(y, n_test, calib_pred=calib_pred, calib_r2=calib_r2)
        if H <= 0 or len(y) < n_test + H:
            return None
        return y[-(n_test + H):-H]
    except Exception as e:  # noqa: BLE001
        print(f"  (y_test 재구성 실패 → DM 생략: {type(e).__name__}: {e})")
        return None


def _dm(loss_a, loss_b):
    """Diebold-Mariano (1-step, two-sided) on two per-period loss vectors. Returns p-value."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    n = len(d)
    if n < 5 or np.allclose(d, 0):
        return float("nan")
    dbar = d.mean()
    var = d.var(ddof=1)
    if var <= 0:
        return float("nan")
    dm = dbar / math.sqrt(var / n)
    # two-sided normal approx
    from math import erf
    return 2 * (1 - 0.5 * (1 + erf(abs(dm) / math.sqrt(2))))


def main():
    rows = _load()
    rows_oof = sorted(rows, key=lambda r: r["oof_wis"])
    shortlist = rows_oof[:SHORTLIST_K]
    # G-322b: REAL_HORIZON 자가보정용 ref 모델(유한 stored R² + 68-len pred) — split offset 검증
    _cal = next((r for r in rows if r.get("pred") and isinstance(r.get("test_r2"), (int, float))
                 and np.isfinite(r["test_r2"]) and len(r["pred"]) >= 68), None)
    yt = _y_test(68, calib_pred=(_cal["pred"] if _cal else None),
                 calib_r2=(_cal["test_r2"] if _cal else None))

    print("=" * 78)
    print("CHAMPION RE-RANK — OOF shortlist → hold-out generalization + DM significance")
    print("=" * 78)
    print(f"\n[1] OOF shortlist (top-{SHORTLIST_K} by oof_wis, eligible cluster):")
    print(f"  {'model':18}{'oof_wis':>9}{'test_wis':>10}{'test_r2':>9}{'mape':>7}{'picp95':>8}")
    for r in shortlist:
        tw = f"{r['test_wis']:.3f}" if isinstance(r['test_wis'], (int, float)) else "-"
        tr = f"{r['test_r2']:.3f}" if isinstance(r['test_r2'], (int, float)) else "-"
        mp = f"{r['mape']:.1f}" if isinstance(r['mape'], (int, float)) else "-"
        pc = f"{r['picp95']:.3f}" if isinstance(r['picp95'], (int, float)) else "-"
        print(f"  {r['model']:18}{r['oof_wis']:>9.3f}{tw:>10}{tr:>9}{mp:>7}{pc:>8}")

    # [2] DIAGNOSTIC ONLY (G-339): hold-out test WIS 순위 — 선정엔 미사용, '챔피언이 미관측
    #     시즌서도 잘했나?' 투명성 표시. 실제 선정 = OOF 1-SE band + fold안정성/parsimony(leak-free).
    gen = sorted([r for r in shortlist if isinstance(r["test_wis"], (int, float))],
                 key=lambda r: r["test_wis"])
    print(f"\n[2] (진단 표시·선정 미사용) hold-out test WIS 순위 = 미관측 시즌 성능 점검:")
    for i, r in enumerate(gen, 1):
        print(f"  {i}. {r['model']:18} test_wis={r['test_wis']:.3f}  test_r2={r['test_r2']:.3f}")

    # [3] DM significance among shortlist vs the generalization-best
    if yt is not None and gen:
        best = gen[0]
        bp = np.asarray(best["pred"], dtype=float) if best.get("pred") else None
        if bp is not None and len(bp) == len(yt):
            loss_best = (bp - yt) ** 2
            print(f"\n[3] DM 유의성 (vs 일반화 1위 {best['model']}, hold-out 제곱오차, p<0.05=유의차):")
            for r in gen[1:6]:
                p = r.get("pred")
                if p and len(p) == len(yt):
                    pv = _dm((np.asarray(p, dtype=float) - yt) ** 2, loss_best)
                    tie = "동률(노이즈)" if not (pv < 0.05) else "유의하게 다름"
                    print(f"  {best['model']} vs {r['model']:16} DM p={pv:.3f}  → {tie}")
        else:
            print("\n[3] DM 생략 (예측 길이 불일치)")
    else:
        print("\n[3] DM 생략 (y_test 없음) — test_wis 근접도로 판단")

    # [4] verdict — 챔피언은 파이프라인과 동일 SSOT(select_champion_g318)로 결정(gen[0] 과 동일하나
    #     in-line 챔피언과 100% 일치 보장).
    _champ_row = select_champion_g318(rows)
    _champ_ho = select_champion_holdout_best(rows)
    champ = _champ_row["model"] if _champ_row else "?"
    champ_ho = _champ_ho["model"] if _champ_ho else "?"
    _agree = (champ == champ_ho)
    print(f"\n[4] 결론 (G-339 leak-free 배포 + hold-out best 진단 병기):")
    print(f"  • G-339 챔피언(배포)  = {champ} (OOF 1-SE band 내 fold안정성·parsimony, **test 미사용**)")
    print(f"  • hold-out best(진단) = {champ_ho} — "
          f"{'G-339와 일치 → 강한 증거(leak-free 선정이 미관측 시즌서도 1위)' if _agree else 'G-339와 다름 → 둘 다 투명 보고(test는 선정 미참여)'}")
    print(f"  • epi 해석가능 count 모델(병기) = NegBinGLM (정식 NB count 모델) — 챔피언 아님")


if __name__ == "__main__":
    main()
