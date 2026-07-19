"""compute_wis_ssot.py — 재현가능 WIS SSOT (S3 옵션 C, 2026-06-27).

논문(docx)이 세 종류의 WIS 슬랩을 혼용해 왔다:
  ① Table1 챔피언 = R10 ``wis`` (hold-out test, leak-free residual, adaptive conformal)
  ② classical-ts 표 = R9 ``test_wis`` (다른 슬랩 — refit-only test, 비교 불가 caveat)
  ③ Supplementary S1 = ``WIS_adapt`` (untracked ``csv/per_model_detailed_complete.csv`` —
     생성 스크립트 부재 → **재현 불가**; 잔차 없는 모델은 test-residual self-calibration 누수로 추정)

이 스크립트는 **committed 코드 + committed 아티팩트만으로** 전 R10 모델의 WIS 3종을
재계산해 단일 SSOT CSV(``simulation/results/wis_ssot.csv``)로 산출한다. 재학습 0·DB 0.

세 칼럼(모두 재현가능 또는 정직 NaN):
  · ``test_wis``     = R10 static empirical WIS (Lei 2018 split-conformal; Bracher 2021).
                      ★이 값이 R10 ``per_model_metrics.csv`` 의 영속 ``wis`` 와 정확 일치
                      (48/48, Δ<0.05) — R10 hold-out test 슬랩이 산출한 것은 STATIC empirical
                      WIS 였음(MPH_ADAPTIVE_CONFORMAL=0 경로). = Table1 챔피언 3.28 의 출처.
                      leak-free residual(R9 in-sample → WF-CV OOF) 보유 모델만; 없으면 NaN.
  · ``oof_wis``      = R9 WF-CV OOF WIS (per_model_optimal/<m>.json val_metrics.oof_wis).
                      ★전 모델 유한·재현가능 = "부족분 보완" 칼럼 (META/Ensemble 만 inf→NaN).
                      챔피언 선정 SSOT 키(G-339 leak-free) 이기도 — 전 모델 비교 가능.
  · ``adaptive_wis`` = adaptive conformal(PID) WIS (G-365; 별개 post-hoc 슬랩).
                      test_wis 와 같은 leak-free residual 사용하나 정점서 구간 동적확장 →
                      커버리지 회복(static PI95 0.67→adaptive 0.90). 옛 Supp ``WIS_adapt`` 의
                      *정직 재현*(orphan CSV 값은 19 모델서 누수). 없으면 NaN.

★ 핵심 원칙 (ENGINEERING_PRINCIPLES.md #5 재현성, G-354 정직성):
  test-residual(y_test − pred) self-calibration 절대 금지(채점 대상 점에 보정 = 낙관 편향).
  leak-free residual(R9 in-sample 또는 WF-CV OOF[:test_start])만 사용. 없으면 정직 NaN.
  → 옛 orphan CSV 가 NaN 이어야 할 22 모델(classical-ts·foundation·Ensemble)에 값을 채운 것은
    누수였음을 본 SSOT 가 정직 NaN 으로 정정.

입력 (전부 committed / read-only):
  · simulation/results/per_model_eval/per_model_metrics.csv  (R10 모델 목록 + R10 ``wis`` 대조용)
  · simulation/results/per_model_optimal/<model>.json        (val_metrics.insample_residuals, oof_wis)
  · simulation/results/csv/predictions_<model>.csv           (test 슬랩 y_true/y_pred)
출력:
  · simulation/results/wis_ssot.csv  (model·test_wis·oof_wis·adaptive_wis·slab·reproducible·nan_reason
                                       + R10 대조 r10_wis·match_r10)

Usage:  .venv/bin/python -m simulation.scripts.compute_wis_ssot
Returns: 0. Side effects: wis_ssot.csv 작성 (DB write 없음, 모델 로드 없음 = 가벼움).
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np

RESULTS = "simulation/results"
R10_METRICS = f"{RESULTS}/per_model_eval/per_model_metrics.csv"
OPTIMAL_DIR = f"{RESULTS}/per_model_optimal"
PRED_DIR = f"{RESULTS}/csv"
OUT = f"{RESULTS}/wis_ssot.csv"


def _leakfree_residuals(name: str, wfcv_oof=None, y_in=None, test_start=None):
    """R10 과 동일 우선순위로 leak-free 잔차 반환 (test-residual 절대 금지).

    우선순위 (per_model_eval.py:715-754 와 동치):
      (1) R9 in-sample residual: per_model_optimal/<name>.json val_metrics.insample_residuals
          (native conformal cal-split 또는 static train-pool fit error — oof_wis 와 동일 레짐).
      (2) WF-CV OOF[:test_start] residual (R9 in-sample 결손 시; 본 SSOT 는 CSV 만으로
          재현해야 하므로 OOF 배열이 영속화돼 있을 때만 — 현 아티팩트엔 없어 (1)만 사용).
      (3) 둘 다 없음 → None (→ WIS=NaN, no test-leak).

    Returns:
        (np.ndarray | None, source_str). source ∈ {r9_leakfree, wfcv_oof, unavailable}.
    """
    f = f"{OPTIMAL_DIR}/{name}.json"
    if os.path.exists(f):
        try:
            d = json.load(open(f, encoding="utf-8"))
            r = (d.get("val_metrics", {}) or {}).get("insample_residuals")
            if r is not None:
                a = np.asarray(r, dtype=np.float64)
                a = a[np.isfinite(a)]
                if len(a) >= 2:
                    return a, "r9_leakfree"
        except Exception:
            pass
    # (2) WF-CV OOF fallback — 현 영속 아티팩트에 OOF 배열 없으면 skip
    if wfcv_oof is not None and name in wfcv_oof and wfcv_oof[name] is not None \
            and y_in is not None and test_start is not None:
        oof = np.asarray(wfcv_oof[name], dtype=np.float64)[:test_start]
        oy = np.asarray(y_in, dtype=np.float64)[:test_start]
        mask = np.isfinite(oof) & np.isfinite(oy)
        a = (oy - oof)[mask]
        if len(a) >= 2:
            return a, "wfcv_oof"
    return None, "unavailable"


def _oof_wis(name: str) -> float:
    """R9 WF-CV OOF WIS (per_model_optimal/<name>.json val_metrics.oof_wis).

    META/Ensemble(OOF 없음)는 inf → NaN 처리. 그 외 전 모델 유한 = 재현가능.
    """
    f = f"{OPTIMAL_DIR}/{name}.json"
    if not os.path.exists(f):
        return float("nan")
    try:
        d = json.load(open(f, encoding="utf-8"))
        v = (d.get("val_metrics", {}) or {}).get("oof_wis")
        if v is None:
            return float("nan")
        v = float(v)
        return v if np.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def _test_slab(name: str):
    """predictions_<name>.csv 의 test 슬랩 → (y_true, y_pred) | (None, None)."""
    import pandas as pd
    f = f"{PRED_DIR}/predictions_{name}.csv"
    if not os.path.exists(f):
        return None, None
    try:
        df = pd.read_csv(f)
        t = df[df["split"] == "test"] if "split" in df.columns else df
        if len(t) < 10:
            return None, None
        return (t["y_true"].to_numpy(np.float64), t["y_pred"].to_numpy(np.float64))
    except Exception:
        return None, None


def _slab_label(name: str) -> str:
    """모델군 라벨 (한 슬랩 라벨 명확화용 — 슬랩 혼용 해소)."""
    ens = name.startswith("Ensemble-")
    return "ensemble_meta" if ens else "hold_out_test"


def compute() -> list[dict]:
    """전 R10 모델의 test_wis·oof_wis·adaptive_wis 재계산. Returns rows."""
    import pandas as pd
    from simulation.analytics.hub_metrics import (
        FLUSIGHT_ALPHAS, k11_pi_widths_from_residuals,
    )
    from simulation.analytics.diagnostics import (
        weighted_interval_score_empirical,
    )
    from simulation.analytics.adaptive_conformal import (
        adaptive_conformal_bounds, wis_from_bounds,
    )

    # R10 모델 목록 + 대조용 R10 wis (이 SSOT 가 R10 을 재현하는지 검증).
    r10 = pd.read_csv(R10_METRICS)
    r10_wis = dict(zip(r10["model"], r10["wis"]))

    rows: list[dict] = []
    for name in sorted(r10["model"]):
        slab = _slab_label(name)
        oofw = _oof_wis(name)

        y, pred = _test_slab(name)
        res, src = _leakfree_residuals(name)

        test_wis = float("nan")
        adapt_wis = float("nan")
        nan_reason = ""

        if y is None:
            nan_reason = "no_test_predictions_csv"
        elif res is None:
            nan_reason = "no_leakfree_residual"  # test-residual self-calibration 금지 → 정직 NaN
        else:
            k11 = k11_pi_widths_from_residuals(np.abs(res), FLUSIGHT_ALPHAS)
            # static empirical WIS (R10 canonical; MPH_ADAPTIVE_CONFORMAL=0 경로)
            try:
                test_wis = float(np.mean(weighted_interval_score_empirical(
                    y, pred, res, alphas=list(FLUSIGHT_ALPHAS))))
            except Exception:
                test_wis = float("nan")
            # adaptive conformal WIS (R10 default MPH_ADAPTIVE_CONFORMAL=1 경로; G-365)
            try:
                b = adaptive_conformal_bounds(pred, k11, res, y, FLUSIGHT_ALPHAS)
                adapt_wis = float(np.mean(
                    wis_from_bounds(y, b, FLUSIGHT_ALPHAS, median=pred)))
            except Exception:
                adapt_wis = float("nan")

        # reproducible = 세 칼럼 중 하나라도 committed 코드로 재계산됨
        reproducible = bool(
            np.isfinite(oofw) or np.isfinite(test_wis) or np.isfinite(adapt_wis))

        # R10 대조: R10 영속 ``wis`` = static empirical WIS (MPH_ADAPTIVE_CONFORMAL=0
        #   경로로 산출됨; adaptive_wis 가 아님). 따라서 본 SSOT 의 ``test_wis``(static)
        #   가 R10 wis 를 정확 재현해야 한다 = 재현성 검증 키.
        r10w = float(r10_wis.get(name, float("nan")))
        if np.isfinite(test_wis) and np.isfinite(r10w):
            match_r10 = abs(test_wis - r10w) < 0.05
        elif (not np.isfinite(test_wis)) and (not np.isfinite(r10w)):
            match_r10 = True  # 둘 다 정직 NaN = 일치
        else:
            match_r10 = False

        rows.append({
            "model": name,
            "slab": slab,
            "test_wis": round(test_wis, 4) if np.isfinite(test_wis) else float("nan"),
            "oof_wis": round(oofw, 4) if np.isfinite(oofw) else float("nan"),
            "adaptive_wis": round(adapt_wis, 4) if np.isfinite(adapt_wis) else float("nan"),
            "residual_source": src if y is not None else "no_predictions",
            "reproducible": reproducible,
            "nan_reason": nan_reason,
            "r10_wis": round(r10w, 4) if np.isfinite(r10w) else float("nan"),
            "match_r10": match_r10,
        })
    return rows


def main() -> int:
    rows = compute()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cols = ["model", "slab", "test_wis", "oof_wis", "adaptive_wis",
            "residual_source", "reproducible", "nan_reason", "r10_wis", "match_r10"]
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if (isinstance(r.get(c), float) and not np.isfinite(r.get(c)))
                            else r.get(c, "")) for c in cols})

    # ── 요약 (정직성 감사) ──
    n = len(rows)
    n_test = sum(1 for r in rows if np.isfinite(r["test_wis"]))
    n_oof = sum(1 for r in rows if np.isfinite(r["oof_wis"]))
    n_adapt = sum(1 for r in rows if np.isfinite(r["adaptive_wis"]))
    n_nan_test = [r["model"] for r in rows if not np.isfinite(r["test_wis"])]
    n_mismatch = [r["model"] for r in rows if not r["match_r10"]]

    print(f"\n  wis_ssot.csv: {n} models", flush=True)
    print(f"    test_wis     재현가능: {n_test}/{n}  (NaN {n - n_test} = leak-free residual 부재)", flush=True)
    print(f"    oof_wis      재현가능: {n_oof}/{n}   (NaN {n - n_oof} = Ensemble/META OOF 부재)", flush=True)
    print(f"    adaptive_wis 재현가능: {n_adapt}/{n} (NaN {n - n_adapt} = leak-free residual 부재)", flush=True)
    print(f"    R10 wis 재현 검증: {n - len(n_mismatch)}/{n} 일치 (Δ<0.05)", flush=True)
    if n_mismatch:
        print(f"      ⚠ 불일치: {n_mismatch}", flush=True)
    print(f"    test_wis=NaN 모델 ({len(n_nan_test)}): {', '.join(n_nan_test)}", flush=True)
    print(f"  → {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
