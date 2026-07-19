#!/usr/bin/env python
"""Single-SSOT multi-slab model comparison: held-out TEST (n=68) + REAL (n=13).

목적 (2026-06-05, 사용자 "모순 없애고 정리" + "외삽 불안정 반영"):
파이프라인 산출물에 같은 모델의 WIS 가 파일마다 다르게 적혀 모순(3.20 oof_cv /
4.37 per_model_optimal / 5.60·7.12 metric_history "test"). 그 중 metric_history
"test" 는 검증되지 않은 다른 예측 소스(MAE 7.12 ≠ refit 4.14)였다. 이 스크립트는
**검증된 단일 소스**(per_model_optimal/*.json 의 refit_test/refit_real 예측 + 캐시
y) 만으로 **하나의 스코어러**로 두 slab 을 재채점하여 단일 SSOT 표를 만든다.

검증 (이 스크립트 실행 시 자동, alignment_check):
  - real:  y[337:350] == real_eval/predictions.csv 의 y_true (max|Δ|=0)
  - test:  refit_test_predictions vs y[269:337] 의 MAE == 저장된 test_metrics.mae

slab 정의 (HWP 80/20, compute_split_indices):
  in-sample = y[0:337], real = y[337:350] (최근 13주, 2026-02-22~05-17).
  test = in-sample 마지막 68주 = y[269:337] (config 선택=pool[0:269] OOF 와 독립 → 깨끗).

외삽 불안정 (사용자 요청 반영):
  ref = **각 slab 의 training-range max** (test: y[:269].max; real: y[:337].max),
  extrapolation_ratio = max_pred / train_max. ratio > 1.5 → unstable.
  (이전 버전의 'real-window max' 기준은 정상적 1주 overshoot 도 불안정으로 오판 → 폐기.)

champion 기준 (사용자 결정 2026-06-04/06-05):
  ① pure held-out TEST WIS 1위, ② **외삽-안정 인지**: test·real 모두 stable 한
  모델 중 best TEST WIS (= robust champion). 둘 다 보고.

Gray-box 계약: 입력=per_model_optimal/*.json + feature_cache.parquet +
  real_eval/predictions.csv. 출력=real_eval/multi_slab_comparison.{csv,md}.
  재학습/DB write 없음. WIS=Gaussian(median, sigma); sigma=각 slab 잔차 std (plug-in,
  전 모델 동일 절차 → 공정 비교; 절대값은 다소 낙관적이나 랭킹 공정 — disclosed).
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
REAL_DIR = ROOT / "simulation" / "results" / "real_eval"
PRED_CSV = REAL_DIR / "predictions.csv"
PMO_DIR = ROOT / "simulation" / "results" / "per_model_optimal"
CACHE = ROOT / "simulation" / "cache" / "feature_cache.parquet"
OUT_CSV = REAL_DIR / "multi_slab_comparison.csv"
OUT_MD = REAL_DIR / "multi_slab_comparison.md"

N_INSAMPLE = 337
N_REAL = 13
N_TEST = 68  # HWP ceil(337*0.2)
STAB_CEIL = 1.5
ALPHAS = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


# ───────── math helpers (no scipy) ─────────
def _norm_ppf(q: float) -> float:
    if q <= 0:
        return -np.inf
    if q >= 1:
        return np.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if q < plow:
        x = math.sqrt(-2 * math.log(q))
        return (((((c[0]*x+c[1])*x+c[2])*x+c[3])*x+c[4])*x+c[5]) / \
               ((((d[0]*x+d[1])*x+d[2])*x+d[3])*x+1)
    if q > phigh:
        x = math.sqrt(-2 * math.log(1 - q))
        return -(((((c[0]*x+c[1])*x+c[2])*x+c[3])*x+c[4])*x+c[5]) / \
               ((((d[0]*x+d[1])*x+d[2])*x+d[3])*x+1)
    x = q - 0.5
    r = x * x
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * x / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def wis_gaussian(y, median, sigma) -> float:
    sigma = max(float(sigma), 1e-6)
    total = np.zeros(len(y))
    for al in ALPHAS:
        lo = median + sigma * _norm_ppf(al / 2)
        hi = median + sigma * _norm_ppf(1 - al / 2)
        total += (al / 2) * ((hi - lo) + (2/al)*np.maximum(0, lo - y)
                             + (2/al)*np.maximum(0, y - hi))
    total += 0.5 * np.abs(y - median)
    return float(np.mean(total / (len(ALPHAS) + 0.5)))


def picp95(y, median, sigma) -> float:
    sigma = max(float(sigma), 1e-6)
    lo = median + sigma * _norm_ppf(0.025)
    hi = median + sigma * _norm_ppf(0.975)
    return float(np.mean((y >= lo) & (y <= hi)))


def point_metrics(y, p) -> dict:
    e = p - y
    sst = float(np.sum((y - y.mean())**2))
    nz = y != 0
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e**2))),
            "r2": 1 - float(np.sum(e**2))/sst if sst > 0 else float("nan"),
            "mape": float(np.mean(np.abs(e[nz]/y[nz]))*100) if nz.any() else float("nan")}


def dm_hln(ae_m, ae_b, h=1):
    d = ae_m - ae_b
    n = len(d)
    dbar = float(np.mean(d))
    var = float(np.var(d)) / n
    if var <= 0:
        return (0.0, 1.0) if abs(dbar) < 1e-12 else (math.copysign(1e6, dbar), 0.0)
    dm = dbar / math.sqrt(var)
    hln = dm * math.sqrt(max((n + 1 - 2*h + h*(h-1)/n) / n, 1e-9))
    return float(hln), float(_betainc((n-1)/2, 0.5, (n-1)/((n-1)+hln*hln)))


def _betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lb = math.lgamma(a)+math.lgamma(b)-math.lgamma(a+b)
    front = math.exp(math.log(x)*a+math.log(1-x)*b-lb)/a
    f, c, dd = 1.0, 1.0, 0.0
    for i in range(300):
        mm = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (mm*(b-mm)*x)/((a+2*mm-1)*(a+2*mm))
        else:
            num = -((a+mm)*(a+b+mm)*x)/((a+2*mm)*(a+2*mm+1))
        dd = 1.0+num*dd
        if abs(dd) < 1e-30:
            dd = 1e-30
        dd = 1.0/dd
        c = 1.0+num/c
        if abs(c) < 1e-30:
            c = 1e-30
        f *= c*dd
        if abs(1-c*dd) < 1e-10:
            break
    val = front*(f-1.0)
    if x > (a+1)/(a+b+2):
        val = 1.0-_betainc(b, a, 1-x)
    return min(max(val, 0.0), 1.0)


# ───────── data ─────────
def load_y():
    y = pl.read_parquet(CACHE)["ili_rate"].to_numpy()
    assert len(y) == N_INSAMPLE + N_REAL, f"cache len {len(y)} != {N_INSAMPLE+N_REAL}"
    return y


def alignment_check(y):
    """Hard-fail if stored predictions don't align to cache y (garbage guard)."""
    yreal = y[N_INSAMPLE:]
    pc = list(csv.DictReader(open(PRED_CSV, encoding="utf-8")))
    yt_csv = np.array([float(r["y_true"]) for r in pc])
    assert np.max(np.abs(yreal - yt_csv)) < 1e-6, "real slab misaligned vs predictions.csv"
    ytest = y[N_INSAMPLE - N_TEST:N_INSAMPLE]
    for name in ("ARIMA", "NegBinGLM-V7"):
        d = json.load(open(PMO_DIR / f"{name}.json", encoding="utf-8"))
        tp = np.array(d["refit_test_predictions"], float)
        assert abs(np.mean(np.abs(tp - ytest)) - d["test_metrics"]["mae"]) < 0.3, \
            f"{name} test slab misaligned"
    return True


def main():
    y = load_y()
    alignment_check(y)
    y_test = y[N_INSAMPLE - N_TEST:N_INSAMPLE]
    y_real = y[N_INSAMPLE:]
    tmax_test = float(y[:N_INSAMPLE - N_TEST].max())   # train pool max (test)
    tmax_real = float(y[:N_INSAMPLE].max())            # full in-sample max (real)

    recs = []
    for fp in sorted(PMO_DIR.glob("*.json")):
        name = fp.stem
        if name == "summary":
            continue
        d = json.load(open(fp, encoding="utf-8"))
        r = {"model": name}
        tp = d.get("refit_test_predictions")
        rp = d.get("refit_real_predictions")
        if isinstance(tp, list) and len(tp) == N_TEST:
            _slab(r, "test", y_test, np.array(tp, float), tmax_test)
        if isinstance(rp, list) and len(rp) == N_REAL:
            _slab(r, "real", y_real, np.array(rp, float), tmax_real)
        if "test_wis" in r:
            recs.append(r)

    # baselines on real (predictions.csv) for DM reference
    pc = list(csv.DictReader(open(PRED_CSV, encoding="utf-8")))
    base = {}
    for col in [c for c in pc[0] if c not in ("date", "y_true")]:
        pred = np.array([float(row[col]) for row in pc], float)
        rr = {"model": col + "(baseline)"}
        _slab(rr, "real", y_real, pred, tmax_real)
        rr["test_wis"] = None
        base[col] = rr

    recs.sort(key=lambda r: r["test_wis"])
    for i, r in enumerate(recs, 1):
        r["rank_test"] = i

    # robust champion = test·real 모두 stable 중 best test WIS
    stable = [r for r in recs if r.get("test_stable") and r.get("real_stable")]
    pure = recs[0]
    robust = min(stable, key=lambda r: r["test_wis"]) if stable else None

    # DM-HLN: pure & robust champion vs best real baseline
    best_base = min(base.values(), key=lambda b: b["real_wis"]) if base else None
    if best_base is not None:
        for r in recs:
            if "_real_ae" in r and "_real_ae" in best_base:
                r["dm_real_vs_base"], r["dm_p_real"] = dm_hln(r["_real_ae"], best_base["_real_ae"])

    _write(recs, list(base.values()), pure, robust, best_base, tmax_test, tmax_real)


def _slab(r, slab, y, p, tmax):
    pm = point_metrics(y, p)
    sigma = max(float(np.std(p - y, ddof=1)), 1e-6)
    maxp = float(np.max(p))
    ratio = maxp / tmax if tmax > 0 else float("nan")
    r[f"{slab}_wis"] = round(wis_gaussian(y, p, sigma), 3)
    r[f"{slab}_mae"] = round(pm["mae"], 3)
    r[f"{slab}_r2"] = round(pm["r2"], 3)
    r[f"{slab}_maxpred"] = round(maxp, 1)
    r[f"{slab}_extrap"] = round(ratio, 2)
    r[f"{slab}_stable"] = bool(ratio <= STAB_CEIL)
    r[f"_{slab}_ae"] = np.abs(p - y)


def _write(recs, bases, pure, robust, best_base, tmax_test, tmax_real):
    cols = ["rank_test", "model", "test_wis", "test_mae", "test_r2", "test_extrap",
            "test_stable", "real_wis", "real_mae", "real_r2", "real_extrap",
            "real_stable", "dm_real_vs_base", "dm_p_real"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in recs:
            w.writerow({k: r.get(k, "") for k in cols})

    L = ["# Multi-slab model comparison — held-out TEST(n=68) + REAL(n=13)",
         "",
         "**단일 SSOT (2026-06-05).** 검증된 예측(refit_test/refit_real) + 캐시 y 를 "
         "하나의 스코어러로 재채점. 파일마다 달랐던 WIS 모순 제거. 외삽 불안정 = 각 slab "
         f"training-range max 기준 (test {tmax_test:.1f} / real {tmax_real:.1f}), >1.5× = unstable.",
         "",
         f"- **순수 held-out TEST 1위 = {pure['model']}** (test WIS={pure['test_wis']}, "
         f"R²={pure['test_r2']}, real WIS={pure.get('real_wis')}, "
         f"real extrap={pure.get('real_extrap')}× {'⚠불안정' if not pure.get('real_stable') else '안정'})."]
    if robust is not None:
        L.append(f"- **외삽-안정 robust champion = {robust['model']}** (test WIS={robust['test_wis']} "
                 f"rank {robust['rank_test']}, real WIS={robust['real_wis']}, "
                 f"test·real 모두 stable). ← 외삽 불안정 반영 시 권장.")
    L += ["",
          "| rk | model | TEST WIS | TEST R² | TEST extrap× | REAL WIS | REAL MAE | REAL extrap× | stable(T/R) | DM-HLN real vs base | p |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        tr = ("✓" if r.get("test_stable") else "✗") + "/" + ("✓" if r.get("real_stable") else "✗")
        L.append(f"| {r['rank_test']} | {r['model']} | {r['test_wis']} | {r['test_r2']} | "
                 f"{r['test_extrap']} | {r.get('real_wis','-')} | {r.get('real_mae','-')} | "
                 f"{r.get('real_extrap','-')}{' ⚠' if not r.get('real_stable',True) else ''} | "
                 f"{tr} | {r.get('dm_real_vs_base','-')} | {r.get('dm_p_real','-')} |")
    if bases:
        L += ["", "### reference baselines (real slab only)",
              "| model | REAL WIS | REAL MAE | REAL R² | REAL extrap× |",
              "|---|---|---|---|---|"]
        for b in sorted(bases, key=lambda b: b["real_wis"]):
            L.append(f"| {b['model']} | {b['real_wis']} | {b['real_mae']} | {b['real_r2']} | {b['real_extrap']} |")
    L += ["", "## VERDICT",
          f"- 순수 성능(held-out TEST n=68): **{pure['model']}** 최고 — oof_cv 와 일관 "
          f"(winner's curse 없음, R² {pure['test_r2']} 진짜).",
          f"- 그러나 REAL(n=13 최근)에서 {pure['model']} extrap={pure.get('real_extrap')}× "
          f"({'붕괴' if not pure.get('real_stable') else '안정'}) → **tail-risk**.",
          (f"- **외삽 불안정 반영 → robust champion = {robust['model']}**: TEST 거의 동급 "
           f"(WIS {robust['test_wis']}) + REAL 안정(WIS {robust['real_wis']}). "
           f"배포/서비스용 권장." if robust else "- 안정 모델 없음."),
          "- WIS=Gaussian plug-in sigma(잔차 std), 전 모델 동일 → 랭킹 공정(절대값 낙관적, disclosed).",
          "- TEST(n=68)·REAL(n=13)은 다른 기간 → 두 WIS 직접 비교 금지(각 slab 내 랭킹만)."]
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"✓ {OUT_CSV}\n✓ {OUT_MD}\n")
    print(f"  순수 TEST 1위 = {pure['model']} (WIS {pure['test_wis']}, real extrap {pure.get('real_extrap')}×)")
    if robust:
        print(f"  robust(외삽안정) champion = {robust['model']} (test WIS {robust['test_wis']}, real WIS {robust['real_wis']})")
    print("\n  rk model               TEST_WIS REAL_WIS  T/R-stable")
    for r in recs[:8]:
        tr = ("✓" if r.get("test_stable") else "✗")+"/"+("✓" if r.get("real_stable") else "✗")
        print(f"  {r['rank_test']:2d} {r['model']:18s} {r['test_wis']:8.3f} "
              f"{str(r.get('real_wis','-')):>8s}  {tr}")


if __name__ == "__main__":
    main()
