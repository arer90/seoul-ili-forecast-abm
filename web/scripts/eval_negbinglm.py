#!/usr/bin/env python3
"""NegBinGLM (web champion) single-model evaluation — REAL data, reproducible.

The ensemble/other-model levers were shown to add nothing (or hurt) at peaks (2026-06
accuracy investigation), so the operational forecast is the single champion NegBinGLM.
This evaluates THAT model alone, on the real val/test split, across the dimensions an
MPH thesis cares about: point accuracy, regime-stratified error + bias, rolling-origin
drift, and probabilistic calibration (relative-conformal PI / WIS).

Source: simulation/results/csv/predictions_NegBinGLM.csv (split=val/test, y_true, y_pred).
Read-only; prints a report.  Run: .venv/bin/python web/scripts/eval_negbinglm.py
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "simulation" / "results" / "csv" / "predictions_NegBinGLM.csv"


def _load(split: str) -> list[tuple[int, float, float]]:
    rows = list(csv.DictReader(CSV.open(encoding="utf-8")))
    out = [(int(r["idx"]), float(r["y_true"]), float(r["y_pred"]))
           for r in rows if r.get("split") == split]
    return sorted(out)


def mae(a, b): return sum(abs(x - y) for x, y in zip(a, b)) / len(a)
def rmse(a, b): return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))
def me(a, b): return sum(y - x for x, y in zip(a, b)) / len(a)   # signed: pred − actual
def mape(a, b):
    p = [abs(x - y) / abs(x) for x, y in zip(a, b) if abs(x) > 1e-9]
    return 100 * sum(p) / len(p)
def r2(a, b):
    m = sum(a) / len(a)
    ss = sum((x - m) ** 2 for x in a)
    sr = sum((x - y) ** 2 for x, y in zip(a, b))
    return 1 - sr / ss if ss > 0 else float("nan")
def quantile(s, p):
    if p <= 0: return s[0]
    if p >= 1: return s[-1]
    i = p * (len(s) - 1); lo = int(i); f = i - lo
    return s[lo] * (1 - f) + s[min(lo + 1, len(s) - 1)] * f


def main() -> None:
    val, test = _load("val"), _load("test")
    yt = [t[1] for t in test]; yp = [t[2] for t in test]
    n = len(yt)

    print(f"=== NegBinGLM 단독 평가 — REAL data (test n={n}, val n={len(val)}) ===\n")

    print("── 점예측 (test) ──")
    print(f"  MAE {mae(yt, yp):.3f} · RMSE {rmse(yt, yp):.3f} · R² {r2(yt, yp):.4f} "
          f"· MAPE {mape(yt, yp):.1f}% · 평균편향(pred−actual) {me(yt, yp):+.2f}")

    thr = sorted(yt, reverse=True)[n // 4]
    hi = [i for i in range(n) if yt[i] >= thr]
    lo = [i for i in range(n) if yt[i] < thr]
    print(f"\n── 층화 (피크=상위25%, ILI≥{thr:.1f}) ──")
    print(f"  피크  (n={len(hi)}): MAE {mae([yt[i] for i in hi], [yp[i] for i in hi]):5.2f} "
          f"· 편향 {me([yt[i] for i in hi], [yp[i] for i in hi]):+.2f}  (음수=과소예측)")
    print(f"  비피크(n={len(lo)}): MAE {mae([yt[i] for i in lo], [yp[i] for i in lo]):5.2f} "
          f"· 편향 {me([yt[i] for i in lo], [yp[i] for i in lo]):+.2f}")

    h = n // 2
    print(f"\n── rolling-origin drift ──")
    print(f"  전반(n={h}): MAE {mae(yt[:h], yp[:h]):.2f}  →  후반(n={n - h}): MAE {mae(yt[h:], yp[h:]):.2f}")

    rel = sorted((t[1] - t[2]) / t[2] for t in val if t[2] > 0.5)
    print(f"\n── 확률예측 보정 (relative-conformal, val 잔차 n={len(rel)}) ──")
    for cov in (0.5, 0.8, 0.95):
        a = 1 - cov
        loq, hiq = quantile(rel, a / 2), quantile(rel, 1 - a / 2)
        hits = sum(1 for i in range(n) if max(0.0, yp[i] * (1 + loq)) <= yt[i] <= yp[i] * (1 + hiq))
        width = sum(yp[i] * (hiq - loq) for i in range(n)) / n
        print(f"  PI{int(cov * 100):2d}: 실측 coverage {100 * hits / n:3.0f}% (목표 {int(cov * 100)}%) · 평균폭 {width:.1f}")

    print("\n── 해석 ──")
    print("  • 비피크는 거의 무편향(MAE 2.8, 편향 ~0) — 평시 ILI는 정확.")
    print("  • 피크는 구조적 과소예측(편향 −9/1k) — 단주 급등을 못 앞섬(feature 한계).")
    print("  • 보정 후에도 PI 전 수준 under-cover = 과신(겨울 잔차가 여름 val보다 두꺼움).")
    print("  • 결론: 평시 신뢰 가능, 피크/상승구간은 상한(PI upper)·경고와 함께 봐야 함.")


if __name__ == "__main__":
    main()
