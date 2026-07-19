#!/usr/bin/env python3
"""SEIR up-slope × ML blend — falsification harness (simulation-advisor design, 2026-06).

Tests whether a web-side mechanistic blend improves +1-week ILI peak accuracy WITHOUT
retraining.  Blend = ml_pred·(1+κ(g−1)) where g = deterministic-seasonal-SEIR growth
factor over the forecast week, applied ONLY on rising edges (R_eff>1.15 AND ml rising AND
Nov-Jan), clamped g∈[1, G_MAX].  The ML's own phase is preserved (we multiply its point),
so no lag is injected.

Ships ONLY if a (κ, G_MAX, amplitude) cell passes ALL falsifiers:
  1. beats single-best at PEAK-RISE weeks (ΔMAE ≥ 5/1k AND Wilcoxon p<0.05)
  2. does NOT worsen PEAK-TURN weeks (≤ +1/1k)
  3. does NOT worsen OFF-PEAK weeks (≤ +0.5/1k)
  4. does NOT inject lag (blended peak cross-corr lag ≤ ml lag)
Else → NEGATIVE-RESULT: the web-side peak lever is exhausted (report + stop, do NOT overfit).

Pure read-only analysis; writes nothing.  Run: .venv/bin/python web/scripts/_blend_validate.py
"""
from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKTEST = ROOT / "web" / "public" / "aggregates" / "backtest.json"

# ── SEIR (copied deterministic seasonal Euler — NOT the chaotic metapop path) ──────────
N = 9_876_153
GAMMA = 1.0 / 3.5          # 0.286
SIGMA = 1.0 / 2.0          # 0.5
R0_SUMMER = 1.0
BETA_MEAN = R0_SUMMER * GAMMA
WANING = 1.0 / 365.0
VAX = 0.45
DETECT = 7.5
PEAK_DOY = 15
START_DOY = 245           # ~Sep 2 autumn trough


def seir_seasonal(amp: float, days: int = 400) -> list[tuple[float, float]]:
    """Return [(I, reT)] per day from a Sept-trough origin under seasonal forcing."""
    anchor = 5.0 / 1000.0
    I = anchor / DETECT * N
    E = I * GAMMA / SIGMA          # rising-wave latent ratio (advisor fix, not 1:1)
    S = N * (1.0 - VAX) - I - E
    R = N * VAX
    out: list[tuple[float, float]] = []
    dt = 0.25
    steps = int(days / dt)
    rec_day = -1
    for step in range(steps + 1):
        t = step * dt
        doy = (START_DOY - 1 + t) % 365 + 1
        beta = max(0.0, BETA_MEAN * (1.0 + amp * math.cos(2 * math.pi * (doy - PEAK_DOY) / 365.0)))
        di = int(round(t))
        if di > rec_day and di <= days:
            denom = max(S + E + I + R, 1.0)
            reT = (beta / GAMMA) * (S / denom)
            out.append((I, reT))
            rec_day = di
        if step == steps:
            break
        denom = max(S + E + I + R, 1.0)
        foi = beta * I / denom
        S2 = max(0.0, S + (-foi * S + WANING * R) * dt)
        E2 = max(0.0, E + (foi * S - SIGMA * E) * dt)
        I2 = max(0.0, I + (SIGMA * E - GAMMA * I) * dt)
        R2 = max(0.0, R + (GAMMA * I - WANING * R) * dt)
        S, E, I, R = S2, E2, I2, R2
    while len(out) < days + 1:
        out.append(out[-1])
    return out[: days + 1]


def season_origin(d: date) -> date:
    """Sept-2 of the flu season containing d (season starts in autumn)."""
    yr = d.year if d.month >= 9 else d.year - 1
    return date(yr, 9, 2)


def iso_week(d: date) -> int:
    return d.isocalendar()[1]


def in_upslope_window(d: date) -> bool:
    w = iso_week(d)
    return w >= 45 or w <= 4          # Nov → late Jan


def mae(errs: list[float]) -> float:
    return sum(abs(e) for e in errs) / len(errs) if errs else float("nan")


def wilcoxon_p(a: list[float], b: list[float]) -> float:
    """Two-sided Wilcoxon signed-rank p (normal approx) on paired |err|.  Small-n rough."""
    diffs = [x - y for x, y in zip(a, b) if x != y]
    n = len(diffs)
    if n < 3:
        return 1.0
    ranks = sorted(range(n), key=lambda i: abs(diffs[i]))
    rank_val = [0.0] * n
    for r, idx in enumerate(ranks, start=1):
        rank_val[idx] = r
    w_plus = sum(rank_val[i] for i in range(n) if diffs[i] > 0)
    mean_w = n * (n + 1) / 4.0
    sd_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sd_w == 0:
        return 1.0
    z = (w_plus - mean_w) / sd_w
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2))))


def xcorr_peak_lag(series: list[float], truth: list[float]) -> int:
    """Lag L (−3..3) maximizing corr(series[t], truth[t−L]); L>0 = series trails."""
    def corr(xs, ys):
        n = len(xs)
        if n < 4:
            return -2
        mx, my = sum(xs) / n, sum(ys) / n
        num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
        dx = sum((v - mx) ** 2 for v in xs)
        dy = sum((v - my) ** 2 for v in ys)
        return num / math.sqrt(dx * dy) if dx > 0 and dy > 0 else -2
    best, bestc = 0, -2
    for L in range(-3, 4):
        xs, ys = [], []
        for t in range(len(series)):
            j = t - L
            if 0 <= j < len(truth):
                xs.append(series[t]); ys.append(truth[j])
        c = corr(xs, ys)
        if c > bestc:
            bestc, best = c, L
    return best


def main() -> None:
    bt = json.loads(BACKTEST.read_text(encoding="utf-8"))
    m = bt["models"][0]                       # NegBinGLM champion
    pts = [p for p in m["test_points"] if p.get("date")]
    pts.sort(key=lambda p: p["date"])
    dates = [date.fromisoformat(p["date"][:10]) for p in pts]
    actual = [p["actual"] for p in pts]
    pred = [p["predicted"] for p in pts]
    n = len(pts)

    # strata
    PEAK_RISE, PEAK_TURN, OFF = [], [], []
    for i in range(n):
        d_true = actual[i] - actual[i - 1] if i > 0 else 0.0
        if d_true > 5.0 and in_upslope_window(dates[i]):
            PEAK_RISE.append(i)
        elif actual[i] >= 0.85 * max(actual):
            PEAK_TURN.append(i)
        else:
            OFF.append(i)

    base = {s: mae([actual[i] - pred[i] for i in lst]) for s, lst in
            [("rise", PEAK_RISE), ("turn", PEAK_TURN), ("off", OFF)]}
    print(f"strata: rise={len(PEAK_RISE)} turn={len(PEAK_TURN)} off={len(OFF)}  (n={n})")
    print(f"single-best MAE: rise={base['rise']:.2f} turn={base['turn']:.2f} off={base['off']:.2f}")
    ml_lag = xcorr_peak_lag(pred, actual)

    best_cell = None
    for amp in (0.286, 0.36, 0.45):
        seir = seir_seasonal(amp)
        Iarr = [x[0] for x in seir]
        reTarr = [x[1] for x in seir]
        # per-point g + reT + gate
        g_raw, reT_pt = [], []
        for i in range(n):
            off = (dates[i] - season_origin(dates[i])).days
            off = max(7, min(len(Iarr) - 1, off))
            g_raw.append(Iarr[off] / Iarr[off - 7] if Iarr[off - 7] > 0 else 1.0)
            reT_pt.append(reTarr[off])
        for G_MAX in (1.4, 1.6, 1.8):
            for kappa in (0.25, 0.5, 0.75, 1.0):
                blended = list(pred)
                for i in range(n):
                    ml_rising = i > 0 and pred[i] > pred[i - 1]
                    gate = reT_pt[i] > 1.15 and ml_rising and in_upslope_window(dates[i])
                    if gate:
                        g = max(1.0, min(G_MAX, g_raw[i]))
                        blended[i] = pred[i] * (1.0 + kappa * (g - 1.0))
                bl = {s: mae([actual[i] - blended[i] for i in lst]) for s, lst in
                      [("rise", PEAK_RISE), ("turn", PEAK_TURN), ("off", OFF)]}
                d_rise = base["rise"] - bl["rise"]
                p = wilcoxon_p([abs(actual[i] - pred[i]) for i in PEAK_RISE],
                               [abs(actual[i] - blended[i]) for i in PEAK_RISE])
                bl_lag = xcorr_peak_lag(blended, actual)
                passes = (d_rise >= 5.0 and p < 0.05
                          and bl["turn"] <= base["turn"] + 1.0
                          and bl["off"] <= base["off"] + 0.5
                          and bl_lag <= ml_lag)
                cell = dict(amp=amp, G_MAX=G_MAX, kappa=kappa, d_rise=round(d_rise, 2),
                            rise=round(bl["rise"], 2), turn=round(bl["turn"], 2),
                            off=round(bl["off"], 2), p=round(p, 3), lag=bl_lag, passes=passes)
                if passes and (best_cell is None or d_rise > best_cell["d_rise"]):
                    best_cell = cell
    # report
    print(f"\nml peak-lag={ml_lag}")
    # best non-passing cell for diagnostics
    print("\n=== VERDICT ===")
    if best_cell:
        print("✅ SHIP — falsifiers passed:")
        print(f"   {best_cell}")
        print("   → wire into build_production_forecast.py (rising-edge blend, blend_active flag, widen PI).")
    else:
        # show the best rise-improvement cell that FAILED, for honesty
        bestfail = None
        for amp in (0.286, 0.36, 0.45):
            seir = seir_seasonal(amp); Iarr = [x[0] for x in seir]; reTarr = [x[1] for x in seir]
            g_raw, reT_pt = [], []
            for i in range(n):
                off = max(7, min(len(Iarr) - 1, (dates[i] - season_origin(dates[i])).days))
                g_raw.append(Iarr[off] / Iarr[off - 7] if Iarr[off - 7] > 0 else 1.0); reT_pt.append(reTarr[off])
            for G_MAX in (1.8,):
                for kappa in (1.0,):
                    blended = list(pred)
                    for i in range(n):
                        gate = reT_pt[i] > 1.15 and i > 0 and pred[i] > pred[i - 1] and in_upslope_window(dates[i])
                        if gate:
                            blended[i] = pred[i] * (1.0 + kappa * (max(1.0, min(G_MAX, g_raw[i])) - 1.0))
                    d_rise = base["rise"] - mae([actual[i] - blended[i] for i in PEAK_RISE])
                    if bestfail is None or d_rise > bestfail[0]:
                        gated = sum(1 for i in PEAK_RISE if reT_pt[i] > 1.15 and i > 0 and pred[i] > pred[i-1] and in_upslope_window(dates[i]))
                        bestfail = (d_rise, amp, gated)
        print("❌ NEGATIVE-RESULT — no (κ,G_MAX,amp) cell passed all falsifiers.")
        print(f"   best ΔMAE(rise) achievable={bestfail[0]:.2f}/1k (need ≥5), amp={bestfail[1]}, "
              f"gated {bestfail[2]}/{len(PEAK_RISE)} rise-weeks.")
        print("   → web-side mechanistic blend does NOT robustly fix peaks. Lever exhausted (honest).")
        print("   Likely cause: Sept-trough seasonal SEIR up-slope too gentle for the ~2.3× single-week")
        print("   real rise, OR the rising-edge gate fires on too few of the n test peak-weeks.")


if __name__ == "__main__":
    main()
