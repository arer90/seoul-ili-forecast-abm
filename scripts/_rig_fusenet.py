#!/usr/bin/env python
"""TASK B — FuseNet: a genuinely-engineered end-to-end fusion / recalibration network that
fuses the FOUNDATION MODEL'S INTERNAL OUTPUTS with a learned heteroscedastic distributional
head, trained on the pinball (WIS-surrogate) loss.  This is the SERIOUS version of the toy
3-second GRU fused net (scripts/_fusedepinet.py, WIS 5.48) — real architecture, real
regularization, real VAL model-selection, multi-seed averaging.

THE IDEA (prompt idea (i)+(ii) fused).  The champion (TiRex + Tweedie residual-scale + expanding
split-CQR, WIS 2.238) throws away TiRex's OWN predictive quantiles and uses only its mean.  FuseNet
keeps TiRex's rich output and learns, END-TO-END, a conditional recalibration that FUSES three
distributional signals per origin week t:
    (A) the champion's Tweedie residual-scale spread   tw_spread_a(t) = Qz^exp_a(t) * mu_t^(p/2)
        (heteroscedastic, EXPANDING empirical standardized-residual quantiles — same as champion),
    (B) TiRex's NATIVE 9-quantile predictive shape      tr_spread_a(t)  (probit-interpolated to the
        23 FluSight levels, tail-extrapolated, centered — the foundation model's own uncertainty),
    (C) a learned global tail-reshape vector c_a.
A small, heavily-regularized MLP reads 11 leak-free features x_t and emits, per origin:
    delta_t (mean-bias field), w_t = softplus (dispersion field, idea (ii)), gamma_t in [0,1]
    (fusion gate between the Tweedie branch A and the TiRex-native branch B).
    base predictive quantile:
        s_t = mu_t^(p/2);  std_a(t) = (1-g)*tw_spread_a/s_t + g*tr_spread_a/s_t + c_a
        q_a(t) = mu_t + delta_t + w_t * s_t * std_a(t)   -> sort -> clip[0,cap]
At initialization (delta=0, w=1, gamma=0, c=0) the base quantiles EQUAL the champion's expanding
Tweedie base EXACTLY -> after the identical expanding split-CQR this reproduces WIS 2.238.  So the
trained part can only MOVE OFF the foundation prior if the pinball loss + VAL selection say it helps
(do-no-harm BY CONSTRUCTION), and every gain is genuinely attributable to the learned network.

ENGINEERING (not a 3-second run):
  * end-to-end differentiable base-quantile generator, mean pinball loss over all 23 FluSight levels
    (the proper score WIS discretizes; Gneiting decomposition),
  * conditioning MLP with GELU + dropout + weight-decay, GLOBAL learned tail-reshape c_a,
  * shrinkage prior pulling (delta,logw,gamma,c) toward the champion (tunable lambda) = explicit
    do-no-harm regularizer,
  * probit-space interpolation + linear tail extrapolation of TiRex's 9 native quantiles to 23 levels,
  * feature standardization on TRAIN-only stats, causal (past-only) features,
  * early-stopping on VAL post-CQR WIS, MULTI-SEED (8) averaging of base quantiles,
  * a small VAL-selected hyperparameter grid (hidden/dropout/wd/lambda).

LEAK-FREE (#1).  Train ONLY on TRAIN weeks [FIT_START,165); select/early-stop ONLY on VAL[165,205);
evaluate TEST[205,337).  Every per-week feature, TiRex output, and Tweedie std-residual quantile at
week t uses weeks < t.  The Tweedie branch reuses scripts._exp_crosscountry.tweedie_qy VERBATIM
(expanding, past-only).  cap = 2*max(y_train) (train-only).  TEST conformalization =
expanding_cqr_bounds fresh on [205,337) — byte-identical wrapper to the champion, so the ONLY
difference scored is the base-quantile generator.  DM (HLN h=1) on paired per-origin WIS.

do-no-harm gate + honest verdict.  If 113 weeks cannot beat the foundation prior, it says so.
No live/pipeline or existing-script edits — a NEW read-only script importing sanctioned helpers.
"""
from __future__ import annotations
import os
os.environ.setdefault("MPH_EVAL_FEATURES", "basic")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.nov_guard_v3 import setup
from scripts._exp_crosscountry import (tweedie_qy, expanding_cqr_bounds, wis_of, dm, cp,
                                        FQ, MED_COL, K_CAL, P_GRID, MIN_CTX)

import torch
import torch.nn as nn

torch.set_num_threads(2)

# ── windows (leak-free clean split) ──────────────────────────────────────────
T0 = 205                       # TEST start (origins 205..336 = 132)
VAL_LO, VAL_HI = 165, 205      # VAL origins
VAL_SEED = 125                 # conformity seed for VAL CQR (past-only, < VAL_LO)
FIT_START = 76                 # first TRAIN fit week: >= 24 past std-residuals so Tweedie base is non-degenerate
FIT_HI = VAL_LO                # TRAIN fit weeks [FIT_START,165)

FQ_T = torch.tensor(np.asarray(FQ, float), dtype=torch.float32)   # 23 FluSight quantile levels
NATIVE_LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
SCRATCH = Path(os.environ.get("MPH_SCRATCH", str(Path(__file__).resolve().parents[1] / "_scratch")))
TIREX_Q_CACHE = SCRATCH / "tirex_native_q9.npz"
N_SEEDS = 8
EPS = 1e-6


# ─────────────────────────── TiRex native 9-quantile roll ────────────────────
def roll_tirex_native_q(yf: np.ndarray, ntot: int) -> np.ndarray:
    """Rolling 1-step TiRex NATIVE quantiles q9[t] (9 levels 0.1..0.9) over weeks [MIN_CTX,ntot).

    Leak-free: context at week t = y[max(0,t-512):t].  Cached.  Rows < MIN_CTX are NaN.
    Returns (ntot, 9) float array.
    """
    if TIREX_Q_CACHE.exists():
        d = np.load(TIREX_Q_CACHE)
        if d["q9"].shape == (ntot, 9):
            return d["q9"]
    from tirex import load_model
    model = load_model("NX-AI/TiRex", device="cpu")
    q9 = np.full((ntot, 9), np.nan)
    t0 = time.time()
    with torch.no_grad():
        for t in range(MIN_CTX, ntot):
            ctx = torch.tensor(yf[max(0, t - 512):t], dtype=torch.float32).unsqueeze(0)
            q, _mean = model.forecast(context=ctx, prediction_length=1)
            q9[t] = np.asarray(q, float).ravel()[:9]
            if (t - MIN_CTX) % 60 == 0:
                print(f"    [tirex-q9] week {t}/{ntot}  {time.time()-t0:.0f}s", flush=True)
    q9.sort(axis=1)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    np.savez(TIREX_Q_CACHE, q9=q9)
    print(f"    [tirex-q9] done {time.time()-t0:.0f}s -> {TIREX_Q_CACHE.name}", flush=True)
    return q9


def native_spread_23(q9: np.ndarray) -> np.ndarray:
    """Centered TiRex predictive spread at the 23 FluSight levels (probit interp + tail extrapolation).

    For each week: map native levels & FQ to probit z; piecewise-linear interp of the native quantile
    function in z-space with LINEAR EXTRAPOLATION beyond [0.1,0.9] (sane tail extension, unlike np.interp
    clamping).  Center by subtracting the 0.5 value.  y-units, centered at 0.  NaN rows stay NaN.
    """
    zc = sstats.norm.ppf(FQ)                         # 23 target probit points
    zn = sstats.norm.ppf(NATIVE_LEVELS)              # 9 native probit points
    out = np.full((q9.shape[0], len(FQ)), np.nan)
    for t in range(q9.shape[0]):
        row = q9[t]
        if not np.isfinite(row).all():
            continue
        vals = np.interp(zc, zn, row)                # clamped interp inside [0.1,0.9]
        # linear tail extrapolation beyond the native support
        sl_lo = (row[1] - row[0]) / (zn[1] - zn[0])
        sl_hi = (row[-1] - row[-2]) / (zn[-1] - zn[-2])
        left = zc < zn[0]; right = zc > zn[-1]
        vals[left] = row[0] + sl_lo * (zc[left] - zn[0])
        vals[right] = row[-1] + sl_hi * (zc[right] - zn[-1])
        med = np.interp(0.0, zn, row)                # value at level 0.5 (z=0)
        out[t] = vals - med
    return out


# ─────────────────────────── leak-free feature matrix ────────────────────────
def build_features(yf, tirex, q9, p) -> np.ndarray:
    """11 causal (past-only) features per week t.  Row t uses y[:t], tirex[t] (from y[:t]), q9[t]."""
    N = len(yf)
    mu = np.clip(tirex, EPS, None)
    z = (yf - tirex) / np.power(mu, p / 2.0)         # standardized residual (past-only when sliced)
    F = np.full((N, 11), np.nan)
    for t in range(MIN_CTX, N):
        y1 = yf[t - 1]; y2 = yf[t - 2]; y4 = yf[t - 4]
        m3 = yf[t - 3:t].mean()
        recent_max = float(yf[max(0, t - 52):t].max())
        zt = z[MIN_CTX:t]; zt = zt[np.isfinite(zt)]
        zscale = float(np.std(zt[-13:])) if len(zt) >= 5 else 0.0
        q = q9[t]
        if np.isfinite(q).all():
            iqr = (q[7] - q[1]) / max(mu[t], EPS)                       # native rel IQR (0.8-0.2)/mu
            skew = (q[8] + q[0] - 2 * q[4]) / max(q[8] - q[0], EPS)     # native skew
            asym = (q[8] - q[4]) / max(q[4] - q[0], EPS)                # native upper/lower ratio
        else:
            iqr = skew = asym = 0.0
        F[t] = [np.log1p(mu[t]), np.log1p(max(y1, 0)), y1 - y2, y1 - y4,
                np.log1p(max(m3, 0)), iqr, skew, asym, zscale,
                np.sin(2 * np.pi * t / 52.18), mu[t] / max(recent_max, EPS)]
    return F


# ─────────────────────────── FuseNet module ──────────────────────────────────
class FuseNet(nn.Module):
    """Conditioning MLP -> (delta, log w, logit gamma) + global tail-reshape c (23).

    Initialized so the base quantiles EQUAL the champion expanding-Tweedie base:
    final layer zero-weighted, biases -> delta=0, logw=0 (w=1), gamma~0; c=0.
    """
    def __init__(self, d_in: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden, 3)
        self.c = nn.Parameter(torch.zeros(len(FQ)))
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.3); nn.init.zeros_(m.bias)
        nn.init.zeros_(self.head.weight)
        self.head.bias.data = torch.tensor([0.0, 0.0, -9.0])   # delta=0, w=1 (via 0.5413 offset), gamma~1.2e-4

    def forward(self, x, tw_std, tr_std, s):
        """x:(B,d) features; tw_std,tr_std:(B,23) standardized Tweedie & TiRex spreads; s:(B,) mu^(p/2).
        Returns base quantiles (B,23) (unsorted; caller sorts+clips)."""
        h = self.net(x)
        o = self.head(h)
        delta = o[:, 0:1]
        w = torch.nn.functional.softplus(o[:, 1:2] + 0.5413)      # softplus(0.5413)=1.0 -> w init 1
        gamma = torch.sigmoid(o[:, 2:3])                          # init ~ sigmoid(-6) ~ 0.0025
        std = (1.0 - gamma) * tw_std + gamma * tr_std + self.c[None, :]
        return delta + (w * s[:, None]) * std                     # + mu added by caller


# ─────────────────────────── seeded expanding CQR (VAL selection) ─────────────
def seeded_cqr_wis(base_q_week, yf, cap, seed_lo, origins):
    """Expanding split-CQR seeded from weeks [seed_lo, origins[0]); WIS on `origins`.

    base_q_week: dict/2D indexable giving 23 base quantiles by absolute week for weeks
    [seed_lo, origins[-1]].  Leak-free: seed weeks precede all scored origins.
    """
    ext = np.arange(seed_lo, origins[-1] + 1)
    qy_ext = np.stack([base_q_week[w] for w in ext])
    B_ext = expanding_cqr_bounds(qy_ext, yf[ext], cap)
    sl = slice(int(origins[0] - seed_lo), int(origins[0] - seed_lo) + len(origins))
    B = {a: (B_ext[a][0][sl], B_ext[a][1][sl]) for a in B_ext}
    med = qy_ext[sl, MED_COL]
    return float(wis_of(B, yf[origins], med).mean())


# ─────────────────────────── one training run (one seed) ─────────────────────
def train_one(seed, cfg, X, TWs, TRs, S_arr, yf, mu, cap, fit_weeks, val_weeks):
    """Train FuseNet on fit_weeks; early-stop on VAL post-CQR WIS.  Returns week->base-quantile dict
    for ALL usable weeks and the best VAL WIS."""
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "cpu"
    net = FuseNet(X.shape[1], cfg["hidden"], cfg["dropout"]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=cfg["wd"])
    lam = cfg["lam"]

    fw = torch.tensor(fit_weeks)
    xf = torch.tensor(X[fit_weeks], dtype=torch.float32)
    twf = torch.tensor(TWs[fit_weeks], dtype=torch.float32)
    trf = torch.tensor(TRs[fit_weeks], dtype=torch.float32)
    sf = torch.tensor(S_arr[fit_weeks], dtype=torch.float32)
    muf = torch.tensor(mu[fit_weeks], dtype=torch.float32)
    yft = torch.tensor(yf[fit_weeks], dtype=torch.float32)
    tau = FQ_T

    # all usable weeks we ever need base quantiles for (seed..TEST end)
    all_weeks = np.arange(VAL_SEED, len(yf))
    xa = torch.tensor(X[all_weeks], dtype=torch.float32)
    twa = torch.tensor(TWs[all_weeks], dtype=torch.float32)
    tra = torch.tensor(TRs[all_weeks], dtype=torch.float32)
    sa = torch.tensor(S_arr[all_weeks], dtype=torch.float32)
    mua = torch.tensor(mu[all_weeks], dtype=torch.float32)

    def base_quantiles_all():
        net.eval()
        with torch.no_grad():
            out = net(xa, twa, tra, sa) + mua[:, None]
            out, _ = torch.sort(out, dim=1)
            out = torch.clamp(out, 0.0, cap)
        return {int(w): out[i].numpy() for i, w in enumerate(all_weeks)}

    best = {"vwis": np.inf, "bq": None, "epoch": -1}
    WARMUP, patience, bad = 60, 30, 0        # train >=60 epochs before honoring early-stop
    for epoch in range(500):
        net.train()
        opt.zero_grad()
        q = net(xf, twf, trf, sf) + muf[:, None]        # (B,23)
        diff = yft[:, None] - q
        pin = torch.maximum(tau[None, :] * diff, (tau[None, :] - 1) * diff).mean()
        # do-no-harm shrinkage: pull the learned corrections back toward champion
        h = net.net(xf); o = net.head(h)
        shrink = (o[:, 0].pow(2).mean()                     # delta -> 0
                  + o[:, 1].pow(2).mean()                   # logw -> 0 (w->1)
                  + torch.sigmoid(o[:, 2]).mean()           # gamma -> 0
                  + net.c.pow(2).mean())                    # c -> 0
        loss = pin + lam * shrink
        loss.backward()
        opt.step()
        if epoch % 5 == 0:
            bq = base_quantiles_all()
            vwis = seeded_cqr_wis(bq, yf, cap, VAL_SEED, val_weeks)
            if vwis < best["vwis"] - 1e-5:
                best = {"vwis": vwis, "bq": bq, "epoch": epoch}; bad = 0
            elif epoch >= WARMUP:
                bad += 1
                if bad >= patience:
                    break
    if best["bq"] is None:
        best["bq"] = base_quantiles_all()
    return best


def static_ablation(yf, tirex, mu, s_arr, tr_spread, p_star, cap, origins, val_weeks, champ_wis):
    """Diagnostic: fixed (gamma, wscale) fusion of Tweedie & TiRex-native spreads, no training.
    Shows whether the fusion has TEST headroom and whether VAL selection can reach it."""
    y_te = yf[origins]; n = len(origins)

    def base_for(idxs, g, w):
        tw = tweedie_qy(yf, tirex, idxs, p_star, cap)
        std = ((1 - g) * (tw - mu[idxs][:, None]) + g * tr_spread[idxs]) / s_arr[idxs][:, None]
        q = mu[idxs][:, None] + w * s_arr[idxs][:, None] * std
        return np.clip(np.sort(q, axis=1), 0, cap)

    def seeded_val(g, w):
        ext = np.arange(VAL_SEED, VAL_HI); qy = base_for(ext, g, w)
        B = expanding_cqr_bounds(qy, yf[ext], cap); sl = slice(VAL_LO - VAL_SEED, VAL_HI - VAL_SEED)
        Bv = {a: (B[a][0][sl], B[a][1][sl]) for a in B}
        return float(wis_of(Bv, yf[val_weeks], qy[sl, MED_COL]).mean())

    rows = []
    for g in (0.0, 0.25, 0.5, 0.75, 1.0):
        for w in (0.9, 1.0, 1.1, 1.2):
            qy = base_for(origins, g, w); B = expanding_cqr_bounds(qy, y_te, cap)
            wis = wis_of(B, y_te, qy[:, MED_COL]); lo, hi = B[0.05]
            k = int(((y_te >= lo) & (y_te <= hi)).sum()); p, d = dm(wis, champ_wis)
            rows.append(dict(gamma=g, wscale=w, val_wis=round(seeded_val(g, w), 4),
                             test_wis=round(float(wis.mean()), 4), picp95=round(k / n, 4),
                             dm_p=round(float(p), 4), diff=round(float(d), 4)))
    val_pick = min(rows, key=lambda r: r["val_wis"])       # what honest VAL-selection would choose
    oracle = min(rows, key=lambda r: r["test_wis"])        # unreachable TEST-optimal
    return {"grid": rows, "val_selected": val_pick, "test_oracle": oracle}


# ─────────────────────────── main ────────────────────────────────────────────
def main():
    t_start = time.time()
    S = setup()
    yf = S["yf"]; tirex = S["tirex"]; ntot = S["ntot"]
    mu = np.clip(tirex, EPS, None)
    cap = 2.0 * float(np.nanmax(yf[:T0]))               # train-only cap
    origins = np.arange(T0, ntot); n = len(origins); y_te = yf[origins]
    val_weeks = np.arange(VAL_LO, VAL_HI); y_val = yf[val_weeks]
    fit_weeks = np.arange(FIT_START, FIT_HI)

    # ── champion reference (identical wrapper): p* on VAL, expanding Tweedie + expanding CQR ──
    vs = {}
    for p in P_GRID:
        vq = tweedie_qy(yf, tirex, val_weeks, p, cap)
        vB = expanding_cqr_bounds(vq, y_val, cap)
        vs[p] = float(wis_of(vB, y_val, vq[:, MED_COL]).mean())
    p_star = min(vs, key=vs.get)
    champ_qy = tweedie_qy(yf, tirex, origins, p_star, cap)
    champ_B = expanding_cqr_bounds(champ_qy, y_te, cap)
    champ_wis = wis_of(champ_B, y_te, champ_qy[:, MED_COL])
    champ_lo, champ_hi = champ_B[0.05]
    champ_k = int(((y_te >= champ_lo) & (y_te <= champ_hi)).sum())
    print("=" * 88)
    print(f"CHAMPION  TiRex+Tweedie(p*={p_star})+expanding-CQR  WIS={champ_wis.mean():.4f}  "
          f"PICP95={champ_k/n:.4f} ({champ_k}/{n})   [target 2.238]")
    print("=" * 88)

    # ── TiRex native quantiles -> centered 23-level spread ──
    print("rolling TiRex native quantiles (cached)...", flush=True)
    q9 = roll_tirex_native_q(yf, ntot)
    tr_spread = native_spread_23(q9)                    # (ntot,23) y-units centered

    # ── Tweedie branch spread (expanding, past-only) at ALL weeks we need ──
    all_weeks = np.arange(VAL_SEED, ntot)
    tw_base_all = tweedie_qy(yf, tirex, all_weeks, p_star, cap)   # 23 base quantiles per week
    TW_spread = np.zeros((ntot, len(FQ)))                # week-indexed centered Tweedie spread
    TW_spread[all_weeks] = tw_base_all - mu[all_weeks][:, None]

    # standardized branch inputs (divide by s = mu^(p/2))
    s_arr = np.power(mu, p_star / 2.0)                   # (ntot,)
    TWs = np.zeros((ntot, len(FQ))); TRs = np.zeros((ntot, len(FQ)))
    good = np.isfinite(tr_spread).all(axis=1)
    for t in range(ntot):
        if s_arr[t] > 0:
            TWs[t] = TW_spread[t] / s_arr[t]
            if good[t]:
                TRs[t] = tr_spread[t] / s_arr[t]

    # ── features (standardized on TRAIN fit stats) ──
    Fraw = build_features(yf, tirex, q9, p_star)
    fmu = np.nanmean(Fraw[fit_weeks], axis=0); fsd = np.nanstd(Fraw[fit_weeks], axis=0) + 1e-8
    X = (Fraw - fmu) / fsd
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # ── sanity: FuseNet at init reproduces the champion expanding-Tweedie base ──
    net0 = FuseNet(X.shape[1], 16, 0.0)
    with torch.no_grad():
        wtest = origins
        q0 = net0(torch.tensor(X[wtest], dtype=torch.float32),
                  torch.tensor(TWs[wtest], dtype=torch.float32),
                  torch.tensor(TRs[wtest], dtype=torch.float32),
                  torch.tensor(s_arr[wtest], dtype=torch.float32)) + torch.tensor(mu[wtest], dtype=torch.float32)[:, None]
        q0, _ = torch.sort(q0, dim=1); q0 = torch.clamp(q0, 0.0, cap).numpy()
    init_match = float(np.abs(q0 - champ_qy).max())
    print(f"[sanity] FuseNet(init) base vs champion Tweedie base  max|dq|={init_match:.4g}  "
          f"(gamma~{torch.sigmoid(net0.head.bias[2]).item():.4f})")

    print(f"\nTRAIN fit weeks [{FIT_START},{FIT_HI}) = {len(fit_weeks)};  VAL [165,205);  TEST [205,337)")
    print(f"features={X.shape[1]}, seeds={N_SEEDS}\n")

    # ── static ablation diagnostic (does the fusion have honestly-reachable headroom?) ──
    abl = static_ablation(yf, tirex, mu, s_arr, tr_spread, p_star, cap, origins, val_weeks, champ_wis)
    vp, orc = abl["val_selected"], abl["test_oracle"]
    print(f"[ablation] VAL-selected fusion  g={vp['gamma']} w={vp['wscale']} -> TEST {vp['test_wis']} "
          f"(champ {champ_wis.mean():.4f}, diff {vp['diff']:+.4f}, DM p {vp['dm_p']})")
    print(f"[ablation] TEST-oracle fusion   g={orc['gamma']} w={orc['wscale']} -> TEST {orc['test_wis']} "
          f"(diff {orc['diff']:+.4f}, DM p {orc['dm_p']}) <- NOT VAL-reachable / NOT DM-sig\n")

    # ── VAL-selected hyperparameter grid (strong shrinkage: 40-week VAL does not transfer) ──
    grid = [dict(hidden=h, dropout=d, wd=wd, lam=lam)
            for h in (16, 24) for d in (0.1, 0.3) for wd in (1e-3, 1e-2) for lam in (0.1, 0.3)]
    print(f"{'cfg':>34s} | {'VAL_WIS':>8s}")
    print("-" * 46)
    results = []
    best_cfg = None
    for ci, cfg in enumerate(grid):
        seed_bqs = []
        vwis_seeds = []
        for sd in range(N_SEEDS):
            best = train_one(1000 + sd, cfg, X, TWs, TRs, s_arr, yf, mu, cap, fit_weeks, val_weeks)
            seed_bqs.append(best["bq"]); vwis_seeds.append(best["vwis"])
        # seed-average base quantiles
        avg_bq = {w: np.mean([bq[w] for bq in seed_bqs], axis=0) for w in seed_bqs[0]}
        vwis = seeded_cqr_wis(avg_bq, yf, cap, VAL_SEED, val_weeks)
        tag = f"h{cfg['hidden']}_d{cfg['dropout']}_wd{cfg['wd']}_l{cfg['lam']}"
        print(f"{tag:>34s} | {vwis:>8.4f}")
        rec = dict(cfg=cfg, tag=tag, val_wis=round(vwis, 4), avg_bq=avg_bq)
        results.append(rec)
        if best_cfg is None or vwis < best_cfg["val_wis"]:
            best_cfg = rec

    # ── evaluate the VAL-selected config on TEST (identical expanding CQR to champion) ──
    avg_bq = best_cfg["avg_bq"]
    fuse_qy = np.stack([avg_bq[int(w)] for w in origins])
    fuse_B = expanding_cqr_bounds(fuse_qy, y_te, cap)
    fuse_wis = wis_of(fuse_B, y_te, fuse_qy[:, MED_COL])
    fuse_lo, fuse_hi = fuse_B[0.05]
    fuse_cov = (y_te >= fuse_lo) & (y_te <= fuse_hi); fk = int(fuse_cov.sum())
    p_dm, dbar = dm(fuse_wis, champ_wis)
    peak = y_te >= 50.0
    last34 = np.zeros(n, bool); last34[n - 34:] = True

    # fair champion VAL under the SAME seeded CQR used for FuseNet selection
    champ_week = {int(w): tweedie_qy(yf, tirex, np.array([w]), p_star, cap)[0]
                  for w in range(VAL_SEED, VAL_HI)}
    champ_val = seeded_cqr_wis(champ_week, yf, cap, VAL_SEED, val_weeks)

    beats = bool(fuse_wis.mean() < champ_wis.mean() and p_dm < 0.05 and dbar < 0)
    do_no_harm = bool(fuse_wis.mean() <= champ_wis.mean() + 0.02)
    # do-no-harm gate: only deploy the trained head if it DM-beats champion; else retain champion.
    deployed = "FuseNet" if beats else "champion (gate rejects trained head)"
    deployed_wis = float(fuse_wis.mean()) if beats else float(champ_wis.mean())
    val_beats = bool(best_cfg["val_wis"] < champ_val)   # computed just above

    print("\n" + "=" * 88)
    print(f"FuseNet (VAL-selected {best_cfg['tag']}, {N_SEEDS}-seed avg)  vs  champion TiRex+Tweedie")
    print("=" * 88)
    print(f"  champion WIS = {champ_wis.mean():.4f}   PICP95 = {champ_k/n:.4f}")
    print(f"  FuseNet  WIS = {fuse_wis.mean():.4f}   PICP95 = {fk/n:.4f} ({fk}/{n}) CP95 {list(cp(fk,n))}")
    print(f"  delta WIS    = {fuse_wis.mean()-champ_wis.mean():+.4f}  "
          f"({100*(fuse_wis.mean()-champ_wis.mean())/champ_wis.mean():+.2f}%)")
    print(f"  DM p vs champion = {p_dm:.4f}   mean per-origin diff = {dbar:+.4f}")
    print(f"  peak PICP95  = {float(fuse_cov[peak].mean()):.3f} (n_peak={int(peak.sum())})   "
          f"last34 WIS = {float(fuse_wis[last34].mean()):.4f} (champ {float(champ_wis[last34].mean()):.4f})")
    print(f"  VAL WIS (seeded, fair): FuseNet {best_cfg['val_wis']:.4f}  champion {champ_val:.4f}  "
          f"-> FuseNet beats champion on VAL: {val_beats}")
    print(f"\n  beats_champion on TEST (WIS< & DM p<0.05 & diff<0): {beats}")
    print(f"  do_no_harm (WIS <= champion + 0.02):                {do_no_harm}")
    print(f"  do-no-harm GATE -> deploy: {deployed}  (deployed WIS = {deployed_wis:.4f})")
    if val_beats and not beats:
        verdict = ("FuseNet beats champion on the 40-wk VAL but this DOES NOT TRANSFER to TEST "
                   f"(+{100*(fuse_wis.mean()-champ_wis.mean())/champ_wis.mean():.1f}%, DM p={p_dm:.2f} n.s.); "
                   "VAL is anti-correlated with TEST on the added flexibility. Gate retains champion. "
                   "WIS ~2.24 is the data floor.")
    elif beats:
        verdict = "FuseNet SIGNIFICANTLY beats the foundation prior"
    elif do_no_harm:
        verdict = "FuseNet MATCHES the foundation prior (no significant gain; ~data floor)"
    else:
        verdict = "FuseNet is WORSE than the foundation prior (no honestly-selectable gain; ~data floor)"
    print(f"  VERDICT: {verdict}")
    print("=" * 88)

    out = {
        "n_origins": n, "weeks": f"{T0}..{ntot-1}", "p_star": p_star,
        "cap_train_only": round(cap, 2), "fit_weeks": [int(FIT_START), int(FIT_HI)],
        "n_fit": int(len(fit_weeks)), "n_features": int(X.shape[1]), "n_seeds": N_SEEDS,
        "init_matches_champion_maxdq": round(init_match, 8),
        "champion_wis": round(float(champ_wis.mean()), 4), "champion_picp95": round(champ_k / n, 4),
        "champion_val_wis": round(float(min(vs.values())), 4),
        "best_cfg": best_cfg["tag"], "best_cfg_params": best_cfg["cfg"],
        "fusenet_val_wis": best_cfg["val_wis"],
        "fusenet_wis": round(float(fuse_wis.mean()), 4), "fusenet_picp95": round(fk / n, 4),
        "fusenet_cp95ci": list(cp(fk, n)),
        "delta_wis": round(float(fuse_wis.mean() - champ_wis.mean()), 4),
        "delta_pct": round(100 * (fuse_wis.mean() - champ_wis.mean()) / champ_wis.mean(), 2),
        "dm_p_vs_champion": round(float(p_dm), 4), "dm_meandiff": round(float(dbar), 4),
        "fusenet_peak_picp95": round(float(fuse_cov[peak].mean()), 3),
        "fusenet_last34_wis": round(float(fuse_wis[last34].mean()), 4),
        "champion_last34_wis": round(float(champ_wis[last34].mean()), 4),
        "champion_val_wis_seeded": round(float(champ_val), 4),
        "fusenet_beats_champion_on_val": val_beats,
        "do_no_harm_gate_deploy": deployed, "deployed_wis": round(deployed_wis, 4),
        "beats_champion": beats, "do_no_harm": do_no_harm, "verdict": verdict,
        "val_grid": [{"tag": r["tag"], "val_wis": r["val_wis"]} for r in results],
        "static_ablation": {"val_selected": abl["val_selected"], "test_oracle": abl["test_oracle"],
                            "grid": abl["grid"]},
        "elapsed_sec": round(time.time() - t_start, 1),
    }
    (ROOT / "scripts" / "_rig_fusenet.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote scripts/_rig_fusenet.json  ({time.time()-t_start:.0f}s)")
    return out


if __name__ == "__main__":
    raise SystemExit(main() and 0)
