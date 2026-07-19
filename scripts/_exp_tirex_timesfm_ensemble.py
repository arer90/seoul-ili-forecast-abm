#!/usr/bin/env python
"""EXPERIMENT: TiRex + TimesFM point ensemble to lower the WIS floor of the bagged-GBM
residual + CQR forecaster, evaluated on the SAME 132 rolling 1-step origins (Seoul ILI
weeks 205..336) against the fair baseline TiRex+CQR (WIS 2.4012, PICP95 0.9545).

Base point  = a*TiRex + (1-a)*TimesFM   (a = TiRex weight in [0,1]).
Candidate   = bagged-GBM (6-cap) conditional residual quantiles anchored on the base,
              + static CQR (seed on [origin_lo-K_CAL, origin_lo)).
Selection   = do-no-harm gate: a chosen ONLY on PAST origins [171,205) (full pipeline
              WIS there); adopt the blend only if it beats pure-TiRex past-WIS by >=0.05,
              else a=1.0 (pure TiRex). Test origins 205..336 are NEVER used to pick a.

Leak-free: (1) TiRex & TimesFM are rolled 1-step past-only (zero-shot / frozen weights);
(2) every GBM block trains on train_end = block_start - K_CAL (strictly before every cal
week it serves); (3) CQR seed calibrated before the eval window. No live/pipeline edits.

Reference (2.4012) per-origin WIS via tirex_empirical_qy + build_bounds_cqr (exact fair
baseline). DM = HLN h=1 paired t on per-origin WIS. Reports WIS, PICP95(k/132)+Clopper-
Pearson CI, DM p, last-34 WIS, mean 95% width.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
os.environ.setdefault("MPH_EVAL_FEATURES", "basic")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import numpy as np
from scipy import stats
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import scripts.dec_boosted_mech as D
from scripts.dec_boosted_mech import (load_split, build_features, cqr_offsets, build_bounds_cqr,
                                       FQ, MED_COL, MIN_CTX, K_CAL, PEAK_Y)
from scripts.dec_boosted_mech_multiorigin import T0, REFIT_K, CONFIGS, fit_gbm, bagged_qy
from scripts._verify_fairbase import tirex_empirical_qy
from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS
from simulation.analytics.adaptive_conformal import wis_from_bounds

ALPHAS = list(FLUSIGHT_ALPHAS)
TFM_CACHE = D.SCRATCH / "exp_timesfm_roll.npz"


def cp(k, nn, a=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(a/2, k, nn-k+1)
    hi = 1.0 if k == nn else stats.beta.ppf(1-a/2, k+1, nn-k)
    return round(float(lo), 3), round(float(hi), 3)

def dm(wa, wb):
    diff = wa - wb; n = len(diff); dbar = diff.mean()
    var = np.var(diff, ddof=1) / n
    if var <= 0: return 1.0, dbar
    st = dbar / np.sqrt(var) * np.sqrt((n+1)/n)
    return float(2*(1-stats.t.cdf(abs(st), df=n-1))), float(dbar)

def wis_of(B, y, med):
    return np.asarray(wis_from_bounds(y, B, ALPHAS, median=med), dtype=float)


def load_state():
    Xtr, ytr, Xte, yte, meta = load_split()
    ntr, nte = len(ytr), len(yte); ntot = ntr + nte
    yf = np.concatenate([ytr, yte]); cap = 2.0*float(yf.max())
    frozen = np.asarray(json.loads((ROOT/"simulation/results/per_model_optimal/TiRex.json").read_text())
                        ["refit_test_predictions"], dtype=float)
    d = np.load(D.TIREX_CACHE); tirex_pool = d["tirex_pool"]
    tirex = np.concatenate([np.full(MIN_CTX, np.nan), tirex_pool, frozen])
    tfm = np.load(TFM_CACHE)["tfm_full"]
    assert len(tfm) == ntot
    return dict(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, yf=yf, cap=cap,
                tirex=tirex, tfm=tfm, ntr=ntr, nte=nte, ntot=ntot)


def build_gbm_qy(feat, base, r, idxs, cap, refit_k=REFIT_K):
    """bagged-GBM conditional FLUSIGHT quantiles anchored on `base` at weeks idxs
    (past-only per-block refit; train_end = block_start - K_CAL)."""
    idxs = np.asarray(idxs)
    qy = np.zeros((len(idxs), len(FQ)))
    lo, hi = idxs.min(), idxs.max()+1
    for bstart in range(lo, hi, refit_k):
        bend = min(bstart+refit_k, hi); train_end = bstart - K_CAL
        tr = np.arange(MIN_CTX, train_end)
        gbm = [fit_gbm(feat[tr], r[tr], cfg) for cfg in CONFIGS.values()]
        mask = (idxs >= bstart) & (idxs < bend)
        if mask.any():
            oi = idxs[mask]
            qy[mask] = bagged_qy(gbm, feat[oi], base[oi], cap)
    return qy


def eval_candidate(S, base, origin_lo, origin_hi):
    """Full leak-free bagged-GBM+CQR pipeline anchored on `base` over origins
    [origin_lo, origin_hi). Returns per-origin WIS array, PICP95 coverage bool, w95 array,
    and the median series. CQR seed on [origin_lo-K_CAL, origin_lo)."""
    yf, cap = S["yf"], S["cap"]
    feat, _ = build_features(S["ytr"], S["yte"], S["Xtr"], S["Xte"], base)
    r = yf - base
    origins = np.arange(origin_lo, origin_hi)
    cal_idx = np.arange(origin_lo - K_CAL, origin_lo)
    qy = build_gbm_qy(feat, base, r, origins, cap)
    qy_cal = build_gbm_qy(feat, base, r, cal_idx, cap)
    cqr = cqr_offsets(qy_cal, yf[cal_idx])
    B = build_bounds_cqr(qy, cqr, cap)
    y = yf[origins]; med = qy[:, MED_COL]
    w = wis_of(B, y, med)
    lo95, hi95 = B[0.05]; cov = (y >= lo95) & (y <= hi95)
    return dict(wis=w, cov=cov, w95=(hi95-lo95), med=med, origins=origins, y=y)


def reference_wis(S, origins):
    """Exact 2.4012 fair baseline per-origin WIS (TiRex point + empirical past-resid CQR)."""
    yf, cap, tirex = S["yf"], S["cap"], S["tirex"]
    r_full = yf - tirex
    cal_idx = np.arange(T0 - K_CAL, T0)
    qy_ref = tirex_empirical_qy(tirex, r_full, origins, cap)
    qy_ref_cal = tirex_empirical_qy(tirex, r_full, cal_idx, cap)
    cqr_ref = cqr_offsets(qy_ref_cal, yf[cal_idx])
    B = build_bounds_cqr(qy_ref, cqr_ref, cap)
    y = yf[origins]; med = qy_ref[:, MED_COL]
    return wis_of(B, y, med), B, med


def main():
    t0 = time.time()
    S = load_state()
    ntot, cap = S["ntot"], S["cap"]
    tirex, tfm, yf = S["tirex"], S["tfm"], S["yf"]
    test_origins = np.arange(T0, ntot); nT = len(test_origins)
    y_test = yf[test_origins]

    # ---------- reference 2.4012 ----------
    ref_wis, ref_B, ref_med = reference_wis(S, test_origins)
    ref_lo, ref_hi = ref_B[0.05]; ref_cov = (y_test>=ref_lo)&(y_test<=ref_hi)
    print(f"REFERENCE fair TiRex+CQR: WIS={ref_wis.mean():.4f} PICP95={ref_cov.mean():.4f} "
          f"k/N={int(ref_cov.sum())}/{nT}")

    # ---------- point accuracy (oracle diagnostic, not used for selection) ----------
    print("\n--- point accuracy over 132 test origins (DIAGNOSTIC, not selection) ---")
    for nm, p in [("TiRex", tirex), ("TimesFM", tfm)]:
        e = yf[test_origins] - p[test_origins]
        print(f"  {nm:9s} MAE={np.abs(e).mean():.4f} RMSE={np.sqrt((e**2).mean()):.4f}")
    print("  blend RMSE (oracle scan): ", end="")
    for a in (1.0,0.8,0.6,0.5,0.4,0.2,0.0):
        b = a*tirex + (1-a)*tfm
        e = yf[test_origins]-b[test_origins]
        print(f"a={a:.1f}:{np.sqrt((e**2).mean()):.3f} ", end="")
    print()

    # ---------- pure-TiRex GBM candidate (harness sanity vs verified 2.2765/0.985) ----------
    pure = eval_candidate(S, tirex, T0, ntot)
    p_pure, _ = dm(pure["wis"], ref_wis)
    print(f"\nSANITY pure-TiRex bagged-GBM+CQR: WIS={pure['wis'].mean():.4f} "
          f"PICP95={pure['cov'].mean():.4f} DMp={p_pure:.4f}  (expect ~2.2765/0.985/0.057)")

    # ---------- leak-free do-no-harm w-selection on PAST origins [171,205) ----------
    A_GRID = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.0]
    SEL_LO, SEL_HI = T0 - 34, T0          # weeks 171..204 (34 past origins, mirror last-34)
    print(f"\n--- do-no-harm w-selection on PAST origins [{SEL_LO},{SEL_HI}) (34 origins) ---")
    sel = {}
    for a in A_GRID:
        base = a*tirex + (1-a)*tfm
        c = eval_candidate(S, base, SEL_LO, SEL_HI)
        sel[a] = float(c["wis"].mean())
        print(f"  a={a:.1f}  past-WIS={sel[a]:.4f}  PICP95={c['cov'].mean():.3f}")
    base_wis = sel[1.0]
    a_star = min(sel, key=lambda k: sel[k])
    gain = base_wis - sel[a_star]
    adopt = a_star if (a_star != 1.0 and gain >= 0.05) else 1.0
    print(f"  pure-TiRex past-WIS={base_wis:.4f} ; best a={a_star} (past-WIS={sel[a_star]:.4f}, "
          f"gain={gain:.4f}) ; do-no-harm(>=0.05) -> ADOPT a={adopt}")

    # ---------- final candidate over 132 test origins ----------
    base_star = adopt*tirex + (1-adopt)*tfm
    cand = eval_candidate(S, base_star, T0, ntot)
    w = cand["wis"]; cov = cand["cov"]; k = int(cov.sum())
    p, dbar = dm(w, ref_wis)
    last34 = np.zeros(nT, bool); last34[nT-34:] = True
    peak = y_test >= PEAK_Y
    print(f"\n=== FINAL ensemble candidate (a={adopt}) over {nT} test origins ===")
    print(f"  WIS={w.mean():.4f}   (ref 2.4012, dbar={dbar:+.4f})")
    print(f"  DM p vs 2.4012 = {p:.4f}   {'<0.05 SIGNIFICANT' if p<0.05 else 'NOT significant'}")
    print(f"  PICP95={cov.mean():.4f}  k/N={k}/{nT}  CP95ci={cp(k,nT)}  "
          f"{'CALIBRATED[0.93,0.96]' if 0.93<=cov.mean()<=0.96 else 'OUT of [0.93,0.96]'}")
    print(f"  last34 WIS={w[last34].mean():.4f}  {'<2.72 OK' if w[last34].mean()<2.72 else '>=2.72 FAIL'}")
    print(f"  mean W95={cand['w95'].mean():.2f}   peak(y>=50) PICP95={cov[peak].mean():.3f} (n={int(peak.sum())})")

    # ---------- also report the (oracle) best-a candidate for context (labeled) ----------
    print("\n--- ORACLE context: best-a on TEST origins (NOT a valid selection) ---")
    orc = {}
    for a in A_GRID:
        base = a*tirex + (1-a)*tfm
        c = eval_candidate(S, base, T0, ntot)
        pa, _ = dm(c["wis"], ref_wis)
        orc[a] = (float(c["wis"].mean()), float(c["cov"].mean()), pa, float(c["wis"][last34].mean()))
        print(f"  a={a:.1f}  WIS={orc[a][0]:.4f}  PICP95={orc[a][1]:.3f}  DMp={orc[a][2]:.4f}  "
              f"last34={orc[a][3]:.4f}")

    out = {
        "reference_wis": round(float(ref_wis.mean()), 4),
        "reference_picp95": round(float(ref_cov.mean()), 4),
        "pure_tirex_gbm": {"wis": round(float(pure["wis"].mean()),4),
                            "picp95": round(float(pure["cov"].mean()),4), "dm_p": round(p_pure,4)},
        "selection": {"grid": A_GRID, "past_window": [int(SEL_LO), int(SEL_HI)],
                      "past_wis": {str(a): round(v,4) for a,v in sel.items()},
                      "a_star": a_star, "gain": round(gain,4), "adopted_a": adopt},
        "final": {"a": adopt, "wis": round(float(w.mean()),4), "dm_p": round(p,4),
                  "picp95": round(float(cov.mean()),4), "k_of_n": f"{k}/{nT}",
                  "cp95ci": list(cp(k,nT)), "last34_wis": round(float(w[last34].mean()),4),
                  "mean_w95": round(float(cand["w95"].mean()),2),
                  "peak_picp95": round(float(cov[peak].mean()),3)},
        "oracle_by_a": {str(a): {"wis": round(v[0],4), "picp95": round(v[1],3),
                                 "dm_p": round(v[2],4), "last34_wis": round(v[3],4)}
                        for a,v in orc.items()},
        "elapsed_sec": round(time.time()-t0,1),
    }
    (ROOT/"scripts"/"_exp_tirex_timesfm_ensemble.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote scripts/_exp_tirex_timesfm_ensemble.json  ({out['elapsed_sec']}s)")

if __name__ == "__main__":
    raise SystemExit(main())
