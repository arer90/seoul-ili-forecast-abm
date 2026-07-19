#!/usr/bin/env python
"""TASK C — genuinely-different ways to USE / MODIFY TiRex itself for the 1-step
probabilistic ILI interval, vs the champion (TiRex point + Tweedie residual-scale
q=mu+Qz*mu^(p/2) + expanding split-CQR = WIS 2.2378 / PICP95 0.9242 on 132 origins).

Not a thin head — three inference-time modifications of the foundation model:

  (i)  NATIVE-QUANTILE forecasting. TiRex.forecast returns 9 native predictive
       quantiles [0.1..0.9] from the xLSTM itself — a CONDITIONAL, heteroscedastic,
       shape-aware distribution, not a parametric residual-scale assumption. Map those
       9 to the 23 FluSight levels by monotone probit interpolation (interior) + linear-
       in-probit (Gaussian) tail extrapolation, then recalibrate with the SAME expanding
       split-CQR as the champion. Two variants: (i-a) native SHAPE on the champion point
       (median-aligned -> isolates the interval), (i-b) FULL native forecast (native
       median as the point too).

  (ii) TEST-TIME AUGMENTATION / in-context ensemble. Query TiRex at multiple context
       lengths {52,104,208,512} and input transforms {level, log1p} (no gradients), then
       Vincentize (average the native quantile functions) across a member set chosen ONLY
       on VAL [165,205). Averaging across contexts sharpens the median and the quantiles;
       map -> 23 levels -> expanding CQR.

  (iii) parametric distributional heads on the TiRex mean (Gamma, NegBin) + CQR — the
       "proper distributional head" probe. Reported honestly (expected null vs Tweedie).

LEAK-FREE: every native forecast at week t uses y[max(0,t-L):t] only (cached separately);
the champion point S['tirex'] and cap=2*max(y[:205]) are train-only; expanding CQR uses
conformity of origins < j; every knob (p, ensemble member set, dispersion family) is chosen
by argmin VAL WIS on [165,205) — never the 132 test origins. DM = HLN h=1 paired per-origin
WIS vs the champion. NO edits to any live/pipeline or existing script — this is a NEW script
that imports the sanctioned reusable helpers.
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
from scipy import stats
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.nov_guard_v3 import setup, dm, cp, wis_of
from scripts._exp_crosscountry import tweedie_qy, expanding_cqr_bounds, FQ, MED_COL, K_CAL, P_GRID

SCR = Path(os.environ.get("MPH_SCRATCH", str(Path(__file__).resolve().parents[1] / "_scratch")))
NATIVE_CACHE = SCR / "native_qcache.npz"

T0 = 205
MIN_CTX = 52
FQ = np.asarray(FQ, float)
FQL = [round(float(q), 4) for q in FQ]
NATIVE_LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
Z_NATIVE = stats.norm.ppf(NATIVE_LEVELS)
Z_TARGET = stats.norm.ppf(FQ)


def dm_block_boot(wa, wb, L=8, B=10000, seed=42):
    """Paired moving-block-bootstrap 2-sided p for mean(wa-wb)=0 (robust to autocorr,
    a stricter sanity check than HLN at n=132). Recentre resampled means to the null."""
    d = np.asarray(wa, float) - np.asarray(wb, float)
    n = len(d)
    obs = float(d.mean())
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / L))
    starts = rng.integers(0, n - L + 1, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(B, -1)[:, :n]
    null = d[idx].mean(axis=1) - obs
    p = 2.0 * min((null <= -abs(obs)).mean(), (null >= abs(obs)).mean())
    return float(min(1.0, p))


# ─────────────────────── native 9 -> 23 FluSight span (probit map) ───────────────────────
def native_to_flusight(native9: np.ndarray, cap: float) -> np.ndarray:
    """Map TiRex's 9 native quantiles [0.1..0.9] to the 23 FluSight levels.

    Monotone PCHIP interpolation on (probit-level, value) for the interior levels; linear-
    in-probit (Gaussian-tail) extrapolation for the 6 tail levels (0.01/0.025/0.05,
    0.95/0.975/0.99) using the outermost native slope. Enforces monotone + [0,cap].

    Args:
        native9: shape (9,) native quantile values (any order); sorted internally.
        cap: upper clip (train-only 2*max(y_train)).
    Returns:
        (23,) monotone-nondecreasing FluSight quantile row.
    """
    v = np.sort(np.asarray(native9, float))
    interp = PchipInterpolator(Z_NATIVE, v, extrapolate=False)
    q = interp(Z_TARGET)
    lo_slope = (v[1] - v[0]) / (Z_NATIVE[1] - Z_NATIVE[0])
    hi_slope = (v[-1] - v[-2]) / (Z_NATIVE[-1] - Z_NATIVE[-2])
    lo_m = Z_TARGET < Z_NATIVE[0]
    hi_m = Z_TARGET > Z_NATIVE[-1]
    q[lo_m] = v[0] + lo_slope * (Z_TARGET[lo_m] - Z_NATIVE[0])
    q[hi_m] = v[-1] + hi_slope * (Z_TARGET[hi_m] - Z_NATIVE[-1])
    q = np.clip(q, 0.0, cap)
    q.sort()
    return q


def span_from_native(native_rows: np.ndarray, cap: float, align_to: np.ndarray | None = None) -> np.ndarray:
    """Build (n,23) FluSight span from (n,9) native quantile rows.

    align_to: if given (n,) point array, median-shift each row so col MED_COL == align_to[i]
    (isolates the interval SHAPE on a fixed point). Else use the native median as-is.
    """
    n = native_rows.shape[0]
    out = np.zeros((n, len(FQ)))
    for i in range(n):
        out[i] = native_to_flusight(native_rows[i], cap)
    if align_to is not None:
        shift = align_to - out[:, MED_COL]
        out = out + shift[:, None]
        out = np.clip(out, 0.0, cap)
        out.sort(axis=1)
    return out


# ─────────────────────── metrics block ───────────────────────
def metrics(B, y, med, ref_wis, n):
    w = wis_of(B, y, med)
    lo95, hi95 = B[0.05]
    covv = (y >= lo95) & (y <= hi95)
    k = int(covv.sum())
    peak = y >= 50.0
    p_dm, dbar = dm(w, ref_wis)   # dbar = mean(w - ref_wis); <0 means we beat the champion
    return dict(
        wis=round(float(w.mean()), 4),
        dm_p_vs_champ=round(float(p_dm), 4), dm_meandiff=round(float(dbar), 4),
        beats=bool(w.mean() < ref_wis.mean() and p_dm < 0.05 and dbar < 0),
        picp95=round(k / n, 4), k_of_n=f"{k}/{n}", cp95ci=[round(v, 3) for v in cp(k, n)],
        peak_picp95=round(float(covv[peak].mean()), 3), n_peak=int(peak.sum()),
        last34_wis=round(float(w[n - 34:].mean()), 4),
        mean_w95=round(float((hi95 - lo95).mean()), 3),
        _wis=w,
    )


def _fmt(nm, m):
    b = "*BEATS*" if m["beats"] else ("under " if m["wis"] < 2.2378 else "      ")
    return (f"  {nm:<26s} WIS={m['wis']:.4f} {b} DMp={m['dm_p_vs_champ']:.4f} "
            f"d%={100*(m['wis']-2.2378)/2.2378:+5.1f} PICP95={m['picp95']:.4f} "
            f"({m['k_of_n']}) peak={m['peak_picp95']:.3f} last34={m['last34_wis']:.4f} "
            f"W95={m['mean_w95']:.2f}")


def main():
    t_start = time.time()
    S = setup()
    yf, tirex, ntot = S["yf"], S["tirex"], S["ntot"]
    origins = np.arange(T0, ntot)
    n = len(origins)
    y = yf[origins]
    val = np.arange(T0 - K_CAL, T0)          # [165,205)
    y_val = yf[val]
    train_max = float(np.nanmax(yf[:T0]))
    cap = 2.0 * train_max                     # train-only leak-free cap (== champion)

    # ── champion reference (reproduced): Tweedie head, p* on VAL, expanding CQR ──
    vw = {}
    for p in P_GRID:
        vqy = tweedie_qy(yf, tirex, val, p, cap)
        vB = expanding_cqr_bounds(vqy, y_val, cap)
        vw[p] = float(wis_of(vB, y_val, vqy[:, MED_COL]).mean())
    p_star = min(vw, key=vw.get)
    champ_qy = tweedie_qy(yf, tirex, origins, p_star, cap)
    champ_B = expanding_cqr_bounds(champ_qy, y, cap)
    champ_wis = wis_of(champ_B, y, champ_qy[:, MED_COL])
    lo95, hi95 = champ_B[0.05]
    champ_k = int(((y >= lo95) & (y <= hi95)).sum())
    print("=" * 100)
    print(f"CHAMPION (reproduced)  Tweedie head p*={p_star} + expanding CQR")
    print(f"   WIS={champ_wis.mean():.4f}  PICP95={champ_k/n:.4f} ({champ_k}/{n})  "
          f"last34={champ_wis[n-34:].mean():.4f}  (target to beat: WIS 2.2378, DM p<0.05)")
    print("=" * 100)

    # ── native quantile cache ──
    d = np.load(NATIVE_CACHE)
    cache = d["cache"]                        # (N, n_ctx, 2, 9)
    ctxs = list(d["ctxs"])                    # [52,104,208,512]
    CTX512 = ctxs.index(512)
    LEVEL, LOG1P = 0, 1
    # native median (col 4) sanity vs champion point on test origins
    nat_med_512 = cache[origins, CTX512, LEVEL, 4]
    med_mae = float(np.mean(np.abs(nat_med_512 - tirex[origins])))
    print(f"\n[sanity] native(ctx512,level) median vs champion point  MAE={med_mae:.4f} "
          f"(weeks 205-268 identical compute; 269-336 fresh-roll vs frozen)")

    results = {"champion": {"wis": round(float(champ_wis.mean()), 4), "p_star": p_star,
                            "picp95": round(champ_k / n, 4), "k_of_n": f"{champ_k}/{n}"},
               "native_median_vs_champ_point_mae": round(med_mae, 4)}
    all_rows = []

    # ══════════════ (i) NATIVE-QUANTILE forecasting ══════════════
    print("\n--- (i) TiRex NATIVE quantiles (9 -> 23 probit map) + expanding CQR ---")
    nat512 = cache[origins, CTX512, LEVEL, :]          # (n,9) test-origin native rows
    # (i-0) RAW native quantiles, NO conformal recalibration (shows what CQR contributes)
    qy_raw = span_from_native(nat512, cap, align_to=None)
    from scripts._exp_crosscountry import ALPHAS
    from scripts.dec_boosted_mech import FQ_COL
    cols = {a: (FQ_COL[round(a / 2.0, 4)], FQ_COL[round(1 - a / 2.0, 4)]) for a in ALPHAS}
    B_raw = {a: (np.clip(qy_raw[:, cl], 0, cap), np.clip(qy_raw[:, ch], 0, cap))
             for a, (cl, ch) in cols.items()}
    m_raw = metrics(B_raw, y, qy_raw[:, MED_COL], champ_wis, n)
    print(_fmt("i-0 native RAW (no CQR)", m_raw))
    # (i-a) native SHAPE on champion point (median-aligned)
    qy_ia = span_from_native(nat512, cap, align_to=tirex[origins])
    B_ia = expanding_cqr_bounds(qy_ia, y, cap)
    m_ia = metrics(B_ia, y, qy_ia[:, MED_COL], champ_wis, n)
    boot_p = dm_block_boot(m_ia["_wis"], champ_wis)
    print(_fmt("i-a native-shape@champ-pt", m_ia) + f"  [block-boot DMp={boot_p:.3f}]")
    # (i-b) FULL native forecast (native median as point)
    qy_ib = span_from_native(nat512, cap, align_to=None)
    B_ib = expanding_cqr_bounds(qy_ib, y, cap)
    m_ib = metrics(B_ib, y, qy_ib[:, MED_COL], champ_wis, n)
    print(_fmt("i-b full-native", m_ib))
    results["approach_i"] = {"i_0_native_raw_noCQR": {k: v for k, v in m_raw.items() if k != "_wis"},
                             "i_a_native_shape": {k: v for k, v in m_ia.items() if k != "_wis"},
                             "i_a_block_boot_dm_p": round(boot_p, 4),
                             "i_b_full_native": {k: v for k, v in m_ib.items() if k != "_wis"}}
    all_rows += [("i-a native-shape", m_ia), ("i-b full-native", m_ib)]

    # ══════════════ (ii) TEST-TIME AUGMENTATION / multi-context ensemble ══════════════
    print("\n--- (ii) TTA: Vincentize native quantiles over contexts/transforms (VAL-selected) ---")

    def member_native(idxs, ctx_idx, tf_idx):
        return cache[idxs, ctx_idx, tf_idx, :]         # (len,9)

    def ensemble_native(idxs, members):
        """Vincentization: average native quantile functions across members -> (len,9)."""
        stack = np.stack([member_native(idxs, ci, ti) for (ci, ti) in members], axis=0)
        return stack.mean(axis=0)

    # candidate member sets (each a list of (ctx_idx, transform_idx))
    lvl = [(ci, LEVEL) for ci in range(len(ctxs))]
    all8 = lvl + [(ci, LOG1P) for ci in range(len(ctxs))]
    cand_sets = {
        "ctx512_level": [(CTX512, LEVEL)],
        "all4_level_mean": lvl,
        "all8_mean": all8,
        "long_level {208,512}": [(ctxs.index(208), LEVEL), (CTX512, LEVEL)],
        "ctx512_level+log": [(CTX512, LEVEL), (CTX512, LOG1P)],
    }
    # also single best member by VAL
    for ci in range(len(ctxs)):
        for ti in (LEVEL, LOG1P):
            cand_sets[f"single_ctx{ctxs[ci]}_{'lvl' if ti==LEVEL else 'log'}"] = [(ci, ti)]

    # score each candidate on VAL (native ensemble -> full-native span -> expanding CQR)
    val_scores = {}
    for nm, mem in cand_sets.items():
        nat_v = ensemble_native(val, mem)
        qv = span_from_native(nat_v, cap, align_to=None)
        Bv = expanding_cqr_bounds(qv, y_val, cap)
        val_scores[nm] = float(wis_of(Bv, y_val, qv[:, MED_COL]).mean())
    best_set = min(val_scores, key=val_scores.get)
    print(f"    VAL WIS by member-set (argmin -> pick):")
    for nm in sorted(val_scores, key=val_scores.get):
        star = "  <== picked" if nm == best_set else ""
        print(f"        {nm:<26s} valWIS={val_scores[nm]:.4f}{star}")

    # TEST with the VAL-picked member set — full-native AND median-aligned to champ point
    nat_te = ensemble_native(origins, cand_sets[best_set])
    ens_med = span_from_native(nat_te, cap, align_to=None)[:, MED_COL]
    ens_mae = float(np.mean(np.abs(ens_med - y)))
    champ_mae = float(np.mean(np.abs(tirex[origins] - y)))
    qy_ii = span_from_native(nat_te, cap, align_to=None)
    B_ii = expanding_cqr_bounds(qy_ii, y, cap)
    m_ii = metrics(B_ii, y, qy_ii[:, MED_COL], champ_wis, n)
    qy_iia = span_from_native(nat_te, cap, align_to=tirex[origins])
    B_iia = expanding_cqr_bounds(qy_iia, y, cap)
    m_iia = metrics(B_iia, y, qy_iia[:, MED_COL], champ_wis, n)
    # document the point-sharpening null: MAE of the median for a few ensembles vs champion
    point_mae = {"champion(=ctx512)": champ_mae}
    for nm in ("all4_level_mean", "all8_mean", "long_level {208,512}"):
        med_nm = span_from_native(ensemble_native(origins, cand_sets[nm]), cap)[:, MED_COL]
        point_mae[nm] = round(float(np.mean(np.abs(med_nm - y))), 4)
    print(f"    point-sharpening check (median MAE): {point_mae}")
    print(f"    ensemble '{best_set}' point MAE={ens_mae:.4f} vs champion point MAE={champ_mae:.4f}")
    print(_fmt(f"ii full-ens[{best_set[:12]}]", m_ii))
    print(_fmt("ii ens-shape@champ-pt", m_iia))
    results["approach_ii"] = {
        "val_scores": {k: round(v, 4) for k, v in val_scores.items()},
        "picked_member_set": best_set,
        "point_mae_by_ensemble": point_mae,
        "ens_point_mae": round(ens_mae, 4), "champ_point_mae": round(champ_mae, 4),
        "full_ensemble": {k: v for k, v in m_ii.items() if k != "_wis"},
        "ens_shape_on_champ_point": {k: v for k, v in m_iia.items() if k != "_wis"},
    }
    all_rows += [("ii full-ensemble", m_ii), ("ii ens-shape", m_iia)]

    # ══════════════ (iii) parametric distributional heads on the TiRex mean ══════════════
    print("\n--- (iii) parametric heads on champion point (Gamma / NegBin) + expanding CQR ---")
    mu = np.clip(tirex, 1e-6, None)

    def gamma_span(idxs):
        out = np.zeros((len(idxs), len(FQ)))
        for i, t in enumerate(idxs):
            past = np.arange(MIN_CTX, t)
            r = yf[past] - tirex[past]
            r = r[np.isfinite(r)]
            if len(r) < 5:
                out[i] = np.clip(tirex[t] + np.zeros(len(FQ)), 0, cap)
                continue
            var = max(float(np.var(r)), 1e-6)
            k = max(mu[t] ** 2 / var, 1e-3)              # shape from MoM (Var=mu^2/k)
            row = stats.gamma.ppf(FQ, a=k, scale=mu[t] / k)
            out[i] = np.clip(np.nan_to_num(row, nan=tirex[t]), 0, cap)
        out.sort(axis=1)
        return out

    def negbin_span(idxs, scale):
        out = np.zeros((len(idxs), len(FQ)))
        for i, t in enumerate(idxs):
            past = np.arange(MIN_CTX, t)
            r = yf[past] - tirex[past]
            r = r[np.isfinite(r)]
            m_c = mu[t] * scale
            if len(r) < 5:
                out[i] = np.clip(tirex[t] + np.zeros(len(FQ)), 0, cap)
                continue
            var_c = max(float(np.var(r)) * scale ** 2, m_c + 1e-6)   # NB needs var>mean
            rr = max(m_c ** 2 / (var_c - m_c), 1e-3)                 # size param
            pp = rr / (rr + m_c)
            row = stats.nbinom.ppf(FQ, rr, pp) / scale
            out[i] = np.clip(np.nan_to_num(row, nan=tirex[t]), 0, cap)
        out.sort(axis=1)
        return out

    heads = {"gamma": gamma_span(origins)}
    for sc in (5.0, 10.0):
        heads[f"negbin_x{int(sc)}"] = negbin_span(origins, sc)
    heads_val = {"gamma": gamma_span(val)}
    for sc in (5.0, 10.0):
        heads_val[f"negbin_x{int(sc)}"] = negbin_span(val, sc)
    # VAL-pick the head family
    head_val_wis = {}
    for nm, qv in heads_val.items():
        Bv = expanding_cqr_bounds(qv, y_val, cap)
        head_val_wis[nm] = float(wis_of(Bv, y_val, qv[:, MED_COL]).mean())
    best_head = min(head_val_wis, key=head_val_wis.get)
    print(f"    VAL WIS by head: {{" + ", ".join(f'{k}:{v:.4f}' for k, v in head_val_wis.items())
          + f"}}  -> pick {best_head}")
    part_iii = {}
    for nm, qte in heads.items():
        B = expanding_cqr_bounds(qte, y, cap)
        m = metrics(B, y, qte[:, MED_COL], champ_wis, n)
        part_iii[nm] = m
        tag = "  <== VAL-pick" if nm == best_head else ""
        print(_fmt(f"iii {nm}", m) + tag)
    results["approach_iii"] = {
        "val_wis": {k: round(v, 4) for k, v in head_val_wis.items()},
        "picked_head": best_head,
        "metrics": {k: {kk: vv for kk, vv in v.items() if kk != "_wis"} for k, v in part_iii.items()},
    }
    all_rows += [(f"iii {best_head}", part_iii[best_head])]

    # ══════════════ verdict ══════════════
    print("\n" + "=" * 100)
    beaters = [(nm, m) for nm, m in all_rows if m["beats"]]
    # honest headline = best VAL-legit config that we would have committed to a-priori:
    # (i-a) native shape, (ii) VAL-picked ensemble, (iii) VAL-picked head. Report their TEST.
    print("VERDICT vs champion WIS 2.2378 (beat = WIS lower AND DM p<0.05 AND mean-diff<0):")
    for nm, m in all_rows:
        flag = "BEATS" if m["beats"] else ("~tie" if m["dm_p_vs_champ"] > 0.05 and m["wis"] < 2.2378
                                           else "loses")
        print(f"    {nm:<22s} WIS={m['wis']:.4f}  DMp={m['dm_p_vs_champ']:.4f}  "
              f"d%={100*(m['wis']-2.2378)/2.2378:+5.1f}  PICP95={m['picp95']:.4f}  -> {flag}")
    best_overall = min(all_rows, key=lambda kv: kv[1]["wis"])
    print(f"\n  best-WIS config: {best_overall[0]}  WIS={best_overall[1]['wis']:.4f}  "
          f"DM p={best_overall[1]['dm_p_vs_champ']:.4f}  beats={best_overall[1]['beats']}")
    print(f"  any DM-significant beater: {[b[0] for b in beaters] or 'NONE'}")
    print("=" * 100)

    results["verdict"] = {
        "champion_wis": 2.2378,
        "best_config": best_overall[0], "best_wis": best_overall[1]["wis"],
        "best_dm_p": best_overall[1]["dm_p_vs_champ"], "best_beats": best_overall[1]["beats"],
        "any_beater": [b[0] for b in beaters],
        "elapsed_sec": round(time.time() - t_start, 1),
    }
    (ROOT / "scripts" / "_rig_modify.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote scripts/_rig_modify.json  ({time.time()-t_start:.0f}s)")
    return results


if __name__ == "__main__":
    main()
