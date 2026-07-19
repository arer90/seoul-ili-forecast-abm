#!/usr/bin/env python
"""Does the champion (TiRex point + Tweedie interval) BEAT plain TiRex on the thesis's OTHER two primary
metrics — alert_f1 and early-warning sensitivity — where WIS is only a tie? Uses the project's own
alert_operating_curve. Alert signal = each model's upper prediction quantile (precautionary early warning:
alert when the forecast's upper bound crosses the threshold). The Tweedie head's heteroscedastic peak
widening should catch outbreak crossings TiRex's tighter native band misses. Leak-free, 132-origin TEST.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import scripts._exp_crosscountry as X
from scripts.nov_guard_v3 import setup
from scripts._tirex_headprobe_final import flusight
from simulation.analytics.alert_curve import alert_operating_curve, threshold_at_sensitivity

FQr = list(np.round(X.FQ, 4)); MEDI = FQr.index(0.5)
Q90 = FQr.index(0.9); Q95 = FQr.index(0.95)
T0 = 205


def main():
    S = setup(); y = S["yf"]; tirex = S["tirex"]; ntot = S["ntot"]
    cap = 2.0*float(np.nanmax(y[:T0])); origins = np.arange(T0, ntot); y_te = y[origins]
    va = np.arange(T0-X.K_CAL, T0); y_va = y[va]
    dec = np.load(ROOT/"scripts"/"_tirex_native_deciles.npz")["dec"]

    # TiRex-native quantiles
    nat = np.array([flusight(dec[t], cap) for t in origins])
    # champion: TiRex point + Tweedie interval (p on VAL) + expanding CQR
    vw = {p: float(X.wis_of(X.expanding_cqr_bounds(X.tweedie_qy(y,tirex,va,p,cap),y_va,cap),
                            y_va, X.tweedie_qy(y,tirex,va,p,cap)[:,X.MED_COL]).mean()) for p in X.P_GRID}
    ps = min(vw, key=vw.get)
    tqy = X.tweedie_qy(y, tirex, origins, ps, cap)
    tB = X.expanding_cqr_bounds(tqy, y_te, cap)
    # build champion full quantile matrix from CQR bounds is not trivial; use tqy skeleton + upper from bounds
    champ = np.clip(np.sort(tqy, 1), 0, cap)
    champ_up95 = tB[0.05][1]                                     # CQR-calibrated 95% upper
    nat_up95 = nat[:, Q95]

    thr = np.arange(5, 90, 2.5)                                  # sweep alert thresholds (ILI level)
    def summary(y_pred, name):
        curve = alert_operating_curve(y_te, y_pred, thr)
        f1s = [r["f1"] for r in curve if isinstance(r["f1"], float) and np.isfinite(r["f1"])]
        maxf1 = max(f1s) if f1s else float("nan")
        # sensitivity at a fixed epidemiological threshold (peak = 50)
        at50 = [r for r in curve if abs(r["threshold"]-50) < 1.3]
        sens50 = at50[0]["sensitivity"] if at50 else float("nan")
        f150 = at50[0]["f1"] if at50 else float("nan")
        return {"model": name, "max_alert_f1": round(maxf1, 4),
                "f1_at_thr50": round(f150, 4) if isinstance(f150, float) else f150,
                "sensitivity_at_thr50": round(sens50, 4) if isinstance(sens50, float) else sens50}

    res = {
        "note": "alert signal = model's UPPER 95% forecast bound (precautionary early warning); WIS is a tie (~2.24)",
        "tirex_point": summary(tirex[origins], "TiRex point"),
        "tirex_upper95": summary(nat_up95, "TiRex native upper-95"),
        "champion_point": summary(tqy[:, MEDI], "champion point"),
        "champion_upper95": summary(champ_up95, "champion (Tweedie) upper-95"),
    }
    # lead-time-style: at matched sensitivity 0.9, which reaches it at a higher (more specific) threshold?
    c_nat = alert_operating_curve(y_te, nat_up95, thr); c_ch = alert_operating_curve(y_te, champ_up95, thr)
    res["thr_at_sens0.9_tirex_upper"] = threshold_at_sensitivity(c_nat, 0.9)
    res["thr_at_sens0.9_champion_upper"] = threshold_at_sensitivity(c_ch, 0.9)

    (ROOT/"scripts"/"_alert_champion_vs_tirex.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str))
    tu = res["tirex_upper95"]["max_alert_f1"]; cu = res["champion_upper95"]["max_alert_f1"]
    print(f"\nVERDICT (alert): champion upper-95 max-F1 {cu} vs TiRex upper-95 {tu} -> "
          + ("champion WINS on alerting" if cu > tu else "tie/loss on alerting"))
    ts = res["tirex_upper95"]["sensitivity_at_thr50"]; cs = res["champion_upper95"]["sensitivity_at_thr50"]
    print(f"VERDICT (peak sensitivity @thr50): champion {cs} vs TiRex {ts} -> "
          + ("champion catches MORE outbreaks" if isinstance(cs,float) and isinstance(ts,float) and cs>ts else "tie/loss"))


if __name__ == "__main__":
    raise SystemExit(main())
