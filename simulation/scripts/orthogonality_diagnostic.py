"""Forecaster⊕ABM point-fusion orthogonality diagnostic (the honest GATE).

Question: does the hybrid ABM add ORTHOGONAL point signal to the FusedEpi champion,
so a champion⊕ABM point fusion could beat the 0.909 champion? Structural fact: the
anchored/assimilated ABM is DOWNSTREAM of the champion forecast (it is fit to track,
and EnKF-corrected toward, that forecast), so its forward trajectory largely
re-expresses the champion and its residuals inherit the champion's residual structure.

This diagnostic makes that concrete and — crucially — bounds the BEST CASE: it fits
the convex blend weight ON THE FORWARD WINDOW ITSELF (a deliberately LEAKY, most-
optimistic upper bound). If even that leaky best case cannot beat the champion by a
meaningful margin, then a legitimate leak-free (in-sample-fit) weight certainly
cannot — so point fusion is not pursued and the 0.909 champion stays the headline
(do-no-harm). The genuine ABM value is elsewhere (mechanism, person-like outputs,
counterfactual/intervention capability), not point accuracy.

Run: .venv/bin/python -m simulation.scripts.orthogonality_diagnostic
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS = Path("simulation/results")


def _r2(y, yhat) -> float:
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    ss = float(np.var(y))
    return float(1.0 - np.mean((y - yhat) ** 2) / ss) if ss > 0 else 0.0


def orthogonality_diagnostic() -> dict:
    """Compute the champion⊕ABM point-fusion diagnostic from the ablation/EnKF artifacts.

    Returns a dict with the leak-free downstream evidence (anchor correlation), the
    descriptive forward-window residual orthogonality, the LEAKY best-case convex-blend
    R² (upper bound), and a do-no-harm verdict. Reads two result JSONs; writes one.
    """
    ab = json.loads((RESULTS / "abm_variant_ablation_anchored.json").read_text())
    en = json.loads((RESULTS / "abm_hybrid_enkf.json").read_text())
    t = en["trajectories"]
    real = np.asarray(t["real"], float)
    champ = np.asarray(t["champion_forecast"], float)
    abm = np.asarray(t["variant_alone"], float)           # hybrid ABM anchored trajectory
    abm_enkf = np.asarray(t["variant_plus_enkf"], float)

    r_champ, r_abm = real - champ, real - abm
    anchor_corr = float(ab["variants"]["H"].get("anchor_corr_sim_vs_forecast", np.nan))

    # descriptive forward-window residual orthogonality (NOT a fusion license)
    resid_corr = float(np.corrcoef(r_champ, r_abm)[0, 1])
    b = float(np.polyfit(r_abm, r_champ, 1)[0])
    incremental_r2 = float(1.0 - np.var(r_champ - b * r_abm) / np.var(r_champ))

    # LEAKY best case: fit the convex weight ON the forward truth (upper bound only)
    ws = np.linspace(0.0, 1.0, 101)
    best = max(((w, _r2(real, w * champ + (1 - w) * abm)) for w in ws), key=lambda x: x[1])
    best_w, best_r2 = float(best[0]), float(best[1])
    best_enkf = max(((w, _r2(real, w * champ + (1 - w) * abm_enkf)) for w in ws),
                    key=lambda x: x[1])

    champ_r2 = _r2(real, champ)
    leaky_gain = best_r2 - champ_r2

    verdict = ("REDUNDANT — point fusion not justified; the leaky best-case blend "
               "barely beats the champion, so a leak-free in-sample weight cannot. "
               "Keep 0.909 as the headline (do-no-harm); ABM value is mechanism / "
               "person-like / counterfactual, not point accuracy."
               if leaky_gain < 0.02 else
               "MARGINAL — a small orthogonal component exists; a leak-free in-sample "
               "weight MIGHT help but n=17 makes it unreliable; do not replace the "
               "champion headline.")

    out = {
        "n_forward_weeks": int(real.size),
        "leak_free_downstream_evidence": {
            "anchor_corr_abm_vs_forecast": round(anchor_corr, 4),
            "note": "ABM trajectory is fit to track the champion forecast (no forward "
                    "truth) — high correlation ⇒ the ABM re-expresses the champion.",
        },
        "forward_window_residual_diagnostic_descriptive": {
            "champion_forward_r2": round(champ_r2, 4),
            "abm_hybrid_forward_r2": round(_r2(real, abm), 4),
            "resid_corr_champion_vs_abm": round(resid_corr, 4),
            "abm_explains_champion_resid_r2": round(incremental_r2, 4),
            "caveat": "computed on the 17-week forward window (descriptive only); NOT "
                      "used to select any weight — a leak-free weight needs in-sample.",
        },
        "leaky_bestcase_upper_bound": {
            "best_convex_weight_on_champion": best_w,
            "blend_r2_weight_fit_on_forward_truth": round(best_r2, 4),
            "champion_alone_r2": round(champ_r2, 4),
            "leaky_gain_over_champion": round(leaky_gain, 4),
            "with_enkf_blend_r2": round(float(best_enkf[1]), 4),
            "interpretation": "even fitting the weight ON the forward truth (leaky, the "
                              "most optimistic case) gains this little — a legitimate "
                              "leak-free weight cannot do better.",
        },
        "verdict": verdict,
    }
    (RESULTS / "abm_fusion_orthogonality.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> None:
    d = orthogonality_diagnostic()
    e = d["leak_free_downstream_evidence"]; f = d["forward_window_residual_diagnostic_descriptive"]
    g = d["leaky_bestcase_upper_bound"]
    print("=== Forecaster⊕ABM point-fusion orthogonality diagnostic ===")
    print(f"  anchor corr (ABM vs forecast, leak-free) = {e['anchor_corr_abm_vs_forecast']}")
    print(f"  champion fwd R² = {f['champion_forward_r2']} | ABM-H fwd R² = {f['abm_hybrid_forward_r2']}")
    print(f"  residual corr = {f['resid_corr_champion_vs_abm']} | ABM explains {f['abm_explains_champion_resid_r2']} of champ resid")
    print(f"  LEAKY best-case blend R² = {g['blend_r2_weight_fit_on_forward_truth']} "
          f"(w*champ={g['best_convex_weight_on_champion']}) vs champion {g['champion_alone_r2']} "
          f"→ gain {g['leaky_gain_over_champion']}")
    print(f"  VERDICT: {d['verdict']}")


if __name__ == "__main__":
    main()
