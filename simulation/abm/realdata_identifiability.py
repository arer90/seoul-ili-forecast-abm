"""
simulation.abm.realdata_identifiability
=======================================
P4 (behavioral identifiability) on REAL Seoul ILI — the title-aligned extension.

The headline P4 (``identifiability.py``) profiles a CALIBRATED (self-consistent)
truth. Here the ABM is fit to the REAL flu sentinel ILI (``sentinel_influenza``)
for each non-COVID season, and the best-fit behavioral parameters are compared
ACROSS seasons:

  * a parameter that lands near the SAME value every season is identifiable from
    the real epidemic curve;
  * one that SCATTERS across seasons is not — the real ILI does not pin it.

This is the real-data counterpart of the P4 result that risk/compliance (α, θ)
are recoverable from ILI while fatigue (τ) is the dynamics the curve integrates
out (needs mobility). A multi-season scatter of τ with stable α/θ confirms the
mobility-identifiability story on real data, not just a calibrated synthetic.

Run:  python -m simulation.abm.realdata_identifiability
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# non-COVID flu seasons present in the sentinel data (2020–21 flu was suppressed
# by COVID NPIs and 2021–22 still distorted, so they are excluded as non-physical
# for behavioral identification).
_SEASONS = (2019, 2022, 2023, 2024)
_WEEK_ORDER = list(range(36, 54)) + list(range(1, 36))


def real_season_series(year0: int) -> np.ndarray:
    """Weekly ILI rate for the flu season starting in ``year0`` (week 36 → 35)."""
    from simulation.abm.multiproxy_behavioral_validation import load_weekly_ili
    ili = load_weekly_ili()
    s = {w[1]: v for w, v in ili.items()
         if (w[0] == year0 and w[1] >= 36) or (w[0] == year0 + 1 and w[1] < 36)}
    return np.array([s[w] for w in _WEEK_ORDER if w in s])


def fit_seasons(seasons=_SEASONS, *, n_agents: int = 1000, days: int = 220) -> list:
    """Fit the ABM to each real season; return per-season fit + behavioral params."""
    from simulation.abm.sim_vs_observed import load_seoul_metapop
    from simulation.abm.validate_real import fit_agent_to_observed
    mp = load_seoul_metapop(days=days)
    out = []
    for y0 in seasons:
        series = real_season_series(y0)
        if len(series) < 30:
            continue
        fit = fit_agent_to_observed(
            series, mp, n_agents=n_agents,
            alpha_grid=(1.0, 2.0, 3.0), kappa_grid=(0.1, 0.2),
            tau_grid=(60.0, 90.0, 120.0), theta_grid=(0.05, 0.10, 0.15),
            beta_mult_grid=(0.8, 0.9, 1.0), gamma_mult_grid=(1.0,))
        p = fit.params if isinstance(fit.params, dict) else {}
        out.append({"season": f"{y0}-{str(y0 + 1)[2:]}", "r2": round(fit.r2, 3),
                    "shift_weeks": fit.shift_weeks, "beta_mult": fit.beta_mult,
                    "alpha": p.get("alpha"), "kappa": p.get("kappa"),
                    "tau": p.get("tau"), "theta": p.get("theta")})
    return out


def cross_season_identifiability(fits: list) -> dict:
    """Which behavioral params are pinned across seasons (identifiable from real
    ILI) vs scatter (not). CV = std/|mean| of the best-fit value across seasons;
    low CV ⇒ identifiable. Only seasons with r²≥0.3 (a usable fit) are pooled."""
    usable = [f for f in fits if (f["r2"] or 0) >= 0.3]
    # ★ caveat (Gemini C4): best-fits are GRID-SNAPPED (3-node grids), so a low CV
    # can be a quantization artifact (all seasons snap to the same node → CV=0) and
    # a true value on a grid boundary inflates CV. The r²≥0.3 gate also drops the
    # scatter-prone seasons (survivorship). Read CV as a COARSE identifiability
    # floor at the grid resolution, not a precise estimate — a continuous optimizer
    # + per-season CI is the rigorous version. The headline P4 (identifiability.py,
    # calibrated profile-likelihood) is the precise result; this is real-data support.
    out = {"n_usable_seasons": len(usable), "grid_resolution_caveat":
           "best-fits grid-snapped (3 nodes/param) → CV is a coarse floor, not precise",
           "params": {}}
    for key in ("alpha", "kappa", "tau", "theta"):
        vals = [f[key] for f in usable if f[key] is not None]
        if len(vals) < 2:
            out["params"][key] = {"error": "too few usable seasons"}
            continue
        m, sd = float(np.mean(vals)), float(np.std(vals))
        cv = sd / abs(m) if m else float("inf")
        out["params"][key] = {
            "values": vals, "mean": round(m, 3), "cv": round(cv, 3),
            "identifiable_from_ili": cv < 0.25,  # pinned within 25% across seasons
        }
    return out


def run(seasons=_SEASONS, **kw) -> dict:
    fits = fit_seasons(seasons, **kw)
    return {"per_season_fit": fits,
            "cross_season_identifiability": cross_season_identifiability(fits)}


def main() -> int:
    rep = run()
    out = Path("simulation/results/realdata_identifiability.json")
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print("실 Seoul ILI 시즌 fit (ABM이 실데이터 재현):")
    for f in rep["per_season_fit"]:
        print(f"  {f['season']}: r²={f['r2']} shift={f['shift_weeks']}주 "
              f"행동=(α={f['alpha']},κ={f['kappa']},τ={f['tau']},θ={f['theta']})")
    ci = rep["cross_season_identifiability"]
    print(f"\n실데이터 식별성 ({ci['n_usable_seasons']}개 usable 시즌, CV<0.25=식별):")
    for key, s in ci["params"].items():
        if "error" in s:
            print(f"  {key:6s} {s['error']}")
        else:
            mark = "✓ 식별됨" if s["identifiable_from_ili"] else "✗ 비식별(mobility 필요)"
            print(f"  {key:6s} CV={s['cv']:.2f} {mark}  (values={s['values']})")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
