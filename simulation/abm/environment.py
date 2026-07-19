"""Environmental covariate layer (ABM heterogeneity enrichment, branch ⑬-환경).

The ABM's only environmental driver was the seasonal-forcing term (β_amp/β_phase).
This adds the explicit, real meteorological covariate behind it: Seoul daily
weather (``weather_historical`` — temperature, humidity, 4174 days). Influenza is
cold/dry-season driven (Lowen 2007; Shaman 2010), so the validation is that the
weather signal anti-correlates with influenza activity. Activity is read from the
real KR flu search-interest (``google_search_trends`` — 독감/인플루엔자), a
literature-validated influenza proxy, since both series are calendar-month
alignable. The fitted coupling gives a covariate a force-of-infection term or a
forecaster can consume.

Additive (no core rewrite); never raises in analysis. Companion to
``comorbidity`` and ``affiliation``.
"""
from __future__ import annotations

import numpy as np

_FLU_TERMS = ("독감", "인플루엔자", "감기")


def load_weather_monthly(db_path: str) -> dict[str, float]:
    """Monthly mean temperature (°C) from ``weather_historical`` → ``{YYYY-MM:
    ta_avg}``. obs_date 'YYYYMMDD' → 'YYYY-MM'. Never raises (→ ``{}``)."""
    out: dict[str, list] = {}
    try:
        from simulation.database import read_only_connect
        con = read_only_connect(db_path)
        try:
            for d, ta in con.execute(
                "SELECT obs_date, ta_avg FROM weather_historical WHERE ta_avg IS NOT NULL"
            ).fetchall():
                s = str(d)
                if len(s) == 8:
                    out.setdefault(f"{s[:4]}-{s[4:6]}", []).append(float(ta))
        finally:
            con.close()
    except Exception:
        return {}
    return {k: float(np.mean(v)) for k, v in out.items() if v}


def load_flu_search_monthly(db_path: str) -> dict[str, float]:
    """Monthly mean KR flu search interest from ``google_search_trends`` →
    ``{YYYY-MM: interest}`` (literature-validated influenza proxy). Never raises."""
    out: dict[str, list] = {}
    ph = ",".join("?" * len(_FLU_TERMS))
    try:
        from simulation.database import read_only_connect
        con = read_only_connect(db_path)
        try:
            for period, interest in con.execute(
                f"SELECT period, interest FROM google_search_trends "
                f"WHERE geo='KR' AND keyword IN ({ph}) AND interest IS NOT NULL",
                _FLU_TERMS,
            ).fetchall():
                ym = str(period)[:7]
                if len(ym) == 7:
                    out.setdefault(ym, []).append(float(interest))
        finally:
            con.close()
    except Exception:
        return {}
    return {k: float(np.mean(v)) for k, v in out.items() if v}


def validate_weather_flu_coupling(db_path: str) -> dict:
    """Validate the cold→flu environmental coupling: monthly temperature must
    anti-correlate with influenza activity (search interest). Returns
    ``{n_months, temp_flu_corr, winter_warm_flu_ratio, match, verdict}``. Never
    raises. ``match`` requires a clearly negative temperature–flu correlation."""
    temp = load_weather_monthly(db_path)
    flu = load_flu_search_monthly(db_path)
    common = sorted(set(temp) & set(flu))
    if len(common) < 12:
        return {"error": f"only {len(common)} aligned months (need ≥12)"}
    t = np.array([temp[m] for m in common])
    f = np.array([flu[m] for m in common])
    corr = float(np.corrcoef(t, f)[0, 1]) if t.std() and f.std() else 0.0
    # winter (DJF) vs summer (JJA) flu activity ratio (should be ≫ 1)
    win = np.mean([flu[m] for m in flu if m[5:7] in ("12", "01", "02")])
    smr = np.mean([flu[m] for m in flu if m[5:7] in ("06", "07", "08")])
    ratio = float(win / smr) if smr > 0 else float("inf")
    match = corr <= -0.3
    verdict = (
        f"{len(common)} aligned months: temperature–flu correlation {corr:+.2f}, "
        f"winter/summer flu activity {ratio:.1f}×. "
        + ("✓ cold/dry-season influenza coupling confirmed on real Seoul weather — "
           "a real-data environmental covariate (temperature anti-correlates with "
           "flu activity), the mechanism behind the seasonal forcing term."
           if match else "✗ no clear cold→flu coupling at this resolution.")
    )
    return {"n_months": len(common), "temp_flu_corr": round(corr, 4),
            "winter_summer_ratio": round(ratio, 3), "match": bool(match),
            "verdict": verdict}
