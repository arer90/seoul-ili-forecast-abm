"""ABM mobility enrichment (branch A): time-resolved daytime population.

The current coupling uses a STATIC commuter matrix. The real
``daily_population_gu_hourly`` table (date × hour × district × age, 1.79M rows)
holds the actual living-population-by-hour, from which the daytime/night swing —
the empirical basis for time-resolved mobility — can be read directly. Business
hubs gain population during the day (people commute in) and residential districts
lose it; this provider exposes that swing and validates it (the known business
hubs must amplify), so a time-varying coupling can be built on a checked footing.

This is the validated DATA FOUNDATION for branch A. The full per-agent,
route-level subway/bus movement integrated into ``agent_kernel`` remains future
work; this module is additive (no core-ABM change) and never raises in the
analysis layer. Companion to ``stratified_validation`` (static daytime match) and
``age_validation`` (age-risk match). See PROOF_VALIDATION_PROTOCOL Pillar 1/3.
"""
from __future__ import annotations

import numpy as np

# districts that empirically gain population in the daytime (CBD / business)
_BUSINESS_HUBS = ["중구", "종로구", "강남구", "서초구", "영등포구"]


def load_daytime_amplification(db_path: str, *, day_hours=(10, 17),
                               night_hours=(0, 6)) -> dict[str, float]:
    """Per-district daytime/night population ratio from the real hourly table.

    ``ratio > 1`` ⇒ the district gains population in the daytime (commuter
    destination); ``< 1`` ⇒ it empties out (commuter origin). Returns
    ``{gu_nm: ratio}``. Never raises (DB error → ``{}``)."""
    try:
        from simulation.database import read_only_connect
        con = read_only_connect(db_path)
        try:
            rows = con.execute(
                "SELECT gu_nm, "
                " AVG(CASE WHEN hour BETWEEN ? AND ? THEN tot_pop END) AS day, "
                " AVG(CASE WHEN hour BETWEEN ? AND ? THEN tot_pop END) AS night "
                "FROM daily_population_gu_hourly GROUP BY gu_nm",
                (day_hours[0], day_hours[1], night_hours[0], night_hours[1]),
            ).fetchall()
        finally:
            con.close()
    except Exception:
        return {}
    out: dict[str, float] = {}
    for gu, day, night in rows:
        if gu and day and night and night > 0:
            out[str(gu)] = float(day) / float(night)
    return out


def validate_temporal_mobility_swing(db_path: str) -> dict:
    """Validate that the real data carries a meaningful, correctly-located
    time-resolved mobility swing — the empirical basis for a time-varying coupling.

    Checks (a) the spatial swing is wide (max/min district ratio ≥ 1.5) and
    (b) the known business hubs amplify (ratio > 1, i.e. gain daytime population).
    Returns ``{n_gu, swing_ratio, hubs_amplify, hubs_in_top5, business_hub_ratios,
    detected, verdict}``. Never raises.
    """
    amp = load_daytime_amplification(db_path)
    if len(amp) < 5:
        return {"error": f"only {len(amp)} districts with day+night population"}
    ratios = np.array(list(amp.values()), dtype=np.float64)
    swing = float(ratios.max() / ratios.min()) if ratios.min() > 0 else float("inf")
    present_hubs = [h for h in _BUSINESS_HUBS if h in amp]
    hubs_amplify = bool(present_hubs) and all(amp[h] > 1.0 for h in present_hubs)
    top5 = sorted(amp, key=amp.get, reverse=True)[:5]
    hubs_in_top5 = sum(1 for h in _BUSINESS_HUBS if h in top5)
    detected = (swing >= 1.5) and hubs_amplify
    verdict = (
        f"{len(amp)} districts, daytime/night swing span {swing:.2f}× "
        f"(max {ratios.max():.2f} / min {ratios.min():.2f}); business hubs "
        f"{'all amplify' if hubs_amplify else 'do NOT all amplify'} "
        f"({hubs_in_top5}/5 in the top-5 amplifiers). "
        + ("✓ real time-resolved mobility swing confirmed — business districts "
           "gain daytime population, residential lose it; a valid basis for a "
           "time-varying coupling (full per-agent route integration = future work)."
           if detected else
           "✗ no clear/located swing — the static coupling is the safer choice here.")
    )
    return {"n_gu": len(amp), "swing_ratio": round(swing, 3),
            "hubs_amplify": hubs_amplify, "hubs_in_top5": hubs_in_top5,
            "business_hub_ratios": {h: round(amp[h], 3) for h in present_hubs},
            "detected": bool(detected), "verdict": verdict}
