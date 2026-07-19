"""C (M7 SCI-grade): stratified ABM validation / matching against real data.

Answers "do the model's stratified attributes match reality?" — the question that
makes a synthetic-population ABM credible. Each validator compares a model
quantity to its real counterpart and returns a match metric + verdict, so we can
SEE where the model agrees with reality (and therefore where the richer
re-implementations — branch A dynamic mobility, branch B deeper stratification —
are actually needed, rather than assuming).

Validators (built incrementally):
  - ``validate_mobility_daytime``  : static commuter-implied daytime population
        per gu  vs  real hourly living population (daily_population_gu_hourly).
        Directly tells us whether branch A (time-resolved mobility) is warranted.

Real ILI age stratification lives in ``sentinel_influenza`` (age_group × ili_rate);
the per-gu daytime truth in ``daily_population_gu_hourly`` (gu × hour × tot_pop).
"""
from __future__ import annotations

from simulation.database.storage import read_only_connect

import numpy as np


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation without scipy (robust to scale/units)."""
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def validate_mobility_daytime(
    db_path: str,
    daytime_hours: tuple[int, int] = (10, 17),
) -> dict:
    """Static commuter mobility vs the real hourly daytime population (branch-A test).

    The metapop FoI uses a static ``commuter_matrix`` to place people by day
    (``daytime_pop[j] = Σ_i M[i,j]·pop_i``). This compares that *implied* daytime
    population per gu to the **observed** hourly living population
    (``daily_population_gu_hourly`` averaged over the daytime window). A high
    rank-correlation means the static commuter assumption already captures where
    people are during the day, so time-resolved mobility (branch A) buys little;
    a low one justifies branch A.

    Args:
        db_path: path to the read-only DB.
        daytime_hours: inclusive (start, end) hour window treated as "daytime".

    Returns:
        ``{n_gu, pearson, spearman, mean_abs_rel_err, verdict, by_gu}`` — or
        ``{error}`` when a table is missing. Never raises.
    """
    try:
        con = read_only_connect(db_path)
        try:
            con.execute("PRAGMA busy_timeout=2000")
            # commuter-implied daytime presence per dest gu:
            #   Σ_origin M[origin,dest]·pop_origin. NOTE: the `commuters` column
            #   is empty (all 0); the real row-stochastic mobility M is in
            #   `coupling` (each origin's coupling sums to 1) — a data-reality
            #   check this validator itself surfaced.
            rows = con.execute(
                "SELECT origin_gu, dest_gu, coupling, night_population "
                "FROM commuter_matrix"
            ).fetchall()
            lo, hi = daytime_hours
            real_rows = con.execute(
                "SELECT gu_nm, AVG(tot_pop) FROM daily_population_gu_hourly "
                "WHERE hour BETWEEN ? AND ? GROUP BY gu_nm", (lo, hi)
            ).fetchall()
        finally:
            con.close()
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}

    if not rows or not real_rows:
        return {"error": "commuter_matrix or daily_population_gu_hourly empty"}

    # `coupling` IS the row-stochastic M[origin,dest]; pop_origin = night_population.
    pop_origin: dict[str, float] = {}
    for o, _d, _coup, night in rows:
        if night is not None:
            pop_origin[o] = float(night)
    daytime_model: dict[str, float] = {}
    for o, d, coup, _night in rows:
        daytime_model[d] = daytime_model.get(d, 0.0) + float(coup or 0) * pop_origin.get(o, 0.0)

    real_day = {gu: float(v or 0) for gu, v in real_rows}
    gus = sorted(set(daytime_model) & set(real_day))
    if len(gus) < 3:
        return {"error": f"only {len(gus)} matching gu between commuter + hourly pop"}

    m = np.array([daytime_model[g] for g in gus], dtype=np.float64)
    r = np.array([real_day[g] for g in gus], dtype=np.float64)
    pearson = float(np.corrcoef(m, r)[0, 1]) if m.std() and r.std() else float("nan")
    spearman = _spearman(m, r)
    # scale model to real total before relative error (units differ)
    m_scaled = m * (r.sum() / m.sum()) if m.sum() else m
    mare = float(np.mean(np.abs(m_scaled - r) / np.maximum(r, 1.0)))

    verdict = ("STATIC-OK (branch A low priority)"
               if (np.isfinite(spearman) and spearman >= 0.8)
               else "MISMATCH (branch A — time-resolved mobility — justified)")
    return {
        "n_gu": len(gus), "pearson": round(pearson, 4),
        "spearman": round(spearman, 4), "mean_abs_rel_err": round(mare, 4),
        "verdict": verdict,
        "by_gu": {g: {"model_daytime": round(daytime_model[g], 0),
                      "real_daytime": round(real_day[g], 0)} for g in gus},
    }
