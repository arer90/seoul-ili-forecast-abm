"""
simulation.abm.density_allocation
=================================
**Density-proportional spatial agent allocation** for the 25-gu Seoul agent-based
model (G-388, user directive 2026-06-27).

The title-bar claims "(1) Multi-Agent + (4) 25 Districts". Until now
:func:`simulation.abm.agent_based.run_agent_abm` placed a *uniform* ``n_agents``
in every district, so a 25-district run had no **spatial** heterogeneity in its
agent budget — every gu carried the same crowd regardless of how many people are
actually there during the day. This module makes the two claims literal: the
total agent budget ``N`` is split across the 25 gu **in proportion to daytime
living-population density** ``d_g`` (낮 생활인구, the population actually exposed to
transmission), so crowded gu (강남·송파·서초 …) carry more decision-makers than
quiet ones.

Pipeline
--------
1. :func:`load_district_density` — per-gu mean ``day_livpop`` (read-only DB).
2. :func:`allocate_agents_by_density` — N → integer per-gu ``n_g`` (largest
   remainder, every gu >= ``floor``, Σ = N).
3. :func:`run_density_abm` — feed the allocation into ``run_agent_abm`` and
   read out the **per-district attack rate** ``a_g`` (cumulative incidence /
   resident population) — the spatial downscale weight.
4. :func:`validate_against_disease` — Spearman ρ of the weight (a_g or d_g)
   against the per-gu annual case distribution of an observed respiratory
   notifiable disease (``seoul_disease_district``).

Honest limitation (carried in the JSON/CSV/figure)
--------------------------------------------------
Weekly per-gu ILI for 2025/26 does **not** exist (KDCA ILI sentinels are
city-level, not gu-resolved). The weight is therefore **density- and
mechanism-based**, and is validated only **indirectly** against the per-gu
*annual* distribution of a notifiable respiratory disease (2020-2024) — NOT a
direct weekly-ILI calibration. A low ρ is reported as low; no fabrication.

Gray-box contract
-----------------
All public functions are pure read / pure compute. ``load_*`` opens the DB with
``read_only_connect`` (mode=ro, never a write path) and 0 SQLite mutation. No
model is retrained — only the ABM simulator is run.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np

from simulation.abm.agent_based import AgentABMResult, run_agent_abm
from simulation.abm.behavioural import BehaviouralParams
from simulation.database.config import SEOUL_GU_ORDERED
from simulation.sim.io import load_metapop_params
from simulation.sim.parameters import IDX_I

log = logging.getLogger(__name__)

__all__ = [
    "DistrictAllocation",
    "load_district_density",
    "allocate_agents_by_density",
    "district_attack_rates",
    "run_density_abm",
    "validate_against_disease",
    "spearman_rho",
]

# Respiratory-route notifiable diseases present in ``seoul_disease_district``
# with full 25-gu coverage (2020-2024). Pneumococcus is the directive's primary
# target; the others are robustness checks (varicella/pertussis carry the most
# cases ⇒ statistically more robust per-gu distributions).
RESP_DISEASES_PRIMARY = "폐렴구균감염증"   # pneumococcal infection (directive primary)
RESP_DISEASES_ROBUST = ("수두", "백일해", "성홍열", "유행성이하선염")  # varicella, pertussis, scarlet, mumps


@dataclass
class DistrictAllocation:
    """Density-proportional per-gu agent allocation + simulated attack rate."""
    districts: list[str]              # (G,) gu names (SEOUL_GU_ORDERED)
    density: np.ndarray               # (G,) mean day_livpop d_g
    n_agents: np.ndarray              # (G,) integer agent count n_g (Σ = total_N)
    total_n: int                      # Σ n_g
    attack_rate: Optional[np.ndarray] = None  # (G,) a_g (filled by run_density_abm)
    weight: Optional[np.ndarray] = None       # (G,) normalised downscale weight


# ── 1. density ─────────────────────────────────────────────────────────
def load_district_density(
    districts: Optional[list[str]] = None,
    *,
    date_min: Optional[str] = None,
    date_max: Optional[str] = None,
) -> np.ndarray:
    """Return (G,) mean **daytime** living-population ``day_livpop`` per gu.

    Daytime living-population (낮 생활인구) is used because that is the crowd
    actually present and exposed to transmission during the day — the right
    proxy for spatial transmission density, not the resident (night) count.

    Args:
        districts: gu ordering; default ``SEOUL_GU_ORDERED`` (25 gu).
        date_min: optional inclusive lower bound on ``stdr_de`` (YYYYMMDD str).
        date_max: optional inclusive upper bound on ``stdr_de`` (YYYYMMDD str).

    Returns:
        (G,) float array of mean ``day_livpop`` aligned with ``districts``.
        Districts absent from the DB get the all-gu mean (loud WARNING).

    Performance: one grouped SELECT (~76k rows). Side effects: none (read-only).
    Caller responsibility: ``districts`` are valid Seoul gu names.
    """
    from simulation.database import read_only_connect

    districts = districts or list(SEOUL_GU_ORDERED)
    G = len(districts)
    where = ["day_livpop IS NOT NULL", "signgu_nm != '서울시'"]
    args: list = []
    if date_min is not None:
        where.append("stdr_de >= ?")
        args.append(str(date_min))
    if date_max is not None:
        where.append("stdr_de <= ?")
        args.append(str(date_max))
    sql = (
        "SELECT signgu_nm, AVG(day_livpop) FROM daily_population_district "
        f"WHERE {' AND '.join(where)} GROUP BY signgu_nm"
    )
    dens_map: dict[str, float] = {}
    con = read_only_connect()
    try:
        rows = con.execute(sql, args).fetchall()
        dens_map = {r[0]: float(r[1]) for r in rows if r and r[0] and r[1] is not None}
    finally:
        con.close()

    if not dens_map:
        raise RuntimeError(
            "load_district_density: daily_population_district returned no "
            "day_livpop rows (check the DB / date filter)."
        )
    fallback = float(np.mean(list(dens_map.values())))
    out = np.empty(G, dtype=float)
    missing: list[str] = []
    for i, name in enumerate(districts):
        if name in dens_map:
            out[i] = dens_map[name]
        else:
            out[i] = fallback
            missing.append(name)
    if missing:
        log.warning(
            "load_district_density: %d/%d gu missing in DB (using all-gu mean): %s",
            len(missing), G, ", ".join(missing[:5]),
        )
    return out


# ── 2. allocation ──────────────────────────────────────────────────────
def allocate_agents_by_density(
    density: np.ndarray, total_n: int, *, floor: int = 100
) -> np.ndarray:
    """Split ``total_n`` agents across districts ∝ density (largest remainder).

    Each district is guaranteed at least ``floor`` agents (so a low-density gu
    never collapses to a strata-less point), and the remainder is distributed in
    proportion to ``density``. The integer counts sum **exactly** to ``total_n``.

    Args:
        density: (G,) non-negative per-gu density d_g (e.g. mean day_livpop).
        total_n: total agent budget N (e.g. 100_000). Must be >= G*floor.
        floor: per-district minimum agent count (>= 1).

    Returns:
        (G,) int64 array ``n_g`` with ``sum == total_n`` and ``min >= floor``.

    Performance: O(G log G). Side effects: none.
    Caller responsibility: total_n >= G*floor; density finite & >= 0.
    """
    d = np.asarray(density, dtype=np.float64).ravel()
    if d.ndim != 1 or d.size == 0:
        raise ValueError(f"density must be 1-D non-empty; got {d.shape}")
    if np.any(d < 0) or not np.all(np.isfinite(d)):
        raise ValueError("density must be finite and non-negative")
    G = d.size
    total_n = int(total_n)
    floor = int(floor)
    if floor < 1:
        raise ValueError(f"floor must be >= 1; got {floor}")
    if total_n < G * floor:
        raise ValueError(
            f"total_n ({total_n}) < G*floor ({G}*{floor}={G * floor}); "
            "increase total_n or lower floor."
        )
    base = np.full(G, floor, dtype=np.int64)
    remaining = total_n - base.sum()
    tot = d.sum()
    share = (remaining * (d / tot)) if tot > 0 else np.full(G, remaining / G)
    add = np.floor(share).astype(np.int64)
    out = base + add
    # largest-remainder rounding so Σ == total_n exactly
    short = total_n - int(out.sum())
    if short > 0:
        order = np.argsort(-(share - add))
        out[order[:short]] += 1
    elif short < 0:  # defensive (shouldn't trigger with floor of the share)
        order = np.argsort(share - add)
        for k in range(-short):
            j = order[k % G]
            if out[j] > floor:
                out[j] -= 1
    return out


# ── 3. simulate → attack rate ───────────────────────────────────────────
def district_attack_rates(result: AgentABMResult) -> np.ndarray:
    """Per-district attack rate a_g = cumulative incidence / resident pop.

    Cumulative incidence = Σ_t daily new infections (the ``incidence`` array the
    SEIR kernel accumulates), divided by each district's resident population.

    Returns:
        (G,) float in [0, ~1]; the spatial transmission intensity per gu.
    """
    inc = np.asarray(result.seir.incidence)          # (T+1, G) daily new infections
    cum = inc.sum(axis=0)                              # (G,) cumulative incidence
    pops = np.asarray(result.seir.params.populations, dtype=float)
    pops = np.maximum(pops, 1.0)
    return cum / pops


def run_density_abm(
    total_n: int = 100_000,
    *,
    behaviour: Optional[BehaviouralParams] = None,
    days: int = 180,
    floor: int = 100,
    theta_sd: float = 0.25,
    seed: int = 42,
    districts: Optional[list[str]] = None,
    date_min: Optional[str] = None,
    date_max: Optional[str] = None,
) -> DistrictAllocation:
    """Density-allocate ``total_n`` agents over the 25 gu, run the ABM, return a_g.

    Loads real Seoul populations + mobility (``load_metapop_params``), allocates
    the agent budget by daytime density, runs the individual-agent ABM, and reads
    the per-district attack rate. **No model is retrained** — only the ABM
    simulator runs.

    Args:
        total_n: total agent budget split across districts (default 100_000).
        behaviour: behavioural params; default a moderate-response BehaviouralParams.
        days: simulation horizon (default 180 ≈ one respiratory season).
        floor: per-district minimum agent count.
        theta_sd, seed: passed through to ``run_agent_abm`` (reproducible).
        districts: gu ordering (default SEOUL_GU_ORDERED, 25 gu).
        date_min/date_max: optional ``stdr_de`` window for the density average.

    Returns:
        DistrictAllocation with density, n_agents, attack_rate, and weight
        (attack-rate-based, normalised to mean 1).

    Performance: O(days * G * max(n_g)). Side effects: read-only DB.
    """
    districts = districts or list(SEOUL_GU_ORDERED)
    behaviour = behaviour or BehaviouralParams(alpha=2.0, kappa=0.2, tau=90.0, theta=0.1)

    density = load_district_density(districts, date_min=date_min, date_max=date_max)
    n_g = allocate_agents_by_density(density, total_n, floor=floor)

    mp = load_metapop_params(days=days, districts=districts)
    mp = replace(mp, days=days)
    result = run_agent_abm(
        mp, behaviour, n_agents_per_district=n_g, theta_sd=theta_sd, seed=seed,
    )
    a_g = district_attack_rates(result)
    # downscale weight = attack rate normalised to mean 1 (relative spatial risk)
    w = a_g / a_g.mean() if a_g.mean() > 0 else np.ones_like(a_g)
    return DistrictAllocation(
        districts=list(districts), density=density, n_agents=n_g,
        total_n=int(n_g.sum()), attack_rate=a_g, weight=w,
    )


# ── 4. validation ───────────────────────────────────────────────────────
def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation ρ (pure-numpy, no scipy dependency).

    Returns NaN if either input has zero rank variance (all-equal).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size or x.size < 3:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank of ``a`` (ties get the mean of the spanned ranks)."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=float)
    ranks[order] = np.arange(1, a.size + 1, dtype=float)
    # average ties
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size)
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def load_disease_distribution(
    disease_nm: str,
    districts: Optional[list[str]] = None,
    *,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
) -> np.ndarray:
    """Per-gu summed annual case count for ``disease_nm`` (read-only DB).

    Args:
        disease_nm: a ``seoul_disease_district.disease_nm`` value.
        districts: gu ordering (default SEOUL_GU_ORDERED).
        year_min/year_max: optional inclusive year window (table = 2020-2024).

    Returns:
        (G,) float of summed cases per gu (0 for gu with no rows).
    """
    from simulation.database import read_only_connect

    districts = districts or list(SEOUL_GU_ORDERED)
    where = ["disease_nm = ?", "gu_nm != '서울시'"]
    args: list = [disease_nm]
    if year_min is not None:
        where.append("year >= ?")
        args.append(int(year_min))
    if year_max is not None:
        where.append("year <= ?")
        args.append(int(year_max))
    sql = (
        "SELECT gu_nm, SUM(cases) FROM seoul_disease_district "
        f"WHERE {' AND '.join(where)} GROUP BY gu_nm"
    )
    con = read_only_connect()
    try:
        rows = con.execute(sql, args).fetchall()
    finally:
        con.close()
    case_map = {r[0]: float(r[1] or 0.0) for r in rows if r and r[0]}
    return np.array([case_map.get(name, 0.0) for name in districts], dtype=float)


def validate_against_disease(
    alloc: DistrictAllocation,
    *,
    primary_disease: str = RESP_DISEASES_PRIMARY,
    robust_diseases: tuple[str, ...] = RESP_DISEASES_ROBUST,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
) -> dict:
    """Spearman ρ of the spatial weight vs observed respiratory case distribution.

    Correlates BOTH the density d_g and the simulated attack-rate weight a_g
    against the per-gu annual case distribution of the primary disease
    (pneumococcus) and a panel of robustness diseases. Honest: a low ρ is
    reported as a low number — no thresholding to "pass".

    Returns:
        dict with per-disease ρ (density-vs-cases and weight-vs-cases), the n of
        gu, and an explicit ``honest_limitation`` string.
    """
    districts = alloc.districts
    density = alloc.density
    weight = alloc.weight if alloc.weight is not None else alloc.density
    results: dict = {"n_districts": len(districts), "validations": {}}

    all_dis = [primary_disease, *robust_diseases]
    for dis in all_dis:
        cases = load_disease_distribution(
            dis, districts, year_min=year_min, year_max=year_max
        )
        rho_density = spearman_rho(density, cases)
        rho_weight = spearman_rho(weight, cases)
        results["validations"][dis] = {
            "total_cases": float(cases.sum()),
            "n_gu_with_cases": int((cases > 0).sum()),
            "spearman_density_vs_cases": rho_density,
            "spearman_weight_vs_cases": rho_weight,
            "is_primary": dis == primary_disease,
        }
    results["honest_limitation"] = (
        "Weekly per-gu ILI for 2025/26 does not exist (KDCA ILI sentinels are "
        "city-level). The spatial weight is density- and mechanism-based and is "
        "validated only INDIRECTLY against the per-gu ANNUAL distribution of "
        "notifiable respiratory diseases (2020-2024) — not a direct weekly-ILI "
        "calibration. Reported ρ values stand as measured."
    )
    return results


# ── figure ────────────────────────────────────────────────────────────────
def make_figure(
    alloc: DistrictAllocation,
    validation: dict,
    out_path: str | Path,
    *,
    primary_disease: str = RESP_DISEASES_PRIMARY,
) -> Path:
    """Two-panel figure: per-gu agent allocation bar + validation scatter.

    Left: per-gu agent count (density-allocated) vs the uniform baseline line.
    Right: density d_g vs observed primary-disease per-gu cases (Spearman ρ
    annotated). Uses matplotlib Agg + a Korean font. Honest: every value is the
    actual simulated/observed number — no placeholder.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    available = {f.name for f in fm.fontManager.ttflist}
    for cand in ("AppleGothic", "NanumGothic", "Apple SD Gothic Neo"):
        if cand in available:
            matplotlib.rcParams["font.family"] = cand
            break
    matplotlib.rcParams["axes.unicode_minus"] = False

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    G = len(alloc.districts)
    uniform = alloc.total_n / G

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))

    # Left: agent allocation bar (sorted by density desc)
    order = np.argsort(-alloc.density)
    names = [alloc.districts[i] for i in order]
    counts = alloc.n_agents[order]
    colors = plt.cm.viridis(alloc.density[order] / alloc.density.max())
    axL.bar(range(G), counts, color=colors)
    axL.axhline(uniform, color="crimson", ls="--", lw=1.5,
                label=f"uniform baseline ({uniform:.0f}/gu)")
    axL.set_xticks(range(G))
    axL.set_xticklabels(names, rotation=90, fontsize=8)
    axL.set_ylabel("agents per district (n_g)")
    axL.set_title(f"Density-proportional agent allocation (N={alloc.total_n:,})\n"
                  "color = daytime living-population density")
    axL.legend()

    # Right: density vs observed primary-disease cases scatter
    cases = load_disease_distribution(primary_disease, alloc.districts)
    rho = validation["validations"][primary_disease]["spearman_density_vs_cases"]
    rho_w = validation["validations"][primary_disease]["spearman_weight_vs_cases"]
    axR.scatter(alloc.density, cases, s=60, alpha=0.7, edgecolor="k")
    for i, nm in enumerate(alloc.districts):
        axR.annotate(nm, (alloc.density[i], cases[i]), fontsize=7,
                     xytext=(3, 3), textcoords="offset points")
    axR.set_xlabel("daytime living-population density d_g")
    axR.set_ylabel(f"{primary_disease} annual cases (2020-2024)")
    axR.set_title(f"Indirect validation (no weekly per-gu ILI exists)\n"
                  f"Spearman ρ(density)={rho:+.3f},  ρ(attack-rate)={rho_w:+.3f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    log.info("density-allocation figure written to %s", out_path)
    return out_path


# ── orchestration / artifacts ────────────────────────────────────────────
def run_and_save(
    out_dir: str | Path,
    *,
    total_n: int = 100_000,
    days: int = 180,
    seed: int = 42,
) -> dict:
    """Full pipeline: allocate → simulate → validate → write CSV + JSON.

    Writes ``district_weights.csv`` (gu, density, n_agents, attack_rate, weight)
    and ``validation.json`` (Spearman ρ vs respiratory diseases + limitation)
    under ``out_dir``. Returns the validation dict.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    alloc = run_density_abm(total_n=total_n, days=days, seed=seed)
    validation = validate_against_disease(alloc)

    # CSV
    import csv
    csv_path = out / "district_weights.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["gu", "day_livpop_density", "n_agents", "attack_rate", "weight"])
        for i, name in enumerate(alloc.districts):
            w.writerow([
                name,
                f"{alloc.density[i]:.1f}",
                int(alloc.n_agents[i]),
                f"{alloc.attack_rate[i]:.6e}",
                f"{alloc.weight[i]:.4f}",
            ])

    # JSON
    payload = {
        "total_agents": alloc.total_n,
        "total_n_requested": total_n,
        "days": days,
        "seed": seed,
        "uniform_baseline_per_gu": total_n // len(alloc.districts),
        "districts": alloc.districts,
        "density": alloc.density.tolist(),
        "n_agents": [int(x) for x in alloc.n_agents],
        "attack_rate": alloc.attack_rate.tolist(),
        "weight": alloc.weight.tolist(),
        "validation": validation,
    }
    (out / "validation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # figure (per-gu allocation bar + validation scatter)
    try:
        make_figure(alloc, validation, out / "fig_density_allocation.png")
    except Exception as e:  # figure is a side-deliverable; never block the data
        log.warning("density-allocation figure skipped: %s", e)

    log.info("density-allocation artifacts written to %s", out)
    return payload
