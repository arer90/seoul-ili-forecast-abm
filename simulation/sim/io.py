"""
simulation.sim.io
=================
DB → ``MetapopParams`` loader for the metapop SEIR-V-D simulator.

Contract
--------
- Uses ``safe_connect`` (gotcha G-117). Direct ``sqlite3.connect`` is
  forbidden anywhere in this codebase.
- Reads from the same ``commuter_matrix`` table that
  ``simulation.collectors.import_external.import_commuter_matrix`` writes.
  Authoritative schema is::

      commuter_matrix(
          id INTEGER PRIMARY KEY,
          collected_at TEXT,
          origin_gu TEXT,            -- Korean gu name ("강남구")
          dest_gu TEXT,              -- Korean gu name
          coupling REAL,             -- already a row-stochastic fraction
          night_population REAL,     -- resident population per origin_gu
          source TEXT
      )

- Populations come from ``commuter_matrix.night_population`` (single source
  of truth — keeps mobility and populations consistent). Falls back to the
  weekly mean of ``daily_population_district.tot_livpop`` (생활인구 총합),
  and finally to a flat 400k / gu constant if both are unavailable.
- Never touches write paths. This module is pure read.
"""
from __future__ import annotations

import functools
import logging
from typing import Optional

import numpy as np

from simulation.database import safe_connect
from simulation.database.config import SEOUL_GU_ORDERED

from .parameters import DiseaseParams, MetapopParams, DEFAULT_FLU_PARAMS


log = logging.getLogger(__name__)

__all__ = [
    "load_metapop_params",
    "load_populations",
    "load_mobility_matrix",
    "clear_metapop_params_cache",
]


# ── Populations ────────────────────────────────────────────────────────
def load_populations(districts: Optional[list[str]] = None) -> np.ndarray:
    """Return (G,) resident-population array aligned with ``districts``.

    Source order (first non-empty wins):
      1. ``commuter_matrix.night_population`` — single source paired with
         the coupling matrix we use for mobility.
      2. ``daily_population_district.tot_livpop`` — all-time mean per gu.
      3. Flat 400,000 / gu constant (dummy) + loud WARNING.
    """
    districts = districts or list(SEOUL_GU_ORDERED)
    G = len(districts)
    pops = np.full(G, 400_000.0)  # ultimate fallback
    pop_map: dict[str, float] = {}

    # Try commuter_matrix.night_population first (keeps pop ≤→ mobility aligned).
    try:
        with safe_connect() as con:
            cur = con.cursor()
            try:
                rows = cur.execute(
                    "SELECT origin_gu, AVG(night_population) "
                    "FROM commuter_matrix "
                    "WHERE night_population IS NOT NULL "
                    "GROUP BY origin_gu"
                ).fetchall()
                pop_map = {
                    r[0]: float(r[1]) for r in rows
                    if r and r[0] and r[1] is not None
                }
            except Exception as e:
                log.debug("commuter_matrix.night_population read failed: %s", e)

            # Fallback 2: daily_population_district.tot_livpop (all-time mean).
            if not pop_map:
                try:
                    rows = cur.execute(
                        "SELECT signgu_nm, AVG(tot_livpop) "
                        "FROM daily_population_district "
                        "WHERE tot_livpop IS NOT NULL "
                        "GROUP BY signgu_nm"
                    ).fetchall()
                    pop_map = {
                        r[0]: float(r[1]) for r in rows
                        if r and r[0] and r[1] is not None
                    }
                except Exception as e:
                    log.debug("daily_population_district read failed: %s", e)
    except Exception as e:
        log.warning("load_populations: DB read failed (%s); using 400k/gu fallback", e)
        return pops

    if not pop_map:
        log.warning(
            "load_populations: no population source found "
            "(commuter_matrix.night_population + daily_population_district "
            "both empty); using 400k/gu fallback"
        )
        return pops

    missing: list[str] = []
    for i, name in enumerate(districts):
        if name in pop_map:
            pops[i] = pop_map[name]
        else:
            missing.append(name)
    if missing:
        log.warning(
            "load_populations: %d/%d districts missing in DB (using 400k default): %s",
            len(missing), G, ", ".join(missing[:5]) + (" …" if len(missing) > 5 else "")
        )
    return pops


# ── Mobility ──────────────────────────────────────────────────────────
def load_mobility_matrix(districts: Optional[list[str]] = None) -> np.ndarray:
    """Return (G, G) row-stochastic mobility matrix.

    Source: ``commuter_matrix(origin_gu, dest_gu, coupling)``. The
    ``coupling`` column is already a fraction (essentially row-stochastic
    from the JSON import), so we just pack it into a (G, G) numpy array
    and re-normalise defensively to handle rounding.

    Fallback: uniform mixing (diag=0.9, off-diag=0.1/(G-1)) with a loud
    WARNING. A row with no DB entries is mapped to full self-loop
    (district isolated in the source data).
    """
    districts = districts or list(SEOUL_GU_ORDERED)
    G = len(districts)
    M = np.zeros((G, G), dtype=float)
    name_to_idx = {name: i for i, name in enumerate(districts)}

    try:
        with safe_connect() as con:
            cur = con.cursor()
            rows = cur.execute(
                "SELECT origin_gu, dest_gu, coupling "
                "FROM commuter_matrix "
                "WHERE coupling IS NOT NULL "
                # ORDER BY is REQUIRED for reproducibility: without it SQLite
                # returns rows in an arbitrary (process-dependent) order, so the
                # ``M[i, j] += coupling`` accumulation rounds differently each
                # process (~1e-16 mobility jitter). The metapop SEIR amplifies
                # that jitter into a cross-process blow-up for some parameter
                # regimes. A fixed row order makes the matrix bit-identical.
                "ORDER BY origin_gu, dest_gu"
            ).fetchall()
            if not rows:
                raise ValueError("commuter_matrix is empty or coupling all-null")
            unknown: set[str] = set()
            for origin, dest, coupling in rows:
                i = name_to_idx.get(origin)
                j = name_to_idx.get(dest)
                if i is None:
                    unknown.add(str(origin))
                    continue
                if j is None:
                    unknown.add(str(dest))
                    continue
                if coupling is None:
                    continue
                M[i, j] += float(coupling)
            if unknown:
                log.warning(
                    "load_mobility_matrix: %d gu names in DB not in districts list: %s",
                    len(unknown),
                    ", ".join(sorted(unknown)[:5])
                    + (" …" if len(unknown) > 5 else ""),
                )
    except Exception as e:
        log.warning(
            "load_mobility_matrix: falling back to uniform mixing (%s)", e
        )
        M = np.full((G, G), 0.1 / (G - 1))
        np.fill_diagonal(M, 0.9)
        return M

    # Row-stochastic defensive re-normalisation. Rows that are all-zero
    # (district not in the commuter JSON) become pure self-loops so the
    # row-sum = 1 invariant holds for the simulator's validate().
    row_sums = M.sum(axis=1)
    zero_rows = row_sums == 0
    if zero_rows.any():
        log.warning(
            "load_mobility_matrix: %d rows with no commuter entries → "
            "pure self-loop fallback for those gu",
            int(zero_rows.sum()),
        )
        M[zero_rows] = 0.0
        M[zero_rows, np.arange(G)[zero_rows]] = 1.0
        row_sums = M.sum(axis=1)
    M = M / row_sums[:, None]
    return M


# ── Cached DB fetch ────────────────────────────────────────────────────
# The (pops, M) pair is the expensive part of load_metapop_params — measured
# at ≈ 3.2 s per call on SQLite.  Commuter matrix + resident populations
# change at most monthly (when import_external runs again), so caching for
# the lifetime of a process is safe.  We keep both arrays read-only so
# callers cannot accidentally corrupt the cache; MetapopParams.__post_init__
# copies them anyway, and MetapopSEIRVD.__init__ calls .astype(float) which
# creates a fresh mutable array.
#
# Cache key is the districts tuple so alternative district orderings get
# independent cache slots.  maxsize=4 covers: default SEOUL_GU_ORDERED +
# a handful of test/subset variants.
#
# Codex non-bio review #10 (sprint 2026-05-06): RACE CONDITION WARNING
# -------------------------------------------------------------------
# Per-process cache. If `simulation collect --groups all` runs concurrently
# with a long sim that has populated this cache, the sim keeps the stale
# (pops, M) snapshot until process exit or `clear_metapop_params_cache()`.
# SQLite WAL mode (`safe_connect`) handles concurrent reads, but the
# python-level cache does NOT auto-invalidate on DB writes. For production
# multi-user ops, wrap with a `db_path.stat().st_mtime` watchdog; for now,
# call `clear_metapop_params_cache()` explicitly after DB-modifying steps.
@functools.lru_cache(maxsize=4)
def _cached_db_params(districts_tuple: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Cached (populations, mobility) for the given districts ordering."""
    districts = list(districts_tuple)
    log.info(
        "load_metapop_params cache MISS — loading %d districts from DB",
        len(districts),
    )
    pops = load_populations(districts)
    M = load_mobility_matrix(districts)
    pops.flags.writeable = False
    M.flags.writeable = False
    return pops, M


def clear_metapop_params_cache() -> None:
    """Invalidate the (pops, M) cache.  Call after DB imports that change
    commuter_matrix or daily_population_district, or in test fixtures.
    """
    _cached_db_params.cache_clear()


# ── Top-level convenience constructor ─────────────────────────────────
def load_metapop_params(
    *,
    disease: Optional[DiseaseParams] = None,
    seed_infected: float = 10.0,
    seed_district: str = "강남구",
    days: int = 200,
    dt: float = 0.25,
    districts: Optional[list[str]] = None,
) -> MetapopParams:
    """Convenience constructor: populate ``MetapopParams`` from the DB.

    The single seed of infection is placed in ``seed_district`` (must be
    present in ``districts``) so scenarios that compare NPI vs vax vs
    baseline start from the identical initial condition.

    Expensive DB reads (populations, mobility) are cached per
    ``districts`` tuple — see ``_cached_db_params``.  Per-call state
    (seed, horizon) is built fresh each time.
    """
    districts = districts or list(SEOUL_GU_ORDERED)
    pops_ro, M_ro = _cached_db_params(tuple(districts))

    G = len(districts)
    I0 = np.zeros(G)
    try:
        idx = districts.index(seed_district)
    except ValueError:
        log.warning(
            "seed_district %r not in districts; seeding gu_0 instead", seed_district
        )
        idx = 0
    I0[idx] = float(seed_infected)

    # MetapopParams constructor + downstream code mutate mobility / pops
    # (e.g. MetapopSEIRVD.__init__ normalises, astype(float) copies).
    # We hand out a writable copy so the cached read-only views stay
    # pristine.
    return MetapopParams(
        disease=disease or DEFAULT_FLU_PARAMS,
        populations=pops_ro.copy(),
        mobility=M_ro.copy(),
        district_names=list(districts),
        initial_infected=I0,
        days=days,
        dt=dt,
        seed=42,
    )
