"""simulation.abm.network_params_from_db
==========================================
DATA-DERIVED, strictly LEAK-FREE contact-network structure parameters.

The contact network (:func:`simulation.abm.contact_network.build_multilayer_network`)
previously hard-coded its structure (``hh_size=(2,4)``, ``community_mean_degree=4.0``,
…). This module replaces those with values derived from the real Seoul DB and from
cited external census/contact-survey constants — never from the epidemic outcome.

Leak-free contract (enforced by the SQL WHERE clauses + guarded by
``tests/test_network_params_from_db.py``):
  * Every DB read is filtered to rows strictly BEFORE the forecast origin
    (``stdr_de < forecast_origin``; ``prd_de <= census_year`` / ``latest_emp_period``).
    The 2026-02-16.. forward window is excluded.
  * No formula reads any ILI / positivity / forward-observation column.
  * No parameter is selected to improve the forward score — the DB supplies only
    *structural* population/mobility/institution signals; the absolute contact
    rates come from documented external surveys, not from a forward-R² sweep.

DB-derived (per-gu, forecast-time structural):
  * ``community_mean_degree`` — per-gu casual-contact degree scaled by living-
    population day/night mobility (``daily_population_district``). Denser daytime-
    influx districts (CBD: 중구/종로/강남) get more casual contacts than residential
    ones (은평/관악), matching the intended density effect.

External, cited (NOT in the 85-table DB — documented, not tuned):
  * ``hh_size`` — Statistics Korea 인구주택총조사: Seoul mean ≈ 2.1 persons/household
    (the old hard-coded (2,4)→mean ~3 over-estimates real Seoul).
  * ``community`` absolute base — POLYMOD-class contact surveys report ~5–10 casual
    ("other/leisure") contacts per day; used as the Seoul-median anchor.
  * ``class_size`` / ``work_size`` — KEDI 학급당 학생수 / KOSIS firm-size means.
"""
from __future__ import annotations

import numpy as np

__all__ = ["derive_network_kwargs", "derive_beta_by_layer", "EXTERNAL_CONSTANTS"]

# ── External, documented constants (NOT in the DB; sourced, not tuned) ─────────
EXTERNAL_CONSTANTS = {
    "hh_size_range": (1, 3),          # KOSIS 인구주택총조사 Seoul mean ≈ 2.1 persons/hh
    "hh_size_source": "Statistics Korea Population & Housing Census (KOSIS 101), Seoul ~2.1/hh",
    "community_base_degree": 8.0,     # POLYMOD-class survey: ~5–10 casual contacts/day
    "community_base_source": "Mossong et al. 2008 (POLYMOD) + KR contact surveys, casual/other ~5–10/day",
    "class_size_range": (20, 28),     # KEDI 교육통계 학급당 학생수 (초 22 / 중 26 / 고 24)
    "class_size_source": "KEDI 교육통계 학급당 학생수 (Seoul 초~22 중~26 고~24)",
    "work_size_range": (3, 12),       # KOSIS 사업체조사 firm-size (dominant-small, ~5-6 mean)
    "work_size_source": "KOSIS 전국사업체조사 firm-size (Korea dominant-small ~5-6 employees)",
    # Relative PER-CONTACT transmission by setting (contact intensity = duration ×
    # physical proximity). Household is the reference (longest, closest, most
    # physical repeated contacts → highest per-contact hazard); community casual
    # contacts are brief → lowest. Documented, NOT tuned to the forward score.
    "per_contact_transmission_weights": {
        "household": 1.00, "school": 0.60, "workplace": 0.45, "community": 0.25},
    "per_contact_transmission_source": (
        "relative per-contact transmission by setting from contact-intensity "
        "(duration × physical proximity), POLYMOD (Mossong et al. 2008) contact "
        "surveys + influenza household secondary-attack-rate studies; household "
        "reference (longest/closest), community casual lowest"),
}


def derive_beta_by_layer(beta: float, layer_degrees: dict,
                         weights: dict | None = None) -> dict:
    """Redistribute a single aggregate ``beta`` into SCALE-PRESERVING per-layer betas.

    The kernel today gives every layer the same per-edge hazard ``beta/deg_total``
    (agent_kernel.py:315), which ignores the well-documented fact that a household
    contact transmits influenza far more readily per contact than a brief community
    contact. This fuses the empirical per-contact intensity ordering (household >
    school > workplace > community, cited constants) into the layer hazards WITHOUT
    changing the aggregate force of infection — so the forward aggregate curve is
    (near) unchanged while the mechanism, per-layer shares, and any layer-resolved
    counterfactual become realistic.

    The aggregate per-agent force is ``prevalence · Σ_L deg_L · beta_L``; the uniform
    baseline makes that ``prevalence · beta``. Choosing ``beta_L = beta · w_L / Σ_k
    deg_k w_k`` keeps ``Σ_L deg_L · beta_L = beta`` exactly (redistribution only),
    while higher-intensity layers (larger ``w_L``) get a larger per-contact hazard.

    Args:
        beta: aggregate transmission scale (the same ``beta`` the uniform kernel path
            divides by ``deg_total``); ``beta ≥ 0``.
        layer_degrees: ``{layer_name: mean_degree}`` per contact layer (e.g. from
            :func:`simulation.abm.contact_network.degree_summary`, excluding
            ``_total``). Degrees must be ``> 0``.
        weights: optional ``{layer_name: relative_per_contact_hazard}``; defaults to
            the cited ``EXTERNAL_CONSTANTS['per_contact_transmission_weights']``.
            Equal weights reproduce the uniform ``beta/deg_total`` baseline exactly.

    Returns:
        ``{layer_name: beta_L}`` with ``Σ_L deg_L · beta_L == beta`` (to float
        precision) and per-contact hazard ordered by ``weights``.

    Raises:
        ValueError: if a layer is missing a weight or ``Σ_k deg_k w_k <= 0``.

    Performance: O(#layers). Side effects: none (pure).
    Caller responsibility: ``layer_degrees`` values > 0; ``beta ≥ 0``.
    """
    w = weights or EXTERNAL_CONSTANTS["per_contact_transmission_weights"]
    layers = list(layer_degrees)
    missing = [L for L in layers if L not in w]
    if missing:
        raise ValueError(f"no per-contact weight for layer(s) {missing}")
    z = float(sum(float(layer_degrees[L]) * float(w[L]) for L in layers))
    if z <= 0.0:
        raise ValueError("Σ deg·weight must be > 0 to preserve scale")
    return {L: float(beta) * float(w[L]) / z for L in layers}


def _sorted_gu_names(conn) -> tuple[str, ...]:
    """Seoul gu names in the SAME order as synthetic_population home_gu codes.

    ``synthetic_population._load_gu_names`` returns ``tuple(sorted(normalized gu))``
    (from commuter_matrix or, as fallback, kosis_age_district). Both sources are the
    same 25 Seoul gu, so ``sorted(kosis gu)`` reproduces the home_gu code order —
    read here positionally so it works with any DB connection.
    """
    from simulation.abm.synthetic_population import _normalize_gu_name
    rows = conn.execute(
        "SELECT DISTINCT gu_nm FROM kosis_age_district WHERE gu_nm IS NOT NULL"
    ).fetchall()
    names = {_normalize_gu_name(r[0]) for r in rows if r[0]}
    gu_names = tuple(sorted(names))
    if len(gu_names) != 25:
        raise RuntimeError(f"expected 25 Seoul gu, found {len(gu_names)}")
    return gu_names


def derive_network_kwargs(db_path=None, *, forecast_origin: str = "20260216",
                          window_start: str = "20250101") -> dict:
    """Derive leak-free contact-network structure kwargs from the Seoul DB.

    Args:
        db_path: SQLite path (default ``simulation.database.config.DB_PATH``).
        forecast_origin: ISO ``yyyymmdd``; DB rows on/after this are EXCLUDED
            (the forward window must never inform the structure).
        window_start: earliest ``stdr_de`` to average mobility over.

    Returns:
        ``{"community_mean_degree": np.ndarray[25], "hh_size", "work_size",
        "class_size", "provenance": {...}}`` — a ``build_multilayer_network``-ready
        kwargs dict. ``community_mean_degree`` is per-gu, indexed to match
        ``home_gu`` codes; the size ranges are Seoul-wide external constants.

    Raises:
        RuntimeError: if the DB lacks the 25 gu mobility rows.

    Side effects: one read-only DB open (closed here). No writes.
    """
    from simulation.database import read_only_connect
    from simulation.database.config import DB_PATH
    from simulation.abm.synthetic_population import _normalize_gu_name

    con = read_only_connect(str(db_path or DB_PATH))
    try:
        gu_names = _sorted_gu_names(con)
        rows = con.execute(
            """
            SELECT signgu_nm,
                   AVG(day_livpop) / NULLIF(AVG(night_livpop), 0) AS dn
            FROM daily_population_district
            WHERE stdr_de < ? AND stdr_de >= ? AND signgu_nm <> '서울시'
            GROUP BY signgu_nm
            """,
            (forecast_origin, window_start),
        ).fetchall()
    finally:
        con.close()

    dn_by_gu = {_normalize_gu_name(r[0]): float(r[1]) for r in rows if r[1] is not None}
    if len(dn_by_gu) < len(gu_names):
        raise RuntimeError(
            f"mobility data covers {len(dn_by_gu)}/{len(gu_names)} gu; cannot derive "
            "per-gu community degree")

    activity = np.array([dn_by_gu[g] for g in gu_names], dtype=np.float64)
    median = float(np.median(activity))
    base = EXTERNAL_CONSTANTS["community_base_degree"]
    # per-gu casual degree: literature base × relative mobility, bounded to a
    # plausible ±50% band (structural, NOT fit to the forward score).
    degree = np.clip(base * activity / max(median, 1e-9), base * 0.5, base * 1.5)

    return {
        "community_mean_degree": degree,                      # DB-derived, per-gu
        "hh_size": EXTERNAL_CONSTANTS["hh_size_range"],       # external, cited
        "work_size": EXTERNAL_CONSTANTS["work_size_range"],   # external, cited
        "class_size": EXTERNAL_CONSTANTS["class_size_range"], # external, cited
        "provenance": {
            "community_mean_degree": "DB daily_population_district day/night mobility "
            f"(leak-free: stdr_de<{forecast_origin}); per-gu, base={base} from "
            f"{EXTERNAL_CONSTANTS['community_base_source']}",
            "hh_size": EXTERNAL_CONSTANTS["hh_size_source"],
            "work_size": EXTERNAL_CONSTANTS["work_size_source"],
            "class_size": EXTERNAL_CONSTANTS["class_size_source"],
            "leak_free": f"all inputs structural, forward window (>= {forecast_origin}) "
                         "excluded; no forward-ILI column read; no param selected on forward-R²",
        },
    }
