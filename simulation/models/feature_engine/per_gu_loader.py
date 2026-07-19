"""Per-gu weekly feature loader for GE-DNN / variants.

Purpose
-------
Build a (T_weeks, 25_gu, K_features) tensor of per-gu node features so that
GE-DNN can learn meaningful spatial structure instead of the degenerate
broadcast-same-features-to-all-nodes layout of v1.

Sources (per ENGINEERING_PRINCIPLES.md data_rules + DB audit 2026-04-21):
 - daily_population_district : 2018-04-05 ~ present, daily, 25 real gu
 columns: tot/day/night/inflow/move_livpop
 - kosis_age_district : 2015-2026, yearly, 25 gu
 columns: age_group × population (we aggregate to 5 coarse bins)
 - seoul_disease_district : 2020-2024, yearly, 25 gu
 columns: ILI-adjacent disease cases (2015-19 missing → carry-back)
 - school_info_seoul : static, count schools per gu
 - commuter_matrix : 25x25 static OD (for the GCN graph itself)

The loader returns
 X_gu ∈ (T, 25, K) float32, z-scored per-feature (train mean/std)
 week_labels ∈ (T) 'YYYY-Www'
 gu_order ∈ (25) list of gu names in the same order as the
 commuter_matrix adjacency rows/cols.

Design notes
------------
1. Weekly aggregation of daily population: mean over Mon-Sun.
2. Forward-fill for yearly sources when week-start year is known, then
 back-fill pre-2018 weeks to the earliest observed value (constant
 extrapolation — flagged in logs).
3. Z-scoring is fit on TRAINING WEEKS ONLY (prevents leakage); call
 ``fit_transform_train`` on train slice, then ``transform`` on val+test.
4. 25 gu ordering follows ``SEOUL_GU_25`` from ``simulation.database.config``
 to align 1-to-1 with ``commuter_matrix`` row/col indices.
5. NaN policy: any remaining NaN after forward-fill + back-fill is set to
 0 (after z-scoring, 0 ≈ the training mean, so this is a benign prior).

This module is ADDITIVE — it does not alter any existing feature engine;
 GE-DNN continues to use the aggregate flat X.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# Canonical 25-gu ordering (matches commuter_matrix indices).
_SEOUL_GU_25 = [
    "종로구", "중구", "용산구", "성동구", "광진구",
    "동대문구", "중랑구", "성북구", "강북구", "도봉구",
    "노원구", "은평구", "서대문구", "마포구", "양천구",
    "강서구", "구로구", "금천구", "영등포구", "동작구",
    "관악구", "서초구", "강남구", "송파구", "강동구",
]


@dataclass
class PerGuFeatureBundle:
    """Container for per-gu feature tensor + metadata."""

    X_gu: np.ndarray          # (T, 25, K) float32
    week_labels: list[str]    # (T,) "YYYY-Www"
    gu_order: list[str]       # (25,) gu names, matches commuter_matrix
    feature_names: list[str]  # (K,) descriptive labels per channel
    train_mu: Optional[np.ndarray] = None   # (K,) used for z-score
    train_sd: Optional[np.ndarray] = None

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.X_gu.shape

    def fit_transform_train(self, n_train: int) -> "PerGuFeatureBundle":
        """Z-score using training weeks [0:n_train] only, return new bundle."""
        train = self.X_gu[:n_train]             # (n_train, 25, K)
        mu = train.mean(axis=(0, 1))            # (K,)
        sd = train.std(axis=(0, 1)) + 1e-6
        Xn = (self.X_gu - mu) / sd
        return PerGuFeatureBundle(
            X_gu=Xn.astype(np.float32),
            week_labels=self.week_labels,
            gu_order=self.gu_order,
            feature_names=self.feature_names,
            train_mu=mu.astype(np.float32),
            train_sd=sd.astype(np.float32),
        )


def _iso_week_label(year: int, week: int) -> str:
    return f"{year}-W{week:02d}"


def _week_date_range(start_ym: tuple[int, int], end_ym: tuple[int, int]) -> list[str]:
    """Generate inclusive list of ISO-week labels between (year, week) endpoints."""
    labels = []
    y, w = start_ym
    ey, ew = end_ym
    while (y, w) <= (ey, ew):
        labels.append(_iso_week_label(y, w))
        w += 1
        # ISO weeks run 1-52 or 1-53. Use polars/pandas-like rollover. For
        # simplicity we use a fixed 52-week year here; weeks 53 are rare and
        # are merged into week 52 of the same year (acceptable for the
        # per-gu feature pipeline where daily sources are already coarse).
        if w > 52:
            w = 1
            y += 1
    return labels


def _daily_to_weekly_population(con) -> tuple[np.ndarray, list[str], list[str]]:
    """Aggregate daily_population_district → (T, 25, 5) weekly mean."""
    import pandas as pd

    rows = con.execute(
        "SELECT stdr_de, signgu_code, signgu_nm, "
        "       tot_livpop, day_livpop, night_livpop, inflow_livpop, move_livpop "
        "FROM daily_population_district "
        "WHERE signgu_code != '11000'"  # drop aggregate row
    ).fetchall()
    if not rows:
        raise RuntimeError("daily_population_district empty")

    df = pd.DataFrame(
        rows,
        columns=[
            "stdr_de", "signgu_code", "signgu_nm",
            "tot_livpop", "day_livpop", "night_livpop", "inflow_livpop", "move_livpop",
        ],
    )
    df["date"] = pd.to_datetime(df["stdr_de"].astype(str), format="%Y%m%d")
    # ISO year/week
    iso = df["date"].dt.isocalendar()
    df["year"] = iso["year"].astype(int)
    df["week"] = iso["week"].astype(int)
    df["wk"] = df.apply(lambda r: _iso_week_label(r["year"], r["week"]), axis=1)

    pop_cols = ["tot_livpop", "day_livpop", "night_livpop", "inflow_livpop", "move_livpop"]
    weekly = (
        df.groupby(["wk", "signgu_nm"])[pop_cols]
        .mean()
        .reset_index()
    )
    week_labels = sorted(weekly["wk"].unique())
    gu_names = _SEOUL_GU_25  # canonical ordering
    W = len(week_labels)
    K = len(pop_cols)

    tensor = np.full((W, 25, K), np.nan, dtype=np.float32)
    wk_idx = {w: i for i, w in enumerate(week_labels)}
    gu_idx = {g: i for i, g in enumerate(gu_names)}
    hits, misses = 0, 0
    for _, r in weekly.iterrows():
        wi = wk_idx.get(r["wk"])
        gi = gu_idx.get(r["signgu_nm"])
        if wi is None or gi is None:
            misses += 1
            continue
        for k, col in enumerate(pop_cols):
            tensor[wi, gi, k] = r[col]
        hits += 1
    log.info(
        f"[per_gu] daily→weekly pop aggregated: "
        f"shape={tensor.shape} hits={hits} missed_gu_join={misses}"
    )
    feat_names = [f"pop_{c}" for c in pop_cols]
    return tensor, week_labels, feat_names


def _yearly_age_to_weekly(con, week_labels: list[str]) -> tuple[np.ndarray, list[str]]:
    """Expand kosis_age_district (yearly, 25 gu, age groups) to weekly tensor.

    Aggregate age bins into 5 coarse bins: [0-9, 10-19, 20-39, 40-59, 60+]
    so the feature count stays modest (25 gu × 5 = 125 channels before
    weekly broadcast).
    """
    import pandas as pd

    rows = con.execute(
        "SELECT prd_de, gu_nm, age_group, population FROM kosis_age_district"
    ).fetchall()
    df = pd.DataFrame(rows, columns=["year", "gu_nm", "age_group", "population"])
    df["year"] = df["year"].astype(int)
    # Map age_group strings → coarse bin.
    def _bin(ag: str) -> str:
        ag = str(ag)
        if ag.startswith("0-") or ag.startswith("5-"):
            return "0_9"
        if ag.startswith("10-"):
            return "10_19"
        if ag.startswith("15-") or ag.startswith("20-") or ag.startswith("25-") or ag.startswith("30-") or ag.startswith("35-"):
            return "20_39"
        if ag.startswith("40-") or ag.startswith("45-") or ag.startswith("50-") or ag.startswith("55-"):
            return "40_59"
        return "60_plus"

    df["bin"] = df["age_group"].apply(_bin)
    yearly = df.groupby(["year", "gu_nm", "bin"])["population"].sum().reset_index()
    bins = ["0_9", "10_19", "20_39", "40_59", "60_plus"]
    yrs = sorted(yearly["year"].unique())
    K = len(bins)
    # yearly_tensor: (Y, 25, K)
    yearly_tensor = np.full((len(yrs), 25, K), np.nan, dtype=np.float32)
    y_idx = {y: i for i, y in enumerate(yrs)}
    g_idx = {g: i for i, g in enumerate(_SEOUL_GU_25)}
    b_idx = {b: i for i, b in enumerate(bins)}
    for _, r in yearly.iterrows():
        yi = y_idx.get(int(r["year"]))
        gi = g_idx.get(r["gu_nm"])
        bi = b_idx.get(r["bin"])
        if None in (yi, gi, bi):
            continue
        yearly_tensor[yi, gi, bi] = r["population"]
    # Expand to weeks
    W = len(week_labels)
    weekly_tensor = np.full((W, 25, K), np.nan, dtype=np.float32)
    for wi, lbl in enumerate(week_labels):
        year = int(lbl.split("-")[0])
        yi = y_idx.get(year)
        if yi is None:
            # pre-first-year: use earliest
            yi = 0
        weekly_tensor[wi] = yearly_tensor[yi]
    feat_names = [f"age_{b}" for b in bins]
    log.info(f"[per_gu] yearly age → weekly: shape={weekly_tensor.shape}")
    return weekly_tensor, feat_names


def _parse_gu_from_address(addr: str) -> Optional[str]:
    """Extract Seoul gu name from a full-address string.

    The school_info_seoul.gu_name column is misleading — it actually stores
    the full address (``서울특별시 XX구 ...``). We find whichever of the 25
    canonical gu names is present in the string.
    """
    if not isinstance(addr, str):
        return None
    for g in _SEOUL_GU_25:
        if g in addr:
            return g
    return None


def _school_static(con) -> tuple[np.ndarray, list[str]]:
    """Count schools per gu × type → static (25, K_school).

    Note: ``school_info_seoul.gu_name`` stores full addresses; we parse the
    actual gu name with :func:`_parse_gu_from_address`.
    """
    import pandas as pd

    rows = con.execute(
        "SELECT gu_name, school_type FROM school_info_seoul"
    ).fetchall()
    df = pd.DataFrame(rows, columns=["addr", "school_type"])
    df["gu_nm"] = df["addr"].apply(_parse_gu_from_address)
    before = len(df)
    df = df.dropna(subset=["gu_nm", "school_type"])
    after = len(df)
    log.info(f"[per_gu] school address→gu parse: kept {after}/{before}")

    agg = df.groupby(["gu_nm", "school_type"]).size().reset_index(name="n")
    types = sorted(agg["school_type"].dropna().unique())
    K = len(types) + 1  # +1 for total
    tensor = np.zeros((25, K), dtype=np.float32)
    g_idx = {g: i for i, g in enumerate(_SEOUL_GU_25)}
    t_idx = {t: i for i, t in enumerate(types)}
    for _, r in agg.iterrows():
        gi = g_idx.get(r["gu_nm"])
        ti = t_idx.get(r["school_type"])
        if gi is None or ti is None:
            continue
        tensor[gi, ti] = r["n"]
        tensor[gi, -1] += r["n"]
    feat_names = [f"school_{t}" for t in types] + ["school_total"]
    log.info(f"[per_gu] school static: shape={tensor.shape} K_types={len(types)}")
    return tensor, feat_names


def _fill_nan_weekly(tensor: np.ndarray) -> np.ndarray:
    """Forward-fill then back-fill NaN along the week axis (T).

    Operates on shape (T, 25, K). If a gu × feature is all NaN, fills with 0.
    """
    T = tensor.shape[0]
    out = tensor.copy()
    # Forward fill
    for t in range(1, T):
        mask = np.isnan(out[t])
        if mask.any():
            out[t] = np.where(mask, out[t - 1], out[t])
    # Back fill for leading NaNs
    for t in range(T - 2, -1, -1):
        mask = np.isnan(out[t])
        if mask.any():
            out[t] = np.where(mask, out[t + 1], out[t])
    # Residual NaN → 0
    out = np.where(np.isnan(out), 0.0, out)
    return out.astype(np.float32)


def build_per_gu_bundle(
    target_week_labels: Optional[list[str]] = None,
) -> PerGuFeatureBundle:
    """Construct the full per-gu feature tensor.

    Args:
        target_week_labels: If given, slice/align tensor to these labels
            (drops any weeks we can't materialise). Useful for aligning
            with the existing aggregate y_train timeline.

    Returns:
        PerGuFeatureBundle with X_gu of shape (T, 25, K).
    """
    from simulation.database import safe_connect

    with safe_connect() as con:
        pop_T, pop_weeks, pop_names = _daily_to_weekly_population(con)
        age_T, age_names = _yearly_age_to_weekly(con, pop_weeks)
        school_static, school_names = _school_static(con)

    # Align static school → per-week broadcast.
    school_T = np.broadcast_to(
        school_static[None, :, :], (len(pop_weeks), 25, school_static.shape[1])
    ).astype(np.float32).copy()

    stacked = np.concatenate([pop_T, age_T, school_T], axis=-1)  # (T, 25, K)
    stacked = _fill_nan_weekly(stacked)
    feat_names = pop_names + age_names + school_names

    if target_week_labels is not None:
        idx = {w: i for i, w in enumerate(pop_weeks)}
        hits, miss = 0, 0
        rows = []
        keep_labels = []
        for w in target_week_labels:
            if w in idx:
                rows.append(stacked[idx[w]])
                keep_labels.append(w)
                hits += 1
            else:
                miss += 1
        if not rows:
            raise RuntimeError(
                "No target weeks overlap with daily_population_district range "
                f"({pop_weeks[0]} .. {pop_weeks[-1]})"
            )
        stacked = np.stack(rows, axis=0)
        week_labels = keep_labels
        log.info(f"[per_gu] aligned to target: kept {hits}, dropped {miss}")
    else:
        week_labels = pop_weeks

    return PerGuFeatureBundle(
        X_gu=stacked,
        week_labels=week_labels,
        gu_order=_SEOUL_GU_25,
        feature_names=feat_names,
    )


__all__ = [
    "PerGuFeatureBundle",
    "build_per_gu_bundle",
]
