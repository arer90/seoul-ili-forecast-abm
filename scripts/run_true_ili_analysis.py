"""
End-to-end TRUE ILI cohort analysis using new modules (§8 paper-grade).

WIRES TOGETHER:
- simulation.pipeline.phase15_true_ili_cohort (cohort definition + data loading)
- simulation.analytics.dtw_plv (DTW + PLV + Mantel + cophenetic)
- simulation.analytics.eda_equal_footing (5 EDA figures)

USAGE:
    .venv/bin/python scripts/run_true_ili_analysis.py [--cohort I-A|I-B] [--out PATH]

OUTPUT:
    simulation/results/phase15_cross_country/
    ├── true_ili_<cohort>_dtw.csv
    ├── true_ili_<cohort>_plv.csv
    ├── true_ili_<cohort>_kr_neighbors.json
    ├── true_ili_<cohort>_mantel.json
    └── figures/eda_<cohort>_*.png  (4 EDA figures)

REPLACES: /tmp/cross_country_smoke.py + /tmp/ili_cohort_v2.py + /tmp/eda_28country_equal.py
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np

from simulation.pipeline.phase15_true_ili_cohort import (
    COHORT_INFO, get_cohort_ia, get_cohort_ib, load_country_ili,
)
from simulation.analytics.dtw_plv import (
    cophenetic_correlation, dtw_matrix, mantel_test, plv_matrix, zscore_interp,
)
from simulation.analytics.eda_equal_footing import generate_all_eda_figures
from simulation.database import safe_connect  # G-116 SSOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

REPO = Path(__file__).parent.parent
DB = REPO / "simulation" / "data" / "db" / "epi_real_seoul.db"
OUT_DEFAULT = REPO / "simulation" / "results" / "phase15_cross_country"

# Capital latitudes for Mantel test (geographic structure)
CAPITAL_LAT = {
    "AT": 48.21, "AU": -35.28, "BE": 50.85, "CN": 39.90, "CZ": 50.08,
    "DE": 52.52, "DK": 55.68, "EE": 59.44, "ES": 40.42, "FI": 60.17,
    "FR": 48.86, "GB": 51.51, "GR": 37.98, "HK": 22.32, "HR": 45.81,
    "HU": 47.50, "IE": 53.35, "IS": 64.13, "IT": 41.90, "JP": 35.68,
    "KR": 37.57, "LT": 54.69, "LU": 49.61, "LV": 56.95, "MT": 35.90,
    "NL": 52.37, "NO": 59.91, "PL": 52.23, "PT": 38.72, "RO": 44.43,
    "SE": 59.33, "SG": 1.35, "SI": 46.06, "SK": 48.15, "US": 38.91,
}


def align_series(rows, year_min, year_max):
    n = (year_max - year_min + 1) * 53
    arr = np.full(n, np.nan)
    for y, w, v in rows:
        if year_min <= y <= year_max and 1 <= w <= 53:
            arr[(y - year_min) * 53 + (w - 1)] = v
    return arr


def run_cohort(cohort_name: str, out_dir: Path):
    """Run TRUE ILI cohort analysis end-to-end."""
    log.info(f"=== Cohort {cohort_name} ===")
    info = COHORT_INFO[cohort_name]
    period_yr = (2019, 2025) if cohort_name == "I-A" else (2021, 2025)
    period_label = info["period"]

    conn = safe_connect(str(DB))
    try:
        countries = get_cohort_ia() if cohort_name == "I-A" else get_cohort_ib(conn)
        log.info(f"Cohort: {len(countries)} countries — {countries}")

        raw_arrs, z_data, sources = {}, {}, {}
        for c in countries:
            rows, src = load_country_ili(conn, c, *period_yr)
            arr = align_series(rows, *period_yr)
            z = zscore_interp(arr)
            if z is None:
                log.warning(f"  {c}: SKIP (insufficient data, src={src})")
                continue
            raw_arrs[c] = arr; z_data[c] = z; sources[c] = src
            log.info(f"  {c}: src={src}, n_valid={int(np.isfinite(arr).sum())}")
    finally:
        conn.close()

    aligned = sorted(z_data.keys())
    log.info(f"Aligned: {len(aligned)}/{len(countries)} countries")

    # DTW + PLV via new module
    log.info("Computing DTW + PLV matrices...")
    D, _ = dtw_matrix(z_data, window=8)
    P, _ = plv_matrix(z_data, fs=52, band_low=0.5, band_high=2.0)
    coph_d = cophenetic_correlation(D)
    coph_p = cophenetic_correlation(1 - P)
    log.info(f"Cophenetic DTW={coph_d:.4f}, PLV={coph_p:.4f}")

    # KR's neighbors
    kr_results = {}
    if "KR" in aligned:
        kr_i = aligned.index("KR")
        dtw_kr = sorted([(aligned[j], float(D[kr_i, j])) for j in range(len(aligned)) if j != kr_i],
                         key=lambda x: x[1])
        plv_kr = sorted([(aligned[j], float(P[kr_i, j])) for j in range(len(aligned)) if j != kr_i],
                         key=lambda x: -x[1])
        kr_results = {"dtw_neighbors": [{"country": c, "dtw": d} for c, d in dtw_kr],
                       "plv_neighbors": [{"country": c, "plv": p} for c, p in plv_kr]}
        log.info(f"KR DTW top 3: {dtw_kr[:3]}")
        log.info(f"KR PLV top 3: {plv_kr[:3]}")

    # Mantel test (DTW vs geographic latitude diff)
    mantel = None
    if all(c in CAPITAL_LAT for c in aligned):
        N = len(aligned)
        geo = np.zeros((N, N))
        for i, ci in enumerate(aligned):
            for j, cj in enumerate(aligned):
                geo[i, j] = abs(CAPITAL_LAT[ci] - CAPITAL_LAT[cj])
        mantel = mantel_test(D, geo, n_permutations=1000, seed=42)
        log.info(f"Mantel (DTW vs lat): r={mantel['r']:.4f}, p={mantel['p_permutation']:.4f}")

    # Save matrices CSV
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, mat in [("dtw", D), ("plv", P)]:
        with open(out_dir / f"true_ili_{cohort_name}_{name}.csv", "w") as f:
            w = csv.writer(f)
            w.writerow(["country"] + aligned)
            for i, ci in enumerate(aligned):
                w.writerow([ci] + [f"{v:.4f}" for v in mat[i]])
        log.info(f"✓ true_ili_{cohort_name}_{name}.csv")

    # Save JSON summary
    summary = {
        "cohort": cohort_name, "info": info, "n_aligned": len(aligned),
        "countries": aligned, "sources": sources, "period_years": period_yr,
        "cophenetic": {"dtw": coph_d, "plv": coph_p},
        "kr_neighbors": kr_results, "mantel_dtw_vs_lat": mantel,
    }
    with open(out_dir / f"true_ili_{cohort_name}_summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log.info(f"✓ true_ili_{cohort_name}_summary.json")

    # EDA figures via new module
    log.info("Generating EDA figures...")
    fig_dir = out_dir / "figures"
    period_full = (period_yr[1] - period_yr[0] + 1) * 53
    generate_all_eda_figures(z_data, raw_arrs, sources, fig_dir,
                              cohort_name=cohort_name, period_label=period_label,
                              period_full_weeks=period_full)
    log.info(f"✓ EDA figures in {fig_dir}/")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["I-A", "I-B", "both"], default="both")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()

    if args.cohort in ("I-A", "both"):
        run_cohort("I-A", args.out)
    if args.cohort in ("I-B", "both"):
        run_cohort("I-B", args.out)
    log.info(f"=== Done. Output: {args.out} ===")


if __name__ == "__main__":
    main()
