"""
Cross-country DTW + PLV + Mantel analytics — paper-grade module.

Status: NEW (2026-05-27) — wiring from §8 TRUE ILI Cohort analysis.
Replaces /tmp/cross_country_smoke.py + /tmp/ili_cohort_v2.py ad-hoc scripts.

DEEP MODULE (ENGINEERING_PRINCIPLES.md D-4):
- small interface: 3 main functions (dtw_distance, plv_matrix, mantel_test)
- rich implementation: Sakoe-Chiba band DTW + Hilbert+bandpass PLV + permutation Mantel

CITATIONS:
- DTW: Berndt & Clifford (1994) KDD Workshop — Sakoe-Chiba constraint
- PLV: Lachaux et al. (1999) Hum Brain Mapp 8(4):194-208
- Mantel: Mantel (1967) Cancer Res 27:209-220
- Permutation null: Künsch (1989) Ann Stat 17:1217-1241 (block bootstrap variant)
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, hilbert


def dtw_distance(x: np.ndarray, y: np.ndarray, window: int = 8) -> float:
    """Dynamic Time Warping distance with Sakoe-Chiba band.

    Args:
        x: 1D series.
        y: 1D series (same len as x for paired comparison).
        window: Sakoe-Chiba band width in time steps (default 8 weeks for biological prior:
                epi peak shift bound ±2 month).

    Returns:
        Normalized DTW distance (path-length normalized).

    Performance: O(n × window) time, O(n × m) space.
    Side effects: None (pure function).
    Caller responsibility: z-score normalize x, y per-country before passing (unit incompatibility).

    Reference: Berndt & Clifford (1994) KDD Workshop "Using dynamic time warping to find patterns".
    """
    n, m = len(x), len(y)
    INF = float("inf")
    cost = np.full((n + 1, m + 1), INF)
    cost[0, 0] = 0
    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m + 1, i + window + 1)
        for j in range(j_start, j_end):
            d = abs(x[i - 1] - y[j - 1])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[n, m] / (n + m))


def dtw_matrix(series_dict: dict[str, np.ndarray], window: int = 8) -> tuple[np.ndarray, list[str]]:
    """Pairwise DTW distance matrix for N countries.

    Args:
        series_dict: {country_code: z-scored 1D series}.
        window: Sakoe-Chiba band (default 8).

    Returns:
        (D matrix N×N symmetric with 0 diagonal, sorted country list).

    Performance: O(N² × n × window).
    """
    countries = sorted(series_dict.keys())
    N = len(countries)
    D = np.zeros((N, N))
    for i, ci in enumerate(countries):
        for j, cj in enumerate(countries):
            if i < j:
                D[i, j] = D[j, i] = dtw_distance(series_dict[ci], series_dict[cj], window=window)
    return D, countries


def plv_matrix(
    series_dict: dict[str, np.ndarray],
    fs: int = 52,
    band_low: float = 0.5,
    band_high: float = 2.0,
    filter_order: int = 4,
) -> tuple[np.ndarray, list[str]]:
    """Phase-Locking Value matrix via Hilbert transform after bandpass.

    Args:
        series_dict: {country_code: z-scored 1D series}.
        fs: sampling frequency (default 52 = weekly).
        band_low/high: bandpass cycles/year (default 0.5-2.0 = annual periodicity).
        filter_order: Butterworth order (default 4).

    Returns:
        (P matrix N×N with 1.0 diagonal, sorted country list).

    Performance: O(N × n log n) Hilbert + O(N² × n) phase difference.
    Side effects: None.

    Reference: Lachaux et al. (1999) Hum Brain Mapp — PLV definition + bandpass-Hilbert pipeline.
    """
    countries = sorted(series_dict.keys())
    N = len(countries)
    b, a = butter(filter_order, [band_low / (fs / 2), band_high / (fs / 2)], btype="band")
    phases = {c: np.angle(hilbert(filtfilt(b, a, series_dict[c]))) for c in countries}
    P = np.zeros((N, N))
    for i, ci in enumerate(countries):
        for j, cj in enumerate(countries):
            if i == j:
                P[i, j] = 1.0
            else:
                P[i, j] = float(np.abs(np.mean(np.exp(1j * (phases[ci] - phases[cj])))))
    return P, countries


def mantel_test(
    A: np.ndarray,
    B: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict:
    """Mantel test — Pearson correlation between flattened upper-triangular matrices.

    Args:
        A, B: N×N symmetric distance/similarity matrices.
        n_permutations: number of permutations (default 1000).
        seed: rng seed.

    Returns:
        dict {r: float, p_permutation: float, n_perm: int}.

    Performance: O(B × N²).
    Side effects: None.

    Reference: Mantel (1967) Cancer Res 27:209-220.
    """
    assert A.shape == B.shape, "Matrices must have same shape"
    iu = np.triu_indices_from(A, k=1)
    r_obs = float(np.corrcoef(A[iu], B[iu])[0, 1])

    rng = np.random.default_rng(seed)
    null = []
    N = A.shape[0]
    for _ in range(n_permutations):
        perm = rng.permutation(N)
        B_perm = B[np.ix_(perm, perm)]
        null.append(np.corrcoef(A[iu], B_perm[iu])[0, 1])
    p = float(np.mean(np.array(null) >= r_obs))
    return {"r": r_obs, "p_permutation": p, "n_perm": n_permutations}


def cophenetic_correlation(D: np.ndarray) -> float:
    """Cophenetic correlation of Ward linkage dendrogram with original distance.

    Args:
        D: square distance matrix.

    Returns:
        Cophenetic correlation coefficient (closer to 1 = better clustering fit).
    """
    import scipy.cluster.hierarchy as sch
    import scipy.spatial.distance as ssd
    condensed = ssd.squareform(D)
    Z = sch.linkage(condensed, method="ward")
    c, _ = sch.cophenet(Z, condensed)
    return float(c)


def zscore_interp(arr: np.ndarray) -> np.ndarray | None:
    """Z-score normalize with linear NaN interpolation.

    Args:
        arr: 1D array with possible NaNs.

    Returns:
        z-scored array, or None if insufficient valid data (std=0 or n<10).
    """
    valid = arr[np.isfinite(arr)]
    if len(valid) < 10 or valid.std() < 1e-9:
        return None
    z = (arr - valid.mean()) / valid.std()
    nans = np.isnan(z)
    if nans.any():
        z[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), z[~nans])
    return z
