"""Dynamical-signature proof (PROOF_VALIDATION_PROTOCOL §1.6).

The observational mobility decomposition is confounded (fatigue ∥ vaccination ∥
policy — see ``behavioral_proof.confounding_check``), so a *coefficient* is not a
proof. A **topological** signature is harder to fake: a memoryless confounder
(weather, a static dose-response, a monotone vaccination trend) maps each
prevalence to ONE contact level — a single-valued curve with ~zero loop area.
A system with **memory/fatigue** (the ABM's F state) is path-dependent: at the
same prevalence, contact differs on the rising vs falling branch — tracing a
**hysteresis loop** with non-zero signed area and a consistent circulation that
random re-orderings cannot reproduce.

These functions are the SAME machinery applied to BOTH the observed mobility and
the simulated trajectory (``behavioral_proof`` panel and the ABM output). The
proof is not "the observed data loops" (it does, but that is confounded) — it is
"the model, given NO mobility and NO vaccination input, reproduces the observed
loop" (see PROOF_VALIDATION_PROTOCOL §1.8 / sim_vs_observed). Never raise.
"""
from __future__ import annotations

import numpy as np


def _zunit(a: np.ndarray) -> np.ndarray:
    """Z-score normalize; constant series → all-zeros (degenerate, no loop)."""
    a = np.asarray(a, dtype=np.float64)
    s = float(a.std())
    return (a - a.mean()) / s if s > 0 else np.zeros_like(a)


def _branch_area(x: np.ndarray, y: np.ndarray, n_grid: int = 50) -> float:
    """Rising-vs-falling BRANCH integral over the driver's dominant cycle (C2-fix).

    Split the time-ordered trajectory at the driver peak; interpolate the response
    on the driver across the branches' common range; integrate (rising − falling).
    A single-valued memoryless curve has identical branches → ~0; a hysteresis loop
    has separated branches → non-zero. Unlike a shoelace closure (``np.roll`` last
    vertex → first) this NEVER invents area for an open, non-returning path (an
    open ramp ``y=x²`` scored 0.166 under closure; here it is ~0).
    """
    n = len(x)
    if n < 6:
        return 0.0
    # Split at whichever turning point is INTERIOR. A trough-bounded season runs
    # trough → peak → trough, so the peak is interior and the branches are
    # rising-then-falling. A span that begins at a peak (a driver sampled from its
    # maximum, e.g. cos over one period) has argmax at index 0; splitting there
    # returned 0.0 and reported a textbook unit circle — enclosed area pi — as no
    # loop at all. Split such a span at its trough instead and swap the branch
    # roles so the returned sign still means the same circulation.
    pk, tr = int(np.argmax(x)), int(np.argmin(x))
    if 2 <= pk <= n - 3:
        first, second, flip = pk, pk, 1.0      # rising | falling
    elif 2 <= tr <= n - 3:
        first, second, flip = tr, tr, -1.0     # falling | rising → negate
    else:
        return 0.0                             # monotone span: no loop to resolve
    xr, yr = x[:first + 1], y[:first + 1]
    xf, yf = x[second:], y[second:]
    lo, hi = max(xr.min(), xf.min()), min(xr.max(), xf.max())
    if hi <= lo:
        return 0.0
    g = np.linspace(lo, hi, n_grid)
    sr, sf = np.argsort(xr), np.argsort(xf)
    return flip * float(
        np.trapezoid(np.interp(g, xr[sr], yr[sr]) - np.interp(g, xf[sf], yf[sf]), g)
    )


def _cycle_bounds(x: np.ndarray) -> list:
    """Trough-bounded cycles of the driver (multi-season support).

    Returns ``[(start, end), …]`` index pairs (one per epidemic cycle); a single
    pair if no interior trough is resolved. Multi-cycle real data (7 flu seasons)
    is segmented so each season's loop is measured and the signed areas summed —
    a consistent circulation across seasons reinforces, an inconsistent one cancels.
    """
    n = len(x)
    troughs = np.array([], dtype=int)
    try:
        from scipy.signal import find_peaks
    except ImportError as e:                     # pragma: no cover - scipy is declared
        # Without find_peaks no trough is resolved, so a multi-season record
        # collapses into ONE segment and the summed per-cycle area comes out
        # 2-3x too small (measured: 3.5 per cycle with scipy, 1.75-2.38 without)
        # while still reporting significance. That is a wrong number, not a
        # missing one, so say so rather than degrading in silence.
        raise ImportError(
            "hysteresis_loop_area needs scipy.signal.find_peaks to segment cycles; "
            "without it multi-cycle areas are understated. scipy is a declared "
            "dependency (pyproject: scipy>=1.11) — install it."
        ) from e
    try:
        amp = float(np.nanmax(x) - np.nanmin(x))
        if amp > 0:
            troughs, _ = find_peaks(-x, prominence=0.15 * amp, distance=max(4, n // 12))
    except Exception:
        pass
    # Only troughs that actually delimit COMPLETE cycles are used as boundaries.
    # With 2+ interior troughs the spans between consecutive troughs are whole
    # seasons. With 0 or 1 trough, cutting at it would hand _branch_area two
    # monotone half-spans, each with its turning point on an edge — which is how a
    # single full cycle got reported as "2 cycle(s)" and scored 0.
    if len(troughs) >= 2:
        bounds = troughs.tolist()
        segs = [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= 6]
        if segs:
            return segs
    return [(0, n - 1)]


def _no_loop(n: int, reason: str) -> dict:
    """Clean negative-control return (NOT an error): a constant/near-constant
    response (behavior OFF → no modulation) is a valid 'no hysteresis' result."""
    return {"loop_area": 0.0, "abs_area": 0.0, "null_p": 1.0, "n": int(n),
            "circulation": "none", "significant": False,
            "verdict": f"no loop — {reason}", "method": "branch_perm", "n_cycles": 0}


def hysteresis_loop_area(driver, response, n_null: int = 2000, seed: int = 42,
                         detrend: bool = False) -> dict:
    """Path-dependence (memory/fatigue) signature of a (driver, response) pair.

    ★ CORRECTED metric (Gemini review C1/C2, verified). Two fixes vs the prior
    shoelace + phase-randomization version:
      * **C2 (area):** rising-vs-falling BRANCH integral (:func:`_branch_area`),
        not a closed-polygon shoelace — a memoryless single-valued curve gives ~0
        (the closure made an open ramp ``y=x²`` score a spurious 0.166).
      * **C1 (null):** branch-label PERMUTATION null (shuffle the response's time
        order → decouple it from the driver's rising/falling phase), not a
        spectrum-preserving phase-randomization. The old null MISSED a genuine
        smooth lagged sinusoid (p≈0.46) and fired on mere waveform sharpness; this
        null detects the lagged sinusoid (p≈0.02), gives ``y=x²`` p≈1, and still
        finds the ABM behavioral loop highly significant (validated probe).
    Multi-cycle: signed per-cycle branch-areas are SUMMED across seasons.

    Args:
        driver: perceived-prevalence proxy over time (ILI/COVID cases), time-ordered.
        response: behavioral response (mobility deviation), aligned, same length.
        n_null: permutation-null replicates. seed: RNG seed (reproducibility, G-#5).
        detrend: subtract a linear trend from each series first (open-ramp guard).

    Returns:
        ``{loop_area, abs_area, null_p, n, circulation, significant, verdict,
        method, n_cycles}`` — ``loop_area`` = summed signed branch area (sign =
        circulation direction), ``null_p`` = P(|null| ≥ |observed|) under response
        time-shuffles (small ⇒ genuine rising≠falling path-dependence a memoryless
        confounder cannot fake). Never raises.

    Limitations (measured 2026-07-19, pinned by
    ``tests/test_hysteresis_positive_control.py``):
      * **A single truncated cycle under-reports.** The branch integral is taken
        over the driver range the two branches SHARE, so a window covering one
        period but starting mid-slope shares only a sliver of it — the same unit
        circle scores 6.24 starting at an extremum and 0.06 starting a sixth of a
        period in, where it is called not significant. Detection is reliable once
        two or more complete cycles are present, which is the multi-season case
        this is used for; do not read a single-season area as an absolute.
      * **Area scales with resolved cycles, not with loop size.** Signed per-cycle
        areas are summed, so compare ``abs_area / n_cycles`` across records of
        different length, never ``abs_area`` directly.

    Performance: O(n_null · n_cycles · n_grid). Side effects: none.
    Caller responsibility: ``driver``/``response`` aligned, length ≥ 6.
    """
    d = np.asarray(driver, dtype=np.float64)
    r = np.asarray(response, dtype=np.float64)
    ok = np.isfinite(d) & np.isfinite(r)
    d, r = d[ok], r[ok]
    if len(d) < 6:
        return {"error": f"only {len(d)} finite aligned points (need ≥6)"}
    # near-constant response = no behavioral modulation = clean negative control
    # (relative threshold: a real modulation is ~20% of the mean; numerical noise
    # is ~1e-10 — 1e-3 cleanly separates and stops z-score amplifying noise into
    # a spurious loop, which is how behavior-OFF leaked a false p≈0 earlier).
    if float(np.std(r)) < 1e-3 * (abs(float(np.mean(r))) + 1e-12):
        return _no_loop(len(d), "response ~constant (no behavioral modulation)")
    if float(np.std(d)) < 1e-3 * (abs(float(np.mean(d))) + 1e-12):
        return _no_loop(len(d), "driver ~constant")
    if detrend and len(d) >= 4:
        t = np.arange(len(d), dtype=np.float64)
        d = d - np.polyval(np.polyfit(t, d, 1), t)
        r = r - np.polyval(np.polyfit(t, r, 1), t)
    x, y = _zunit(d), _zunit(r)
    if x.std() == 0 or y.std() == 0:
        return _no_loop(len(d), "no variation after normalization")
    segs = _cycle_bounds(x)

    def _total(yy: np.ndarray) -> float:
        return float(sum(_branch_area(x[a:b + 1], yy[a:b + 1]) for a, b in segs))

    obs = _total(y)
    rng = np.random.default_rng(seed)
    null = np.array([_total(rng.permutation(y)) for _ in range(n_null)], dtype=np.float64)
    p = float((np.abs(null) >= abs(obs)).mean())
    circ = "counterclockwise (falling-branch above)" if obs > 0 else "clockwise"
    sig = p < 0.05
    verdict = (
        f"hysteresis {'PRESENT' if sig else 'not significant'} "
        f"(branch area={obs:+.3f}, p={p:.3f}, {circ}, {len(segs)} cycle(s)) — "
        + ("path-dependent rising≠falling response a memoryless confounder cannot "
           "fake; the proof is whether the ABM reproduces it (sim_vs_observed)"
           if sig else "single-valued / too few cycles to resolve a loop")
    )
    return {"loop_area": round(obs, 4), "abs_area": round(abs(obs), 4),
            "null_p": round(p, 4), "n": len(d), "circulation": circ,
            "significant": bool(sig), "verdict": verdict,
            "method": "branch_perm", "n_cycles": len(segs)}


def spectral_peak(series, samples_per_year: float = 12.0) -> dict:
    """Dominant oscillation period via periodogram — flags NON-annual cycles.

    A purely seasonal driver oscillates at exactly 1/year. A behavioral feedback
    loop (prevalence→fear→contact→prevalence, predator-prey-like) introduces an
    oscillation at a DIFFERENT period — a signature seasonality alone cannot
    produce. Returns ``{period_years, is_annual, verdict}``. Never raises.

    Performance: O(n log n). Side effects: none.
    """
    a = np.asarray(series, dtype=np.float64)
    a = a[np.isfinite(a)]
    if len(a) < 8:
        return {"error": f"only {len(a)} finite points (need ≥8)"}
    a = a - a.mean()
    if a.std() == 0:
        return {"error": "constant series"}
    # real FFT periodogram; ignore the DC bin
    power = np.abs(np.fft.rfft(a)) ** 2
    freqs = np.fft.rfftfreq(len(a), d=1.0 / samples_per_year)  # cycles/year
    power[0] = 0.0
    k = int(np.argmax(power))
    f = float(freqs[k])
    period = float(1.0 / f) if f > 0 else float("inf")
    is_annual = abs(period - 1.0) < 0.25  # within 3 months of annual
    verdict = (
        f"dominant period ≈ {period:.2f} yr — "
        + ("annual (seasonal driver; behavioral cycle not separable at this length)"
           if is_annual else
           "NON-annual oscillation, consistent with a behavioral feedback cycle")
    )
    return {"period_years": round(period, 3), "is_annual": bool(is_annual),
            "dominant_freq_per_year": round(f, 3), "verdict": verdict}


def lag_cross_correlation(driver, response, max_lag: int = 6) -> dict:
    """Lag (in samples) at which response best correlates with driver.

    A reactive behavioral response LAGS the prevalence signal (perception +
    decision delay). A zero-lag or negative-lag peak argues against a causal
    behavioral reaction. Returns ``{best_lag, best_corr, verdict}``. Never raises.
    """
    d = np.asarray(driver, dtype=np.float64)
    r = np.asarray(response, dtype=np.float64)
    ok = np.isfinite(d) & np.isfinite(r)
    d, r = d[ok], r[ok]
    if len(d) < 8:
        return {"error": f"only {len(d)} aligned points (need ≥8)"}
    d = (d - d.mean()) / (d.std() or 1.0)
    r = (r - r.mean()) / (r.std() or 1.0)
    best_lag, best_abs, best_corr = 0, -1.0, 0.0
    for lag in range(0, max_lag + 1):
        if lag >= len(d):
            break
        # response at t vs driver at t-lag
        a = d[: len(d) - lag] if lag else d
        b = r[lag:] if lag else r
        if len(a) < 4:
            break
        c = float(np.corrcoef(a, b)[0, 1])
        if abs(c) > best_abs:
            best_lag, best_abs, best_corr = lag, abs(c), c
    verdict = (f"response best matches driver at lag {best_lag} "
               f"(r={best_corr:+.3f}) — "
               + ("delayed reaction, consistent with perception+decision lag"
                  if best_lag >= 1 else "contemporaneous / no resolvable delay"))
    return {"best_lag": best_lag, "best_corr": round(best_corr, 4),
            "verdict": verdict}
