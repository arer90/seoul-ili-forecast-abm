"""P4 — the headline contribution: mobility BREAKS the (α, θ) identifiability
degeneracy (PROOF_VALIDATION_PROTOCOL §1.7).

The behavioral parameters (α risk-sensitivity, θ compliance-threshold) are
*equifinal* from prevalence/ILI alone: a higher α (faster response) is
compensated by a higher θ (later activation) to yield nearly the same epidemic
curve, because prevalence is a SMOOTHED, integrated response to the compliance
trajectory — it integrates out the timing detail. The mobility / β_scale
observable is a DIRECT read of the compliance trajectory, so it retains the
detail prevalence loses.

Profile likelihood (Raue et al. 2009; Kreutz et al. 2013): fix the target
parameter on a grid, minimise the objective over the nuisance parameter, and
read identifiability off the profile's curvature — a FLAT profile is structurally
non-identifiable (the flat-bottomed valley), a CURVED profile with a finite
threshold-crossing interval is identifiable. The claim is that the ILI-only
profile is flat and the ILI+mobility profile is curved — i.e. mobility is the
data that resolves the degeneracy. Never raises in the analysis layer.
"""
from __future__ import annotations

import numpy as np

from .sim_vs_observed import simulate_response


def _nsse(sig: np.ndarray, truth: np.ndarray) -> float:
    """Normalised SSE = ||sig − truth||² / ||truth − mean(truth)||² (≈ 1 − R²;
    scale-free so prevalence and β_scale are comparable in a joint objective)."""
    sig = np.asarray(sig, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    denom = float(np.sum((truth - truth.mean()) ** 2))
    if denom <= 0:
        return float("inf")
    return float(np.sum((sig - truth) ** 2) / denom)


def objective_grid(metapop, truth_kwargs: dict, x_grid, y_grid,
                   *, param_x: str = "alpha", param_y: str = "theta",
                   fixed: dict | None = None) -> dict:
    """Run the ABM over a 2-D behavioral-parameter grid and return the prevalence-
    and response-error surfaces vs the truth.

    Generalised to ANY two behavioral parameters — the risk pair ``(alpha, theta)``
    OR the FATIGUE pair ``(kappa, tau)``. Fatigue is the dynamics the epidemic
    curve integrates out MOST (κ,τ shape the *slow* compliance erosion), so the
    mobility identifiability gain is expected to be even larger there than for
    (α,θ).

    Args:
        metapop: MetapopParams (reused for every cell).
        truth_kwargs: behavioral params that GENERATE the synthetic truth (all 4).
        x_grid, y_grid: 1-D arrays scanned on the two axes.
        param_x, param_y: which behavioral params the axes vary (e.g. "kappa",
            "tau"). The non-scanned params are held at truth via ``fixed``.
        fixed: behavioral kwargs held constant (the OTHER two params at truth).

    Returns:
        ``{param_x, param_y, x_grid, y_grid, sse_prev, sse_resp, truth_kwargs}``
        — ``sse_*`` are (|x|, |y|) normalised-SSE surfaces. Never raises (failed
        cells → inf). Performance: |x|·|y| sims (~0.2 s each on Seoul).
    """
    fixed = dict(fixed or {})
    truth = simulate_response(metapop, {**truth_kwargs})
    pa, pr = truth["prevalence"], truth["response"]
    X, Y = np.asarray(x_grid, float), np.asarray(y_grid, float)
    sse_prev = np.full((len(X), len(Y)), np.inf)
    sse_resp = np.full((len(X), len(Y)), np.inf)
    for i, xv in enumerate(X):
        for j, yv in enumerate(Y):
            try:
                r = simulate_response(
                    metapop, {param_x: float(xv), param_y: float(yv), **fixed})
                sse_prev[i, j] = _nsse(r["prevalence"], pa)
                sse_resp[i, j] = _nsse(r["response"], pr)
            except Exception:
                continue
    return {"param_x": param_x, "param_y": param_y, "x_grid": X, "y_grid": Y,
            "sse_prev": sse_prev, "sse_resp": sse_resp,
            "truth_kwargs": dict(truth_kwargs)}


def _profile(J: np.ndarray, axis_vals: np.ndarray, along_axis: int) -> np.ndarray:
    """Profile objective: minimise J over the nuisance axis, leaving a 1-D curve
    over ``along_axis``."""
    nuisance_axis = 1 - along_axis
    return np.min(J, axis=nuisance_axis)


def _ci_width(profile: np.ndarray, vals: np.ndarray, threshold: float = 0.1) -> float:
    """Fraction of the scanned axis where profile < min + threshold. ~1.0 = flat
    (non-identifiable); small = a sharp, identifiable minimum."""
    finite = profile[np.isfinite(profile)]
    if finite.size == 0:
        return 1.0
    lo = float(finite.min())
    within = np.isfinite(profile) & (profile <= lo + threshold)
    return float(within.sum()) / float(len(profile))


def identifiability_gain(grid: dict, target: str | None = None,
                         mobility_weight: float = 1.0, threshold: float = 0.1) -> dict:
    """The headline metric: does adding mobility turn a FLAT ILI-only profile into
    a CURVED (identified) one?

    Args:
        grid: output of :func:`objective_grid`.
        target: which parameter to profile — must equal the grid's ``param_x`` or
            ``param_y`` (e.g. ``"kappa"``). ``None`` → ``param_x``.
        mobility_weight: weight on the response-SSE in the joint objective.
        threshold: profile-likelihood threshold (in normalised-SSE units) for the
            identifiability interval.

    Returns:
        ``{target, ili_interval_frac, joint_interval_frac, gain,
        identified_by_mobility, verdict}`` — ``*_interval_frac`` is the fraction of
        the target axis inside the threshold (≈1 ⇒ flat/non-identifiable); ``gain``
        = ili/joint width ratio (>1 ⇒ mobility sharpened it). Never raises.
    """
    px, py = grid["param_x"], grid["param_y"]
    target = target or px
    if target == px:
        along, vals = 0, grid["x_grid"]
    elif target == py:
        along, vals = 1, grid["y_grid"]
    else:
        return {"error": f"target '{target}' not in grid params ({px}, {py})"}
    J_ili = grid["sse_prev"]
    J_joint = grid["sse_prev"] + mobility_weight * grid["sse_resp"]
    prof_ili = _profile(J_ili, vals, along)
    prof_joint = _profile(J_joint, vals, along)
    w_ili = _ci_width(prof_ili, vals, threshold)
    w_joint = _ci_width(prof_joint, vals, threshold)
    gain = (w_ili / w_joint) if w_joint > 0 else float("inf")
    # practical-identifiability (Raue 2009): the parameter is poorly constrained
    # by ILI alone (a BROAD threshold interval — the flat-bottomed valley) AND
    # adding the behavioral observable tightens it ≥2× (substantial CI reduction).
    identified = (w_ili >= 0.33) and (gain >= 2.0) and (w_joint < w_ili)
    verdict = (
        f"profile({target}): ILI-only interval={w_ili:.0%} of axis"
        + (" (BROAD ⇒ practically non-identifiable)" if w_ili >= 0.33 else
           " (already constrained by ILI)")
        + f", ILI+mobility interval={w_joint:.0%} (CI tightened {gain:.1f}×). "
        + ("⇒ MOBILITY RESOLVES THE DEGENERACY: a parameter the epidemic curve "
           "integrates out becomes identifiable once the behavioral observable is "
           "added — the data that breaks the (α,θ) equifinality."
           if identified else
           "⇒ ILI already constrains this parameter; mobility refines but is not "
           "the decisive data here (report honestly).")
    )
    return {"target": target, "ili_interval_frac": round(w_ili, 4),
            "joint_interval_frac": round(w_joint, 4), "gain": round(gain, 3),
            "identified_by_mobility": bool(identified),
            "ili_profile": [round(float(x), 4) for x in prof_ili],
            "joint_profile": [round(float(x), 4) for x in prof_joint],
            "axis": [round(float(x), 4) for x in vals], "verdict": verdict}


def _gain_from_widths(w_ili: float, w_joint: float, target: str,
                      threshold: float) -> dict:
    """Shared identifiability verdict from ILI-only and joint CI widths."""
    gain = (w_ili / w_joint) if w_joint > 0 else float("inf")
    identified = (w_ili >= 0.33) and (gain >= 2.0) and (w_joint < w_ili)
    verdict = (
        f"profile({target}): ILI-only={w_ili:.0%}"
        + (" (BROAD ⇒ practically non-identifiable)" if w_ili >= 0.33 else
           " (already constrained by ILI)")
        + f", ILI+mobility={w_joint:.0%} (tightened {gain:.1f}×). "
        + ("⇒ mobility resolves the degeneracy." if identified else
           "⇒ ILI already constrains it; mobility refines.")
    )
    return {"target": target, "ili_interval_frac": round(w_ili, 4),
            "joint_interval_frac": round(w_joint, 4), "gain": round(gain, 3),
            "identified_by_mobility": bool(identified), "verdict": verdict}


def _weekly(daily: np.ndarray, n_weeks: int) -> np.ndarray:
    """Mean-downsample a daily series to ``n_weeks`` weekly points."""
    d = np.asarray(daily, dtype=np.float64)
    return np.array([d[w * 7:(w + 1) * 7].mean()
                     for w in range(n_weeks) if (w + 1) * 7 <= len(d)])


def _affine_r2(model: np.ndarray, target: np.ndarray) -> float:
    """R² of the best affine fit ``target ≈ a + b·model`` (scale-free shape match)."""
    m, t = np.asarray(model, float), np.asarray(target, float)
    n = min(len(m), len(t))
    m, t = m[:n], t[:n]
    if n < 4 or m.std() == 0:
        return -np.inf
    b, a = np.polyfit(m, t, 1)
    sse = float(np.sum((t - (a + b * m)) ** 2))
    var = float(np.sum((t - t.mean()) ** 2))
    return 1.0 - sse / var if var > 0 else -np.inf


def calibrate_behavioral_to_ili(metapop, ili_weekly, grids: dict, *,
                                seed_infected: float = 1000.0) -> dict:
    """Coarse grid-search calibration of the behavioral params to a REAL ILI wave
    (run_coupled_abm prevalence, affine shape-matched) — a CALIBRATED, not assumed,
    operating point for the identifiability analysis.

    Returns ``{params, r2, n_eval}`` with the best-fitting (α,κ,τ,θ). The ABM is
    run for ``len(ili_weekly)*7`` days so the weekly downsample aligns. Never
    raises (failed cells skipped). Performance: ∏|grid| sims.
    """
    ili = np.asarray(ili_weekly, dtype=np.float64)
    n_weeks = len(ili)
    days = n_weeks * 7
    mp = _with_days(metapop, days, seed_infected)
    names = list(grids.keys())
    arrs = [np.asarray(grids[n], float) for n in names]
    best, n_eval = None, 0
    for idx in np.ndindex(*[len(a) for a in arrs]):
        params = {names[k]: float(arrs[k][idx[k]]) for k in range(len(names))}
        try:
            r = simulate_response(mp, params)
            r2 = _affine_r2(_weekly(r["prevalence"], n_weeks), ili)
        except Exception:
            continue
        n_eval += 1
        if best is None or r2 > best["r2"]:
            best = {"params": params, "r2": round(float(r2), 4)}
    if best is None:
        return {"error": "no successful calibration run"}
    best["n_eval"] = n_eval
    return best


def _with_days(metapop, days: int, seed_infected: float):
    """Return a copy of ``metapop`` with a new horizon/seed (deep enough for sims)."""
    import dataclasses
    import numpy as _np
    G = int(_np.asarray(metapop.populations).size)
    return dataclasses.replace(metapop, days=int(days),
                               initial_infected=_np.full(G, float(seed_infected)))


def _with_r0(metapop, r0: float, days: int, seed_infected: float):
    """Copy of ``metapop`` with a new disease R0 (forcing) + horizon/seed."""
    import dataclasses
    import numpy as _np
    G = int(_np.asarray(metapop.populations).size)
    disease = dataclasses.replace(metapop.disease, R0=float(r0))
    return dataclasses.replace(metapop, disease=disease, days=int(days),
                               initial_infected=_np.full(G, float(seed_infected)))


def calibrate_forcing_then_behavior(metapop, ili_weekly, *, r0_grid, seed_grid,
                                    behavior_grids: dict) -> dict:
    """Co-calibrate the FORCING (R0, seed — wave timing) THEN the behavior, the way
    epi_proof does, so the identifiability truth sits at a genuinely fitted point
    rather than one where only behavior was tuned over a mistimed wave.

    Step 1 fixes R0 and the seed by matching a behavior-OFF wave to the real ILI
    shape (timing); step 2 grid-searches the behavioral params at that forcing.
    Returns ``{r0, seed, forcing_r2, params, r2}``. Never raises.
    """
    ili = np.asarray(ili_weekly, dtype=np.float64)
    n_weeks = len(ili)
    days = n_weeks * 7
    off = {"alpha": 0.0, "kappa": 0.0, "tau": 1.0e9, "theta": 0.0}
    best_f = None
    for r0 in r0_grid:
        for seed in seed_grid:
            try:
                r = simulate_response(_with_r0(metapop, r0, days, seed), off)
                r2 = _affine_r2(_weekly(r["prevalence"], n_weeks), ili)
            except Exception:
                continue
            if best_f is None or r2 > best_f["r2"]:
                best_f = {"r0": float(r0), "seed": float(seed), "r2": round(float(r2), 4)}
    if best_f is None:
        return {"error": "no successful forcing run"}
    mp_best = _with_r0(metapop, best_f["r0"], days, best_f["seed"])
    bcal = calibrate_behavioral_to_ili(mp_best, ili, behavior_grids,
                                       seed_infected=best_f["seed"])
    if "error" in bcal:
        return {"error": bcal["error"], "forcing": best_f}
    return {"r0": best_f["r0"], "seed": best_f["seed"], "forcing_r2": best_f["r2"],
            "params": bcal["params"], "r2": bcal["r2"]}


def objective_grid_nd(metapop, truth_kwargs: dict, grids: dict) -> dict:
    """Full N-D objective grid over ANY set of behavioral params (the proper
    profile-likelihood basis — every nuisance param is itself gridded, not fixed).

    Args:
        grids: ``{param_name: 1-D array}`` for every param to vary (e.g. all four
            α,κ,τ,θ). truth_kwargs generates the synthetic truth.

    Returns ``{names, grids, sse_prev, sse_resp, truth_kwargs}`` with N-D SSE
    tensors. Never raises. Performance: ∏|grid| sims (4 params × 5 = 625).
    """
    names = list(grids.keys())
    arrs = [np.asarray(grids[n], float) for n in names]
    shape = tuple(len(a) for a in arrs)
    truth = simulate_response(metapop, {**truth_kwargs})
    pa, pr = truth["prevalence"], truth["response"]
    sse_prev = np.full(shape, np.inf)
    sse_resp = np.full(shape, np.inf)
    for idx in np.ndindex(*shape):
        params = {names[k]: float(arrs[k][idx[k]]) for k in range(len(names))}
        try:
            r = simulate_response(metapop, params)
            sse_prev[idx] = _nsse(r["prevalence"], pa)
            sse_resp[idx] = _nsse(r["response"], pr)
        except Exception:
            continue
    return {"names": names, "grids": arrs, "sse_prev": sse_prev,
            "sse_resp": sse_resp, "truth_kwargs": dict(truth_kwargs)}


def profile_nd(grid_nd: dict, target: str, mobility_weight: float = 1.0,
               threshold: float = 0.1) -> dict:
    """Proper profile likelihood for ``target``: minimise the objective over ALL
    other params (the full N-D nuisance space), then compare ILI-only vs
    ILI+mobility CI widths. Returns the same shape as
    :func:`identifiability_gain`. Never raises."""
    names = grid_nd["names"]
    if target not in names:
        return {"error": f"target '{target}' not in {names}"}
    ax = names.index(target)
    other = tuple(i for i in range(len(names)) if i != ax)
    J_ili = grid_nd["sse_prev"]
    J_joint = grid_nd["sse_prev"] + mobility_weight * grid_nd["sse_resp"]
    prof_ili = np.min(J_ili, axis=other) if other else J_ili
    prof_joint = np.min(J_joint, axis=other) if other else J_joint
    vals = grid_nd["grids"][ax]
    w_ili = _ci_width(prof_ili, vals, threshold)
    w_joint = _ci_width(prof_joint, vals, threshold)
    out = _gain_from_widths(w_ili, w_joint, target, threshold)
    out["ili_profile"] = [round(float(x), 4) for x in prof_ili]
    out["joint_profile"] = [round(float(x), 4) for x in prof_joint]
    return out
