"""
simulation.scripts.abm_identifiability_4param
=============================================
Practical identifiability analysis of the FOUR behavioural ABM parameters
(α risk-perception, κ fatigue-weight, τ fatigue-time-constant, θ
compliance-threshold) against REAL Seoul national ILI.

WHY THIS SCRIPT EXISTS
----------------------
The thesis claims the behavioural-layer parameters are "plausible rather than
sharply identified" but only *asserts* it; the S3.3 SBC diagnostic validates a
**2-parameter toy** Gaussian simulator (``simulation/results/abm_sbc/result.json``
= ``"toy only (ABM skipped)"``). The cited reference [26] (Verelst et al.) flags
exactly this weak identifiability of behaviour-change models. This script
quantifies it on the actual 4-parameter ABM with two complementary,
textbook-standard diagnostics:

1. PROFILE LIKELIHOOD (Raue et al. 2009 *Bioinformatics*; Kreutz et al. 2013):
   fix one parameter on a grid, minimise the fit loss over the other three
   (nuisance) parameters, and read identifiability off the profile's curvature.
   A FLAT profile (a wide threshold-crossing interval) ⇒ practically
   non-identifiable; a SHARP minimum (a narrow interval) ⇒ identifiable.

2. POSTERIOR PAIRWISE CORRELATION (sloppiness; Gutenkunst et al. 2007 *PLoS
   Comp Biol*; Brown & Sethna 2003): an ABC-SMC posterior over the 4 parameters
   given the same observed; strong pairwise correlation ⇒ a degenerate
   (sloppy) direction = two parameters trade off and cannot be separated.

OBSERVATION MODEL
-----------------
Observed = real KDCA national sentinel ILI (``sentinel_influenza``) for one
non-COVID flu season (default 2023-24), reduced to a SCALE-INVARIANT SHAPE
summary (peak position, normalised rise/fall, peak/mean) — identical to the
existing ``scripts/sbi_posterior_calibration.py`` ``_summary``. The ABM
city-wide prevalence I(t) is reduced with the same summary, so only epidemic
SHAPE (not absolute scale, which the affine map absorbs) drives the fit. This
is the honest information content available from single-city ILI alone — the
whole point of the [26] critique.

HONESTY
-------
The verdict reports per-parameter identifiable / sloppy exactly as the data
show it. Single-city ILI is expected to leave at least the fatigue parameters
(κ, τ) sloppy (the epidemic curve integrates out the slow compliance-erosion
dynamics — the P4 ``identifiability.py`` story). That is NOT a failure to hide;
it is the quantification of the [26] critique and a contribution in itself.

READ-ONLY GUARANTEE
-------------------
- DB: read-only (``load_weekly_ili`` → ``read_only_connect``).
- No model RETRAINING (the forecasting champion is untouched).
- No SQLite WRITE. Only writes plain CSV/JSON/PNG under
  ``simulation/results/abm_identifiability/`` and a figure under
  ``simulation/results/figures/``.

Run:
    .venv/bin/python -m simulation.scripts.abm_identifiability_4param
    .venv/bin/python -m simulation.scripts.abm_identifiability_4param --season 2023 \
        --grid-points 13 --nuisance-iters 200 --abc-particles 300
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger("abm_identifiability_4param")

# ---------------------------------------------------------------------------
# Parameter space (matches scripts/sbi_posterior_calibration.PRIORS exactly so
# the two analyses live in the same box — no silent re-scoping).
# ---------------------------------------------------------------------------
PARAM_NAMES = ("alpha", "kappa", "tau", "theta")
PRIORS: dict[str, tuple[float, float]] = {
    "alpha": (0.5, 3.5),
    "kappa": (0.05, 0.40),
    "tau": (40.0, 140.0),
    "theta": (0.03, 0.30),
}
PARAM_LABEL = {
    "alpha": "α (risk perception)",
    "kappa": "κ (fatigue weight)",
    "tau": "τ (fatigue time-constant, d)",
    "theta": "θ (compliance threshold)",
}


# ---------------------------------------------------------------------------
# Observation model — scale-invariant epidemic SHAPE summary.
# Identical to scripts/sbi_posterior_calibration._summary (single SSOT of the
# observation model; copied here only to keep this script self-contained and
# avoid importing a __main__-side helper).
# ---------------------------------------------------------------------------
def _summary(traj) -> np.ndarray:
    """Scale-invariant shape statistic (peak position, normalised rise/fall,
    peak/mean). Returns 4-vector; NaN-vector when the trajectory is degenerate."""
    t = np.asarray(traj, dtype=np.float64)
    t = t[np.isfinite(t)]
    if len(t) < 6 or t.max() <= 0:
        return np.array([np.nan] * 4)
    pk, n = int(np.argmax(t)), len(t)
    rise = (t[pk] - t[0]) / (t[pk] + 1e-9) / max(pk, 1)
    fall = (t[pk] - t[-1]) / (t[pk] + 1e-9) / max(n - pk, 1)
    return np.array([pk / n, rise * 52, fall * 52, t[pk] / (t.mean() + 1e-9)])


def _shape_loss(x_sim: np.ndarray, x_obs: np.ndarray) -> float:
    """Dimension-normalised Euclidean distance between shape summaries.

    Returns +inf for non-finite simulations (blow-up guard) so a diverged run
    never wins the profile minimisation. This is the SAME distance ABC-SMC uses,
    so the two diagnostics share one objective surface."""
    if not np.all(np.isfinite(x_sim)) or not np.all(np.isfinite(x_obs)):
        return float("inf")
    diff = x_sim - x_obs
    return float(np.sqrt(np.dot(diff, diff) / x_obs.size))


# ---------------------------------------------------------------------------
# Profile likelihood
# ---------------------------------------------------------------------------
@dataclass
class ProfilePoint:
    param: str
    value: float
    profile_loss: float           # min loss over the 3 nuisance params
    best_nuisance: dict           # the argmin nuisance params at this grid value


def _random_nuisance(rng: np.random.Generator, fixed_param: str,
                     fixed_value: float, n: int) -> list[dict]:
    """Latin-ish random sample of the OTHER three params (uniform in prior box).

    The fixed param is pinned at ``fixed_value``; the three nuisance params are
    sampled uniformly from their priors. n samples per grid value."""
    others = [p for p in PARAM_NAMES if p != fixed_param]
    out = []
    for _ in range(n):
        kw = {fixed_param: float(fixed_value)}
        for p in others:
            lo, hi = PRIORS[p]
            kw[p] = float(rng.uniform(lo, hi))
        out.append(kw)
    return out


def profile_likelihood(simulator, x_obs, *, grid_points: int,
                       nuisance_iters: int, seed: int) -> list[ProfilePoint]:
    """Profile-likelihood curve for each of the 4 params.

    For each param p and each grid value v in its prior range, fix p=v and
    minimise the shape loss over the other 3 params by random search
    (``nuisance_iters`` samples). The resulting (v, min_loss) curve is the
    profile. A flat curve ⇒ non-identifiable; a sharp valley ⇒ identifiable.

    Random search (not gradient descent) is used because the compliance
    Heaviside makes the loss piecewise-flat / non-smooth, so a local optimiser
    would stall; a few-hundred-sample random search over a 3-D box is robust and
    fully reproducible (seeded).

    Returns a flat list of ProfilePoint (param-major, then grid-major)."""
    rng = np.random.default_rng(seed)
    points: list[ProfilePoint] = []
    for p in PARAM_NAMES:
        lo, hi = PRIORS[p]
        grid = np.linspace(lo, hi, grid_points)
        for v in grid:
            best_loss = float("inf")
            best_kw: dict = {}
            for kw in _random_nuisance(rng, p, v, nuisance_iters):
                x_sim = simulator(kw)
                loss = _shape_loss(x_sim, x_obs)
                if loss < best_loss:
                    best_loss, best_kw = loss, dict(kw)
            points.append(ProfilePoint(param=p, value=float(v),
                                       profile_loss=best_loss,
                                       best_nuisance=best_kw))
        log.info("  profile[%s] done (%d grid pts)", p, grid_points)
    return points


def _profile_interval_frac(values: np.ndarray, losses: np.ndarray,
                           threshold_abs: float) -> float:
    """Fraction of the scanned axis with profile_loss <= min + threshold_abs.

    ~1.0 = flat profile (practically non-identifiable); small = a sharp,
    identifiable minimum. Non-finite losses are excluded from the support but
    counted in the denominator (a diverged region is not 'inside' the CI)."""
    finite = np.isfinite(losses)
    if not finite.any():
        return 1.0
    lo = float(losses[finite].min())
    within = finite & (losses <= lo + threshold_abs)
    return float(within.sum()) / float(len(losses))


# ---------------------------------------------------------------------------
# Posterior pairwise correlation (ABC-SMC)
# ---------------------------------------------------------------------------
def posterior_correlation(simulator_rng, x_obs, *, n_particles: int,
                          tolerance_schedule, seed: int) -> dict:
    """ABC-SMC posterior over the 4 params + pairwise correlation matrix.

    Uses the project's leak-free ``abc_smc`` (Toni et al. 2009). Strong pairwise
    correlation ⇒ a sloppy (degenerate) direction = the two params trade off and
    cannot be separated from the data — the sloppiness diagnostic
    (Gutenkunst et al. 2007).

    Returns ``{particles, weights, param_names, posterior_mean, posterior_std,
    ci95, ci_width_vs_prior, corr (4x4), accept_counts, posterior_std_per_round}``.
    """
    from simulation.abm.abc_smc import abc_smc

    res = abc_smc(
        simulator_rng,
        x_obs,
        PRIORS,
        n_particles=n_particles,
        tolerance_schedule=tolerance_schedule,
        seed=seed,
        max_tries_per_particle=4000,  # shape-distance floor ~0.86 (model can't
        # match single-city ILI shape exactly) → low tolerances need headroom.
    )
    particles = np.asarray(res["particles"], dtype=np.float64)
    weights = np.asarray(res["weights"], dtype=np.float64)
    names = list(res["param_names"])

    # Weighted Pearson correlation across the posterior particles.
    mean = np.average(particles, axis=0, weights=weights)
    centered = particles - mean
    cov = (centered * weights[:, None]).T @ centered  # weighted cov (sum w=1)
    std = np.sqrt(np.diag(cov))
    denom = np.outer(std, std)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(denom > 0, cov / denom, 0.0)
    np.fill_diagonal(corr, 1.0)

    # Marginal CI widths vs prior (ABC-SMC weighted percentiles).
    ci95, widths = [], []
    for i, p in enumerate(names):
        s = particles[:, i]
        order = np.argsort(s)
        cw = np.cumsum(weights[order])
        lo_v = float(np.interp(0.025, cw, s[order]))
        hi_v = float(np.interp(0.975, cw, s[order]))
        pr_lo, pr_hi = PRIORS[p]
        ci95.append([round(lo_v, 4), round(hi_v, 4)])
        widths.append(round((hi_v - lo_v) / (pr_hi - pr_lo), 3))

    return {
        "particles": particles,
        "weights": weights,
        "param_names": names,
        "posterior_mean": [round(float(m), 4) for m in res["posterior_mean"]],
        "posterior_std": [round(float(s), 4) for s in res["posterior_std"]],
        "ci95": ci95,
        "ci_width_vs_prior": widths,
        "corr": corr,
        "accept_counts": res["accept_counts"],
        "tolerance_schedule": res["tolerance_schedule"],
        "posterior_std_per_round": res["posterior_std_per_round"],
    }


# ---------------------------------------------------------------------------
# Verdict synthesis
# ---------------------------------------------------------------------------
def synthesize_verdict(profile_pts: list[ProfilePoint], post: dict,
                       *, profile_threshold: float,
                       width_identifiable: float,
                       corr_strong: float) -> dict:
    """Combine the two diagnostics into a per-parameter honest verdict.

    A parameter is called IDENTIFIABLE only if BOTH agree:
      - profile interval-fraction < ``width_identifiable`` (sharp valley), AND
      - posterior CI width < ``width_identifiable`` × prior.
    Otherwise SLOPPY / non-identifiable. Strong pairwise correlations
    (|r| >= ``corr_strong``) are reported as the degenerate directions."""
    # profile interval fraction per param
    prof_frac: dict[str, float] = {}
    prof_curve: dict[str, dict] = {}
    for p in PARAM_NAMES:
        pts = [pp for pp in profile_pts if pp.param == p]
        vals = np.array([pp.value for pp in pts])
        losses = np.array([pp.profile_loss for pp in pts])
        frac = _profile_interval_frac(vals, losses, profile_threshold)
        prof_frac[p] = round(frac, 4)
        prof_curve[p] = {"values": [round(float(v), 4) for v in vals],
                         "profile_loss": [round(float(x), 4) for x in losses]}

    names = post["param_names"]
    width_by = dict(zip(names, post["ci_width_vs_prior"]))
    corr = np.asarray(post["corr"])

    per_param: dict[str, dict] = {}
    for p in PARAM_NAMES:
        pf = prof_frac[p]
        w = float(width_by[p])
        prof_id = pf < width_identifiable
        post_id = w < width_identifiable
        if prof_id and post_id:
            status = "identifiable"
        elif (not prof_id) and (not post_id):
            status = "sloppy"  # both flag it
        else:
            status = "weakly_identifiable"  # diagnostics disagree → honest middle
        per_param[p] = {
            "label": PARAM_LABEL[p],
            "profile_interval_frac": pf,
            "profile_identifiable": bool(prof_id),
            "posterior_ci_width_vs_prior": w,
            "posterior_identifiable": bool(post_id),
            "status": status,
        }

    # strong correlation pairs (degenerate / sloppy directions)
    strong_pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = float(corr[i, j])
            if abs(r) >= corr_strong:
                strong_pairs.append({
                    "pair": [names[i], names[j]], "r": round(r, 3),
                    "note": ("positive trade-off (degenerate ridge)" if r > 0
                             else "negative trade-off (compensating)"),
                })

    ident = [p for p in PARAM_NAMES if per_param[p]["status"] == "identifiable"]
    sloppy = [p for p in PARAM_NAMES if per_param[p]["status"] == "sloppy"]
    weak = [p for p in PARAM_NAMES if per_param[p]["status"] == "weakly_identifiable"]

    paragraph = (
        f"Practical-identifiability analysis of the four behavioural parameters "
        f"against real single-city Seoul ILI shape. "
        f"IDENTIFIABLE (sharp profile AND tight posterior): "
        f"{', '.join(ident) if ident else 'none'}. "
        f"SLOPPY / non-identifiable (flat profile AND broad posterior): "
        f"{', '.join(sloppy) if sloppy else 'none'}. "
        f"WEAKLY identifiable (diagnostics split): "
        f"{', '.join(weak) if weak else 'none'}. "
        + (f"Degenerate posterior directions (|r|>={corr_strong}): "
           + "; ".join(f"{q['pair'][0]}~{q['pair'][1]} (r={q['r']})"
                       for q in strong_pairs) + ". "
           if strong_pairs else "No strong pairwise posterior correlation. ")
        + "This empirically quantifies the [26] (Verelst et al.) critique that "
          "behaviour-change parameters are only weakly identified from epidemic "
          "curves alone, and complements the toy-only S3.3 SBC with the real "
          "4-parameter ABM."
    )

    return {
        "per_param": per_param,
        "identifiable": ident,
        "sloppy": sloppy,
        "weakly_identifiable": weak,
        "strong_correlation_pairs": strong_pairs,
        "thresholds": {
            "profile_loss_threshold_abs": profile_threshold,
            "interval_frac_identifiable": width_identifiable,
            "posterior_ci_width_identifiable": width_identifiable,
            "corr_strong": corr_strong,
        },
        "verdict_paragraph": paragraph,
        "_profile_curve": prof_curve,
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def make_figure(profile_pts: list[ProfilePoint], post: dict, verdict: dict,
                out_png: Path) -> None:
    """4-panel profile-likelihood + a posterior correlation heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 8), dpi=140)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.15], hspace=0.38, wspace=0.32)

    # --- profile-likelihood 4-panel (2x2 over the first two columns) ---
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    for ax, p in zip(axes, PARAM_NAMES):
        pts = [pp for pp in profile_pts if pp.param == p]
        vals = np.array([pp.value for pp in pts])
        losses = np.array([pp.profile_loss for pp in pts])
        fin = np.isfinite(losses)
        ax.plot(vals[fin], losses[fin], "o-", color="#1f77b4", markersize=4, lw=1.5)
        if fin.any():
            lo = losses[fin].min()
            thr = lo + verdict["thresholds"]["profile_loss_threshold_abs"]
            ax.axhline(thr, color="#d62728", ls="--", lw=1.0,
                       label=f"min+{verdict['thresholds']['profile_loss_threshold_abs']:.2f}")
            ax.axhline(lo, color="#2ca02c", ls=":", lw=0.8)
        st = verdict["per_param"][p]["status"]
        frac = verdict["per_param"][p]["profile_interval_frac"]
        col = {"identifiable": "#2ca02c", "sloppy": "#d62728",
               "weakly_identifiable": "#ff7f0e"}[st]
        ax.set_title(f"{PARAM_LABEL[p]}\nprofile interval={frac:.0%}  [{st}]",
                     fontsize=10, color=col)
        ax.set_xlabel(p, fontsize=9)
        ax.set_ylabel("profile loss (min over 3 nuisance)", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.25)

    # --- posterior correlation heatmap (third column, spanning both rows) ---
    axc = fig.add_subplot(gs[:, 2])
    corr = np.asarray(post["corr"])
    names = post["param_names"]
    im = axc.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")
    axc.set_xticks(range(len(names)), names, fontsize=9, rotation=30)
    axc.set_yticks(range(len(names)), names, fontsize=9)
    for i in range(len(names)):
        for j in range(len(names)):
            axc.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                     fontsize=10,
                     color="white" if abs(corr[i, j]) > 0.55 else "black")
    axc.set_title("ABC-SMC posterior pairwise correlation\n"
                  "(|r|→1 = sloppy/degenerate direction)", fontsize=10)
    cbar = fig.colorbar(im, ax=axc, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r", fontsize=9)

    fig.suptitle(
        "ABM behavioural 4-parameter practical identifiability "
        "(real Seoul ILI shape)\n"
        "profile likelihood (Raue 2009) + posterior sloppiness "
        "(Gutenkunst 2007)",
        fontsize=12, y=0.99)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("wrote %s", out_png)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def write_profile_csv(profile_pts: list[ProfilePoint], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["param", "value", "profile_loss",
                    "best_alpha", "best_kappa", "best_tau", "best_theta"])
        for pp in profile_pts:
            bn = pp.best_nuisance
            w.writerow([pp.param, f"{pp.value:.6f}",
                        ("inf" if not np.isfinite(pp.profile_loss)
                         else f"{pp.profile_loss:.6f}"),
                        bn.get("alpha", ""), bn.get("kappa", ""),
                        bn.get("tau", ""), bn.get("theta", "")])


def write_corr_csv(post: dict, path: Path) -> None:
    names = post["param_names"]
    corr = np.asarray(post["corr"])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + names)
        for i, p in enumerate(names):
            w.writerow([p] + [f"{corr[i, j]:.6f}" for j in range(len(names))])
        w.writerow([])
        w.writerow(["param", "posterior_mean", "posterior_std",
                    "ci95_lo", "ci95_hi", "ci_width_vs_prior"])
        for i, p in enumerate(names):
            w.writerow([p, post["posterior_mean"][i], post["posterior_std"][i],
                        post["ci95"][i][0], post["ci95"][i][1],
                        post["ci_width_vs_prior"][i]])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, default=2023,
                    help="flu season start year for the observed ILI (default 2023-24)")
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--grid-points", type=int, default=13,
                    help="profile-likelihood grid points per parameter")
    ap.add_argument("--nuisance-iters", type=int, default=160,
                    help="random-search samples over the 3 nuisance params per grid value")
    ap.add_argument("--abc-particles", type=int, default=240)
    ap.add_argument("--abc-tolerance", type=float, nargs="+",
                    default=[1.25, 1.15, 1.05, 0.95],
                    help="ABC-SMC tolerance-annealing schedule (shape-distance "
                         "units). NB: the empirical model-vs-real-ILI shape-distance "
                         "FLOOR is ~0.86 (the mean-field ABM cannot match single-city "
                         "ILI shape exactly), so tolerances below ~0.9 are "
                         "unreachable — the schedule anneals down toward that floor.")
    ap.add_argument("--profile-threshold", type=float, default=0.10,
                    help="absolute shape-loss threshold for the profile CI interval")
    ap.add_argument("--width-identifiable", type=float, default=0.5,
                    help="interval-frac / CI-width below this = identifiable")
    ap.add_argument("--corr-strong", type=float, default=0.6,
                    help="|r| above this = a strong (sloppy) posterior correlation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    from simulation.abm.realdata_identifiability import real_season_series
    from simulation.abm.sim_vs_observed import load_seoul_metapop, simulate_response

    if args.out_dir is None:
        try:
            from simulation.utils.paths import get_results_dir
            base = get_results_dir()
        except Exception:
            base = Path("simulation/results")
        out_dir = Path(base) / "abm_identifiability"
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = Path("simulation/results/figures/abm_identifiability.png")

    t0 = time.time()
    log.info("loading Seoul metapop (cached ~90s first call)...")
    mp = load_seoul_metapop(days=args.days)
    x_obs = _summary(real_season_series(args.season))
    log.info("observed ILI season %d-%s shape summary x_obs=%s",
             args.season, str(args.season + 1)[2:], np.round(x_obs, 3))
    if not np.all(np.isfinite(x_obs)):
        raise RuntimeError(f"observed shape summary is non-finite: {x_obs}")

    # Deterministic simulator (profile) — kwargs in, shape-summary out.
    n_sim_calls = {"n": 0}

    def simulator(kw: dict) -> np.ndarray:
        n_sim_calls["n"] += 1
        try:
            return _summary(simulate_response(mp, kw)["prevalence"])
        except Exception:
            return np.array([np.nan] * 4)

    # ABC-SMC-compatible simulator: (theta_vec, rng) -> shape summary.
    def simulator_rng(theta: np.ndarray, rng) -> np.ndarray:  # noqa: ARG001
        kw = {PARAM_NAMES[j]: float(theta[j]) for j in range(len(PARAM_NAMES))}
        n_sim_calls["n"] += 1
        try:
            return _summary(simulate_response(mp, kw)["prevalence"])
        except Exception:
            return np.array([np.nan] * 4)

    # ---- 1. Profile likelihood ----
    log.info("=== profile likelihood: %d params × %d grid × %d nuisance iters ===",
             len(PARAM_NAMES), args.grid_points, args.nuisance_iters)
    profile_pts = profile_likelihood(
        simulator, x_obs,
        grid_points=args.grid_points,
        nuisance_iters=args.nuisance_iters,
        seed=args.seed)

    # ---- 2. Posterior correlation (ABC-SMC) ----
    log.info("=== ABC-SMC posterior: %d particles, schedule %s ===",
             args.abc_particles, args.abc_tolerance)
    post = posterior_correlation(
        simulator_rng, x_obs,
        n_particles=args.abc_particles,
        tolerance_schedule=tuple(args.abc_tolerance),
        seed=args.seed)

    # ---- 3. Verdict ----
    verdict = synthesize_verdict(
        profile_pts, post,
        profile_threshold=args.profile_threshold,
        width_identifiable=args.width_identifiable,
        corr_strong=args.corr_strong)

    elapsed = time.time() - t0

    # ---- persist ----
    write_profile_csv(profile_pts, out_dir / "profile_likelihood.csv")
    write_corr_csv(post, out_dir / "posterior_corr.csv")
    verdict_out = {
        "method": {
            "profile_likelihood": "Raue et al. 2009 (fix-param, minimise over nuisance)",
            "posterior_correlation": "ABC-SMC (Toni 2009) + sloppiness (Gutenkunst 2007)",
            "observation_model": "scale-invariant epidemic SHAPE summary of city I(t) vs real national ILI",
            "observed_season": f"{args.season}-{str(args.season + 1)[2:]}",
            "x_obs": [round(float(v), 4) for v in x_obs],
        },
        "config": {
            "priors": {k: list(v) for k, v in PRIORS.items()},
            "grid_points": args.grid_points,
            "nuisance_iters": args.nuisance_iters,
            "abc_particles": args.abc_particles,
            "abc_tolerance_schedule": args.abc_tolerance,
            "seed": args.seed,
            "total_simulator_calls": n_sim_calls["n"],
            "elapsed_sec": round(elapsed, 1),
        },
        "posterior_summary": {
            "param_names": post["param_names"],
            "posterior_mean": post["posterior_mean"],
            "posterior_std": post["posterior_std"],
            "ci95": post["ci95"],
            "ci_width_vs_prior": post["ci_width_vs_prior"],
            "accept_counts": post["accept_counts"],
            "tolerance_schedule": post["tolerance_schedule"],
            "posterior_std_per_round": post["posterior_std_per_round"],
        },
        **{k: v for k, v in verdict.items() if k != "_profile_curve"},
        "profile_curves": verdict["_profile_curve"],
        "read_only_guarantee": {
            "db_writes": 0, "model_retraining": 0, "sqlite_writes": 0,
            "db_access": "read_only (load_weekly_ili → read_only_connect)",
        },
    }
    (out_dir / "verdict.json").write_text(
        json.dumps(verdict_out, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- figure ----
    try:
        make_figure(profile_pts, post, verdict, fig_path)
    except Exception as exc:  # figure is a nicety; analysis already persisted
        log.warning("figure generation failed (%s) — CSV/JSON still written",
                    type(exc).__name__)

    # ---- console summary ----
    print("\n" + "=" * 72)
    print("ABM behavioural 4-parameter practical identifiability")
    print(f"  observed: real Seoul national ILI season "
          f"{args.season}-{str(args.season + 1)[2:]}  x_obs={np.round(x_obs, 3)}")
    print(f"  simulator calls: {n_sim_calls['n']}   elapsed: {elapsed:.1f}s")
    print("-" * 72)
    print(f"{'param':7s} {'profile_frac':>12s} {'post_ci_w':>10s} {'status':>20s}")
    for p in PARAM_NAMES:
        pp = verdict["per_param"][p]
        print(f"{p:7s} {pp['profile_interval_frac']:>12.0%} "
              f"{pp['posterior_ci_width_vs_prior']:>10.2f} {pp['status']:>20s}")
    if verdict["strong_correlation_pairs"]:
        print("-" * 72)
        print("strong posterior correlations (sloppy directions):")
        for q in verdict["strong_correlation_pairs"]:
            print(f"  {q['pair'][0]:6s} ~ {q['pair'][1]:6s}  r={q['r']:+.3f}  ({q['note']})")
    print("-" * 72)
    print(verdict["verdict_paragraph"])
    print("=" * 72)
    print(f"\nwrote:\n  {out_dir / 'profile_likelihood.csv'}\n"
          f"  {out_dir / 'posterior_corr.csv'}\n  {out_dir / 'verdict.json'}\n"
          f"  {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
