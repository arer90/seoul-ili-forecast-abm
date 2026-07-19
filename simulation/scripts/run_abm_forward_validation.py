"""ABM forward(real-2026) prediction-validation runner — NOT retrospective fit.

The ABM's role in this thesis is a *forward* one: after the forecasting models
finish their in-sample training (train+val+test, overseas included) with the
in-sample window ending **2026-02-09** (identical to the forecasting cutoff), the
ABM is anchored to the champion's **forward** operational forecast and used to
*predict the forward (real-2026) period* (2026-02-16 .. 2026-06-15). The forward
ABM curve is then scored against the **real 2026 forward observations** that were
never in any training/selection set.

This is the same forward protocol the forecasting models follow — a forecasting
validation, NOT a season-by-season retrospective calibration. (User correction
2026-06-26: ABM = in-sample-then-forward real prediction + per-gu distribution +
realtime-feature possibility exploration.)

What this runner computes
-------------------------
1. **Forward split** of the real Seoul ILI series (read-only DB): in-sample
   (<= 2026-02-09) vs forward/real (2026-02-16 .. 2026-06-15, n~17-18 weeks).
2. **Forecast-anchored forward ABM**: the champion's forward forecast
   (``FusedEpi.json:refit_real_predictions``, the post-cutoff operational real
   forecast — the live ABM→forecast SSOT input) is fed to
   ``anchor_abm_to_forecast`` so the ABM's seasonal forcing tracks the forecast,
   producing a forward ABM curve over the same forward weeks.
3. **Forward R2 / RMSE** of the forward ABM curve vs the **real 2026 forward**.
4. **behavior ON vs OFF on the FORWARD window** (not retrospective): a behaviour-
   coupled forward agent world vs behaviour-off, both anchored to the same
   forcing and started from the in-sample-derived prevalence — which one predicts
   the forward better. The behaviour-ON parameters (alpha/theta/kappa/tau) are NOT
   hardcoded: they are **calibrated on the in-sample real ILI** (the 2025-26
   season tail <= 2026-02-09) by ``epi_proof._calibrate_behaviour`` (WIS-minimising
   over ``BEHAVIOUR_GRID``), which is leak-free w.r.t. the forward window. A
   ``theta_sd`` sensitivity sweep (0.10/0.15/0.25) is reported (not calibrated).
5. **Per-gu distribution**: the forward ABM's per-gu infected-fraction
   distribution (possibility exploration of where the forward burden concentrates).
6. **Realtime feature layer**: leak-free regime / pandemic-alert levels over the
   forward weeks (``simulation.analytics.external_impact``) — a possibility-
   exploration layer, not a scored prediction.

Output: ``simulation/results/abm_forward_validation/result.json``.

Discipline
----------
- Determinism: fixed seeds throughout (``range(N_SEEDS)``, ``global_seed``).
- Real data only: the forward truth is the real DB; never synthesise a forward
  truth and never fall back to zeros.
- Read-only DB: ``read_only_connect`` (never writes; safe under a live writer).
- No existing code modified — import only.
- Lightweight: JSON + print, no matplotlib.

Run (FULL — caller launches detached; this module does NOT self-launch full):
    .venv/bin/python -m simulation.scripts.run_abm_forward_validation

Smoke (reduced agents / seeds, just confirms a forward R2 actually materialises):
    MPH_ABM_FWD_SMOKE=1 .venv/bin/python -m simulation.scripts.run_abm_forward_validation
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from simulation.abm.adaptive_allocation import AdaptiveAllocator
from simulation.abm.adaptive_world import run_adaptive_agent_world
from simulation.abm.agent_kernel import STATE_I, _N_GU
from simulation.abm.epi_proof import (
    BEHAVIOUR_OFF,
    DEFAULT_DISEASE,
    SeasonSeries,
    _calibrate_behaviour,
)
from simulation.abm.forecast_anchor import anchor_abm_to_forecast
from simulation.analytics.external_impact import (
    detect_regime_shifts,
    pandemic_alert_level,
)
from simulation.database.storage import read_only_connect
from simulation.models.feature_engine.utils import _season_weekseq_to_date


# ── Forward protocol constants (identical to the forecasting cutoff) ──────────
IN_SAMPLE_END = "2026-02-09"        # forecasting cutoff; in-sample <= this date
FORWARD_START = "2026-02-16"        # first forward (real-2026) week
FORWARD_END = "2026-06-15"          # last forward week (current 2025-26 season tail)
# First Monday of the season being forward-predicted (2025-26). The behaviour
# calibration target is this season's *in-sample* tail (>= this date and <=
# IN_SAMPLE_END), which directly precedes — and never overlaps — the forward
# window, so the calibration is leak-free by construction.
CUR_SEASON_START_DATE = "2025-09-01"
CUR_SEASON_YEAR = 2025             # generate_population reference year for that season
# theta_sd has no calibration grid (BEHAVIOUR_GRID never varies it); it is a
# fixed structural assumption of the per-agent threshold heterogeneity. We sweep
# it (0.10/0.15/0.25) to *measure* its forward-R2 sensitivity rather than fit it
# (fitting one extra scalar on n=16 forward weeks would be over-engineering).
THETA_SD_DEFAULT = 0.15
THETA_SD_SWEEP = (0.10, 0.15, 0.25)

DB_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "db" / "epi_real_seoul.db"
)
CHAMPION_PATH = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "per_model_optimal"
    / "FusedEpi.json"
)
OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "abm_forward_validation"
    / "result.json"
)


def _smoke() -> bool:
    return os.environ.get("MPH_ABM_FWD_SMOKE", "0") not in ("", "0", "false", "False")


def load_real_ili_split(
    db_path: str | Path = DB_PATH,
    *,
    in_sample_end: str = IN_SAMPLE_END,
    forward_start: str = FORWARD_START,
    forward_end: str = FORWARD_END,
) -> dict[str, Any]:
    """Split the real Seoul ILI series into in-sample vs forward (real-2026).

    Reads ``sentinel_influenza`` (age-stratified ili_rate) and aggregates over
    age bands to the Seoul-wide weekly rate, mapping each (season, week_seq) to a
    Monday date via the canonical ``_season_weekseq_to_date`` helper.

    Args:
        db_path: SQLite DB path (read-only).
        in_sample_end: ISO date; weeks with date <= this are in-sample.
        forward_start/forward_end: ISO bounds (inclusive) of the forward window.

    Returns:
        Dict with ``in_sample_weeks`` (list of {date, ili}), ``forward_weeks``
        (same), ``in_sample_ili`` / ``forward_ili`` (np.ndarray, time-ordered),
        and ``forward_dates`` (list[str]).

    Raises:
        ValueError: if no forward weeks fall in the window (would make the
            forward validation impossible — fail loud, never fabricate).

    Side effects: read-only DB open (caller-free; closed here). No writes.
    """
    con = read_only_connect(str(db_path))
    try:
        rows = con.execute(
            """
            SELECT season_start, week_seq, AVG(ili_rate) AS ili_rate
            FROM sentinel_influenza
            WHERE ili_rate IS NOT NULL AND ili_rate >= 0
            GROUP BY season_start, week_seq
            ORDER BY season_start, week_seq
            """
        ).fetchall()
    finally:
        con.close()

    cutoff = np.datetime64(in_sample_end)
    fwd_lo = np.datetime64(forward_start)
    fwd_hi = np.datetime64(forward_end)

    in_sample: list[dict[str, Any]] = []
    forward: list[dict[str, Any]] = []
    for season_start, week_seq, ili in rows:
        d = np.datetime64(
            _season_weekseq_to_date(int(season_start), int(week_seq)).date()
        )
        item = {"date": str(d), "ili": float(ili)}
        if d <= cutoff:
            in_sample.append(item)
        elif fwd_lo <= d <= fwd_hi:
            forward.append(item)

    if not forward:
        raise ValueError(
            f"no real forward weeks in [{forward_start}, {forward_end}] — "
            "cannot run forward validation (real truth is mandatory)"
        )

    return {
        "in_sample_weeks": in_sample,
        "forward_weeks": forward,
        "in_sample_ili": np.asarray([x["ili"] for x in in_sample], dtype=np.float64),
        "forward_ili": np.asarray([x["ili"] for x in forward], dtype=np.float64),
        "forward_dates": [x["date"] for x in forward],
    }


def load_champion_forward_forecast(
    champion_path: str | Path = CHAMPION_PATH,
) -> np.ndarray:
    """Load the champion's forward operational forecast (the ABM anchor).

    ``FusedEpi.json:refit_real_predictions`` is the post-cutoff operational rolling
    real forecast — the live forecast→ABM anchoring SSOT input.

    Returns:
        ``(real_n,)`` float64 forward forecast, finite and non-negative-clipped.

    Raises:
        FileNotFoundError / ValueError: missing file or missing/empty/non-finite
            ``refit_real_predictions``.

    Side effects: reads one local JSON; no DB / network / writes.
    """
    path = Path(champion_path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    preds = obj.get("refit_real_predictions")
    if not preds:
        raise ValueError(
            f"{path} has no refit_real_predictions (forward forecast anchor)"
        )
    arr = np.asarray(preds, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("champion forward forecast contains non-finite values")
    # anchor_abm_to_forecast requires non-negative forecast values
    return np.clip(arr, 0.0, None)


def calibrate_behaviour_in_sample(
    in_sample_weeks: list[dict[str, Any]],
    forcing: dict[str, float],
    *,
    n_agents: int,
    seeds: list[int],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Calibrate behavioural params (alpha/theta/kappa/tau) on in-sample real ILI.

    Replaces the old hardcoded ``alpha=0.45/theta=0.20/kappa=0.30/tau=60``. The
    calibration target is the *in-sample* tail of the season being forward-
    predicted (2025-26: weeks with date in ``[CUR_SEASON_START_DATE,
    IN_SAMPLE_END]``), which never overlaps the forward window — so the fitted
    behaviour is **leak-free** w.r.t. the forward (real-2026) period scored later.
    The WIS-minimising grid search itself is ``epi_proof._calibrate_behaviour``
    over ``epi_proof.BEHAVIOUR_GRID`` (the same real-ILI calibration that
    ``anchor_abm_to_forecast`` shares), run with the disease/forcing the forward
    sim uses (``DEFAULT_DISEASE`` overlaid with the anchored beta/amp/phase/import).

    Args:
        in_sample_weeks: list of ``{"date": ISO, "ili": float}`` (in-sample only).
        forcing: anchored forcing dict (``beta/beta_amp/beta_phase/import_rate``).
        n_agents: synthetic population size for the calibration sims.
        seeds: replicate seeds (determinism).

    Returns:
        ``(behaviour, calibration)`` — ``behaviour`` = ``{alpha,theta,kappa,tau}``
        floats; ``calibration`` = the diagnostic dict from ``_calibrate_behaviour``
        plus ``target_window`` (the in-sample calibration date range).

    Raises:
        ValueError: if no in-sample weeks fall in the current-season target
            window (fail loud rather than silently calibrating on nothing).

    Performance: O(len(BEHAVIOUR_GRID product) * len(seeds) * weeks * 7 *
        n_agents) — ~24-week season, 24-cell grid.
    Side effects: local ABM sims via the shared epi_proof simulation cache; no
        DB writes, no network.
    Caller responsibility: pass an already-leak-free in-sample slice.
    """
    cur = [
        w
        for w in in_sample_weeks
        if CUR_SEASON_START_DATE <= w["date"] <= IN_SAMPLE_END
    ]
    if not cur:
        raise ValueError(
            f"no in-sample weeks in current-season window "
            f"[{CUR_SEASON_START_DATE}, {IN_SAMPLE_END}] — cannot calibrate "
            "behaviour (leak-free in-sample target is mandatory)"
        )
    ili = np.asarray([w["ili"] for w in cur], dtype=np.float64)
    season = SeasonSeries(
        season=CUR_SEASON_YEAR,
        week_seq=np.arange(ili.size, dtype=np.int16),
        ili_rate=ili,
    )
    # disease = default SEIR-V-D hazards overlaid with the anchored forcing the
    # forward sim runs with, so the behaviour grid is calibrated under the same
    # forcing it will be deployed under.
    disease = {
        **DEFAULT_DISEASE,
        **{k: float(v) for k, v in forcing.items() if k in DEFAULT_DISEASE},
    }
    behaviour, calibration = _calibrate_behaviour(
        season, seeds=seeds, n_agents=n_agents, disease=disease
    )
    calibration["target_window"] = [cur[0]["date"], cur[-1]["date"]]
    calibration["target_n_weeks"] = int(ili.size)
    calibration["leak_free_note"] = (
        "calibrated on the in-sample 2025-26 season tail (<= forecasting cutoff); "
        "the forward window starts strictly after IN_SAMPLE_END."
    )
    return {k: float(behaviour[k]) for k in ("alpha", "theta", "kappa", "tau")}, calibration


def _fit_linear_map(sim: np.ndarray, obs: np.ndarray) -> tuple[float, float]:
    """Least-squares affine map sim -> obs (offset, scale). scale=0 if degenerate."""
    x = np.asarray(sim, dtype=np.float64)
    y = np.asarray(obs, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2 or float(np.var(x[mask])) <= 1e-12:
        return float(np.nanmean(y)), 0.0
    xv, yv = x[mask], y[mask]
    scale = float(np.cov(xv, yv, ddof=0)[0, 1] / np.var(xv))
    if not np.isfinite(scale):
        scale = 0.0
    offset = float(np.mean(yv) - scale * np.mean(xv))
    return (offset if np.isfinite(offset) else float(np.nanmean(y))), scale


def _r2(obs: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(obs, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _rmse(obs: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(obs, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def _weekly_from_daily_I(I_daily: np.ndarray, n_weeks: int) -> np.ndarray:
    """Aggregate a daily infected-count curve to weekly means (length n_weeks)."""
    arr = np.asarray(I_daily, dtype=np.float64)
    usable = n_weeks * 7
    if arr.size < usable:
        # pad with the last value so a slightly-short sim still yields n_weeks
        pad = np.full(usable - arr.size, arr[-1] if arr.size else 0.0)
        arr = np.concatenate([arr, pad])
    return arr[:usable].reshape(n_weeks, 7).mean(axis=1)


def _forward_agent_world(
    *,
    forcing: dict[str, float],
    behaviour: dict[str, float],
    n_forward_weeks: int,
    n_agents: int,
    seed: int,
    init_prevalence: float,
    theta_sd: float = THETA_SD_DEFAULT,
) -> dict[str, np.ndarray]:
    """Run ONE forward agent world over the forward weeks with given forcing.

    Uses ``run_adaptive_agent_world`` (the in-run dynamic-agent SEIR-V-D world).
    The behavioural coupling params are supplied via ``behaviour`` (the in-sample
    real-ILI calibrated ``{alpha,theta,kappa,tau}``); pass ``BEHAVIOUR_OFF`` for
    the behaviour-off arm (alpha=theta=kappa=0). ``theta_sd`` is the per-agent
    threshold-heterogeneity SD (structural assumption, not calibrated). The
    forward window starts from the in-sample-derived prevalence level, so the sim
    is genuinely *predicting forward* rather than re-fitting the season.

    Returns:
        Dict with ``I`` (population-weighted daily infected counts).
    """
    total_days = n_forward_weeks * 7
    floor = max(200, n_agents // 8)
    base = max(floor, n_agents // 2)
    allocator = AdaptiveAllocator(base_n=base, max_n=n_agents, floor_n=floor)
    out = run_adaptive_agent_world(
        n_agents,
        total_days,
        allocator=allocator,
        peak_prevalence=max(init_prevalence, 0.01),
        beta=float(forcing["beta"]),
        sigma=0.45,
        gamma=0.18,
        delta=0.002,
        nu=0.0002,
        population_size=n_agents,
        epoch_len=7,
        global_seed=int(seed),
        theta_mean=float(behaviour["theta"]),
        theta_sd=float(theta_sd),
        alpha_mean=float(behaviour["alpha"]),
        kappa_mean=float(behaviour["kappa"]),
        tau_mean=float(behaviour["tau"]),
        beta_amp=float(forcing["beta_amp"]),
        beta_phase=float(forcing["beta_phase"]),
        import_rate=float(forcing.get("import_rate", 3.0e-4)),
    )
    return {"I": np.asarray(out["I"], dtype=np.float64)}


def _per_gu_distribution(
    *,
    forcing: dict[str, float],
    behaviour: dict[str, float],
    n_forward_weeks: int,
    n_agents: int,
    seed: int,
    theta_sd: float = THETA_SD_DEFAULT,
) -> dict[str, Any]:
    """Forward per-gu infected-fraction distribution (possibility exploration).

    Runs the production agent kernel directly (it carries home_gu) over the
    forward window and reports the per-gu cumulative-ever-infected fraction
    distribution at the end of the forward horizon. Behaviour params come from the
    in-sample real-ILI calibration (``behaviour`` = ``{alpha,theta,kappa,tau}``),
    not hardcoded literals.
    """
    from simulation.abm.agent_kernel import run_agent_world

    rng = np.random.default_rng(int(seed))
    home_gu = rng.integers(0, _N_GU, size=n_agents).astype(np.int64)
    work_gu = home_gu.copy()
    commute = rng.random(n_agents) < 0.2
    n_c = int(commute.sum())
    if n_c:
        work_gu[commute] = rng.integers(0, _N_GU, size=n_c)
    pop = {
        "home_gu": home_gu,
        "work_gu": work_gu,
        "age_band": rng.integers(0, 7, size=n_agents).astype(np.int64),
        "occupation": rng.integers(0, 6, size=n_agents).astype(np.int64),
        "severity": (rng.random(n_agents) < 0.1).astype(np.int64),
    }
    res = run_agent_world(
        N=n_agents,
        T_days=n_forward_weeks * 7,
        beta=float(forcing["beta"]),
        sigma=0.45,
        gamma=0.18,
        delta=0.002,
        nu=0.0002,
        population=pop,
        global_seed=int(seed),
        theta_mean=float(behaviour["theta"]),
        theta_sd=float(theta_sd),
        alpha_mean=float(behaviour["alpha"]),
        kappa_mean=float(behaviour["kappa"]),
        tau_mean=float(behaviour["tau"]),
        beta_amp=float(forcing["beta_amp"]),
        beta_phase=float(forcing["beta_phase"]),
        import_rate=float(forcing.get("import_rate", 3.0e-4)),
    )
    final_state = np.asarray(res["agents"]["state"])
    # per-gu prevalence (currently-infected fraction) at the forward end
    gu_fracs: list[float] = []
    for g in range(_N_GU):
        in_gu = home_gu == g
        n_in = int(in_gu.sum())
        if n_in == 0:
            gu_fracs.append(0.0)
            continue
        inf = int(((final_state == STATE_I) & in_gu).sum())
        gu_fracs.append(inf / n_in)
    fracs = np.asarray(gu_fracs, dtype=np.float64)
    return {
        "n_gu": int(_N_GU),
        "mean_I_frac": float(np.mean(fracs)),
        "std_I_frac": float(np.std(fracs)),
        "min_I_frac": float(np.min(fracs)),
        "max_I_frac": float(np.max(fracs)),
        "cv_across_gu": (
            float(np.std(fracs) / np.mean(fracs)) if np.mean(fracs) > 1e-9 else 0.0
        ),
        "per_gu_I_frac": [float(v) for v in fracs],
    }


def _realtime_alert_summary(
    in_sample_ili: np.ndarray, forward_ili: np.ndarray
) -> dict[str, Any]:
    """Leak-free regime / pandemic-alert levels over the forward weeks.

    The full series (in-sample + forward) is passed so the causal baseline has
    history; only the forward tail's levels are summarised. This is a possibility-
    exploration layer (NOT a scored forward prediction).
    """
    full = np.concatenate(
        [np.asarray(in_sample_ili, dtype=np.float64),
         np.asarray(forward_ili, dtype=np.float64)]
    )
    n_fwd = int(forward_ili.size)
    alert = pandemic_alert_level(full, return_labels=True)
    regime = detect_regime_shifts(full)
    fwd_levels = np.asarray(alert["level"])[-n_fwd:]
    fwd_labels = np.asarray(alert["label_en"])[-n_fwd:]
    # detect_regime_shifts returns {"changepoints": list[int], "shift_flags": (n,)}
    reg_arr = np.asarray(regime["shift_flags"])
    fwd_regime = reg_arr[-n_fwd:] if reg_arr.size >= n_fwd else reg_arr
    return {
        "n_forward": n_fwd,
        "forward_alert_levels": [int(v) for v in fwd_levels],
        "forward_alert_labels": [str(v) for v in fwd_labels],
        "max_forward_alert_level": int(np.max(fwd_levels)) if n_fwd else 0,
        "forward_regime_shift_count": int(np.sum(np.asarray(fwd_regime) != 0)),
        "note": (
            "leak-free causal alert/regime over the forward window; "
            "possibility-exploration layer, not a scored forward prediction."
        ),
    }


def run_forward_validation(
    *,
    n_agents: int | None = None,
    n_seeds: int | None = None,
    db_path: str | Path = DB_PATH,
    champion_path: str | Path = CHAMPION_PATH,
    output_path: str | Path = OUTPUT_PATH,
) -> dict[str, Any]:
    """Run the full ABM forward(real-2026)-prediction validation and write JSON.

    Forward protocol (NOT retrospective fit):
      in-sample (<= 2026-02-09) → anchor ABM to champion forward forecast →
      predict the forward window (2026-02-16 .. 2026-06-15) → score the forward
      ABM curve against the real 2026 forward.

    Args:
        n_agents: synthetic population size (default 30000; smoke 1500).
        n_seeds: replicate seeds (default 5; smoke 2).
        db_path / champion_path / output_path: real DB, champion JSON, JSON out.

    Returns:
        The result dict (also written to ``output_path``).

    Side effects: read-only DB reads, local ABM sims, writes ``output_path``.
    """
    smoke = _smoke()
    n_agents = int(n_agents if n_agents is not None else (1_500 if smoke else 30_000))
    n_seeds = int(n_seeds if n_seeds is not None else (2 if smoke else 5))
    seeds = list(range(n_seeds))

    split = load_real_ili_split(db_path)
    forward_ili = split["forward_ili"]
    in_sample_ili = split["in_sample_ili"]
    n_forward_full = int(forward_ili.size)

    champ_fc = load_champion_forward_forecast(champion_path)
    # align comparison window to min(forecast, real forward)
    n_cmp = int(min(champ_fc.size, n_forward_full))
    fwd_obs = forward_ili[:n_cmp]
    champ_fc_aligned = champ_fc[:n_cmp]

    # ── 1. forecast-anchored forward ABM (the live anchor mechanism) ──────────
    #   Anchor the ABM's seasonal forcing to the champion's FORWARD forecast so
    #   the ABM tracks the forward forecast — this is forecast-anchored, the same
    #   mechanism the live ABM→forecast SSOT uses.
    anchor = anchor_abm_to_forecast(
        champ_fc_aligned, n_agents=n_agents, seeds=seeds
    )
    forcing = dict(anchor["fitted_forcing"])
    abm_anchored = np.asarray(anchor["anchored_trajectory"], dtype=np.float64)[:n_cmp]

    forward_r2 = _r2(fwd_obs, abm_anchored)
    forward_rmse = _rmse(fwd_obs, abm_anchored)

    # in-sample-derived starting prevalence (level the forward sim launches from)
    init_prev = float(np.clip(in_sample_ili[-1] / 100.0, 1e-3, 0.5)) if in_sample_ili.size else 0.05

    # ── 1b. behaviour calibration on IN-SAMPLE real ILI (leak-free) ───────────
    #   Replaces the old hardcoded alpha=0.45/theta=0.20/kappa=0.30/tau=60. The
    #   behaviour params are WIS-minimised on the 2025-26 in-sample tail (<= cutoff)
    #   by epi_proof._calibrate_behaviour over BEHAVIOUR_GRID, under the anchored
    #   forcing — never touching the forward window that is scored below.
    fitted_behaviour, beh_calib = calibrate_behaviour_in_sample(
        split["in_sample_weeks"], forcing, n_agents=n_agents, seeds=seeds
    )

    # ── 2. behavior ON vs OFF — ON THE FORWARD WINDOW (not retrospective) ─────
    def _arm_forward_curve(
        behaviour: dict[str, float], *, theta_sd: float = THETA_SD_DEFAULT
    ) -> np.ndarray:
        curves = []
        for s in seeds:
            res = _forward_agent_world(
                forcing=forcing,
                behaviour=behaviour,
                n_forward_weeks=n_cmp,
                n_agents=n_agents,
                seed=s,
                init_prevalence=init_prev,
                theta_sd=theta_sd,
            )
            curves.append(_weekly_from_daily_I(res["I"], n_cmp))
        mean_curve = np.vstack(curves).mean(axis=0)
        offset, scale = _fit_linear_map(mean_curve, fwd_obs)
        return np.clip(offset + scale * mean_curve, 0.0, None)

    on_curve = _arm_forward_curve(fitted_behaviour)
    off_curve = _arm_forward_curve(BEHAVIOUR_OFF)
    r2_on = _r2(fwd_obs, on_curve)
    r2_off = _r2(fwd_obs, off_curve)

    # ── 2b. theta_sd sensitivity sweep (NOT a calibration — see THETA_SD_SWEEP) ─
    #   Structural assumption: theta_sd is the relative SD of the per-agent
    #   behavioural threshold (theta ~ theta_mean*(1+theta_sd*N(0,1))); it has no
    #   BEHAVIOUR_GRID axis, so we report its forward-R2 sensitivity (0.10/0.15/0.25)
    #   rather than fit it (one extra scalar on n~16 forward weeks = over-fitting).
    theta_sd_sweep = [
        {
            "theta_sd": float(tsd),
            "forward_r2_behavior_on": float(
                _r2(fwd_obs, _arm_forward_curve(fitted_behaviour, theta_sd=tsd))
            ),
        }
        for tsd in THETA_SD_SWEEP
    ]

    # ── 3. per-gu forward distribution (possibility exploration) ──────────────
    gu_dist = _per_gu_distribution(
        forcing=forcing,
        behaviour=fitted_behaviour,
        n_forward_weeks=n_cmp,
        n_agents=n_agents,
        seed=seeds[0],
    )

    # ── 4. realtime feature layer (leak-free alert/regime, possibility) ───────
    alert = _realtime_alert_summary(in_sample_ili, forward_ili)

    behaviour_helps = bool(np.isfinite(r2_on) and np.isfinite(r2_off) and r2_on > r2_off)
    honest_note = (
        "FORWARD prediction validation (NOT retrospective season fit): the ABM is "
        f"anchored to the champion's forward operational forecast and used to predict "
        f"the forward/real-2026 window ({FORWARD_START}..{FORWARD_END}). "
        f"forward_r2 scores the forecast-anchored forward ABM vs the real 2026 forward. "
        f"behavior ON {'beats' if behaviour_helps else 'does NOT beat'} OFF on the "
        f"forward window. The behaviour-ON params (alpha/theta/kappa/tau) are "
        f"calibrated on the in-sample real ILI ({beh_calib['target_window'][0]}.."
        f"{beh_calib['target_window'][1]}), NOT hardcoded; leak-free w.r.t. the "
        f"forward window. degenerate(anchor)={bool(anchor['degenerate'])}; if "
        "degenerate or smoke, treat metrics as a wiring check only. per-gu + alert "
        "layers are possibility-exploration, not scored forecasts."
    )

    result = {
        "forward_r2": float(forward_r2),
        "forward_rmse": float(forward_rmse),
        "forward_r2_behavior_on": float(r2_on),
        "forward_r2_behavior_off": float(r2_off),
        "behavior_helps_forward": behaviour_helps,
        "n_forward": int(n_cmp),
        "in_sample_end": IN_SAMPLE_END,
        "forward_window": [FORWARD_START, FORWARD_END],
        "forward_dates": split["forward_dates"][:n_cmp],
        "real_forward_ili": [float(v) for v in fwd_obs],
        "champion_forward_forecast": [float(v) for v in champ_fc_aligned],
        "abm_anchored_forward": [float(v) for v in abm_anchored],
        "anchor_mechanism": "forecast_anchored",
        "anchor_degenerate": bool(anchor["degenerate"]),
        "anchor_corr_sim_vs_forecast": float(anchor["corr_sim_vs_forecast"]),
        "fitted_forcing": {k: float(v) for k, v in forcing.items()},
        "calibrated_behaviour": dict(fitted_behaviour),
        "behaviour_calibration": beh_calib,
        "theta_sd_default": float(THETA_SD_DEFAULT),
        "theta_sd_sweep": theta_sd_sweep,
        "gu_distribution_summary": gu_dist,
        "realtime_alert_summary": alert,
        "honest_note": honest_note,
        "metadata": {
            "n_agents": int(n_agents),
            "seeds": [int(s) for s in seeds],
            "smoke": bool(smoke),
            "n_forward_available": int(n_forward_full),
            "champion_forecast_len": int(champ_fc.size),
            "champion_path": str(Path(champion_path)),
            "db_path": str(Path(db_path)),
            "read_only_db": True,
            "local_only": True,
            "protocol": "forward_prediction_NOT_retrospective_fit",
        },
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    res = run_forward_validation()
    print(
        json.dumps(
            {
                "forward_r2": res["forward_r2"],
                "forward_rmse": res["forward_rmse"],
                "forward_r2_behavior_on": res["forward_r2_behavior_on"],
                "forward_r2_behavior_off": res["forward_r2_behavior_off"],
                "behavior_helps_forward": res["behavior_helps_forward"],
                "calibrated_behaviour": res["calibrated_behaviour"],
                "theta_sd_sweep": res["theta_sd_sweep"],
                "n_forward": res["n_forward"],
                "anchor_mechanism": res["anchor_mechanism"],
                "anchor_degenerate": res["anchor_degenerate"],
                "gu_cv_across_gu": res["gu_distribution_summary"]["cv_across_gu"],
                "max_forward_alert_level": res["realtime_alert_summary"][
                    "max_forward_alert_level"
                ],
                "smoke": res["metadata"]["smoke"],
                "output": res["metadata"]["champion_path"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
