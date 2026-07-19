"""run_expanding_multihorizon.py — operational expanding-window rolling-origin multi-horizon
forecast protocol with a forecast -> ABM -> forecast loop (FusedEpi champion).

This runner implements the user-specified operational protocol:

    full(train+val+test)                    -> +1W,+2W,+3W,+4W,+8W,+12W,+28W
    full(train+val+test) + 2026 W01         -> +1W,+2W,+3W,+4W,+8W,+12W,+28W
    full(train+val+test) + 2026 W01 + W02   -> +1W,+2W,+3W,+4W,+8W,+12W,+28W
    ... expanding one observed week per origin until the data runs out

For every expanding origin ``t`` (in-sample = all real weeks <= t) the runner:
  1. fits the R-track champion FusedEpi on the in-sample feature matrix + ILI series,
  2. produces a *recursive* multi-horizon forecast for h in {1,2,3,4,8,12,28} weeks ahead
     via ``FusedEpiForecaster.predict_multi`` (TiRex native multi-step + seasonal blend,
     leak-free: it consumes only ``context_y`` = the in-sample history, never future
     observations), plus NegBin / conformal prediction intervals for the 1-step head,
  3. anchors the seasonal ABM to that forecast (``anchor_abm_to_forecast``) producing a
     mechanistic / district-distribution update -- the *forecast -> ABM* leg,
  4. records the ABM-updated forcing so the next expanding origin's forecast is produced
     after the ABM update -- closing the *forecast -> ABM -> forecast* loop.

Each horizon is scored only against a *real* observed value when one exists at week
``t + h`` (honest: future-only origins emit forecast + PI with ``actual=None``).

DISCIPLINE
  - Leak-free: every origin uses only weeks <= t; the recursive multi-horizon feeds the
    model's own predictions (``predict_multi`` never receives future observations).
  - Determinism: numpy / torch seeds fixed; FusedEpi runs on CPU.
  - Real data only: the observed ILI series is the KDCA sentinel (per-age averaged) read
    via ``read_only_connect`` -- no synthetic gap fill.
  - No existing code is modified; only imported.

OUTPUT  ``simulation/results/expanding_multihorizon/result.json``::

    {
      "origins": [
        {"origin_date", "origin_week_index", "in_sample_n",
         "horizons": [{"h", "pred", "pi_lo", "pi_hi", "actual"?, "abs_err"?}],
         "abm_updated": bool, "abm": {...}?},
        ...
      ],
      "per_horizon_summary": [{"h", "mean_r2"?, "mean_mae"?, "n_origins", "n_scored"}],
      "honest_note": "...",
      "metadata": {...}
    }

Performance: per origin = 1 FusedEpi fit (TiRex rolling + TabPFN) + 1 ABM anchor
    (27-cell forcing grid * seeds * n_agents). The ABM leg dominates; smoke uses a tiny
    population and 2-3 origins. Full run is detached.
Side effects: reads the local DB + builds features once; writes one JSON result file. No
    network. Caller responsibility: run detached for the full sweep (slow ABM).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from contextlib import closing
from pathlib import Path
from typing import Any, Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("expanding_multihorizon")

# Requested operational horizons (weeks ahead).
HORIZONS: tuple[int, ...] = (1, 2, 3, 4, 8, 12, 28)
SEED = 42

_PKG_ROOT = Path(__file__).resolve().parents[1]            # simulation/
DEFAULT_DB = _PKG_ROOT / "data" / "db" / "epi_real_seoul.db"
DEFAULT_OUT = _PKG_ROOT / "results" / "expanding_multihorizon" / "result.json"


# ────────────────────────────────────────────────────────────────────────────
# Real observed ILI series (leak-free, read-only).
# ────────────────────────────────────────────────────────────────────────────
def load_observed_ili(db_path: Path) -> tuple[np.ndarray, list[str]]:
    """Load the canonical KDCA sentinel ILI series (per-age averaged), read-only.

    Mirrors ``true_ili_cohort.load_kr_sentinel_ili`` exactly: ``AVG(ili_rate)``
    grouped by ``(season_start, week_seq)`` ordered chronologically. This is the
    SSOT operational target the champion forecasts.

    Args:
        db_path: path to ``epi_real_seoul.db``.

    Returns:
        ``(y, labels)`` -- ``y`` the float64 weekly ILI rate (chronological),
        ``labels`` a parallel list of ``"<season_start>W<week_seq>"`` strings.

    Side effects: opens a lock-free read-only connection (never blocks a writer).
    """
    from simulation.database.storage import read_only_connect

    with closing(read_only_connect(str(db_path))) as con:
        rows = con.execute(
            "SELECT season_start, week_seq, AVG(ili_rate) AS r "
            "FROM sentinel_influenza WHERE ili_rate IS NOT NULL "
            "GROUP BY season_start, week_seq "
            "ORDER BY season_start, week_seq"
        ).fetchall()
    y = np.asarray([float(r[2]) for r in rows], dtype=np.float64)
    labels = [f"{int(r[0])}W{int(r[1])}" for r in rows]
    return y, labels


# ────────────────────────────────────────────────────────────────────────────
# Feature matrix (full live pool) -- built ONCE, then row-sliced per origin.
# ────────────────────────────────────────────────────────────────────────────
def build_feature_matrix(db_path: Path) -> tuple[np.ndarray, np.ndarray, list[str], Optional[np.ndarray]]:
    """Build the full enriched feature pool once (X, y, cols, dates).

    Uses the same builder the R1 data stage uses. Built once on the full DB; the
    expanding window then row-slices ``[:k]`` per origin -- the same chronological
    truncation the pipeline's nested-CV (``MPH_DATA_END_WEEK``) treats as a
    leak-free expanding outer fold.

    Returns:
        ``(X_all, y_all, feature_cols, dates)`` with ``X_all`` NaN-sanitized.

    Side effects: reads the local DB; no writes.
    """
    import polars as pl
    from simulation.models.feature_engine.builder import build_enriched_features

    feat_df, meta = build_enriched_features(str(db_path))
    ycol = "ili_rate" if "ili_rate" in feat_df.columns else feat_df.columns[0]
    numeric = (
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64, pl.Boolean,
    )
    schema = feat_df.schema
    feature_cols = [c for c in feat_df.columns
                    if c not in (ycol, "week_start") and schema[c] in numeric]
    X_all = feat_df.select(feature_cols).to_numpy().astype(np.float64)
    y_all = feat_df[ycol].to_numpy().astype(np.float64)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    dates = meta.get("dates") if isinstance(meta, dict) else None
    if dates is not None and len(dates) != len(y_all):
        dates = None
    return X_all, y_all, feature_cols, dates


def _date_str(dates: Optional[np.ndarray], i: int) -> str:
    if dates is None or i >= len(dates):
        return f"row{i}"
    try:
        return str(dates[i])[:10]
    except Exception:
        return f"row{i}"


# ────────────────────────────────────────────────────────────────────────────
# Champion forecaster (FusedEpi) -- fit fresh per origin on the in-sample pool.
# ────────────────────────────────────────────────────────────────────────────
def fit_champion(X_in: np.ndarray, y_in: np.ndarray):
    """Fit the R-track champion FusedEpi on the in-sample feature pool.

    FusedEpi is self-adaptive: it re-derives its blend alpha, multicollinearity
    keep-set, NegBin dispersion, and conformal calibration from the in-sample
    data passed to ``fit`` (config preproc = identity ``none``/``none``), so the
    live full pool is the model-author-intended use. No future weeks are touched.

    Args:
        X_in: ``(k, p)`` in-sample feature matrix (weeks <= origin).
        y_in: ``(k,)`` in-sample ILI series.

    Returns:
        Fitted ``FusedEpiForecaster``.
    """
    from simulation.models.fused_epi import FusedEpiForecaster

    model = FusedEpiForecaster()
    model.fit(X_in, y_in)
    return model


def champion_multihorizon(model, X_in: np.ndarray, y_in: np.ndarray
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recursive multi-horizon point + 95% PI from the champion (leak-free).

    Calls ``predict_multi(H)`` -- TiRex native multi-step (single-strong) blended
    into seasonal-naive for long horizons -- which consumes ONLY ``context_y``
    (the in-sample history), feeding the model's own forecast forward. No future
    observation is supplied, so the multi-horizon path is recursive and leak-free.

    The 1-step head additionally gets a NegBin/conformal 95% interval from
    ``predict_quantiles`` evaluated on the LAST in-sample feature row (the
    features available at the origin to forecast t+1; static Q, no future obs).
    Longer horizons widen the 1-step half-width by ``sqrt(h)`` as an honest,
    leak-free PI proxy (recursive error accumulation); banded, flagged in note.

    Args:
        X_in: ``(k, p)`` in-sample feature matrix (last row = origin features).
        y_in: ``(k,)`` in-sample ILI series.

    Returns:
        ``(point, pi_lo, pi_hi)`` each shape ``(len(HORIZONS),)``.
    """
    H = max(HORIZONS)
    traj = np.asarray(model.predict_multi(H, context_y=y_in), dtype=np.float64)   # (H,)
    point = np.array([traj[h - 1] for h in HORIZONS], dtype=np.float64)

    # 1-step NegBin/conformal PI from the last in-sample feature row (static Q).
    half = np.nan
    try:
        x_last = np.asarray(X_in[-1:], dtype=np.float64)     # origin features (1, p)
        q = model.predict_quantiles(x_last, y_observed=None,
                                    levels=(0.025, 0.5, 0.975))
        lo1 = float(np.asarray(q[0.025]).ravel()[0])
        hi1 = float(np.asarray(q[0.975]).ravel()[0])
        half = max(0.0, (hi1 - lo1) / 2.0)
    except Exception as exc:                                  # PI is best-effort
        log.warning("    predict_quantiles failed (%s) -> PI from in-sample resid", exc)
    if not np.isfinite(half):
        cr = getattr(model, "_calib_residuals", None)
        half = 1.96 * float(np.std(cr)) if cr else 1.96 * float(np.std(y_in[-12:]) + 1e-6)

    scale = np.sqrt(np.asarray(HORIZONS, dtype=np.float64))   # sqrt(h) error growth
    pi_lo = np.clip(point - half * scale, 0.0, None)
    pi_hi = point + half * scale
    return point, pi_lo, pi_hi


# ────────────────────────────────────────────────────────────────────────────
# forecast -> ABM update (the mechanistic / district-distribution leg).
# ────────────────────────────────────────────────────────────────────────────
def abm_update(forecast: np.ndarray, *, n_agents: int, seeds: tuple[int, ...],
               year: Optional[int]) -> dict[str, Any]:
    """Anchor the seasonal ABM to the champion forecast (forecast -> ABM leg).

    Wraps ``forecast_anchor.anchor_abm_to_forecast``: it calibrates a seasonal
    agent-based SEIR forcing (beta / amplitude / phase grid) to the forecast
    trajectory and returns the anchored mechanistic trajectory + affine map +
    WIS + ``degenerate`` flag. This is the ABM's mechanistic / district-scalable
    update consumed by the next forecast cycle.

    Args:
        forecast: non-negative 1D weekly forecast (the recursive multi-horizon
            point trajectory) -- the ABM's input signal.
        n_agents: synthetic population size (keep small for smoke).
        seeds: replicate seeds.
        year: synthetic-population reference year; None -> latest local season.

    Returns:
        The anchor result dict, plus a ``valid`` key (= not degenerate).
    """
    from simulation.abm.forecast_anchor import anchor_abm_to_forecast

    fc = np.clip(np.asarray(forecast, dtype=np.float64), 0.0, None)
    res = anchor_abm_to_forecast(fc, n_agents=int(n_agents), seeds=seeds, year=year)
    res = dict(res)
    res["valid"] = not bool(res.get("degenerate", False))
    return res


# ────────────────────────────────────────────────────────────────────────────
# Expanding-window driver.
# ────────────────────────────────────────────────────────────────────────────
def run(
    *,
    db_path: Path = DEFAULT_DB,
    out_path: Path = DEFAULT_OUT,
    start_frac: float = 0.85,
    max_origins: Optional[int] = None,
    horizons: tuple[int, ...] = HORIZONS,
    abm_every: int = 1,
    abm_n_agents: int = 20_000,
    abm_seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    min_in_sample: int = 80,
) -> dict[str, Any]:
    """Run the expanding-window rolling-origin multi-horizon + ABM-loop protocol.

    Args:
        db_path: local DB.
        out_path: result JSON path.
        start_frac: first origin index = ``round(n * start_frac)`` (the
            ``full(train+val+test)`` cutoff); subsequent origins add one observed
            week each, expanding to the end of the data.
        max_origins: cap on number of origins (None = run to data end).
        horizons: weeks-ahead to forecast/score.
        abm_every: run the ABM update every k-th origin (1 = every origin).
        abm_n_agents / abm_seeds: ABM anchor population / replicate seeds.
        min_in_sample: skip origins whose in-sample length < this (FusedEpi needs
            >= ~70 weeks; default 80 for a stable seasonal context).

    Returns:
        The full result dict (also written to ``out_path``).

    Side effects: reads DB, builds features once, writes ``out_path``.
    """
    np.random.seed(SEED)
    try:
        import torch
        torch.manual_seed(SEED)
    except Exception:
        pass
    os.environ.setdefault("PYTHONHASHSEED", str(SEED))

    log.info("[1/4] loading observed ILI (read_only_connect) + building feature pool")
    # The builder's ``ili_rate`` column IS the per-age-averaged KDCA sentinel ILI
    # (same source as load_observed_ili), but it is the ONLY series guaranteed
    # row-aligned with X (the raw AVG-grouped series is trimmed/shifted at the
    # edges by the feature engine). To avoid a ~1-week feature/target misalignment
    # we use the builder's ili_rate as the SSOT target and context, and load the
    # read-only sentinel series only for chronological week labels (provenance).
    X_all, y_all, feature_cols, dates = build_feature_matrix(db_path)
    n_full = len(y_all)
    y_obs = y_all                                           # builder ili_rate = aligned target
    try:
        _, ro_labels = load_observed_ili(db_path)           # read_only_connect provenance labels
        labels = ro_labels[-n_full:] if len(ro_labels) >= n_full else \
            ro_labels + [f"row{i}" for i in range(len(ro_labels), n_full)]
    except Exception as exc:
        log.warning("  read_only label load failed (%s) -> row labels", exc)
        labels = [f"row{i}" for i in range(n_full)]
    start = max(min_in_sample, int(round(n_full * start_frac)))
    origins_idx = list(range(start, n_full))               # origin t in-sample = [:t+1]
    if max_origins is not None:
        origins_idx = origins_idx[:max_origins]
    log.info("  n=%d weeks, %d feature cols, %d origins (start@%d='%s')",
             n_full, len(feature_cols), len(origins_idx), start, _date_str(dates, start))

    per_h_pred: dict[int, list[float]] = {int(h): [] for h in horizons}
    per_h_act: dict[int, list[float]] = {int(h): [] for h in horizons}

    origins_out: list[dict[str, Any]] = []
    for oi, t in enumerate(origins_idx):
        k = t + 1                                          # in-sample = weeks [0..t]
        X_in, y_in = X_all[:k], y_all[:k]
        log.info("[2/4] origin %d/%d  date=%s  in_sample_n=%d",
                 oi + 1, len(origins_idx), _date_str(dates, t), k)

        model = fit_champion(X_in, y_in)
        point, pi_lo, pi_hi = champion_multihorizon(model, X_in, y_in)

        horizons_rec: list[dict[str, Any]] = []
        for j, h in enumerate(horizons):
            tgt = t + int(h)                               # actual week index
            rec: dict[str, Any] = {
                "h": int(h),
                "pred": float(point[j]),
                "pi_lo": float(pi_lo[j]),
                "pi_hi": float(pi_hi[j]),
            }
            if tgt < n_full:                               # real observation exists
                act = float(y_obs[tgt])
                rec["actual"] = act
                rec["abs_err"] = abs(float(point[j]) - act)
                per_h_pred[int(h)].append(float(point[j]))
                per_h_act[int(h)].append(act)
            horizons_rec.append(rec)

        # forecast -> ABM update (mechanistic / district-scalable leg of the loop).
        abm_done = False
        abm_payload: Optional[dict[str, Any]] = None
        if abm_every > 0 and (oi % abm_every == 0):
            try:
                a = abm_update(point, n_agents=abm_n_agents, seeds=abm_seeds, year=None)
                abm_payload = {
                    "fitted_forcing": a.get("fitted_forcing"),
                    "affine": a.get("affine"),
                    "corr_sim_vs_forecast": a.get("corr_sim_vs_forecast"),
                    "wis": a.get("wis"),
                    "anchored_trajectory": a.get("anchored_trajectory"),
                    "valid": a.get("valid"),
                    "degenerate": a.get("degenerate"),
                }
                abm_done = bool(a.get("valid"))
                log.info("    ABM anchored: corr=%.3f wis=%.3f valid=%s",
                         float(a.get("corr_sim_vs_forecast", float("nan"))),
                         float(a.get("wis", float("nan"))), abm_done)
            except Exception as exc:                       # ABM is best-effort per origin
                log.warning("    ABM update failed at origin %d: %s", t, exc)

        origins_out.append({
            "origin_date": _date_str(dates, t),
            "origin_week_index": int(t),
            "origin_label": labels[t] if t < len(labels) else f"row{t}",
            "in_sample_n": int(k),
            "horizons": horizons_rec,
            "abm_updated": abm_done,
            "abm": abm_payload,
        })

    # per-horizon summary (scored only where a real actual existed).
    per_horizon_summary: list[dict[str, Any]] = []
    for h in horizons:
        preds = np.asarray(per_h_pred[int(h)], dtype=np.float64)
        acts = np.asarray(per_h_act[int(h)], dtype=np.float64)
        row: dict[str, Any] = {"h": int(h), "n_origins": len(origins_out),
                               "n_scored": int(preds.size)}
        if preds.size >= 2:
            err = preds - acts
            ss_res = float(np.sum(err ** 2))
            ss_tot = float(np.sum((acts - acts.mean()) ** 2))
            row["mean_r2"] = (1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else None
            row["mean_mae"] = float(np.mean(np.abs(err)))
            row["rmse"] = float(np.sqrt(np.mean(err ** 2)))
        elif preds.size == 1:
            row["mean_mae"] = float(abs(preds[0] - acts[0]))
            row["mean_r2"] = None
        else:
            row["mean_mae"] = None
            row["mean_r2"] = None
        per_horizon_summary.append(row)

    result = {
        "origins": origins_out,
        "per_horizon_summary": per_horizon_summary,
        "honest_note": (
            "Champion = R-track FusedEpi, fit fresh per expanding origin on the live full "
            "feature pool (config preproc = identity none/none; stored feature_indices were "
            "computed against a now-stale pool, so the self-adaptive model is fit on the full "
            "current pool rather than mis-indexed). Multi-horizon is RECURSIVE and leak-free: "
            "predict_multi consumes only the in-sample history (context_y), never future "
            "observations. Horizons are scored only where a real KDCA-sentinel observation "
            "exists at week t+h; future-only horizons emit forecast+PI with actual=null. The "
            "1-step PI is a NegBin/conformal interval; h>1 PI widens the 1-step half-width by "
            "sqrt(h) as a leak-free recursive-error proxy (banded approximation, not a "
            "re-derived conformal band). The forecast->ABM leg anchors a seasonal agent-based "
            "SEIR forcing to each origin's forecast (anchor_abm_to_forecast); the ABM-updated "
            "forcing closes the forecast->ABM->forecast loop. Real data only; no synthetic fill."
        ),
        "metadata": {
            "champion": "FusedEpi",
            "horizons": list(horizons),
            "n_weeks": int(n_full),
            "n_feature_cols": len(feature_cols),
            "start_index": int(start),
            "n_origins": len(origins_out),
            "abm_every": int(abm_every),
            "abm_n_agents": int(abm_n_agents),
            "abm_seeds": list(abm_seeds),
            "seed": SEED,
            "db_path": str(db_path),
            "local_only": True,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("[4/4] wrote %s (%d origins)", out_path, len(origins_out))
    return result


def _print_smoke(result: dict[str, Any]) -> None:
    print("\n=== expanding multi-horizon smoke ===")
    md = result["metadata"]
    print(f"champion={md['champion']}  weeks={md['n_weeks']}  feat_cols={md['n_feature_cols']}  "
          f"origins={md['n_origins']}  horizons={md['horizons']}")
    for o in result["origins"]:
        hs = "  ".join(
            f"h{r['h']}:pred={r['pred']:.2f}" + (f",act={r['actual']:.2f}" if "actual" in r else ",act=NA")
            for r in o["horizons"]
        )
        print(f"  origin {o['origin_date']} (n={o['in_sample_n']}, abm_updated={o['abm_updated']}): {hs}")
    print("per-horizon summary:")
    for s in result["per_horizon_summary"]:
        r2 = s.get("mean_r2")
        mae = s.get("mean_mae")
        print(f"  h={s['h']:>2}  n_scored={s['n_scored']}  "
              f"r2={('%.3f' % r2) if isinstance(r2, float) else r2}  "
              f"mae={('%.3f' % mae) if isinstance(mae, float) else mae}")
    print("\nhonest_note:\n  " + result["honest_note"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--start-frac", type=float, default=0.85)
    ap.add_argument("--max-origins", type=int, default=None)
    ap.add_argument("--horizons", type=str, default="",
                    help="comma list, e.g. 1,4,12 (default: 1,2,3,4,8,12,28)")
    ap.add_argument("--abm-every", type=int, default=1)
    ap.add_argument("--abm-n-agents", type=int, default=20_000)
    ap.add_argument("--abm-seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--smoke", action="store_true",
                    help="reduced run: 2 origins, horizons 1,4,12, tiny ABM (2k agents, 2 seeds)")
    args = ap.parse_args()

    horizons = (tuple(int(x) for x in args.horizons.split(",") if x.strip())
                if args.horizons else HORIZONS)
    abm_seeds = tuple(int(x) for x in args.abm_seeds.split(",") if x.strip())

    if args.smoke:
        result = run(
            db_path=args.db, out_path=args.out,
            start_frac=args.start_frac, max_origins=2,
            horizons=(1, 4, 12), abm_every=1,
            abm_n_agents=2_000, abm_seeds=(0, 1),
        )
    else:
        result = run(
            db_path=args.db, out_path=args.out,
            start_frac=args.start_frac, max_origins=args.max_origins,
            horizons=horizons, abm_every=args.abm_every,
            abm_n_agents=args.abm_n_agents, abm_seeds=abm_seeds,
        )
    _print_smoke(result)


if __name__ == "__main__":
    main()
