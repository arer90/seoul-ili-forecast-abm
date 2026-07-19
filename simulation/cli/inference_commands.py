"""Pinf inference CLI commands — extracted from __main__.py.

Phase C2 partial (2026-05-12 cont.): cmd_predict_real moved here.
Heavy handler (~263 lines) — uses ChampionLog + ChampionArtifact +
build_enriched_features + Pinf inference pipeline.
"""
from __future__ import annotations

import logging
import sys


log = logging.getLogger(__name__)


def cmd_predict_real(args) -> None:
    """`python -m simulation predict-real` — Pinf inference using champions.

    Builds the same feature matrix the training pipeline uses, slices the
    requested window, then for every champion in ``models/champion_log.json``
    (or the ``--models`` subset) loads the ChampionArtifact (model + fitted
    scaler + transform state + feature_indices) and replays the full
    training-time pipeline on the inference window. Writes:

      simulation/results/inference/<ts>/
        ├── predictions.csv
        ├── inference_metrics.json   (if actual ILI rate available)
        ├── champions_used.json
        └── REPORT.md
    """
    from pathlib import Path as _PPath

    models_dir = _PPath(getattr(args, "models_dir", None) or "models")

    # ── --list-champions: just print the log and exit ──
    if getattr(args, "list_champions", False):
        from simulation.utils.champion_log import ChampionLog
        cl = ChampionLog(models_dir=models_dir,
                         log_path=models_dir / "champion_log.json")
        sm = cl.summary()
        if not sm:
            print("(no champions yet — run `simulation train --per-model-optimize` "
                  "to populate models/champion_log.json)")
            return
        # Tier-grouped output
        by_tier: dict = {"paper": [], "extra": [], "negative": [], "unknown": []}
        for nm, s in sm.items():
            by_tier.setdefault(s.get("tier", "unknown"), []).append((nm, s))

        print(f"\n{'Tier':<8} {'Model':<25} {'v#':<4} {'test_WIS':<10} "
              f"{'test_MAE':<10} {'test_R²':<8}  promoted_at")
        print("-" * 100)
        for tier in ("paper", "extra", "negative", "unknown"):
            tier_glyph = {"paper": "⭐ paper",
                          "extra": "  extra",
                          "negative": "✗ neg",
                          "unknown": "? unkn"}.get(tier, "?")
            for nm, s in sorted(by_tier.get(tier, [])):
                wis = s.get("current_test_wis")
                mae = s.get("current_test_mae")
                r2 = s.get("current_test_r2")
                wis_s = f"{wis:.3f}" if isinstance(wis, (int, float)) else "?"
                mae_s = f"{mae:.3f}" if isinstance(mae, (int, float)) else "?"
                r2_s = f"{r2:.3f}" if isinstance(r2, (int, float)) else "?"
                print(f"{tier_glyph:<8} {nm:<25} {s.get('current_version', '?'):<4} "
                      f"{wis_s:<10} {mae_s:<10} {r2_s:<8}  "
                      f"{s.get('promoted_at', '?')}")
        # Tier counts
        n_paper = len(by_tier["paper"])
        n_extra = len(by_tier["extra"])
        n_neg = len(by_tier["negative"])
        n_unk = len(by_tier["unknown"])
        from simulation.models.registry import EXTRA_MODELS_FLAT, PAPER_PRIMARY_11
        print(f"\nTotal: {len(sm)} champion(s) "
              f"= ⭐ {n_paper}/{len(PAPER_PRIMARY_11)} paper "
              f"+ {n_extra}/{len(EXTRA_MODELS_FLAT)} extra "
              f"+ {n_neg} neg + {n_unk} unknown")
        return

    # ── Build feature matrix (mirrors phase1_data.py sanitization) ──
    log.info("[predict-real] building enriched feature matrix …")
    import numpy as np
    import pandas as pd
    import polars as pl

    from simulation.database.config import DB_PATH
    from simulation.models.feature_engine import build_enriched_features

    db_path = str(DB_PATH)
    feat_df, meta = build_enriched_features(db_path=db_path)
    target_col = meta.get("target_col", "ili_rate")
    dates_arr = meta.get("dates")  # may be None if weather not joined

    # Drop non-numeric columns the same way phase1_data does — some realtime
    # loaders leak string-typed enums (예: '여유' / '보통' / '혼잡' for road
    # congestion) which break to_numpy(dtype=float64).
    numeric_dtypes = (
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64, pl.Boolean,
    )
    schema = feat_df.schema
    feature_cols = [c for c in feat_df.columns
                    if c != target_col and schema[c] in numeric_dtypes]
    dropped_str = [c for c in feat_df.columns
                   if c != target_col and schema[c] not in numeric_dtypes]
    if dropped_str:
        log.info(f"[predict-real] dropped {len(dropped_str)} non-numeric "
                 f"column(s): {dropped_str[:5]}"
                 f"{'…' if len(dropped_str) > 5 else ''}")

    X_full = feat_df.select(feature_cols).to_numpy().astype(np.float64)
    y_full = (feat_df[target_col].to_numpy().astype(np.float64)
              if target_col in feat_df.columns else None)

    # ── Mirror R1 (data)'s COVID-mode indicator augmentation ──
    # If any champion was trained with --covid-mode indicator, X has an
    # extra `covid_era_indicator` column (320→321). Inference X must match.
    # Detect by comparing X_full's column count to the first artifact's
    # expected feature count; pad with zeros (current/future weeks are
    # always outside the 2020-03..2022-12 COVID era → indicator=0).
    try:
        from simulation.utils.champion_log import ChampionLog
        from simulation.utils.model_artifact import load_artifact
        cl_probe = ChampionLog(models_dir=models_dir,
                               log_path=models_dir / "champion_log.json")
        sm_probe = cl_probe.summary()
        # find max expected feature count across all champions
        expected_total = 0
        for nm in sm_probe:
            art = load_artifact(models_dir / f"{nm}.pt")
            if art is None or art.scaler is None:
                continue
            try:
                exp = int(art.scaler.n_features_in_)
            except Exception:
                exp = 0
            # If artifact uses feature_indices, the scaler size IS the
            # subset size, but indices are into the FULL X. We need the
            # max index + 1 to know the full X width that was used.
            if art.feature_indices:
                exp = max(exp, max(art.feature_indices) + 1)
            expected_total = max(expected_total, exp)
        cur_total = X_full.shape[1]
        if expected_total > cur_total:
            pad_cols = expected_total - cur_total
            X_full = np.hstack([X_full, np.zeros((len(X_full), pad_cols),
                                                 dtype=np.float64)])
            feature_cols = list(feature_cols) + (
                ["covid_era_indicator"] if pad_cols == 1
                else [f"_pad_{i}" for i in range(pad_cols)])
            log.info(f"[predict-real] padded X with {pad_cols} zero col(s) "
                     f"(expected {expected_total}, got {cur_total}) — "
                     f"covid_era_indicator=0 for current/future weeks")
    except Exception as _pe:
        log.debug(f"[predict-real] padding probe skipped: {_pe}")

    # ── Pick the inference window ──
    # Priority: --weeks-ahead > (--start-date, --end-date) > "last test slab"
    n = len(X_full)
    if dates_arr is not None and len(dates_arr) == n:
        dates_pd = pd.to_datetime(pd.Series(dates_arr))
    else:
        dates_pd = pd.Series([pd.NaT] * n)

    weeks_ahead = getattr(args, "weeks_ahead", None)
    start_date = getattr(args, "start_date", None)
    end_date = getattr(args, "end_date", None)

    if weeks_ahead and weeks_ahead > 0:
        i0 = max(0, n - int(weeks_ahead))
        i1 = n
        window_label = f"last {weeks_ahead} weeks"
    elif start_date or end_date:
        if dates_arr is None:
            log.error("[predict-real] feature matrix has no dates — "
                      "cannot honour --start-date / --end-date "
                      "(use --weeks-ahead instead)")
            sys.exit(2)
        s = pd.to_datetime(start_date) if start_date else dates_pd.min()
        e = pd.to_datetime(end_date) if end_date else dates_pd.max()
        mask = (dates_pd >= s) & (dates_pd <= e)
        if not mask.any():
            log.error(f"[predict-real] no rows in window [{s} .. {e}]")
            sys.exit(2)
        i0 = int(mask.idxmax())
        i1 = int((mask[::-1].idxmax())) + 1
        window_label = f"{s.date()} → {e.date()}"
    else:
        # Default: last 68 weeks (= HWP §3 test slab) + any post-cutoff real
        try:
            from simulation.pipeline.config import SplitConfig
            cfg_split = SplitConfig()
            tail = (cfg_split.in_sample_test_ratio
                    * cfg_split.paper_cutoff_week if cfg_split.paper_cutoff_week
                    else 68)
            tail = int(round(tail)) or 68
        except Exception:
            tail = 68
        i0 = max(0, n - tail)
        i1 = n
        window_label = f"default tail (last {i1 - i0} weeks)"

    X_inf = X_full[i0:i1]
    y_inf = y_full[i0:i1] if y_full is not None else None
    dates_inf = (dates_arr[i0:i1] if dates_arr is not None else None)
    log.info(f"[predict-real] window: {window_label} "
             f"(rows {i0}:{i1}, n={i1 - i0})")

    # ── Resolve --models filter ──
    raw_models = getattr(args, "models", None)
    if raw_models:
        wanted = [m.strip() for m in str(raw_models).split(",") if m.strip()]
    else:
        wanted = None

    # ── Run Pinf inference ──
    out_dir = (_PPath(args.out_dir) if getattr(args, "out_dir", None)
               else None)
    actuals = y_inf if getattr(args, "with_actuals", False) else None

    from simulation.pipeline.inference import run_inference
    res = run_inference(
        X_inference=X_inf,
        inference_dates=dates_inf,
        actuals=actuals,
        model_names=wanted,
        models_dir=models_dir,
        log_path=models_dir / "champion_log.json",
        out_dir=out_dir,
    )

    if res.get("skipped"):
        log.warning(f"[predict-real] skipped: {res.get('reason', '?')}")
        sys.exit(1)

    log.info(f"[predict-real] done in {res.get('elapsed', 0):.1f}s")
    log.info(f"[predict-real] report: {res.get('report_path')}")
    if res.get("metrics_per_model"):
        # Aggregate (all horizons) — keep for back-compat
        print("\n──── aggregate metrics (all horizons combined) ────")
        for nm, m in sorted(res["metrics_per_model"].items(),
                            key=lambda kv: kv[1].get("wis", float("inf"))):
            print(f"  {nm:<22}  WIS={m.get('wis', float('nan')):>7.3f}  "
                  f"MAE={m.get('mae', float('nan')):>7.3f}  "
                  f"R²={m.get('r2', float('nan')):>+6.2f}  "
                  f"n={m.get('n')}")
        # Per-horizon — h=1 is the operationally critical one
        first_per_h = next(iter(res["metrics_per_model"].values())).get(
            "per_horizon", {})
        if first_per_h:
            horizons = sorted(first_per_h.keys(),
                              key=lambda k: int(k.replace("h", "")))
            print(f"\n──── per-horizon absolute error (h=1 = next-week, primary KPI) ────")
            header = f"  {'model':<22} " + " ".join(f"{h:>7}" for h in horizons)
            print(header)
            print("  " + "-" * (len(header) - 2))
            # Sort by h=1 ae (operational ranking)
            def _h1_ae(model_metrics):
                return (model_metrics.get("per_horizon", {})
                        .get(horizons[0], {}).get("ae", float("inf")))
            for nm, m in sorted(res["metrics_per_model"].items(),
                                key=lambda kv: _h1_ae(kv[1])):
                ph = m.get("per_horizon", {})
                row = []
                for h in horizons:
                    ae = ph.get(h, {}).get("ae", float("nan"))
                    row.append(f"{ae:>7.2f}" if (ae == ae) else f"{'?':>7}")
                print(f"  {nm:<22} " + " ".join(row))
            # Print actuals for context
            actuals_row = [str(first_per_h.get(h, {}).get("actual", "?"))
                           for h in horizons]
            actuals_fmt = [f"{float(a):>7.2f}" if a != "?" else f"{'?':>7}"
                           for a in actuals_row]
            print(f"  {'(actual ILI rate)':<22} " + " ".join(actuals_fmt))


__all__ = [
    "cmd_predict_real",
]
