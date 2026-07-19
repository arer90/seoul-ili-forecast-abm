"""
generate_mc_comparison_csv.py
==============================
Compare multicollinearity method results (none/vif/corr/pca) across all models.

Reads per_model_optimal_METHOD_* directories (R9 per_model_optimize / Phase B outputs),
loads test_metrics from each JSON, computes additional metrics where possible,
and writes two CSVs:

  simulation/results/mc_comparison_metrics.csv     -- one row per (model, method)
  simulation/results/mc_comparison_predictions.csv -- one row per (model, method, week_idx)

Usage:
  python -m simulation.scripts.generate_mc_comparison_csv
  python -m simulation.scripts.generate_mc_comparison_csv --output-dir <path>
  python simulation/scripts/generate_mc_comparison_csv.py

--output-dir (Codex audit 2026-05-26 P1 fix): overrides default
simulation/results/ destination so Pov overseas (phase18_overseas.py) can route
outputs into its own out_dir. Both CSVs land in the same dir.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from simulation.database import safe_connect

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent  # MPH_infection_simulation/
_SIM_ROOT = _PROJECT_ROOT / "simulation"
_RESULTS_ROOT = _SIM_ROOT / "results"
_DB_PATH = _SIM_ROOT / "data" / "db" / "epi_real_seoul.db"
# Default destinations — overridable via --output-dir CLI arg.
_DEFAULT_OUTPUT_METRICS = _RESULTS_ROOT / "mc_comparison_metrics.csv"
_DEFAULT_OUTPUT_PREDS = _RESULTS_ROOT / "mc_comparison_predictions.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

TEST_SLAB_N = 68  # canonical test window length

# ---------------------------------------------------------------------------
# 1. Directory discovery
# ---------------------------------------------------------------------------

def _discover_method_dirs() -> dict[str, Path]:
    """Return {method_label: most_recent_dir} for all known mc methods.

    Priority order (highest first):
      - per_model_optimal_<method>_*/  (R9 per_model_optimize method runs, latest timestamp)
      - per_model_optimal_b_<method>_*/  (Phase B runs)
      - per_model_optimal/  (base dir → "none" fallback)
    """
    methods = ["none", "vif", "corr", "pca"]
    result: dict[str, Path] = {}

    for method in methods:
        candidates: list[Path] = []

        # R9 per_model_optimize style: per_model_optimal_<method>_YYYYMMDD[_*]
        candidates += sorted(_RESULTS_ROOT.glob(f"per_model_optimal_{method}_*/"))
        # Phase B style: per_model_optimal_b_<method>_YYYYMMDD[_*]
        candidates += sorted(_RESULTS_ROOT.glob(f"per_model_optimal_b_{method}_*/"))

        # Filter to directories that actually contain at least one JSON
        valid = [p for p in candidates if p.is_dir() and list(p.glob("*.json"))]

        if valid:
            # Take the lexicographically latest (timestamp suffix sorts naturally)
            chosen = sorted(valid)[-1]
            result[method] = chosen
            log.info("method=%-6s  dir=%s", method, chosen.name)
        elif method == "none":
            # Fallback to base per_model_optimal/ for "none"
            base = _RESULTS_ROOT / "per_model_optimal"
            if base.is_dir() and list(base.glob("*.json")):
                result[method] = base
                log.info("method=none  dir=per_model_optimal (fallback)")
            else:
                log.warning("method=none: no directory found")
        else:
            log.warning("method=%-6s: no directory found — skipped", method)

    return result


# ---------------------------------------------------------------------------
# 2. y_true from DB (last 68 weeks, average across all age groups)
# ---------------------------------------------------------------------------

def _load_y_true() -> np.ndarray:
    """Load last 68 weeks of ILI rate from sentinel_influenza.

    Returns a float64 array of shape (68,) ordered chronologically.
    Averages across all age groups for each (season_start, week_seq).
    Falls back to all age groups if '전체' does not exist.
    """
    if not _DB_PATH.exists():
        raise FileNotFoundError(f"DB not found: {_DB_PATH}")

    conn = safe_connect(str(_DB_PATH))
    try:
        cur = conn.cursor()

        # Check whether a '전체' (national total) row exists
        cur.execute(
            "SELECT COUNT(*) FROM sentinel_influenza WHERE age_group = '전체'"
        )
        total_count = cur.fetchone()[0]

        if total_count > 0:
            age_filter = "WHERE age_group = '전체'"
        else:
            age_filter = ""  # average across all age groups

        cur.execute(f"""
            SELECT season_start, week_seq, AVG(ili_rate) AS avg_ili
            FROM sentinel_influenza
            {age_filter}
            GROUP BY season_start, week_seq
            ORDER BY season_start, week_seq
        """)
        rows = cur.fetchall()
    finally:
        conn.close()

    if len(rows) < TEST_SLAB_N:
        raise ValueError(
            f"DB has only {len(rows)} weeks; need at least {TEST_SLAB_N}"
        )

    y_true = np.array([r[2] for r in rows[-TEST_SLAB_N:]], dtype=np.float64)
    log.info(
        "y_true loaded: n=%d  min=%.2f  max=%.2f  mean=%.2f",
        len(y_true), float(np.nanmin(y_true)),
        float(np.nanmax(y_true)), float(np.nanmean(y_true)),
    )
    return y_true


def _load_y_train() -> np.ndarray:
    """Load training pool ILI (all weeks except last 68) for MASE denominator."""
    conn = safe_connect(str(_DB_PATH))
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM sentinel_influenza WHERE age_group = '전체'"
        )
        age_filter = (
            "WHERE age_group = '전체'"
            if cur.fetchone()[0] > 0
            else ""
        )
        cur.execute(f"""
            SELECT season_start, week_seq, AVG(ili_rate) AS avg_ili
            FROM sentinel_influenza
            {age_filter}
            GROUP BY season_start, week_seq
            ORDER BY season_start, week_seq
        """)
        rows = cur.fetchall()
    finally:
        conn.close()

    y_all = np.array([r[2] for r in rows], dtype=np.float64)
    return y_all[:-TEST_SLAB_N]  # training pool only


# ---------------------------------------------------------------------------
# 3. Load all JSON results for a method directory
# ---------------------------------------------------------------------------

def _load_method_results(directory: Path) -> dict[str, dict[str, Any]]:
    """Return {model_name: json_content} for all *.json in directory.

    Ignores backup files (*.backup_*) and files without 'model' key.
    """
    results: dict[str, dict[str, Any]] = {}
    for fp in sorted(directory.glob("*.json")):
        if ".backup" in fp.name or fp.stem.startswith("_"):
            continue
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("JSON parse error %s: %s", fp.name, exc)
            continue
        model = data.get("model") or fp.stem
        if not isinstance(data.get("test_metrics"), dict):
            log.debug("Skipping %s — no test_metrics dict", fp.name)
            continue
        results[model] = data
    return results


# ---------------------------------------------------------------------------
# 4. Compute additional metrics beyond the 26 stored in test_metrics
# ---------------------------------------------------------------------------

def _safe(fn, *args, default: float = float("nan"), **kwargs) -> float:
    """Call fn(*args, **kwargs), return default on any exception."""
    try:
        val = fn(*args, **kwargs)
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            return float(val) if val is not None else default
        return float(val)
    except Exception:
        return default


def _compute_extra_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    sigma: float,
) -> dict[str, float]:
    """Compute additional metrics not in the 26-key test_metrics store.

    All computations are wrapped in try/except — never raises.
    """
    extra: dict[str, float] = {}

    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(yt) & np.isfinite(yp)

    if not mask.any():
        return extra

    yt_m = yt[mask]
    yp_m = yp[mask]
    resid = yp_m - yt_m

    # MASE at multiple horizons
    for h, key in [(4, "mase_h4"), (13, "mase_h13"), (26, "mase_h26"), (52, "mase_h52")]:
        try:
            if len(y_train) > h:
                naive = np.abs(y_train[h:] - y_train[:-h])
                naive = naive[np.isfinite(naive)]
                mae_naive = float(np.mean(naive)) if len(naive) > 0 else np.nan
                mae_fc = float(np.mean(np.abs(yp_m - yt_m)))
                extra[key] = mae_fc / mae_naive if mae_naive > 1e-10 else np.nan
            else:
                extra[key] = np.nan
        except Exception:
            extra[key] = np.nan

    # MSLE
    try:
        pos = (yt_m > 0) & (yp_m > 0)
        if pos.any():
            extra["msle"] = float(
                np.mean((np.log1p(yt_m[pos]) - np.log1p(yp_m[pos])) ** 2)
            )
        else:
            extra["msle"] = np.nan
    except Exception:
        extra["msle"] = np.nan

    # Theil's U
    try:
        if len(yt_m) > 1:
            num = float(np.sqrt(np.mean((yp_m - yt_m) ** 2)))
            denom = float(np.sqrt(np.mean(yt_m ** 2)))
            extra["theils_u"] = num / denom if denom > 1e-10 else np.nan
        else:
            extra["theils_u"] = np.nan
    except Exception:
        extra["theils_u"] = np.nan

    # Log-score (Gaussian)
    try:
        if sigma > 0 and np.isfinite(sigma):
            log_scores = (
                -0.5 * np.log(2 * np.pi * sigma ** 2)
                - 0.5 * ((yt_m - yp_m) ** 2) / sigma ** 2
            )
            extra["log_score_gauss"] = float(np.mean(log_scores))
        else:
            extra["log_score_gauss"] = np.nan
    except Exception:
        extra["log_score_gauss"] = np.nan

    # PI width metrics (from sigma, Gaussian assumption)
    try:
        from scipy.stats import norm
        for level, key_w, key_cov in [
            (0.95, "pi95_width", None),
            (0.80, "pi80_width", None),
            (0.50, "pi50_width", None),
            (0.99, "pi99_width", None),
        ]:
            z = float(norm.ppf(0.5 + level / 2))
            half = z * sigma
            lo = yp_m - half
            hi = yp_m + half
            extra[f"pi{int(level*100)}_width"] = float(2 * half)
            cov = float(np.mean((yt_m >= lo) & (yt_m <= hi)))
            extra[f"pi{int(level*100)}_coverage_computed"] = cov
    except Exception:
        pass

    # Residual autocorrelation lag-1
    try:
        if len(resid) > 2:
            r = np.corrcoef(resid[:-1], resid[1:])[0, 1]
            extra["residual_acf_lag1"] = float(r) if np.isfinite(r) else np.nan
        else:
            extra["residual_acf_lag1"] = np.nan
    except Exception:
        extra["residual_acf_lag1"] = np.nan

    # Shapiro-Wilk on residuals
    try:
        from scipy.stats import shapiro
        if 3 <= len(resid) <= 5000:
            _, p = shapiro(resid)
            extra["shapiro_wilk_p"] = float(p)
        else:
            extra["shapiro_wilk_p"] = np.nan
    except Exception:
        extra["shapiro_wilk_p"] = np.nan

    # Ljung-Box Q at lag 10
    try:
        from scipy.stats import chi2
        n = len(resid)
        max_lag = min(10, n // 2 - 1)
        if max_lag >= 1:
            acf_vals = []
            for lag in range(1, max_lag + 1):
                c = np.corrcoef(resid[:-lag], resid[lag:])[0, 1]
                acf_vals.append(float(c) if np.isfinite(c) else 0.0)
            q = float(
                n * (n + 2) * sum(
                    acf_vals[i] ** 2 / (n - i - 1)
                    for i in range(max_lag)
                )
            )
            p = 1.0 - float(chi2.cdf(q, df=max_lag))
            extra["ljung_box_q"] = q
            extra["ljung_box_p"] = p
        else:
            extra["ljung_box_q"] = np.nan
            extra["ljung_box_p"] = np.nan
    except Exception:
        extra["ljung_box_q"] = np.nan
        extra["ljung_box_p"] = np.nan

    # Alert / clinical metrics (threshold = 15.0 ILI rate, WHO ILI standard)
    try:
        threshold = 15.0
        y_alert = (yt_m >= threshold).astype(int)
        pred_alert = (yp_m >= threshold).astype(int)
        tp = int(np.sum((pred_alert == 1) & (y_alert == 1)))
        fp = int(np.sum((pred_alert == 1) & (y_alert == 0)))
        fn = int(np.sum((pred_alert == 0) & (y_alert == 1)))
        tn = int(np.sum((pred_alert == 0) & (y_alert == 0)))
        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
        f1_num = 2 * tp
        f1_den = 2 * tp + fp + fn
        f1 = f1_num / f1_den if f1_den > 0 else np.nan
        extra["alert_threshold"] = threshold
        extra["sensitivity"] = float(sens) if np.isfinite(sens) else np.nan
        extra["specificity"] = float(spec) if np.isfinite(spec) else np.nan
        extra["ppv"] = float(ppv) if np.isfinite(ppv) else np.nan
        extra["npv"] = float(npv) if np.isfinite(npv) else np.nan
        extra["alert_f1"] = float(f1) if np.isfinite(f1) else np.nan
        # MCC
        denom_mcc = (
            (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        ) ** 0.5
        extra["mcc"] = (
            (tp * tn - fp * fn) / denom_mcc if denom_mcc > 0 else np.nan
        )
        # Brier score
        extra["brier_score"] = float(
            np.mean((pred_alert - y_alert) ** 2)
        )
        # Brier skill (vs climatological prevalence)
        clim = float(np.mean(y_alert))
        bs_ref = float(np.mean((clim - y_alert) ** 2))
        extra["brier_skill"] = (
            1.0 - extra["brier_score"] / bs_ref if bs_ref > 0 else np.nan
        )
    except Exception:
        pass

    # Growth rate correlation
    try:
        if len(yt_m) > 1:
            gr_true = np.diff(yt_m) / np.where(yt_m[:-1] > 0.01, yt_m[:-1], 0.01)
            gr_pred = np.diff(yp_m) / np.where(yp_m[:-1] > 0.01, yp_m[:-1], 0.01)
            fmask = np.isfinite(gr_true) & np.isfinite(gr_pred)
            if fmask.sum() > 2:
                r = np.corrcoef(gr_true[fmask], gr_pred[fmask])[0, 1]
                extra["growth_rate_corr"] = float(r) if np.isfinite(r) else np.nan
            else:
                extra["growth_rate_corr"] = np.nan
        else:
            extra["growth_rate_corr"] = np.nan
    except Exception:
        extra["growth_rate_corr"] = np.nan

    # Attack rate relative error
    try:
        ar_true = float(np.sum(yt_m))
        ar_pred = float(np.sum(yp_m))
        extra["attack_rate_relerr"] = (
            abs(ar_pred - ar_true) / ar_true if ar_true > 0 else np.nan
        )
    except Exception:
        extra["attack_rate_relerr"] = np.nan

    return extra


# ---------------------------------------------------------------------------
# 5. Build records for a single (model, method) pair
# ---------------------------------------------------------------------------

def _build_row(
    model: str,
    method: str,
    source_dir: str,
    data: dict[str, Any],
    y_true: np.ndarray,
    y_train: np.ndarray,
) -> dict[str, Any]:
    """Return a flat metrics dict for one (model, method) pair."""
    row: dict[str, Any] = {
        "model": model,
        "method": method,
        "source_dir": source_dir,
    }

    test_metrics: dict = data.get("test_metrics", {})
    row.update(test_metrics)

    # Predictions
    preds_raw = data.get("refit_test_predictions", [])
    if not isinstance(preds_raw, list) or len(preds_raw) == 0:
        # Try best_metrics path
        preds_raw = data.get("best_metrics", {}).get("refit_test_predictions", [])

    y_pred = (
        np.array(preds_raw, dtype=np.float64)
        if len(preds_raw) > 0
        else np.full(TEST_SLAB_N, np.nan)
    )
    if len(y_pred) != TEST_SLAB_N:
        log.debug(
            "%s/%s predictions length %d != %d — padding/truncating",
            model, method, len(y_pred), TEST_SLAB_N,
        )
        if len(y_pred) < TEST_SLAB_N:
            y_pred = np.pad(
                y_pred, (0, TEST_SLAB_N - len(y_pred)), constant_values=np.nan
            )
        else:
            y_pred = y_pred[:TEST_SLAB_N]

    sigma = float(test_metrics.get("sigma_in_sample", 0.0) or 0.0)

    # Compute extra metrics
    extra = _compute_extra_metrics(y_true, y_pred, y_train, sigma)
    # Only add extra keys not already present from test_metrics
    for k, v in extra.items():
        if k not in row:
            row[k] = v

    return row, y_pred


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args. --output-dir is the only configurable knob (Codex P1 fix).

    Args:
        argv: optional argv override (test harness). Default = sys.argv[1:].

    Returns:
        Namespace with `output_dir: Path | None`.
    """
    p = argparse.ArgumentParser(
        description="Compare multicollinearity method results across models",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help=("CSV 출력 디렉토리. 미지정 시 simulation/results/ 사용 (legacy default). "
              "Pov overseas (phase18_overseas) 호출 시 out_dir 전달용."),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.output_dir is not None:
        out_dir = args.output_dir
        output_metrics = out_dir / "mc_comparison_metrics.csv"
        output_preds = out_dir / "mc_comparison_predictions.csv"
    else:
        out_dir = _RESULTS_ROOT
        output_metrics = _DEFAULT_OUTPUT_METRICS
        output_preds = _DEFAULT_OUTPUT_PREDS

    log.info("=== generate_mc_comparison_csv ===")
    log.info("project_root=%s", _PROJECT_ROOT)
    log.info("output_dir=%s", out_dir)

    # Discover directories
    method_dirs = _discover_method_dirs()
    if not method_dirs:
        log.error("No per_model_optimal_* directories found. Exiting.")
        sys.exit(1)

    # Load y_true and y_train
    y_true = _load_y_true()
    y_train = _load_y_train()

    metrics_rows: list[dict] = []
    pred_rows: list[dict] = []

    for method, directory in sorted(method_dirs.items()):
        model_data = _load_method_results(directory)
        log.info("method=%-6s  models=%d  dir=%s", method, len(model_data), directory.name)

        for model, data in sorted(model_data.items()):
            try:
                row, y_pred = _build_row(
                    model=model,
                    method=method,
                    source_dir=directory.name,
                    data=data,
                    y_true=y_true,
                    y_train=y_train,
                )
            except Exception as exc:
                log.warning("Error building row for %s/%s: %s", model, method, exc)
                continue

            metrics_rows.append(row)

            # Prediction rows
            for i in range(TEST_SLAB_N):
                pred_rows.append({
                    "model": model,
                    "method": method,
                    "week_idx": i,
                    "y_true": float(y_true[i]) if i < len(y_true) else np.nan,
                    "y_pred": float(y_pred[i]),
                })

    if not metrics_rows:
        log.error("No rows collected — check directory contents.")
        sys.exit(1)

    # Write CSVs
    out_dir.mkdir(parents=True, exist_ok=True)

    df_metrics = pd.DataFrame(metrics_rows)
    # Ensure consistent column order: identity cols first, then alphabetical metrics
    id_cols = ["model", "method", "source_dir"]
    other_cols = sorted(c for c in df_metrics.columns if c not in id_cols)
    df_metrics = df_metrics[id_cols + other_cols]
    df_metrics.to_csv(output_metrics, index=False)
    log.info("Wrote metrics CSV: %s  (%d rows x %d cols)", output_metrics, len(df_metrics), len(df_metrics.columns))

    df_preds = pd.DataFrame(pred_rows)
    df_preds.to_csv(output_preds, index=False)
    log.info("Wrote predictions CSV: %s  (%d rows)", output_preds, len(df_preds))

    # Summary table
    _print_summary(df_metrics, output_metrics, output_preds)


def _print_summary(df: pd.DataFrame, output_metrics: Path, output_preds: Path) -> None:
    """Print a compact summary table to stdout.

    Args:
        df: metrics DataFrame.
        output_metrics: resolved output path for metrics CSV (passed in to avoid
            stale module-level references — Gemini audit 2026-05-26 P1 fix).
        output_preds: resolved output path for predictions CSV.
    """
    print("\n" + "=" * 72)
    print("  MC COMPARISON SUMMARY")
    print("=" * 72)

    key_metrics = ["r2", "mae", "rmse", "mape", "wis", "pi95_coverage"]
    available = [m for m in key_metrics if m in df.columns]

    for method in sorted(df["method"].unique()):
        sub = df[df["method"] == method]
        print(f"\nMethod: {method}  (n_models={len(sub)})")
        header = f"  {'model':<30}" + "".join(f" {m:>14}" for m in available)
        print(header)
        print("  " + "-" * (30 + 15 * len(available)))
        for _, row in sub.sort_values("model").iterrows():
            vals = "".join(
                f" {row[m]:>14.4f}" if pd.notna(row.get(m)) else f" {'--':>14}"
                for m in available
            )
            print(f"  {row['model']:<30}{vals}")

    print("\n" + "=" * 72)
    print(f"  Output files:")
    print(f"    {output_metrics}")
    print(f"    {output_preds}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
