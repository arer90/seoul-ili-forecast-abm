"""
visualize — per-model figures from saved ChampionArtifact `.pt` files.
========================================================================

Without re-training, this module loads each champion `.pt` (=
`ChampionArtifact` bundle of model + scaler + transform_state +
feature_indices), reconstructs predictions across the **whole timeline**
(train / val / test / real if available), and renders:

  1. **Time-series overview per model** — actual ILI vs predicted, with
     vertical bands shaded for train (gray) / val (yellow) / test
     (orange) / real (red). Peak weeks annotated.

  2. **Combined timeseries** — every champion on one axes, sharing
     ground-truth — for visual comparison of forecast trajectories.

  3. **Residual diagnostic per model** — time series + scatter (actual
     vs predicted) + Q-Q plot of residuals.

  4. **Per-horizon error bar (real slab only)** — h=1..h=N AE per model.

  5. **Optuna learning curve** if `optuna_feature_selection.db` exists —
     trial value vs trial number, per model.

All figures written to
``simulation/results/visualizations/<ts>/`` with ``manifest.json``
listing every produced PNG. Open via OS file viewer or
``ls -la simulation/results/visualizations/<ts>/``.

CLI:
    simulation visualize                       # all champions, full timeline
    simulation visualize --models XGBoost,LightGBM
    simulation visualize --no-residuals
    simulation visualize --include-optuna      # also draw trial history
    simulation visualize --out-dir <path>
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Color scheme (consistent across all plots)
# ─────────────────────────────────────────────────────────────────
SLAB_COLORS = {
    "train": ("#cfd8dc", "Train"),       # cool gray
    "val":   ("#fff59d", "Val"),         # pale yellow
    "test":  ("#ffcc80", "Test"),        # light orange
    "real":  ("#ef9a9a", "Real"),        # pale red
}


def _setup_mpl():
    """Lazy import + non-interactive backend (for headless / CI)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 140,
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.labelsize": 9.5,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return plt


# ─────────────────────────────────────────────────────────────────
# Build the full timeline (train + val + test + real)
# ─────────────────────────────────────────────────────────────────
def _build_full_timeline(repo_root: Path) -> dict:
    """Run feature_engine + sanitize same way the R1 data phase does. Returns
    dict with X_full / y_full / dates / split_indices."""
    from simulation.models.feature_engine import build_enriched_features
    from simulation.database.config import DB_PATH
    from simulation.pipeline.config import SplitConfig
    from simulation.pipeline.data import compute_split_indices
    import polars as pl

    feat_df, meta = build_enriched_features(db_path=str(DB_PATH))
    target_col = meta.get("target_col", "ili_rate")
    dates_arr = meta.get("dates")

    schema = feat_df.schema
    num_dt = (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8,
              pl.UInt16, pl.UInt32, pl.UInt64,
              pl.Float32, pl.Float64, pl.Boolean)
    feature_cols = [c for c in feat_df.columns
                    if c != target_col and schema[c] in num_dt]
    X_full = feat_df.select(feature_cols).to_numpy().astype(np.float64)
    y_full = (feat_df[target_col].to_numpy().astype(np.float64)
              if target_col in feat_df.columns else None)

    n = len(X_full)
    cfg = type("Cfg", (), {"split": SplitConfig()})()
    paper_cutoff = cfg.split.paper_cutoff_week or n
    n_in_sample = min(paper_cutoff, n)
    n_real = max(0, n - n_in_sample)
    n_train, n_val, n_test = compute_split_indices(n_in_sample, cfg)

    # boundaries (inclusive lower, exclusive upper):
    bounds = {
        "train": (0,                         n_train),
        "val":   (n_train,                   n_train + n_val),
        "test":  (n_train + n_val,           n_in_sample),
        "real":  (n_in_sample,               n),
    }

    return {
        "X_full":       X_full,
        "y_full":       y_full,
        "dates":        dates_arr,
        "feature_cols": feature_cols,
        "n":            n,
        "bounds":       bounds,
        "n_in_sample":  n_in_sample,
        "n_real":       n_real,
        "paper_cutoff": paper_cutoff,
    }


def _pad_X_for_artifact(X: np.ndarray, artifact) -> np.ndarray:
    """Pad zeros if artifact expects more features than X has.

    Three places hold the expected count, in priority order:
      (a) artifact.scaler.n_features_in_   (sklearn fitted scaler)
      (b) max(artifact.feature_indices) + 1 (Optuna feature subset upper bound)
      (c) artifact.model.n_features_in_     (sklearn / xgboost / lightgbm fit'd estimator)
      (d) artifact.model.booster_.num_feature() (lightgbm low-level API)

    The covid_era_indicator pad-with-zero is safe: indicator is always 0 for
    weeks outside 2020-03..2022-12, which inference always is.
    """
    expected = 0
    # (e) artifact.config['n_features'] — recorded by R9 (per_model_optimize) at fit time;
    #     covers the case where the wrapped estimator hides its own scaler
    #     (e.g. some BayesianRidge / LightGBM wrappers internalize StandardScaler
    #     so artifact.scaler is None but the model still expects n_features+1).
    cfg = getattr(artifact, "config", {}) or {}
    if isinstance(cfg.get("n_features"), int):
        expected = max(expected, int(cfg["n_features"]))
    if artifact.scaler is not None:
        try:
            expected = max(expected, int(artifact.scaler.n_features_in_))
        except Exception:
            pass
    if artifact.feature_indices:
        expected = max(expected, max(artifact.feature_indices) + 1)
    # Model-level introspection
    m = artifact.model
    for attr in ("n_features_in_", "n_features_"):
        v = getattr(m, attr, None)
        if isinstance(v, int) and v > expected:
            expected = v
    # XGBoost: get_booster().num_features()
    try:
        b = m.get_booster()
        nf = getattr(b, "num_features", None)
        if callable(nf):
            v = int(nf())
            if v > expected:
                expected = v
    except Exception:
        pass
    # LightGBM: model.booster_.num_feature()
    try:
        b = getattr(m, "booster_", None)
        if b is not None and hasattr(b, "num_feature"):
            v = int(b.num_feature())
            if v > expected:
                expected = v
    except Exception:
        pass
    # (f) walk every nested sub-estimator/transformer with n_features_in_
    #     (e.g. sklearn Pipeline with internal PCA / StandardScaler that
    #     the top-level model object hides). Walks .steps / .named_steps /
    #     .transformer_ / .pipeline_ etc.
    seen = set()
    def _walk(o, depth=0):
        if depth > 4 or id(o) in seen:
            return
        seen.add(id(o))
        v = getattr(o, "n_features_in_", None)
        if isinstance(v, int):
            yield v
        for sub_attr in ("steps", "named_steps", "transformer_",
                            "pipeline_", "estimator_", "regressor_",
                            "preprocessor_", "estimators_", "_pre"):
            sub = getattr(o, sub_attr, None)
            if sub is None:
                continue
            if isinstance(sub, (list, tuple)):
                for it in sub:
                    if isinstance(it, tuple) and len(it) >= 2:
                        yield from _walk(it[1], depth+1)
                    else:
                        yield from _walk(it, depth+1)
            elif isinstance(sub, dict):
                for it in sub.values():
                    yield from _walk(it, depth+1)
            else:
                yield from _walk(sub, depth+1)
    for v in _walk(m):
        if v > expected:
            expected = v
    cur = X.shape[1]
    if expected > cur:
        return np.hstack([X, np.zeros((len(X), expected - cur), dtype=np.float64)])
    return X


# ─────────────────────────────────────────────────────────────────
# Plot 1 — Per-model time series (4 slab bands + peaks)
# ─────────────────────────────────────────────────────────────────
def _plot_per_model_timeseries(model_name: str, y_pred: np.ndarray,
                                  y_full: np.ndarray, dates: np.ndarray,
                                  bounds: dict, out_path: Path,
                                  meta_summary: dict) -> None:
    plt = _setup_mpl()
    import pandas as pd

    fig, ax = plt.subplots(figsize=(13, 4.5))
    x = pd.to_datetime(dates) if dates is not None else np.arange(len(y_full))

    # Slab bands
    for slab, (lo, hi) in bounds.items():
        if hi <= lo or hi > len(x):
            continue
        color, label = SLAB_COLORS[slab]
        ax.axvspan(x[lo] if hasattr(x, "__getitem__") else lo,
                    x[min(hi-1, len(x)-1)] if hi <= len(x) else x[-1],
                    alpha=0.35, color=color, label=f"{label} (n={hi-lo})",
                    zorder=0)

    # Actual
    mask = np.isfinite(y_full)
    ax.plot(x[mask], y_full[mask], "k-", lw=1.4, label="Actual ILI", zorder=3)

    # Predicted
    pred_mask = np.isfinite(y_pred)
    ax.plot(x[pred_mask], y_pred[pred_mask], "C0--", lw=1.4,
            label=f"{model_name} prediction", zorder=4)

    # Annotate peaks (top-3 in actual)
    if mask.sum() > 0:
        peak_idx = np.argsort(-y_full[mask])[:3]
        for pi in peak_idx:
            tx = x[mask][pi] if hasattr(x, "__getitem__") else int(pi)
            ax.annotate(f"{y_full[mask][pi]:.0f}", xy=(tx, y_full[mask][pi]),
                          xytext=(0, 8), textcoords="offset points",
                          fontsize=8, ha="center",
                          arrowprops=dict(arrowstyle="-", lw=0.5, color="gray"))

    # Title with config + metrics summary
    cfg_str = (f"transform={meta_summary.get('transform','?')}, "
                f"scaler={meta_summary.get('scaler','?')}, "
                f"n_feat={meta_summary.get('n_features','?')}")
    test_wis = meta_summary.get("test_wis")
    title = f"[{model_name}]  {cfg_str}"
    if test_wis is not None:
        title += f"  ·  test_WIS={test_wis:.2f}"
    ax.set_title(title)
    ax.set_xlabel("week")
    ax.set_ylabel("ILI rate (per 1,000)")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
# Plot 2 — Combined timeseries (all models on one axes)
# ─────────────────────────────────────────────────────────────────
def _plot_combined_timeseries(predictions: dict[str, np.ndarray],
                                  y_full: np.ndarray, dates: np.ndarray,
                                  bounds: dict, out_path: Path) -> None:
    plt = _setup_mpl()
    import pandas as pd

    fig, ax = plt.subplots(figsize=(14, 5.5))
    x = pd.to_datetime(dates) if dates is not None else np.arange(len(y_full))

    for slab, (lo, hi) in bounds.items():
        if hi <= lo or hi > len(x):
            continue
        color, label = SLAB_COLORS[slab]
        ax.axvspan(x[lo] if hasattr(x, "__getitem__") else lo,
                    x[min(hi-1, len(x)-1)] if hi <= len(x) else x[-1],
                    alpha=0.30, color=color, label=f"{label}", zorder=0)

    mask = np.isfinite(y_full)
    ax.plot(x[mask], y_full[mask], "k-", lw=1.7, label="Actual ILI",
            zorder=10)

    cmap = plt.get_cmap("tab10")
    for i, (nm, yp) in enumerate(sorted(predictions.items())):
        pm = np.isfinite(yp)
        ax.plot(x[pm], yp[pm], "--", lw=1.0, alpha=0.85,
                color=cmap(i % 10), label=nm, zorder=5)

    ax.set_title(f"Combined champions overview · {len(predictions)} models")
    ax.set_xlabel("week")
    ax.set_ylabel("ILI rate (per 1,000)")
    ax.legend(loc="upper left", fontsize=8, ncols=2, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
# Plot 3 — Residual diagnostics per model (3-panel)
# ─────────────────────────────────────────────────────────────────
def _plot_residuals(model_name: str, y_pred: np.ndarray, y_full: np.ndarray,
                      dates: np.ndarray, bounds: dict, out_path: Path) -> None:
    plt = _setup_mpl()
    import pandas as pd
    from scipy import stats

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.2))
    mask = np.isfinite(y_pred) & np.isfinite(y_full)
    resid = y_pred[mask] - y_full[mask]
    x = pd.to_datetime(dates) if dates is not None else np.arange(len(y_full))

    # 1. Residuals over time, slab-colored
    for slab, (lo, hi) in bounds.items():
        if hi <= lo:
            continue
        idx = np.arange(lo, min(hi, len(y_full)))
        idx = idx[np.isfinite(y_pred[idx]) & np.isfinite(y_full[idx])]
        if len(idx) == 0:
            continue
        color, label = SLAB_COLORS[slab]
        ax1.scatter(x[idx] if hasattr(x, "__getitem__") else idx,
                     y_pred[idx] - y_full[idx],
                     s=14, color=color, edgecolor="k", lw=0.4,
                     label=label, alpha=0.85)
    ax1.axhline(0, color="k", lw=0.6)
    ax1.set_title("Residual over time")
    ax1.set_xlabel("week"); ax1.set_ylabel("pred − actual")
    ax1.legend(fontsize=8)

    # 2. Scatter actual vs predicted
    for slab, (lo, hi) in bounds.items():
        if hi <= lo:
            continue
        idx = np.arange(lo, min(hi, len(y_full)))
        idx = idx[np.isfinite(y_pred[idx]) & np.isfinite(y_full[idx])]
        if len(idx) == 0:
            continue
        color, label = SLAB_COLORS[slab]
        ax2.scatter(y_full[idx], y_pred[idx], s=18, color=color, edgecolor="k",
                     lw=0.4, label=label, alpha=0.85)
    if mask.any():
        lim = (min(y_full[mask].min(), y_pred[mask].min()),
               max(y_full[mask].max(), y_pred[mask].max()))
        ax2.plot(lim, lim, "k--", lw=0.6, label="y=x")
        ax2.set_xlim(lim); ax2.set_ylim(lim)
    ax2.set_title("Predicted vs Actual")
    ax2.set_xlabel("actual ILI"); ax2.set_ylabel("predicted ILI")
    ax2.legend(fontsize=8)

    # 3. Q-Q normal
    if len(resid) >= 5:
        stats.probplot(resid, dist="norm", plot=ax3)
        ax3.set_title("Q-Q plot — residual normality")
        ax3.get_lines()[0].set_markersize(4)
        ax3.get_lines()[0].set_markeredgecolor("k")
        ax3.get_lines()[0].set_markerfacecolor("#1f77b4")

    fig.suptitle(f"[{model_name}] residual diagnostics  "
                  f"(n={mask.sum()}, μ={resid.mean():+.2f}, σ={resid.std():.2f})",
                  fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
# Plot 4 — Per-horizon error bar (real slab only)
# ─────────────────────────────────────────────────────────────────
def _per_slab_metrics(yp: np.ndarray, y: np.ndarray,
                         bounds: dict) -> dict:
    """Compute MAE / RMSE / R² / MAPE / sMAPE / bias per slab."""
    out: dict = {}
    for slab, (lo, hi) in bounds.items():
        if hi <= lo or hi > len(y):
            continue
        idx = np.arange(lo, min(hi, len(y)))
        mask = np.isfinite(yp[idx]) & np.isfinite(y[idx])
        if not mask.any():
            out[slab] = {"n": 0}
            continue
        yp_s = yp[idx][mask]; y_s = y[idx][mask]
        err = yp_s - y_s
        ae = np.abs(err)
        mae = float(np.mean(ae))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        sst = float(np.sum((y_s - y_s.mean()) ** 2))
        sse = float(np.sum(err ** 2))
        r2 = (1.0 - sse / sst) if sst > 1e-9 else float("nan")
        # MAPE / sMAPE only on positive y
        pos = y_s > 1e-3
        mape = (float(np.mean(ae[pos] / y_s[pos]) * 100)
                  if pos.any() else float("nan"))
        smape = (float(np.mean(2.0 * ae[pos] /
                                  (np.abs(yp_s[pos]) + np.abs(y_s[pos])))
                          * 100)
                    if pos.any() else float("nan"))
        out[slab] = {
            "n": int(mask.sum()),
            "mae":   mae,
            "rmse":  rmse,
            "r2":    r2,
            "mape":  mape,
            "smape": smape,
            "bias":  float(np.mean(err)),
        }
    return out


def _build_per_model_md(model_name: str, summary: dict, slab_metrics: dict,
                          per_horizon_real: list[dict],
                          has_residuals: bool, has_optuna: bool,
                          out_dir: Path) -> Path:
    """Generate per-model markdown report with figure + metrics tables."""
    md = [
        f"# {model_name} — champion report",
        "",
        f"_생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
        "## Champion configuration",
        "",
        f"| field | value |",
        f"|---|---|",
        f"| transform | `{summary.get('transform','?')}` |",
        f"| scaler | `{summary.get('scaler') or 'none'}` |",
        f"| n_features used | {summary.get('n_features','?')} |",
        f"| test_WIS @ promotion | {summary.get('test_wis')} |",
        f"| test_MAE @ promotion | {summary.get('test_mae')} |",
        f"| test_R² @ promotion | {summary.get('test_r2')} |",
        "",
        "## Time-series overview (train / val / test / real)",
        "",
        f"![{model_name}_overview](./{model_name}_overview.png)",
        "",
        "## Per-slab metrics",
        "",
        f"| slab | n | MAE | RMSE | R² | MAPE% | sMAPE% | bias |",
        f"|---|---|---|---|---|---|---|---|",
    ]
    for slab in ("train", "val", "test", "real"):
        m = slab_metrics.get(slab, {})
        n = m.get("n", 0)
        if n == 0:
            md.append(f"| {slab} | 0 | — | — | — | — | — | — |")
            continue
        def f(k, p=3):
            v = m.get(k)
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                return "?"
            return f"{v:.{p}f}"
        md.append(f"| {slab} | {n} | {f('mae')} | {f('rmse')} | "
                  f"{f('r2')} | {f('mape',2)} | {f('smape',2)} | "
                  f"{f('bias')} |")
    md.append("")

    if per_horizon_real:
        md += [
            "## Per-horizon (real slab) — h=1 is operational KPI",
            "",
            f"| horizon | actual | pred | AE | APE% |",
            f"|---|---|---|---|---|",
        ]
        for r in per_horizon_real:
            ape = r.get('ape')
            ape_s = f"{ape:.2f}" if isinstance(ape,(int,float)) and np.isfinite(ape) else "?"
            md.append(f"| h={r['h']} | {r['actual']:.3f} | "
                      f"{r['pred']:.3f} | {r['ae']:.3f} | {ape_s} |")
        md.append("")

    if has_residuals:
        md += ["## Residual diagnostics",
                "",
                f"![{model_name}_residuals](./{model_name}_residuals.png)",
                ""]

    if has_optuna:
        md += ["## Optuna trial history (this model)",
                "",
                f"![{model_name}_optuna](./{model_name}_optuna.png)",
                ""]

    md += ["---", "",
            "_Generated by `simulation visualize` — uses ChampionArtifact `.pt` files only,_",
            "_no re-training required._"]
    out_md = out_dir / f"{model_name}_report.md"
    out_md.write_text("\n".join(md))
    return out_md


def _plot_per_horizon_real(predictions: dict[str, np.ndarray],
                              y_full: np.ndarray, bounds: dict,
                              out_path: Path) -> None:
    plt = _setup_mpl()
    real_lo, real_hi = bounds.get("real", (0, 0))
    if real_hi <= real_lo:
        return
    horizons = list(range(1, real_hi - real_lo + 1))
    actual = y_full[real_lo:real_hi]
    if not np.isfinite(actual).any():
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    cmap = plt.get_cmap("tab10")
    width = 0.8 / max(len(predictions), 1)
    sorted_models = sorted(predictions.keys(),
                            key=lambda nm: float(np.mean(np.abs(
                                predictions[nm][real_lo:real_hi] - actual))
                                if np.isfinite(predictions[nm][real_lo:real_hi]).any()
                                else 9e9))

    x_pos = np.arange(len(horizons))
    for i, nm in enumerate(sorted_models):
        pred = predictions[nm][real_lo:real_hi]
        ae = np.abs(pred - actual)
        ax.bar(x_pos + i * width - 0.4, ae, width=width,
                label=nm, color=cmap(i % 10), edgecolor="k", lw=0.4)

    # Annotate h=1 KPI
    ax.axvspan(-0.5, 0.5, alpha=0.12, color="red", zorder=0)
    ax.text(0, ax.get_ylim()[1] * 0.95,
              "h=1 (operational KPI)", ha="center", fontsize=9,
              color="darkred", fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"h={h}\n(actual {actual[i]:.1f})"
                          for i, h in enumerate(horizons)])
    ax.set_xlabel("Horizon (weeks ahead)")
    ax.set_ylabel("Absolute error")
    ax.set_title("Per-horizon AE on real slab — sorted by h=1 (lower = better)")
    ax.legend(fontsize=8, ncols=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
# Plot 5 — Optuna trial history
# ─────────────────────────────────────────────────────────────────
def _read_optuna_studies(db_path: Path) -> dict[str, list[tuple[int, float]]]:
    """Return {study_name: [(trial_number, value), ...]}. Empty if unavailable."""
    if not db_path.exists():
        return {}
    try:
        # G-116/G-117: 직접 연결 대신 safe_connect (PRAGMA quick_check 손상방어).
        # Optuna study DB(read-only SELECT); get_conn 은 fresh conn 반환 → conn.close() 안전.
        # DatabaseCorruptError 는 아래 except 가 graceful 처리(빈 dict).
        from simulation.database import safe_connect
        conn = safe_connect(str(db_path))
        cur = conn.cursor()
        rows = cur.execute("""
            SELECT s.study_name, t.number, tv.value
            FROM trials t
            JOIN studies s ON t.study_id = s.study_id
            JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE t.state = 'COMPLETE'
            ORDER BY s.study_name, t.number
        """).fetchall()
        conn.close()
    except Exception as e:
        log.warning(f"  [visualize] optuna DB read failed: {e}")
        return {}
    by_study: dict[str, list[tuple[int, float]]] = {}
    for sn, n_trial, val in rows:
        by_study.setdefault(sn, []).append((n_trial, val))
    return by_study


def _classify_optuna_study(study_name: str) -> str:
    """Classify an Optuna study name → 'feature' / 'hp' / 'joint' / 'other'.

    Convention from `_optuna_torch.py`:
      - feat_only_*  / feat_r0_*..feat_r9_*    → feature selection (split-CV)
      - hp_*                                    → HP-only optimization
      - joint_*                                 → joint feature × HP search
      - everything else                         → 'other' (rare)
    """
    sn = study_name.lower()
    if sn.startswith("feat_") or "_feat_" in sn or "feature" in sn:
        return "feature"
    if sn.startswith("joint_"):
        return "joint"
    if sn.startswith("hp_"):
        return "hp"
    return "other"


def _plot_optuna_history(out_dir: Path, by_study: dict,
                            kind: str = "all",
                            top_models: Optional[list[str]] = None) -> Optional[Path]:
    """Combined Optuna history filtered by ``kind`` (feature / hp / joint / all).

    Each line = one study's running-best objective. Color encodes which
    *model* the study belongs to (substring match on top_models if given).
    """
    if not by_study:
        return None
    if kind != "all":
        by_study = {k: v for k, v in by_study.items()
                    if _classify_optuna_study(k) == kind}
    if top_models:
        wanted = set(top_models)
        by_study = {k: v for k, v in by_study.items()
                    if any(m.lower() in k.lower() for m in wanted)}
    if not by_study:
        return None

    plt = _setup_mpl()
    fig, ax = plt.subplots(figsize=(14, 5.5))
    cmap = plt.get_cmap("tab20")
    # Color by model — derived from study name
    def _model_from_study(sn: str) -> str:
        # feat_only_xgboost_v15 → "xgboost"
        # hp_lightgbm_v15 → "lightgbm"
        parts = sn.split("_")
        # skip prefix tokens
        for skip in ("feat", "only", "r0", "r1", "r2", "r3", "r4",
                       "r5", "r6", "r7", "r8", "r9",
                       "hp", "joint"):
            if parts and parts[0] == skip:
                parts.pop(0)
        return parts[0] if parts else sn
    by_model: dict[str, list] = {}
    for sn, hist in by_study.items():
        by_model.setdefault(_model_from_study(sn), []).append((sn, hist))

    for i, (mdl, items) in enumerate(sorted(by_model.items())):
        col = cmap(i % 20)
        for sn, hist in items:
            ns, vs = zip(*hist)
            running_best = np.minimum.accumulate(np.array(vs, dtype=np.float64))
            ax.plot(ns, running_best, "-", lw=1.0, alpha=0.6,
                    color=col, label=None)
        # Add a legend entry per model with mean final value
        finals = [np.minimum.accumulate(np.array([v for _, v in h],
                                                    dtype=np.float64))[-1]
                   for _, h in items]
        ax.plot([], [], "-", lw=1.6, color=col,
                label=f"{mdl}  (n_studies={len(items)}, "
                      f"final≈{np.mean(finals):.3f})")

    title_map = {
        "all":     f"Optuna trial-history — ALL studies ({len(by_study)})",
        "feature": f"Optuna FEATURE-selection studies ({len(by_study)})",
        "hp":      f"Optuna HP-only studies ({len(by_study)})",
        "joint":   f"Optuna JOINT (feature×HP) studies ({len(by_study)})",
    }
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Optuna objective (running best, lower=better)")
    ax.set_title(title_map.get(kind, "Optuna trial-history"))
    ax.legend(fontsize=8, ncols=2, loc="upper right",
              title="model (running best, mean over studies)",
              title_fontsize=8.5)
    fig.tight_layout()
    name_map = {"all": "optuna_trial_history.png",
                  "feature": "optuna_feature_selection.png",
                  "hp": "optuna_hp_history.png",
                  "joint": "optuna_joint_history.png"}
    out = out_dir / name_map.get(kind, "optuna_trial_history.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def _plot_optuna_per_model(out_path: Path, model_name: str,
                              by_study: dict,
                              kind: str = "all") -> Optional[Path]:
    """Single-model Optuna history. Optionally filter by kind.

    ``kind`` ∈ {'all', 'feature', 'hp', 'joint'}. The figure shows raw
    trial values (scatter) + running best (line) per matching study.
    """
    nm_lower = model_name.lower().replace("-", "")
    matching: dict = {}
    for k, v in by_study.items():
        if kind != "all" and _classify_optuna_study(k) != kind:
            continue
        if (model_name.lower() in k.lower() or
              nm_lower in k.lower().replace("-", "").replace("_", "")):
            matching[k] = v
    if not matching:
        return None
    plt = _setup_mpl()
    fig, ax = plt.subplots(figsize=(12, 5))
    cmap = plt.get_cmap("tab10")
    for i, (sn, hist) in enumerate(sorted(matching.items())):
        ns, vs = zip(*hist)
        ns = np.array(ns); vs = np.array(vs, dtype=np.float64)
        running_best = np.minimum.accumulate(vs)
        ax.scatter(ns, vs, s=14, color=cmap(i % 10), alpha=0.45,
                     edgecolor="k", lw=0.3, label=f"{sn} (raw, n={len(vs)})")
        ax.plot(ns, running_best, "-", lw=1.5, color=cmap(i % 10),
                  label=f"{sn} (running best)")
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Optuna objective (lower = better)")
    title_map = {
        "all":     f"[{model_name}] Optuna trial-history (all studies)",
        "feature": f"[{model_name}] Optuna FEATURE-selection",
        "hp":      f"[{model_name}] Optuna HP-only",
        "joint":   f"[{model_name}] Optuna JOINT (feature×HP)",
    }
    ax.set_title(title_map.get(kind, f"[{model_name}] Optuna"))
    ax.legend(fontsize=8, loc="upper right", ncols=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────────────────────────
# Public entry
# ─────────────────────────────────────────────────────────────────
def run_visualize(*, models_filter: Optional[list[str]] = None,
                    include_residuals: bool = True,
                    include_optuna: bool = True,         # default ON now
                    out_dir: Optional[Path] = None,
                    repo_root: Optional[Path] = None) -> dict:
    """Generate all visualizations from saved champion .pt files."""
    if repo_root is None:
        repo_root = Path.cwd()
    repo_root = Path(repo_root)
    if out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = repo_root / "simulation" / "results" / "visualizations" / ts
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    log.info(f"[visualize] output → {out_dir}")

    # 1. Build the full timeline
    log.info("[visualize] building full feature matrix …")
    tl = _build_full_timeline(repo_root)
    X_full = tl["X_full"]
    y_full = tl["y_full"]
    dates = tl["dates"]
    bounds = tl["bounds"]

    # 2. Load champions
    from simulation.utils.champion_log import ChampionLog
    from simulation.utils.model_artifact import load_artifact

    cl = ChampionLog(models_dir=repo_root / "models",
                       log_path=repo_root / "models" / "champion_log.json")
    sm = cl.summary()
    if not sm:
        log.warning("[visualize] no champions found in models/champion_log.json")
        return {"skipped": True, "reason": "no champions",
                "elapsed": time.time() - t0}

    if models_filter:
        sm = {k: v for k, v in sm.items() if k in models_filter}
        if not sm:
            log.warning(f"[visualize] no champions match filter {models_filter}")
            return {"skipped": True, "reason": "no models match filter",
                    "elapsed": time.time() - t0}

    log.info(f"[visualize] {len(sm)} champions: {sorted(sm.keys())}")

    # 3. Predict each champion across the full timeline
    import re
    predictions: dict[str, np.ndarray] = {}
    summaries: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    for nm, info in sm.items():
        pt = repo_root / "models" / f"{nm}.pt"
        art = load_artifact(pt)
        if art is None:
            skipped[nm] = "load_artifact returned None (file unloadable)"
            continue

        # torch state_dict-only saves are dicts → no .predict method
        if not hasattr(art.model, "predict"):
            skipped[nm] = (f"loaded object is {type(art.model).__name__} "
                            f"(no .predict — torch state_dict, needs class re-construct)")
            log.warning(f"[visualize] {nm} skipped: {skipped[nm]}")
            continue

        Xp = _pad_X_for_artifact(X_full, art)
        # try the static padding; if it fails with shape mismatch, parse
        # the error and pad to the requested width and retry once.
        try:
            yp = np.asarray(art.predict(Xp), dtype=np.float64)
        except Exception as e:
            msg = str(e)
            m = re.search(r"expecting (\d+)\s*feature", msg)
            if not m:
                m = re.search(r"training data \((\d+)\)", msg)
            if m:
                want = int(m.group(1))
                if want > Xp.shape[1]:
                    Xp2 = np.hstack([Xp, np.zeros((len(Xp), want - Xp.shape[1]),
                                                       dtype=np.float64)])
                    try:
                        yp = np.asarray(art.predict(Xp2), dtype=np.float64)
                    except Exception as e2:
                        skipped[nm] = f"predict failed even after retry pad to {want}: {str(e2)[:120]}"
                        log.warning(f"[visualize] {nm} retry-pad failed: {e2}")
                        continue
                else:
                    skipped[nm] = f"predict failed: {msg[:120]}"
                    log.warning(f"[visualize] {nm} predict failed: {e}")
                    continue
            else:
                skipped[nm] = f"predict failed: {msg[:120]}"
                log.warning(f"[visualize] {nm} predict failed: {e}")
                continue

        if len(yp) != len(y_full):
            skipped[nm] = f"prediction length {len(yp)} ≠ {len(y_full)}"
            continue
        predictions[nm] = yp
        sm_summary = art.summary()
        summaries[nm] = {
            "transform": sm_summary.get("transform_name"),
            "scaler":    sm_summary.get("scaler_class"),
            "n_features": sm_summary.get("n_features_used"),
            "test_wis":  info.get("current_test_wis"),
            "test_mae":  info.get("current_test_mae"),
            "test_r2":   info.get("current_test_r2"),
            "is_legacy": bool((art.config or {}).get("legacy", False)),
        }

    if not predictions:
        log.error("[visualize] no champions produced predictions")
        return {"skipped": True, "reason": "all champions failed predict",
                "elapsed": time.time() - t0}

    # 4. Per-model timeseries + residuals + Optuna (per model) + markdown report
    manifest: dict = {"figures": [], "per_model_reports": [],
                       "skipped":   skipped,
                       "metadata": {
        "n_champions_total": len(sm),
        "n_champions_rendered": len(predictions),
        "n_skipped":   len(skipped),
        "n_in_sample": tl["n_in_sample"],
        "n_real":      tl["n_real"],
        "paper_cutoff": tl["paper_cutoff"],
        "bounds":      bounds,
        "out_dir":     str(out_dir),
    }}
    per_model_dir = out_dir / "per_model"
    per_model_dir.mkdir(exist_ok=True)

    # Pre-load Optuna once (reused across models)
    op_db = repo_root / "simulation" / "results" / "optuna_feature_selection.db"
    optuna_studies = _read_optuna_studies(op_db) if include_optuna else {}

    for nm, yp in sorted(predictions.items()):
        # 4a. Time-series overview
        ts_path = per_model_dir / f"{nm}_overview.png"
        try:
            _plot_per_model_timeseries(nm, yp, y_full, dates, bounds, ts_path,
                                          summaries.get(nm, {}))
            manifest["figures"].append({"model": nm, "kind": "timeseries",
                                          "path": str(ts_path)})
        except Exception as e:
            log.warning(f"[visualize] {nm} timeseries plot failed: {e}")

        # 4b. Residual diagnostic
        has_resid = False
        if include_residuals:
            rs_path = per_model_dir / f"{nm}_residuals.png"
            try:
                _plot_residuals(nm, yp, y_full, dates, bounds, rs_path)
                manifest["figures"].append({"model": nm, "kind": "residuals",
                                              "path": str(rs_path)})
                has_resid = True
            except Exception as e:
                log.warning(f"[visualize] {nm} residual plot failed: {e}")

        # 4c. Optuna learning curves (per model, split by kind)
        has_optuna_for_model = False
        if include_optuna and optuna_studies:
            for kind, suffix in (("all",     "_optuna.png"),
                                  ("feature", "_optuna_feature.png"),
                                  ("hp",      "_optuna_hp.png"),
                                  ("joint",   "_optuna_joint.png")):
                op_path = per_model_dir / f"{nm}{suffix}"
                try:
                    got = _plot_optuna_per_model(op_path, nm, optuna_studies,
                                                   kind=kind)
                    if got:
                        manifest["figures"].append({
                            "model": nm,
                            "kind": f"optuna_{kind}",
                            "path": str(op_path),
                        })
                        if kind == "all":
                            has_optuna_for_model = True
                except Exception as e:
                    log.warning(f"[visualize] {nm} optuna {kind} plot failed: {e}")

        # 4d. Per-slab metrics + per-horizon (real) tables
        slab_metrics = _per_slab_metrics(yp, y_full, bounds)
        per_h: list[dict] = []
        real_lo, real_hi = bounds.get("real", (0, 0))
        for j in range(real_lo, min(real_hi, len(y_full))):
            if not (np.isfinite(yp[j]) and np.isfinite(y_full[j])):
                continue
            ae = float(abs(yp[j] - y_full[j]))
            ape = (ae / float(y_full[j]) * 100
                   if abs(y_full[j]) > 1e-3 else float("nan"))
            per_h.append({
                "h":       j - real_lo + 1,
                "actual":  float(y_full[j]),
                "pred":    float(yp[j]),
                "ae":      ae,
                "ape":     ape,
            })

        # 4e. Per-model markdown report (figure + metrics tables)
        try:
            md_path = _build_per_model_md(nm, summaries.get(nm, {}),
                                              slab_metrics, per_h,
                                              has_resid, has_optuna_for_model,
                                              per_model_dir)
            manifest["per_model_reports"].append({"model": nm,
                                                     "md": str(md_path)})
        except Exception as e:
            log.warning(f"[visualize] {nm} md report failed: {e}")

    # 5. Combined timeseries
    combined_path = out_dir / "all_models_timeseries.png"
    try:
        _plot_combined_timeseries(predictions, y_full, dates, bounds, combined_path)
        manifest["figures"].append({"kind": "combined", "path": str(combined_path)})
    except Exception as e:
        log.warning(f"[visualize] combined timeseries failed: {e}")

    # 6. Per-horizon real slab
    if bounds["real"][1] > bounds["real"][0]:
        ph_path = out_dir / "per_horizon_real_AE.png"
        try:
            _plot_per_horizon_real(predictions, y_full, bounds, ph_path)
            manifest["figures"].append({"kind": "per_horizon", "path": str(ph_path)})
        except Exception as e:
            log.warning(f"[visualize] per-horizon plot failed: {e}")

    # 7. Optuna learning curves — combined, separated by kind
    #    Generates 4 panels: ALL / FEATURE / HP / JOINT
    if include_optuna and optuna_studies:
        # Classification breakdown for the manifest
        breakdown: dict[str, int] = {}
        for sn in optuna_studies:
            breakdown[_classify_optuna_study(sn)] = (
                breakdown.get(_classify_optuna_study(sn), 0) + 1)
        manifest["metadata"]["optuna_studies_by_kind"] = breakdown

        for kind in ("all", "feature", "hp", "joint"):
            op_path = _plot_optuna_history(out_dir, optuna_studies, kind=kind)
            if op_path:
                manifest["figures"].append({"kind": f"optuna_{kind}",
                                              "path": str(op_path)})

    # 8. Manifest + INDEX.md
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str))

    md = ["# Visualization Index", "",
          f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
          f"- Champions in log: {len(sm)}",
          f"- Successfully rendered: {len(predictions)}",
          f"- Skipped: {len(skipped)} (see bottom of report)",
          f"- Figures: {len(manifest['figures'])}",
          f"- Bounds: train [0:{bounds['train'][1]}], "
          f"val [{bounds['val'][0]}:{bounds['val'][1]}], "
          f"test [{bounds['test'][0]}:{bounds['test'][1]}], "
          f"real [{bounds['real'][0]}:{bounds['real'][1]}]",
          ""]
    md.append("## Combined")
    if (out_dir / "all_models_timeseries.png").exists():
        md.append(f"![combined](all_models_timeseries.png)")
    md.append("")
    md.append("## Per-horizon (real slab)")
    if (out_dir / "per_horizon_real_AE.png").exists():
        md.append(f"![per_horizon](per_horizon_real_AE.png)")
    md.append("")
    if include_optuna:
        breakdown = manifest["metadata"].get("optuna_studies_by_kind", {})
        if any((out_dir / fn).exists() for fn in
                ("optuna_trial_history.png", "optuna_feature_selection.png",
                 "optuna_hp_history.png", "optuna_joint_history.png")):
            md.append("## Optuna trial history")
            md.append("")
            if breakdown:
                md.append(f"- **Total studies**: {sum(breakdown.values())}")
                for k in ("feature", "hp", "joint", "other"):
                    if breakdown.get(k):
                        md.append(f"- {k}: {breakdown[k]} studies")
                md.append("")
            for fn, label in (
                ("optuna_feature_selection.png", "Feature-selection studies"),
                ("optuna_hp_history.png",        "HP-only studies"),
                ("optuna_joint_history.png",     "Joint (feature × HP) studies"),
                ("optuna_trial_history.png",     "ALL studies (combined)"),
            ):
                if (out_dir / fn).exists():
                    md.append(f"### {label}")
                    md.append(f"![{fn[:-4]}]({fn})")
                    md.append("")
    md.append("## Per-model deep-dives")
    md.append("")
    md.append("각 모델의 전체 보고서 (그래프 + slab 별 metrics 표 + per-horizon 표) — 클릭:")
    md.append("")
    for nm in sorted(predictions.keys()):
        s = summaries.get(nm, {})
        wis = s.get('test_wis')
        wis_s = f"{wis:.3f}" if isinstance(wis,(int,float)) else "?"
        md.append(f"- **[{nm}](per_model/{nm}_report.md)** — "
                  f"transform=`{s.get('transform','?')}`, "
                  f"scaler=`{s.get('scaler') or 'none'}`, "
                  f"test_WIS={wis_s}")
    md.append("")
    md.append("### 미리보기 (썸네일)")
    md.append("")
    for nm in sorted(predictions.keys()):
        md.append(f"#### {nm}")
        md.append(f"![{nm}_ts](per_model/{nm}_overview.png)")
        md.append("")

    # Skipped models — transparency
    if skipped:
        md.append("---")
        md.append("")
        md.append(f"## ⚠️ Skipped models ({len(skipped)})")
        md.append("")
        md.append("이 모델들은 champion_log 에 등록되어 있으나 inference 실행 불가:")
        md.append("")
        md.append(f"| model | reason |")
        md.append(f"|---|---|")
        for nm in sorted(skipped.keys()):
            md.append(f"| {nm} | {skipped[nm]} |")
        md.append("")
        md.append("**복구 방법**:")
        md.append("- `'dict' object has no attribute 'predict'` → torch state_dict 만 저장됨. "
                  "원본 모델 클래스 + 하이퍼파라미터로 재구성 필요. **권장: R9 (per_model_optimize) 재실행** 으로 ChampionArtifact 형식으로 재저장.")
        md.append("- `expecting N features` (padding fail) → 자동 padding retry 도 fail. "
                  "X_full 의 feature 차원과 학습 시점 차원 불일치 → 학습 데이터셋이 변경된 듯.")
        md.append("- `Ran out of input` → 파일 손상. 재학습 필요.")
        md.append("")
    (out_dir / "INDEX.md").write_text("\n".join(md))

    elapsed = time.time() - t0
    log.info(f"[visualize] done in {elapsed:.1f}s — {len(manifest['figures'])} figures")
    return {
        "out_dir":   str(out_dir),
        "n_figures": len(manifest["figures"]),
        "n_models":  len(predictions),
        "elapsed":   elapsed,
        "index_md":  str(out_dir / "INDEX.md"),
    }


__all__ = ["run_visualize"]
