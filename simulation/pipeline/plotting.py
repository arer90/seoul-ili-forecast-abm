"""plotting & CSV layer
==================================
End-of-pipeline artifacts:
 * plots_matplotlib/ — PNG (learning curves, R² bar, pred-vs-actual)
 * plots_plotly/ — interactive HTML (same three families)
 * plots_seaborn/ — seaborn polished PNG
 * csv/ — summary, history_<model>.csv, predictions_<model>.csv

The hook is idempotent and failure-isolated: any exception degrades to a
logged warning without killing the pipeline. Absence of matplotlib /
plotly / seaborn / pandas is tolerated (produces whatever is available).

Input contract
--------------
phase2_runner_result: dict returned by MultiModelRunner.run, i.e.
 {
 "individual_results": {name: {val_pred, test_pred, val_metrics,
 test_metrics, history?, ...}},
 "ensemble_results": {name: {val_pred, test_pred, ...}},
 "summary": pd.DataFrame (optional),
 ...
 }

y_val, y_test: ndarray — the original-scale ground truth arrays. The
caller must supply these from phase1/phase2 scope.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


__all__ = ["generate_all"]


# ══════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════

def _safe_mkdir(p: Path) -> Path:
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        log.warning(f"  [plot] mkdir({p}) failed: {e}")
    return p


def _clip_arr(a) -> np.ndarray:
    out = np.asarray(a, dtype=float)
    out = out[np.isfinite(out)]
    return out


def _sorted_names(models: dict) -> list:
    """Return model names sorted by ascending test RMSE (best first)."""
    rows = []
    for name, r in models.items():
        if "test_metrics" not in r:
            continue
        rmse = r["test_metrics"].get("rmse", float("inf"))
        rows.append((rmse, name))
    rows.sort()
    return [n for _, n in rows]


def _collect_records(models: dict) -> list[dict]:
    out = []
    for name, r in models.items():
        if "test_metrics" not in r:
            continue
        vm = r.get("val_metrics", {}) or {}
        tm = r["test_metrics"]
        am = r.get("test_metrics_ar", {}) or {}
        out.append({
            "name": name,
            "category": r.get("category", ""),
            "val_r2": vm.get("r2"),
            "val_rmse": vm.get("rmse"),
            "test_r2": tm.get("r2"),
            "test_rmse": tm.get("rmse"),
            "test_mae": tm.get("mae"),
            "test_mape": tm.get("mape"),
            "test_smape": tm.get("smape"),
            "ar_r2": am.get("r2"),
            "ar_rmse": am.get("rmse"),
            "elapsed_s": r.get("elapsed_s", 0),
        })
    return out


def _try_import_pd():
    try:
        import pandas as pd
        return pd
    except ImportError:
        log.warning("  [plot] pandas unavailable — CSV export disabled")
        return None


def _try_import_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        log.warning("  [plot] matplotlib unavailable")
        return None


def _try_import_plotly():
    try:
        import plotly.graph_objects as go
        return go
    except ImportError:
        log.warning("  [plot] plotly unavailable")
        return None


def _try_import_sns():
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid", palette="muted")
        return sns
    except ImportError:
        log.warning("  [plot] seaborn unavailable")
        return None


# ══════════════════════════════════════════════════════════════════
# CSV export
# ══════════════════════════════════════════════════════════════════

def _export_csv(models: dict, y_val: np.ndarray, y_test: np.ndarray,
                csv_dir: Path) -> None:
    pd = _try_import_pd()
    if pd is None:
        return

    _safe_mkdir(csv_dir)

    # Summary
    rows = _collect_records(models)
    if rows:
        df = pd.DataFrame(rows).sort_values("test_rmse", na_position="last")
        try:
            df.to_csv(csv_dir / "summary_metrics.csv", index=False, encoding="utf-8")
            log.info(f"  [plot] summary_metrics.csv: {len(df)} rows")
        except Exception as e:
            log.warning(f"  [plot] summary CSV failed: {e}")

    # Per-model history (DL only — has _history)
    for name, r in models.items():
        hist = r.get("history")
        if not hist:
            continue
        try:
            dfh = pd.DataFrame(hist)
            safe = name.replace(" ", "_").replace("/", "_")
            dfh.to_csv(csv_dir / f"history_{safe}.csv", index=False, encoding="utf-8")
        except Exception as e:
            log.debug(f"  [plot] history CSV {name} failed: {e}")

    # Per-model predictions aligned with y_val / y_test
    y_val = np.asarray(y_val, dtype=float) if y_val is not None else None
    y_test = np.asarray(y_test, dtype=float) if y_test is not None else None
    for name, r in models.items():
        if "val_pred" not in r or "test_pred" not in r:
            continue
        try:
            vp = np.asarray(r["val_pred"], dtype=float)
            tp = np.asarray(r["test_pred"], dtype=float)
            rows2 = []
            if y_val is not None and len(y_val) == len(vp):
                for i in range(len(vp)):
                    rows2.append({"split": "val", "idx": i,
                                  "y_true": float(y_val[i]),
                                  "y_pred": float(vp[i])})
            if y_test is not None and len(y_test) == len(tp):
                for i in range(len(tp)):
                    rows2.append({"split": "test", "idx": i,
                                  "y_true": float(y_test[i]),
                                  "y_pred": float(tp[i])})
            if rows2:
                dfp = pd.DataFrame(rows2)
                safe = name.replace(" ", "_").replace("/", "_")
                dfp.to_csv(csv_dir / f"predictions_{safe}.csv",
                           index=False, encoding="utf-8")
        except Exception as e:
            log.debug(f"  [plot] predictions CSV {name} failed: {e}")


# ══════════════════════════════════════════════════════════════════
# matplotlib plots
# ══════════════════════════════════════════════════════════════════

def _plot_mpl(models: dict, y_val: np.ndarray, y_test: np.ndarray,
              out_dir: Path) -> None:
    plt = _try_import_mpl()
    if plt is None:
        return
    _safe_mkdir(out_dir)

    records = _collect_records(models)
    if not records:
        return

    # 1) Test R² bar (descending)
    try:
        recs = sorted(records, key=lambda x: (x["test_r2"] is None,
                                              -(x["test_r2"] or -999)))
        names = [r["name"] for r in recs]
        r2s = [r["test_r2"] or 0 for r in recs]
        fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.4), 5))
        colors = ["#2ecc71" if v >= 0.8 else "#3498db" if v >= 0.5
                  else "#e67e22" if v >= 0 else "#e74c3c" for v in r2s]
        ax.bar(range(len(names)), r2s, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
        ax.axhline(0, color="k", lw=0.5)
        ax.axhline(0.8, color="green", lw=0.7, ls="--", alpha=0.5, label="R²=0.8")
        ax.set_ylabel("Test R²")
        ax.set_title("Test R² by Model (descending)")
        ax.legend(loc="lower left", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "r2_bar.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        log.warning(f"  [plot] mpl r2_bar failed: {e}")

    # 2) Pred vs actual — G-356 (2026-06-25, 사용자 "전부다"): 옛 top-6([:6]) → 전 모델.
    #   예측(test_pred)이 있는 모든 모델에 pred_vs_actual 생성(논문/검토용 완전 커버). test_pred 결손은
    #   루프 안에서 skip. learning_curve(아래)는 epoch history 있는 모델(deep)만이라 이미 해당 전체.
    best = _sorted_names(models)
    y_test = np.asarray(y_test, dtype=float) if y_test is not None else None
    for name in best:
        r = models.get(name, {})
        if "test_pred" not in r or y_test is None:
            continue
        try:
            tp = np.asarray(r["test_pred"], dtype=float)
            if len(tp) != len(y_test):
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(y_test, label="actual", color="#2c3e50", lw=1.6)
            ax.plot(tp, label="predicted", color="#c0392b", lw=1.2, alpha=0.85)
            r2 = r.get("test_metrics", {}).get("r2", 0) or 0
            rmse = r.get("test_metrics", {}).get("rmse", 0) or 0
            ax.set_title(f"{name} — test split (R²={r2:.3f}, RMSE={rmse:.2f})")
            ax.set_xlabel("test index")
            ax.set_ylabel("ILI rate")
            ax.legend()
            fig.tight_layout()
            safe = name.replace(" ", "_").replace("/", "_")
            fig.savefig(out_dir / f"pred_vs_actual_{safe}.png",
                        dpi=120, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            log.debug(f"  [plot] mpl pred_vs_actual {name} failed: {e}")

    # 3) Learning curves (only models with history)
    for name, r in models.items():
        hist = r.get("history")
        if not hist:
            continue
        try:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            # split by seed if present
            seeds = sorted({h.get("seed") for h in hist if h.get("seed") is not None})
            if seeds:
                for s in seeds:
                    h_s = [h for h in hist if h.get("seed") == s]
                    ep = [h["epoch"] for h in h_s]
                    tr = [h.get("train_loss", np.nan) for h in h_s]
                    va = [h.get("val_loss", np.nan) for h in h_s]
                    ax.plot(ep, tr, alpha=0.35, lw=0.8, label=f"train seed={s}")
                    ax.plot(ep, va, alpha=0.85, lw=1.2, label=f"val seed={s}")
            else:
                ep = [h["epoch"] for h in hist]
                tr = [h.get("train_loss", np.nan) for h in hist]
                va = [h.get("val_loss", np.nan) for h in hist]
                ax.plot(ep, tr, label="train", alpha=0.7, color="#3498db")
                ax.plot(ep, va, label="val", color="#e74c3c", lw=1.4)
            ax.set_title(f"{name} — learning curve")
            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax.set_yscale("symlog", linthresh=1e-3)
            ax.legend(fontsize=7, loc="upper right", ncol=2)
            fig.tight_layout()
            safe = name.replace(" ", "_").replace("/", "_")
            fig.savefig(out_dir / f"learning_curve_{safe}.png",
                        dpi=120, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            log.debug(f"  [plot] mpl learning_curve {name} failed: {e}")

    log.info(f"  [plot] matplotlib PNGs → {out_dir}")


# ══════════════════════════════════════════════════════════════════
# plotly plots (interactive)
# ══════════════════════════════════════════════════════════════════

def _plot_plotly(models: dict, y_val: np.ndarray, y_test: np.ndarray,
                 out_dir: Path) -> None:
    go = _try_import_plotly()
    if go is None:
        return
    _safe_mkdir(out_dir)

    records = _collect_records(models)
    if not records:
        return

    # 1) R² bar
    try:
        recs = sorted(records, key=lambda x: (x["test_r2"] is None,
                                              -(x["test_r2"] or -999)))
        names = [r["name"] for r in recs]
        r2s = [r["test_r2"] or 0 for r in recs]
        colors = ["#2ecc71" if v >= 0.8 else "#3498db" if v >= 0.5
                  else "#e67e22" if v >= 0 else "#e74c3c" for v in r2s]
        fig = go.Figure(data=[go.Bar(x=names, y=r2s, marker_color=colors)])
        fig.update_layout(
            title="Test R² by Model (descending)",
            xaxis_title="model", yaxis_title="R²",
            xaxis={"tickangle": -60},
            template="plotly_white",
            height=500,
        )
        fig.add_hline(y=0.8, line_dash="dash", line_color="green",
                      annotation_text="R²=0.8")
        fig.write_html(str(out_dir / "r2_bar.html"))
    except Exception as e:
        log.warning(f"  [plot] plotly r2_bar failed: {e}")

    # 2) Pred vs actual (overlay — top 8 models)
    best = _sorted_names(models)[:8]
    y_test = np.asarray(y_test, dtype=float) if y_test is not None else None
    if y_test is not None:
        try:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=y_test, name="actual",
                                     line=dict(color="black", width=2)))
            for name in best:
                r = models.get(name, {})
                if "test_pred" not in r:
                    continue
                tp = np.asarray(r["test_pred"], dtype=float)
                if len(tp) != len(y_test):
                    continue
                r2 = r.get("test_metrics", {}).get("r2", 0) or 0
                fig.add_trace(go.Scatter(
                    y=tp, name=f"{name} (R²={r2:.3f})",
                    opacity=0.75, line=dict(width=1.2),
                ))
            fig.update_layout(
                title="Test split — actual vs top-8 predictions",
                xaxis_title="test index", yaxis_title="ILI rate",
                template="plotly_white", height=550,
                hovermode="x unified",
            )
            fig.write_html(str(out_dir / "pred_vs_actual_top8.html"))
        except Exception as e:
            log.warning(f"  [plot] plotly pred_vs_actual failed: {e}")

    # 3) Learning curves overlay (log scale, all DL)
    try:
        dl_models = [(n, r) for n, r in models.items() if r.get("history")]
        if dl_models:
            fig = go.Figure()
            for name, r in dl_models:
                hist = r["history"]
                seeds = sorted({h.get("seed") for h in hist
                               if h.get("seed") is not None})
                if seeds:
                    # plot the first seed only for plotly overlay legibility
                    s0 = seeds[0]
                    h_s = [h for h in hist if h.get("seed") == s0]
                    ep = [h["epoch"] for h in h_s]
                    va = [h.get("val_loss") for h in h_s]
                    fig.add_trace(go.Scatter(x=ep, y=va, name=f"{name} (seed {s0})",
                                             mode="lines"))
                else:
                    ep = [h["epoch"] for h in hist]
                    va = [h.get("val_loss") for h in hist]
                    fig.add_trace(go.Scatter(x=ep, y=va, name=name, mode="lines"))
            fig.update_layout(
                title="Learning curves — val loss (DL models)",
                xaxis_title="epoch", yaxis_title="val loss",
                yaxis_type="log", template="plotly_white", height=550,
            )
            fig.write_html(str(out_dir / "learning_curves_all.html"))
    except Exception as e:
        log.warning(f"  [plot] plotly learning_curves failed: {e}")

    log.info(f"  [plot] plotly HTMLs → {out_dir}")


# ══════════════════════════════════════════════════════════════════
# seaborn plots
# ══════════════════════════════════════════════════════════════════

def _plot_sns(models: dict, y_val: np.ndarray, y_test: np.ndarray,
              out_dir: Path) -> None:
    sns = _try_import_sns()
    plt = _try_import_mpl()
    pd = _try_import_pd()
    if sns is None or plt is None or pd is None:
        return
    _safe_mkdir(out_dir)

    records = _collect_records(models)
    if not records:
        return
    df = pd.DataFrame(records)

    # 1) R² bar with category hue
    try:
        df_plot = df.dropna(subset=["test_r2"]).sort_values("test_r2", ascending=False)
        fig, ax = plt.subplots(figsize=(max(8, len(df_plot) * 0.4), 5.5))
        sns.barplot(data=df_plot, x="name", y="test_r2",
                    hue="category", dodge=False, ax=ax, palette="Set2",
                    legend=True)
        ax.axhline(0.8, color="green", ls="--", alpha=0.6, label="R²=0.8")
        ax.set_title("Test R² — seaborn view (category-hued)")
        # Use tick positions as anchor to avoid FixedLocator warning
        ax.set_xticks(range(len(df_plot)))
        ax.set_xticklabels(df_plot["name"].tolist(), rotation=60,
                           ha="right", fontsize=8)
        ax.legend(loc="lower left", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "r2_bar.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        log.warning(f"  [plot] sns r2_bar failed: {e}")

    # 2) Metrics heatmap (normalized)
    try:
        mdf = df[["name", "test_r2", "test_rmse", "test_mae", "test_mape"]].copy()
        mdf = mdf.dropna(subset=["test_r2"]).sort_values("test_rmse")
        mdf = mdf.set_index("name")
        # normalize per-column for comparability (higher=worse except r2)
        norm = mdf.copy()
        norm["test_r2"] = 1 - (mdf["test_r2"] - mdf["test_r2"].min()) / max(
            mdf["test_r2"].max() - mdf["test_r2"].min(), 1e-9)
        for col in ["test_rmse", "test_mae", "test_mape"]:
            if col in norm and norm[col].notna().any():
                lo, hi = norm[col].min(), norm[col].max()
                norm[col] = (norm[col] - lo) / max(hi - lo, 1e-9)
        fig, ax = plt.subplots(figsize=(7, max(4, len(norm) * 0.25)))
        sns.heatmap(norm, cmap="RdYlGn_r", annot=mdf.round(3),
                    fmt="", cbar_kws={"label": "normalized worse→0 better"},
                    ax=ax)
        ax.set_title("Test metrics — heatmap (lower = better for RMSE/MAE/MAPE)")
        fig.tight_layout()
        fig.savefig(out_dir / "metrics_heatmap.png",
                    dpi=120, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        log.warning(f"  [plot] sns metrics_heatmap failed: {e}")

    # 3) Residual distribution (test split, top-6 models)
    y_test = np.asarray(y_test, dtype=float) if y_test is not None else None
    if y_test is not None:
        try:
            rows = []
            for name in _sorted_names(models)[:6]:
                r = models.get(name, {})
                if "test_pred" not in r:
                    continue
                tp = np.asarray(r["test_pred"], dtype=float)
                if len(tp) != len(y_test):
                    continue
                for resid in (y_test - tp):
                    rows.append({"model": name, "residual": float(resid)})
            if rows:
                rdf = pd.DataFrame(rows)
                model_order = list(rdf["model"].unique())
                fig, ax = plt.subplots(figsize=(9, 5))
                sns.violinplot(data=rdf, x="model", y="residual",
                               hue="model", legend=False, ax=ax,
                               palette="Set3", order=model_order)
                ax.axhline(0, color="k", lw=0.8)
                ax.set_xticks(range(len(model_order)))
                ax.set_xticklabels(model_order, rotation=45,
                                   ha="right", fontsize=9)
                ax.set_title("Test residual distribution — top 6 models")
                fig.tight_layout()
                fig.savefig(out_dir / "residual_violin.png",
                            dpi=120, bbox_inches="tight")
                plt.close(fig)
        except Exception as e:
            log.warning(f"  [plot] sns residual_violin failed: {e}")

    log.info(f"  [plot] seaborn PNGs → {out_dir}")


# ══════════════════════════════════════════════════════════════════
# public entry point
# ══════════════════════════════════════════════════════════════════

def generate_all(runner_result: dict,
                 y_val: Optional[np.ndarray],
                 y_test: Optional[np.ndarray],
                 output_root: str,
                 tag: str = "phase4") -> dict:
    """Produce CSVs + 3-library plots at ``{output_root}/plots_*`` and
    ``{output_root}/csv``. Returns a manifest dict of written files.

    Parameters
    ----------
    runner_result : dict returned by MultiModelRunner.run().
    y_val, y_test : ground-truth arrays (original scale) from phase1.
    output_root   : base directory (usually ``config.save_dir``). Will
                    create subfolders if missing.
    tag           : label added to the manifest for traceability
                    (e.g. "phase4_baseline" vs "phase5_external").
    """
    try:
        indiv = runner_result.get("individual_results", {}) or {}
        ens = runner_result.get("ensemble_results", {}) or {}
        # merge with ensemble-first-wins conflict resolution
        models = {**indiv, **ens}
        if not models:
            log.warning(f"  [plot] no models in runner_result — skipping")
            return {"tag": tag, "status": "empty"}

        root = Path(output_root)
        csv_dir = root / "csv"
        mpl_dir = root / "plots_matplotlib"
        ply_dir = root / "plots_plotly"
        sns_dir = root / "plots_seaborn"

        _export_csv(models, y_val, y_test, csv_dir)
        _plot_mpl(models, y_val, y_test, mpl_dir)
        _plot_plotly(models, y_val, y_test, ply_dir)
        _plot_sns(models, y_val, y_test, sns_dir)

        manifest = {
            "tag": tag,
            "status": "ok",
            "output_root": str(root),
            "csv_dir": str(csv_dir),
            "mpl_dir": str(mpl_dir),
            "plotly_dir": str(ply_dir),
            "seaborn_dir": str(sns_dir),
            "n_models": len(models),
        }
        try:
            (root / f"plot_manifest_{tag}.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            log.debug(f"  [plot] manifest write failed: {e}")
        return manifest
    except Exception as e:
        log.warning(f"  [plot] generate_all failed: {e}")
        return {"tag": tag, "status": "error", "error": str(e)}
