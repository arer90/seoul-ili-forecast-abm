"""
R12: Comprehensive Evaluation Aggregator
==============================================

Master phase that consolidates R2~R9 outputs into:

  • **Master CSV** — model × metric × phase × horizon × regime grid
  • **Per-model deep-dive MD** — one comprehensive report per model
    (forecast trajectory, residual diagnostics, calibration, SHAP top-15,
     PIT histogram, peak metrics, alert performance, statistical tests)
  • **Statistical comparison tables** — DM (R6), Hansen SPA (R10),
    pairwise tournament (R10), McNemar (test slab)
  • **Visualization scripts** — forest plot of WIS with bootstrap CI,
    heatmap of model × metric, calibration curve overlay, horizon-decay plot
  • **Audit pointer** — links to {run_id}_audit.json for full reproducibility

This is the single document an MPH-thesis reviewer / IRB defender opens to
see "what did this pipeline actually do, what did it find, and why is it
defensible". All outputs are machine-readable (CSV/JSON) plus
human-readable (MD + PNG).

CLI: enabled by default; opt-out via `--no-comprehensive-eval`.
Heavy plotting requires matplotlib (auto-fallback to no-plots if missing).

Output:
  simulation/results/comprehensive_eval/
    ├── MASTER_GRID.csv              ← cube view: rows = model × metric × phase
    ├── ranking_consolidated.json    ← cross-phase ranking with uncertainty
    ├── per_model/<model>.md         ← one deep-dive per model
    ├── tables/                      ← statistical comparison tables
    │   ├── dm_pvalues.csv
    │   ├── pairwise_relative_wis.csv
    │   ├── hansen_spa.json
    │   └── horizon_decay.csv
    ├── figures/                     ← matplotlib outputs
    │   ├── forest_plot_wis.png
    │   ├── heatmap_model_x_metric.png
    │   ├── calibration_curve.png
    │   └── horizon_decay.png
    └── REPORT.md                    ← top-level summary linking everything
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def _load_per_model_metrics(out_dir_phase11: Path) -> list[dict]:
    """Read R10's per-model CSV if it exists."""
    csv_path = out_dir_phase11 / "per_model_metrics.csv"
    if not csv_path.exists():
        return []
    import csv as _csv
    with csv_path.open(encoding="utf-8") as f:
        return list(_csv.DictReader(f))


def _build_master_grid(all_results: dict, out_dir: Path) -> Path:
    """One row per (phase, model, metric, horizon, slab) tuple. Long format
    for downstream pivot / plotting."""
    import csv as _csv
    rows = []

    def _push(phase: str, model: str, metric: str, value, **dims):
        if value is None:
            return
        try:
            v = float(value)
        except (TypeError, ValueError):
            v = None
        rec = {
            "phase": phase, "model": model, "metric": metric,
            "value": v, "value_str": str(value),
            **dims,
        }
        rows.append(rec)

    # ── Defensive helpers — every phase returns slightly different shapes,
    #    and some entries are scalar / string / None instead of dict.
    def _safe_dict(x):
        return x if isinstance(x, dict) else {}
    def _push_metrics(phase: str, model: str, mr, slab: str):
        if not isinstance(mr, dict):
            return
        for k, v in mr.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                _push(phase, model, k, v, slab=slab)
            elif isinstance(v, dict):
                # Flatten one level for nested {peak_week: {abs_weeks: ...}} etc.
                for kk, vv in v.items():
                    if isinstance(vv, (int, float, np.integer, np.floating)):
                        _push(phase, model, f"{k}.{kk}", vv, slab=slab)

    # R2 baseline — runner.run() returns nested
    # {individual_results: {model: {...}}, ensemble_results: {...}, ...}
    bl = _safe_dict(_safe_dict(all_results.get("baseline")).get("model_results"))
    bl_flat: dict = {}
    for k in ("individual_results", "ensemble_results"):
        inner = _safe_dict(bl.get(k))
        bl_flat.update(inner)
    if not bl_flat:
        # Fallback: legacy flat shape {model: {r2: ..., ...}}
        bl_flat = {k: v for k, v in bl.items()
                   if isinstance(v, dict) and "test_metrics" not in v
                   and any(isinstance(v.get(mk), (int, float)) for mk in ("r2", "rmse", "mae"))}
    for m, mr in bl_flat.items():
        # Each model's `test_metrics` is the right place for r2/rmse/mae
        if isinstance(mr, dict):
            tm = _safe_dict(mr.get("test_metrics"))
            if tm:
                _push_metrics("baseline", m, tm, slab="test_baseline")
            # phase 8 AR_correction RETIRED (2026-06-05) — no AR variant push
            #   (stale test_metrics_ar from baseline sidecar no longer surfaced)

    # R4 WF-CV — actual key is 'wf_results' (model_results is R2's)
    wf_root = _safe_dict(all_results.get("wfcv"))
    wf = _safe_dict(wf_root.get("wf_results") or wf_root.get("model_results"))
    for m, mr in wf.items():
        if isinstance(mr, dict):
            # Nested 'overall_metrics' is the R4 convention
            om = _safe_dict(mr.get("overall_metrics"))
            if om:
                _push_metrics("wfcv", m, om, slab="oof")
            else:
                _push_metrics("wfcv", m, mr, slab="oof")

    # R7 PI
    pi_root = _safe_dict(all_results.get("prediction_intervals"))
    # R7 returns {pi_results: {model: {...}}, ...}
    pi = _safe_dict(pi_root.get("pi_results")) or pi_root
    for m, mr in pi.items():
        _push_metrics("prediction_intervals", m, mr, slab="conformal_holdout")

    # R8 scoring — actual key is 'scores' (not 'composite_scores')
    sc_root = _safe_dict(all_results.get("scoring"))
    sc = _safe_dict(sc_root.get("scores"))
    for m, mr in sc.items():
        _push_metrics("scoring", m, mr, slab="oof")

    # R12 decoupled from P1 (2026-06-20, 사용자 지시): comprehensive report sources its
    # champion + families from R9 (per_model_optimize) + R10 (per_model_eval) below — NOT
    # from P1's real-slab. P1/real_eval now runs AFTER R12 (production-track start), so
    # all_results["real_eval"] is intentionally absent here.

    # R10 per-model uniform (per_model_eval — families, test-slab)
    p11 = all_results.get("per_model_eval", {}) or {}
    if isinstance(p11, dict) and not p11.get("skipped"):
        out_p11 = Path(p11.get("metrics_csv") or "").parent if p11.get("metrics_csv") else Path()
        if out_p11.exists():
            try:
                p11_rows = _load_per_model_metrics(out_p11)
                for r in p11_rows:
                    m = r.get("model", "")
                    for k, v in r.items():
                        if k in ("model",): continue
                        _push("per_model_eval", m, k, v, slab="test")
            except Exception:
                pass

    # R9 per-model individual optimization
    p12 = all_results.get("per_model_optimize", {}) or {}
    if isinstance(p12, dict) and not p12.get("skipped"):
        configs = p12.get("per_model_configs", {}) or {}
        for m, c in configs.items():
            best = c.get("best_metrics", {}) if isinstance(c, dict) else {}
            for k, v in best.items():
                _push("per_model_optimize", m, f"best_{k}", v, slab="val_search")
            cfg = c.get("best_config", {}) if isinstance(c, dict) else {}
            _push("per_model_optimize", m, "best_transform", cfg.get("transform"),
                   slab="val_search")
            _push("per_model_optimize", m, "best_scaler", cfg.get("scaler"),
                   slab="val_search")

    # Persist
    out = out_dir / "MASTER_GRID.csv"
    if rows:
        cols = sorted({k for r in rows for k in r})
        with out.open("w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    return out


def _ranking_consolidated(all_results: dict, out_dir: Path) -> dict:
    """Cross-phase ranking with uncertainty.

    Consolidated Borda ranking is sourced ONLY from OOF-skill + proper-scoring
    rankings, consistent with the pure best-WIS champion policy (A4/M7):
      - R4 OOF rank (R² descending)
      - R10 test-slab WIS rank
      - R10 pairwise tournament rank
      - Hansen SPA p-values (reported, not Borda-summed)

    The Phase-11 ad-hoc composite (R²+RMSE+DM+stability+conformal weighted blend)
    is retained under ``diagnostics`` only — it is NOT folded into the Borda
    consolidation, because that weighted blend contradicts the project's
    best-WIS champion (``per_model_eval.py:77``) and a reviewer would flag the
    self-inconsistency of ranking on a composite the methods section disowns.
    """
    out: dict = {"sources": {}, "diagnostics": {}, "consolidated": []}  # consolidated MUST be list (sliced [:15] in REPORT.md)

    wf_root = all_results.get("wfcv") if isinstance(all_results.get("wfcv"), dict) else {}
    wf = (wf_root.get("wf_results") or wf_root.get("model_results")
          if isinstance(wf_root, dict) else {}) or {}
    if isinstance(wf, dict) and wf:
        def _r2(node):
            if not isinstance(node, dict): return -np.inf
            if isinstance(node.get("r2"), (int, float)): return float(node["r2"])
            om = node.get("overall_metrics")
            if isinstance(om, dict) and isinstance(om.get("r2"), (int, float)):
                return float(om["r2"])
            return -np.inf
        order = sorted(wf.keys(), key=lambda m: -_r2(wf.get(m)))
        out["sources"]["phase7_oof_r2"] = order

    sc_root = all_results.get("scoring") if isinstance(all_results.get("scoring"), dict) else {}
    # R8 returns 'scores' dict {model: {composite: 1.0, ...}}
    sc = sc_root.get("scores") if isinstance(sc_root, dict) else {}
    sc = sc if isinstance(sc, dict) else {}
    if sc:
        def _comp(node):
            if isinstance(node, dict) and isinstance(node.get("composite"), (int, float)):
                return float(node["composite"])
            if isinstance(node, (int, float)): return float(node)
            return -np.inf
        order = sorted(sc.keys(), key=lambda m: -_comp(sc.get(m)))
        # A4 (M7): diagnostic only — NOT a Borda source. The composite blend
        # contradicts the best-WIS champion, so it must not move the consolidated
        # ranking; it stays here for inspection / Phase-11 continuity.
        out["diagnostics"]["composite_order"] = order

    p11 = all_results.get("per_model_eval", {}) or {}
    if isinstance(p11, dict) and p11.get("ranking_top10"):
        out["sources"]["phase11_wis"] = p11["ranking_top10"]
    if isinstance(p11, dict) and p11.get("pairwise_relative_wis"):
        rwis = p11["pairwise_relative_wis"]
        order = sorted(rwis.keys(),
                       key=lambda m: rwis.get(m, float("inf")))
        out["sources"]["phase11_pairwise"] = order

    if isinstance(p11, dict) and p11.get("spa_test"):
        out["spa_test"] = p11["spa_test"]

    # Borda-count consolidation across the available rankings
    # Lower position = better; sum positions across rankings; output is
    # the model with smallest summed position.
    all_models: set = set()
    for lst in out["sources"].values():
        if isinstance(lst, list):
            all_models.update(lst)
    if all_models:
        scores = {m: 0.0 for m in all_models}
        for lst in out["sources"].values():
            if not isinstance(lst, list):
                continue
            for pos, m in enumerate(lst, 1):
                scores[m] += pos
        consolidated = sorted(scores.items(), key=lambda kv: kv[1])
        out["consolidated"] = [{"model": m, "borda_score": s} for m, s in consolidated]

    (out_dir / "ranking_consolidated.json").write_text(
        json.dumps(out, indent=2, default=str)
    , encoding="utf-8")
    return out


def _per_model_deep_dive(model: str, all_results: dict, out_dir: Path) -> Path:
    """모델별 종합 evaluation MD 생성 (G-163, D-4 deep module).

    R2~R9 의 모든 결과를 단일 모델 관점으로 통합. 사용자 thesis appendix
    또는 모델 비교 시 참조용 single source-of-truth.

    Args:
        model: model name (e.g., "XGBoost", "DNN-Conformer"). per_model_optimal/
               + R10 metrics_csv 에서 lookup.
        all_results: pipeline 전체 결과 dict (`runner.run()` 반환). 다음 키 포함:
                     - baseline (R2): individual_results / ensemble_results
                     - wfcv (R4): wf_results / overall_metrics
                     - prediction_intervals (R7): conformal PI
                     - feature_importance (R11): SHAP top features
                     - per_model_eval (R10): test-slab uniform (52 metric)
                     - per_model_optimize (R9): per-model optimal config
                     (real_eval/P1 디커플됨, 2026-06-20 — R12는 R9/R10 챔피언·family만 소비;
                      P1은 R12 뒤 production-track 시작에서 실행)
        out_dir: phase16 output dir. `out_dir/per_model/<model>.md` 에 저장.

    Returns:
        Path — 생성된 MD path (out_dir/per_model/<model>.md).

    Raises:
        절대 raise X — phase 별 결과 결손 시 graceful skip.

    Performance: O(n_metrics + n_features) — 1 모델당 ~100ms (52 metric × 25 features).
    Side effects:
        - file write: out_dir/per_model/<model>.md (~5-15KB)
        - dir create: out_dir/per_model/ (없으면)

    Caller responsibility:
        - all_results 가 phase 별 결과 dict 형식 (else graceful skip).
        - out_dir 쓰기 권한.

    Output format (markdown sections):
        - R2 baseline (R²/MAE/RMSE)
        - R4 WF-CV (OOF R²/MAE)
        - R7 conformal PI (95% coverage / width)
        - R11 SHAP top 15 features
        - R9 real-slab nowcast
        - R10 test-slab UNIFORM evaluation (26 핵심 metric + quality emoji)
        - R9 best (transform × scaler) + search grid

    See: G-163 (R12 figures + deep-dive 4종 docstring 약속 이행),
         G-168 (26 metric 보존), `metric_rubric.RUBRIC` (quality threshold).
    """
    def _d(x):
        return x if isinstance(x, dict) else {}

    md = [
        f"# {model} — Comprehensive Evaluation",
        "",
    ]

    # R2 baseline — drill into individual_results / ensemble_results
    bl_root = _d(_d(all_results.get("baseline")).get("model_results"))
    bl = _d(_d(bl_root.get("individual_results")).get(model)) \
         or _d(_d(bl_root.get("ensemble_results")).get(model)) \
         or _d(bl_root.get(model))   # legacy flat
    if bl and isinstance(bl.get("test_metrics"), dict):
        bl_metrics = bl["test_metrics"]
    else:
        bl_metrics = bl  # legacy or fallback
    if bl_metrics:
        md += [
            "## R2 — Baseline (test-slab on first split)",
            "",
            f"- R² = {bl_metrics.get('r2', float('nan'))}",
            f"- MAE = {bl_metrics.get('mae', float('nan'))}",
            f"- RMSE = {bl_metrics.get('rmse', float('nan'))}",
            "",
        ]

    # R4 WF-CV (key is 'wf_results', not 'model_results')
    wf_root = _d(all_results.get("wfcv"))
    wf_pm = _d(wf_root.get("wf_results") or wf_root.get("model_results")).get(model)
    wf = _d(wf_pm)
    om = _d(wf.get("overall_metrics")) or wf
    if om:
        md += [
            "## R4 — Walk-Forward CV (OOF)",
            "",
            f"- R² = {om.get('r2', float('nan'))}",
            f"- MAE = {om.get('mae', float('nan'))}",
            f"- folds = {wf.get('n_folds', om.get('n_folds', '?'))}",
            "",
        ]

    # R7 PI (root has pi_results dict)
    pi_root = _d(all_results.get("prediction_intervals"))
    pi = _d(_d(pi_root.get("pi_results")).get(model)) or _d(pi_root.get(model))
    if pi:
        md += [
            "## R7 — Conformal PI",
            "",
            f"- 95% PI coverage = {pi.get('coverage', float('nan'))}",
            f"- 95% PI width = {pi.get('width', float('nan'))}",
            f"- regime breakdown = {pi.get('regime_coverage', '?')}",
            "",
        ]

    # R11 SHAP
    fi = _d(all_results.get("feature_importance"))
    if not fi.get("error"):
        # AUDIT 2026-06-01: shap 출력은 model_importance/shap_analysis = {model: [{feature,score}]}.
        #   옛 코드 fi[model]["top_features"] 는 미존재 키 → SHAP 섹션 영구 공백이었음. 실제 키로 정정.
        _mi = (_d(fi.get("model_importance")).get(model)
               or _d(fi.get("shap_analysis")).get(model) or [])
        top = [(it.get("feature"), it.get("score")) for it in _mi[:15] if isinstance(it, dict)]
        if top:
            md += [
                "## R11 — feature importance top 15 (descriptive only)",
                "",
                "| feature | importance |",
                "|---------|-----------:|",
            ]
            for f, v in top:
                try:
                    md.append(f"| {f} | {float(v):.4f} |")
                except (TypeError, ValueError):
                    md.append(f"| {f} | {v} |")
            md.append("")

    # (R12 decoupled from P1, 2026-06-20 — per-model real-slab section removed; P1/real_eval
    #  now runs AFTER R12 so real_eval is absent here. Operational real-slab lives with P1.)

    # R10 per-model uniform (per_model_eval) — with quality tags
    try:
        from simulation.analytics.metric_rubric import RUBRIC, quality_emoji
    except Exception:
        RUBRIC, quality_emoji = {}, lambda q: "?"
    p11 = _d(all_results.get("per_model_eval"))
    if p11.get("metrics_csv"):
        try:
            with open(p11["metrics_csv"], encoding="utf-8") as f:
                import csv as _csv
                rows = list(_csv.DictReader(f))
                row = next((r for r in rows if r.get("model") == model), None)
                if row:
                    def _flag(metric_key):
                        rule = RUBRIC.get(metric_key)
                        v = row.get(metric_key)
                        if not rule or v in (None, "", "nan"): return ""
                        try:
                            q = rule.quality(float(v))
                        except (TypeError, ValueError):
                            return ""
                        return f" {quality_emoji(q)} ({q})"
                    md += [
                        "## R10 — Test-slab UNIFORM evaluation (n=68, primary inferential)",
                        "",
                        "### Forecasting metrics with quality tags",
                        "",
                        "| metric | value | quality | rule (excellent / good / acceptable) |",
                        "|---|---|---|---|",
                    ]
                    for k in ("r2", "mae", "rmse", "mape", "smape", "mdape",
                             "mase_h1", "mase_h52", "wis", "log_wis",
                             "crps_gaussian", "pinball_q50",
                             "pit_mean", "pit_ks_p",
                             "pi95_coverage", "pi80_coverage", "pi50_coverage",
                             "direction_acc", "peak_week_err", "peak_int_relerr",
                             "alert_f1", "brier_score", "brier_skill",
                             "sensitivity", "specificity",
                             "relative_wis_pairwise"):
                        rule = RUBRIC.get(k)
                        v = row.get(k)
                        if v in (None, "", "nan") or rule is None:
                            continue
                        try:
                            q = rule.quality(float(v))
                        except (TypeError, ValueError):
                            q = "n/a"
                        if rule.direction == "calibration":
                            tgt = rule.excellent
                            rng = (f"|x−{tgt}| ≤ {rule.good} / "
                                    f"≤ {rule.acceptable} / else poor")
                        else:
                            arrow = "≤" if rule.direction == "lower" else "≥"
                            rng = (f"{arrow} {rule.excellent} / "
                                    f"{arrow} {rule.good} / "
                                    f"{arrow} {rule.acceptable}")
                        md.append(
                            f"| **{rule.name}** | `{v}` | "
                            f"{quality_emoji(q)} {q} | {rng} |"
                        )
                    md += [
                        "",
                        f"- MAE 95% BCa CI: ({row.get('mae_ci95_lo','?')}, {row.get('mae_ci95_hi','?')})",
                        f"- 95% PI Wilson CI: ({row.get('pi95_ci_lo','?')}, {row.get('pi95_ci_hi','?')})",
                        f"- σ (in-sample residual): {row.get('sigma_in_sample','?')}",
                        f"- WIS rank: {row.get('rank_wis','?')}",
                        f"- log-WIS rank: {row.get('rank_log_wis','?')}",
                        "",
                    ]
        except Exception:
            pass

    # R9 per-model individual optimization
    p12 = _d(all_results.get("per_model_optimize"))
    if p12.get("per_model_configs"):
        cfg = _d(p12["per_model_configs"].get(model))
        if cfg:
            best = _d(cfg.get("best_config"))
            metrics = _d(cfg.get("best_metrics"))
            md += [
                "## R9 — Individual optimization",
                "",
                f"- Best transform = `{best.get('transform', '?')}`",
                f"- Best scaler = `{best.get('scaler', '?')}`",
                f"- # features used = {best.get('n_features', '?')}",
                f"- Validation WIS at optimum = {metrics.get('wis', '?'):.3f}"
                  if isinstance(metrics.get('wis'), (int, float)) else
                f"- Validation WIS at optimum = {metrics.get('wis', '?')}",
                "",
            ]
            grid = cfg.get("optuna_trial_results", cfg.get("search_grid", []))  # search_grid = legacy key
            if grid:
                md += [
                    "### Optuna trial results (preproc)",
                    "",
                    "| transform | scaler | WIS | MAE |",
                    "|-----------|--------|-----|-----|",
                ]
                for cell in grid[:20]:
                    try:
                        md.append(
                            f"| {cell.get('transform','?')} | {cell.get('scaler','?')} | "
                            f"{float(cell.get('wis', float('nan'))):.3f} | "
                            f"{float(cell.get('mae', float('nan'))):.3f} |"
                        )
                    except (TypeError, ValueError):
                        pass
                md.append("")

    md += [
        "---",
        "",
        "*Generated by R12 comprehensive evaluator. Cross-references:*",
        "- Master grid: `MASTER_GRID.csv`",
        "- Stat comparisons: `tables/`",
        "- Audit metadata: `simulation/results/eval_logs/{run_id}_audit.json`",
    ]
    path = out_dir / "per_model" / f"{model}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(md), encoding="utf-8")
    return path


def _stat_comparison_tables(all_results: dict, out_dir: Path) -> dict:
    """Statistical comparison tables (DM, Hansen SPA, pairwise tournament)."""
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out: dict = {}

    # R6 DM tests
    dm = all_results.get("dm_tests", {}) or {}
    dm_rows = []
    if isinstance(dm, dict):
        regimes = dm.get("dm_tests_by_regime") or {"global": dm.get("dm_tests", {})}
        for regime, pairs in regimes.items() if isinstance(regimes, dict) else []:
            if not isinstance(pairs, dict):
                continue
            for pair_key, stats in pairs.items() if isinstance(pairs, dict) else []:
                if not isinstance(stats, dict):
                    continue
                dm_rows.append({
                    "regime": regime, "pair": pair_key,
                    "stat": stats.get("stat"),
                    "pvalue": stats.get("pvalue") or stats.get("p"),
                })
    if dm_rows:
        import csv as _csv
        with (tables_dir / "dm_pvalues.csv").open("w", newline="", encoding="utf-8") as f:
            cols = sorted({k for r in dm_rows for k in r})
            w = _csv.DictWriter(f, fieldnames=cols); w.writeheader()
            for r in dm_rows: w.writerow(r)
        out["dm_pvalues"] = str(tables_dir / "dm_pvalues.csv")

    # R10 pairwise relative WIS + Hansen SPA
    p11 = all_results.get("per_model_eval", {}) or {}
    if isinstance(p11, dict):
        rel = p11.get("pairwise_relative_wis", {})
        if rel:
            import csv as _csv
            with (tables_dir / "pairwise_relative_wis.csv").open("w", newline="", encoding="utf-8") as f:
                w = _csv.writer(f)
                w.writerow(["model", "relative_wis_pairwise"])
                for m, v in sorted(rel.items(), key=lambda x: x[1]):
                    w.writerow([m, v])
            out["pairwise_relative_wis"] = str(tables_dir / "pairwise_relative_wis.csv")
        spa = p11.get("spa_test", {})
        if spa:
            (tables_dir / "hansen_spa.json").write_text(
                json.dumps(spa, indent=2, default=str)
            , encoding="utf-8")
            out["hansen_spa"] = str(tables_dir / "hansen_spa.json")

    return out


def _figures(all_results: dict, out_dir: Path) -> dict:
    """R12 종합 figure 4종 생성 (G-163, D-4 deep module).

    R10 의 `per_model_metrics.csv` (52 metric × N 모델) 를 종합 시각화.
    matplotlib 미설치 시 graceful skip (학습 영향 X). 각 figure 는 독립 try-except
    (1개 실패 시 다른 3개는 계속).

    Args:
        all_results: pipeline 전체 결과 dict. 다음 키 필요:
                     - per_model_eval (R10): {metrics_csv: str path}
        out_dir: phase16 output dir. `out_dir/figures/` 에 PNG 저장.

    Returns:
        dict — 생성된 figure 의 path mapping:
          - forest_plot_wis: str (top 20 모델 의 WIS forest plot + MAE 95% CI bar)
          - heatmap_model_x_metric: str (z-score normalized, green=better, 9 metric)
          - calibration_curve: str (PI coverage vs nominal, top 20 모델)
          - horizon_decay: str (MASE h=1/4/13/52 4-horizon decay, top 15)
        키 누락 = 해당 figure 생성 실패 (matplotlib import / data 결손).

    Raises:
        절대 raise X — 각 figure try-except 로 graceful skip.

    Performance:
        - matplotlib import ~1초 (cold)
        - 4 figure 총 ~2-5초 (n_models ≤ 30)
        - 각 figure 24-67 KB PNG

    Side effects:
        - file write: out_dir/figures/<figure>.png × 4
        - dir create: out_dir/figures/ (없으면)
        - log.warning: matplotlib 미설치 시
        - log.debug: 각 figure 실패 시

    Caller responsibility:
        - all_results 의 per_model_eval.metrics_csv 가 valid path.
        - matplotlib 설치 (요구사항 X — graceful skip).

    Example:
        >>> from pathlib import Path
        >>> r = _figures(all_results, Path("simulation/results/comprehensive_eval"))
        >>> sorted(r.keys())
        ['calibration_curve', 'forest_plot_wis', 'heatmap_model_x_metric', 'horizon_decay']

    See: G-163 (4 figure docstring 약속 이행, 이전 forest 1개만 코드 — 거짓 docstring 해소).
    """
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out: dict = {}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as _np
    except ImportError:
        log.warning("  [phase16] matplotlib unavailable — skipping figures")
        return out

    # phase14 metrics_csv 한 번 로드 후 4 figure 가 공유
    p11 = all_results.get("per_model_eval", {}) or {}
    rows: list[dict] = []
    if isinstance(p11, dict) and p11.get("metrics_csv"):
        try:
            import csv as _csv
            with open(p11["metrics_csv"], encoding="utf-8") as f:
                rows = list(_csv.DictReader(f))
        except Exception as e:
            log.debug(f"  [phase16] metrics_csv load failed: {e}")

    # ── 1. Forest plot of WIS with MAE 95% CI (test slab) ───────────
    if rows:
        try:
            rows_with_wis = [r for r in rows
                             if r.get("wis") not in (None, "", "nan")]
            rows_with_wis.sort(key=lambda r: float(r["wis"]))
            top = rows_with_wis[:20]
            if top:
                fig, ax = plt.subplots(figsize=(7, max(4, len(top) * 0.3)))
                ys = list(range(len(top)))
                wis = [float(r["wis"]) for r in top]
                lo = [float(r.get("mae_ci95_lo", "0") or "0") for r in top]
                hi = [float(r.get("mae_ci95_hi", "0") or "0") for r in top]
                xerr = [[w - l if w > l else 0 for w, l in zip(wis, lo)],
                        [h - w if h > w else 0 for w, h in zip(wis, hi)]]
                ax.errorbar(wis, ys, xerr=xerr, fmt="o", capsize=3, color="#0f766e")
                ax.set_yticks(ys)
                ax.set_yticklabels([r["model"][:25] for r in top], fontsize=7)
                ax.invert_yaxis()
                ax.set_xlabel("WIS (test slab) — lower is better")
                ax.set_title("Forest plot — top 20 models (test slab WIS, MAE CI bars)")
                ax.grid(axis="x", alpha=0.3)
                fig.tight_layout()
                f = fig_dir / "forest_plot_wis.png"
                fig.savefig(f, dpi=120); plt.close(fig)
                out["forest_plot_wis"] = str(f)
        except Exception as e:
            log.debug(f"  [phase16] forest plot failed: {e}")

    # ── 2. Heatmap of model × metric (z-score normalized per metric) ─
    # G-163: docstring 약속의 heatmap 구현. 행=model, 열=metric (low-better:
    # WIS/MAE/RMSE/MAPE 부호 뒤집어서 high-better 통일 → z-score → diverging cmap).
    if rows:
        try:
            # Pick a stable subset of metrics that all 4 model categories produce
            metric_cols = [
                ("r2", True),                # higher = better
                ("mae", False), ("rmse", False), ("mape", False), ("smape", False),
                ("wis", False), ("crps_gaussian", False),
                ("pi95_coverage", "calib"),  # closer to 0.95 = better
                ("direction_acc", True),
            ]
            valid_rows = []
            mat = []
            row_names = []
            for r in rows:
                row_vec = []
                ok = True
                for col, _ in metric_cols:
                    v = r.get(col)
                    if v in (None, "", "nan"):
                        ok = False; break
                    try:
                        row_vec.append(float(v))
                    except (TypeError, ValueError):
                        ok = False; break
                if ok:
                    valid_rows.append(r)
                    mat.append(row_vec)
                    row_names.append(r.get("model", "?"))
            if mat:
                M = _np.array(mat, dtype=float)
                # Normalize each column to z-score with sign so higher = better.
                # For "calib" treat as |x − 0.95| (lower is better, then sign flip).
                M_norm = _np.zeros_like(M)
                for j, (_, dirn) in enumerate(metric_cols):
                    col = M[:, j]
                    if dirn == "calib":
                        col_eff = -_np.abs(col - 0.95)
                    elif dirn is True:
                        col_eff = col
                    else:
                        col_eff = -col
                    sd = _np.std(col_eff)
                    M_norm[:, j] = (col_eff - col_eff.mean()) / sd if sd > 0 else 0.0

                # Sort rows by mean z-score (best on top)
                order = _np.argsort(-_np.nanmean(M_norm, axis=1))
                M_norm = M_norm[order]
                row_names_sorted = [row_names[i] for i in order]

                fig, ax = plt.subplots(figsize=(8, max(3, len(row_names_sorted) * 0.25 + 1)))
                im = ax.imshow(M_norm, cmap="RdYlGn", aspect="auto",
                               vmin=-2, vmax=2)
                ax.set_xticks(range(len(metric_cols)))
                ax.set_xticklabels([c for c, _ in metric_cols], rotation=45, ha="right",
                                   fontsize=8)
                ax.set_yticks(range(len(row_names_sorted)))
                ax.set_yticklabels(row_names_sorted, fontsize=7)
                ax.set_title("Heatmap — model × metric (z-score, green=better)")
                plt.colorbar(im, ax=ax, label="z-score (higher = better)")
                fig.tight_layout()
                f = fig_dir / "heatmap_model_x_metric.png"
                fig.savefig(f, dpi=120); plt.close(fig)
                out["heatmap_model_x_metric"] = str(f)
        except Exception as e:
            log.debug(f"  [phase16] heatmap failed: {e}")

    # ── 3. Calibration curve (PI coverage vs nominal — G-163) ─────────
    # 행=model, x=nominal coverage (50/80/95), y=empirical pi*_coverage
    # 대각선 = perfect calibration. 너무 높으면 over-coverage (PI 과대), 낮으면 under.
    if rows:
        try:
            nominal = _np.array([0.50, 0.80, 0.95])
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="perfect calibration")
            for r in rows[:20]:  # top 20 to avoid clutter
                emp = []
                for nom, key in zip(nominal, ("pi50_coverage", "pi80_coverage",
                                              "pi95_coverage")):
                    v = r.get(key)
                    if v in (None, "", "nan"):
                        emp.append(_np.nan)
                    else:
                        try:
                            emp.append(float(v))
                        except (TypeError, ValueError):
                            emp.append(_np.nan)
                if any(_np.isfinite(emp)):
                    ax.plot(nominal, emp, "o-", alpha=0.6, lw=1,
                             label=r.get("model", "?")[:20])
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
            ax.set_xlabel("Nominal coverage")
            ax.set_ylabel("Empirical coverage")
            ax.set_title("Calibration curve — PI coverage vs nominal (top 20 models)")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=6, ncol=2, loc="lower right")
            fig.tight_layout()
            f = fig_dir / "calibration_curve.png"
            fig.savefig(f, dpi=120); plt.close(fig)
            out["calibration_curve"] = str(f)
        except Exception as e:
            log.debug(f"  [phase16] calibration curve failed: {e}")

    # ── 4. Horizon decay (MASE h=1, 4, 13, 52 — G-163) ────────────────
    # MASE 가 horizon 따라 어떻게 증가/감소하는지 — short-term skill 가시화
    if rows:
        try:
            horizons = [("mase_h1", 1), ("mase_h4", 4), ("mase_h13", 13),
                        ("mase_h52", 52)]
            fig, ax = plt.subplots(figsize=(7, 5))
            for r in rows[:15]:  # top 15 to keep readable
                ys = []
                for key, _ in horizons:
                    v = r.get(key)
                    if v in (None, "", "nan"):
                        ys.append(_np.nan)
                    else:
                        try:
                            ys.append(float(v))
                        except (TypeError, ValueError):
                            ys.append(_np.nan)
                xs = [h for _, h in horizons]
                if any(_np.isfinite(ys)):
                    ax.plot(xs, ys, "o-", alpha=0.7, lw=1,
                             label=r.get("model", "?")[:20])
            ax.axhline(1.0, color="red", ls="--", lw=1, alpha=0.5,
                       label="MASE=1 (naive)")
            ax.set_xscale("log")
            ax.set_xticks([1, 4, 13, 52])
            ax.set_xticklabels(["h=1", "h=4", "h=13", "h=52"])
            ax.set_xlabel("Horizon (weeks, log scale)")
            ax.set_ylabel("MASE — lower is better")
            ax.set_title("Horizon decay — MASE across 4 seasonality scales (top 15)")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=6, ncol=2, loc="upper left")
            fig.tight_layout()
            f = fig_dir / "horizon_decay.png"
            fig.savefig(f, dpi=120); plt.close(fig)
            out["horizon_decay"] = str(f)
        except Exception as e:
            log.debug(f"  [phase16] horizon decay failed: {e}")

    return out


from simulation.utils.resource_tracker import track_resources


@track_resources("comprehensive_eval")
def run_comprehensive_eval(
    phase1: dict,
    all_results: dict,
    config,
    eval_logger=None,
) -> dict:
    """Comprehensive evaluation aggregator."""
    t0 = time.time()
    out_dir = Path(getattr(config, "save_dir", "simulation/results")) / "comprehensive_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"  [R12] aggregating results from R2-R10 → {out_dir}")

    master_csv = _build_master_grid(all_results, out_dir)
    log.info(f"  [phase16] MASTER_GRID written: {master_csv}")

    ranking = _ranking_consolidated(all_results, out_dir)
    log.info(f"  [phase16] consolidated ranking ({len(ranking.get('consolidated', []))} models)")

    # Per-model deep-dive — wipe stale files from prior runs first.
    # When the user re-runs with --force or --models filter, prior runs'
    # deep-dive .md files for models no longer evaluated would otherwise
    # remain and inflate the directory listing (75 stale + 3 new = 78
    # confusing files). Clean slate per run.
    per_model_dir = out_dir / "per_model"
    if per_model_dir.exists():
        for stale in per_model_dir.glob("*.md"):
            try:
                stale.unlink()
            except OSError:
                pass
    per_model_dir.mkdir(parents=True, exist_ok=True)
    models_seen: set = set()
    # Baseline: drill into nested individual_results + ensemble_results
    bl_root = (all_results.get("baseline", {}) or {}).get("model_results", {})
    if isinstance(bl_root, dict):
        for k in ("individual_results", "ensemble_results"):
            inner = bl_root.get(k)
            if isinstance(inner, dict):
                models_seen.update(inner.keys())
        # Legacy flat fallback
        if not models_seen:
            models_seen.update(k for k, v in bl_root.items() if isinstance(v, dict))
    # WF-CV: actual key is 'wf_results' (not 'model_results')
    wf_root = all_results.get("wfcv", {}) or {}
    wf_dict = wf_root.get("wf_results") or wf_root.get("model_results") or {}
    if isinstance(wf_dict, dict):
        models_seen.update(wf_dict.keys())

    # Filter out non-model meta keys that leak from runner output dicts
    META_KEYS = {
        "best_individual", "best_overall",
        "ensemble_results", "individual_results",
        "failed_models", "n_ensembles_run", "n_models_run",
        "summary", "epi_validity_gate", "elapsed",
        "holdout_predictions", "holdout_start", "fold_val_indices",
        "wf_results", "model_results", "plot_manifest", "split",
    }
    models_seen = {m for m in models_seen if m not in META_KEYS}

    # Respect --models CLI filter when present
    selected = set(getattr(config, "_selected_models", None) or [])
    if selected:
        models_seen = models_seen & selected
        log.info(f"  [phase16] --models filter: kept {len(models_seen)} of "
                 f"{len(selected)} requested models")
    p11 = all_results.get("per_model_eval", {}) or {}
    if isinstance(p11, dict):
        models_seen.update(p11.get("ranking_top10", []))

    # R10's metrics CSV is the authoritative record of what was actually
    # evaluated, so treat it as a peer source rather than a fallback. The two
    # in-memory sources above come from phases the checkpoints cannot restore
    # (R2 stores only a model count, R4 a subset), so on any resumed or
    # `--models`-filtered run they are empty and this report would otherwise
    # announce "Models evaluated: 0" while the CSV holds all 48.
    if isinstance(p11, dict) and p11.get("metrics_csv"):
        csv_dir = Path(p11["metrics_csv"]).parent
        before = len(models_seen)
        models_seen.update(r.get("model", "") for r in _load_per_model_metrics(csv_dir))
        models_seen.discard("")
        if len(models_seen) > before:
            log.info(f"  [R12] +{len(models_seen) - before} models from "
                     f"{csv_dir.name}/per_model_metrics.csv "
                     f"(total {len(models_seen)})")
    n_dive = 0
    for m in sorted(models_seen):
        try:
            _per_model_deep_dive(m, all_results, out_dir)
            n_dive += 1
        except Exception as e:
            log.debug(f"  [phase16] per-model {m} failed: {e}")
    log.info(f"  [phase16] {n_dive} per-model deep-dive reports written")

    tables = _stat_comparison_tables(all_results, out_dir)
    log.info(f"  [phase16] statistical tables: {list(tables.keys())}")

    figures = _figures(all_results, out_dir)
    log.info(f"  [phase16] figures: {list(figures.keys())}")

    # Top-level REPORT.md
    md = [
        "# R12 — Comprehensive Evaluation Report",
        "",
        f"- Models evaluated: {len(models_seen)}",
        f"- Per-model deep-dive reports: {n_dive}",
        f"- Master grid CSV: `{master_csv.name}`",
        f"- Statistical tables: {list(tables.keys())}",
        f"- Figures: {list(figures.keys())}",
        "",
        "## Consolidated ranking (Borda-count across phases)",
        "",
        "| rank | model | Borda score |",
        "|------|-------|-------------|",
    ]
    for i, item in enumerate(ranking.get("consolidated", [])[:15], 1):
        md.append(f"| {i} | {item['model']} | {item['borda_score']:.1f} |")
    md += [
        "",
        "## R/P coverage",
        "",
        "| R/P | Result key | Status |",
        "|-----|------------|--------|",
    ]
    # R12 decoupled from P1 (2026-06-20): real_eval not listed — it runs AFTER R12
    # (production track), so it is intentionally absent from this research-track coverage.
    for phase, key in [
        ("R2", "baseline"), ("R4", "wfcv"), ("R5", "diagnostics"),
        ("R6", "dm_tests"),
        ("R7", "prediction_intervals"), ("R8", "scoring"),
        ("R9", "per_model_optimize"),
        ("R10", "per_model_eval"), ("R11", "feature_importance"),
    ]:
        present = key in all_results and not (
            isinstance(all_results[key], dict) and all_results[key].get("skipped"))
        md.append(f"| {phase} | `{key}` | {'OK' if present else 'missing/skipped'} |")
    # R6 MAJOR #4 (2026-05-26): auto-load fairness + LOSO JSONs if present
    # Reviewer critique: 'Fairness/LOSO scripts not integrated into R12 report'
    md += [
        "",
        "## Per-age Fairness (TRIPOD-AI 5g) — auto-loaded from phase11_fairness/",
        "",
    ]
    fairness_json = Path(getattr(config, "save_dir", "simulation/results")) / "phase11_fairness" / "per_age.json"
    if fairness_json.exists():
        try:
            fdata = json.loads(fairness_json.read_text(encoding="utf-8"))
            di = fdata.get("disparity", {}).get("disparate_impact_sens", float("nan"))
            di_pass = fdata.get("disparity", {}).get("di_4_5_rule_pass", False)
            md.append(f"- **Disparate Impact (Sens)**: {di} (4/5-rule: {'PASS ✓' if di_pass else 'FAIL ✗'})")
            md.append("")
            md.append("| Age group | MAE | R² | Sens | F1 |")
            md.append("|-----------|------|------|------|------|")
            for r in fdata.get("per_age", []):
                md.append(f"| {r['age']} | {r['mae']} | {r['r2']} | {r['sensitivity']} | {r['f1']} |")
            md.append(f"\nSource: `phase11_fairness/per_age.json` (seed=42, "
                      f"train through {fdata.get('train_through')}, test {fdata.get('test_season')}).")
        except Exception as _e:
            md.append(f"_(failed to parse fairness JSON: {_e})_")
    else:
        md.append("_(no fairness output found — run `python -m simulation.scripts.phase11_fairness`)_")

    md += [
        "",
        "## Cross-Season LOSO (TRIPOD-AI 5j) — auto-loaded from phase11_loso/",
        "",
    ]
    loso_json = Path(getattr(config, "save_dir", "simulation/results")) / "phase11_loso" / "loso_per_season.json"
    if loso_json.exists():
        try:
            ldata = json.loads(loso_json.read_text(encoding="utf-8"))
            era = ldata.get("era_stratified", {})
            if era:
                md.append(f"- **Normal era mean MAE**: {era.get('normal_mean_mae', 'N/A')} "
                          f"({', '.join(map(str, era.get('normal_seasons', [])))})")
                md.append(f"- **COVID era mean MAE**: {era.get('covid_mean_mae', 'N/A')} "
                          f"({', '.join(map(str, era.get('covid_seasons', [])))})")
                if "covid_vs_normal_ratio" in era:
                    md.append(f"- **COVID/normal ratio**: {era['covid_vs_normal_ratio']}× "
                              f"(OOD detected: {era.get('ood_detected', False)})")
            md.append("")
            md.append("| Held-out | Context | n_train | n_test | MAE | R² | F1 |")
            md.append("|----------|---------|---------|--------|------|------|------|")
            for r in ldata.get("per_season", []):
                md.append(f"| {r['season']} | {r['context']} | {r['n_train']} | "
                          f"{r['n_test']} | {r['mae']} | {r['r2']} | {r['f1']} |")
            md.append(f"\nSource: `phase11_loso/loso_per_season.json` (seed=42).")
        except Exception as _e:
            md.append(f"_(failed to parse LOSO JSON: {_e})_")
    else:
        md.append("_(no LOSO output found — run `python -m simulation.scripts.phase11_loso_full`)_")

    md += [
        "",
        "## Per-model deep-dive index",
        "",
    ]
    for m in sorted(models_seen):
        md.append(f"- [{m}](per_model/{m}.md)")
    md += [
        "",
        "## Reproducibility",
        "",
        "- Audit metadata: `simulation/results/eval_logs/{run_id}_audit.json`",
        "- Per-record evaluation log: `simulation/results/eval_logs/{run_id}.jsonl`",
        "- All-runs index: `simulation/results/eval_logs/INDEX.csv`",
        "",
    ]
    # Append metric rubric
    try:
        from simulation.analytics.metric_rubric import render_rubric_markdown
        md.append(render_rubric_markdown())
    except Exception as _e:
        log.debug(f"  [phase16] metric rubric render failed: {_e}")
    report_path = out_dir / "REPORT.md"
    report_path.write_text("\n".join(md), encoding="utf-8")
    log.info(f"  [phase16] top-level report: {report_path}")

    # Log to EvalLogger if provided
    if eval_logger is not None:
        try:
            eval_logger.log(phase="phase16", model="<phase>",
                            metric="comprehensive_done",
                            value=1, n_models=len(models_seen),
                            n_deep_dives=n_dive,
                            n_tables=len(tables), n_figures=len(figures))
        except Exception:
            pass

    # 2026-05-28 V1 사용자 명시: eLife figure auto-call
    # R9 결과 (per_model_optimal/) 가져와서 elife figure 자동 생성.
    # scripts/run_elife_phase12.py 의 main() subprocess 호출 — 격리 + 학습 영향 0.
    elife_status = "skipped"
    try:
        import subprocess as _sp_elife, sys as _sys_elife
        from pathlib import Path as _P_elife
        _elife_script = _P_elife(__file__).parent.parent.parent / "scripts" / "run_elife_phase12.py"
        if _elife_script.exists():
            _r = _sp_elife.run(
                [_sys_elife.executable, str(_elife_script)],
                check=False, timeout=600,
                capture_output=True, text=True,
            )
            if _r.returncode == 0:
                elife_status = "completed → simulation/results/phase12_elife/"
                log.info(f"  [phase16] eLife figure 생성 완료 (subprocess)")
            else:
                elife_status = f"subprocess returncode={_r.returncode}"
                log.warning(f"  [phase16] eLife subprocess fail: {_r.stderr[:200]}")
        else:
            elife_status = f"script not found: {_elife_script}"
    except Exception as _elife_err:
        elife_status = f"error: {_elife_err}"
        log.warning(f"  [phase16] eLife auto-call skip: {_elife_err}")

    return {
        "n_models": len(models_seen),
        "n_per_model_dives": n_dive,
        "master_grid_csv": str(master_csv),
        "ranking_consolidated_path": str(out_dir / "ranking_consolidated.json"),
        "tables": tables,
        "figures": figures,
        "report_path": str(report_path),
        "elapsed": time.time() - t0,
        # 2026-05-28 V1: eLife figure auto-call status
        "elife_phase12_status": elife_status,
    }


# back-compat aliases (2026-06-02 semantic rename — 옛 run_phaseN)
run_phase16 = run_comprehensive_eval
