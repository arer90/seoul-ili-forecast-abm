"""Utility CLI commands — extracted from __main__.py.

Phase C2 partial (2026-05-12 cont.): 6 bounded utility handlers (extract-pdf,
verify-audit, freeze-paper-primary, visualize, feature-importance, rehydrate)
moved here. Each handler is small (15-40 lines) and self-contained.
"""
from __future__ import annotations

import logging
import sys


log = logging.getLogger(__name__)


def cmd_extract_pdf(args) -> None:
    """`python -m simulation extract-pdf` — Seoul annual-report PDF → DB."""
    from simulation.collectors.extract_pdf import (
        DEFAULT_SOURCE_TAG, extract_pdf, find_pdf,
    )
    from simulation.database.config import DB_PATH

    pdf_path = find_pdf(getattr(args, "pdf", None))
    print(f"PDF: {pdf_path.name}")
    tag = getattr(args, "source_tag", None) or DEFAULT_SOURCE_TAG
    result = extract_pdf(
        pdf_path,
        str(DB_PATH),
        source_tag=tag,
        force=bool(getattr(args, "force", False)),
    )
    if result.get("skipped"):
        print("Skipped: data already present. Use --force to re-extract.")
    else:
        print(f"Done. district={result['district']}, monthly={result['monthly']} rows")


def cmd_verify_audit(args) -> None:
    """`python -m simulation verify-audit` — scan simulation/ for FORBIDDEN_PATTERNS."""
    from simulation.verifier import AstChecker

    path = getattr(args, "path", "simulation")
    print(f"[verify-audit] scanning {path} ...")
    report = AstChecker().scan_dir(path)
    print(report.summary())
    exit_code = 0
    if report.n_fail > 0:
        exit_code = 2
    elif report.n_warn > 0 and getattr(args, "fail_on_warn", False):
        exit_code = 1
    if exit_code:
        sys.exit(exit_code)


def cmd_freeze_paper_primary(args) -> None:
    """`python -m simulation freeze-paper-primary` — snapshot PAPER_PRIMARY_11."""
    from simulation.models.registry import (
        build_snapshot, persist_snapshot, verify_paper_primary_frozen,
    )

    if getattr(args, "verify", False):
        report = verify_paper_primary_frozen()
        print(f"[freeze-paper-primary] frozen={report.get('n_frozen', 0)}  "
              f"ok={report['ok']}")
        for m in report.get("mismatches", []):
            print(f"  MISMATCH {m['model_name']:<30s}  "
                  f"{m['source_file']}")
            print(f"     frozen : {m['frozen_sha']}")
            print(f"     current: {m['current_sha']}")
        return

    snapshots = build_snapshot(
        mark_paper_primary=True,
        freeze_paper_primary=bool(getattr(args, "freeze", False)),
    )
    rows = persist_snapshot(snapshots, replace=True)
    n_primary = sum(1 for s in snapshots if s.is_paper_primary)
    n_frozen = sum(1 for s in snapshots if s.frozen_at)
    print(f"[freeze-paper-primary] persisted {rows} rows  "
          f"(PAPER_PRIMARY={n_primary}, frozen={n_frozen})")


def cmd_feature_importance(args) -> None:
    """`python -m simulation feature-importance` — Optuna freq + SHAP figures."""
    from pathlib import Path as _PPath

    from simulation.utils.feature_importance import run_feature_importance

    models = None
    if getattr(args, "models", None):
        models = [m.strip() for m in str(args.models).split(",") if m.strip()]
    out_dir = _PPath(args.out_dir) if getattr(args, "out_dir", None) else None
    res = run_feature_importance(
        models_filter=models,
        top_k=int(getattr(args, "top_k", 30)),
        include_shap=not bool(getattr(args, "no_shap", False)),
        out_dir=out_dir,
    )
    if res.get("skipped"):
        log.warning(f"[feat-imp] skipped: {res.get('reason','?')}")
        sys.exit(1)
    log.info(f"[feat-imp] {res.get('n_figures',0)} figures, "
             f"{res.get('n_models',0)} models in {res.get('elapsed',0):.1f}s")
    log.info(f"[feat-imp] index: {res.get('index_md')}")


def cmd_rehydrate(args) -> None:
    """`python -m simulation rehydrate` — register legacy bare-model .pt as champions.

    Bridges R2 baseline (which writes bare-model `.pt`) with R9
    (per_model_optimize)'s champion-challenger ledger. Imports
    `test_wis`/`test_mae` from `post_E_eval.json` when available so legacy
    entries have real metrics.
    """
    from simulation.utils.rehydrate import run_rehydrate

    res = run_rehydrate(
        dry_run=bool(getattr(args, "dry_run", False)),
        force=bool(getattr(args, "force", False)),
        eval_source=getattr(args, "eval_source", "post_E"),
    )
    log.info(f"[rehydrate] {res.get('n_registered')} models in champion_log "
             f"after rehydrate. log: {res.get('log_path')}")


def cmd_list_models(args) -> None:
    """`python -m simulation list-models` — list registry by tier with champion status."""
    import json as _json
    from pathlib import Path as _PPath

    from simulation.models.registry import (
        EXTRA_MODELS, NEGATIVE_CONTROL, PAPER_PRIMARY_11,
    )

    tier = getattr(args, "tier", "all")
    with_champ = bool(getattr(args, "with_champion_status", False))

    # Champion lookup
    champ_set: set = set()
    champ_meta: dict = {}
    if with_champ:
        log_path = _PPath("models") / "champion_log.json"
        if log_path.exists():
            try:
                j = _json.loads(log_path.read_text())
                for nm, rec in j.items():
                    cur = (rec or {}).get("current") or {}
                    if cur.get("filename"):
                        champ_set.add(nm)
                        champ_meta[nm] = {
                            "test_wis": cur.get("test_wis"),
                            "v": cur.get("version"),
                        }
            except Exception:
                pass

    # ── Print paper-primary 11 ──
    if tier in ("all", "paper"):
        print("\n" + "=" * 80)
        print(f"  ⭐ PAPER_PRIMARY_11 ({len(PAPER_PRIMARY_11)} models)")
        print("=" * 80)
        print(f"  {'#':>2}  {'model':<22}  {'category':<14}  {'source file':<32}"
              + ("  champion" if with_champ else ""))
        print("  " + "-" * (78 if with_champ else 70))
        # Sub-categorize paper-11
        paper_cats = {
            "ts (classical)":      ["SARIMA"],
            "linear":              ["ElasticNet"],
            "tree":                ["XGBoost"],
            "epi":                 ["NegBinGLM", "BayesianMCMC"],
            "physics":             ["PINN-Lite"],
            "DL":                  ["TabularDNN-Lite", "TFT", "PatchTST"],
            "foundation":          ["TimesFM-2.5"],
            "ensemble (meta)":     ["Ensemble-Stacking"],
        }
        rank = 0
        for cat in paper_cats:
            for nm in paper_cats[cat]:
                rank += 1
                # Find source file
                src = next((s for n, s in PAPER_PRIMARY_11 if n == nm), "?")
                ch = ""
                if with_champ:
                    if nm in champ_set:
                        m = champ_meta[nm]
                        wis = m.get("test_wis")
                        wis_s = f"{wis:.2f}" if isinstance(wis, (int, float)) else "?"
                        ch = f"  ✅ v{m.get('v','?')}  WIS={wis_s}"
                    else:
                        ch = "  ❌ no champion"
                print(f"  {rank:>2}.  {nm:<22}  {cat:<14}  {src:<32}{ch}")

    # ── Print extras ──
    if tier in ("all", "extra"):
        print("\n" + "=" * 80)
        n_extras = sum(len(v) for v in EXTRA_MODELS.values())
        print(f"  EXTRA_MODELS ({n_extras} models, paper-외 ablation/variants)")
        print("=" * 80)
        for cat in EXTRA_MODELS:
            names = EXTRA_MODELS[cat]
            n_in_champ = sum(1 for nm in names if nm in champ_set)
            print(f"\n  [{cat:<22}]  {len(names)} models"
                  + (f"  ({n_in_champ} champions)" if with_champ else ""))
            for nm in names:
                ch = ""
                if with_champ:
                    if nm in champ_set:
                        m = champ_meta[nm]
                        wis = m.get("test_wis")
                        wis_s = f"{wis:.2f}" if isinstance(wis, (int, float)) else "?"
                        ch = f"  ✅ v{m.get('v','?')}  WIS={wis_s}"
                    else:
                        ch = "  ❌"
                print(f"      {nm:<25}{ch}")

    # ── Print negative ──
    if tier in ("all", "negative") and NEGATIVE_CONTROL:
        print("\n" + "=" * 80)
        print(f"  ✗ NEGATIVE_CONTROL ({len(NEGATIVE_CONTROL)} models, "
              f"intentionally excluded from ensembles)")
        print("=" * 80)
        for nm in sorted(NEGATIVE_CONTROL):
            print(f"      {nm}")

    # ── Summary ──
    print("\n" + "=" * 80)
    n_paper = len(PAPER_PRIMARY_11)
    n_extras = sum(len(v) for v in EXTRA_MODELS.values())
    n_neg = len(NEGATIVE_CONTROL)
    print(f"  Total registered: ⭐ {n_paper} paper + {n_extras} extras + "
          f"{n_neg} negative = {n_paper + n_extras + n_neg}")
    if with_champ:
        n_paper_ch = sum(1 for n, _ in PAPER_PRIMARY_11 if n in champ_set)
        n_extra_ch = sum(1 for grp in EXTRA_MODELS.values()
                         for n in grp if n in champ_set)
        print(f"  Champion status: ⭐ {n_paper_ch}/{n_paper} paper + "
              f"{n_extra_ch}/{n_extras} extras "
              f"= {n_paper_ch + n_extra_ch} total")


def cmd_visualize(args) -> None:
    """`python -m simulation visualize` — render per-model + combined figures.

    Generates (no re-training needed):
      • Time-series overview per model (train/val/test/real bands + actual vs pred)
      • Combined timeseries (all champions on one axes)
      • Residual diagnostic per model (3-panel: time, scatter, Q-Q)
      • Per-horizon AE bar chart on real slab (h=1 = primary KPI)
      • Optuna trial history (combined + per-model running best)
      • Per-model markdown report (figure + slab metrics + per-horizon tables)
      • INDEX.md gallery linking everything
    """
    from pathlib import Path as _PPath

    from simulation.utils.visualize import run_visualize

    models = None
    if getattr(args, "models", None):
        models = [m.strip() for m in str(args.models).split(",") if m.strip()]
    out_dir = _PPath(args.out_dir) if getattr(args, "out_dir", None) else None
    res = run_visualize(
        models_filter=models,
        include_residuals=not bool(getattr(args, "no_residuals", False)),
        include_optuna=not bool(getattr(args, "no_optuna", False)),
        out_dir=out_dir,
    )
    if res.get("skipped"):
        log.warning(f"[visualize] skipped: {res.get('reason','?')}")
        sys.exit(1)
    log.info(f"[visualize] done in {res.get('elapsed', 0):.1f}s — "
             f"{res.get('n_figures',0)} figures, "
             f"{res.get('n_models',0)} models")
    log.info(f"[visualize] index: {res.get('index_md')}")


__all__ = [
    "cmd_extract_pdf",
    "cmd_verify_audit",
    "cmd_freeze_paper_primary",
    "cmd_feature_importance",
    "cmd_rehydrate",
    "cmd_list_models",
    "cmd_visualize",
]
