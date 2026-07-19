#!/usr/bin/env python3
"""Regenerate R12 comprehensive_eval REPORT from ON-DISK R10 results (NO retraining).

Why this exists
---------------
``simulation/pipeline/comprehensive_eval.py::run_comprehensive_eval`` consumes an
**in-memory** ``all_results`` dict that a live pipeline run hands it. When R12 is run
standalone (or the in-memory dict is empty for the model-bearing keys), it writes a
hollow ``REPORT.md`` that says "Models evaluated: 0" even though the real per-model
metrics are sitting on disk. That is exactly the stale state evidence-sync #4 found
(REPORT.md=0 models, while ``per_model_eval/per_model_metrics.csv`` holds 48 models
with FusedEpi as champion).

This script reconstructs the *minimal* ``all_results`` dict that the existing R12
functions need — pointing ``per_model_eval.metrics_csv`` at the real on-disk CSV and
injecting the ``ranking.json`` summary (top-10 by WIS, SPA test, pairwise relative
WIS) — then calls the SAME R12 functions (no logic duplicated). The result:

  • REPORT.md          — 48-model coverage, FusedEpi champion, consolidated ranking
  • per_model/<m>.md   — one deep-dive per evaluated model (R10 metric table)
  • tables/            — pairwise_relative_wis.csv + hansen_spa.json
  • figures/           — forest_plot / heatmap / calibration / horizon_decay (matplotlib,
                         regenerated from the on-disk metrics CSV)

Honesty contract (no fabrication)
---------------------------------
  • Figures ARE regenerated when matplotlib is present and the metrics CSV is readable
    (they are pure functions of that CSV — no in-memory training state needed).
  • R6 DM-test table requires the in-memory ``dm_tests`` object which is NOT persisted to
    the live results dir; it is therefore honestly ABSENT (the REPORT R/P-coverage row
    for R6 will read 'missing/skipped'). We do NOT copy a stale archived dm_pvalues.csv.
  • The script touches ZERO model weights and runs ZERO training. DB is opened read-only
    only if a downstream function needs it (none here do).

Run from project root:
    .venv/bin/python -m simulation.scripts.regenerate_comprehensive_report
"""
from __future__ import annotations

import argparse
import csv as _csv
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("regenerate_comprehensive_report")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "simulation" / "results"
PER_MODEL_EVAL_DIR = RESULTS_DIR / "per_model_eval"
METRICS_CSV = PER_MODEL_EVAL_DIR / "per_model_metrics.csv"
RANKING_JSON = PER_MODEL_EVAL_DIR / "ranking.json"
PER_MODEL_OPTIMAL_DIR = RESULTS_DIR / "per_model_optimal"


def _load_ranking_summary() -> dict:
    """Load R10 ranking.json into the keys the R12 functions expect.

    The on-disk ``ranking.json`` stores ``top10_by_wis`` / ``spa_test`` /
    ``pairwise_relative_wis``. ``comprehensive_eval`` reads them under the keys
    ``ranking_top10`` / ``spa_test`` / ``pairwise_relative_wis``. This adapter maps them.

    Returns:
        dict with keys: ranking_top10 (list[str]), spa_test (dict),
        pairwise_relative_wis (dict[str,float]), metrics_csv (str path).
        Empty/absent fields are simply omitted (downstream is graceful).
    """
    summary: dict = {"metrics_csv": str(METRICS_CSV)}
    if RANKING_JSON.is_file():
        r = json.loads(RANKING_JSON.read_text(encoding="utf-8"))
        if r.get("top10_by_wis"):
            summary["ranking_top10"] = r["top10_by_wis"]
        if r.get("spa_test"):
            summary["spa_test"] = r["spa_test"]
        if r.get("pairwise_relative_wis"):
            summary["pairwise_relative_wis"] = r["pairwise_relative_wis"]
        summary["n_models_evaluated"] = r.get("n_models_evaluated")
        summary["n_test_weeks"] = r.get("n_test_weeks")
    return summary


def _load_per_model_optimize() -> dict:
    """Reconstruct R9 per_model_optimize.per_model_configs from per_model_optimal/*.json.

    Each ``per_model_optimal/<model>.json`` holds ``best_config`` + ``val_metrics``.
    R12's deep-dive and master grid read ``best_config``/``best_metrics`` — map them.

    Returns:
        {"per_model_configs": {model: {best_config, best_metrics}}}; empty if dir absent.
    """
    configs: dict = {}
    if PER_MODEL_OPTIMAL_DIR.is_dir():
        for jf in sorted(PER_MODEL_OPTIMAL_DIR.glob("*.json")):
            try:
                d = json.loads(jf.read_text(encoding="utf-8"))
            except Exception as e:  # noqa: BLE001
                log.warning("  skip %s: %s", jf.name, e)
                continue
            model = d.get("model") or jf.stem
            configs[model] = {
                "best_config": d.get("best_config", {}),
                "best_metrics": d.get("val_metrics", {}),
            }
    return {"per_model_configs": configs} if configs else {}


def _model_names_from_csv() -> list[str]:
    """Read the model column from the on-disk per_model_metrics.csv (for coverage count)."""
    if not METRICS_CSV.is_file():
        return []
    with METRICS_CSV.open(encoding="utf-8") as f:
        return [r["model"] for r in _csv.DictReader(f) if r.get("model")]


def _full_model_table() -> str:
    """Build a full WIS-sorted markdown table of ALL evaluated models from the CSV.

    Columns mirror ``per_model_eval/report.md`` (the SSOT R10 report) so the
    comprehensive REPORT is self-contained for a reviewer. Models with no WIS
    (foundation/pf rolling-only models) sort last but are still listed.

    Returns:
        Markdown table string (header + 48 rows), or "" if CSV unreadable.
    """
    if not METRICS_CSV.is_file():
        return ""
    with METRICS_CSV.open(encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))

    def _f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("inf")

    rows.sort(key=lambda r: _f(r.get("wis")))
    lines = [
        "| rank | model | WIS | log-WIS | MAE | R² | 95% PI cov | rel-WIS-pair | champion |",
        "|------|-------|-----|---------|-----|-----|-----------|--------------|----------|",
    ]

    def _s(v, nd=3):
        try:
            return f"{float(v):.{nd}f}"
        except (TypeError, ValueError):
            return "nan"

    for i, r in enumerate(rows, 1):
        champ = "★" if str(r.get("champion_best_wis", "")).lower() == "true" else ""
        lines.append(
            f"| {i} | {r.get('model', '?')} | {_s(r.get('wis'))} | "
            f"{_s(r.get('log_wis'))} | {_s(r.get('mae'))} | {_s(r.get('r2'))} | "
            f"{_s(r.get('pi95_coverage'))} | {_s(r.get('relative_wis_pairwise'))} | {champ} |"
        )
    return "\n".join(lines)


def _inject_full_sections(body: str, full_table: str, all_models: list[str]) -> str:
    """Insert the full 48-model table and replace the deep-dive index with all models.

    Args:
        body: current REPORT.md text.
        full_table: markdown table from _full_model_table().
        all_models: sorted list of every evaluated model name.

    Returns:
        Updated REPORT.md text.
    """
    # 1. Insert the full table right after the consolidated-ranking section header block,
    #    before "## R/P coverage".
    full_section = (
        "## All evaluated models — full test-slab table (n=68, WIS-sorted)\n\n"
        "_Regenerated from `per_model_eval/per_model_metrics.csv` (no retraining). "
        "★ = champion (best WIS). 'nan' WIS = foundation/pf rolling-only models "
        "(scored on MAE/R² instead)._\n\n"
        + full_table
        + "\n\n## R/P coverage\n"
    )
    if "\n## R/P coverage\n" in body:
        body = body.replace("\n## R/P coverage\n", "\n" + full_section, 1)

    # 2. Replace the (top-10) deep-dive index with the full set.
    idx_lines = ["## Per-model deep-dive index", ""]
    idx_lines += [f"- [{m}](per_model/{m}.md)" for m in all_models]
    new_index = "\n".join(idx_lines)
    start = body.find("## Per-model deep-dive index")
    if start != -1:
        end = body.find("\n## ", start + 1)
        if end == -1:
            end = len(body)
        body = body[:start] + new_index + "\n" + body[end:]
    return body


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir",
        default=str(RESULTS_DIR / "comprehensive_eval"),
        help="output dir (default: simulation/results/comprehensive_eval)",
    )
    args = ap.parse_args()

    if not METRICS_CSV.is_file():
        log.error("On-disk R10 metrics CSV not found: %s — cannot regenerate.", METRICS_CSV)
        return 1

    sys.path.insert(0, str(PROJECT_ROOT))
    from simulation.pipeline.comprehensive_eval import (
        _per_model_deep_dive,
        run_comprehensive_eval,
    )

    model_names = _model_names_from_csv()
    log.info("On-disk R10 metrics CSV: %d models (%s)", len(model_names), METRICS_CSV)

    # Reconstruct the minimal all_results the R12 functions consume — from disk only.
    per_model_eval = _load_ranking_summary()
    per_model_optimize = _load_per_model_optimize()
    log.info(
        "Reconstructed: per_model_eval(top10=%d, spa=%s, pairwise=%d) · "
        "per_model_optimize(%d configs)",
        len(per_model_eval.get("ranking_top10", [])),
        "yes" if per_model_eval.get("spa_test") else "no",
        len(per_model_eval.get("pairwise_relative_wis", {})),
        len(per_model_optimize.get("per_model_configs", {})),
    )

    all_results: dict = {
        "per_model_eval": per_model_eval,
        # R9 (best transform/scaler per model) — populates deep-dives + master grid
        "per_model_optimize": per_model_optimize,
        # R2/R4/R5/R6/R7/R8 in-memory objects are NOT persisted to the live dir.
        # They are honestly absent; the REPORT R/P-coverage rows reflect that.
        "feature_importance": {},  # R11 SHAP is on disk under phase11_* but not consumed via all_results here
    }

    config = SimpleNamespace(save_dir=str(RESULTS_DIR), _selected_models=None)

    log.info("Calling existing R12 run_comprehensive_eval (consume-only, no training)...")
    result = run_comprehensive_eval(
        phase1={}, all_results=all_results, config=config, eval_logger=None
    )

    out_dir = Path(args.out_dir)
    report = out_dir / "REPORT.md"

    # ── Complete the per-model deep-dives for ALL evaluated models ──────────────
    # run_comprehensive_eval derives ``models_seen`` from the in-memory R2/R4 dicts
    # (absent here) + ranking_top10 (only 10), so it wrote 10 deep-dives. The metrics
    # CSV holds all 48 evaluated models — generate the remaining deep-dives directly
    # (the function reads from the CSV, no training).
    n_dives_all = 0
    for m in sorted(set(model_names)):
        try:
            _per_model_deep_dive(m, all_results, out_dir)
            n_dives_all += 1
        except Exception as e:  # noqa: BLE001
            log.warning("  deep-dive %s failed: %s", m, e)
    log.info("  per-model deep-dives (all evaluated models): %d", n_dives_all)

    # ── Build a full WIS-sorted 48-model table from the on-disk CSV ─────────────
    full_table = _full_model_table()

    # ── Correct the misleading headline count + inject full table + full index ──
    body = report.read_text(encoding="utf-8")
    body = body.replace(
        f"- Models evaluated: {(result or {}).get('n_models', 0)}",
        f"- Models evaluated: {len(set(model_names))}",
    ).replace(
        f"- Per-model deep-dive reports: {(result or {}).get('n_per_model_dives', 0)}",
        f"- Per-model deep-dive reports: {n_dives_all}",
    )
    # Replace the (top-10-only) deep-dive index with the full set, and add full table.
    body = _inject_full_sections(body, full_table, sorted(set(model_names)))
    report.write_text(body, encoding="utf-8")

    # ── Honesty stamp: prepend a provenance + stale-figure banner to the REPORT ──
    fig_keys = list((result or {}).get("figures", {}).keys())
    n_models = len(set(model_names))
    banner = [
        "<!-- REGENERATED from on-disk R10 results by "
        "simulation/scripts/regenerate_comprehensive_report.py (no retraining). -->",
        "",
        "> **Provenance / honesty stamp** — This report was regenerated from the on-disk "
        f"R10 artifact `per_model_eval/per_model_metrics.csv` ({len(model_names)} models, "
        "FusedEpi champion by WIS) **without any retraining**. ",
        ">",
        "> - **Champion (R10, by WIS):** FusedEpi. The full per-model table lives in the "
        "consolidated ranking + per-model deep-dives below and in "
        "`per_model_eval/report.md`.",
        "> - **Figures regenerated from the metrics CSV:** "
        + (", ".join(f"`{k}`" for k in fig_keys) if fig_keys
           else "_none — matplotlib unavailable or CSV unreadable_") + ".",
        "> - **R6 DM-test table is STALE / requires a full run to regenerate:** the "
        "Diebold-Mariano pairwise object is an in-memory pipeline product that is not "
        "persisted to the live results dir, so it is honestly absent here (not copied "
        "from an archive). Run `bash scripts/launch_full_run.sh` to repopulate it.",
        "> - **Per-age fairness / cross-season LOSO** sections auto-load from "
        "`phase11_fairness/` and `phase11_loso/` if present; absent = honestly marked.",
        "",
    ]
    body = report.read_text(encoding="utf-8")
    report.write_text("\n".join(banner) + body, encoding="utf-8")

    log.info("=== Regeneration complete (NO retraining) ===")
    log.info("  REPORT.md models-evaluated: %d (deep-dives: %d)", n_models, n_dives_all)
    log.info("  figures: %s", fig_keys or "(none)")
    log.info("  tables:  %s", list((result or {}).get("tables", {}).keys()) or "(none)")
    log.info("  report:  %s", report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
