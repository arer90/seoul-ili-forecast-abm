"""
V16 Pipeline Orchestrator
==========================
Runs all Phases sequentially with checkpoint support.
CLI interface with --dry-run for pre-flight testing.
"""
import argparse
import gc
import json
import logging
import os
import sys
import time
import platform
import importlib
from pathlib import Path

# ── UTF-8 encoding guard for cross-platform logging ───────────
import sys as _sys
if _sys.platform == "win32" and hasattr(_sys.stdout, "reconfigure"):
    try:
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from simulation.config_global import GLOBAL  # SSOT (2026-05-28)
from simulation.pipeline import phases


log = logging.getLogger(__name__)


def resolve_resume_from(value):
    """`--resume-from` 값을 canonical phase order index 로 해석.

    R/P 라벨("R9") 또는 의미이름("per_model_optimize")만 허용한다 — phase 번호는 제거됨.
    반환값은 :mod:`simulation.pipeline.phases` 의 ordered index 이다.

    Args:
        value: "R9" 같은 R/P 라벨, 또는 "per_model_optimize" 같은 의미이름.
            대소문자/공백 무시. None/"" → None.

    Returns:
        int ordered phase index (None/"" 입력 시 None).

    Raises:
        argparse.ArgumentTypeError: 등록된 R/P 라벨/이름이 아닐 때(숫자 포함).
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return phases.resume_index(s)
    except KeyError as e:
        valid = ", ".join(phases.all_labels() + [p[2] for p in phases.PHASES])
        raise argparse.ArgumentTypeError(
            f"--resume-from: 알 수 없는 phase '{value}'. "
            f"R/P 라벨(예: R9) 또는 이름 중 하나: {valid}"
        ) from e


def build_cli_parser() -> argparse.ArgumentParser:
    """CLI argument parser."""
    p = argparse.ArgumentParser(description="MPH Infection Simulation V16 Pipeline")

    p.add_argument("--config", type=str, default=None, help="YAML config file path")
    p.add_argument("--preset", type=str, default="aggressive",
                   help="Preset (aggressive/moderate/conservative)")
    p.add_argument("--optuna-mode", dest="optuna_mode", type=str, default=None,
                   choices=["none", "external", "inline", "all"], help="Optuna mode")
    p.add_argument("--optuna-trials", dest="optuna_trials", type=int, default=None)
    p.add_argument("--optuna-strategy", dest="optuna_strategy", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None, help="Training epochs (final fit)")
    # Stage 3 full_light: decouple Optuna inline-trial epochs from final-fit epochs.
    # Rationale: Optuna search needs fast trials (e.g. 50 ep) but the final fit
    # after HP selection should train longer (e.g. 200 ep).
    # If --inline-epochs is omitted, Optuna inherits --epochs (back-compat).
    p.add_argument("--inline-epochs", dest="inline_epochs", type=int, default=None,
                   help="Optuna inline-trial epochs (overrides --epochs for Optuna trials)")
    p.add_argument("--early-stopping-patience", dest="early_stopping_patience",
                   type=int, default=None,
                   help="Early-stopping patience for NN training (default 10)")
    p.add_argument("--train-ratio", dest="train_ratio", type=float, default=None)
    # HWP §3 4-way split (in-sample / real)
    p.add_argument("--paper-cutoff-week", dest="paper_cutoff_week", type=int,
                   default=None,
                   help="HWP §3 in-sample boundary (week count, default 337)")
    p.add_argument("--in-sample-end", dest="in_sample_end", type=str,
                   default=None,
                   help="ISO date override for in-sample boundary "
                        "(takes priority over --paper-cutoff-week)")
    p.add_argument("--no-real-eval", dest="no_real_eval", action="store_true",
                   help="Skip P1 real forecast evaluation")
    p.add_argument("--per-model-optimize", dest="per_model_optimize",
                   action="store_true",
                   help="R9: per-model individual optimization "
                        "(transform × scaler × feature × HP grid search per "
                        "model). Heavy — adds hours to runtime.")
    p.add_argument("--no-comprehensive-eval", dest="no_comprehensive_eval",
                   action="store_true",
                   help="Skip R12 (master aggregator + per-model "
                        "deep-dives + figures + audit roll-up).")
    p.add_argument("--covid-mode", dest="covid_inclusion_mode",
                   choices=["include", "exclude", "indicator"], default=None,
                   help="R1 COVID-era 3-way sensitivity")
    p.add_argument("--conformal-method", dest="real_conformal_method",
                   choices=["split", "aci", "agaci"], default=None,
                   help="P1 conformal PI method")
    p.add_argument("--ensemble-method", dest="ensemble_method",
                   choices=["nnls", "bma", "stacking", "median"], default=None,
                   help="R8 ensemble combination method")
    p.add_argument("--weather-mode", dest="weather_mode",
                   choices=["observed", "climatology", "hybrid"],
                   default=None,
                   help="P1 weather feature handling on real slab "
                        "(default: observed = perfect-foresight, optimistic). "
                        "climatology = week-of-year mean from in-sample. "
                        "hybrid = KMA fcst_* where available + climatology fallback.")
    p.add_argument("--wf-step", dest="wf_step", type=int, default=None, help="WF-CV step (weeks)")
    p.add_argument("--wf-retune", dest="wf_retune", type=int, default=None, help="Inline retune interval")
    # C-step
    p.add_argument("--paper-primary-only", dest="paper_primary_only",
                   action="store_true",
                   help="WF-CV: registry.PAPER_PRIMARY_11 과 factory 교집합만 실행하고 "
                        "step_size 를 wfcv.step_size_paper_primary (기본 4) 로 압축")
    p.add_argument("--resume-from", dest="resume_from", type=resolve_resume_from, default=None,
                   help="Resume from R/P 라벨(예: R9) 또는 의미이름(예: per_model_optimize). "
                        "번호는 미지원(거부됨) — phases.py registry order 로 변환")
    p.add_argument("--save-dir", dest="save_dir", type=str, default=None)
    p.add_argument("--lite", action="store_true", help="Lite mode (fewer trials)")
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="Pre-flight check: verify libraries, OS, DB, dirs without training")
    p.add_argument("--force", dest="force_overwrite", action="store_true",
                   help="Force overwrite existing results without prompt")
    p.add_argument("--no-cache", dest="no_cache", action="store_true",
                   help="Ignore FE parquet cache, rebuild from DB")
    # Stage 4 — epi-validity gate toggles
    p.add_argument("--epi-validity-disable", dest="epi_validity_disable",
                   action="store_true",
                   help="Disable the epi-validity gate after R5 (default: on)")
    p.add_argument("--epi-validity-strict", dest="epi_validity_strict",
                   action="store_true",
                   help="Flip exclude_from_ensemble=True on any gate failure")

    return p

def run_dry_run(config) -> bool:
    """Pre-flight check: libraries, OS, DB, memory, dirs.
    Returns True if all checks pass."""
    print("=" * 60)
    print("  V16 Pipeline DRY-RUN (Pre-flight Check)")
    print("=" * 60)
    ok_count = 0
    fail_count = 0
    warn_count = 0

    def _ok(msg):
        nonlocal ok_count; ok_count += 1; print(f"  [OK]   {msg}")
    def _fail(msg):
        nonlocal fail_count; fail_count += 1; print(f"  [FAIL] {msg}")
    def _warn(msg):
        nonlocal warn_count; warn_count += 1; print(f"  [WARN] {msg}")

    # --- 1. OS & Python ---
    # Project convention: Python 3.12 on Windows (uv venv). Other versions
    # are accepted (3.10+) but flagged as WARN so drift is visible.
    print(f"\n  --- OS & Python ---")
    _ok(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    _ok(f"Interpreter: {sys.executable}")
    py_ver = sys.version.split()[0]
    py_tuple = tuple(int(x) for x in py_ver.split(".")[:2])
    if py_tuple == (3, 12):
        _ok(f"Python {py_ver}  (matches project target 3.12)")
    elif py_tuple >= (3, 10):
        _warn(f"Python {py_ver}  (project validated on 3.12; behavior on {py_ver} not guaranteed)")
    else:
        _fail(f"Python {py_ver} (need >= 3.10, target 3.12)")
    # --- 2. Core Libraries ---
    # Per ENGINEERING_PRINCIPLES.md hard rule: numpy < 2.2 for SHAP/numba compat.
    def _check_version(pkg_ver: str, spec: tuple) -> bool:
        """Return True if pkg_ver satisfies ('<', '2.2') etc."""
        op, target = spec
        try:
            pv = tuple(int(x) for x in pkg_ver.split(".")[:2])
            tv = tuple(int(x) for x in target.split(".")[:2])
        except Exception:
            return True  # can't parse → don't block
        if op == "<":   return pv < tv
        if op == "<=":  return pv <= tv
        if op == ">=":  return pv >= tv
        if op == ">":   return pv > tv
        return True

    print(f"\n  --- Core Libraries ---")
    core_libs = [
        ("numpy", "numpy", ("<", "2.2")),        # SHAP/numba 호환 (ENGINEERING_PRINCIPLES.md)
        ("pandas", "pandas", None),
        ("scipy", "scipy", None),
        ("sklearn", "scikit-learn", None),
        ("yaml", "PyYAML", None),
        ("statsmodels", "statsmodels", None),
    ]
    for mod_name, pip_name, constraint in core_libs:
        try:
            m = importlib.import_module(mod_name)
            ver = getattr(m, "__version__", "?")
            if constraint and ver != "?" and not _check_version(ver, constraint):
                op, target = constraint
                _fail(f"{pip_name} {ver}  (need {op}{target} -- see ENGINEERING_PRINCIPLES.md)")
            else:
                _ok(f"{pip_name} {ver}")
        except ImportError:
            _fail(f"{pip_name} not installed (uv pip install {pip_name})")

    # --- 3. Optional Libraries ---
    print(f"\n  --- Optional Libraries ---")
    opt_libs = [
        ("polars", "polars"),
        ("duckdb", "duckdb"),           # analytical overlay (ENGINEERING_PRINCIPLES.md)
        ("xgboost", "xgboost"),
        ("lightgbm", "lightgbm"),
        ("optuna", "optuna"),
        ("shap", "shap"),
        ("psutil", "psutil"),
        ("torch", "pytorch"),
        ("neuralforecast", "neuralforecast"),
    ]
    for mod_name, pip_name in opt_libs:
        try:
            m = importlib.import_module(mod_name)
            ver = getattr(m, "__version__", "?")
            _ok(f"{pip_name} {ver}")
        except ImportError:
            hint = "analytical overlay" if mod_name == "duckdb" else "optional"
            _warn(f"{pip_name} not installed ({hint})  uv pip install {pip_name}")
    # --- 4. GPU / CUDA / MPS (Apple Silicon) ---
    print(f"\n  --- GPU / CUDA ---")
    try:
        import torch
        if torch.cuda.is_available():
            _ok(f"CUDA available: {torch.cuda.get_device_name(0)}")
            _ok(f"CUDA version: {torch.version.cuda}")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            # MPS is the official PyTorch GPU backend on Apple Silicon —
            # equivalent role to CUDA on Linux/Windows. Don't warn.
            _ok(f"Apple MPS available (Metal Performance Shaders)")
        else:
            _ok("Running in CPU mode (no CUDA, no MPS detected)")
    except ImportError:
        _warn("PyTorch not installed, DL models may fail")

    # --- 5. Database ---
    print(f"\n  --- Database ---")
    db_path = Path(config.data.db_path)
    if db_path.exists():
        size_mb = db_path.stat().st_size / (1024 * 1024)
        _ok(f"DB found: {db_path} ({size_mb:.1f} MB)")
        try:
            # : safe_connect 로 dry-run 조회도 일원화
            from simulation.database import safe_connect
            conn = safe_connect(str(db_path))
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            conn.close()
            _ok(f"DB readable: {len(tables)} tables")
        except Exception as e:
            _fail(f"DB read error: {e}")
    else:
        _fail(f"DB not found: {db_path}")
    # --- 6. Memory ---
    print(f"\n  --- Memory ---")
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        avail_gb = mem.available / (1024**3)
        _ok(f"Total RAM: {total_gb:.1f} GB")
        if avail_gb > 2.0:
            _ok(f"Available: {avail_gb:.1f} GB")
        else:
            _warn(f"Available: {avail_gb:.1f} GB (low, may OOM)")
    except ImportError:
        _warn("psutil not installed, cannot check memory")

    # --- 7. Directories ---
    print(f"\n  --- Directories ---")
    save_dir = Path(config.save_dir)
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
        test_file = save_dir / ".write_test"
        test_file.write_text("test", encoding="utf-8")
        test_file.unlink()
        _ok(f"Save dir writable: {save_dir}")
    except Exception as e:
        _fail(f"Save dir error: {save_dir} - {e}")

    # D7 (M7): reproducibility manifest at run start (git SHA + frozen-deps hash +
    # seed + DB data vintage + key MPH_* env) → populates config_sha256 and is
    # surfaced by the MCP provenance envelope (D1). Best-effort, never blocks a run.
    try:
        from simulation.pipeline.run_manifest import write_run_manifest
        _seed = int(getattr(config, "seed", 42) or 42)
        _man = write_run_manifest(save_dir, seed=_seed)
        _ok(f"Run manifest: config_sha256={_man['config_sha256'][:12]} "
            f"git={(_man.get('git_sha') or '?')[:8]}")
    except Exception as _me:
        log.debug("run manifest skipped: %s", _me)

    log_dir = Path(config.output.log_dir)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        _ok(f"Log dir writable: {log_dir}")
    except Exception as e:
        _fail(f"Log dir error: {log_dir} - {e}")
    # --- 8. External Optuna JSON ---
    if config.optuna.mode in ("external", "all"):
        print(f"\n  --- External Optuna JSON ---")
        json_dir = config.optuna.external_json_dir or str(save_dir)
        json_path = Path(json_dir)
        if json_path.exists():
            jsons = list(json_path.glob("optuna_*.json"))
            if jsons:
                _ok(f"Optuna JSONs found: {len(jsons)} files in {json_dir}")
            else:
                _warn(f"No optuna_*.json files in {json_dir}")
        else:
            _fail(f"Optuna JSON dir not found: {json_dir}")

    # --- 9. Pipeline modules ---
    print(f"\n  --- Pipeline Modules ---")
    modules = [
        "simulation.pipeline.data",
        "simulation.pipeline.baseline",
        "simulation.pipeline.external",
        "simulation.pipeline.diagnostics",
        "simulation.pipeline.dm_test",
        "simulation.pipeline.intervals",
        "simulation.pipeline.wfcv",
        "simulation.pipeline.shap_analysis",
        "simulation.pipeline.scoring",
        "simulation.pipeline.real_eval",
        "simulation.pipeline.per_model_eval",
        "simulation.pipeline.per_model_optimize",
        "simulation.pipeline.comprehensive_eval",
        "simulation.pipeline.inference",
        "simulation.utils.eval_logger",
        "simulation.utils.audit_log",
        "simulation.utils.champion_log",
        "simulation.utils.model_artifact",
        "simulation.utils.doctor",
        "simulation.utils.eta",
        "simulation.utils.prune",
        "simulation.utils.auto_update",
        "simulation.utils.visualize",
        "simulation.utils.rehydrate",
        "simulation.models._optuna_subprocess",
        "simulation.models._optuna_samplers",
        "simulation.utils.feature_importance",
    ]
    for mod in modules:
        try:
            importlib.import_module(mod)
            _ok(f"{mod.split('.')[-1]}")
        except Exception as e:
            _fail(f"{mod.split('.')[-1]}: {e}")
    # --- 10. Config Summary ---
    print(f"\n  --- Config Summary ---")
    print(f"    Preset:         {config.preset}")
    print(f"    Optuna mode:    {config.optuna.mode}")
    print(f"    Optuna trials:  {config.optuna.trials}")
    print(f"    Epochs:         {config.training.epochs}")
    print(f"    Train ratio:    {config.split.train_ratio}")
    print(f"    WF-CV step:     {config.wfcv.step_size}")
    print(f"    WF-CV min_train:{config.wfcv.min_train_weeks}")
    print(f"    Resume from:    ordered index {config.resume_from_phase}")
    print(f"    Save dir:       {config.save_dir}")

    # --- Summary ---
    print(f"\n" + "=" * 60)
    print(f"  Result: {ok_count} OK, {warn_count} WARN, {fail_count} FAIL")
    if fail_count == 0:
        print(f"  >> All checks passed! Ready to run.")
    else:
        print(f"  >> {fail_count} critical issue(s) found. Fix before running.")
    print("=" * 60)

    return fail_count == 0


# ── G-237: critical-phase fail-loud gate ───────────────────────────────────
# P1/R9/R10 (operational forecast / HP-optimize / SSOT eval) are paper-critical:
# a silent {"error":…} → "계속 진행" there must never read as a successful run.
_CRITICAL_PHASES = {
    "real_eval":          f"{phases.display('P1')} operational forecast (real)",
    "per_model_optimize": f"{phases.display('R9')} HP-optimize",
    "per_model_eval":     f"{phases.display('R10')} SSOT eval",
}


# A critical phase may report {"skipped": True} for two very different reasons.
# "disabled" means the operator did not ask for the phase — legitimate. Any other
# reason means the phase intended to run and could not, which voids the run just
# as an error does.
_DELIBERATE_SKIP_REASONS = {"disabled"}


def _collect_critical_failures(all_results: dict) -> list:
    """Critical phases (P1/R9/R10) that failed or voided → [(label, reason_str), …].

    Pure function, no side effects. Empty list ⇒ no critical failure. Used by the
    end-of-run fail-loud gate so a voided operational forecast / SSOT eval surfaces loudly
    instead of passing as "Pipeline Complete!" (the 2026-05-30 incident, G-237).

    Catches two shapes:
      - ``{"error": …}``  — the phase raised.
      - ``{"skipped": True, "reason": …}`` with a reason other than "disabled" —
        the phase produced nothing. This shape used to pass the gate, and the way
        to reach it is to restart an interrupted run: ``all_results`` starts empty
        on every launch and is never rehydrated from the phase checkpoints, so
        resuming at R10 leaves the SSOT eval with no predictions to score. It
        returned ``{"skipped": True, "reason": "no predictions"}`` and the run
        reported success having evaluated nothing.

    Args:
        all_results: pipeline output dict keyed by phase name.

    Returns:
        list[tuple[str, str]]: (human label, reason string) per failed critical phase.
    """
    out = []
    for _key, _label in _CRITICAL_PHASES.items():
        node = all_results.get(_key)
        if not isinstance(node, dict):
            continue
        if node.get("error"):
            out.append((_label, str(node.get("error"))))
        elif node.get("skipped"):
            reason = str(node.get("reason", "no reason recorded"))
            if reason not in _DELIBERATE_SKIP_REASONS:
                out.append((_label, f"produced nothing (skipped: {reason})"))
    return out


# all_results phase-key → evaluation slab (from metric_rubric.py per-phase table).
_PHASE_HISTORY_SLAB = {
    "baseline": "test", "external": "test", "wfcv": "oof", "diagnostics": "oof",
    "dm_tests": "oof", "prediction_intervals": "oof",
    "scoring": "oof", "real_eval": "real", "per_model_optimize": "oof_cv",
    "per_model_eval": "test", "overseas": "country",
}


def _resolve_eval_features(X_all, feature_cols, *, eval_basic=True, basic_cols=None):
    """R2-P1 의 평가 feature set 을 결정 (사용자 2026-06-02: R2-P1 = BASIC, R9 = full).

    eval_basic 이면 BASIC(lag+계절성) 컬럼만 슬라이스해 반환 — R4(WF-CV champion 비교) +
    P1(real_eval/operational forecast) 가 이 좁은 feature 로 평가. R9 은 caller 가 full(phase1)
    을 그대로 넘겨 STABILITY/nested 로 per-model 선택(full=최종 fallback). 둘은 분리됨(영향 0).

    Args:
        X_all: (n, p) 전체 feature 행렬.
        feature_cols: 길이 p 의 컬럼명.
        eval_basic: True 면 BASIC 슬라이스, False 면 full 그대로 (MPH_EVAL_FEATURES).
        basic_cols: BASIC 컬럼명 (None → phase4_baseline.BASIC_FEATURE_COLS).
    Returns:
        (X_eval, feature_cols_eval, basic_idx | None). basic_idx=None 이면 full
        (eval_basic=False 또는 BASIC 컬럼 0개 → 안전 fallback).
    """
    if not eval_basic:
        return X_all, feature_cols, None
    if basic_cols is None:
        from .baseline import BASIC_FEATURE_COLS
        basic_cols = BASIC_FEATURE_COLS
    bset = set(basic_cols)
    idx = [i for i, c in enumerate(feature_cols) if c in bset]
    if not idx:
        return X_all, feature_cols, None
    return X_all[:, idx], [feature_cols[i] for i in idx], idx


def _write_metric_history(all_results, save_dir):
    """G-238: unified model × 129-key × phase comparison history (long-format CSV).

    Walks all_results, extracts every nested ``phase_eval_r8`` (the 129-key SSOT block
    from phase_evaluator.evaluate_predictions_full), and writes one row per
    (phase, model, slab, metric, value) to ``<save_dir>/eval_logs/metric_history.csv``.

    This is the cross-phase comparison artifact that previously did NOT exist: the
    129-key dicts were computed across R/P phases but only persisted nested in-memory
    (G-237/G-238 audit 2026-05-30 — eval_logs JSONL drained only 3 phases + flat keys).
    Long format → trivially pivotable to model×metric per phase. R10 still emits
    the wide test-slab CSV; this adds the missing cross-phase trajectory.

    Args:
        all_results: pipeline output dict keyed by phase name.
        save_dir: results dir (Path or str); CSV written under save_dir/eval_logs/.

    Returns:
        Path to the written CSV, or None if no phase_eval_r8 blocks were found.

    Side effects: writes one CSV. Pure read over all_results — never mutates it.
    Caller responsibility: wrap in try/except (a history-writer failure must not
    fail the pipeline).
    """
    import csv as _csv
    import re as _re
    from pathlib import Path as _Path
    rows = []

    # Metric-container keys must NOT masquerade as the model name. R9's
    # per-model results nest the 129-key block under e.g. {"<model>":
    # {"test_metrics": {phase_eval_r8: ...}}}; without this guard the walk
    # descended into "test_metrics" and labelled every model "test_metrics"
    # (real name only surviving inside phase_id like "phase13_refit_ARIMA").
    _CONTAINER_KEYS = {"test_metrics", "oof_metrics", "val_metrics", "metrics",
                       "test", "oof", "val", "oof_cv", "test_slab"}

    def _model_name(r8, model_key):
        if model_key in _CONTAINER_KEYS or model_key == "<phase>":
            pid = r8.get("phase_id") if isinstance(r8, dict) else None
            if isinstance(pid, str):
                recovered = _re.sub(r"^phase\d+_[a-z]+_", "", pid)
                if recovered and recovered != pid:
                    return recovered
        return model_key

    def _walk(node, phase, model_key):
        if not isinstance(node, dict):
            return
        r8 = node.get("phase_eval_r8")
        if isinstance(r8, dict):
            slab = _PHASE_HISTORY_SLAB.get(phase, "?")
            mk = _model_name(r8, model_key)
            for _k, _v in r8.items():
                if (_v is None or isinstance(_v, (int, float, str, bool))) and not _k.startswith("_"):
                    rows.append({"phase": phase, "model": mk,
                                 "slab": slab, "metric": _k, "value": _v})
        for _key, _val in node.items():
            if _key != "phase_eval_r8" and isinstance(_val, dict):
                # keep the parent's model name when descending into a container
                child_key = model_key if _key in _CONTAINER_KEYS else _key
                _walk(_val, phase, child_key)

    for _phase, _pdata in all_results.items():
        if isinstance(_pdata, dict):
            _walk(_pdata, _phase, "<phase>")

    # G-238 completeness: phase14 (R10 per_model_eval compat key) merges its 129-key R8 block
    # FLAT into per_model_metrics.csv instead of nesting phase_eval_r8, so the
    # walk above misses the most authoritative (test-slab, paper Table 1)
    # evaluation. Absorb that CSV as long-format rows when the walk produced
    # none for per_model_eval (guard avoids double-count if that ever changes).
    if not any(r["phase"] == "per_model_eval" for r in rows):
        _pme = all_results.get("per_model_eval")
        if isinstance(_pme, dict) and not _pme.get("error"):
            _pme_csv = _pme.get("metrics_csv")
            if _pme_csv and _Path(_pme_csv).exists():
                try:
                    with _Path(_pme_csv).open(encoding="utf-8") as _fh:
                        for _row in _csv.DictReader(_fh):
                            _m = _row.get("model")
                            if not _m:
                                continue
                            for _ck, _cv in _row.items():
                                if _ck == "model" or _ck.startswith("_") or _cv in (None, ""):
                                    continue
                                rows.append({"phase": "per_model_eval", "model": _m,
                                             "slab": "test", "metric": _ck, "value": _cv})
                except Exception as _pme_err:
                    log.warning(f"  [metric-history] phase14 CSV absorb failed: {_pme_err}")

    if not rows:
        return None
    out_dir = _Path(save_dir) / "eval_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metric_history.csv"
    with out_path.open("w", newline="", encoding="utf-8") as _fh:
        _w = _csv.DictWriter(_fh, fieldnames=["phase", "model", "slab", "metric", "value"])
        _w.writeheader()
        _w.writerows(rows)
    _n_pm = len({(r["phase"], r["model"]) for r in rows})
    log.info(f"  [metric-history] {len(rows)} cells across {_n_pm} (phase,model) → {out_path}")
    return out_path


def repair_metric_history(csv_path):
    """One-off repair for metric_history.csv files written before the
    container-key guard: re-key rows whose model column is a metric-container
    sentinel (e.g. ``test_metrics``) back to the real model name carried in the
    block's ``phase_id`` value (``phase13_refit_ARIMA`` -> ``ARIMA``).

    Args:
        csv_path: path to an existing metric_history.csv.

    Returns:
        Number of rows re-keyed (0 if the file is already clean / absent).

    Side effects: rewrites ``csv_path`` in place when any row is repaired.
    Idempotent — running it twice changes nothing the second time.
    """
    import csv as _csv
    import re as _re
    from pathlib import Path as _Path
    _containers = {"test_metrics", "oof_metrics", "val_metrics", "metrics",
                   "test", "oof", "val", "oof_cv", "test_slab", "<phase>"}
    p = _Path(csv_path)
    if not p.exists():
        return 0
    rows = list(_csv.DictReader(p.open(encoding="utf-8")))
    current: dict = {}
    repaired = 0
    for r in rows:
        if r.get("model") in _containers:
            if r.get("metric") == "phase_id" and isinstance(r.get("value"), str):
                name = _re.sub(r"^phase\d+_[a-z]+_", "", r["value"])
                if name and name != r["value"]:
                    current[r["phase"]] = name
            name = current.get(r["phase"])
            if name:
                r["model"] = name
                repaired += 1
    if repaired:
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = _csv.DictWriter(fh, fieldnames=["phase", "model", "slab", "metric", "value"])
            w.writeheader()
            w.writerows(rows)
    return repaired


def run_pipeline(config=None):
    """V16 Pipeline full execution.

    R/P registry dispatch (R/P label mechanical rename):
        R1=data, R2=baseline, R3=external, R4=WF-CV, R5=diagnostics,
        R6=DM_test, R7=intervals, R8=scoring, R9=per_model_optimize,
        R10=per_model_eval, P1=real_forecaster, R11=SHAP/XAI,
        R12=comprehensive_eval, P2=inference(별도 CLI), P3=overseas(별도 CLI).
    Legacy numbers are accepted only at the CLI boundary through phases.resume_index().
    """
    from .config import PipelineConfig
    from .utils.logging_util import setup_logging, fmt_time
    from .utils.memory import MemoryGuard
    from .utils.checkpoint import CheckpointManager

    if config is None:
        parser = build_cli_parser()
        args = parser.parse_args()
        config = PipelineConfig.from_cli(args)

    # --- Dry-run mode ---
    if config.dry_run:
        success = run_dry_run(config)
        sys.exit(0 if success else 1)

    # --- determinism block (S1-2 / S1-5) -----------------------
    # MPH 논문 재현성 확보. cudnn deterministic + warn_only=True.
    # GPU 10-30% 느려지지만 학술 재현성이 성능보다 우선.
    _seed = int(getattr(config, "seed", 42) or 42)
    import os as _os, random as _random
    _os.environ["PYTHONHASHSEED"] = str(_seed)
    _os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    _random.seed(_seed)
    try:
        import numpy as _np
        _np.random.seed(_seed)
    except ImportError:
        pass
    try:
        import torch as _torch
        _torch.manual_seed(_seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(_seed)
        _torch.backends.cudnn.deterministic = True
        _torch.backends.cudnn.benchmark = False
        try:
            _torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    except ImportError:
        pass
    log.info(f"  [seed] determinism locked at seed={_seed} (cudnn.deterministic=True)")

    # --- Force overwrite check ---
    save_dir = config.get_save_dir()
    report_path = save_dir / config.output.report_name
    checkpoint_dir = save_dir / "checkpoints"
    if report_path.exists() and not config.force_overwrite and config.resume_from_phase == 0:
        print(f"\n  [WARNING] Results already exist: {report_path}")
        print(f"  Use --force to overwrite, or --resume-from N to continue.")
        sys.exit(1)
    if config.force_overwrite:
        import shutil
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
            log.info(f"  Force overwrite: cleared checkpoints in {checkpoint_dir}")
        if report_path.exists():
            report_path.unlink()
            log.info(f"  Force overwrite: removed {report_path}")
        # --- : force also clears Optuna JSON caches so studies restart fresh ---
        # R3 external Optuna reuses optuna_feat_sel_*.json / optuna_all_strategies_*.json
        # from save_dir when present. Without clearing them, --force only wipes
        # checkpoints but Optuna feature-selection results are silently reused.
        optuna_json_patterns = ("optuna_feat_sel_*.json",
                                 "optuna_all_strategies_*.json")
        n_cleared = 0
        for pat in optuna_json_patterns:
            for p in save_dir.glob(pat):
                try:
                    p.unlink()
                    n_cleared += 1
                except OSError as e:
                    log.warning(f"  Could not delete {p}: {e}")
        if n_cleared:
            log.info(f"  Force overwrite: cleared {n_cleared} Optuna JSON cache files")
        # Also clear saved model artifacts so models re-fit from scratch.
        # SAFETY: config.training.model_dir defaults to './models' (relative).
        # Refuse to rmtree unless the resolved path lives inside save_dir or
        # the package root, so a stray cwd can never nuke unrelated folders.
        model_dir = None
        try:
            md = config.get_model_dir().resolve()
            sd = save_dir.resolve()
            try:
                from pathlib import Path as _Path
                pkg_root = _Path(__file__).resolve().parents[2]
            except Exception:
                pkg_root = sd
            def _is_inside(child, parent):
                try:
                    child.relative_to(parent)
                    return True
                except ValueError:
                    return False
            if _is_inside(md, sd) or _is_inside(md, pkg_root):
                model_dir = md
            else:
                log.warning(
                    f"  Force overwrite: refusing to delete model dir "
                    f"{md} (outside save_dir/pkg_root)"
                )
        except AttributeError as e:
            log.warning(f"  Force overwrite: could not resolve model_dir: {e}")
        if model_dir is not None and model_dir.exists():
            shutil.rmtree(model_dir, ignore_errors=True)
            log.info(f"  Force overwrite: cleared model dir {model_dir}")
        # Also clear R2 baseline sidecar (stale-merge prevention).
        # When the user passes --models X,Y the sidecar would otherwise
        # merge in 50+ stale model results from prior FULL runs, so the
        # "61 models evaluated in R8" includes models that weren't
        # trained under the current config (covid mode, weather mode...).
        # Clear it on --force so the current run is the only source.
        for sidecar_name in ("phase4_baseline_sidecar.pkl",
                              "checkpoint_R4_oof.pkl",):
            p = save_dir / sidecar_name
            if p.exists():
                try:
                    p.unlink()
                    log.info(f"  Force overwrite: cleared {sidecar_name}")
                except OSError as e:
                    log.warning(f"  Could not delete {p}: {e}")
        # --- : --no-cache also wipes FE parquet cache on disk ---
        if getattr(config, "no_cache", False):
            from pathlib import Path as _P
            cache_path = _P(config.data.cache_dir) / "feature_cache.parquet"
            if cache_path.exists():
                try:
                    cache_path.unlink()
                    log.info(f"  --no-cache: removed FE cache {cache_path}")
                except OSError as e:
                    log.warning(f"  Could not delete {cache_path}: {e}")

    # --- Normal execution ---
    log_file = setup_logging(config.output.log_dir, config.output.structured_logging)
    log.info(f"  V16 Pipeline Start")
    log.info(f"  Log: {log_file}")
    log.info(f"  Save: {config.get_save_dir()}")
    log.info(f"  Optuna mode: {config.optuna.mode}")
    log.info(f"  Optuna trials: {config.optuna.trials}, epochs: {config.optuna.epochs_per_trial}")
    log.info(f"  Train/Test: {config.split.train_ratio}/{config.split.test_ratio}")
    log.info(f"  WF-CV: min_train={config.wfcv.min_train_weeks}, step={config.wfcv.step_size}")
    t_global = time.time()
    mem = MemoryGuard(config.memory.min_free_mb, config.memory.use_float32)
    ckpt = CheckpointManager(config.get_save_dir())
    resume_from = 0 if config.resume_from_phase is None else config.resume_from_phase

    # Phase boundary banner — wraps ckpt.start_timer / stop_timer so each
    # R/P entry prints "<label> <name> · ETA <range> · prev <elapsed>"
    # without touching every phase block. Bypass by env: NO_PHASE_BANNER=1.
    # G-251: per-phase resource/timing accounting — populated by the wrapped
    # start/stop timers below, summarized in the Final Report. Defined at function
    # scope (not inside the banner block) so the summary survives NO_PHASE_BANNER=1.
    _phase_res_log: list = []          # [{phase, elapsed_s, rss_gb, peak_gb, sys_cpu}]
    _peak_rss = [0.0]                  # running peak RSS (GB, incl. children)
    _phase_t0: dict = {}              # phase → (t_start, rss_at_start)

    def _res_snapshot():
        """(rss_gb incl children, sys_cpu%, sys_mem%, n_children). Safe zeros on error."""
        try:
            import psutil
            p = psutil.Process()
            rss = p.memory_info().rss
            nch = 0
            for c in p.children(recursive=True):
                try:
                    rss += c.memory_info().rss
                    nch += 1
                except Exception:
                    pass
            return (rss / 1e9, psutil.cpu_percent(interval=0.1),
                    psutil.virtual_memory().percent, nch)
        except Exception:
            return (0.0, 0.0, 0.0, 0)

    if not os.environ.get("NO_PHASE_BANNER"):
        try:
            from simulation.utils.eta import print_phase_banner as _phase_banner
            _prev_phase_elapsed = [0.0]
            _orig_start = ckpt.start_timer
            _orig_stop = ckpt.stop_timer

            def _phase_label(_phase) -> Optional[str]:
                try:
                    return phases.label_of(_phase)
                except Exception:
                    return None

            def _phase_display(_phase) -> str:
                try:
                    return phases.display(_phase)
                except Exception:
                    return str(_phase)

            def _wrapped_start(_n):
                try:
                    prev = _prev_phase_elapsed[0]
                    _lbl = _phase_label(_n)
                    if _lbl is not None:
                        _phase_banner(_lbl, prev_elapsed_sec=prev if prev > 0 else None)
                    _r = _res_snapshot()                      # G-251 자원 스냅샷
                    _phase_t0[_n] = (time.time(), _r[0])
                    log.info(f"  ▶ {_phase_display(_n)} 시작 | RSS={_r[0]:.2f}GB "
                             f"sysCPU={_r[1]:.0f}% sysMEM={_r[2]:.0f}% children={_r[3]}")
                except Exception as _e:
                    log.debug(f"[phase-banner] {_n} skipped: {_e}")
                return _orig_start(_n)

            def _wrapped_stop(_n):
                el = _orig_stop(_n)
                try:
                    _prev_phase_elapsed[0] = float(el or 0.0)
                    _r = _res_snapshot()                      # G-251 완료시 자원 + Δ + peak
                    _peak_rss[0] = max(_peak_rss[0], _r[0])
                    _rss0 = _phase_t0.get(_n, (None, _r[0]))[1]
                    log.info(f"  ✓ {_phase_display(_n)} 완료 [{fmt_time(el or 0.0)}] | "
                             f"RSS={_r[0]:.2f}GB (Δ{_r[0] - _rss0:+.2f}) "
                             f"peak={_peak_rss[0]:.2f}GB sysCPU={_r[1]:.0f}% children={_r[3]}")
                    _phase_res_log.append({"phase": _n, "elapsed_s": float(el or 0.0),
                                           "rss_gb": round(_r[0], 2),
                                           "peak_gb": round(_peak_rss[0], 2),
                                           "sys_cpu": round(_r[1], 0)})
                except Exception:
                    pass
                return el

            ckpt.start_timer = _wrapped_start  # type: ignore[assignment]
            ckpt.stop_timer = _wrapped_stop    # type: ignore[assignment]
        except Exception as _be:
            log.debug(f"[phase-banner] wiring skipped: {_be}")

    all_results = {}

    # ───── Resume: bring back what earlier phases already produced ─────
    # Without this the dict stays empty on every launch, so a run resumed at R10
    # reaches the evaluation phases with nothing to evaluate and reports success
    # having evaluated nothing. R2 and R4 cannot be restored — their checkpoints
    # store a model count and a subset, not the predictions — so the gap is
    # logged rather than papered over: the champion pool really is narrower on a
    # resumed run, and the operator needs to know that before trusting it.
    if resume_from:
        try:
            from simulation.pipeline.rehydrate import (
                LABEL_TO_KEY,
                rehydrate_all_results,
            )

            _state = rehydrate_all_results(Path(config.get_save_dir()))
            # resume_from is already the ordered index (resolve_resume_from ran
            # at parse time). Passing it back through resume_index() raises,
            # because numbers are no longer accepted as phase references.
            _resume_idx = int(resume_from)
            _key_to_label = {v: k for k, v in LABEL_TO_KEY.items()}
            for _key, _val in _state.results.items():
                # Only phases that ran BEFORE the resume point; anything at or
                # after it is about to be recomputed and must not be preloaded.
                _lbl = _key_to_label.get(_key)
                if _lbl and phases.order(_lbl) < _resume_idx:
                    all_results[_key] = _val
            log.info(f"  [resume] rehydrated {len(all_results)} phase results from disk")
            for _k, _why in _state.missing.items():
                log.warning(f"  [resume] {_k} NOT restored — {_why}")
        except Exception as _re:
            log.warning(f"  [resume] rehydration skipped: {type(_re).__name__}: {_re}")

    # ───── Initialize EvalLogger + AuditLog ─────
    # Structured JSONL evaluation log + reproducibility audit metadata.
    # Every phase writes to this; R9 reads it for the master report.
    global _EVAL_LOGGER
    try:
        from simulation.utils.eval_logger import EvalLogger
        from simulation.utils.audit_log import capture_audit
        _EVAL_LOGGER = EvalLogger.from_config(config)
        _EVAL_LOGGER.log_audit(capture_audit(config))
        log.info(f"  [audit] eval_log: {_EVAL_LOGGER.jsonl_path}")
        log.info(f"  [audit] audit:    {_EVAL_LOGGER.audit_path}")
    except Exception as e:
        log.warning(f"  [audit] EvalLogger init failed: {e} — continuing without it")
        _EVAL_LOGGER = None

    def _log_phase_metrics(phase: str, results: dict, slab: str = "?"):
        """Helper: drain a phase's per-model metric dict into EvalLogger."""
        if _EVAL_LOGGER is None or not isinstance(results, dict):
            return
        # Common shape: {"model_results": {model: {metric: value}}}
        per_model = results.get("model_results") or results.get("metrics") or {}
        if isinstance(per_model, dict):
            for m, mr in per_model.items():
                if isinstance(mr, dict):
                    _EVAL_LOGGER.log_metrics_dict(phase=phase, model=m,
                                                    metrics=mr, slab=slab)
                    # G-238: also drain the nested 129-key SSOT block — log_metrics_dict
                    # skips dicts >20 keys, so phase_eval_r8 (129) would otherwise be lost.
                    _r8 = mr.get("phase_eval_r8")
                    if isinstance(_r8, dict):
                        _EVAL_LOGGER.log_metrics_dict(phase=phase, model=m,
                                                        metrics=_r8, slab=slab)

    # ===== R1: Data =====
    if phases.should_run("R1", resume_from):
        from .data import run_data
        ckpt.start_timer("R1")
        phase1 = run_data(config)
        ckpt.save("R1", {k: v for k, v in phase1.items()
                       if k not in ("X_all", "y_all", "dates")}, "Data + FE")
        all_results["phase1"] = phase1
        log.info(f"  {phases.display('R1')} elapsed: {fmt_time(ckpt.stop_timer('R1'))}")
    else:
        phase1 = ckpt.load("R1")
        if phase1 is None:
            log.error(f"  {phases.display('R1')} checkpoint missing -- restart from beginning")
            sys.exit(1)
        from .data import run_data
        phase1 = run_data(config)
        all_results["phase1"] = phase1

    X_all = phase1["X_all"]
    y_all = phase1["y_all"]
    feature_cols = phase1["feature_cols"]
    holdout_start = phase1.get("holdout_start", len(y_all))  # S0-1
    # ── EVAL feature set (사용자 2026-06-02): R2-P1 = BASIC(lag+계절성); feature 최적화는 R9
    #    (full pool, full=최종 fallback) 에서만. R9 은 full(R1) 그대로. MPH_EVAL_FEATURES=full 회귀.
    import os as _os_ef
    _eval_basic = _os_ef.environ.get("MPH_EVAL_FEATURES", "basic").strip().lower() == "basic"
    X_eval, feature_cols_eval, _basic_idx = _resolve_eval_features(X_all, feature_cols, eval_basic=_eval_basic)
    if _basic_idx is not None:
        log.info(f"  [eval-features] BASIC {len(_basic_idx)}/{len(feature_cols)} cols → R2-P1 (R9 = full pool)")
    elif _eval_basic:
        log.warning("  [eval-features] BASIC cols 0 발견 → full fallback")
    # F2: inject week_start into DataConfig so R6 dm_test and
    # R7 intervals can use calendar-accurate COVID regime boundaries
    # instead of the 47/36/17 proportional fallback.
    dates = phase1.get("dates")
    if dates is not None:
        config.data.dates = dates
        log.info(
            f"  [F2] dates 벡터 주입: n={len(dates)}, "
            f"range=[{dates[0]} → {dates[-1]}] → R3/R4"
        )
    else:
        log.warning(
            "  [F2] phase1 이 dates 를 돌려주지 않음 — R3/R4 는 "
            "47/36/17 proportional fallback 을 사용한다."
        )
    gc.collect()

    # --- S0-3 fix: cap Optuna trials as a function of n_train ------
    # At n=343 weeks with train_ratio=0.8 we get n_train≈274. trials=100
    # is way too many: in a p≈n regime this overfits the TPE surrogate
    # and produces fragile best_params. Cap at min(50, 0.3 * n_train).
    # Use the non-holdout window so the cap reflects what models see.
    try:
        n_total = len(y_all)
        n_main = holdout_start  # S0-1: effective training window
        n_train = int(n_main * config.split.train_ratio)
        dyn_cap = max(10, min(50, int(0.3 * n_train)))
        old_trials = config.optuna.trials
        if old_trials > dyn_cap:
            config.optuna.trials = dyn_cap
            config.optuna.n_trials = dyn_cap
            log.info(f"  [S0-3] Optuna trials capped: {old_trials} -> {dyn_cap} "
                     f"(n_train={n_train}, rule=min(50, 0.3*n_train))")
        # : per_model_trials 를 dyn_cap 상한에 맞춰 재조정 후
        # env-var 로 subprocess worker 에 전달
        # 2026-04-28: MPH_FAST_TRAIN=1 시 추가 50% cut (smoke / 빠른 검증)
        try:
            from simulation.models._optuna_budget import set_budget
            _pmt = {k: min(v, dyn_cap) for k, v in config.optuna.per_model_trials.items()}

            _fast = GLOBAL.training.fast_train
            if _fast:
                _pmt = {k: max(5, v // 2) for k, v in _pmt.items()}
                log.info(f"  [MPH_FAST_TRAIN=1] trials 50% cut applied")

            set_budget(_pmt)
            log.info(f" [] per_model_trials 설정: {_pmt}")
        except Exception as _be:
            log.warning(f" [] per_model_trials 설정 실패: {_be}")
    except Exception as _e:
        log.warning(f"  [S0-3] could not cap Optuna trials: {_e}")

    # ===== Legacy Phase 2-3: 의도적으로 비움 (2026-06-01, 사용자 명시) =====
    # feature 선택(STABILITY) + multicollinearity 는 **R9 (per-model) 에서만** 수행.
    # R2~P1 는 **BASIC eval feature**(lag+계절성 13; MPH_EVAL_FEATURES=basic, _resolve_eval_features
    #   → X_eval) 위에서 모델 비교 — champion gate(P1)도 BASIC. full pool(401)은 **R9 에서만**
    #   per-model STABILITY 선택(full=최종 fallback). 사용자 명시 2026-06-02.
    #   · 옛 frozen pre-stage feature-optuna LOAD 폐기 (stale/degenerate n/p≈2.8; B2/B4 2026-06-01).
    #   · global stability stage 도 폐기 — feature 선택은 R9 에서만 (사용자 명시 2026-06-01).
    # per_model_feature_map: R4 가 받음 (default 빔 → per_model_features=None = 주어진 X_eval 전체 사용).
    #   2026-06-02 codex+Gemini fix: external(R3)은 `full` 시나리오서 제거(optuna_mode=none) →
    #   이 map 은 항상 빔. R9 은 자체 per-model STABILITY(이 map 무관). R10 은 미사용.
    per_model_feature_map = {}
    per_model_pca_feat_set: dict = {}  # external 제거(2026-06-02) → 항상 빔

    mem.check_and_gc("pre-R2 GC (feature/mc 는 R9 에서만)")

    # ===== R2: Baseline Training =====

    if phases.should_run("R2", resume_from):
        ckpt.start_timer("R2")

        if config.optuna.mode in ("none", "all"):
            from .baseline import run_baseline
            log.info(f"  === {phases.display('R2')}: Baseline Training ===")
            baseline = run_baseline(X_all, y_all, feature_cols, config)
            all_results["baseline"] = baseline
            ckpt.save("R2", {"baseline_n_models": baseline.get("n_models", 0)}, "Baseline")
            _log_phase_metrics("phase4_baseline", baseline, slab="test_baseline")

        log.info(f"  {phases.display('R2')} elapsed: {fmt_time(ckpt.stop_timer('R2'))}")

    # ===== R3: External Optuna Training =====

    if phases.should_run("R3", resume_from):
        ckpt.start_timer("R3")

        if config.optuna.mode in ("external", "all"):
            from .external import run_external
            log.info(f"  === {phases.display('R3')}: External Optuna Training ===")
            external = run_external(X_all, y_all, feature_cols, config)
            all_results["external"] = external
            for mname, info in external.get("feature_selection_log", {}).items():
                if "features" in info:
                    per_model_feature_map[mname] = info["features"]

        if config.optuna.mode in ("inline", "all"):
            log.info(f"  === {phases.display('R3')}b: Inline Optuna -> {phases.display('R4')} ===")

        log.info(f"  {phases.display('R3')} elapsed: {fmt_time(ckpt.stop_timer('R3'))}")

    mem.check_and_gc("R2/R3 (training) done")

    # (G-232a mc_filter 는 R9 per-model 로 이동 — 2026-06-01 B2; legacy phase 2-3 = 빔. 옛 위치 흔적 주석.)

    # ===== R4: Walk-Forward CV =====
    wf_data = {}
    if phases.should_run("R4", resume_from):
        from .wfcv import run_wfcv
        ckpt.start_timer("R4")
        wf_data = run_wfcv(
            X_eval, y_all, feature_cols_eval, config,   # BASIC eval features (R9 = full)
            per_model_feature_map=per_model_feature_map,
            per_model_pca_feat_set=per_model_pca_feat_set or None,
            memory_guard=mem,
            holdout_start=holdout_start,   # S0-1
        )
        all_results["wfcv"] = wf_data
        _log_phase_metrics("phase6_wfcv", wf_data, slab="oof")
        # F3: fold_holdout_predictions is a dict of (K, H) matrices
        # per model — large, and only needed in-memory for R4 CV+.
        # Skip it in the checkpoint dump.
        _wf_ckpt = {k: v for k, v in wf_data.items()
                    if k not in ("oof_predictions", "fold_holdout_predictions")}
        ckpt.save("R4", _wf_ckpt, "WF-CV")
        # B-4: persist R4 array outputs as a pickle sidecar so that
        # --resume-from R4/R8 can re-run the light downstream phases without
        # redoing R4 (~60min of the 1h45m budget). Fields: everything
        # R5/R6/R7/R8/R11 reads from wf_data besides what's in checkpoint_R4.json.
        # partial-refit: 만약 --models 필터로 이번 R4 가 일부 모델만
        #   재학습한 경우, 기존 sidecar 를 먼저 로드한 뒤 새 결과를 덮어써서
        #   나머지 모델의 OOF/holdout 을 유지한다.
        try:
            import pickle
            _sidecar_path = ckpt.checkpoint_dir / "checkpoint_R4_oof.pkl"
            _partial = bool(getattr(config, "_selected_models", None))
            _merged = {
                "oof_predictions": {},
                "holdout_predictions": {},
                "fold_holdout_predictions": {},
                "fold_val_indices": {},
            }
            if _partial and _sidecar_path.exists():
                try:
                    with _sidecar_path.open("rb") as _f:
                        _prev = pickle.load(_f)
                    for _k in _merged.keys():
                        _merged[_k].update(_prev.get(_k, {}) or {})
                    log.info(
                        f"  [partial] 기존 sidecar 로드: "
                        f"OOF {len(_merged['oof_predictions'])}개, "
                        f"holdout {len(_merged['holdout_predictions'])}개 유지"
                    )
                except Exception as _pe:
                    log.warning(f"  [partial] 기존 sidecar 로드 실패 → 새로 작성: {_pe}")
            # 이번 run 의 결과 덮어쓰기 (partial 이면 merge, full 이면 override)
            for _k in _merged.keys():
                _merged[_k].update(wf_data.get(_k, {}) or {})
            with _sidecar_path.open("wb") as _f:
                pickle.dump(_merged, _f)
            # : 다운스트림 phase 가 merged sidecar 를 보도록 wf_data 도 갱신
            if _partial:
                wf_data["oof_predictions"] = _merged["oof_predictions"]
                wf_data["holdout_predictions"] = _merged["holdout_predictions"]
                wf_data["fold_holdout_predictions"] = _merged["fold_holdout_predictions"]
                wf_data["fold_val_indices"] = _merged["fold_val_indices"]
            log.info(
                f"  [B-4] R4 sidecar 저장: {_sidecar_path} "
                f"(OOF {len(_merged['oof_predictions'])}개, "
                f"holdout {len(_merged['holdout_predictions'])}개"
                + (", partial merge" if _partial else "")
                + ")"
            )
        except Exception as _e:
            log.warning(f"  [B-4] R4 sidecar 저장 실패: {_e}")
        log.info(f"  {phases.display('R4')} elapsed: {fmt_time(ckpt.stop_timer('R4'))}")
    else:
        # B-4: resume path — reload sidecar so R5/R6/R7/R8/R11
        # don't crash with empty oof/holdout predictions.
        try:
            import pickle
            _sidecar_path = ckpt.checkpoint_dir / "checkpoint_R4_oof.pkl"
            if _sidecar_path.exists():
                with _sidecar_path.open("rb") as _f:
                    _sc = pickle.load(_f)
                wf_data["oof_predictions"] = _sc.get("oof_predictions", {})
                wf_data["holdout_predictions"] = _sc.get("holdout_predictions", {})
                wf_data["fold_holdout_predictions"] = _sc.get(
                    "fold_holdout_predictions", {}
                )
                wf_data["fold_val_indices"] = _sc.get("fold_val_indices", {})
                log.info(
                    f"  [B-4] R4 sidecar 로드: "
                    f"OOF {len(wf_data['oof_predictions'])}개, "
                    f"holdout {len(wf_data['holdout_predictions'])}개 복원"
                )
            else:
                log.warning(
                    f"  [B-4] R4 sidecar 없음 → R4/R6 빈 OOF 로 동작"
                )
        except Exception as _e:
            log.warning(f"  [B-4] R4 sidecar 로드 실패: {_e}")

    oof_predictions = wf_data.get("oof_predictions", {})
    oof_predictions = {k: v for k, v in oof_predictions.items()
                       if v is not None and hasattr(v, '__len__')}
    mem.check_and_gc("R4 done")

    # ===== R5: Residual Diagnostics =====
    if phases.should_run("R5", resume_from):
        from .diagnostics import run_diagnostics
        ckpt.start_timer("R5")
        diag = run_diagnostics(y_all, oof_predictions, config)
        all_results["diagnostics"] = diag
        ckpt.save("R5", diag, "Diagnostics")

    # ===== R6: DM Test =====
    if phases.should_run("R6", resume_from):
        from .dm_test import run_dm_test
        ckpt.start_timer("R6")
        dm = run_dm_test(y_all, oof_predictions, config)
        all_results["dm_tests"] = dm
        ckpt.save("R6", dm, "DM Test")

    # ===== R7: Prediction Intervals =====
    if phases.should_run("R7", resume_from):
        from .intervals import run_intervals
        ckpt.start_timer("R7")
        holdout_preds_map = wf_data.get("holdout_predictions", {})
        # F3: pass per-fold holdout matrices so R4 can compute CV+
        fold_holdout_map = wf_data.get("fold_holdout_predictions", {})
        fold_val_idx_map = wf_data.get("fold_val_indices", {})
        pi = run_intervals(
            y_all, oof_predictions, config,
            holdout_predictions=holdout_preds_map,
            holdout_start=holdout_start,
            fold_holdout_predictions=fold_holdout_map,
            fold_val_indices=fold_val_idx_map,
        )
        all_results["prediction_intervals"] = pi
        ckpt.save("R7", pi, "PI")

    mem.check_and_gc("R7 done")

    # ===== R8: Composite Scoring =====
    if phases.should_run("R8", resume_from):
        from .scoring import run_scoring
        ckpt.start_timer("R8")
        scoring = run_scoring(
            wf_results=all_results.get("wfcv", {}),
            dm_results=all_results.get("dm_tests", {}),
            pi_results=all_results.get("prediction_intervals", {}),
            config=config,
            oof_predictions=oof_predictions,
            y_all=y_all,
            holdout_start=holdout_start,
        )
        all_results["scoring"] = scoring
        ckpt.save("R8", scoring, "Composite Scoring")

    # ===== P1 (real_eval) RELOCATED → now runs AFTER R10 (see below) =====
    # G-306 Step2 (2026-06-17): real_eval moved to after per_model_eval(R10) so the
    # operational / deployment forecast (gate → ABM/ARIA) uses the FINAL best-WIS
    # champion + its R9 OPTIMIZED real prediction, NOT a default-HP stand-in
    # (research ⊥ production: real-slab metrics = R9 service-zone; deployment
    # gate operates on the confirmed champion). The stale R9-swap note was removed —
    # its claim (scoring-after-per_model_optimize) did not match the live dispatch.

    # ===== R9: Per-Model INDIVIDUAL OPTIMIZATION (preproc × HP) ====
    # Each model gets its own best (target transform, scaler, feature subset)
    # via WF-CV grid search. Heavy: ~hours for 50+ models. Opt-in via flag.
    if phases.should_run("R9", resume_from) and (
        bool(getattr(config, "per_model_optimize", False)) or
        bool(getattr(config.split, "per_model_optimize", False))
    ):
        from .per_model_optimize import run_per_model_optimize
        ckpt.start_timer("R9")
        # ABLATION A1 (2026-06-02, 엄밀 ablation): MPH_PHASE13_FEATURE_POOL=basic 면 R9 을
        #   BASIC pool 에서 최적화(preproc+HP) → A1 arm(=BASIC+tune). A2(기본)=full pool. 둘의 차이가
        #   ΔFeature 격리. main run(=full)엔 무영향. 별도 A1 run: MPH_PHASE13_FEATURE_POOL=basic launch.
        import os as _os_a1
        _p13 = phase1
        if _os_a1.environ.get("MPH_PHASE13_FEATURE_POOL", "full").strip().lower() == "basic":
            from .baseline import BASIC_FEATURE_COLS as _BFC
            _Xb, _fcb, _bidx = _resolve_eval_features(
                phase1["X_all"], phase1["feature_cols"], eval_basic=True, basic_cols=_BFC)
            _p13 = dict(phase1); _p13["X_all"] = _Xb; _p13["feature_cols"] = _fcb
            if _bidx is not None and phase1.get("real_X") is not None:
                _p13["real_X"] = phase1["real_X"][:, _bidx]
            log.info(f"  [ABLATION A1] R9 = BASIC pool ({len(_fcb)} feat) — HP/preproc 격리 arm")
        try:
            per_model_opt = run_per_model_optimize(_p13, all_results, config)
            all_results["per_model_optimize"] = per_model_opt
            ckpt.save("R9", per_model_opt, "Per-Model Optimize")
            log.info(f"  {phases.display('R9')} elapsed: {fmt_time(ckpt.stop_timer('R9'))}")
        except Exception as e:
            log.error(f"  [CRITICAL] {phases.display('R9')} 실패: {type(e).__name__}: {e}")
            all_results["per_model_optimize"] = {"error": f"{type(e).__name__}: {e}", "critical": True}

    # ===== Sprint E2 (R4 큰 변경, opt-in): R4 re-OOF with optimized HP =====
    # 사용자 명시 (Sprint E2): "R4 re-OOF with optimized configs"
    # Codex/Gemini HARD-FIX: R4 OOF 가 baseline HP → R8 SSOT 가 un-tuned 평가.
    # Fix: R9 best HP 받아 R4 다시 학습 → paper SSOT 가 tuned OOF 평가.
    # Cost: 학습 wall-clock 2배 → opt-in via MPH_ENABLE_PHASE6_REOPT=1.
    # 현재는 scaffolding — actual re-OOF logic 은 별도 sprint (phase6_wfcv signature 변경 필요).
    if GLOBAL.ops.enable_phase6_reopt:
        log.warning(
            "  [E2] MPH_ENABLE_PHASE6_REOPT=1 — R4 re-OOF with optimized configs "
            "is scaffolded but NOT YET IMPLEMENTED. Wall-clock 2x impact expected. "
            "See Sprint E2 (2026-05-26) for next-sprint implementation plan."
        )
        # TODO: per_model_opt 의 best_config 받아서 phase6_wfcv 재호출 → all_results["wfcv"] 덮어쓰기
        # TODO: R8 input source 가 새 OOF 사용

    # ===== R10: Per-Model UNIFORM Evaluation on Test Slab (n=68) ====
    # R10: SSOT 129-key row reflects optimized HP.
    if phases.should_run("R10", resume_from):
        from .per_model_eval import run_per_model_eval
        ckpt.start_timer("R10")
        try:
            per_model = run_per_model_eval(phase1, all_results, config)
            all_results["per_model_eval"] = per_model
            ckpt.save("R10", per_model, "Per-Model Eval")
            if _EVAL_LOGGER is not None and per_model.get("metrics_csv"):
                # Drain R10's CSV into the JSONL log
                try:
                    import csv as _csv
                    with open(per_model["metrics_csv"], encoding="utf-8") as fh:
                        for row in _csv.DictReader(fh):
                            mname = row.pop("model", "?")
                            for k, v in row.items():
                                if v not in (None, ""):
                                    _EVAL_LOGGER.log(phase="phase14", model=mname,
                                                     metric=k, value=v, slab="test")
                except Exception as _e:
                    log.debug(f"  [audit] R10 drain failed: {_e}")
            log.info(f"  {phases.display('R10')} elapsed: {fmt_time(ckpt.stop_timer('R10'))}")
        except Exception as e:
            log.error(f"  [CRITICAL] {phases.display('R10')} 실패: {type(e).__name__}: {e}")
            all_results["per_model_eval"] = {"error": f"{type(e).__name__}: {e}", "critical": True}

    # ===== P1 (real_eval): MOVED → now runs AFTER R12 (see below, before Final Report) =====
    # 2026-06-20 (사용자 지시): P1 relocated to the production-track start = after ALL R
    # (R9 optimize → R10 eval → R11 SHAP → R12 comprehensive). R12 is now DECOUPLED from
    # real_eval — the comprehensive report sources its champion + families from R9
    # (per_model_optimize) + R10 (per_model_eval), NOT from P1's real-slab. P1 still uses
    # the R9 best-WIS champion and feeds the ABM/ARIA deployment gate, just at the end.

    # ===== R11: SHAP =====
    if phases.should_run("R11", resume_from):
        from .shap_analysis import run_shap
        ckpt.start_timer("R11")
        try:
            shap_data = run_shap(X_all, y_all, feature_cols, config)
            all_results["feature_importance"] = shap_data
            ckpt.save("R11", shap_data, "SHAP")
        except Exception as e:
            log.error(f"  {phases.display('R11')} (SHAP) 실패 — 파이프라인 계속 진행: {e}")
            all_results["feature_importance"] = {"error": str(e)}
            ckpt.save("R11", {"error": str(e)}, "SHAP (failed)")

    # R11 (XAI): SHAP + Permutation Importance on the trained champion.
    # Runs after R10 eval so it uses the champion .pt (not a stale model).
    # SHAP (above) + XAI together = R11. opt-out: --no-xai / MPH_NO_XAI=1.
    _xai_skip = (bool(getattr(config, "no_xai", False))
                 or GLOBAL.ops.no_xai)
    if phases.should_run("R11", resume_from) and not _xai_skip:
        from .xai import run_xai
        try:
            xai = run_xai(phase1, all_results, config)
            all_results["xai"] = xai
            log.info(f"  {phases.display('R11')} (XAI) status: {xai.get('status', 'completed')}")
        except Exception as e:
            log.warning(f"  {phases.display('R11')} (XAI) 실패 (non-fatal): {e}")
            all_results["xai"] = {"error": str(e)}

    # ===== R12: Comprehensive Evaluation Aggregator =====
    # Consolidates upstream outputs into MASTER_GRID + per-model deep-dives
    # + statistical tables + figures. Default ON; opt-out via --no-comprehensive-eval.
    if phases.should_run("R12", resume_from) and not bool(getattr(config, "no_comprehensive_eval", False)):
        from .comprehensive_eval import run_comprehensive_eval
        ckpt.start_timer("R12")
        try:
            comp = run_comprehensive_eval(phase1, all_results, config,
                                eval_logger=globals().get("_EVAL_LOGGER"))
            all_results["comprehensive_eval"] = comp
            ckpt.save("R12", comp, "Comprehensive Eval")
            log.info(f"  {phases.display('R12')} elapsed: {fmt_time(ckpt.stop_timer('R12'))}")
        except Exception as e:
            import traceback as _tb
            log.error(f"  {phases.display('R12')} 실패 — 계속 진행: {e}\n"
                      + _tb.format_exc())
            all_results["comprehensive_eval"] = {"error": str(e)}

    # ===== P1 (real_eval): Real Forecast + Deployment Gate — production-track start =====
    # Runs AFTER all R (R9 optimize → R10 eval → R11 SHAP → R12 comprehensive).
    # _select_champion_and_real_pred (in real_eval) picks best_name = R9 best-WIS champion
    # + best_pred = R9 optimized real prediction; the gate (→ABM/ARIA) operates on the
    # confirmed champion. resume gate is P1 so a resume from R9..R12 re-runs it with the
    # fresh champion. R12 no longer consumes real_eval — real-slab lives in R9 service-zone.
    if phases.should_run("P1", resume_from) and bool(getattr(config.split, "real_eval_enabled", True)):
        from .real_eval import run_real_eval
        ckpt.start_timer("P1")
        try:
            # real_eval on BASIC eval features (R9 used full pool); real_X+X_all BASIC slice.
            _phase1_eval = phase1
            if _basic_idx is not None:
                _phase1_eval = dict(phase1)
                _phase1_eval["X_all"] = X_eval
                _phase1_eval["feature_cols"] = feature_cols_eval
                _rX = phase1.get("real_X")
                if _rX is not None:
                    _phase1_eval["real_X"] = _rX[:, _basic_idx]
            real_eval = run_real_eval(_phase1_eval, all_results, config)
            all_results["real_eval"] = real_eval
            ckpt.save("P1", real_eval, "Real Forecast Eval")
            _log_phase_metrics("real_eval", real_eval, slab="real")
            log.info(f"  {phases.display('P1')} (post-champion) elapsed: {fmt_time(ckpt.stop_timer('P1'))}")
        except Exception as e:
            import traceback as _tb
            log.error(f"  [CHAMPION_GATE_FAILED] {phases.display('P1')}: {type(e).__name__}: {e}\n"
                      + _tb.format_exc())
            # G-237: critical phase — mark so the end-of-run gate surfaces it loudly.
            all_results["real_eval"] = {"error": f"{type(e).__name__}: {e}", "critical": True}

    # ===== Final Report =====
    from .utils.checkpoint import _make_serializable
    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": "1.0",
        "config": config.to_dict(),
        "total_elapsed": fmt_time(time.time() - t_global),
        "peak_rss_gb": round(_peak_rss[0], 2),          # G-251 자원 요약
        "phase_resource_log": _phase_res_log,           # G-251 phase별 소요시간·자원
    }

    # G-251: 전체 진행 요약 — phase별 소요시간 · RSS · CPU 한눈에 (로그)
    if _phase_res_log:
        log.info("  ┌─ 전체 phase 요약 (소요시간 · 자원) ──────────────────")
        for _e in _phase_res_log:
            log.info(f"  │ {_e['phase']:>3}: {fmt_time(_e['elapsed_s']):>9} "
                     f"│ RSS {_e['rss_gb']:.2f}GB (peak {_e['peak_gb']:.2f}) "
                     f"│ sysCPU {_e['sys_cpu']:.0f}%")
        log.info(f"  └─ 총 {fmt_time(time.time() - t_global)} · peak RSS "
                 f"{_peak_rss[0]:.2f}GB · {len(_phase_res_log)} phase ───────")

    for key in ["diagnostics", "dm_tests",
                "prediction_intervals", "feature_importance", "scoring"]:
        if key in all_results:
            report[key] = all_results[key]

    if "wfcv" in all_results:
        report["walk_forward_cv"] = all_results["wfcv"].get("wf_results", {})

    # ── G-237: critical-phase fail-loud gate (logic in _collect_critical_failures) ──
    # R9/R10/P1 silent failure must NOT read as a successful run (2026-05-30).
    # ADDITIVE — per-submetric NaN-fallback + optional phases R11/R12 untouched.
    _crit_fail = _collect_critical_failures(all_results)
    if _crit_fail:
        report["CRITICAL_FAILURES"] = [{"phase": _p, "error": _e} for _p, _e in _crit_fail]

    # ───── G-238: unified model×129×phase comparison history ─────
    try:
        _mh = _write_metric_history(all_results, config.get_save_dir())
        if _mh is not None:
            report["metric_history_csv"] = str(_mh)
    except Exception as _mhe:
        log.warning(f"  [metric-history] writer failed (non-fatal): {type(_mhe).__name__}: {_mhe}")

    report_path = config.get_save_dir() / config.output.report_name
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(_make_serializable(report), f, indent=2,
                  ensure_ascii=False, default=str)

    # ───── Close EvalLogger + roll up INDEX.csv ─────
    if _EVAL_LOGGER is not None:
        try:
            _EVAL_LOGGER.checkpoint()
            _EVAL_LOGGER.close()
            from simulation.utils.eval_logger import build_run_index
            idx = build_run_index()
            log.info(f"  [audit] eval-runs INDEX: {idx}")
        except Exception as _ce:
            log.warning(f"  [audit] EvalLogger close failed: {_ce}")

    log.info("")
    if _crit_fail:
        log.error("  " + "█" * 58)
        for _p, _e in _crit_fail:
            log.error(f"  █ CRITICAL FAILURE — {_p}: {_e}")
        log.error("  █ Champion gate / SSOT outputs are INVALID — do NOT trust this run.")
        log.error("  " + "█" * 58)
    log.info("  " + "=" * 58)
    _done = ("Pipeline completed WITH CRITICAL FAILURES (see banner above)"
             if _crit_fail else "Pipeline Complete!")
    log.info(f"  {_done} (Total: {fmt_time(time.time() - t_global)})")
    log.info(f"  Report: {report_path}")
    log.info("  " + "=" * 58)

    return report


if __name__ == "__main__":
    run_pipeline()
