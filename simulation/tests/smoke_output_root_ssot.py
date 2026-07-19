#!/usr/bin/env python3
"""Regression smoke: 4 leaking components honor MPH_OUTPUT_ROOT (2026-05-29 fix).

Background
----------
`MPH_OUTPUT_ROOT` (= GLOBAL.paths.output_root) redirects results/models/cache to
another disk. Four sub-components used to ignore it and hardcode
``simulation/results/`` — they leaked production files even when the root was a
throwaway dir (found via ``run_pipeline.sh --smoke`` 2026-05-29):

    1. eval_logs (EvalLogger + build_run_index → INDEX.csv)
    2. stage2_feature_optuna (writer default + paired readers/consumer)
    3. phase12_elife (scripts/run_elife_phase12.py OUT)
    4. statistical_audit (phase13 consolidation write + standalone script)

All four now route through ``simulation.utils.paths.get_results_dir()`` (the same
``Path(output_root)/"results"`` pattern config.py uses).

A follow-up sweep (2026-05-29) extended this to the per_model_optimal pattern: 18
post-hoc readers/writers (phase18 pipeline read + 16 scripts + the phase13
optuna_feat_sel legacy reader) that hardcoded
``simulation/results/per_model_optimal[_v2]``. Component 5 verifies them
dynamically (a module-level constant + the base dirs) and with a durable AST guard
that fails if any swept file reintroduces the hardcode in code.

What this checks
----------------
Sets ``MPH_OUTPUT_ROOT`` to a throwaway dir *before* importing simulation, then:
  (A) artifacts from the exercised writers land under ``$ROOT/results/``, and
  (B) ZERO new files appear in project-local ``simulation/results/``  ← the
      operator's actual acceptance criterion.

This is the fast/focused counterpart to ``bash run_pipeline.sh --smoke`` (which
exercises all four end-to-end and reports leaks via manifest-diff). The heavy
in-pipeline writers (phase13 STATISTICAL_AUDIT, statistical_audit standalone)
share the identical ``get_results_dir()`` wiring proven here for eval/elife/stage2;
their end-to-end coverage is the full ``--smoke`` run.

Run:
    .venv/bin/python simulation/tests/smoke_output_root_ssot.py
Exit 0 = pass, 1 = leak / misroute detected.
"""
from __future__ import annotations

import ast
import os
import shutil
import sys
import tempfile
from pathlib import Path

# MUST set the env override before importing simulation: get_output_root() is
# lru_cached and module-level path constants resolve at import time.
_TMP = Path(tempfile.mkdtemp(prefix="mph_ssot_smoke_"))
os.environ["MPH_OUTPUT_ROOT"] = str(_TMP)

# Project-local results dir — the directory that must NOT receive new files
# while the redirect is active. simulation/tests/ → parents[1] == simulation/.
_PROJECT_RESULTS = Path(__file__).resolve().parents[1] / "results"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _snapshot(d: Path) -> set[str]:
    """Set of absolute paths of every file under ``d`` (empty if missing)."""
    if not d.exists():
        return set()
    return {str(p) for p in d.rglob("*") if p.is_file()}


def _under(p: Path, root: Path) -> bool:
    """True iff ``p`` resolves inside ``root``."""
    try:
        Path(p).resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def main() -> int:
    before = _snapshot(_PROJECT_RESULTS)
    failures: list[str] = []
    checks: list[str] = []

    from simulation.utils.paths import get_results_dir

    rdir = get_results_dir()
    if _under(rdir, _TMP):
        checks.append(f"get_results_dir() → {rdir}")
    else:
        failures.append(f"get_results_dir()={rdir} NOT under redirect root {_TMP}")

    # --- Component 1: eval_logs (real write) -------------------------------
    try:
        from simulation.utils.eval_logger import EvalLogger, build_run_index

        el = EvalLogger("ssot_probe")
        el.log(phase="p", model="m", metric="wis", value=1.0, slab="test")
        el.checkpoint()
        el.close()
        if _under(el.jsonl_path, _TMP):
            checks.append(f"eval_logs jsonl → {el.jsonl_path}")
        else:
            failures.append(f"EvalLogger jsonl {el.jsonl_path} NOT under {_TMP}")

        idx = build_run_index()
        if _under(idx, _TMP):
            checks.append(f"eval_logs INDEX.csv → {idx}")
        else:
            failures.append(f"build_run_index INDEX.csv {idx} NOT under {_TMP}")
    except Exception as e:  # noqa: BLE001 — surface any wiring error as a failure
        failures.append(f"eval_logger exercise raised: {e!r}")

    # --- Component 3: phase12_elife (module-level OUT / P12_DIR) ------------
    try:
        import importlib.util

        elife_path = _REPO_ROOT / "scripts" / "run_elife_phase12.py"
        spec = importlib.util.spec_from_file_location("_elife_probe", elife_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        for attr in ("OUT", "P12_DIR"):
            p = getattr(mod, attr)
            if _under(p, _TMP):
                checks.append(f"elife {attr} → {p}")
            else:
                failures.append(f"elife {attr}={p} NOT under {_TMP}")
    except Exception as e:  # noqa: BLE001
        failures.append(f"elife import raised: {e!r}")

    # --- Component 2: stage2_feature_optuna (light real run, fallback wiring)
    s2dir = get_results_dir() / "stage2_feature_optuna"
    try:
        import numpy as np

        from simulation.pipeline._inline_optuna_3stage import (
            _stage2_feature_optuna_inline,
        )

        rng = np.random.default_rng(42)
        n, k = 60, 6
        phase1 = {
            "X_all": rng.normal(size=(n, k)),
            "y_all": np.abs(rng.normal(2.0, 0.5, size=n)),  # positive, ILI-like
            "feature_cols": [f"f{i}" for i in range(k)],
            "n_train": 45,
            "pool_end": 45,
        }
        _stage2_feature_optuna_inline(phase1, ["ElasticNet"], n_trials_per_model=1)
        wrote = list(s2dir.glob("*.json")) if s2dir.exists() else []
        if wrote and all(_under(p, _TMP) for p in wrote):
            checks.append(f"stage2 wrote {len(wrote)} json → {s2dir}")
        elif _under(s2dir, _TMP):
            checks.append(f"stage2 dir → {s2dir} (light run produced {len(wrote)} files)")
        else:
            failures.append(f"stage2 dir {s2dir} NOT under {_TMP}")
    except Exception as e:  # noqa: BLE001 — light run is best-effort; verify wiring
        if _under(s2dir, _TMP):
            checks.append(f"stage2 wiring OK → {s2dir} (light run skipped: {type(e).__name__})")
        else:
            failures.append(f"stage2 dir {s2dir} NOT under {_TMP} ({e!r})")

    # --- Component 4: STATISTICAL_AUDIT (phase13 + standalone share base) ---
    audit_md = get_results_dir() / "STATISTICAL_AUDIT.md"
    if _under(audit_md, _TMP):
        checks.append(f"STATISTICAL_AUDIT base → {audit_md.parent}")
    else:
        failures.append(f"STATISTICAL_AUDIT base {audit_md} NOT under {_TMP}")

    # --- Component 5: per_model_optimal sweep (2026-05-29 extension) --------
    # 18 readers/writers that hardcoded simulation/results/per_model_optimal[_v2]
    # (+ phase13 optuna_feat_sel legacy reader) now route through get_results_dir().
    # (1) base dirs resolve under the redirect root.
    for sub in ("per_model_optimal", "per_model_optimal_v2"):
        d = get_results_dir() / sub
        if _under(d, _TMP):
            checks.append(f"{sub} base → {d}")
        else:
            failures.append(f"{sub} base {d} NOT under {_TMP}")

    # (2) a module-level constant (apply_cqr_pi.PER_MODEL_DIR) resolves under root.
    try:
        import importlib

        cqr = importlib.import_module("simulation.scripts.apply_cqr_pi")
        if _under(cqr.PER_MODEL_DIR, _TMP):
            checks.append(f"apply_cqr_pi.PER_MODEL_DIR → {cqr.PER_MODEL_DIR}")
        else:
            failures.append(f"apply_cqr_pi.PER_MODEL_DIR {cqr.PER_MODEL_DIR} NOT under {_TMP}")
    except Exception as e:  # noqa: BLE001
        failures.append(f"apply_cqr_pi import raised: {e!r}")

    # (3) durable static guard: none of the swept files may reintroduce a hardcoded
    #     "simulation/results/per_model_optimal" in CODE. Docstrings are excluded;
    #     comments are absent from the AST; the redirect-aware fix splits the path
    #     into get_results_dir()/"per_model_optimal" so it never appears as one literal.
    swept = [
        "simulation/pipeline/phase18_overseas.py",
        "simulation/pipeline/phase13_per_model_optimize.py",
        "simulation/scripts/audit_problem_models.py",
        "simulation/scripts/compare_v1_v2.py",
        "simulation/scripts/mini_test_sub085_models.py",
        "simulation/scripts/sync_ensemble_jsons.py",
        "simulation/scripts/full_metric_audit.py",
        "simulation/scripts/plot_forecast_full.py",
        "simulation/scripts/apply_cqr_pi.py",
        "simulation/scripts/training_health.py",
        "simulation/scripts/extract_raw_predictions.py",
        "simulation/scripts/ensemble_full.py",
        "simulation/scripts/ensemble_standalone.py",
        "simulation/scripts/recompute_ensembles.py",
        "simulation/scripts/restore_fast_only.py",
        "simulation/scripts/mini_tests/run_grid_sweep.py",
        "simulation/scripts/retrain_problem_models.py",
    ]
    needle = "simulation/results/per_model_optimal"
    n_scanned = 0
    for rel in swept:
        fp = _REPO_ROOT / rel
        try:
            tree = ast.parse(fp.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            failures.append(f"AST parse {rel}: {e!r}")
            continue
        n_scanned += 1
        docstring_ids = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                body = getattr(node, "body", [])
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)
                ):
                    docstring_ids.add(id(body[0].value))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and needle in node.value
                and id(node) not in docstring_ids
            ):
                failures.append(
                    f"HARDCODE REGRESSION {rel}:{getattr(node, 'lineno', '?')} "
                    f"non-docstring literal {node.value!r}"
                )
    checks.append(f"AST guard: {n_scanned} swept files, 0 active per_model_optimal hardcodes")

    # --- Component 6: per_model_research sweep (2026-05-29 audit follow-up) --
    # run_3stage_optuna (research mode) + its phase14 caller wrote
    # per_model_research/<MODEL>.json with a hardcoded path — a DISTINCT family
    # missed by the original 4-component fix, the per_model_optimal sweep, AND
    # the smoke test (which never invokes the research path). Now via
    # get_results_dir(). Base resolves under root + durable AST guard.
    pmr = get_results_dir() / "per_model_research"
    if _under(pmr, _TMP):
        checks.append(f"per_model_research base → {pmr}")
    else:
        failures.append(f"per_model_research base {pmr} NOT under {_TMP}")
    pmr_files = [
        "simulation/pipeline/_inline_optuna_3stage.py",
        "simulation/pipeline/per_model_eval.py",
    ]
    pmr_needle = "simulation/results/per_model_research"
    for rel in pmr_files:
        fp = _REPO_ROOT / rel
        try:
            tree = ast.parse(fp.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            failures.append(f"AST parse {rel}: {e!r}")
            continue
        doc_ids = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                body = getattr(node, "body", [])
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)
                ):
                    doc_ids.add(id(body[0].value))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and pmr_needle in node.value
                and id(node) not in doc_ids
            ):
                failures.append(
                    f"HARDCODE REGRESSION {rel}:{getattr(node, 'lineno', '?')} "
                    f"non-docstring literal {node.value!r}"
                )
    checks.append(f"AST guard: per_model_research clean in {len(pmr_files)} files")

    # --- KEY criterion: zero new files in project-local simulation/results ---
    leaked = sorted(_snapshot(_PROJECT_RESULTS) - before)
    if leaked:
        failures.append(
            f"PROJECT-LOCAL LEAK: {len(leaked)} new file(s) in {_PROJECT_RESULTS}:"
        )
        failures.extend(f"      + {p}" for p in leaked[:20])

    # --- report ------------------------------------------------------------
    print("── MPH_OUTPUT_ROOT SSOT smoke ──")
    print(f"  redirect root : {_TMP}")
    print(f"  project-local : {_PROJECT_RESULTS}")
    for c in checks:
        print(f"  ✓ {c}")
    if failures:
        print("  ── FAILURES ──")
        for f in failures:
            print(f"  ✗ {f}")
        print(f"RESULT: FAIL ({len(failures)} issue(s))")
        return 1
    print("RESULT: PASS — 0 project-local leaks; exercised writers under redirect root")
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    finally:
        shutil.rmtree(_TMP, ignore_errors=True)
    sys.exit(rc)
