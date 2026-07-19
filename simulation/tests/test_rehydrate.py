"""Guard: resume must actually recover earlier phases, and say what it cannot.

Two defects this pins, both hit while writing the feature:

1. The runner's resume block converted `resume_from` with `phases.resume_index()`,
   but `resume_from` is already an ordered index by that point, and numbers are
   no longer accepted as phase references — so it raised. The surrounding
   try/except turned that into a one-line warning and rehydration silently never
   ran. The pipeline would have looked fixed while behaving exactly as before.

2. `CheckpointManager(save_dir)` appends "checkpoints" itself. Passing it the
   checkpoints directory writes to checkpoints/checkpoints/, so the sync script
   reported success while the real checkpoints were untouched.

The honest half matters as much as the recovery: R2 stores only a model count
and R4 a subset, so those phases cannot come back. A resumed run really does
select the champion from a narrower pool, and `missing` is how the operator
learns that instead of reading a confident report built on half the candidates.

Run standalone — macOS needs per-file pytest runs (LightGBM/OpenMP):
    .venv/bin/python -m pytest simulation/tests/test_rehydrate.py -q
"""

import json
from pathlib import Path

import pytest

from simulation.pipeline.rehydrate import (
    LABEL_TO_KEY,
    LOSSY_LABELS,
    rehydrate_all_results,
)

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "simulation" / "results"


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A minimal results tree with checkpoints and per-model artifacts."""
    ck = tmp_path / "checkpoints"
    ck.mkdir(parents=True)
    for label, payload in [
        ("R5", {"diag": 1}),
        ("R6", {"dm_tests": {"A_vs_B": {"p_value": 0.5}}}),
        ("R8", {"scoring": [1, 2]}),
        ("R10", {"skipped": True, "reason": "filter excluded all"}),
    ]:
        (ck / f"checkpoint_{label}.json").write_text(
            json.dumps({"phase": label, "data": payload}), encoding="utf-8"
        )
    pmo = tmp_path / "per_model_optimal"
    pmo.mkdir()
    for name in ("Alpha", "Beta", "Gamma"):
        (pmo / f"{name}.json").write_text(json.dumps({"model": name}), encoding="utf-8")
    ev = tmp_path / "per_model_eval"
    ev.mkdir()
    (ev / "per_model_metrics.csv").write_text("model,wis\nAlpha,1.0\nBeta,2.0\n", encoding="utf-8")
    (ev / "ranking.json").write_text(
        json.dumps({"top10_by_oof_wis": [{"model": "Alpha", "oof_wis": 1.0}]}), encoding="utf-8"
    )
    return tmp_path


def test_recovers_the_phases_it_can(tree):
    s = rehydrate_all_results(tree)
    assert "diagnostics" in s.results
    assert "dm_tests" in s.results
    assert "scoring" in s.results


def test_r9_comes_from_per_model_files_not_the_checkpoint(tree):
    """A --models re-run overwrites checkpoint_R9 but not per_model_optimal/."""
    s = rehydrate_all_results(tree)
    configs = s.results["per_model_optimize"]["per_model_configs"]
    assert set(configs) == {"Alpha", "Beta", "Gamma"}
    assert "per_model_optimal" in s.recovered["per_model_optimize"]


def test_a_skipped_checkpoint_is_never_carried_forward_as_results(tree):
    """R10's checkpoint here says `skipped: filter excluded all`.

    Its payload must not be handed to a consuming phase — that is exactly how an
    empty evaluation gets reported as a success. Recovering the same phase from
    the metrics CSV instead is fine and is what R12 needs, so the assertion is
    on the payload, not on the key's absence.
    """
    s = rehydrate_all_results(tree)
    node = s.results.get("per_model_eval", {})
    assert not node.get("skipped"), "a skipped checkpoint was carried into all_results"
    assert "reason" not in node
    if node:
        assert node["metrics_csv"].endswith("per_model_metrics.csv"), (
            "if per_model_eval is present it must be backed by the CSV, "
            "not by the skipped checkpoint"
        )
        assert "per_model_metrics" in s.recovered["per_model_eval"], (
            "provenance must name the CSV so a reader knows the checkpoint "
            "was not the source"
        )


def test_metrics_csv_and_ranking_are_wired_for_r12(tree):
    s = rehydrate_all_results(tree)
    node = s.results["per_model_eval"]
    assert node["metrics_csv"].endswith("per_model_metrics.csv")
    assert node["ranking_top10"] == ["Alpha"]


def test_unrecoverable_phases_are_named(tree):
    s = rehydrate_all_results(tree)
    for key in ("baseline", "wfcv"):
        assert key in s.missing, f"{key} is not recoverable and must be reported"
    assert "baseline_n_models" in s.missing["baseline"] or "count" in s.missing["baseline"]


def test_absent_tree_yields_missing_not_a_crash(tmp_path):
    s = rehydrate_all_results(tmp_path / "nope")
    assert s.results == {}
    assert s.missing


# ── the two defects that nearly shipped ──────────────────────────────────────
def test_runner_uses_the_index_directly_not_resume_index():
    """`resume_from` is already an ordered index; converting it raises."""
    src = (ROOT / "simulation" / "pipeline" / "runner.py").read_text(encoding="utf-8")
    block = src[src.index("[resume] rehydrated") - 2000: src.index("[resume] rehydrated")]
    assert "phases.resume_index(resume_from)" not in block, (
        "resume_index() rejects integers, and the surrounding try/except would "
        "swallow the error — rehydration would silently never run"
    )


def test_checkpoint_manager_is_given_the_results_root():
    """It appends 'checkpoints' itself; passing that dir writes one level too deep."""
    src = (ROOT / "scripts" / "regenerate_r12.py").read_text(encoding="utf-8")
    assert 'CheckpointManager(results_dir / "checkpoints")' not in src
    assert "CheckpointManager(results_dir)" in src


def test_label_map_covers_every_key_the_runner_assigns():
    """A key the runner writes but the map omits comes back empty on resume."""
    src = (ROOT / "simulation" / "pipeline" / "runner.py").read_text(encoding="utf-8")
    import re
    assigned = set(re.findall(r'all_results\["([a-z_0-9]+)"\]\s*=', src))
    known = set(LABEL_TO_KEY.values()) | {"phase1", "baseline", "wfcv", "external", "xai"}
    missing = assigned - known
    assert not missing, f"runner writes all_results keys the rehydrator does not know: {missing}"


def test_shipped_tree_rehydrates():
    """The real results directory must come back, not just the fixture."""
    if not (RESULTS / "checkpoints").is_dir():
        pytest.skip("no shipped results tree")
    s = rehydrate_all_results(RESULTS)
    assert len(s.recovered) >= 5, f"only recovered {sorted(s.recovered)}"
    assert set(s.missing) >= {"baseline", "wfcv"}, "the lossy phases must be declared"
