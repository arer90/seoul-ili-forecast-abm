"""Guard: the result validator must actually fail on corrupted results.

A validator that passes on a clean tree proves nothing on its own — it could be
checking conditions that no realistic corruption would ever violate. Each test
below plants one specific defect in a throwaway copy of the result files and
asserts the validator rejects it.

The defects are the ones that would actually mislead a reader: a truncated
metric table, a number in the README that no longer matches its source, a
result file corrupted in transit, and a new model appearing with no leak-free
score.

Run standalone — macOS needs per-file pytest runs (LightGBM/OpenMP):
    .venv/bin/python -m pytest simulation/tests/test_validate_results.py -q
"""

import csv
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "validate_results.py"

METRICS = Path("simulation/results/per_model_eval/per_model_metrics.csv")
ABM = Path("simulation/results/abm_forward_validation/result.json")


def _load():
    """Import the script fresh so its module-level counters start clean."""
    spec = importlib.util.spec_from_file_location("validate_results", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(root: Path) -> int:
    return _load().main(["--root", str(root)])


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    """A minimal copy of the files the validator reads."""
    for rel in (METRICS, ABM):
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, dst)
    shutil.copytree(ROOT / "simulation/results/checkpoints",
                    tmp_path / "simulation/results/checkpoints")
    shutil.copytree(ROOT / "web/public/aggregates",
                    tmp_path / "web/public/aggregates")
    shutil.copy(ROOT / "README.md", tmp_path / "README.md")
    return tmp_path


def test_clean_tree_passes(sandbox):
    assert _run(sandbox) == 0, "the validator rejects an unmodified checkout"


def test_catches_readme_number_drift(sandbox):
    """The README quoting a number the result file no longer supports."""
    p = sandbox / ABM
    d = json.loads(p.read_text(encoding="utf-8"))
    d["forward_r2"] = 0.999
    p.write_text(json.dumps(d), encoding="utf-8")
    assert _run(sandbox) == 1, "README/result mismatch went undetected"


def test_catches_corrupted_json(sandbox):
    (sandbox / ABM).write_text("{ truncated", encoding="utf-8")
    assert _run(sandbox) == 1, "unparseable result file went undetected"


def test_catches_missing_result_file(sandbox):
    (sandbox / ABM).unlink()
    assert _run(sandbox) == 1, "missing result file went undetected"


def test_catches_new_model_without_a_leak_free_score(sandbox):
    """A model may skip test WIS, but never the out-of-fold WIS it is ranked on."""
    p = sandbox / METRICS
    with open(p, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows, cols = list(reader), reader.fieldnames
    ghost = dict.fromkeys(cols, "")
    ghost["model"] = "Ghost-Model"
    ghost["wis"] = "nan"
    ghost["oof_wis"] = "nan"
    rows.append(ghost)
    with open(p, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    assert _run(sandbox) == 1, "unscored new model went undetected"


def test_catches_truncated_metric_table(sandbox):
    """Only the header survives — the table is technically valid CSV."""
    with open(sandbox / METRICS, encoding="utf-8") as fh:
        header = fh.readline()
    (sandbox / METRICS).write_text(header, encoding="utf-8")
    assert _run(sandbox) == 1, "empty metric table went undetected"


def test_catches_champion_demotion(sandbox):
    """Another model overtaking FusedEpi without the README being updated."""
    p = sandbox / METRICS
    with open(p, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows, cols = list(reader), reader.fieldnames
    for r in rows:
        if r["model"] == "ARIMA":
            r["relative_wis_vs_baseline"] = "0.0001"
    with open(p, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    assert _run(sandbox) == 1, "champion demotion went undetected"
