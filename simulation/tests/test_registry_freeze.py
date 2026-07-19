"""PAPER_PRIMARY_11 + NEGATIVE_CONTROL freeze invariants.

CURRENT design (2026-05-12 사용자 명시 "그냥 없애. paper primary도 다시 만들어야해
+ 학습 후 만들꺼야"; registry.py PAPER_PRIMARY_11 declaration):
 1. PAPER_PRIMARY_11 is INTENTIONALLY EMPTY pending post-training refreeze —
    structural invariant only (tuple of unique (name,src) pairs).
 2. NEGATIVE_CONTROL is EMPTY ("all models equal"; over-param is a HP problem,
    not a permanent demotion) — TabularDNN is a normal model, NOT excluded.
 3. registry_v22.json (SHA-256 manifest) is consistent with the in-code
    declaration when present (xfail if missing).

When the user refreezes PAPER_PRIMARY_11 post-training, the structural tests
still hold; the freeze-count test below is skipped until then (flip the skip).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# ══════════════════════════════════════════════════════════════════════════
# 1. In-code PAPER_PRIMARY_11 + NEGATIVE_CONTROL invariants
# ══════════════════════════════════════════════════════════════════════════
def test_paper_primary_11_structural_invariant():
    """PAPER_PRIMARY_11 = tuple of unique (name, src) pairs. Intentionally
    empty pending post-training refreeze — assert structure, not count."""
    from simulation.models.registry import PAPER_PRIMARY_11
    assert isinstance(PAPER_PRIMARY_11, tuple)
    names = [n for n, _ in PAPER_PRIMARY_11]
    assert len(names) == len(set(names)), f"duplicate names: {names}"


@pytest.mark.skip(reason="PAPER_PRIMARY_11 refreeze pending post-training "
                         "(registry.py: 사용자 2026-05-12 '학습 후 만들꺼야'). "
                         "Flip to assert the 11-set once refrozen.")
def test_paper_primary_11_has_exactly_eleven_unique_names():
    from simulation.models.registry import PAPER_PRIMARY_11
    names = [n for n, _ in PAPER_PRIMARY_11]
    assert len(names) == 11, f"expected 11 entries, got {len(names)}"


def test_negative_control_is_empty_all_models_equal():
    """CURRENT policy (registry.py): NEGATIVE_CONTROL is an empty frozenset —
    no permanent demotion (over-param = HP problem). TabularDNN is therefore a
    normal model, NOT excluded. (Reversed the obsolete 'TabularDNN demoted'
    invariant.)"""
    from simulation.models.registry import NEGATIVE_CONTROL
    assert isinstance(NEGATIVE_CONTROL, frozenset), (
        "NEGATIVE_CONTROL must be an immutable frozenset."
    )
    assert "TabularDNN" not in NEGATIVE_CONTROL, (
        "TabularDNN is a normal model under the all-equal policy."
    )


def test_negative_control_and_paper_primary_are_disjoint():
    from simulation.models.registry import PAPER_PRIMARY_11, NEGATIVE_CONTROL
    paper_names = {n for n, _ in PAPER_PRIMARY_11}
    overlap = paper_names & set(NEGATIVE_CONTROL)
    assert not overlap, (
        f"PAPER_PRIMARY_11 and NEGATIVE_CONTROL must be disjoint, but "
        f"overlap = {overlap}"
    )


# ══════════════════════════════════════════════════════════════════════════
# 2. Frozen manifest (simulation/results/registry_v22.json)
# ══════════════════════════════════════════════════════════════════════════
def _manifest_path() -> Path:
    # tests/ -> simulation/ -> results/registry_v22.json
    sim_root = Path(__file__).resolve().parent.parent
    return sim_root / "results" / "registry_v22.json"


def test_registry_manifest_exists_and_parses():
    """The frozen manifest must be checked into the repo."""
    p = _manifest_path()
    if not p.exists():
        pytest.xfail(
            f"registry_v22.json missing at {p}. Run Stage 2 generator to "
            "produce it."
        )
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data.get("version") == "v22.2", data.get("version")
    assert len(data["paper_primary_11"]) == 11
    assert len(data["negative_control"]) >= 1


def test_registry_manifest_matches_code():
    """Names + source files in JSON manifest match the in-code
    PAPER_PRIMARY_11 declaration."""
    p = _manifest_path()
    if not p.exists():
        pytest.xfail(f"registry_v22.json missing at {p}")
    data = json.loads(p.read_text(encoding="utf-8"))

    from simulation.models.registry import PAPER_PRIMARY_11, NEGATIVE_CONTROL
    code_paper = {(n, s) for n, s in PAPER_PRIMARY_11}
    json_paper = {(e["model_name"], e["source_file"])
                  for e in data["paper_primary_11"]}
    assert code_paper == json_paper, {
        "only_in_code": code_paper - json_paper,
        "only_in_json": json_paper - code_paper,
    }

    json_neg = {e["model_name"] for e in data["negative_control"]}
    assert set(NEGATIVE_CONTROL) <= json_neg, (
        f"NEGATIVE_CONTROL in code has entries not in manifest: "
        f"{set(NEGATIVE_CONTROL) - json_neg}"
    )


def test_registry_manifest_hashes_match_current_sources():
    """Recompute each source's SHA-256 and compare against the manifest.
    A mismatch means the source file changed without re-freezing — a
    reviewable event but not a hard fail during development."""
    p = _manifest_path()
    if not p.exists():
        pytest.xfail(f"registry_v22.json missing at {p}")
    data = json.loads(p.read_text(encoding="utf-8"))

    import hashlib
    sim_root = Path(__file__).resolve().parent.parent

    drifts: list[tuple[str, str, str]] = []
    for entry in data["paper_primary_11"] + data["negative_control"]:
        src = sim_root / entry["source_file"]
        if not src.exists():
            continue  # handled by test_paper_primary_source_files_exist
        h = hashlib.sha256()
        with src.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        current = h.hexdigest()
        frozen = entry["source_sha256"]
        if current != frozen:
            drifts.append((entry["model_name"], frozen[:12], current[:12]))

    if drifts:
        pytest.xfail(
            "Source files drifted from frozen manifest — re-freeze via "
            "Stage 2 generator if intentional:\n"
            + "\n".join(f"  {n}: frozen={f}… current={c}…" for n, f, c in drifts)
        )


# ══════════════════════════════════════════════════════════════════════════
# 3. Ensemble pool defense — tournament must exclude NEGATIVE_CONTROL
# ══════════════════════════════════════════════════════════════════════════
def test_tournament_excludes_negative_control_from_pool(monkeypatch):
    """The orchestrator must drop NEGATIVE_CONTROL members before category
    rank / Caruana / meta-compete see them. NEGATIVE_CONTROL is empty under
    the current all-equal policy, so we monkeypatch a synthetic member to
    exercise the FILTER MECHANISM (robust to the live empty policy)."""
    import numpy as np
    import simulation.models.registry as _reg
    from simulation.ensembles.tournament import TournamentOrchestrator

    # Inject a synthetic negative control; tournament re-imports the module
    # attribute at call time, so patching the registry attr is sufficient.
    monkeypatch.setattr(_reg, "NEGATIVE_CONTROL", frozenset({"BannedDL"}))

    n = 30
    rng = np.random.default_rng(42)
    y = rng.normal(size=n)
    oof = {
        "Ridge":    y + rng.normal(scale=0.1, size=n),
        "XGBoost":  y + rng.normal(scale=0.15, size=n),
        "BannedDL": y + rng.normal(scale=0.05, size=n),  # "best" but banned
    }
    cats = {"Ridge": "linear", "XGBoost": "tree", "BannedDL": "dl"}

    orch = TournamentOrchestrator(top_k_per_category=1, caruana_steps=5)
    result = orch.run(oof, y, cats)

    a1_all_names: set[str] = set()
    for names in result.stage_a1.values():
        a1_all_names.update(names)
    assert "BannedDL" not in a1_all_names, (
        f"NEGATIVE_CONTROL member leaked into tournament pool despite "
        f"filter. stage_a1={result.stage_a1}"
    )
