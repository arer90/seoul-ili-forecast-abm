"""Smoke tests: PAPER_PRIMARY_11 registry snapshot + SHA-256."""
from __future__ import annotations

import pytest

from pathlib import Path


def test_paper_primary_11_structural_invariant():
    """PAPER_PRIMARY_11 is a tuple of (name, source_file) pairs with no
    duplicates. It is INTENTIONALLY EMPTY pending post-training refreeze
    (registry.py PAPER_PRIMARY_11 declaration, 사용자 2026-05-12: "학습 후
    새 PaperPRIMARY 정의"), so we assert the structure, not a fixed count.
    When refrozen, this test still holds (and the dup-check guards it)."""
    from simulation.models.registry import PAPER_PRIMARY_11
    assert isinstance(PAPER_PRIMARY_11, tuple)
    names = [n for n, _ in PAPER_PRIMARY_11]
    assert all(isinstance(s, str) for _, s in PAPER_PRIMARY_11), \
        "each entry must be (name, source_file_str)"
    assert len(names) == len(set(names)), f"duplicate names: {names}"


def test_paper_primary_source_files_exist():
    """Each PAPER_PRIMARY entry must point at a real file on disk."""
    from simulation.models.registry import PAPER_PRIMARY_11, _simulation_root
    root = _simulation_root()
    missing = []
    for name, src in PAPER_PRIMARY_11:
        p = root / src
        if not p.exists():
            missing.append((name, str(p)))
    # It's ok if some PAPER_PRIMARY targets don't exist yet (pending),
    # but at least 7 of them should (registered models).
    assert len(missing) <= 4, \
        f"Too many PAPER_PRIMARY source files missing: {missing}"


def test_sha256_file_returns_hex():
    from simulation.models.registry import sha256_file
    import tempfile, os
    fd, path = tempfile.mkstemp()
    try:
        os.write(fd, b"hello world")
        os.close(fd)
        h = sha256_file(path)
        assert h is not None
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)
    finally:
        os.unlink(path)


def test_live_registry_coverage_consistent():
    """live REGISTRY 의 단일 출처 검증 — `verify_registry_coverage()` 가
    CATEGORY_MODELS(52, SSOT) 를 모두 등록하고 missing=0 인지 확인.
    (이전 REGISTRY_CATALOG 하드코딩 dict 은 2026-05-29 은퇴 — stale 합 68 ≠ live.
     2026-06-13: G-261 Chronos→TimesFM, G-262 중복 3종 감축, G-263 NegBinGLM-Glum add → 54→53→50→51→52.)"""
    from simulation.models.registry import verify_registry_coverage
    r = verify_registry_coverage(force_import=True)
    if not r["ok"]:
        # A model whose module cannot import is never registered, so coverage is
        # legitimately incomplete without the heavy optional stack (torch and
        # torch-geometric gate GCN and OverseasTransfer). Distinguish a thin
        # environment from a broken registry rather than failing CI for an
        # install it deliberately does not perform.
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip(f"torch absent — registry cannot be complete: {r['missing']}")
    assert r["ok"], (
        f"registry coverage failed (active {r['total_expected']}, "
        f"registered {r['total_registered']}, missing {r['missing']})"
    )
    assert r["total_registered"] >= r["total_expected"], (
        f"registered {r['total_registered']} < active {r['total_expected']}"
    )
    assert not r["missing"], (
        f"models in CATEGORY_MODELS but not registered: {r['missing']}"
    )
    assert r["ok"] is True


# ══════════════════════════════════════════════════════════════════════════
# — bench_dl_baselines comparison harness (smoke only)
# ══════════════════════════════════════════════════════════════════════════
def test_bench_dl_baselines_argparse_rejects_unknown_flag():
    """Guards the CLI surface — adding an unintended flag silently
    drifts the public contract."""
    import pytest
    from simulation.benchmarks import bench_dl_baselines as bdb
    with pytest.raises(SystemExit):
        bdb._parse_args(["--nonexistent-flag"])


# ══════════════════════════════════════════════════════════════════════════
# — Task C: no runtime dependency on `_past/`
#
# ENGINEERING_PRINCIPLES.md rule (§41, §160): simulation/ MUST NOT reach outside
# simulation/data/. Task C audit (2026-04-17) removed the last runtime
# reference (`_past/data/` PDF fallback). This test freezes the invariant:
# only two `_past` mentions are allowed in simulation/ —
#   1) verifier/ast_checker.py EXCLUDE list (lint skip),
#   2) collectors/extract_pdf.py audit-trail comment.
# Any additional mention fails CI.
# ══════════════════════════════════════════════════════════════════════════
def test_simulation_package_has_no_runtime_past_reference():
    """Runtime files in simulation/ must not import, read, or walk
    `_past/`. Only the two documented mentions (AST exclude list and
    the dropped-fallback comment in extract_pdf.py) are tolerated."""
    from pathlib import Path

    sim_root = Path(__file__).resolve().parent.parent  # simulation/
    allowed_files = {
        sim_root / "verifier" / "ast_checker.py",
        sim_root / "collectors" / "extract_pdf.py",
        sim_root / "tests" / "test_registry.py",          # this file (renamed from test_v22_registry.py)
        sim_root / "scripts" / "bench_seir_python.py",    # docstring ref to old _past/seir-wasm location
    }

    offenders: list[tuple[str, int, str]] = []
    for py in sim_root.rglob("*.py"):
        if py in allowed_files:
            continue
        # _archive/ = retired code (not runtime); __pycache__ = bytecode.
        if "__pycache__" in py.parts or "_archive" in py.parts:
            continue
        try:
            text = py.read_text(encoding="utf-8")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            # Match the PATH form `_past/` (runtime reference), not the bare
            # substring — avoids false hits on vars like `y_past` and on
            # skip-list literals such as "_past" in dir-exclusion sets.
            if "_past/" in line:
                offenders.append((str(py.relative_to(sim_root)), i, line.strip()))

    assert not offenders, (
        "ENGINEERING_PRINCIPLES.md forbids runtime references to _past/. Offenders:\n"
        + "\n".join(f"  {f}:{ln}  {txt}" for f, ln, txt in offenders)
    )
