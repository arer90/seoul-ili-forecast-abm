"""TDD — `--resume-from` resolves R/P labels + semantic names (Phase 0b: numbers removed).

`resolve_resume_from` now returns the ordered phase INDEX (simulation.pipeline.phases), not the
old magic number, and accepts ONLY R/P labels ("R9") or semantic names ("per_model_optimize").
Legacy phase NUMBERS are rejected (the whole point of the R/P rename).

run PER-FILE on macOS: KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 pytest <thisfile>
"""
import argparse

import pytest

from simulation.pipeline import phases
from simulation.pipeline.runner import resolve_resume_from


def test_label_resolves_to_order_index():
    assert resolve_resume_from("R9") == phases.order("R9")
    assert resolve_resume_from("R1") == phases.order("R1")
    assert resolve_resume_from("P1") == phases.order("P1")


def test_semantic_name_resolves_same_as_label():
    assert resolve_resume_from("per_model_optimize") == phases.order("R9")
    assert resolve_resume_from("real_eval") == phases.order("P1")   # alias → real_forecaster
    assert resolve_resume_from("wfcv") == phases.order("R4")
    assert resolve_resume_from("data") == phases.order("R1")


def test_case_and_whitespace_insensitive():
    assert resolve_resume_from("r9") == phases.order("R9")
    assert resolve_resume_from("  PER_MODEL_OPTIMIZE  ") == phases.order("R9")


def test_none_and_empty():
    assert resolve_resume_from(None) is None
    assert resolve_resume_from("") is None
    assert resolve_resume_from("   ") is None


def test_shap_xai_share_one_phase():
    assert (resolve_resume_from("shap") == resolve_resume_from("shap_analysis")
            == resolve_resume_from("xai") == phases.order("R11"))


def test_numbers_are_rejected():
    # phase numbers are GONE — numeric resume must raise
    for bad in (13, "13", "6", "phase13"):
        with pytest.raises(argparse.ArgumentTypeError):
            resolve_resume_from(bad)


def test_unknown_name_raises():
    with pytest.raises(argparse.ArgumentTypeError):
        resolve_resume_from("bogus_phase")


def test_index_to_label_roundtrip_stable():
    """training_commands re-passes the resolved index as a LABEL (index→label→re-resolve).

    Regression: it used to re-pass str(index) (e.g. '8'), which the inner parse rejected as a
    removed phase number → `--resume-from R9` crashed with "알 수 없는 phase '8'".
    """
    for lbl in phases.all_labels():
        idx = resolve_resume_from(lbl)
        relabel = phases.PHASES[idx][0]                  # index → canonical label
        assert resolve_resume_from(relabel) == idx, f"round-trip broke: {lbl}→{idx}→{relabel}"
