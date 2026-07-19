"""R/P phase registry (SSOT) + label-based checkpoint — Phase 0b rename safety net.

The pipeline is migrating from non-sequential magic phase numbers (1,4,5,...,12,13,14 with
real_eval=12 after 14) to an ordered R(research)/P(production) layout. These tests lock the
resolve/order contract (label ⇄ name ⇄ legacy-number) and the checkpoint round-trip BEFORE the
runner dispatch is rewired, so a mistake in the big rename fails loudly here.

macOS: run PER-FILE.
"""
import pytest

from simulation.pipeline import phases


def test_labels_unique_and_ordered():
    labels = phases.all_labels()
    assert labels == sorted(set(labels), key=labels.index)  # unique, preserves order
    assert labels[0] == "R1"                                # data is first
    # the standalone-CLI production phases (inference/overseas) come last
    assert phases.name_of(labels[-1]) == "overseas"


def test_resolve_label_and_name_agree():
    # label and semantic name resolve to the SAME phase (phase NUMBERS are removed)
    assert phases.order("R9") == phases.order("per_model_optimize")
    assert phases.label_of("per_model_optimize") == "R9"
    assert phases.name_of("R9") == "per_model_optimize"
    assert phases.track_of("R9") == "research"


def test_numbers_are_gone():
    # legacy phase numbers no longer resolve — only R/P labels + names
    import pytest as _pt
    for n in (13, 12, "13", 1):
        with _pt.raises(KeyError):
            phases.order(n)


def test_real_eval_is_now_P1_production():
    # the old real_eval becomes P1 real_forecaster on the PRODUCTION track (alias resolves)
    assert phases.label_of("real_eval") == "P1"
    assert phases.name_of("P1") == "real_forecaster"
    assert phases.track_of("P1") == "production"


def test_real_runs_after_comprehensive():
    # de27fdf (2026-06-20): P1(real_forecaster) 디커플 — R10→P1→R11→R12 중간끼임 제거하고
    # **R1..R12 전부 끝난 뒤** P1 실행(R12 comprehensive 가 R9 챔피언/family 만 소비, P1 미소비).
    # 따라서 P1 order > R10 AND > comprehensive_eval(R12).
    assert phases.order("P1") > phases.order("R10")
    assert phases.order("P1") > phases.order("comprehensive_eval")


def test_xai_shap_aliases():
    assert phases.label_of("xai") == phases.label_of("shap_analysis") == phases.label_of("shap")


def test_should_run_gate_semantics():
    # resume from 0 → run everything
    assert all(phases.should_run(l, 0) for l in phases.all_labels())
    # resume from R9 → R9 and later run; R1..R8 skip
    idx = phases.resume_index("R9")
    assert phases.should_run("R9", idx) and phases.should_run("R10", idx)
    assert not phases.should_run("R1", idx) and not phases.should_run("R8", idx)
    # label + semantic name resolve identically (no numbers)
    assert phases.resume_index("R9") == phases.resume_index("per_model_optimize") == idx
    # de27fdf: real (P1) 가 R1..R12 뒤로 이동 → R10·R11 어디서 resume 해도 P1 은 뒤라 실행됨.
    assert phases.should_run("P1", phases.resume_index("R10"))
    assert phases.should_run("P1", phases.resume_index("R11"))


def test_unknown_raises():
    with pytest.raises(KeyError):
        phases.order("R99")
    with pytest.raises(KeyError):
        phases.order(999)
    assert phases.is_known("R1") and not phases.is_known("nope")


def test_checkpoint_label_roundtrip(tmp_path):
    """CheckpointManager must save/load/exist by R/P label and report last-completed by order."""
    from simulation.pipeline.utils.checkpoint import CheckpointManager
    cm = CheckpointManager(tmp_path)
    cm.save("R9", {"x": 1}, "per_model_optimize")
    cm.save("R1", {"y": 2}, "data")
    assert cm.phase_exists("R9") and cm.phase_exists("R1")
    assert cm.load("R9") == {"x": 1}
    assert cm.load("P2") is None                 # not saved
    # last completed = the one with the highest pipeline order (R9 > R1)
    assert cm.get_last_completed_phase() == "R9"
    # file is named by label, not a magic number
    assert (tmp_path / "checkpoints" / "checkpoint_R9.json").exists()
