"""G-306 + 2026-06-20: real_eval(P1) dispatch invariants.

G-306 (preserved): the operational/deployment forecast must use the FINAL best-WIS
champion + its R9 optimized real prediction, so real_eval must dispatch AFTER both
per_model_optimize(R9) and per_model_eval(R10) — never before (which would silently
revert to default-HP / WF-CV-best behaviour).

2026-06-20 (사용자 지시): P1 further relocated to AFTER the ENTIRE research track
(R11 SHAP, R12 comprehensive) = production-track start. R12 is DECOUPLED from real_eval —
it sources champion/families from R9/R10, not P1's real-slab. These tests guard against
(a) a re-reorder back to real_eval-before-R9, and (b) re-coupling R12 to P1.

macOS: run PER-FILE.
"""
from pathlib import Path

import simulation.pipeline.runner as _runner

_SRC = Path(_runner.__file__).read_text(encoding="utf-8")


def test_real_eval_dispatched_after_per_model_eval():
    """G-306 invariant: real_eval after optimize(R9) + eval(R10) → champion available."""
    i_opt = _SRC.index("run_per_model_optimize(")
    i_eval = _SRC.index("run_per_model_eval(")
    i_real = _SRC.index("run_real_eval(")
    assert i_opt < i_eval < i_real, (
        "dispatch order must be per_model_optimize(R9) → per_model_eval(R10) → "
        "real_eval(P1); real_eval must come AFTER so the champion is available"
    )


def test_real_eval_runs_after_comprehensive():
    """2026-06-20: P1 now dispatches AFTER all research incl. SHAP(R11) + comprehensive(R12)."""
    i_real = _SRC.index("run_real_eval(")
    i_shap = _SRC.index("run_shap(")
    i_comp = _SRC.index("run_comprehensive_eval(")
    assert i_shap < i_comp < i_real, (
        "P1(real_eval) must dispatch AFTER R11(shap) and R12(comprehensive) — the "
        "production track starts only once the research track is complete"
    )


def test_p1_gate_uses_rp_label():
    """Gate uses should_run(\"P1\") (R/P label); numeric resume_from gates are removed."""
    assert 'should_run("P1", resume_from)' in _SRC, \
        "P1 gate must use the R/P label should_run(\"P1\")"
