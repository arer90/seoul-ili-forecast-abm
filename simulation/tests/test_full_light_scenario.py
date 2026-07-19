"""Freeze tests for the Stage 3 full_light scenario.

Stage 3 introduces a mid-weight scenario that decouples Optuna
inline-trial epochs from the final-fit epochs. These tests fail loudly
if someone silently changes the preset or removes the wiring.
"""
from __future__ import annotations

import pytest


def test_full_light_registered():
    """full_light must be present in the SCENARIOS dict."""
    from simulation.__main__ import SCENARIOS
    assert "full_light" in SCENARIOS, "Stage 3: full_light scenario is missing"


def test_full_light_values_frozen():
    """The Stage 3 preset values are an intentional choice; freeze them.

    If you need to change these numbers, update the Stage 3 section of
    docs/internal/stage_plan.md at the same time.
    """
    from simulation.__main__ import SCENARIOS
    scn = SCENARIOS["full_light"]
    # 2026-06-02 (codex+Gemini NO-GO fix): feature 선택 = phase 13 STABILITY 전용 → 옛 external/
    # pre-pipeline feature-Optuna 제거. optuna_mode all→none, rerun_feature_optuna/inline_epochs 삭제.
    assert scn["optuna_mode"] == "none", "full_light: feature 선택은 phase 13 — external Optuna 제거"
    assert "rerun_feature_optuna" not in scn, "full_light: 누수성 pre-pipeline feature-Optuna 제거됨"
    assert scn["optuna_trials"] == 30, "full_light: trials frozen at 30"
    assert scn["epochs"] == 200, "full_light: final-fit epochs frozen at 200"
    assert scn["early_stopping_patience"] == 20, (
        "full_light: early_stopping_patience frozen at 20"
    )
    assert scn["conformal_holdout_weeks"] == 26, "full_light: honest split-conformal holdout"


def test_cli_parser_accepts_new_flags():
    """The train subparser must expose --inline-epochs and --early-stopping-patience."""
    from simulation.__main__ import build_parser
    parser = build_parser()
    args = parser.parse_args([
        "train", "--scenario", "full_light",
        "--inline-epochs", "77",
        "--early-stopping-patience", "13",
    ])
    assert args.inline_epochs == 77, "--inline-epochs did not reach args"
    assert args.early_stopping_patience == 13, (
        "--early-stopping-patience did not reach args"
    )


def test_pipeline_runner_distinguishes_epochs_from_inline():
    """When both --epochs and --inline-epochs are given, they go to different config fields."""
    from simulation.pipeline.runner import build_cli_parser
    from simulation.pipeline.config import PipelineConfig

    parser = build_cli_parser()
    args = parser.parse_args([
        "--epochs", "200",
        "--inline-epochs", "50",
        "--early-stopping-patience", "20",
        "--optuna-mode", "all",
        "--optuna-trials", "30",
    ])
    cfg = PipelineConfig.from_cli(args)

    # Full-fit epochs
    assert cfg.training.epochs == 200
    # Optuna inline epochs — must NOT equal training.epochs
    assert cfg.optuna.epochs_per_trial == 50, (
        "inline_epochs must override optuna.epochs_per_trial independently"
    )
    assert cfg.training.early_stopping_patience == 20
    assert cfg.optuna.mode == "all"
    assert cfg.optuna.trials == 30


def test_epochs_alone_still_sets_both_for_back_compat():
    """Back-compat: callers that only pass --epochs (no --inline-epochs) should
    still see both fields updated to the same value. This was the pre-Stage-3
    behavior and existing scenarios (dl-only, quick-test) depend on it."""
    from simulation.pipeline.runner import build_cli_parser
    from simulation.pipeline.config import PipelineConfig

    parser = build_cli_parser()
    args = parser.parse_args(["--epochs", "100"])
    cfg = PipelineConfig.from_cli(args)
    assert cfg.training.epochs == 100
    assert cfg.optuna.epochs_per_trial == 100, (
        "back-compat broken: --epochs alone should still set both"
    )


def test_optuna_mode_all_is_accepted():
    """Stage 3 uses optuna_mode='all'. The train subparser must accept it."""
    from simulation.__main__ import build_parser
    parser = build_parser()
    args = parser.parse_args(["train", "--optuna-mode", "all"])
    assert args.optuna_mode == "all"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
