"""--oof-folds argparse + GLOBAL.training.oof_folds wiring.

사용자 명시 (2026-05-31): "oof_cv를 argparse로 기본을 5로 해서 해줘. 학위논문 제출용이니까."
= OOF WF-CV fold 수 기본값을 paper-grade 5 로 (이전 3), argparse 노출.

검증:
  1. config 기본값 = 5 (in-process + subprocess 모두 paper-grade)
  2. MPH_OOF_FOLDS env override (정상값 / 범위밖→default 5 / 비정수→default 5)
  3. argparse --oof-folds 기본 5, 명시값 파싱
  4. cmd_train(args) 가 args.oof_folds → GLOBAL.training.oof_folds + os.environ 전파
     (env 전파 = OPTUNA_ISOLATE subprocess 격리 학습에서도 fold 수 일관)
  5. _inline 의 oof_cv fold 수가 GLOBAL.training.oof_folds 를 읽음 (하드코딩 3 회귀 방지)

macOS: run PER-FILE with KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1.
"""
import os
import pytest

pytestmark = pytest.mark.filterwarnings("ignore")


def _fresh_training_config(monkeypatch, env_val):
    """env 적용 후 새 TrainingConfig 인스턴스 (default_factory 재평가)."""
    import simulation.config_global as cg
    if env_val is None:
        monkeypatch.delenv("MPH_OOF_FOLDS", raising=False)
    else:
        monkeypatch.setenv("MPH_OOF_FOLDS", str(env_val))
    return cg.TrainingConfig()


def test_config_default_5(monkeypatch):
    cfg = _fresh_training_config(monkeypatch, None)
    assert cfg.oof_folds == 5, f"학위논문 제출용 기본 5 여야 함, got {cfg.oof_folds}"


def test_env_override(monkeypatch):
    assert _fresh_training_config(monkeypatch, 7).oof_folds == 7
    assert _fresh_training_config(monkeypatch, 3).oof_folds == 3  # service 빠른 모드


def test_env_out_of_range_falls_back(monkeypatch):
    # _env_int: 범위(2..10) 밖이거나 비정수면 default 5 반환 (clamp 아님)
    assert _fresh_training_config(monkeypatch, 1).oof_folds == 5     # < lo=2
    assert _fresh_training_config(monkeypatch, 99).oof_folds == 5    # > hi=10
    assert _fresh_training_config(monkeypatch, "abc").oof_folds == 5  # non-int


def test_argparse_default_sentinel():
    """미지정 시 None sentinel (config 기본 5 가 effective). env 를 안 덮어쓰기 위함."""
    from simulation.__main__ import build_parser
    args = build_parser().parse_args(["train"])
    assert args.oof_folds is None, (
        f"미지정 시 None 이어야 (effective 5 는 config default). got {args.oof_folds} — "
        f"5 로 두면 export MPH_OOF_FOLDS=3 을 argparse 기본이 덮어씀")


def test_effective_default_is_5_via_config(monkeypatch):
    """argparse None + config default 5 → effective 기본 = 5 (학위논문 paper-grade)."""
    cfg = _fresh_training_config(monkeypatch, None)  # MPH_OOF_FOLDS unset
    from simulation.__main__ import build_parser
    args = build_parser().parse_args(["train"])
    effective = args.oof_folds if args.oof_folds is not None else cfg.oof_folds
    assert effective == 5, f"effective 기본 5 여야 함, got {effective}"


def test_argparse_explicit():
    from simulation.__main__ import build_parser
    args = build_parser().parse_args(["train", "--oof-folds", "8"])
    assert args.oof_folds == 8


def test_precedence_flag_absent_does_not_clobber_global(monkeypatch):
    """--oof-folds 미지정 → cmd_train 이 GLOBAL 미변경 (env/config 값 유지; CLI > env > 5)."""
    import simulation.config_global as cg
    import simulation.cli.training_commands as tc
    import simulation.pipeline.runner as runner

    sentinel_val = 4  # GLOBAL 에 env-유래 값이 있다고 가정 (≠ argparse 기본 5)
    orig = cg.GLOBAL.training.oof_folds
    orig_env = os.environ.get("MPH_OOF_FOLDS")
    object.__setattr__(cg.GLOBAL.training, "oof_folds", sentinel_val)

    class _Sentinel(Exception):
        pass

    monkeypatch.setattr(runner, "run_pipeline",
                        lambda *a, **k: (_ for _ in ()).throw(_Sentinel()), raising=True)
    monkeypatch.setattr(runner, "run_dry_run", lambda *a, **k: None, raising=False)

    from simulation.__main__ import build_parser
    args = build_parser().parse_args(["train"])  # --oof-folds 미지정 → None
    try:
        with pytest.raises(_Sentinel):
            tc.cmd_train(args)
        assert cg.GLOBAL.training.oof_folds == sentinel_val, \
            "flag 미지정인데 GLOBAL 이 5 로 덮어써짐 — env override 가 깨짐"
    finally:
        object.__setattr__(cg.GLOBAL.training, "oof_folds", orig)
        if orig_env is None:
            os.environ.pop("MPH_OOF_FOLDS", None)
        else:
            os.environ["MPH_OOF_FOLDS"] = orig_env


def test_cmd_train_wires_global(monkeypatch):
    """cmd_train(args) 가 args.oof_folds → GLOBAL.training.oof_folds + env 전파 (heavy phase 진입 전)."""
    import simulation.config_global as cg
    import simulation.cli.training_commands as tc
    import simulation.pipeline.runner as runner

    orig = cg.GLOBAL.training.oof_folds
    orig_env = os.environ.get("MPH_OOF_FOLDS")

    class _Sentinel(Exception):
        pass

    def _boom(*a, **k):
        raise _Sentinel()

    # run_pipeline/run_dry_run 직전까지만 실행 — 실제 학습 phase 진입 차단.
    # cmd_train 의 local `from ...runner import run_pipeline` 가 patch 된 binding 을 가져감.
    monkeypatch.setattr(runner, "run_pipeline", _boom, raising=True)
    monkeypatch.setattr(runner, "run_dry_run", _boom, raising=False)

    from simulation.__main__ import build_parser
    args = build_parser().parse_args(["train", "--oof-folds", "8"])
    try:
        with pytest.raises(_Sentinel):
            tc.cmd_train(args)
        assert cg.GLOBAL.training.oof_folds == 8, "wiring: GLOBAL.training.oof_folds 미설정"
        assert os.environ.get("MPH_OOF_FOLDS") == "8", "wiring: env 미전파 (subprocess 격리 영향)"
    finally:
        # GLOBAL.training 은 frozen → 복원도 object.__setattr__ (cmd_train 과 동일 escape)
        object.__setattr__(cg.GLOBAL.training, "oof_folds", orig)
        if orig_env is None:
            os.environ.pop("MPH_OOF_FOLDS", None)
        else:
            os.environ["MPH_OOF_FOLDS"] = orig_env


def test_cmd_train_clamps_out_of_range(monkeypatch):
    """--oof-folds 범위[2,10] 밖이면 5 (paper-grade) — _env_int 와 동일 의미."""
    import simulation.config_global as cg
    import simulation.cli.training_commands as tc
    import simulation.pipeline.runner as runner

    orig = cg.GLOBAL.training.oof_folds
    orig_env = os.environ.get("MPH_OOF_FOLDS")

    class _Sentinel(Exception):
        pass

    monkeypatch.setattr(runner, "run_pipeline",
                        lambda *a, **k: (_ for _ in ()).throw(_Sentinel()), raising=True)
    monkeypatch.setattr(runner, "run_dry_run", lambda *a, **k: None, raising=False)

    from simulation.__main__ import build_parser
    args = build_parser().parse_args(["train", "--oof-folds", "99"])  # > hi=10
    try:
        with pytest.raises(_Sentinel):
            tc.cmd_train(args)
        assert cg.GLOBAL.training.oof_folds == 5, "범위밖 → 5 fallback 이어야"
        assert os.environ.get("MPH_OOF_FOLDS") == "5"
    finally:
        object.__setattr__(cg.GLOBAL.training, "oof_folds", orig)
        if orig_env is None:
            os.environ.pop("MPH_OOF_FOLDS", None)
        else:
            os.environ["MPH_OOF_FOLDS"] = orig_env


def test_cmd_train_no_attr_is_noop(monkeypatch):
    """args 에 oof_folds 없으면 (구버전 Namespace) GLOBAL 미변경 — getattr None gate."""
    import argparse
    import simulation.config_global as cg
    import simulation.cli.training_commands as tc
    import simulation.pipeline.runner as runner

    orig = cg.GLOBAL.training.oof_folds

    class _Sentinel(Exception):
        pass

    monkeypatch.setattr(runner, "run_pipeline", lambda *a, **k: (_ for _ in ()).throw(_Sentinel()), raising=True)
    monkeypatch.setattr(runner, "run_dry_run", lambda *a, **k: None, raising=False)

    # oof_folds 속성을 일부러 누락한 최소 Namespace (list_models 로 즉시 early-return)
    args = argparse.Namespace(list_models=True)
    tc.cmd_train(args)  # list_models 분기 → return (wiring 도달 X)
    assert cg.GLOBAL.training.oof_folds == orig, "list_models early-return 시 GLOBAL 불변이어야"


def test_inline_reads_global_not_hardcoded():
    """_inline 의 oof_cv fold 수 = GLOBAL.training.oof_folds (하드코딩 `else 3` 회귀 방지)."""
    import inspect
    import simulation.pipeline._inline_optuna_3stage as mod
    src = inspect.getsource(mod)
    assert "else GLOBAL.training.oof_folds" in src, \
        "oof_cv fold 수가 GLOBAL.training.oof_folds 를 읽어야 함 (하드코딩 3 회귀)"
    assert 'best_by == "research_5fold"' in src, "research_5fold 은 명시 5 유지"


def test_phase13_gate_recompute_uses_global_folds():
    """champion-gate 4지표 OOF 재계산이 selection 과 동일 fold 수(GLOBAL.training.oof_folds) 사용.

    selection(_inline:493)=GLOBAL, gate(_oof_cv_metrics)=default 면 --oof-folds 3 시 불일치
    (selection 3 / gate 5). gate 호출에 n_folds=GLOBAL.training.oof_folds 명시 회귀 방지.
    """
    import inspect
    import simulation.pipeline.per_model_optimize as mod
    src = inspect.getsource(mod)
    assert "n_folds=GLOBAL.training.oof_folds" in src, \
        "champion-gate _oof_cv_metrics 가 GLOBAL.training.oof_folds 를 명시 전달해야 (selection 일치)"
