"""Smoke test for cli/ extracted handlers (Day 8 C2 split + Day 10 fixes).

Phase C2 (Day 8) 에서 25 cmd_* handlers 를 9 cli/ modules 로 추출. Day 9 에
SCENARIOS NameError regression 발견 (cmd_train 이 module-level dict 참조).
이번 smoke test 는 그런 hidden regression 을 사전 차단.

검증 항목 (per-module):
    1. import 성공 (모듈 로드)
    2. 모든 export 가 callable (cmd_X 함수)
    3. __main__ re-export identity 일치
    4. SCENARIOS / ALL_MODELS 의 module-level access (training_commands)
    5. _state helpers (state_path / save_state / load_state / clear_state) 동작

직접 실행:
    .venv/bin/python -m simulation.tests.smoke_cli_extracted

pytest auto-skip:
    test_smoke_cli_extracted_skipped() — pytest 가 collect 만 하고 skip
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def smoke_db_commands() -> tuple[int, int]:
    """Verify cli.db_commands: 4 cmd_* handlers + __main__ re-export identity."""
    from simulation.cli import db_commands as m
    expected = ["cmd_db_init", "cmd_db_status", "cmd_db_optimize", "cmd_db_migrate_v22"]
    n_ok, n_fail = 0, 0
    for name in expected:
        if not callable(getattr(m, name, None)):
            print(f"  ✗ db_commands.{name}: not callable"); n_fail += 1
        else:
            n_ok += 1
    # __main__ re-export identity
    import simulation.__main__ as mn
    for name in expected:
        if getattr(mn, name, None) is not getattr(m, name):
            print(f"  ✗ __main__.{name} ≠ cli.db_commands.{name}"); n_fail += 1
        else:
            n_ok += 1
    return n_ok, n_fail


def smoke_maintenance_commands() -> tuple[int, int]:
    from simulation.cli import maintenance_commands as m
    expected = ["cmd_maintain", "cmd_prune", "cmd_doctor", "cmd_auto_update"]
    n_ok, n_fail = 0, 0
    import simulation.__main__ as mn
    for name in expected:
        f1 = getattr(m, name, None)
        f2 = getattr(mn, name, None)
        if not callable(f1):
            print(f"  ✗ {name} not callable"); n_fail += 1
        elif f1 is not f2:
            print(f"  ✗ __main__.{name} ≠ cli.maintenance.{name}"); n_fail += 1
        else:
            n_ok += 2
    return n_ok, n_fail


def smoke_utility_commands() -> tuple[int, int]:
    from simulation.cli import utility_commands as m
    expected = [
        "cmd_extract_pdf", "cmd_verify_audit", "cmd_freeze_paper_primary",
        "cmd_visualize", "cmd_feature_importance", "cmd_rehydrate", "cmd_list_models",
    ]
    n_ok, n_fail = 0, 0
    import simulation.__main__ as mn
    for name in expected:
        f1 = getattr(m, name, None)
        f2 = getattr(mn, name, None)
        if not callable(f1):
            print(f"  ✗ {name} not callable"); n_fail += 1
        elif f1 is not f2:
            print(f"  ✗ __main__.{name} ≠ cli.utility.{name}"); n_fail += 1
        else:
            n_ok += 2
    return n_ok, n_fail


def smoke_sim_commands() -> tuple[int, int]:
    from simulation.cli import sim_commands as m
    expected = ["cmd_sim", "cmd_mcp_server"]
    n_ok, n_fail = 0, 0
    import simulation.__main__ as mn
    for name in expected:
        f1 = getattr(m, name, None)
        f2 = getattr(mn, name, None)
        if not callable(f1):
            print(f"  ✗ {name} not callable"); n_fail += 1
        elif f1 is not f2:
            print(f"  ✗ __main__.{name} ≠ cli.sim.{name}"); n_fail += 1
        else:
            n_ok += 2
    return n_ok, n_fail


def smoke_data_commands() -> tuple[int, int]:
    from simulation.cli import data_commands as m
    expected = ["cmd_import_external", "cmd_orchestrate"]
    n_ok, n_fail = 0, 0
    import simulation.__main__ as mn
    for name in expected:
        f1 = getattr(m, name, None)
        f2 = getattr(mn, name, None)
        if not callable(f1):
            print(f"  ✗ {name} not callable"); n_fail += 1
        elif f1 is not f2:
            print(f"  ✗ __main__.{name} ≠ cli.data.{name}"); n_fail += 1
        else:
            n_ok += 2
    return n_ok, n_fail


def smoke_pipeline_commands() -> tuple[int, int]:
    from simulation.cli import pipeline_commands as m
    expected = ["cmd_bootstrap"]
    n_ok, n_fail = 0, 0
    import simulation.__main__ as mn
    for name in expected:
        f1 = getattr(m, name, None)
        f2 = getattr(mn, name, None)
        if not callable(f1):
            print(f"  ✗ {name} not callable"); n_fail += 1
        elif f1 is not f2:
            print(f"  ✗ __main__.{name} ≠ cli.pipeline.{name}"); n_fail += 1
        else:
            n_ok += 2
    return n_ok, n_fail


def smoke_inference_commands() -> tuple[int, int]:
    from simulation.cli import inference_commands as m
    expected = ["cmd_predict_real"]
    n_ok, n_fail = 0, 0
    import simulation.__main__ as mn
    for name in expected:
        f1 = getattr(m, name, None)
        f2 = getattr(mn, name, None)
        if not callable(f1):
            print(f"  ✗ {name} not callable"); n_fail += 1
        elif f1 is not f2:
            print(f"  ✗ __main__.{name} ≠ cli.inference.{name}"); n_fail += 1
        else:
            n_ok += 2
    return n_ok, n_fail


def smoke_training_commands() -> tuple[int, int]:
    """Day 9 SCENARIOS NameError regression 차단 — module-level globals 검증."""
    from simulation.cli import training_commands as tc
    n_ok, n_fail = 0, 0
    expected = ["cmd_collect", "cmd_train", "cmd_train_all", "cmd_run_all"]
    import simulation.__main__ as mn
    for name in expected:
        f1 = getattr(tc, name, None)
        f2 = getattr(mn, name, None)
        if not callable(f1):
            print(f"  ✗ {name} not callable"); n_fail += 1
        elif f1 is not f2:
            print(f"  ✗ __main__.{name} ≠ cli.training.{name}"); n_fail += 1
        else:
            n_ok += 2
    # SCENARIOS / ALL_MODELS module-level access (Day 9 regression)
    if not hasattr(tc, "SCENARIOS"):
        print("  ✗ training_commands missing SCENARIOS (Day 9 regression)"); n_fail += 1
    elif len(tc.SCENARIOS) < 15:
        print(f"  ✗ training_commands.SCENARIOS too small: {len(tc.SCENARIOS)}"); n_fail += 1
    else:
        n_ok += 1
    if not hasattr(tc, "ALL_MODELS"):
        print("  ✗ training_commands missing ALL_MODELS"); n_fail += 1
    elif sum(len(v) for v in tc.ALL_MODELS.values()) < 50:
        print(f"  ✗ ALL_MODELS too small: {sum(len(v) for v in tc.ALL_MODELS.values())}"); n_fail += 1
    else:
        n_ok += 1
    return n_ok, n_fail


def smoke_scenarios() -> tuple[int, int]:
    """SCENARIOS dict + ALL_MODELS 정합 검증."""
    from simulation.cli._scenarios import ALL_MODELS, SCENARIOS
    n_ok, n_fail = 0, 0
    if len(SCENARIOS) < 15:
        print(f"  ✗ SCENARIOS count {len(SCENARIOS)} < 15"); n_fail += 1
    else:
        n_ok += 1
    total = sum(len(v) for v in ALL_MODELS.values())
    if total < 50:
        print(f"  ✗ ALL_MODELS total {total} < 50"); n_fail += 1
    else:
        n_ok += 1
    # Verify dl-only scenario uses live registry models (no LSTM/Transformer stale)
    dl_models = SCENARIOS.get("dl-only", {}).get("models", [])
    if "LSTM" in dl_models or "Transformer" in dl_models:
        print("  ✗ SCENARIOS[dl-only].models still has stale LSTM/Transformer"); n_fail += 1
    else:
        n_ok += 1
    return n_ok, n_fail


def smoke_state_helpers() -> tuple[int, int]:
    """_state.py: save_state/load_state roundtrip."""
    from simulation.cli._state import (
        _clear_state, _load_state, _save_state, _state_path,
    )
    n_ok, n_fail = 0, 0
    test_name = "smoke_cli_test_marker"
    payload = {"k1": "v1", "k2": 42, "list": [1, 2, 3]}
    try:
        _save_state(test_name, payload)
        loaded = _load_state(test_name)
        if loaded == payload:
            n_ok += 1
        else:
            print(f"  ✗ roundtrip mismatch: {loaded} != {payload}"); n_fail += 1
        _clear_state(test_name)
        # After clear, load should return empty dict
        if _load_state(test_name) == {}:
            n_ok += 1
        else:
            print("  ✗ clear_state didn't fully clear"); n_fail += 1
    except Exception as e:
        print(f"  ✗ state roundtrip exception: {type(e).__name__}: {e}"); n_fail += 1
    return n_ok, n_fail


def smoke_main_dispatch() -> tuple[int, int]:
    """build_parser() works + dispatch table reaches all 25 cmd_*."""
    import simulation.__main__ as mn
    n_ok, n_fail = 0, 0
    try:
        parser = mn.build_parser()
        n_ok += 1
    except Exception as e:
        print(f"  ✗ build_parser failed: {type(e).__name__}: {e}"); n_fail += 1
        return n_ok, n_fail
    # Verify all 25 cmd_* present
    all_25 = [
        "cmd_db_init", "cmd_db_status", "cmd_db_optimize", "cmd_db_migrate_v22",
        "cmd_maintain", "cmd_prune", "cmd_doctor", "cmd_auto_update",
        "cmd_sim", "cmd_mcp_server",
        "cmd_extract_pdf", "cmd_verify_audit", "cmd_freeze_paper_primary",
        "cmd_feature_importance", "cmd_rehydrate", "cmd_visualize", "cmd_list_models",
        "cmd_import_external", "cmd_orchestrate",
        "cmd_bootstrap",
        "cmd_predict_real",
        "cmd_collect", "cmd_train", "cmd_train_all", "cmd_run_all",
    ]
    for name in all_25:
        if not hasattr(mn, name):
            print(f"  ✗ __main__ missing {name}"); n_fail += 1
        else:
            n_ok += 1
    return n_ok, n_fail


def main() -> int:
    """Run all 10 smoke checks. Exit code = total failures."""
    suites = [
        ("db_commands", smoke_db_commands),
        ("maintenance_commands", smoke_maintenance_commands),
        ("utility_commands", smoke_utility_commands),
        ("sim_commands", smoke_sim_commands),
        ("data_commands", smoke_data_commands),
        ("pipeline_commands", smoke_pipeline_commands),
        ("inference_commands", smoke_inference_commands),
        ("training_commands", smoke_training_commands),
        ("scenarios", smoke_scenarios),
        ("state_helpers", smoke_state_helpers),
        ("main_dispatch", smoke_main_dispatch),
    ]
    print("=" * 64)
    print("  cli/ smoke test (Day 8 C2 split + Day 9/10 fixes)")
    print("=" * 64)
    total_ok = 0
    total_fail = 0
    for name, fn in suites:
        try:
            ok, fail = fn()
        except Exception as e:
            print(f"  ✗ [{name}] EXCEPTION: {type(e).__name__}: {e}")
            ok, fail = 0, 1
        total_ok += ok
        total_fail += fail
        sym = "✓" if fail == 0 else "✗"
        print(f"  {sym} [{name:<22s}] {ok} ok, {fail} fail")
    print("=" * 64)
    print(f"  TOTAL: {total_ok} ok, {total_fail} fail")
    print("=" * 64)
    return total_fail


def test_smoke_cli_skipped() -> None:
    """Pytest collection target — skipped (run via CLI for proper output)."""
    import pytest

    pytest.skip(
        "Run smoke directly: `python -m simulation.tests.smoke_cli_extracted`"
    )


if __name__ == "__main__":
    raise SystemExit(main())
