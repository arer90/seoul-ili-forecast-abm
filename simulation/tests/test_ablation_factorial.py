"""TDD — 2^3 factorial orchestrator (safe approximation, 2026-06-02).

orchestration 로직 전수 (CELLS / cell_env / select_panel / run_factorial / aggregate). 실제 cell 적합은
fit_fn 주입(여기선 mock) — 무거운 optimize_one_model 통합은 실행(post-main-run)서 검증.

run PER-FILE on macOS: KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 pytest <thisfile>
"""
import numpy as np
import pytest

from simulation.analytics.ablation_factorial import (
    FACTORS, CELLS, cell_env, select_panel, run_factorial, aggregate_effects,
    cell_run_env, cell_cli_command, collect_cell_predictions, reshape_to_per_model)


def test_cells_are_8_unique_factorial():
    assert len(CELLS) == 8 and len(set(CELLS)) == 8
    assert all(len(c) == 3 and set(c) <= {0, 1} for c in CELLS)
    assert FACTORS == ("preproc", "hp", "feature")


def test_cell_env_depth_toggles():
    # DEPTH ablation: bit=1=deep, bit=0=shallow (preproc always-ON·HP min5 → off 불가)
    e_deep = cell_env((1, 1, 1), hp_trials=15, preproc_trials=8)
    assert e_deep["MPH_PREPROC_TRIALS"] == "8"           # preproc deep
    assert e_deep["MPH_HP_OPTUNA_TRIALS"] == "15"        # HP deep
    assert e_deep["MPH_PHASE13_FEATURE_POOL"] == "full"
    e_shallow = cell_env((0, 0, 0))
    assert e_shallow["MPH_PREPROC_TRIALS"] == "1"        # preproc shallow=1 (PREPROC_SHALLOW)
    assert e_shallow["MPH_HP_OPTUNA_TRIALS"] == "5"      # HP shallow=5 (HP_SHALLOW, get_trials 하한)
    assert e_shallow["MPH_PHASE13_FEATURE_POOL"] == "basic"
    assert "MPH_PREPROC_OPTUNA" not in e_shallow         # no-op env 제거 (depth)


def test_cell_env_default_deep_values():
    """기본 deep = HP 20 / preproc 10 (사용자 B depth 채택 2026-06-02)."""
    e = cell_env((1, 1, 0))                              # preproc+HP deep
    assert e["MPH_HP_OPTUNA_TRIALS"] == "20"
    assert e["MPH_PREPROC_TRIALS"] == "10"


def test_select_panel_covers_all_families_plus_champion():
    cat = {"tree": ["XGBoost", "LightGBM"], "linear": ["ElasticNet"],
           "dl": ["DNN", "TimesNet"], "epi": ["EpiEstim"]}
    panel = select_panel("TimesNet", cat, n_per_family=1)
    # 4 family 전부 대표 1개씩 + champion(TimesNet) 포함
    assert "TimesNet" in panel                       # champion 포함
    assert len(panel) == 4                            # 4 family × 1 (champion = dl 대표)
    # 각 family 에서 최소 1개
    assert any(m in ("XGBoost", "LightGBM") for m in panel)
    assert "ElasticNet" in panel and "EpiEstim" in panel


def test_select_panel_champion_not_in_models_appended():
    cat = {"tree": ["XGBoost"], "linear": ["ElasticNet"]}
    panel = select_panel("Ensemble-NNLS", cat, n_per_family=1)
    assert "Ensemble-NNLS" in panel                  # 어느 family 에도 없으면 추가
    assert "XGBoost" in panel and "ElasticNet" in panel


def test_run_factorial_detects_hp_effect_via_mock_fit():
    """mock fit_fn: HP on → 작은 오차 → factorial 이 HP 주효과 검출."""
    rng = np.random.default_rng(0)
    n = 68
    y = rng.normal(10, 2, n)
    base_err = rng.normal(0, 3, n)

    def mock_fit(model, cell, env):
        p, h, f = cell
        err = base_err.copy()
        if h:                       # HP on → 큰 개선
            err = err * 0.3
        if f and h:                 # feature 는 HP 있을 때만 (상호작용)
            err = err * 0.6
        return y + err

    res = run_factorial(["XGBoost", "DNN"], y, fit_fn=mock_fit)
    assert set(res) == {"XGBoost", "DNN"}
    for model, fe in res.items():
        eff = {m["factor"]: m for m in fe["main"]}
        assert eff["hp"]["effect"] > 0 and eff["hp"]["sig"] == "yes", f"{model}: HP 주효과"
        inter = {d["pair"]: d["effect"] for d in fe["interactions"]}
        assert inter["hp:feature"] > 0, f"{model}: hp:feature 시너지"


def test_run_factorial_skips_none_cells():
    n = 12; y = np.arange(n, dtype=float)
    def mock_fit(model, cell, env):
        return None if cell == (0, 0, 0) else y + 1.0   # 한 cell 실패
    res = run_factorial(["M"], y, fit_fn=mock_fit)
    assert "M" in res                                    # None cell 제외해도 동작


def test_aggregate_effects():
    results = {
        "A": {"main": [{"factor": "hp", "effect": 1.0, "sig": "yes"},
                       {"factor": "feature", "effect": 0.01, "sig": "no"}]},
        "B": {"main": [{"factor": "hp", "effect": 0.8, "sig": "yes"},
                       {"factor": "feature", "effect": -0.02, "sig": "no"}]},
    }
    agg = aggregate_effects(results)
    assert abs(agg["hp"]["mean_effect"] - 0.9) < 1e-9
    assert agg["hp"]["n_sig"] == 2 and agg["hp"]["n_models"] == 2
    assert agg["feature"]["n_sig"] == 0               # feature 비유의 (null)


def test_cell_run_env_has_toggles_isolation_and_preserves_base():
    """subprocess env: depth 토글 + MPH_OUTPUT_ROOT 격리 + OPTUNA_ISOLATE(G-158) + base_env 보존."""
    # cell (preproc deep, HP shallow, feature deep)
    env = cell_run_env((1, 0, 1), "/tmp/cellA", hp_trials=10, base_env={"PATH": "/x", "HOME": "/h"})
    assert env["MPH_PREPROC_TRIALS"] == "10"              # preproc deep (1,_,_)
    assert env["MPH_HP_OPTUNA_TRIALS"] == "5"             # HP shallow (_,0,_) = 5
    assert env["MPH_PHASE13_FEATURE_POOL"] == "full"      # feature deep (_,_,1)
    assert env["MPH_OUTPUT_ROOT"] == "/tmp/cellA"         # main run 과 격리
    assert env["OPTUNA_ISOLATE"] == "1"                   # G-158 메모리 격리
    assert env["MPH_MULTI_SEED_RUN"] == "0"               # 단일 seed (main 5-seed 상속 차단)
    assert env["PATH"] == "/x" and env["HOME"] == "/h"    # base_env(venv/PATH) 보존


def test_cell_run_env_shallow_cell_basic_pool():
    env = cell_run_env((0, 0, 0), "/tmp/cellB")           # 전부 shallow
    assert env["MPH_PREPROC_TRIALS"] == "1"               # preproc shallow
    assert env["MPH_HP_OPTUNA_TRIALS"] == "5"             # HP shallow
    assert env["MPH_PHASE13_FEATURE_POOL"] == "basic"     # feature shallow = BASIC
    assert env["MPH_OUTPUT_ROOT"] == "/tmp/cellB"


def test_cell_cli_command_restricts_panel_and_phase():
    cmd = cell_cli_command(["XGBoost", "DNN"], scenario="full_light")
    assert "--models" in cmd and "XGBoost,DNN" in cmd      # 12-panel 만
    assert "--resume-from" in cmd and "per_model_optimize" in cmd   # phase 13
    assert "--scenario" in cmd and "full_light" in cmd
    # 빈 panel → --models 생략 (전체 fallback)
    assert "--models" not in cell_cli_command([])


def test_collect_cell_predictions_reads_json(tmp_path):
    """per_model_optimal/{model}.json 의 refit_test_predictions 수집 (list + str-repr + 결측)."""
    base = tmp_path / "results" / "per_model_optimal"
    base.mkdir(parents=True)
    (base / "XGBoost.json").write_text('{"refit_test_predictions": [1.0, 2.0, 3.0, 4.0]}', encoding="utf-8")
    # str-repr (json default=str 가 np.array 를 str 로 한 경우)
    (base / "DNN.json").write_text('{"refit_test_predictions": "[5.0 6.0 7.0 8.0]"}', encoding="utf-8")
    (base / "Bad.json").write_text('{"refit_test_predictions": null}', encoding="utf-8")   # 예측 없음 → skip
    got = collect_cell_predictions(tmp_path, ["XGBoost", "DNN", "Bad", "Missing"], n_test=3)
    assert set(got) == {"XGBoost", "DNN"}                  # Bad(null)/Missing(파일없음) 제외
    assert list(got["XGBoost"]) == [2.0, 3.0, 4.0]         # n_test=3 → 끝 3개
    assert list(got["DNN"]) == [6.0, 7.0, 8.0]             # str-repr 파싱 + 끝 3개


def test_reshape_to_per_model_transposes():
    import numpy as np
    per_cell = {(0, 0, 0): {"A": np.array([1.0]), "B": np.array([2.0])},
                (1, 0, 0): {"A": np.array([3.0])}}          # B 는 이 cell 결측
    out = reshape_to_per_model(per_cell, ["A", "B"])
    assert set(out["A"]) == {(0, 0, 0), (1, 0, 0)}          # A: 2 cell
    assert set(out["B"]) == {(0, 0, 0)}                     # B: 1 cell (결측 cell 제외)
    assert list(out["A"][(1, 0, 0)]) == [3.0]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
