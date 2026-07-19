"""Factorial ablation 전체 드라이버 (2026-06-02) — 8 cell × panel 순차 격리 실행 + 수집 + 리포트.

각 cell = factorial_cell_runner subprocess (cell_run_env: preproc/HP/feature 토글 + MPH_OUTPUT_ROOT
격리 + OPTUNA_ISOLATE G-158). **순차**(1 cell-run at a time → 메모리 안전) + **nice -19**(main 양보).
수집 = 각 cell 의 per_model_optimal/{model}.json refit_test_predictions → reshape → factorial_effects
(주효과+2-way 상호작용, HLN-DM 소표본 보정 + Holm 다중검정) → aggregate → factorial_report.{json,md}.

메커니즘 근거: GLOBAL config = frozen → in-process 토글 불가 → cell당 subprocess(env→fresh GLOBAL).
상세 = simulation/analytics/ablation_factorial.py docstring.

사용:
  .venv/bin/python -m simulation.scripts.run_factorial "<m1,m2,...>" [output_base] [hp_trials] [preproc_trials]
  panel 미지정("") → select_panel(champion="", CATEGORY_MODELS) = 12-family 대표 1개씩.
"""
import os
import sys
import json
import subprocess
from pathlib import Path


def _load_y_test():
    """run_data 1회(공유 캐시 hit) → (y_test actuals, n_test). 모든 cell 공통 held-out 심판."""
    import numpy as np
    from simulation.pipeline.config import PipelineConfig
    from simulation.pipeline.data import run_data
    cfg = PipelineConfig()
    cfg.data.cache_dir = str(Path(__file__).resolve().parents[1] / "cache")
    p1 = run_data(cfg)
    n_train, n_val = int(p1["n_train"]), int(p1["n_val"])
    n_test = int(p1.get("n_test") or 0)
    pool_end = n_train + n_val
    y = np.asarray(p1["y_all"], dtype=float)
    return y[pool_end:pool_end + n_test], n_test


def main(argv) -> int:
    from simulation.analytics.ablation_factorial import (
        CELLS, FACTORS, cell_run_env, collect_cell_predictions, reshape_to_per_model,
        aggregate_effects, select_panel)
    from simulation.analytics.ablation_stats import factorial_effects

    # ── panel ──
    if len(argv) > 1 and argv[1].strip():
        panel = [m.strip() for m in argv[1].split(",") if m.strip()]
    else:
        from simulation.models.registry import CATEGORY_MODELS   # 12-family SSOT (registry.py L405)
        panel = select_panel("", CATEGORY_MODELS, n_per_family=1)
    output_base = Path(argv[2]) if len(argv) > 2 else Path("simulation/results/factorial")
    hp_trials = int(argv[3]) if len(argv) > 3 else 20      # DEPTH: HP deep=20, shallow=5(고정). bit별 적용
    preproc_trials = int(argv[4]) if len(argv) > 4 else 10  # DEPTH: preproc deep=10, shallow=1(고정)
    output_base.mkdir(parents=True, exist_ok=True)
    print(f"[factorial] panel({len(panel)}): {panel}", flush=True)
    print(f"[factorial] budgets: hp_trials={hp_trials} preproc_trials={preproc_trials}", flush=True)

    y_test, n_test = _load_y_test()
    print(f"[factorial] y_test n={n_test}", flush=True)

    # ── 8 cell × per-model 격리 subprocess + 수집 (RESUME + 크래시 격리) ──
    # 모델별 별도 subprocess: 한 모델이 SIGABRT(예: CQR-LightGBM 의 OMP pthread_mutex_init 자원고갈,
    # 8모델 누적 후 9번째 폭발)로 죽어도 **그 모델만** 잃고 나머지는 진행. 게다가 각 모델이 fresh
    # 프로세스(누적 OMP 상태 0)라 크래셔도 단독 실행서 살 수 있음.
    # RESUME 2단계: ① 모델별 {model}.json 있으면 spawn 생략(run_data 도 skip) ② per_model_optimize
    # 내부 L2666 skip. 어디서 끊겨도/크래시나도 이어서.
    per_cell: dict = {}
    for i, cell in enumerate(CELLS, 1):
        cdir = output_base / ("cell_%d%d%d" % cell)
        cdir.mkdir(parents=True, exist_ok=True)
        env = cell_run_env(cell, cdir, hp_trials=hp_trials, preproc_trials=preproc_trials,
                           base_env=os.environ.copy())
        env.update({"KMP_DUPLICATE_LIB_OK": "TRUE", "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"})
        print(f"[factorial] ({i}/{len(CELLS)}) cell {cell} "
              f"preproc={cell[0]} hp={cell[1]} feature={cell[2]} → per-model 격리 실행...", flush=True)
        crashed = []
        # ensemble(메타 결합기)은 base 모델 예측이 필요 → base 먼저, ensemble 마지막.
        ensembles = [m for m in panel if m.startswith("Ensemble")]
        base_models = [m for m in panel if m not in ensembles]
        # 1) base 모델 (per-model 격리, factorial_cell_runner)
        for model in base_models:
            if model in collect_cell_predictions(cdir, [model], n_test=n_test):
                continue                          # RESUME: 이미 완료 → spawn/run_data 생략
            cmd = ["nice", "-n", "19", sys.executable, "-m",
                   "simulation.scripts.factorial_cell_runner", model]
            with (cdir / f"{model}.log").open("w", encoding="utf-8") as logf:
                rc = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT).returncode
            if rc != 0:
                crashed.append(f"{model}(rc={rc})")
        # 2) ensemble (base 완료 후 — base val(replay)+test 예측 NNLS 결합, factorial_ensemble_runner)
        for ens in ensembles:
            if ens in collect_cell_predictions(cdir, [ens], n_test=n_test):
                continue
            cmd = ["nice", "-n", "19", sys.executable, "-m",
                   "simulation.scripts.factorial_ensemble_runner", ens, ",".join(base_models)]
            with (cdir / f"{ens}.log").open("w", encoding="utf-8") as logf:
                rc = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT).returncode
            if rc != 0:
                crashed.append(f"{ens}(rc={rc})")
        preds = collect_cell_predictions(cdir, panel, n_test=n_test)
        per_cell[cell] = preds
        msg = f"[factorial] ({i}/{len(CELLS)}) cell {cell} collected={len(preds)}/{len(panel)}"
        if crashed:
            msg += f" | 크래시/실패(격리됨): {', '.join(crashed)}"
        print(msg, flush=True)

    # ── reshape → per-model factorial_effects → aggregate ──
    per_model = reshape_to_per_model(per_cell, panel)
    results = {m: factorial_effects(cells, y_test, factors=FACTORS)
               for m, cells in per_model.items()}
    agg = aggregate_effects(results)

    report = {"panel": panel, "n_test": n_test, "hp_trials": hp_trials,
              "preproc_trials": preproc_trials,
              "cells": ["".join(map(str, c)) for c in CELLS],
              "per_model": results, "aggregate": agg}
    (output_base / "factorial_report.json").write_text(json.dumps(report, indent=2, default=str))

    # ── MD 요약 ──
    md = ["# Factorial Ablation Report (2^3: preproc × HP × feature)", "",
          f"- panel: **{len(panel)}** models — {', '.join(panel)}",
          f"- n_test (held-out 심판): {n_test}",
          f"- budgets: hp_trials={hp_trials}, preproc_trials={preproc_trials}",
          f"- models with usable cells: {len(results)}/{len(panel)}", "",
          "## 요인 종합 (panel aggregate)", "",
          "| factor | mean_effect | n_sig / n_models |", "|---|---|---|"]
    for f, a in agg.items():
        md.append(f"| {f} | {a['mean_effect']:+.4f} | {a['n_sig']} / {a['n_models']} |")
    md += ["", "> mean_effect>0 = 그 요인 ON 이 손실(|y−ŷ|) 감소(개선). "
           "n_sig = HLN-DM(소표본 보정)+Holm(다중검정) p<0.05 모델 수.", "",
           "## 모델별 주효과 + 상호작용", "",
           "| model | preproc | hp | feature | hp:feature |", "|---|---|---|---|---|"]
    for m, r in results.items():
        eff = {x["factor"]: x for x in r.get("main", [])}
        inter = {d["pair"]: d["effect"] for d in r.get("interactions", [])}

        def _c(fac):
            e = eff.get(fac)
            if not e:
                return "—"
            return f"{e['effect']:+.3f}{'*' if e.get('sig') == 'yes' else ''}"
        md.append(f"| {m} | {_c('preproc')} | {_c('hp')} | {_c('feature')} | "
                  f"{inter.get('hp:feature', float('nan')):+.3f} |")
    md += ["", "> `*` = HLN-DM+Holm 유의(p<0.05). hp:feature>0 = HP 있을 때 feature 가 더 도움(시너지)."]
    (output_base / "factorial_report.md").write_text("\n".join(md))

    print(f"[factorial] DONE → {output_base}/factorial_report.{{json,md}}", flush=True)
    for f, a in agg.items():
        print(f"  {f}: mean_effect={a['mean_effect']:+.4f} "
              f"sig={a['n_sig']}/{a['n_models']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
