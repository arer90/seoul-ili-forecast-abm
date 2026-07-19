"""Training pipeline CLI commands — extracted from __main__.py.

Phase C2 partial 5차 (2026-05-12 final): the 4 large training-pipeline
handlers (collect / train / train_all / run_all) + 3 helpers
(_OPTUNA_FEAT_MODEL_MAP / _map_models_to_optuna_keys / _rerun_feature_optuna)
moved here. ~925 lines extracted as a single batch — the handlers have
closures over each other and must be co-located.

Inter-module deps (closures):
    - cmd_train_all calls cmd_train (same module)
    - cmd_run_all calls cmd_bootstrap (← cli.pipeline_commands),
                        cmd_collect (same module),
                        cmd_db_optimize (← cli.db_commands),
                        cmd_train_all (same module)

State helpers (_save_state / _load_state / _clear_state / _state_path) are
imported from cli._state (Phase C2 day7).

References:
    - G-170 v1/v2 (`_map_models_to_optuna_keys` + feature-optuna pre-stage skip)
    - G-175 multi-criteria filter env (MAPE 20 / PICP95 0.90)
"""
from __future__ import annotations

import argparse
import logging
import sys

from simulation.cli._scenarios import ALL_MODELS, SCENARIOS
from simulation.cli._state import (
    _clear_state,
    _load_state,
    _save_state,
    _state_path,
)
from simulation.cli.db_commands import cmd_db_optimize
from simulation.cli.pipeline_commands import cmd_bootstrap


log = logging.getLogger(__name__)


def cmd_collect(args):
    from simulation.collectors import run_collection, list_groups

    if args.list_groups:
        groups = list_groups()
        print(f"\n{'Group':<6} {'Available':<10} {'Est.':>6}  Description")
        print("-" * 60)
        for g, info in sorted(groups.items()):
            avail = "YES" if info["available"] else "NO"
            print(f"  {g:<4}   {avail:<10} {info['est_sec']:>5}s  {info['desc']}")
        return

    # fix: accept "all" as an explicit "run every group" sentinel
    # (matches the `--groups` help text "(default: all)" and the default
    # value of --collect-groups in cmd_train / cmd_run_all). Previously
    # passing `--groups all` silently skipped every group because
    # orchestrator.GROUP_INFO has no entry named "all".
    #
    # fix: --groups now accepts both space- and comma-separated
    # inputs (plus mixed). nargs='+' gives us a list from space-separation;
    # any element containing commas is split further. "all" sentinel still
    # wins. Also keeps back-compat with callers that pass a raw string via
    # argparse.Namespace(groups="E,D,B") (see cmd_run_all → _run_collect_stage).
    raw = args.groups
    if raw is None:
        groups = None
    else:
        tokens: list[str] = []
        # argparse with nargs='+' yields list; old callers may still pass str
        iterable = raw if isinstance(raw, (list, tuple)) else [str(raw)]
        for item in iterable:
            for token in str(item).split(","):
                t = token.strip()
                if t:
                    tokens.append(t)
        if not tokens or any(t.lower() == "all" for t in tokens):
            groups = None  # → run_collection falls back to DEFAULT_ORDER
        else:
            groups = tokens
    run_collection(groups=groups, force=args.force,
                   backfill_days=getattr(args, "backfill_days", None))

    # Auto-refresh the web map's file-backed aggregates so the dashboard reflects
    # the freshly-collected DB (사용자 채택 2026-06-06: orchestrate 자동 연결).
    # The LIVE /api/mcp + /api/overlays layers already update per-request; this
    # syncs the snapshot layers (air/subway/weather/POI/disease-vax/bus/seir-init).
    # Degrade-and-continue + subprocess isolation: a web-builder failure or hang
    # must never fail data collection (--no-web-refresh opts out).
    if not getattr(args, "no_web_refresh", False):
        try:
            import subprocess
            from pathlib import Path
            refresh = Path(__file__).resolve().parents[2] / "web" / "scripts" / "refresh_web_data.py"
            if refresh.exists():
                r = subprocess.run([sys.executable, str(refresh)],
                                   capture_output=True, text=True, timeout=900)
                last = (r.stdout or "").strip().splitlines()
                print(f"[collect] web data refresh: {last[-1] if last else '(no output)'}")
        except Exception as e:  # noqa: BLE001 — web refresh is best-effort
            print(f"[collect] web data refresh skipped: {type(e).__name__}: {e}")


# G-170 (2026-05-03): 사용자 모델명 → run_optuna_feature_selection.py key 매핑
# train_by_category.sh:get_models() 의 한글-친화 이름과 ALL_MODELS (lower_underscore) 매핑.
_OPTUNA_FEAT_MODEL_MAP = {
    # Tree
    "XGBoost": "xgboost", "LightGBM": "lightgbm",
    "RandomForest": "randomforest", "GradientBoosting": "gradientboosting",
    "CatBoost": None,   # G-169 신규 — run_optuna_feature_selection.py 미지원 (다음 sprint)
    # Linear
    "ElasticNet": "elasticnet", "BayesianRidge": "bayesianridge",
    "NegBinGLM": "negbinglm", "NegBinGLM-V7": "negbinglm",
    "PoissonAutoreg": "poissonautoreg",
    # Kernel
    "KRR": "krr", "SVR-Linear": "svr_linear", "SVR-RBF": "svr_rbf",
    # Other
    "GAM-Spline": "gam", "GP-RBF-Periodic": "gp_rbf_periodic",
    "BayesianMCMC": "bayesianmcmc",
    # DL — DNN proxy
    "DNN": "dnn", "DNN-Optuna": "dnn",
    "TabularDNN-Lite": "tabular_dnn",
    # DL-seq
    "TCN": "tcn", "TCN-Optuna": "tcn",
    # Modern TS
    "PatchTST": "patchtst", "iTransformer": "itransformer",
    "Mamba": "mamba", "TimesNet": "timesnet",
    "N-BEATS": "nbeats", "N-HiTS": "nhits", "TiDE": "tide", "TFT": "tft",
    "DeepAR": "dnn", "RNN": "dnn",  # pytorch-forecasting → DNN proxy
    # Graph
    "GCN": "ge_dnn", "GAT": "ge_dnn",  # graph models → ge_dnn feature proxy
    # Mech — H6 (2026-05-03): PINN-Lite 만 X 사용 (X_train index uses=2),
    #        나머지 (MP-PINN proxy 동일, SEIR-V2-Forced/Rt-Augmented 는 X 무시)
    "PINN-Lite": "dnn",       # PINN proxy → DNN feature optuna
    "MP-PINN": "dnn",         # 동일 proxy (PINN family)
    # SEIR-V2-Forced / Rt-Augmented = X 무시 → 매핑 dict 부재 = SKIP
    # Foundation / Ensemble — feature optuna 무관 (skip)
}


def _map_models_to_optuna_keys(models_csv: str | None) -> str | None:
    """사용자 친화 모델명 (XGBoost / DNN-Optuna 등) → optuna feature selection key (G-170, D-4).

    `train_by_category.sh` 또는 `--models` CLI 가 받은 사용자 친화 이름을
    `simulation/tools/run_optuna_feature_selection.py` 의 `--model` 인자가
    이해하는 lower_underscore key (xgboost / lightgbm / dnn 등) 로 변환.
    Dedup + unsupported skip 적용.

    Args:
        models_csv: 콤마구분 사용자 친화 모델명. 예:
                    "XGBoost,LightGBM,RandomForest,GradientBoosting,CatBoost".
                    None / "" → None 반환 (caller 가 "all" fallback).

    Returns:
        콤마구분 lower_underscore key string (dedup 후, dict 순서 유지). 예:
            "xgboost,lightgbm,randomforest,gradientboosting"
        모든 모델 미지원 / valid 0 시 None (caller "all" fallback).

    Raises:
        절대 raise X — 미지원 모델은 stdout 에 print 후 skip.

    Side effects:
        - stdout print: skip 된 모델 list (예: "[feature-optuna] skip (미지원): ['CatBoost']")

    Performance: O(N) where N = colon-separated entry count (~5-15).
    Caller responsibility: caller 가 "all" fallback 처리 (예: `_model_arg = result or "all"`).

    Example:
        >>> _map_models_to_optuna_keys("XGBoost,LightGBM,CatBoost")
        # CatBoost 미지원 (run_optuna_feature_selection.py ALL_MODELS 에 없음)
        '<stdout>: [feature-optuna] skip: [CatBoost]'
        'xgboost,lightgbm'

        >>> _map_models_to_optuna_keys("DNN,DNN-Optuna,TinyMLP")
        # 모두 dnn proxy 로 dedup
        'dnn'

        >>> _map_models_to_optuna_keys("PINN-Lite,Bayesian-SEIR")
        # 모두 미지원 (mech 카테고리는 feature optuna 무관)
        None

    See: `_OPTUNA_FEAT_MODEL_MAP` (32 entry mapping dict), G-170 (bottleneck fix).
    """
    if not models_csv:
        return None
    out: list[str] = []
    seen: set[str] = set()
    skipped: list[str] = []
    for raw in models_csv.split(","):
        name = raw.strip()
        if not name:
            continue
        key = _OPTUNA_FEAT_MODEL_MAP.get(name)
        if key is None:
            skipped.append(name)
            continue
        if key not in seen:
            seen.add(key)
            out.append(key)
    if skipped:
        print(f"  [feature-optuna] skip (미지원): {skipped}")
    return ",".join(out) if out else None


def _rerun_feature_optuna(scope: str, strategy: str, n_trials: int,
                           force: bool = False,
                           models_filter: str | None = None) -> int:
    """Optuna feature-selection subprocess re-run with category filter (G-170 v2, D-4).

    Stale JSON cache 검출 시 `simulation/tools/run_optuna_feature_selection.py`
    subprocess 호출. G-170 v2 (2026-05-03): models_filter 가 명시되고 매핑 결과
    None (ts/mech/foundation/ensemble — feature_indices 사용 X 카테고리) 이면
    feature-optuna pre-stage 자체 SKIP (return 0) — bottleneck 영구 차단.

    Args:
        scope: feature optuna scope ("quick" / "representative" / "individual").
        strategy: search strategy ("feature_only" / "joint" / "hp_then_feature" / "all").
        n_trials: per-strategy Optuna trial 수 (default scenario 50).
        force: True 시 stale check 무관 무조건 re-run.
        models_filter: 콤마구분 사용자 친화 모델명 (`args.models` 그대로).
                       None / "" → "all" fallback (legacy / full scenario).
                       매핑 결과 valid 키 있음 → 그 키만 subprocess 전달.
                       **매핑 결과 None (G-170 v2) → feature-optuna pre-stage 자체 SKIP (return 0)**.

    Returns:
        int — subprocess returncode (0 = 성공). feature-optuna pre-stage skip 시도 0 반환.

    Raises:
        절대 raise X — subprocess error / script 부재 / 매핑 실패 모두 graceful skip
        (return 1 또는 0). 학습 자체는 stale JSON fall-through 로 계속.

    Performance:
        - models_filter 매핑 결과 None (skip): 즉시 0 반환 (~10ms)
        - models_filter 매핑 valid (예: tree 4 모델): subprocess 5-30분
        - models_filter None + "all" fallback: 70 모델 1-3시간 (legacy 동작)

    Side effects:
        - subprocess.run (`run_optuna_feature_selection.py`) 호출
        - stdout: stale-check reasons + filter 결과 + skip/proceed 메시지
        - JSON files 생성 (`optuna_feat_sel_<key>.json`)

    Caller responsibility:
        - models_filter = `getattr(args, "models", None)` 권장 (G-170 v2 카테고리 필터).
        - scenario 의 `rerun_feature_optuna: True` 시만 호출.

    Example:
        # tree 카테고리 학습 (G-170 v1+v2 모두 적용)
        >>> _rerun_feature_optuna(scope="representative", strategy="all",
        ...                        n_trials=50, models_filter="XGBoost,LightGBM")
        # → run_optuna_feature_selection.py --model xgboost,lightgbm 실행

        # ts 카테고리 학습 (G-170 v2 — feature-optuna pre-stage skip)
        >>> _rerun_feature_optuna(..., models_filter="ARIMA,SARIMA,SARIMAX")
        # → "G-170 v2: ARIMA,SARIMA,SARIMAX = feature optuna 무관 → SKIP"
        # → return 0 (즉시 종료)

        # legacy (full scenario, models_filter 명시 X)
        >>> _rerun_feature_optuna(scope="representative", strategy="all", n_trials=50)
        # → run_optuna_feature_selection.py --model all 실행 (70 모델, 1-3h)

    See: G-170 v1 (`--model all` hardcoded bottleneck fix),
         G-170 v2 (ts/mech/foundation/ensemble feature-optuna pre-stage skip),
         `_map_models_to_optuna_keys` (매핑 helper).
    """
    import os as _os_b4
    # B4 (2026-06-01): frozen pre-stage feature-optuna 생성기 default gate-off.
    # B2 가 runner 의 stage2_feature_optuna LOAD 폐기(phase6 = STABILITY) → 이 생성기 산출물
    # (stage2_feature_optuna/*.json)은 기본 경로서 더 이상 소비 안 됨 (phase13 자체 stability,
    # phase14 미사용). default = SKIP(즉시 0). external 모드(phase5_external 가 JSON 의존)는
    # MPH_LEGACY_PERMODEL_FEATURES=1 로 opt-in. 코드 full 제거는 재학습 검증(B7) 후.
    if not _os_b4.environ.get("MPH_LEGACY_PERMODEL_FEATURES"):
        print("  [feature-optuna] DEPRECATED (B2/B4 2026-06-01): frozen pre-stage 폐기 → "
              "phase13 STABILITY 사용 → SKIP. (external 모드는 MPH_LEGACY_PERMODEL_FEATURES=1)")
        return 0
    import json
    import subprocess
    import time
    from pathlib import Path
    from simulation.utils.paths import get_optuna_dir

    # Cross-platform storage root: set MPH_OUTPUT_ROOT=/path/to/disk to
    # redirect to another drive when the main one is low on space. Falls
    # back to project-local `simulation/results/` otherwise.
    save_dir = get_optuna_dir()
    db_path = Path("simulation/data/db/epi_real_seoul.db")
    script_path = Path("simulation/tools/run_optuna_feature_selection.py")

    # Probe the known key models so all required JSONs exist
    probe_keys = [
        "lightgbm", "xgboost", "elasticnet", "dnn", "tabular_dnn",
        "negbinglm", "randomforest", "gradientboosting", "svr_rbf",
    ]
    need_rerun = force
    reasons: list[str] = []

    if not need_rerun:
        try:
            db_mtime = db_path.stat().st_mtime if db_path.exists() else 0
        except Exception:
            db_mtime = 0
        feature_tokens = ("rt_pm", "rt_sdot_", "rt_air_")

        for key in probe_keys:
            jp = save_dir / f"optuna_feat_sel_{key}.json"
            if not jp.exists():
                reasons.append(f"missing {jp.name}")
                need_rerun = True
                break
            if jp.stat().st_mtime < db_mtime - 3600:  # 1h grace
                reasons.append(f"{jp.name} older than DB")
                need_rerun = True
                break
            try:
                data = json.load(open(jp, encoding="utf-8"))
                sel = set(data.get("selected_features", []))
                # If none of the new feature families appear, treat as stale
                # (they were wired in 2026-04-20)
                if not any(any(f.startswith(t) for t in feature_tokens) for f in sel):
                    reasons.append(
                        f"{jp.name} lacks new feature families "
                        f"({'/'.join(feature_tokens)})"
                    )
                    need_rerun = True
                    break
            except Exception as e:
                reasons.append(f"{jp.name} unreadable: {e}")
                need_rerun = True
                break

    print("=" * 70)
    print("[rerun-feature-optuna]")
    for r in reasons or ["forced by --force / scenario"]:
        print(f"  stale-check: {r}")

    if not need_rerun:
        print("  → JSONs are fresh and mention new feature families; skipping rerun")
        return 0

    if not script_path.exists():
        print(f"  [warn] script not found: {script_path} — skipping")
        return 1

    # G-170 v2 (2026-05-03): ts/mech/foundation/ensemble = feature optuna 무관 (코드 grep 검증).
    # 이전 v1 (k1): _map None → "all" fallback → 70 모델 stuck (1-2h+, 12.96GB MEM).
    # v2 fix: models_filter 명시 + 매핑 결과 None = 진짜 skip (feature-optuna pre-stage 자체 우회).
    # 매핑 dict 에 등록된 카테고리 (tree/linear/kernel/other/dl-tabular/dl-seq/modern-ts/anchor/graph)
    # 만 feature-optuna pre-stage 진행. ts/mech/foundation/ensemble = skip (statsmodels / mechanistic / pretrained
    # / meta-ensemble 은 feature_indices 사용 X).
    _model_arg = _map_models_to_optuna_keys(models_filter)
    if models_filter and _model_arg is None:
        # ts/mech/foundation/ensemble — feature-optuna pre-stage 자체 skip (G-170 v2 핵심)
        print(f"  [feature-optuna] G-170 v2: '{models_filter}' = feature optuna 무관 카테고리 → SKIP")
        print(f"  → feature-optuna pre-stage 우회, 즉시 R1+ (data 이후) 진입")
        return 0
    if not models_filter and _model_arg is None:
        # caller 가 models_filter 명시 X (legacy / full scenario) → 기존 'all' 동작
        _model_arg = "all"
    print(f"  → invoking {script_path.name} "
          f"(scope={scope}, strategy={strategy}, n_trials={n_trials})")
    print(f"  → output JSONs will be written under: {save_dir}")
    if models_filter:
        print(f"  [feature-optuna] 카테고리 필터: '{models_filter}' → '{_model_arg}'")
    cmd = [
        sys.executable, str(script_path),
        "--scope", scope,
        "--strategy", strategy,
        "--n-trials", str(n_trials),
        "--model", _model_arg,
    ]
    t0 = time.time()
    rc = subprocess.run(cmd).returncode
    elapsed = time.time() - t0
    print(f"  → exit code {rc} after {elapsed/60:.1f} min")
    print("=" * 70)
    return rc


def cmd_train(args):
    # --- Utility commands ---
    if getattr(args, "list_models", False):
        total = 0
        for cat, models in ALL_MODELS.items():
            gpu = " [GPU]" if cat == "DL" else ""
            print(f"\n  [{cat}]{gpu} ({len(models)})")
            for m in models:
                print(f"    {m}")
            total += len(models)
        print(f"\n  Total: {total} models")
        return

    if getattr(args, "list_scenarios", False):
        print("\nAvailable Scenarios:")
        print("=" * 60)
        for name, info in SCENARIOS.items():
            print(f"  {name:22s}  {info['desc']}")
        return

    if getattr(args, "export_config", None):
        from simulation.pipeline.config import PipelineConfig
        config = PipelineConfig()
        config.save_yaml(args.export_config)
        print(f"Config exported: {args.export_config}")
        return

    # 2026-05-31 (사용자 명시, 학위논문 제출용): OOF WF-CV fold 수 = --oof-folds (기본 5, paper-grade).
    # GLOBAL 은 frozen 싱글톤(import 시 1회) — CLI flag 가 명시 override 하는 단일 진입점이므로
    #   ① object.__setattr__ 로 in-process 읽기(_inline:493 등) 갱신,
    #   ② os.environ 으로 OPTUNA_ISOLATE subprocess 자식(config_global 재import)도 동일 fold 수 보장.
    # 범위 [2,10] guard 는 _env_int("MPH_OOF_FOLDS", 5, lo=2, hi=10) 와 동일 의미 (D-2 일관성).
    _oof_folds = getattr(args, "oof_folds", None)
    if _oof_folds is not None:
        import os as _os_of
        from simulation.config_global import GLOBAL as _G_of
        _n = int(_oof_folds)
        if not (2 <= _n <= 10):
            log.warning(f"--oof-folds={_n} 범위[2,10] 밖 → 5 (paper-grade default)")
            _n = 5
        object.__setattr__(_G_of.training, "oof_folds", _n)
        _os_of.environ["MPH_OOF_FOLDS"] = str(_n)

    # --- Auto-collect (optional pre-flight DB refresh) ---
    if getattr(args, "auto_collect", False):
        import time as _time
        from pathlib import Path as _Path
        from simulation.database.config import DB_PATH as _DB_PATH
        db_path = _Path(_DB_PATH)
        stale_days = float(getattr(args, "stale_days", 7.0))
        stale_sec = stale_days * 86400
        if not db_path.exists():
            age_sec = float("inf")
            age_str = "missing"
        else:
            age_sec = _time.time() - db_path.stat().st_mtime
            age_str = f"{age_sec/86400:.1f}d"
        if age_sec > stale_sec:
            print(f"[auto-collect] DB age={age_str} > {stale_days}d threshold "
                  f"-- running collect --groups {args.collect_groups}")
            ns = argparse.Namespace(
                groups=args.collect_groups,
                list_groups=False,
                force=False,
                backfill_days=None,
            )
            try:
                cmd_collect(ns)
            except Exception as e:
                log.warning(f"[auto-collect] failed: {e} -- continuing with existing DB")
            # Invalidate FE cache so phase1 rebuilds against fresh DB
            args.no_cache = True
        else:
            print(f"[auto-collect] DB age={age_str} <= {stale_days}d -- skipping collect")

    # --- Apply scenario preset ---
    if getattr(args, "scenario", None):
        scn = SCENARIOS[args.scenario]
        # P0-1: target-transform preset inheritance (explicit --preset wins)
        if getattr(args, "preset", None) is None and "preset" in scn:
            args.preset = scn["preset"]
        if args.optuna_mode is None and "optuna_mode" in scn:
            args.optuna_mode = scn["optuna_mode"]
        if args.optuna_trials is None and "optuna_trials" in scn:
            args.optuna_trials = scn["optuna_trials"]
        # Stage-3: Optuna search topology selector
        if (getattr(args, "optuna_strategy", None) is None
                and "optuna_strategy" in scn):
            args.optuna_strategy = scn["optuna_strategy"]
        if args.epochs is None and "epochs" in scn:
            args.epochs = scn["epochs"]
        # Stage 3 full_light: inline_epochs / early_stopping_patience are
        # independent of --epochs; only inherit from scenario if not set.
        if getattr(args, "inline_epochs", None) is None and "inline_epochs" in scn:
            args.inline_epochs = scn["inline_epochs"]
        if (getattr(args, "early_stopping_patience", None) is None
                and "early_stopping_patience" in scn):
            args.early_stopping_patience = scn["early_stopping_patience"]
        if args.resume_from is None and "resume_from" in scn:
            args.resume_from = scn["resume_from"]
        if scn.get("lite"):
            args.lite = True
        if args.models is None and "models" in scn:
            args.models = ",".join(scn["models"])
        print(f"  Scenario: {args.scenario} -- {scn['desc']}")

        # P0-H: auto-rerun Optuna feature-selection when the scenario
        # opts in (full / full_light / aggressive). Keeps feature caches aligned
        # with the current feature matrix (e.g. rt_pm*, rt_sdot_* added 2026-04-20).
        # `--skip-feature-optuna` bypasses this (safety valve for GAM timeout etc.).
        #
        # 2026-05-28 (사용자 명시 design A): 본 호출 = **feature-optuna pre-stage actual entry**
        # (cmd_train auto-path). cmd_train 가 R9(per_model_optimize) 보다 먼저 호출 → R9 가
        # 결과 file load (stage2_feature_optuna/<model>.json). feature-optuna pre-stage 명명.
        # 관련 module: simulation.pipeline._inline_optuna_3stage._stage2_feature_optuna_inline
        # (사용자 명시 design A — R9(per_model_optimize) research mode 호출 시 사용).
        if (scn.get("rerun_feature_optuna")
                and not getattr(args, "dry_run", False)
                and not getattr(args, "skip_feature_optuna", False)):
            # G-170 (2026-05-03): args.models (--models XGBoost,LightGBM,...) 전달
            # → run_optuna_feature_selection.py 가 카테고리 모델만 feature search.
            # 이전: hardcoded "all" → tree 학습 시도 1h+ feature-optuna pre-stage stuck (bottleneck 사건).
            # 2026-05-28: scope default = "individual" (53 model 각각, 사용자 명시)
            _rerun_feature_optuna(
                scope=scn.get("feature_optuna_scope", "individual"),  # 2026-05-28: 사용자 명시 default
                strategy=scn.get("feature_optuna_strategy", "all"),
                n_trials=int(scn.get("feature_optuna_trials", 20)),    # 2026-05-28: 사용자 명시 budget
                force=bool(getattr(args, "force", False)),
                models_filter=getattr(args, "models", None),  # G-170: 카테고리 필터 전달
            )
        elif (scn.get("rerun_feature_optuna")
                and getattr(args, "skip_feature_optuna", False)):
            print("  [rerun-feature-optuna] SKIPPED by --skip-feature-optuna")

    # --- Build pipeline args ---
    from simulation.pipeline.runner import build_cli_parser, run_dry_run, run_pipeline
    from simulation.pipeline.config import PipelineConfig

    train_argv = []
    if args.config:
        train_argv += ["--config", args.config]
    if args.dry_run:
        train_argv.append("--dry-run")
    # P0-1: forward preset to pipeline CLI (default 'aggressive' in
    # runner.build_cli_parser kept for back-compat if nothing is injected).
    if getattr(args, "preset", None):
        train_argv += ["--preset", args.preset]
    if args.optuna_mode and args.optuna_mode != "none":
        train_argv += ["--optuna-mode", args.optuna_mode]
    if args.optuna_trials:
        train_argv += ["--optuna-trials", str(args.optuna_trials)]
    if getattr(args, "optuna_strategy", None):
        train_argv += ["--optuna-strategy", args.optuna_strategy]
    if args.epochs:
        train_argv += ["--epochs", str(args.epochs)]
    # Stage 3 full_light: forward new Optuna/training fine-grain knobs
    if getattr(args, "inline_epochs", None):
        train_argv += ["--inline-epochs", str(args.inline_epochs)]
    if getattr(args, "early_stopping_patience", None):
        train_argv += ["--early-stopping-patience", str(args.early_stopping_patience)]
    if args.resume_from is not None:
        # args.resume_from is an ordered phase INDEX (top-level parse already ran
        # resolve_resume_from). The inner train parse re-resolves it, so re-pass the canonical
        # R/P LABEL — a stringified index would hit "phase numbers removed". A raw label/name
        # (e.g. from a scenario dict) passes through unchanged.
        from simulation.pipeline import phases as _ph
        _rf = args.resume_from
        if isinstance(_rf, int) and 0 <= _rf < len(_ph.PHASES):
            _rf = _ph.PHASES[_rf][0]
        train_argv += ["--resume-from", str(_rf)]
    if args.lite:
        train_argv.append("--lite")
    if args.force:
        train_argv.append("--force")
    if args.no_cache:
        train_argv.append("--no-cache")
    # HWP §3 4-way split overrides
    if getattr(args, "paper_cutoff_week", None) is not None:
        train_argv += ["--paper-cutoff-week", str(args.paper_cutoff_week)]
    if getattr(args, "in_sample_end", None):
        train_argv += ["--in-sample-end", str(args.in_sample_end)]
    if getattr(args, "no_real_eval", False):
        train_argv.append("--no-real-eval")
    if getattr(args, "weather_mode", None):
        train_argv += ["--weather-mode", str(args.weather_mode)]
    if getattr(args, "covid_inclusion_mode", None):
        train_argv += ["--covid-mode", str(args.covid_inclusion_mode)]
    if getattr(args, "real_conformal_method", None):
        train_argv += ["--conformal-method", str(args.real_conformal_method)]
    if getattr(args, "ensemble_method", None):
        train_argv += ["--ensemble-method", str(args.ensemble_method)]
    if getattr(args, "per_model_optimize", False):
        train_argv.append("--per-model-optimize")
    if getattr(args, "no_comprehensive_eval", False):
        train_argv.append("--no-comprehensive-eval")

    pipeline_parser = build_cli_parser()

    # ─── Sweep mode: parse `--sweep "dim1:v1,v2;dim2:v3,v4"` and run
    # the pipeline once per Cartesian product, aggregating into
    # simulation/results/sweeps/<timestamp>/.
    sweep_spec = getattr(args, "sweep_spec", None)
    if sweep_spec:
        from itertools import product
        from datetime import datetime
        from pathlib import Path

        DIM_TO_FLAG = {
            "covid":     "--covid-mode",
            "weather":   "--weather-mode",
            "conformal": "--conformal-method",
            "ensemble":  "--ensemble-method",
        }
        # Parse "dim1:v1,v2;dim2:v3,v4" → {dim: [v1, v2], ...}
        sweep_dims: dict[str, list[str]] = {}
        for chunk in sweep_spec.split(";"):
            chunk = chunk.strip()
            if not chunk or ":" not in chunk:
                continue
            dim, vals = chunk.split(":", 1)
            dim = dim.strip().lower()
            if dim not in DIM_TO_FLAG:
                print(f"  [sweep] unknown dim {dim!r}; valid: {list(DIM_TO_FLAG)}")
                continue
            sweep_dims[dim] = [v.strip() for v in vals.split(",") if v.strip()]
        if not sweep_dims:
            print(f"  [sweep] no valid dims parsed from {sweep_spec!r}; "
                  f"running single config")
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
            sweep_dir = get_results_dir() / "sweeps" / ts
            sweep_dir.mkdir(parents=True, exist_ok=True)
            combos = list(product(*[sweep_dims[d] for d in sweep_dims]))
            n_combos = len(combos)
            print(f"  [sweep] {n_combos} configurations: "
                  f"{ {d: vs for d, vs in sweep_dims.items()} }")
            print(f"  [sweep] aggregating into {sweep_dir}")

            run_outcomes = []
            for i, combo in enumerate(combos, 1):
                # Build per-config train_argv
                per_argv = list(train_argv)
                cfg_label_parts = []
                for dim, val in zip(sweep_dims.keys(), combo):
                    flag = DIM_TO_FLAG[dim]
                    # If the flag was already in train_argv, replace it
                    if flag in per_argv:
                        idx = per_argv.index(flag)
                        per_argv[idx + 1] = val
                    else:
                        per_argv += [flag, val]
                    cfg_label_parts.append(f"{dim}={val}")
                cfg_label = ",".join(cfg_label_parts)
                print(f"\n  [sweep {i}/{n_combos}] {cfg_label}")
                pa = pipeline_parser.parse_args(per_argv)
                if pa.config:
                    cfg = PipelineConfig.from_yaml(pa.config)
                else:
                    cfg = PipelineConfig.from_cli(pa)
                # scenario conformal_holdout override
                if getattr(args, "scenario", None):
                    _scn = SCENARIOS.get(args.scenario, {})
                    if "conformal_holdout_weeks" in _scn:
                        cfg.split.conformal_holdout_weeks = int(
                            _scn["conformal_holdout_weeks"])
                if args.models:
                    cfg._selected_models = [m.strip() for m in args.models.split(",")]
                try:
                    if pa.dry_run:
                        run_dry_run(cfg)
                        outcome = "dry_run_ok"
                    else:
                        run_pipeline(cfg)
                        outcome = "ok"
                except Exception as e:
                    log.error(f"  [sweep {i}/{n_combos}] FAILED: {e}", exc_info=True)
                    outcome = f"FAIL: {type(e).__name__}: {str(e)[:120]}"
                run_outcomes.append({"label": cfg_label, "outcome": outcome})

            # Aggregate INDEX.csv
            try:
                from simulation.utils.eval_logger import build_run_index
                idx_path = build_run_index()
                print(f"\n  [sweep] all-runs INDEX rolled up: {idx_path}")
            except Exception as e:
                log.warning(f"  [sweep] INDEX roll-up failed: {e}")
            # Persist sweep manifest
            import json as _json
            manifest = {
                "timestamp": ts,
                "sweep_dims": sweep_dims,
                "n_combos": n_combos,
                "outcomes": run_outcomes,
                "scenario": getattr(args, "scenario", None),
            }
            (sweep_dir / "manifest.json").write_text(
                _json.dumps(manifest, indent=2, default=str)
            )
            print(f"  [sweep] manifest: {sweep_dir / 'manifest.json'}")
            return  # sweep complete; skip the single-run path below

    pipeline_args = pipeline_parser.parse_args(train_argv)

    if pipeline_args.config:
        config = PipelineConfig.from_yaml(pipeline_args.config)
    else:
        config = PipelineConfig.from_cli(pipeline_args)

    # scenario 가 conformal_holdout_weeks 를 명시했으면 직접 주입.
    # CLI 와 from_cli 를 우회 — SCENARIOS dict 가 source of truth.
    # 필요 이유: config.py default 가 0 으로 바뀌어 PI baseline (PICP=80.77%) 재현 불가.
    if getattr(args, "scenario", None):
        _scn = SCENARIOS.get(args.scenario, {})
        if "conformal_holdout_weeks" in _scn:
            config.split.conformal_holdout_weeks = int(_scn["conformal_holdout_weeks"])
            print(f"  [scenario] conformal_holdout_weeks = "
                  f"{config.split.conformal_holdout_weeks} (overridden from default 0)")

    # Inject model filter
    if args.models:
        selected = [m.strip() for m in args.models.split(",")]
        config._selected_models = selected
        print(f"  Model filter: {selected}")

    if pipeline_args.dry_run:
        run_dry_run(config)
    else:
        run_pipeline(config)


def cmd_train_all(args):
    """Sweep every scenario. Each scenario is a fully isolated subprocess so
    that Optuna studies, global seeds, and torch CUDA state can't leak across
    runs. By default runs with --force --no-cache so every scenario starts
    from a clean slate (fresh Optuna, no FE parquet reuse, no model pickles).
    """
    import subprocess
    import time

    # Scenarios that are not standalone full runs
    SKIP_BY_DEFAULT = {"diagnostics-only", "wfcv-only", "quick-test"}

    if args.scenarios:
        wanted = [s.strip() for s in args.scenarios.split(",") if s.strip()]
        unknown = [s for s in wanted if s not in SCENARIOS]
        if unknown:
            print(f"[error] Unknown scenarios: {unknown}")
            print(f"        Available: {list(SCENARIOS.keys())}")
            sys.exit(2)
    else:
        wanted = [s for s in SCENARIOS.keys() if s not in SKIP_BY_DEFAULT]

    if args.skip:
        skip_set = {s.strip() for s in args.skip.split(",") if s.strip()}
        wanted = [s for s in wanted if s not in skip_set]

    if not wanted:
        print("[error] Nothing to run after filtering.")
        sys.exit(2)

    # --- Resume state ---
    if getattr(args, "restart", False):
        _clear_state("train_all")
        print("  [restart] previous resume state cleared")
    state = _load_state("train_all")
    already_ok = {s for s, v in state.items()
                  if isinstance(v, dict) and v.get("status") == "OK"}
    pending = [s for s in wanted if s not in already_ok]
    skipped_resume = [s for s in wanted if s in already_ok]

    print("=" * 70)
    print(f"  train-all sweep  --  {len(wanted)} scenario(s)")
    print("=" * 70)
    if skipped_resume:
        print(f"  [resume] skipping {len(skipped_resume)} already-OK: {skipped_resume}")
    if not pending:
        print("  [resume] nothing to do -- all scenarios already completed.")
        print("          use --restart to force a fresh run.")
        return
    for i, s in enumerate(pending, 1):
        print(f"    {i:>2}. {s:22s}  {SCENARIOS[s]['desc']}")
    force_fresh = not args.no_force
    print(f"  Force fresh: {force_fresh}  (Optuna JSON caches + FE parquet cleared per run)")
    print(f"  Continue on error: {args.continue_on_error}")
    print()

    if args.dry_run:
        print("  [dry-run] not executing.")
        return

    continue_on_error = getattr(args, "continue_on_error", False)
    results = []  # (scenario, status, elapsed, returncode)
    aborted = False
    t_total = time.time()
    # NOTE: sys.executable is the interpreter that launched this process,
    # which is exactly `.venv\Scripts\python.exe` when invoked per project
    # convention (`.venv\Scripts\python.exe -m simulation train-all`). Using
    # sys.executable is therefore the correct way to guarantee children run
    # in the same uv venv.
    for i, scn in enumerate(pending, 1):
        print("-" * 70)
        print(f"  [{i}/{len(pending)}] START  {scn}")
        print("-" * 70)
        cmd = [sys.executable, "-m", "simulation", "train", "--scenario", scn]
        if force_fresh:
            cmd += ["--force", "--no-cache"]
        t0 = time.time()
        proc = None
        try:
            # Use Popen so Ctrl+C can terminate the child deterministically.
            proc = subprocess.Popen(cmd)
            rc = proc.wait()
        except KeyboardInterrupt:
            print(f"\n[abort] Ctrl+C received during {scn} -- terminating child")
            if proc is not None:
                try:
                    proc.terminate()
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                except Exception as e:
                    log_warn = getattr(sys, "stderr", None)
                    if log_warn:
                        print(f"  terminate error: {e}", file=log_warn)
            rc = 130
            aborted = True
        elapsed = time.time() - t0
        status = "OK" if rc == 0 else f"FAIL(rc={rc})"
        results.append((scn, status, elapsed, rc))
        # Persist resume state after every scenario (crash-safe)
        state[scn] = {
            "status": status, "rc": rc, "elapsed": round(elapsed, 1),
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _save_state("train_all", state)
        print(f"  [{i}/{len(pending)}] DONE   {scn}  --  {status}  ({elapsed:.0f}s)")
        if aborted:
            break
        if rc != 0 and not continue_on_error:
            print(f"\n[abort] {scn} failed and --continue-on-error not set.")
            break

    total = time.time() - t_total
    print("\n" + "=" * 70)
    print(f"  Sweep summary  ({total:.0f}s total)")
    print("=" * 70)
    n_ok = sum(1 for _, s, _, _ in results if s == "OK")
    for scn, status, elapsed, _ in results:
        print(f"    {scn:22s}  {status:12s}  {elapsed:>7.0f}s")
    print(f"\n  {n_ok}/{len(results)} scenarios succeeded.")

    # --- S2-4: returncode aggregation report --------------------
    # When --continue-on-error is set the failures used to be easy to
    # miss once the summary table scrolled past. Make them unmissable:
    #   (1) loud "FAILED SCENARIOS" block on stderr
    #   (2) structured JSON report on disk for CI to pick up
    failed = [(scn, rc, elapsed) for (scn, _, elapsed, rc) in results if rc != 0]
    if failed:
        print("", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        print(f"  FAILED SCENARIOS ({len(failed)}/{len(results)})", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        for scn, rc, elapsed in failed:
            print(f"    {scn:22s}  rc={rc:<5d}  ({elapsed:.0f}s)", file=sys.stderr)
        print("", file=sys.stderr)

    # Always persist the sweep summary (success or failure) for downstream
    # tooling. Path mirrors the resume-state layout.
    try:
        import json as _json
        report_path = _state_path("train_all").with_name("train_all_summary.json")
        summary = {
            "total_elapsed_s": round(total, 1),
            "n_total": len(results),
            "n_ok": n_ok,
            "n_failed": len(failed),
            "continue_on_error": bool(continue_on_error),
            "aborted_by_keyboard_interrupt": bool(aborted),
            "results": [
                {
                    "scenario": scn,
                    "status": status,
                    "rc": rc,
                    "elapsed_s": round(elapsed, 1),
                }
                for (scn, status, elapsed, rc) in results
            ],
            "failed_scenarios": [
                {"scenario": scn, "rc": rc, "elapsed_s": round(elapsed, 1)}
                for (scn, rc, elapsed) in failed
            ],
        }
        report_path.write_text(
            _json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\n  [report] sweep summary written → {report_path}")
    except Exception as _e:
        print(f"  [warn] failed to write sweep summary: {_e}", file=sys.stderr)

    # Non-zero exit if any failed
    if any(rc != 0 for _, _, _, rc in results):
        sys.exit(1)


def cmd_run_all(args):
    """End-to-end lifecycle: bootstrap -> collect -> db-optimize -> train-all.

    Each stage is invoked through the same argparse dispatch that a user would
    type manually. Stages are independently skippable so the command works for
    cold-start builds, training-only reruns, or anything in between.

    Returns non-zero if any non-skipped stage fails (unless --continue-on-error
    is set, which only applies to the training sweep inside train-all).
    """
    import time

    stages = []  # (name, enabled, callable_thunk)

    # ---- stage 1: bootstrap ----
    def _run_bootstrap():
        ns = argparse.Namespace(
            skip_pdf=False, skip_maintain=False, vacuum=False,
        )
        return cmd_bootstrap(ns)

    # ---- stage 2: collect ----
    def _run_collect():
        ns = argparse.Namespace(
            groups=args.collect_groups,
            list_groups=False,
            force=False,
            backfill_days=getattr(args, "backfill_days", None),
        )
        return cmd_collect(ns)

    # ---- stage 3: db-optimize ----
    def _run_optimize():
        ns = argparse.Namespace(vacuum=bool(args.vacuum))
        return cmd_db_optimize(ns)

    # ---- stage 4: train-all ----
    def _run_train_all():
        ns = argparse.Namespace(
            scenarios=args.scenarios,
            skip=args.skip_scenarios,
            no_force=bool(args.no_force),
            continue_on_error=bool(args.continue_on_error),
            dry_run=False,
        )
        return cmd_train_all(ns)

    stages.append(("bootstrap",   not args.skip_bootstrap, _run_bootstrap))
    stages.append(("collect",     not args.skip_collect,   _run_collect))
    stages.append(("db-optimize", not args.skip_optimize,  _run_optimize))
    stages.append(("train-all",   not args.skip_train,     _run_train_all))

    # --- Resume state (stage-level) ---
    # train-all stage is always replayed because it has its own scenario-level
    # resume; the other three stages are cheap enough to rerun but we still
    # persist state so user can see what's been done.
    if getattr(args, "restart", False):
        _clear_state("run_all")
        _clear_state("train_all")
        print("  [restart] previous resume state cleared")
    state = _load_state("run_all")
    resumable_stages = {"bootstrap", "collect", "db-optimize"}
    resumed = {k for k, v in state.items()
               if isinstance(v, dict) and v.get("status", "").startswith("OK")
               and k in resumable_stages}

    print("=" * 70)
    print("  run-all  --  full lifecycle")
    print("=" * 70)
    for name, enabled, _ in stages:
        if enabled and name in resumed:
            mark = "DONE"
        elif enabled:
            mark = "RUN "
        else:
            mark = "SKIP"
        print(f"    [{mark}] {name}")
    if args.dry_run:
        print("\n  [dry-run] nothing executed.")
        return

    print()
    t_total = time.time()
    results = []  # (name, status, elapsed)
    for name, enabled, thunk in stages:
        if not enabled:
            results.append((name, "SKIPPED", 0.0))
            continue
        if name in resumed:
            prev = state.get(name, {})
            print(f"  >>> STAGE: {name} -- RESUMED (previously OK at "
                  f"{prev.get('completed_at', '?')})")
            results.append((name, "OK(resumed)", 0.0))
            continue
        print("-" * 70)
        print(f"  >>> STAGE: {name}")
        print("-" * 70)
        t0 = time.time()
        status = "OK"
        try:
            rc = thunk()
            if isinstance(rc, int) and rc != 0:
                status = f"FAIL(rc={rc})"
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
            status = "OK" if code == 0 else f"FAIL(rc={code})"
        except KeyboardInterrupt:
            print(f"\n[abort] Ctrl+C during {name}")
            results.append((name, "ABORT", time.time() - t0))
            break
        except Exception as e:
            status = f"ERROR({type(e).__name__}: {e})"
        elapsed = time.time() - t0
        results.append((name, status, elapsed))
        # Persist stage state after each stage (crash-safe)
        state[name] = {
            "status": status, "elapsed": round(elapsed, 1),
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _save_state("run_all", state)
        print(f"  <<< {name} -- {status} ({elapsed:.0f}s)")
        if not status.startswith("OK"):
            # Only train-all knows --continue-on-error; the upstream ETL stages
            # are hard dependencies for training, so failing there always stops.
            print(f"\n[abort] {name} did not succeed; stopping run-all.")
            break

    total = time.time() - t_total
    print("\n" + "=" * 70)
    print(f"  run-all summary  ({total:.0f}s total)")
    print("=" * 70)
    for name, status, elapsed in results:
        print(f"    {name:14s}  {status:20s}  {elapsed:>7.0f}s")
    if any(not s.startswith("OK") and s != "SKIPPED" for _, s, _ in results):
        sys.exit(1)




__all__ = [
    "cmd_collect",
    "cmd_train",
    "cmd_train_all",
    "cmd_run_all",
]

