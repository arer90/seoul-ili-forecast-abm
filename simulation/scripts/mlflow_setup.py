"""MLflow setup (Tier 1 #6).

Optuna study 결과 + R9(per_model_optimize) best HP 를 MLflow 에 자동 logging.

사용:
    # 1. MLflow 서버 시작
    .venv/bin/python -m simulation.scripts.mlflow_setup --serve

    # 2. URL 환경변수 설정 (config_global 에서 자동 사용)
    export MPH_MLFLOW_URI=sqlite:///simulation/results/mlflow.db
    .venv/bin/python -m simulation.scripts.mlflow_setup --import-optuna

    # 3. Web UI
    mlflow ui --backend-store-uri sqlite:///simulation/results/mlflow.db --port 5000
    # → http://localhost:5000

ENGINEERING_PRINCIPLES.md §원칙 #5 (재현성): trial 결과 자동 기록.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
DEFAULT_URI = f"sqlite:///{get_results_dir()}/mlflow.db"
DEFAULT_OPTUNA_DB = get_results_dir() / "optuna_feature_selection.db"


def serve_ui(uri: str, port: int = 5000) -> int:
    """MLflow web UI 시작."""
    print(f"MLflow UI 시작")
    print(f"  Backend: {uri}")
    print(f"  URL:     http://localhost:{port}")
    print()
    try:
        return subprocess.call([
            "mlflow", "ui",
            "--backend-store-uri", uri,
            "--port", str(port),
        ])
    except FileNotFoundError:
        print("✗ mlflow 미설치")
        print("  설치: uv pip install mlflow")
        return 1
    except KeyboardInterrupt:
        return 0


def import_optuna(uri: str, optuna_db: Path) -> int:
    """Optuna study 결과 → MLflow runs.

    각 study 의 best trial = 1 run.
    """
    try:
        import mlflow
    except ImportError:
        print("✗ mlflow 미설치 — uv pip install mlflow")
        return 1

    if not optuna_db.exists():
        print(f"✗ Optuna DB 없음: {optuna_db}")
        return 1

    mlflow.set_tracking_uri(uri)
    print(f"MLflow tracking → {uri}")

    from simulation.database import safe_connect  # G-116/G-117 SSOT
    conn = safe_connect(str(optuna_db))
    studies = conn.execute(
        "SELECT study_id, study_name FROM studies"
    ).fetchall()
    print(f"Optuna studies: {len(studies)}")

    n_imported = 0
    for study_id, study_name in studies:
        # Best trial
        best = conn.execute("""
            SELECT t.trial_id, MIN(tv.value)
            FROM trials t
            JOIN trial_values tv ON tv.trial_id = t.trial_id
            WHERE t.study_id = ? AND tv.value IS NOT NULL AND tv.value < 1e10
        """, (study_id,)).fetchone()
        if not best or best[0] is None:
            continue

        trial_id, value = best

        # Params
        params = dict(conn.execute(
            "SELECT param_name, param_value FROM trial_params WHERE trial_id = ?",
            (trial_id,)
        ).fetchall())

        # Trial count + pruning
        states = dict(conn.execute(
            "SELECT state, COUNT(*) FROM trials WHERE study_id = ? GROUP BY state",
            (study_id,)
        ).fetchall())

        # MLflow run
        try:
            mlflow.set_experiment(study_name)
            with mlflow.start_run(run_name=f"best_trial_{trial_id}"):
                mlflow.log_metric("best_value", float(value))
                mlflow.log_metric("n_trials_total", sum(states.values()))
                mlflow.log_metric("n_trials_complete", states.get("COMPLETE", 0))
                mlflow.log_metric("n_trials_pruned", states.get("PRUNED", 0))
                mlflow.log_param("optuna_study", study_name)
                # use_* 같은 boolean 파라미터 그대로 log
                for k, v in params.items():
                    if len(str(v)) < 250:  # MLflow param 길이 제한
                        try:
                            mlflow.log_param(k, v)
                        except Exception:
                            pass
            n_imported += 1
        except Exception as e:
            print(f"  ⚠ {study_name}: {type(e).__name__}: {e}")

    print(f"✓ {n_imported} studies imported")
    return 0


def main():
    ap = argparse.ArgumentParser()
    from simulation.config_global import GLOBAL  # SSOT (2026-05-28)
    ap.add_argument("--uri", default=(GLOBAL.ops.mlflow_uri or DEFAULT_URI))
    ap.add_argument("--port", type=int, default=5000)
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("serve", help="MLflow UI 시작")
    sub.add_parser("import-optuna", help="Optuna → MLflow")

    # 단일 인자 형태도 지원
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--import-optuna", action="store_true",
                    dest="import_optuna_flag")
    ap.add_argument("--optuna-db", default=str(DEFAULT_OPTUNA_DB))

    args = ap.parse_args()

    if args.serve or args.cmd == "serve":
        return serve_ui(args.uri, args.port)
    if args.import_optuna_flag or args.cmd == "import-optuna":
        return import_optuna(args.uri, Path(args.optuna_db))

    # 기본: 도움말
    print("MLflow setup")
    print()
    print("사용:")
    print(f"  --serve              MLflow UI (port {args.port})")
    print(f"  --import-optuna      Optuna → MLflow runs")
    print()
    print(f"기본 URI: {DEFAULT_URI}")
    print(f"환경변수 override: MPH_MLFLOW_URI")
    return 0


if __name__ == "__main__":
    sys.exit(main())
