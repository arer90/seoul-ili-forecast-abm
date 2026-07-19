"""학습 progress 모니터 CLI (Tier 1 #5).

매 점검 시 사용했던 명령들을 통합. 한 번 실행으로:
  - 학습 PID 상태 (살아있나, CPU/MEM)
  - pipeline 진행 (pre-R9 / R9 per_model_optimize / 'OK [N/M]')
  - α-blend 분포 (G-141 collapse 검사)
  - Test R²/RMSE 분포
  - Optuna trials + pruning rate
  - Best WIS 추세

사용:
    .venv/bin/python -m simulation.scripts.progress
    .venv/bin/python -m simulation.scripts.progress --watch    # 30s 주기
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

# Windows 대응: "/tmp" 리터럴 대신 프로젝트 SSOT 임시 디렉터리.
from simulation.config_global import GLOBAL as _G
_TMP = _G.paths.fast_tmp

ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = ROOT / "simulation/logs"


def find_training_pid() -> int | None:
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "simulation train|run_optuna_feature_selection"],
            text=True,
        ).strip()
        return int(out.split("\n")[0]) if out else None
    except subprocess.CalledProcessError:
        return None


def ps_info(pid: int) -> dict:
    try:
        out = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "etime=,stat=,%cpu=,%mem=,rss="],
            text=True,
        ).strip()
        parts = out.split(None, 4)
        return {
            "etime": parts[0],
            "stat": parts[1],
            "cpu": float(parts[2]),
            "mem": float(parts[3]),
            "rss_mb": int(parts[4]) // 1024,
        }
    except (subprocess.CalledProcessError, ValueError, IndexError):
        return {}


def find_latest_log() -> Path | None:
    candidates: list[Path] = []
    candidates += list(_TMP.glob("training_resume_*.log"))
    candidates += list(LOGS_DIR.glob("train_*.log"))
    candidates = [p for p in candidates if p.exists() and p.stat().st_size > 1024]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def parse_log(log_path: Path) -> dict:
    if not log_path or not log_path.exists():
        return {}
    txt = log_path.read_text(encoding="utf-8", errors="replace")
    if len(txt) > 1_000_000:
        txt = txt[-1_000_000:]

    ok_pattern = re.compile(r"OK \[(\d+)/(\d+)\] (\S+)\s+Test R2=\s*(-?\d+\.\d+)\s+RMSE=\s*(\d+\.\d+)")
    ok_matches = ok_pattern.findall(txt)

    # 옛 학습 로그의 persisted 출력 형식 "post-anchor: α=..." 파싱 — 리터럴 변경 불가
    alpha_pattern = re.compile(r"post-anchor: α=([-+]?\d+\.\d+)")
    alphas = [float(m) for m in alpha_pattern.findall(txt)]

    best_pattern = re.compile(r"best=\+?(-?\d+\.\d+)")
    bests = [float(m) for m in best_pattern.findall(txt)]

    progress = re.findall(
        r"▶ ([a-z]+) ([a-z_]+) \[P(\d+):([^\]]+)\] \| (\d+)/(\d+) \((\d+)%\) \| 경과 ([^|]+)\| 남은 (\S+)",
        txt,
    )

    return {
        "log_path": str(log_path),
        "log_size_mb": log_path.stat().st_size // 1024 // 1024,
        "log_modified": time.strftime("%H:%M:%S", time.localtime(log_path.stat().st_mtime)),
        "ok_count": len(ok_matches),
        "ok_recent": ok_matches[-5:] if ok_matches else [],
        "alpha_count": len(alphas),
        "alpha_recent": alphas[-10:] if alphas else [],
        "best_recent": bests[-10:] if bests else [],
        "progress_recent": progress[-3:] if progress else [],
    }


def optuna_stats() -> list[dict]:
    out = []
    from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
    for db in sorted(glob.glob(str(get_results_dir() / "optuna_*.db"))):
        try:
            from simulation.database import safe_connect  # G-116 (2026-05-29)
            conn = safe_connect(db)
            n = conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0]
            states = dict(conn.execute(
                "SELECT state, COUNT(*) FROM trials GROUP BY state"
            ).fetchall())
            best = conn.execute(
                "SELECT MIN(value) FROM trial_values WHERE value IS NOT NULL AND value < 1e10"
            ).fetchone()[0]
            size_mb = os.path.getsize(db) // 1024 // 1024
            prune_rate = (states.get("PRUNED", 0) /
                          max(states.get("PRUNED", 0) + states.get("COMPLETE", 0), 1) * 100)
            out.append({
                "db": Path(db).name,
                "trials": n,
                "states": states,
                "size_mb": size_mb,
                "best": best,
                "prune_rate": prune_rate,
            })
        except Exception as e:
            out.append({"db": Path(db).name, "error": str(e)})
    return out


def collapse_check(ok_matches: list) -> dict:
    if not ok_matches:
        return {"checked": False}
    metric_pairs = [(round(float(r2), 4), round(float(rmse), 4))
                    for _, _, _, r2, rmse in ok_matches]
    counts = Counter(metric_pairs)
    suspect = [(pair, cnt) for pair, cnt in counts.most_common() if cnt >= 3]
    return {
        "checked": True,
        "n_total": len(ok_matches),
        "n_unique": len(set(metric_pairs)),
        "suspect_collapses": suspect[:5],
    }


def render(state: dict) -> str:
    lines = []
    lines.append("═" * 70)
    lines.append(f"  학습 progress 보고 — {time.strftime('%H:%M:%S')}")
    lines.append("═" * 70)

    if state.get("worker_pid"):
        info = state.get("worker_info", {})
        lines.append(f"\n▸ Worker PID {state['worker_pid']}")
        if info:
            lines.append(f"    elapsed: {info['etime']}, CPU {info['cpu']}%, "
                         f"MEM {info['rss_mb']} MB ({info['mem']}%)")
    else:
        lines.append("\n▸ Worker PID: 없음 (학습 종료 또는 시작 전)")

    log = state.get("log", {})
    if log.get("progress_recent"):
        lines.append(f"\n▸ 진행 단계")
        for p in log["progress_recent"]:
            lines.append(f"    {p[0]}/{p[1]} [P{p[2]}:{p[3]}] {p[4]}/{p[5]} ({p[6]}%) "
                         f"경과 {p[7].strip()} 남은 {p[8]}")

    lines.append(f"\n▸ R9 per_model_optimize 'OK [N/M]'")
    if log.get("ok_count", 0) > 0:
        lines.append(f"    완료: {log['ok_count']} 모델")
        for m in log["ok_recent"]:
            lines.append(f"      [{m[0]}/{m[1]}] {m[2]:30s} R²={m[3]} RMSE={m[4]}")
    else:
        lines.append("    아직 미진입 (R9 이전 phase 진행 중)")

    if log.get("alpha_count", 0) > 0:
        lines.append(f"\n▸ α-blend 분포 ({log['alpha_count']} 모델)")
        alphas = log["alpha_recent"]
        if alphas:
            unique_alpha = sorted(set(round(a, 2) for a in alphas))
            lines.append(f"    최근 10: {alphas}")
            lines.append(f"    {len(unique_alpha)} 고유값 — "
                         f"min={min(alphas):.3f}, max={max(alphas):.3f}, "
                         f"mean={sum(alphas)/len(alphas):.3f}")
            zero_count = sum(1 for a in alphas if a < 0.05)
            if zero_count > 0:
                lines.append(f"    ⚠ α<0.05: {zero_count}/{len(alphas)} (G-141 의심)")

    cc = collapse_check(log.get("ok_recent", []))
    if cc.get("checked"):
        lines.append(f"\n▸ R²/RMSE collapse 검사 (G-141)")
        lines.append(f"    {cc['n_unique']}/{cc['n_total']} 고유값")
        if cc["suspect_collapses"]:
            for pair, cnt in cc["suspect_collapses"]:
                lines.append(f"    ⚠ {cnt}개 동일: R²={pair[0]}, RMSE={pair[1]}")
        else:
            lines.append(f"    ✓ collapse 없음")

    if log.get("best_recent"):
        lines.append(f"\n▸ Best WIS 추세 (최근 10)")
        bests = log["best_recent"]
        lines.append(f"    {bests}")
        lines.append(f"    min: {min(bests):.3f}, max: {max(bests):.3f}")

    opt = state.get("optuna", [])
    if opt:
        lines.append(f"\n▸ Optuna DB")
        for db in opt:
            if "error" in db:
                lines.append(f"    {db['db']}: {db['error']}")
            else:
                lines.append(f"    {db['db']}: {db['trials']:,} trials, "
                             f"prune {db['prune_rate']:.1f}%, "
                             f"best {db.get('best', 'N/A')}, "
                             f"{db['size_mb']} MB")

    if log.get("log_path"):
        lines.append(f"\n▸ Log: {log['log_path']}")
        lines.append(f"    {log['log_size_mb']} MB, modified {log['log_modified']}")

    lines.append("\n" + "═" * 70)
    return "\n".join(lines)


def collect_state() -> dict:
    worker_pid = find_training_pid()
    state = {"worker_pid": worker_pid}
    if worker_pid:
        state["worker_info"] = ps_info(worker_pid)
    log_path = find_latest_log()
    if log_path:
        state["log"] = parse_log(log_path)
    state["optuna"] = optuna_stats()
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true",
                    help="watch mode (30s 주기 갱신)")
    ap.add_argument("--interval", type=int, default=30)
    args = ap.parse_args()

    if args.watch:
        try:
            while True:
                os.system("clear" if os.name != "nt" else "cls")
                print(render(collect_state()))
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n중단됨")
            return 0

    print(render(collect_state()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
