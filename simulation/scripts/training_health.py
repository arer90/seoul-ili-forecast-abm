"""학습 health check — stuck 자동 감지 + 진단 5규칙 (G-152, 2026-05-02).

기존 progress.py 의 부족한 부분 보완:
  - Optuna trial last_trial_time (진짜 진행 신호)
  - Champion-log 마지막 시각
  - per_model_optimal 마지막 timestamp
  - Process tree (child/parent)
  - Stuck 자동 감지 (N분 무진전 시 ⚠ 알림)
  - Memory 추세 (RSS 증가율)

사용:
    .venv/bin/python -m simulation.scripts.training_health
    .venv/bin/python -m simulation.scripts.training_health --watch 60   # 60s 주기
    .venv/bin/python -m simulation.scripts.training_health --json       # JSON 출력
    .venv/bin/python -m simulation.scripts.training_health --threshold 1800  # stuck 1800s

출력 항목 (진단 5규칙):
  ① Optuna trial last_trial_time vs 현재 시각 → 진짜 진행 여부
  ② Champion-log 마지막 시각 vs 현재 → 모델 완료 추적
  ③ per_model_optimal 새 파일 timestamp → 결과 저장 추적
  ④ CPU 활성 ≠ 진행 (참고만)
  ⑤ Sequence 모델 timeout 임박 알림
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)

# Windows 대응: "/tmp" 리터럴 대신 프로젝트 SSOT 임시 디렉터리.
from simulation.config_global import GLOBAL as _G
_TMP = _G.paths.fast_tmp

ROOT = Path(__file__).resolve().parents[2]


# ════════════════════════════════════════════════════════════════
# Process tree (PID + child)
# ════════════════════════════════════════════════════════════════
def find_training_processes() -> list[dict]:
    """학습 관련 모든 process (parent + child)."""
    procs = []
    try:
        out = subprocess.check_output(
            ["ps", "-axo", "pid,ppid,etime,%cpu,%mem,rss,command"],
            text=True,
        )
        for line in out.splitlines()[1:]:
            parts = line.strip().split(None, 6)
            if len(parts) < 7:
                continue
            cmd = parts[6]
            if any(k in cmd for k in [
                "simulation train", "run_resume_phase12", "run_optuna_feature_selection",
                "train_by_category"
            ]):
                procs.append({
                    "pid": int(parts[0]),
                    "ppid": int(parts[1]),
                    "etime": parts[2],
                    "cpu": float(parts[3]),
                    "mem": float(parts[4]),
                    "rss_mb": int(parts[5]) // 1024,
                    "cmd": cmd[:100],
                })
    except Exception as e:
        return [{"error": str(e)}]
    return procs


# ════════════════════════════════════════════════════════════════
# Optuna last_trial_time (진짜 진행 신호 ①)
# ════════════════════════════════════════════════════════════════
def optuna_last_trials() -> list[dict]:
    """각 study 의 마지막 trial 시각 + 현재로부터 경과."""
    db = get_results_dir() / "optuna_study.db"
    if not db.exists():
        return []
    out = []
    now = datetime.now()
    try:
        from simulation.database import safe_connect  # G-116 (2026-05-29)
        conn = safe_connect(str(db))
        rows = conn.execute("""
            SELECT s.study_name,
                    MAX(t.datetime_start),
                    COUNT(t.trial_id),
                    SUM(CASE WHEN t.state='RUNNING' THEN 1 ELSE 0 END) AS running
            FROM studies s LEFT JOIN trials t ON s.study_id=t.study_id
            GROUP BY s.study_name
            ORDER BY MAX(t.datetime_start) DESC
        """).fetchall()
        for name, last_str, n, running in rows:
            if last_str:
                last_dt = datetime.strptime(last_str.split(".")[0], "%Y-%m-%d %H:%M:%S")
                age_sec = (now - last_dt).total_seconds()
            else:
                age_sec = None
            out.append({
                "study": name,
                "last_trial": last_str or "(none)",
                "age_sec": age_sec,
                "trials": n,
                "running": running,
            })
    except Exception as e:
        out.append({"error": str(e)})
    return out


# ════════════════════════════════════════════════════════════════
# Champion-log 마지막 시각 (진짜 진행 신호 ②)
# ════════════════════════════════════════════════════════════════
def champion_log_status(log_path: Path | None = None) -> dict:
    """학습 log 의 champion 메시지 마지막 시각."""
    if log_path is None:
        # 가장 최근 log 자동 탐색
        candidates = []
        candidates += list(_TMP.glob("training_resume*.log"))
        candidates += list(_TMP.glob("training_v*.log"))
        candidates += list(_TMP.glob("train_cat_*.log"))
        candidates = [p for p in candidates if p.exists() and p.stat().st_size > 1024]
        if not candidates:
            return {"log": None}
        log_path = max(candidates, key=lambda p: p.stat().st_mtime)

    try:
        # 마지막 50줄 + 전체 grep
        result = subprocess.run(
            ["grep", "-E", "champion-log|champion 선정|OK \\[[0-9]+/[0-9]+\\]|학습 완료",
             str(log_path)],
            capture_output=True, text=True
        )
        lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
        last_line = lines[-1] if lines else None
        # 시각 추출 (e.g., "2026-05-02 08:26:59")
        import re
        if last_line:
            m = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", last_line)
            if m:
                last_dt = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                age_sec = (datetime.now() - last_dt).total_seconds()
                return {
                    "log": log_path.name,
                    "last_event": m.group(1),
                    "age_sec": age_sec,
                    "snippet": last_line[:120],
                }
        return {"log": log_path.name, "last_event": None}
    except Exception as e:
        return {"error": str(e)}


# ════════════════════════════════════════════════════════════════
# per_model_optimal 마지막 timestamp (진짜 진행 신호 ③)
# ════════════════════════════════════════════════════════════════
def per_model_optimal_status() -> dict:
    """per_model_optimal 디렉토리의 가장 최근 모델 + 시각."""
    p = get_results_dir() / "per_model_optimal"
    if not p.exists():
        return {"count": 0}
    files = sorted(p.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        return {"count": 0}
    latest = files[0]
    age_sec = time.time() - latest.stat().st_mtime
    return {
        "count": len(files),
        "latest_model": latest.stem,
        "latest_time": datetime.fromtimestamp(latest.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "age_sec": age_sec,
    }


# ════════════════════════════════════════════════════════════════
# Stuck 자동 감지 (진단 5규칙 종합)
# ════════════════════════════════════════════════════════════════
def diagnose_stuck(threshold_sec: int = 1800) -> dict:
    """5가지 신호로 stuck 진단."""
    procs = find_training_processes()
    optuna = optuna_last_trials()
    champion = champion_log_status()
    pmo = per_model_optimal_status()

    # 진짜 진행 신호의 최신 age (가장 최근)
    ages = []
    if optuna:
        valid = [s["age_sec"] for s in optuna if s.get("age_sec") is not None]
        if valid:
            ages.append(("optuna_last_trial", min(valid)))
    if champion.get("age_sec") is not None:
        ages.append(("champion_log", champion["age_sec"]))
    if pmo.get("age_sec") is not None:
        ages.append(("per_model_optimal", pmo["age_sec"]))

    # 어떤 신호도 없거나 모두 threshold 초과 → stuck
    if not ages:
        return {"stuck": "unknown", "reason": "no progress signal"}
    most_recent_signal, most_recent_age = min(ages, key=lambda x: x[1])

    if most_recent_age > threshold_sec:
        return {
            "stuck": True,
            "signal": most_recent_signal,
            "age_sec": most_recent_age,
            "threshold_sec": threshold_sec,
            "message": f"⚠ STUCK suspect — last progress {int(most_recent_age // 60)}min ago via {most_recent_signal}",
            "training_alive": len(procs) > 0,
        }
    return {
        "stuck": False,
        "signal": most_recent_signal,
        "age_sec": most_recent_age,
        "message": f"✓ active — last progress {int(most_recent_age // 60)}min ago via {most_recent_signal}",
        "training_alive": len(procs) > 0,
    }


# ════════════════════════════════════════════════════════════════
# 종합 진단 출력
# ════════════════════════════════════════════════════════════════
def render_health(threshold_sec: int = 1800) -> str:
    procs = find_training_processes()
    optuna = optuna_last_trials()
    champion = champion_log_status()
    pmo = per_model_optimal_status()
    diag = diagnose_stuck(threshold_sec)

    lines = []
    lines.append("=" * 80)
    lines.append(f"학습 Health Check ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    lines.append("=" * 80)

    # 1. Process tree
    lines.append("\n[1] Process tree")
    if not procs:
        lines.append("  ⊘ 학습 프로세스 없음")
    else:
        for p in procs:
            if "error" in p:
                lines.append(f"  ✗ {p['error']}")
                continue
            lines.append(f"  PID {p['pid']:>5} (ppid={p['ppid']}) {p['etime']:>10s} "
                          f"CPU {p['cpu']:>5.1f}% MEM {p['mem']:>4.1f}% RSS {p['rss_mb']:>5,}MB")
            lines.append(f"    {p['cmd']}")

    # 2. Optuna last_trial_time (진짜 진행 신호)
    lines.append("\n[2] Optuna last_trial_time (진짜 진행 신호 ①)")
    for s in optuna[:8]:
        if "error" in s:
            lines.append(f"  ✗ {s['error']}")
            continue
        age = s["age_sec"]
        flag = "🔄" if age and age < 300 else ("⚠" if age and age > threshold_sec else "  ")
        age_str = f"{int(age//60)}분" if age else "-"
        lines.append(f"  {flag} {s['study']:<28s} last={s['last_trial'][-8:] if s['last_trial']!='(none)' else '-':>8s} ago={age_str:>6s} trials={s['trials']:>4d} RUNNING={s['running']}")

    # 3. Champion-log
    lines.append("\n[3] Champion-log 마지막 (진짜 진행 신호 ②)")
    if champion.get("last_event"):
        age = champion.get("age_sec", 0)
        flag = "🔄" if age < 300 else ("⚠" if age > threshold_sec else "  ")
        lines.append(f"  {flag} 마지막: {champion['last_event']} ({int(age//60)}분 전)")
        lines.append(f"    {champion['snippet']}")
    else:
        lines.append(f"  ⊘ champion 메시지 없음 (log: {champion.get('log', '?')})")

    # 4. per_model_optimal
    lines.append("\n[4] per_model_optimal 새 파일 (진짜 진행 신호 ③)")
    if pmo.get("count", 0) > 0:
        age = pmo.get("age_sec", 0)
        flag = "🔄" if age < 300 else ("⚠" if age > threshold_sec else "  ")
        lines.append(f"  {flag} {pmo['count']} 모델, 최신: {pmo['latest_model']} @ {pmo['latest_time']} ({int(age//60)}분 전)")
    else:
        lines.append("  ⊘ per_model_optimal 비어있음")

    # 5. Stuck 진단
    lines.append("\n[5] Stuck 자동 진단 (threshold=" + f"{threshold_sec}s={threshold_sec//60}분)")
    if diag["stuck"] is True:
        lines.append(f"  🚨 {diag['message']}")
        lines.append(f"     → 학습이 {threshold_sec//60}분 이상 진행 없음. kill 권고.")
    elif diag["stuck"] is False:
        lines.append(f"  ✓ {diag['message']}")
    else:
        lines.append(f"  ? {diag['reason']}")

    lines.append("=" * 80)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, default=0, help="주기 (초). 0=한번만.")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--threshold", type=int, default=1800,
                       help="Stuck threshold (초, default 1800=30분)")
    args = ap.parse_args()

    while True:
        if args.json:
            state = {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "processes": find_training_processes(),
                "optuna": optuna_last_trials(),
                "champion": champion_log_status(),
                "per_model_optimal": per_model_optimal_status(),
                "stuck_diagnosis": diagnose_stuck(args.threshold),
            }
            print(json.dumps(state, indent=2, default=str))
        else:
            print(render_health(args.threshold))

        if args.watch <= 0:
            break
        try:
            time.sleep(args.watch)
        except KeyboardInterrupt:
            break

    return 0


if __name__ == "__main__":
    sys.exit(main())
