"""per-model 시간·CPU 를 training log 에서 post-hoc 파싱 → CSV (G-357, 2026-06-25, 사용자).

R9(per_model_optimize) subprocess 격리 모니터가 매 60s 찍는
``[R9-isolate] <model> 진행 elapsed=<N>s ... CPU=<N>%`` 라인을 모델별로 집계한다.
peak RSS 는 로그에 없어(per-phase ResourceTracker 만) 여기선 시간·CPU 만 — peak-memory 는 G-358
(R9 subprocess 측정→구조화 저장)이 다음 run 부터 채운다.

baseline 시간(summary_metrics.csv elapsed_s)·phase 자원(ResourceTracker)과 상보적. read-only(파이프라인 무수정).

Usage:
    .venv/bin/python -m simulation.scripts.parse_per_model_resources [training_log] [out_csv]
    (인자 없으면 최신 $TMPDIR/training_resume_*.log → simulation/results/csv/per_model_resources.csv;
     $TMPDIR = paths.fast_tmp = MPH_FAST_TMPDIR 또는 tempfile.gettempdir())

Returns: 작성한 CSV 경로(print). per-model {model, r9_elapsed_s, cpu_mean, cpu_peak, n_samples}.
"""
from __future__ import annotations

import csv
import glob
import os
import re
import sys

_RE = re.compile(r"\[R9-isolate\]\s+(?:mc:)?(\S+)\s+진행\s+elapsed=(\d+)s.*?CPU=([\d.]+)%")


def parse(log_path: str) -> dict:
    """training log → {model: {elapsed_max, cpu_samples[]}}.

    Args:
        log_path: R9-isolate 라인을 가진 training_resume_*.log 경로.
    Returns:
        모델별 집계 dict.
    """
    agg: dict[str, dict] = {}
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = _RE.search(line)
            if not m:
                continue
            name, elapsed, cpu = m.group(1), int(m.group(2)), float(m.group(3))
            a = agg.setdefault(name, {"elapsed_max": 0, "cpu": []})
            a["elapsed_max"] = max(a["elapsed_max"], elapsed)
            a["cpu"].append(cpu)
    return agg


def write_csv(agg: dict, out_csv: str) -> str:
    """집계 → CSV. Returns out_csv 경로. Side effects: out_csv 작성."""
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "r9_elapsed_s", "cpu_mean_pct", "cpu_peak_pct", "n_samples"])
        for name in sorted(agg):
            a = agg[name]
            cpu = a["cpu"]
            cm = round(sum(cpu) / len(cpu), 1) if cpu else 0.0
            cp = round(max(cpu), 1) if cpu else 0.0
            w.writerow([name, a["elapsed_max"], cm, cp, len(cpu)])
    return out_csv


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    # paths.fast_tmp is the SSOT for the temp directory (MPH_FAST_TMPDIR, else
    # tempfile.gettempdir()). A literal "/tmp" here would miss the log on macOS,
    # where each user gets a private TMPDIR, and on Windows entirely — which is
    # exactly the discovery-by-glob failure the SSOT exists to prevent.
    from simulation.config_global import GLOBAL as _G
    pattern = str(_G.paths.fast_tmp / "training_resume_*.log")
    hits = sorted(glob.glob(pattern), key=os.path.getmtime)
    log_path = argv[0] if argv else (hits[-1] if hits else None)
    out_csv = argv[1] if len(argv) > 1 else "simulation/results/csv/per_model_resources.csv"
    if not log_path or not os.path.exists(log_path):
        print("training log 없음", file=sys.stderr)
        return 1
    agg = parse(log_path)
    write_csv(agg, out_csv)
    print(f"{out_csv} ({len(agg)} models from {os.path.basename(log_path)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
