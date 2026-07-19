"""Reconstruct a per-backend accuracy ranking from a factual_bench log.

`run_factual_benchmark` writes `factual_report.json` only at the very end (after
ALL backends), so an interrupted run leaves nothing on disk. This salvages the
accuracy ranking from the per-item log lines
(``backend=... item=... rep=0 acc=X``). It recovers ACCURACY ONLY — the full
SCI stats (Wilcoxon+Holm, Fleiss κ, repro_manifest) need the completed run.

Usage:  python scripts/recover_bench_from_log.py [$TMPDIR/full_combined.log]
"""
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

log = sys.argv[1] if len(sys.argv) > 1 else str(Path(tempfile.gettempdir()) / "full_combined.log")
pat = re.compile(r"backend=(\S+?) item=(\S+) rep=0 acc=([0-9.]+)")
acc: dict[str, list[float]] = defaultdict(list)
errs: dict[str, int] = defaultdict(int)
with open(log, encoding="utf-8") as f:
    for line in f:
        m = pat.search(line)
        if not m:
            continue
        b, _item, a = m.group(1), m.group(2), float(m.group(3))
        if "ERR=" in line:
            errs[b] += 1          # logged but errored (auth/timeout) — count separately
        else:
            acc[b].append(a)

rows = [(b, sum(v) / len(v), len(v), errs.get(b, 0)) for b, v in acc.items() if v]
rows.sort(key=lambda x: x[1], reverse=True)
print(f"# Recovered accuracy ranking from {log}\n")
print(f"{'rank':<5}{'backend':<46}{'accuracy':<11}{'n':<5}{'errs'}")
print("-" * 72)
for i, (b, a, n, e) in enumerate(rows, 1):
    print(f"{i:<5}{b.split('@')[0]:<46}{a:<11.4f}{n:<5}{e}")
excluded = [b for b in errs if b not in {r[0] for r in rows}]
if excluded:
    print(f"\nexcluded (all errored): {excluded}")
print("\n※ 정확도만 복구 — Wilcoxon/Fleiss κ/repro_manifest는 완주한 run 필요.")
