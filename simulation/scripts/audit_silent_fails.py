"""Silent fail pattern audit (G-169, 2026-05-03 — G-159 후속).

학습 path 의 silent fail 패턴 grep 후 분류:

  1. **CATASTROPHIC** (위험): except 후 caller 못 알아챔
     - `except: pass` (no log, no return)
     - `return None` 단독 (no log)
     - `score = 100.0 / -1.0` (sentinel)
  2. **LOGGED** (안전): log.warning/error 후 fallback
  3. **BEST-EFFORT** (안전): cleanup/atexit/heartbeat — fail 해도 정상

사용:
    .venv/bin/python -m simulation.scripts.audit_silent_fails
    .venv/bin/python -m simulation.scripts.audit_silent_fails --root simulation/models
    .venv/bin/python -m simulation.scripts.audit_silent_fails --strict   # exit 1 if catastrophic
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


# Pattern 분류
CATASTROPHIC_PATTERNS = [
    # except: pass 단독 (no log) — 진짜 silent
    (re.compile(r"^\s*except\s+(Exception)?\s*:\s*$"), "naked_except"),
    # score = 100.0 / -1.0 sentinel (G-159 같은)
    (re.compile(r"\bscore\s*=\s*(100\.0|-1\.0)\b"), "score_sentinel"),
    (re.compile(r"\bfold_scores\.append\((100\.0|-1\.0)\)"), "fold_score_sentinel"),
    # log.debug 후 silent — log.warning 으로 격상 권장 (G-159 fix 패턴)
    (re.compile(r"log\.debug\(.*fail|fail.*log\.debug"), "debug_log_for_fail"),
]

# log.warning / log.error / log.info / log.exception 인접 → logged (안전)
LOGGED_NEARBY = re.compile(r"log\.(warning|error|exception|info)")

# best-effort 키워드 (안전 분류)
BEST_EFFORT_KEYWORDS = re.compile(
    r"cleanup|atexit|heartbeat|empty_cache|gc\.collect|tmp|temp|"
    r"unlink|rmtree|close|flush|garbage|finalize"
)

# G-171 (2026-05-03, Q19 A): graceful fallback 패턴 인식 (false-positive 차단).
# 이전 audit: phase12 23 catastrophic 모두 actually graceful fallback (NaN/default)
# 이었음. context 에 다음 키워드 있으면 catastrophic 분류 X → graceful_fallback:
#   - "= float('nan')" / "= np.nan"        (NaN default)
#   - "= None" / "= []" / "= {}" / "= 0"    (default value)
#   - "fallback" / "graceful" / "fail-safe" / "default" / "intentional"  (의도 comment)
#   - "continue" (loop next iteration — graceful skip)
#   - "import.*optional" / "ImportError"   (optional dependency)
GRACEFUL_FALLBACK_KEYWORDS = re.compile(
    r"=\s*(float\(.nan.\)|np\.nan|None|\[\]|\{\}|0\.0|0\b)"
    r"|\bfallback\b|\bgraceful\b|fail-safe|fail_safe|\bdefault\b|\bintentional\b"
    r"|\bcontinue\b|optional|ImportError"
)

# G-237 (2026-05-30): critical-phase swallow — `except Exception` in a critical
# phase wrapper (champion gate / HP-optimize / SSOT eval) that continues WITHOUT a
# fail-loud marker. This is NOT "안전" even when log.error-adjacent — exactly the
# blind spot that let a 1-token NameError silently void phase12 for a 10h run.
# A fixed wrapper carries a marker ("critical": True / CRITICAL_FAILURES / raise),
# so post-fix this fires 0; only NEW unmarked critical swallows are flagged.
_CRIT_EXCEPT = re.compile(r"^\s*except\s+Exception(\s+as\s+\w+)?\s*:")
_CRIT_PHASE_CTX = re.compile(r"real_eval|per_model_optimize|per_model_eval|champion[ _]gate")
_FAILLOUD_MARKER = re.compile(r'"critical"\s*:\s*True|CRITICAL_FAILURES|_collect_critical|\braise\b')
# Orchestrator-swallow signature: stores an {"error": …} sentinel into all_results[…]
# (the phase-wrapper shape). Sub-operation excepts (log reads, debug) lack this, so
# requiring it keeps precision high — only true phase-wrapper swallows are flagged.
_SWALLOW_SIG = re.compile(r'all_results\s*\[[^\]]+\]\s*=\s*\{[^}]*error')


def audit_file(path: Path) -> dict:
    """단일 .py 파일의 silent-fail 패턴 audit + 분류 (G-169, D-4).

    `CATASTROPHIC_PATTERNS` (4 regex: naked_except / score_sentinel /
    fold_score_sentinel / debug_log_for_fail) 을 line-by-line scan. 각 매치를
    context window (5 lines before + after) 로 분류:
      - **catastrophic**: log/best-effort 키워드 없음 → 진짜 silent
      - **logged**: log.warning/error/exception 인접 → 안전
      - **best_effort**: cleanup/atexit/heartbeat/empty_cache 인접 → 안전

    Args:
        path: .py file path. 읽기 실패 시 빈 dict 반환.

    Returns:
        dict:
          - catastrophic (list[dict]): {line: int, kind: str, code: str}
          - logged (list[dict]): 동일 형식
          - best_effort (list[dict]): 동일 형식

    Raises:
        절대 raise X — read 실패는 빈 dict 반환.

    Performance: O(n_lines × n_patterns) — 4 patterns × ~1000 lines = 4ms.
    Side effects: 없음 (read-only).

    Caller responsibility:
        - `EXCLUDE_DIRS` 같은 sweeper level filtering.
        - `--strict` flag 처리 (exit 1 if catastrophic).

    Example:
        >>> r = audit_file(Path("simulation/models/runner.py"))
        >>> len(r["catastrophic"]), len(r["logged"]), len(r["best_effort"])
        (12, 8, 15)

    See: G-159 (silent 100.0 sentinel root cause),
         G-169 (audit_silent_fails.py 신규).
    """
    out = {"catastrophic": [], "logged": [], "best_effort": [], "critical_swallow": []}
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return out
    lines = content.splitlines()

    for i, line in enumerate(lines):
        stripped = line.strip()
        for pat, kind in CATASTROPHIC_PATTERNS:
            if pat.search(line):
                # context window: 5 before + 5 after (logged 가 나중에 올 수 있음)
                context = "\n".join(lines[max(0, i - 5):min(len(lines), i + 6)])
                # G-171 (2026-05-03): 분류 우선순위 (4-tier):
                #   best_effort > graceful_fallback > logged > catastrophic
                if BEST_EFFORT_KEYWORDS.search(context):
                    out["best_effort"].append({
                        "line": i + 1, "kind": kind, "code": stripped,
                    })
                elif GRACEFUL_FALLBACK_KEYWORDS.search(context):
                    # G-171: NaN/default/fallback/continue 등 의도적 graceful degrade
                    # → catastrophic 아님. "best_effort" 분류로 처리 (audit script 의 정확도 ↑).
                    out["best_effort"].append({
                        "line": i + 1, "kind": kind + "_graceful", "code": stripped,
                    })
                elif LOGGED_NEARBY.search(context):
                    out["logged"].append({
                        "line": i + 1, "kind": kind, "code": stripped,
                    })
                else:
                    out["catastrophic"].append({
                        "line": i + 1, "kind": kind, "code": stripped,
                    })
                break

        # G-237: critical-phase swallow — `except Exception` in a critical-phase
        # wrapper WITHOUT a fail-loud marker. Flagged regardless of log.error
        # adjacency (the prior blind spot that hid the phase12 champion-gate swallow).
        if _CRIT_EXCEPT.search(line):
            _ctx = "\n".join(lines[max(0, i - 2):min(len(lines), i + 9)])
            if (_CRIT_PHASE_CTX.search(_ctx) and _SWALLOW_SIG.search(_ctx)
                    and not _FAILLOUD_MARKER.search(_ctx)):
                out["critical_swallow"].append({
                    "line": i + 1, "kind": "critical_phase_swallow", "code": stripped,
                })

    return out


def main():
    """CLI entry — silent fail pattern audit + 분류 (G-159, G-169, D-4).

    simulation/ root 의 모든 .py file scan (legacy `_archive` / `pipeline_demo`
    /`tests` 제외). 각 file 의 `audit_file()` 호출 → catastrophic / logged /
    best_effort 분류 후 report.

    CLI args:
        --root: audit root dir (default "simulation").
        --strict: catastrophic >0 시 exit 1 (CI 게이트).
        --show-best-effort: best-effort 패턴도 출력 (verbose).

    Returns: int — exit code (0 = 통과, 1 = strict 시 catastrophic 발견).

    Side effects:
        - stdout: per-file catastrophic list + 요약 (catastrophic / logged /
                  best-effort count)
        - 빈 dict (read 실패 file 무시)

    Example:
        # 모든 simulation/ audit
        $ .venv/bin/python -m simulation.scripts.audit_silent_fails
        # strict (CI)
        $ ... --strict
        # 단일 dir
        $ ... --root simulation/models

    Performance: ~500ms (471 file × 4 patterns × ~1000 lines).
    Caller responsibility: --strict 사용 시 false-positive 검토 필요.

    See: G-159 (silent 100.0 sentinel root cause),
         G-169 (audit_silent_fails.py 신규 + audit script 자체).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="simulation",
                    help="audit root dir (default: simulation)")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if any CATASTROPHIC found")
    ap.add_argument("--show-best-effort", action="store_true",
                    help="show best-effort patterns too (verbose)")
    args = ap.parse_args()

    root = ROOT / args.root
    if not root.exists():
        print(f"✗ root 없음: {root}")
        return 1

    py_files = sorted(root.rglob("*.py"))
    # G-169: 제외 디렉토리 — legacy archive / demo / test 는 학습 path 영향 X
    EXCLUDE_DIRS = {"__pycache__", "_archive", "pipeline_demo", "_trash",
                    "abm_v1", "tests", "test"}
    py_files = [p for p in py_files
                if not any(part in EXCLUDE_DIRS for part in p.parts)]

    print(f"=== Silent fail pattern audit (G-159, G-169) ===")
    print(f"Root: {root}")
    print(f"Files: {len(py_files)}")
    print()

    total_cat = total_log = total_be = total_crit = 0
    cat_files = []
    crit_files = []
    for p in py_files:
        result = audit_file(p)
        n_cat = len(result["catastrophic"])
        n_log = len(result["logged"])
        n_be = len(result["best_effort"])
        n_crit = len(result.get("critical_swallow", []))
        total_cat += n_cat
        total_log += n_log
        total_be += n_be
        total_crit += n_crit

        if n_cat > 0:
            rel = p.relative_to(ROOT)
            cat_files.append((rel, result["catastrophic"]))
        if n_crit > 0:
            crit_files.append((p.relative_to(ROOT), result["critical_swallow"]))

    # Report
    print(f"Total CRITICAL-PHASE SWALLOW (G-237, 위험): {total_crit}")
    print(f"Total CATASTROPHIC (위험): {total_cat}")
    print(f"Total LOGGED (안전 — logged):       {total_log}")
    print(f"Total BEST-EFFORT (안전 — cleanup): {total_be}")
    print()

    if crit_files:
        print(f"=== CRITICAL-PHASE SWALLOW (G-237 — {len(crit_files)} files) ===")
        print("  (critical-phase `except Exception` without fail-loud marker — review)")
        for rel, items in crit_files:
            print(f"\n{rel}:")
            for item in items[:10]:
                print(f"  L{item['line']:4d} [{item['kind']}]: {item['code']}")
        print()

    if cat_files:
        print(f"=== CATASTROPHIC patterns ({len(cat_files)} files) ===")
        for rel, items in cat_files:
            print(f"\n{rel}:")
            for item in items[:10]:  # cap 10 per file to avoid spam
                print(f"  L{item['line']:4d} [{item['kind']}]: {item['code']}")
            if len(items) > 10:
                print(f"  ... +{len(items) - 10} more")

    print()
    print("=" * 60)
    if args.strict and total_cat > 0:
        print(f"✗ STRICT mode: {total_cat} catastrophic silent fails → exit 1")
        return 1
    elif total_cat > 0:
        print(f"⚠ {total_cat} catastrophic silent fails — fix 권장")
        print(f"  fix 가이드: log.warning(f\"[ctx] {{type(e).__name__}}: {{e}}\")"
              " 추가 후 명시적 fallback")
        return 0
    else:
        print(f"✓ No catastrophic silent fails detected")
        return 0


if __name__ == "__main__":
    sys.exit(main())
