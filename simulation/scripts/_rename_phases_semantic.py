"""One-off migration (2026-06-02): phase 번호-기반 모듈 → 의미이름 + back-compat alias.

사용자 결정 "코드도 넘버링 없이 그냥 이름으로". Step A = file 이동(git mv) + 옛 경로에
deprecation-alias 생성만. **참조 사이트(import/call)는 미변경** — 옛 import 는 alias 로 100% 작동
(G-150 per-site: 기계적 일괄 참조변경 금지 → canonical 마이그레이션은 Step B 에서 per-site Edit).

함수 rename / runner canonical 마이그레이션 / resume resolver = Step B/C (이 스크립트 범위 밖).
실행: .venv/bin/python -m simulation.scripts._rename_phases_semantic [--dry-run]
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PIPELINE = Path(__file__).resolve().parents[1] / "pipeline"

# (옛 번호-기반 모듈, 신규 의미이름). shap→shap_analysis (라이브러리 `import shap` 충돌 회피).
RENAME = [
    ("phase1_data", "data"),
    ("phase4_baseline", "baseline"),
    ("phase5_external", "external"),
    ("phase6_wfcv", "wfcv"),
    ("phase7_diagnostics", "diagnostics"),
    ("phase8_ar_correction", "ar_correction"),
    ("phase9_dm_test", "dm_test"),
    ("phase10_intervals", "intervals"),
    ("phase11_scoring", "scoring"),
    ("real_eval", "real_eval"),
    ("phase13_per_model_optimize", "per_model_optimize"),
    ("per_model_eval", "per_model_eval"),
    ("shap", "shap_analysis"),
    ("phase15_xai", "xai"),
    ("comprehensive_eval", "comprehensive_eval"),
    ("inference", "inference"),
    ("phase18_overseas", "overseas"),
    ("phase18_seoul_gu", "seoul_gu"),
    ("phase18_true_ili_cohort", "true_ili_cohort"),
]

ALIAS_TEMPLATE = '''"""Deprecation alias → :mod:`simulation.pipeline.{new}` (2026-06-02 phase semantic rename).

옛 번호-기반 모듈명. 신규 코드는 ``from simulation.pipeline.{new} import ...`` 사용.
번호↔의미이름 매핑: ``docs/PHASE_MAPPING.md``. back-compat (기존 caller/checkpoint/test 무변경).
"""
from __future__ import annotations

import warnings as _warnings

import simulation.pipeline.{new} as _canon

_warnings.warn(
    "simulation.pipeline.{old} 는 deprecation alias 입니다 (2026-06-02 semantic rename). "
    "simulation.pipeline.{new} 에서 import 하세요.",
    DeprecationWarning,
    stacklevel=2,
)
# 공개 + 단일밑줄 private 전체 re-export (dunder 제외) — 기존 import 100% 호환
globals().update({{_k: getattr(_canon, _k) for _k in dir(_canon) if not _k.startswith("__")}})
del _canon, _warnings
'''


# Step B: 함수 rename + import canonical 마이그레이션 ──────────────────────────────
# 옛 run_phaseN → 의미이름 (verified names; \b 경계로 prefix 충돌 회피: run_phase1 ≠ run_phase10).
FN_RENAME = {
    "data.py": [("run_phase1", "run_data")],
    "baseline.py": [("run_phase4_baseline", "run_baseline")],
    "external.py": [("run_phase5_external", "run_external")],
    "wfcv.py": [("run_phase6", "run_wfcv")],
    "diagnostics.py": [("run_phase7", "run_diagnostics")],
    "ar_correction.py": [("run_phase8", "run_ar_correction")],
    "dm_test.py": [("run_phase9", "run_dm_test")],
    "intervals.py": [("run_phase10_extended", "run_intervals_extended"),
                     ("run_phase10", "run_intervals")],
    "scoring.py": [("run_phase11", "run_scoring")],
    "real_eval.py": [("run_phase12", "run_real_eval")],
    "per_model_optimize.py": [("run_phase13", "run_per_model_optimize")],
    "per_model_eval.py": [("run_phase14", "run_per_model_eval")],
    "shap_analysis.py": [("run_phase15", "run_shap")],
    "xai.py": [("run_phase15_xai", "run_xai")],
    "comprehensive_eval.py": [("run_phase16", "run_comprehensive_eval")],
    "inference.py": [("run_phase17", "run_inference")],
}
# 전역 함수-토큰 치환용 평탄 목록 (긴 이름 먼저 — \b 로 무관하나 방어적).
ALL_FN = sorted(
    {pair for pairs in FN_RENAME.values() for pair in pairs},
    key=lambda p: -len(p[0]),
)
MODMAP = {old: new for old, new in RENAME}  # 옛 모듈 → 신규 모듈
CANONICAL_FILES = {f"{new}.py" for _old, new in RENAME}
# 처리 제외: 옛 alias 파일(19) + 기존 alias(phase2/3) + 이 스크립트.
SKIP_FILES = {f"{old}.py" for old, _new in RENAME} | {
    "phase2_multicollinearity.py", "phase3_feature_optuna.py",
    "_rename_phases_semantic.py",
}


def _migrate_text(txt: str) -> str:
    """import 모듈경로 + 함수 호출 토큰을 canonical 로 (참조 사이트 전용 — 문자열/키 미변경)."""
    import re
    for old_m, new_m in MODMAP.items():
        txt = txt.replace(f"from .{old_m} import", f"from .{new_m} import")
        txt = txt.replace(f"from simulation.pipeline.{old_m} import",
                          f"from simulation.pipeline.{new_m} import")
        txt = txt.replace(f"from simulation.pipeline import {old_m}",
                          f"from simulation.pipeline import {new_m}")
        txt = re.sub(rf"import simulation\.pipeline\.{old_m}\b",
                     f"import simulation.pipeline.{new_m}", txt)
    for old_fn, new_fn in ALL_FN:
        txt = re.sub(rf"\b{old_fn}\b", new_fn, txt)
    return txt


def step_b(dry_run: bool = False) -> int:
    import re
    root = Path(__file__).resolve().parents[1]  # simulation/
    changed = 0
    # 1) canonical 파일: 마이그레이션 + 자기 함수 back-compat alias append
    for cfile, pairs in FN_RENAME.items():
        p = PIPELINE / cfile
        txt = p.read_text(encoding="utf-8")
        new = _migrate_text(txt)
        alias_block = "\n\n# back-compat aliases (2026-06-02 semantic rename — 옛 run_phaseN)\n" + \
            "\n".join(f"{o} = {n}" for o, n in pairs) + "\n"
        new = new.rstrip() + "\n" + alias_block
        if new != txt:
            print(f"  canonical+alias: pipeline/{cfile}", flush=True)
            if not dry_run:
                p.write_text(new, encoding="utf-8")
            changed += 1
    # canonical 파일 중 함수 없는 것(overseas/seoul_gu/true_ili_cohort)도 cross-import 마이그레이션
    for cfile in CANONICAL_FILES - set(FN_RENAME):
        p = PIPELINE / cfile
        txt = p.read_text(encoding="utf-8")
        new = _migrate_text(txt)
        if new != txt:
            print(f"  canonical(import): pipeline/{cfile}", flush=True)
            if not dry_run:
                p.write_text(new, encoding="utf-8")
            changed += 1
    # 2) 그 외 모든 .py (runner/cli/server/scripts/tests/benchmarks) — alias/archive 제외
    for p in sorted(root.rglob("*.py")):
        if "_archive" in p.parts or "__pycache__" in p.parts:
            continue
        if p.name in SKIP_FILES or p.name in CANONICAL_FILES:
            continue
        txt = p.read_text(encoding="utf-8")
        new = _migrate_text(txt)
        if new != txt:
            print(f"  migrate: {p.relative_to(root)}", flush=True)
            if not dry_run:
                p.write_text(new, encoding="utf-8")
            changed += 1
    print(f"\n{'(dry-run) ' if dry_run else ''}Step B: {changed} 파일 변경.", flush=True)
    return 0


def main(dry_run: bool = False) -> int:
    moved = 0
    for old, new in RENAME:
        old_p = PIPELINE / f"{old}.py"
        new_p = PIPELINE / f"{new}.py"
        if not old_p.exists():
            print(f"  SKIP (없음): {old}.py", flush=True)
            continue
        if new_p.exists():
            print(f"  SKIP (대상 존재): {new}.py", flush=True)
            continue
        print(f"  mv {old}.py → {new}.py  + alias", flush=True)
        if dry_run:
            moved += 1
            continue
        # git mv (history 보존)
        r = subprocess.run(["git", "mv", str(old_p), str(new_p)],
                           cwd=PIPELINE, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"    git mv 실패: {r.stderr.strip()}", file=sys.stderr)
            return 1
        # 옛 경로에 alias 파일 생성
        old_p.write_text(ALIAS_TEMPLATE.format(old=old, new=new), encoding="utf-8")
        moved += 1
    print(f"\n{'(dry-run) ' if dry_run else ''}{moved} 모듈 처리 완료.", flush=True)
    return 0


if __name__ == "__main__":
    _dry = "--dry-run" in sys.argv
    if "--step" in sys.argv and sys.argv[sys.argv.index("--step") + 1] == "b":
        sys.exit(step_b(dry_run=_dry))
    sys.exit(main(dry_run=_dry))
