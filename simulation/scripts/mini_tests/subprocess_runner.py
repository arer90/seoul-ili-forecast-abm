"""Subprocess-isolated mini test — heavy/OMP-segfault 모델 격리 실행.

KNOWN_OMP_SEGFAULT (CQR family) + KNOWN_OOM (Foundation/pf/heavy DL) 검증용.
각 모델 별도 python process spawn → fail/segfault/OOM 격리, 다음 모델 진행 보장.

Usage:
    .venv/bin/python -m simulation.scripts.mini_tests.subprocess_runner \\
        --models "CQR-LightGBM,CQR-GBR,TimesFM-2.5,TFT-pf"
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from collections import Counter


# 격리 실행 child process 코드
_CHILD_CODE_TEMPLATE = '''
import sys, json, pickle, warnings, os
warnings.filterwarnings('ignore')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('KMP_INIT_AT_FORK', 'FALSE')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

# Force model module load (top-level + modern_ts subpackage)
import simulation.models
import importlib
from pathlib import Path
for p in sorted(Path("simulation/models").glob("*.py")):
    if p.name.startswith("_") or p.name in ("__init__.py", "base.py"):
        continue
    try:
        importlib.import_module(f"simulation.models.{{p.stem}}")
    except Exception:
        pass
# modern_ts subpackage
mt_dir = Path("simulation/models/modern_ts")
if mt_dir.exists():
    for p in sorted(mt_dir.glob("*.py")):
        if p.name.startswith("_") or p.name == "__init__.py":
            continue
        try:
            importlib.import_module(f"simulation.models.modern_ts.{{p.stem}}")
        except Exception:
            pass

from simulation.models.base import REGISTRY
from simulation.scripts.mini_tests.diagnose import diagnose_model

# Load data
with open("{data_path}", "rb") as f:
    data = pickle.load(f)

# Get model class
cls = REGISTRY._models.get("{model_name}")
if cls is None:
    print("__JSON_RESULT__" + json.dumps({{"name": "{model_name}", "status": "NOT_REGISTERED", "issues": ["model name not in REGISTRY"]}}))
    sys.exit(0)

# Run diagnose
result = diagnose_model(cls, data, model_name="{model_name}", timeout_sec={inner_timeout})
print("__JSON_RESULT__" + json.dumps(result, default=str))
'''


def _setup_data():
    """Synthetic data + pickle 저장 (subprocess 공유)."""
    from simulation.scripts.mini_tests.synthetic import make_synthetic
    data = make_synthetic(n_train=240, n_val=30, n_test=50, n_features=20, seed=42)
    f = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
    pickle.dump(data, f)
    f.close()
    return data, f.name


def diagnose_subprocess(model_name: str, data_path: str, *,
                        timeout: int = 200, inner_timeout: int = 150) -> dict:
    """Single-model subprocess isolation."""
    code = _CHILD_CODE_TEMPLATE.format(
        data_path=data_path,
        model_name=model_name,
        inner_timeout=inner_timeout,
    )
    t0 = time.time()
    try:
        # 2026-05-26: hardcoded user path → repo-relative (ENGINEERING_PRINCIPLES.md §원칙 #1 portability)
        from pathlib import Path as _Path
        _repo = _Path(__file__).resolve().parents[3]   # …/MPH_infection_simulation
        proc = subprocess.run(
            [sys.executable, '-c', code],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(_repo),
        )
        elapsed = time.time() - t0
        # stdout 마지막에 __JSON_RESULT__ 찾기
        result = None
        for line in proc.stdout.splitlines():
            if line.startswith('__JSON_RESULT__'):
                try:
                    result = json.loads(line[len('__JSON_RESULT__'):])
                except Exception as je:
                    result = {'name': model_name, 'status': 'FAIL_PARSE',
                              'issues': [f'JSON parse: {je}']}
                break

        if result is None:
            result = {
                'name': model_name,
                'status': 'FAIL_NO_RESULT',
                'issues': [f'no __JSON_RESULT__ marker (returncode={proc.returncode}, '
                           f'stderr 200b: {proc.stderr[-200:].strip()[:200]})'],
                'fix_hints': ['subprocess crashed before result emit — segfault/OOM/SIGKILL 추정']
            }
            if proc.returncode == -11:  # SIGSEGV
                result['issues'].append('SIGSEGV (signal 11) — OMP fork/native crash')
                result['fix_hints'].append('per-trial subprocess + KMP_INIT_AT_FORK=FALSE')
            elif proc.returncode == -9:  # SIGKILL (OS OOM kill)
                result['issues'].append('SIGKILL (signal 9) — OS OOM-killer')
                result['fix_hints'].append('memory cap 또는 모델 size 축소')
            elif proc.returncode == 137:  # 128+9
                result['issues'].append('exit 137 = OOM kill (Linux)')
        result['time_sec_subprocess'] = elapsed
        return result

    except subprocess.TimeoutExpired:
        return {
            'name': model_name,
            'status': 'FAIL_TIMEOUT',
            'time_sec_subprocess': float(timeout),
            'issues': [f'subprocess timeout {timeout}s — Lightning Trainer / Foundation HF load 가능'],
            'fix_hints': ['G-152 max_time 강제 또는 G-177 Foundation 별도 sprint'],
        }
    except Exception as e:
        return {
            'name': model_name,
            'status': 'FAIL_SUBPROCESS',
            'time_sec_subprocess': time.time() - t0,
            'issues': [f'subprocess: {type(e).__name__}: {str(e)[:150]}'],
            'fix_hints': [],
        }


def run_subprocess_models(models: list[str], *, timeout: int = 200,
                          output_dir: Path | None = None) -> dict:
    """Run subprocess-isolated mini test on a list of models."""
    if output_dir is None:
        from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
        output_dir = get_results_dir() / "mini_test_all_20260505"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'=== Subprocess-isolated mini test: {len(models)} models ===')
    print(f'  per-model timeout: {timeout}s, est total: {len(models)*timeout/60:.0f}min')
    print()

    # Setup data once
    _, data_path = _setup_data()

    results = {}
    status_counter = Counter()
    try:
        for i, name in enumerate(models, 1):
            print(f'[{i}/{len(models)}] {name:<28s} ', end='', flush=True)
            res = diagnose_subprocess(name, data_path, timeout=timeout,
                                      inner_timeout=int(timeout * 0.75))
            results[name] = res
            status_counter[res.get('status', 'UNKNOWN')] += 1
            sym_map = {'PASS': '✓', 'WARN': '⚠', 'FAIL': '✗',
                       'FAIL_TIMEOUT': '⏱', 'FAIL_NO_RESULT': '💥',
                       'FAIL_PARSE': '?', 'NOT_REGISTERED': '∅',
                       'FAIL_SUBPROCESS': '✗', 'SKIP': '−'}
            sym = sym_map.get(res.get('status', '?'), '?')
            r2 = res.get('metrics', {}).get('r2') if isinstance(res.get('metrics'), dict) else None
            r2_str = f'R²={r2:+.2f}' if r2 is not None and isinstance(r2, (int, float)) else '       '
            t = res.get('time_sec_subprocess', 0)
            print(f'{sym} {res.get("status", "?"):<18s} {r2_str}  ({t:.1f}s)')
            if res.get('issues'):
                for iss in res['issues'][:2]:
                    print(f'      ⚠ {iss[:100]}')
            # save per-model
            (output_dir / f'{name}_subprocess.json').write_text(json.dumps(res, indent=2, default=str))
    finally:
        try:
            os.unlink(data_path)
        except Exception:
            pass

    # Summary
    summary = {
        'generated_at': '2026-05-05',
        'mode': 'subprocess_isolated',
        'total_models': len(models),
        'status_counter': dict(status_counter),
        'results': results,
    }
    summary_file = output_dir / '_subprocess_summary.json'
    summary_file.write_text(json.dumps(summary, indent=2, default=str))

    print()
    print('=' * 60)
    print(f'Subprocess mini test 완료: {len(models)} models')
    for status, cnt in sorted(status_counter.items()):
        print(f'  {status}: {cnt}')
    print()
    print(f'Results: {summary_file}')
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', type=str, required=True,
                   help='Comma-separated model names')
    p.add_argument('--timeout', type=int, default=200,
                   help='Per-model timeout seconds')
    args = p.parse_args()

    models = [m.strip() for m in args.models.split(',')]
    run_subprocess_models(models, timeout=args.timeout)


if __name__ == '__main__':
    main()
