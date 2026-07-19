"""Mini test runner — 69 모델 일괄 검증.

Usage:
    .venv/bin/python -m simulation.scripts.mini_tests.runner
    .venv/bin/python -m simulation.scripts.mini_tests.runner --category dl
    .venv/bin/python -m simulation.scripts.mini_tests.runner --models DNN,TFT
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import Counter

import warnings
warnings.filterwarnings('ignore')

# Disable threading to avoid contention (LightGBM/XGBoost/CatBoost OMP)
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
# macOS-specific: prevent OMP fork issues (CQR-LightGBM Error #179)
os.environ.setdefault('KMP_INIT_AT_FORK', 'FALSE')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')


# OMP fork segfault 모델 (subprocess isolation 필요)
KNOWN_OMP_SEGFAULT = {'CQR-LightGBM', 'CQR-GBR'}

# Foundation 모델 OOM (G-177 — 별도 진단 sprint)
# G-261 (2026-06-13): Chronos 전 변형 제거 — Chronos retire (대체 = TimesFM-2.5 + TiRex).
KNOWN_OOM = {
    'FoundationModelTransfer', 'OverseasTransfer',
    # pytorch-forecasting wrappers (G-152 stuck — 90s timeout 으로도 부족)
    'TFT-pf', 'TiDE-pf', 'N-BEATS-pf', 'N-HiTS-pf', 'DeepAR-pf', 'RNN-pf',
    # Graph attention 무거움 (60s timeout 부족)
    'GE-DNN-GAT', 'GE-Transformer', 'GE-PNA', 'GE-ResGated',
    # Heavy DL (논문 학습 30분+, mini 부적합)
    'TFT', 'Mamba', 'PatchTST', 'iTransformer', 'TimesNet', 'TiDE',
    # Foundation/transfer
}


def _load_registry():
    """Force load all model modules → REGISTRY 채움."""
    import importlib
    import simulation.models  # base init

    models_dir = Path(__file__).resolve().parents[2] / 'models'
    skipped = []
    for p in sorted(models_dir.glob('*.py')):
        if p.name.startswith('_') or p.name == '__init__.py' or p.name == 'base.py':
            continue
        mod = f'simulation.models.{p.stem}'
        try:
            importlib.import_module(mod)
        except Exception as e:
            skipped.append((mod, type(e).__name__, str(e)[:60]))

    from simulation.models.base import REGISTRY
    return REGISTRY, skipped


def run_all(*, categories: list[str] | None = None,
            models: list[str] | None = None,
            timeout_sec: int = 60,
            output_dir: Path | None = None) -> dict:
    """Run mini test on all (or filtered) models.

    Args:
        categories: optional category filter (e.g., ['dl', 'tree'])
        models: optional explicit model name list
        timeout_sec: per-model time budget
        output_dir: result destination (default: simulation/results/mini_test_all_<date>/)

    Returns:
        summary dict
    """
    from simulation.scripts.mini_tests.synthetic import make_synthetic
    from simulation.scripts.mini_tests.diagnose import diagnose_model

    REGISTRY, skipped = _load_registry()
    if skipped:
        print(f'⚠ 모듈 import skip {len(skipped)}:')
        for m, et, msg in skipped:
            print(f'  {m}: {et}: {msg}')

    # 모델 목록 결정
    target_models = []
    for name, cls in sorted(REGISTRY._models.items()):
        cat = getattr(cls.meta, 'category', '?')
        if categories and cat not in categories:
            continue
        if models and name not in models:
            continue
        target_models.append((name, cls))

    print(f'\n=== Mini test: {len(target_models)} models ===')
    print(f'  timeout: {timeout_sec}s/model, total budget ~{len(target_models) * timeout_sec / 60:.0f}min')
    print()

    # Synthetic data (한 번만) — 실제 split 근사
    data = make_synthetic(n_train=240, n_val=30, n_test=50, n_features=20, seed=42)
    print(f'  data: train {len(data["y_train"])}, val {len(data["y_val"])}, test {len(data["y_test"])}, '
          f'features {data["X_train"].shape[1]}, y range {data["meta"]["y_range"]}')
    print()

    # Output dir
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[2] / 'results' / 'mini_test_all_20260505'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run per model
    results = {}
    status_counter = Counter()
    issue_counter = Counter()

    for i, (name, cls) in enumerate(target_models, 1):
        t0 = time.time()
        print(f'[{i}/{len(target_models)}] {name:<28s} ', end='', flush=True)
        if name in KNOWN_OMP_SEGFAULT:
            print(f'− SKIP_OMP  (known fork issue)')
            results[name] = {'name': name, 'status': 'SKIP_OMP',
                             'category': getattr(cls.meta, 'category', '?'),
                             'level': getattr(cls.meta, 'level', None),
                             'time_sec': 0.0, 'metrics': {},
                             'issues': ['OMP fork segfault (Error #179) — subprocess isolation 필요'],
                             'fix_hints': ['CQR family LightGBM/GBR backend OMP 충돌. subprocess spawn 적용 필요.']}
            status_counter['SKIP_OMP'] += 1
            (output_dir / f'{name}.json').write_text(json.dumps(results[name], indent=2, default=str))
            continue
        if name in KNOWN_OOM:
            print(f'− SKIP_OOM  (foundation/pytorch-forecasting — G-177/G-152)')
            results[name] = {'name': name, 'status': 'SKIP_OOM',
                             'category': getattr(cls.meta, 'category', '?'),
                             'level': getattr(cls.meta, 'level', None),
                             'time_sec': 0.0, 'metrics': {},
                             'issues': ['foundation OOM 또는 pytorch-forecasting Lightning stuck (별도 sprint)'],
                             'fix_hints': ['G-177 (chronos OOM), G-152 (Lightning max_time) — 별도 진단 sprint']}
            status_counter['SKIP_OOM'] += 1
            (output_dir / f'{name}.json').write_text(json.dumps(results[name], indent=2, default=str))
            continue
        result = diagnose_model(cls, data, model_name=name, timeout_sec=timeout_sec)
        elapsed = time.time() - t0
        results[name] = result
        status_counter[result['status']] += 1
        for iss in result['issues']:
            # bucket: first word of issue
            issue_counter[iss.split(':')[0].strip()] += 1
        # Compact print
        sym = {'PASS': '✓', 'WARN': '⚠', 'FAIL': '✗', 'SKIP': '−'}.get(result['status'], '?')
        r2 = result['metrics'].get('r2')
        r2_str = f'R²={r2:+.2f}' if r2 is not None else '       '
        print(f'{sym} {result["status"]:<5s} {r2_str}  ({elapsed:.1f}s)')
        if result['status'] == 'FAIL' or result['issues']:
            for iss in result['issues'][:3]:
                print(f'      ⚠ {iss[:100]}')

        # Save individual JSON
        out_file = output_dir / f'{name}.json'
        out_file.write_text(json.dumps(result, indent=2, default=str))

    # Summary
    summary = {
        'generated_at': '2026-05-05',
        'total_models': len(target_models),
        'status_counter': dict(status_counter),
        'issue_counter': dict(issue_counter.most_common(20)),
        'data_meta': data['meta'],
        'results_by_status': {
            'PASS': sorted([n for n, r in results.items() if r['status'] == 'PASS']),
            'WARN': sorted([n for n, r in results.items() if r['status'] == 'WARN']),
            'FAIL': [{'name': n, 'issues': r['issues'][:3], 'fix_hints': r['fix_hints'][:3]}
                     for n, r in sorted(results.items()) if r['status'] == 'FAIL'],
            'SKIP': sorted([n for n, r in results.items() if r['status'] == 'SKIP']),
        },
    }
    summary_file = output_dir / '_summary.json'
    summary_file.write_text(json.dumps(summary, indent=2, default=str))

    # Print final
    print()
    print('=' * 60)
    print(f'완료: {len(target_models)} models')
    for status, count in sorted(status_counter.items()):
        sym = {'PASS': '✓', 'WARN': '⚠', 'FAIL': '✗', 'SKIP': '−'}.get(status, '?')
        print(f'  {sym} {status}: {count}')
    print()
    print(f'Top issues:')
    for iss, cnt in issue_counter.most_common(10):
        print(f'  {cnt:>3d}× {iss[:80]}')
    print()
    print(f'Results: {output_dir}/_summary.json')

    return summary


def main():
    p = argparse.ArgumentParser(description='Mini test runner')
    p.add_argument('--category', type=str, default=None,
                   help='Filter by category (comma-separated, e.g., dl,tree)')
    p.add_argument('--models', type=str, default=None,
                   help='Filter by model name (comma-separated)')
    p.add_argument('--timeout', type=int, default=60,
                   help='Per-model timeout seconds (default 60)')
    args = p.parse_args()

    cats = [c.strip() for c in args.category.split(',')] if args.category else None
    models = [m.strip() for m in args.models.split(',')] if args.models else None

    run_all(categories=cats, models=models, timeout_sec=args.timeout)


if __name__ == '__main__':
    main()
