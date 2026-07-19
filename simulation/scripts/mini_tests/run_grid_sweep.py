"""Run grid sweep on all catastrophic models — 25 combo (5 transform × 5 scaler).

사용자 요청 (2026-05-05): "전처리를 할 수 있는 것들을 다 해봐" — catastrophic 모델 진단.

Identifies:
- 'default 에서만 fail' (transform 잘못) → grid sweep 후 PASS
- 'all combos fail' (모델 자체 발산) → 별도 fix 필요

사용:
    .venv/bin/python -m simulation.scripts.mini_tests.run_grid_sweep
    .venv/bin/python -m simulation.scripts.mini_tests.run_grid_sweep --models "TiDE-pf,N-BEATS-pf"
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('KMP_INIT_AT_FORK', 'FALSE')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')


def _load_registry():
    """Force load all model modules including modern_ts subpackage."""
    import importlib
    import simulation.models
    models_dir = Path(__file__).resolve().parents[2] / 'models'
    for p in sorted(models_dir.glob('*.py')):
        if p.name.startswith('_') or p.name in ('__init__.py', 'base.py'):
            continue
        try:
            importlib.import_module(f'simulation.models.{p.stem}')
        except Exception:
            pass
    # modern_ts subpackage
    mt_dir = models_dir / 'modern_ts'
    if mt_dir.exists():
        for p in sorted(mt_dir.glob('*.py')):
            if p.name.startswith('_') or p.name == '__init__.py':
                continue
            try:
                importlib.import_module(f'simulation.models.modern_ts.{p.stem}')
            except Exception:
                pass

    from simulation.models.base import REGISTRY
    return REGISTRY


def _get_catastrophic_models() -> list[str]:
    """Catastrophic 25 from real R9 (per_model_optimize) + synthetic mini test."""
    from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
    PMO = get_results_dir() / 'per_model_optimal'
    catastrophic = []
    if PMO.exists():
        for j in PMO.glob('*.json'):
            if j.stem in ('summary',) or j.stem.startswith('_'):
                continue
            try:
                d = json.loads(j.read_text())
                r2 = d.get('test_metrics', {}).get('r2')
                if isinstance(r2, (int, float)) and r2 < 0:
                    catastrophic.append(j.stem)
            except Exception:
                pass
    # Synthetic mini test catastrophic
    from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
    MTD = get_results_dir() / "mini_test_all_20260505"
    if MTD.exists():
        for j in MTD.glob('*.json'):
            if j.stem.startswith('_') or j.stem == 'summary':
                continue
            try:
                d = json.loads(j.read_text())
                # main run + subprocess run
                metrics = d.get('metrics') or {}
                r2 = metrics.get('r2') if isinstance(metrics, dict) else None
                if isinstance(r2, (int, float)) and r2 < 0:
                    name = j.stem.replace('_subprocess', '')
                    if name not in catastrophic:
                        catastrophic.append(name)
            except Exception:
                pass
    return sorted(set(catastrophic))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', type=str, default=None,
                   help='Comma-separated model names (default: catastrophic auto-detect)')
    p.add_argument('--timeout-per-combo', type=int, default=30)
    from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
    p.add_argument('--output-dir', type=str,
                   default=str(get_results_dir() / "mini_test_all_20260505" / "grid_sweep"))
    args = p.parse_args()

    REGISTRY = _load_registry()

    if args.models:
        target = [m.strip() for m in args.models.split(',')]
    else:
        target = _get_catastrophic_models()

    target = [m for m in target if m in REGISTRY._models]
    print(f'=== Grid Sweep: {len(target)} catastrophic models × {len(__import__("simulation.scripts.mini_tests.grid_sweep", fromlist=["TRANSFORMS"]).TRANSFORMS) * len(__import__("simulation.scripts.mini_tests.grid_sweep", fromlist=["SCALERS"]).SCALERS)} combos ===')
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from simulation.scripts.mini_tests.grid_sweep import diagnose_model_with_grid
    from simulation.scripts.mini_tests.synthetic import make_synthetic
    data = make_synthetic(n_train=240, n_val=30, n_test=50, n_features=20, seed=42)

    summary = {'models': {}, 'verdict_counter': {}}
    t_overall = time.time()

    for i, name in enumerate(target, 1):
        cls = REGISTRY._models[name]
        t0 = time.time()
        print(f'\n[{i}/{len(target)}] {name:<28s} grid sweep...')
        try:
            result = diagnose_model_with_grid(
                cls, data, model_name=name,
                timeout_per_combo=args.timeout_per_combo)
        except Exception as e:
            result = {'name': name, 'verdict': 'EXCEPTION',
                      'error': f'{type(e).__name__}: {str(e)[:200]}',
                      'all_combos': []}

        summary['models'][name] = result
        verdict = result.get('verdict', '?')
        summary['verdict_counter'][verdict] = summary['verdict_counter'].get(verdict, 0) + 1

        # 결과 출력
        elapsed = time.time() - t0
        if result.get('best'):
            b = result['best']
            print(f'   verdict: {verdict}')
            print(f'   best: transform={b.get("transform")}, scaler={b.get("scaler")}, R²={b.get("r2"):+.3f}, MAPE={b.get("mape"):.1f}%')
            print(f'   stats: PASS={result["n_pass"]} / CATASTROPHIC={result["n_catastrophic"]} / FAIL={result["n_fail"]} ({elapsed:.0f}s)')
        else:
            print(f'   verdict: {verdict} (no valid combo)')

        # save per-model
        (output_dir / f'{name}_grid.json').write_text(json.dumps(result, indent=2, default=str))

    summary['total_time_sec'] = time.time() - t_overall

    # Save summary
    (output_dir / '_grid_summary.json').write_text(json.dumps(summary, indent=2, default=str))

    # Final report
    print()
    print('=' * 70)
    print(f'Grid sweep 완료: {len(target)} models, {time.time()-t_overall:.0f}s')
    for v, c in sorted(summary['verdict_counter'].items()):
        print(f'  {v}: {c}')
    print()
    print(f'Saved: {output_dir}/_grid_summary.json')

    # 모델별 best 요약
    print()
    print('=== 모델별 best (R² descending) ===')
    sorted_results = sorted(
        [(n, r) for n, r in summary['models'].items() if r.get('best')],
        key=lambda x: -x[1]['best']['r2']
    )
    for name, res in sorted_results[:30]:
        b = res['best']
        verdict_short = 'PASS' if b['r2'] >= 0.8 else ('BORDER' if b['r2'] >= 0 else 'CATAS')
        print(f'  {name:<26s}  best R²={b["r2"]:+.3f}  ({b["transform"]:<12s} × {b["scaler"]:<10s})  {verdict_short}')


if __name__ == '__main__':
    main()
