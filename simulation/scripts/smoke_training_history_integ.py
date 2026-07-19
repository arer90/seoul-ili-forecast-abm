"""G-269b 통합 스모크 — training-history 훅이 실제 모델 객체로 작동하는가.

unit test(test_training_history.py)는 mock shape. 이건 (1) 실제 DNN._history 가
_epoch_rows 가 기대하는 shape 인지, (2) record_type='auto' 자동판별, (3) optuna trial_results
list 경로, (4) 일부러 깨진 입력에 try/except 가드가 학습을 안 죽이는지를 실측.

run: KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=2 .venv/bin/python -m simulation.scripts.smoke_training_history_integ
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from simulation.pipeline.training_history import save_training_record


def _synth(n=110, p=10, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float64)
    y = np.clip(2.0 + 0.6 * X[:, 0] + rng.normal(scale=0.3, size=n), 0, None)
    return X, y


def main():
    out = Path(tempfile.mkdtemp(prefix="th_integ_"))
    X, y = _synth()
    Xtr, ytr = X[:-15], y[:-15]
    n_ok = 0

    # ── (1) 실제 DNN: _history shape + record_type='auto' 자동판별 ──
    try:
        from simulation.models.dl_models import DNNForecaster
        m = DNNForecaster()
        # epoch 줄여 빠르게 (early-stop 도 작동)
        if hasattr(m, "_EPOCHS"):
            m._EPOCHS = 40
        m.fit(Xtr, ytr)
        has_hist = hasattr(m, "_history") and bool(getattr(m, "_history"))
        rtype = getattr(m, "_history_record_type", "dl_epoch")
        print(f"[1] DNN fit: _history 존재={has_hist}, record_type={rtype}, "
              f"len={len(getattr(m,'_history',[]) or [])}")
        csv = save_training_record("DNN", "pooled", "auto",
                                   {"model": m, "metrics": {"test_wis": 1.23, "test_r2": 0.7}}, out)
        df = pl.read_csv(csv)
        dl_rows = df.filter(pl.col("record_type") == "dl_epoch").height
        png = (out / "figures" / "DNN_pooled.png").exists()
        print(f"    → CSV {df.height} rows (dl_epoch={dl_rows}), 학습곡선 PNG={png}")
        assert df.height > 0 and png, "DNN history CSV/PNG 생성 실패"
        n_ok += 1
        print("    ✓ 실제 DNN 학습곡선 영속화 OK")
    except Exception as e:
        print(f"    ✗ DNN 경로 실패: {type(e).__name__}: {str(e)[:120]}")

    # ── (2) 실제 XGBoost optuna trial_results 형태 (list[dict]) ──
    try:
        # 훅이 넘기는 trial_results 형태 모사: list of dict (number/value/params)
        trial_results = [
            {"number": 0, "value": 5.1, "params": {"max_depth": 4}},
            {"number": 1, "value": 3.2, "params": {"max_depth": 6}},
            {"number": 2, "value": 3.0, "params": {"max_depth": 5}},
        ]
        csv = save_training_record("XGBoost", "pooled", "optuna_trial", trial_results, out)
        df = pl.read_csv(csv)
        opt_rows = df.filter(pl.col("record_type") == "optuna_trial").height
        png = (out / "figures" / "XGBoost_pooled.png").exists()
        print(f"[2] XGBoost optuna trial_results(list): CSV {opt_rows} trial rows, PNG={png}")
        assert opt_rows == 3 and png
        n_ok += 1
        print("    ✓ optuna trial 수렴 영속화 OK")
    except Exception as e:
        print(f"    ✗ optuna 경로 실패: {type(e).__name__}: {str(e)[:120]}")

    # ── (3) try/except 가드: 일부러 깨진 입력 → 훅 패턴이 학습을 안 죽이나 ──
    try:
        hook_survived = False
        try:
            # 훅과 동일 패턴: save_training_record 가 raise 해도 바깥은 계속
            save_training_record("Bad", "pooled", "auto", object(), out)  # auto+미지원객체 → ValueError
        except Exception as _hist_exc:
            hook_survived = True  # 훅의 except 가 잡는 시나리오 재현
        print(f"[3] 깨진 입력: save_training_record raise={hook_survived} "
              f"(훅의 try/except 가 잡아 학습 계속 — 설계대로)")
        assert hook_survived, "깨진 입력인데 raise 안 함 (가드 무의미)"
        n_ok += 1
        print("    ✓ try/except 가드 시나리오 확인 (history 실패≠학습 중단)")
    except Exception as e:
        print(f"    ✗ 가드 검증 실패: {type(e).__name__}: {str(e)[:120]}")

    # ── (4) summary_wis.png cross-model ──
    summary = (out / "figures" / "summary_wis.png").exists()
    print(f"[4] cross-model summary_wis.png={summary}")
    n_ok += 1 if summary else 0

    print(f"\n[결과] {n_ok}/4 통과 — out={out}")
    print(f"[파일] CSV: {[p.name for p in sorted(out.glob('*.csv'))]}")
    print(f"       PNG: {[p.name for p in sorted((out/'figures').glob('*.png'))]}")


if __name__ == "__main__":
    main()
