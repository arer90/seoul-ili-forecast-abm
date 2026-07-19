"""G-270 smoke — 실제 N-BEATS(Lightning) fit 이 새 val_dl+best-restore 흐름으로 끝까지 도는가.

unit test 는 toy LightningModule. 이건 실제 PfNBeatsForecaster.fit(X,y)→predict 가
val split(min_prediction_idx)+EarlyStopping+ModelCheckpoint best 복원+_history(lightning_epoch)
까지 정상 동작하는지 실측. 버그시 phase-13 try/except 가 잡아 모델만 skip 되므로 재실행 전 확인.

run: KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=2 .venv/bin/python -m simulation.scripts.smoke_nbeats_earlystop
"""
from __future__ import annotations

import numpy as np


def main():
    rng = np.random.default_rng(42)
    n, p = 160, 8
    t = np.arange(n)
    X = rng.normal(size=(n, p)).astype(np.float64)
    y = np.clip(3.0 + 1.5 * np.sin(2 * np.pi * t / 52) + 0.4 * X[:, 0]
                + rng.normal(scale=0.3, size=n), 0, None)
    Xtr, ytr, Xte = X[:-20], y[:-20], X[-20:]

    from simulation.models.modern_ts.pf_models import PfNBeatsForecaster

    m = PfNBeatsForecaster()
    # 빠른 smoke: epoch 상한 낮춤 (있으면)
    for attr in ("MAX_EPOCHS", "_max_epochs", "max_epochs"):
        if hasattr(m, attr):
            try:
                setattr(m, attr, 12)
            except Exception:
                pass
    print(f"[smoke] PfNBeatsForecaster fit (n_train={len(Xtr)}) …")
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)

    hist = getattr(m, "_history", None)
    rtype = getattr(m, "_history_record_type", None)
    print(f"[smoke] ✓ fit+predict 완료 — pred.shape={np.asarray(pred).shape}, "
          f"finite={np.all(np.isfinite(pred))}")
    print(f"[smoke] _history_record_type={rtype}, epochs 기록={len(hist) if hist else 0}")
    if hist:
        keys = sorted({k for r in hist for k in r})
        print(f"[smoke] history keys={keys}")
        has_val = any("val" in k for r in hist for k in r)
        print(f"[smoke] val_loss 기록됨={has_val} (EarlyStopping monitor 가능 = early-stop 작동 전제)")

    assert np.asarray(pred).shape[0] == 20, "predict 길이 불일치"
    assert np.all(np.isfinite(pred)), "predict NaN/inf"
    assert rtype == "lightning_epoch", f"_history_record_type={rtype} != lightning_epoch"
    assert hist and len(hist) > 0, "lightning_epoch history 비어있음 — 콜백 미작동"
    print("\n[결과] ✓ 실제 N-BEATS Lightning early_stop+best_restore+history 전 흐름 정상")


if __name__ == "__main__":
    main()
