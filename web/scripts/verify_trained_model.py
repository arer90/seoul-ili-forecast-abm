#!/usr/bin/env python3
"""학습이 완료되어 저장된 모델(.pt 아티팩트)을 불러와 REAL 을 제대로 예측하는지 검증.

사용자: "가지고 있는 데이터로, 학습이 완료된 모델에서 real 을 제대로 예측하는지 확인."

그동안 web(build_production_forecast)은 매번 model.fit() **재적합**했음. 이 스크립트는 그게
아니라 **이미 학습이 끝나 디스크에 저장된 trained 아티팩트**(models/*.pt,
simulation/checkpoints_history/*.pt = pickle 된 fitted 모델)를 `load_artifact()` 로 불러와:

  1. 충실도: 불러온 모델의 예측이 학습 때 저장된 predictions_<name>.csv(val/test)를 재현하는가?
     → 재현되면 "저장된 아티팩트 = 학습 완료된 그 모델" 확인.
  2. 정확도: 그 예측이 REAL ILI 와 얼마나 맞는가? (val/test = held-out real)
  3. 사용성: 아티팩트가 에러 없이 로드+예측되는가? (web 이 재적합 대신 이걸 써도 되는가)

Read-only. Run: .venv/bin/python web/scripts/verify_trained_model.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "web" / "scripts"))

from build_production_forecast import _load_feature_matrix, _extract_basic_features  # noqa: E402
from simulation.utils.model_artifact import load_artifact  # noqa: E402

ARTIFACTS = [
    ("NegBinGLM-V7 (models/, 등록 champion)", ROOT / "models" / "NegBinGLM-V7.pt"),
    ("NegBinGLM (checkpoints_history/)", ROOT / "simulation" / "checkpoints_history" / "NegBinGLM.pt"),
]


def _metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) < 2:
        return dict(n=len(y), mae=float("nan"), rmse=float("nan"), r2=float("nan"))
    err = p - y
    ss = float(np.sum((y - y.mean()) ** 2))
    return dict(n=len(y), mae=float(np.mean(np.abs(err))),
                rmse=float(np.sqrt(np.mean(err ** 2))),
                r2=(1 - float(np.sum(err ** 2)) / ss) if ss > 0 else float("nan"))


def _saved_preds(name: str):
    f = ROOT / "simulation" / "results" / "csv" / f"predictions_{name}.csv"
    if not f.is_file():
        return None
    rows = list(csv.DictReader(f.open(encoding="utf-8")))
    return [(r["split"], float(r["y_true"]), float(r["y_pred"])) for r in rows]


def main() -> None:
    X_all, y_all, fcols, ws = _load_feature_matrix()
    X_all = np.asarray(X_all, float)
    y_all = np.asarray(y_all, float)
    Xb, _bc, _bi = _extract_basic_features(X_all, fcols)
    n = len(y_all)
    print(f"=== 저장된 trained 모델 → REAL 예측 검증 (feature matrix {X_all.shape}, n={n}) ===\n")

    for label, path in ARTIFACTS:
        print(f"── {label}")
        print(f"   {path.relative_to(ROOT)}  ({path.stat().st_size if path.is_file() else 0} bytes)")
        if not path.is_file():
            print("   ✗ 파일 없음\n"); continue
        try:
            art = load_artifact(path)
        except Exception as e:
            print(f"   ✗ load_artifact 실패: {type(e).__name__}: {e}\n"); continue
        if art is None:
            print("   ✗ load_artifact → None\n"); continue

        mdl = art.model
        nfeat = getattr(mdl, "n_features_in_", None)
        print(f"   로드 OK · model={type(mdl).__name__} · transform={art.transform_name} "
              f"· feature_indices={'set('+str(len(art.feature_indices))+')' if art.feature_indices else 'None'} "
              f"· n_features_in_={nfeat}")

        # choose feature matrix matching the model's expected width
        candidates = [("full", X_all), ("basic", Xb)]
        Xuse, used = None, None
        for tag, M in candidates:
            try:
                test_pred = art.predict(M[:5])
                if np.asarray(test_pred).shape[0] == 5:
                    Xuse, used = M, tag
                    break
            except Exception:
                continue
        if Xuse is None:
            print("   ✗ 어떤 feature width(full/basic)로도 predict 실패\n"); continue
        print(f"   predict 입력 = {used} features ({Xuse.shape[1]} cols)")

        pred_all = np.asarray(art.predict(Xuse), float).ravel()

        # full real series (대부분 in-sample → 낙관) + 최근 held-out 근사(마지막 68주=test 크기)
        m_full = _metrics(y_all, pred_all)
        m_tail = _metrics(y_all[-68:], pred_all[-68:])
        print(f"   REAL 전체 {m_full['n']}주: R²={m_full['r2']:+.3f} MAE={m_full['mae']:.2f} "
              f"(대부분 in-sample → 낙관)")
        print(f"   최근 68주(≈test 크기): R²={m_tail['r2']:+.3f} MAE={m_tail['mae']:.2f}")

        # faithfulness: reproduce saved predictions_<name>.csv?
        nm = "NegBinGLM" if "checkpoints_history" in str(path) else "NegBinGLM-V7"
        sp = _saved_preds(nm)
        if sp:
            tv = [t for t in sp if t[0] == "test"]
            yt = np.array([t[1] for t in tv]); ypsaved = np.array([t[2] for t in tv])
            # align: assume test = last len(tv) rows chronologically
            k = len(tv)
            pred_tail = pred_all[-k:]
            real_tail = y_all[-k:]
            align_ok = np.allclose(real_tail, yt, atol=0.5)
            faith = _metrics(ypsaved, pred_tail)   # loaded vs saved predictions
            msaved = _metrics(yt, ypsaved)          # saved-pred accuracy (학습 기록)
            print(f"   저장 예측(predictions_{nm}.csv) test n={k}: 기록 R²={msaved['r2']:+.3f} "
                  f"MAE={msaved['mae']:.2f}")
            print(f"   충실도(불러온 예측 vs 저장 예측): {'✓ 재현' if faith['r2'] > 0.98 else '✗ 불일치'} "
                  f"(R²={faith['r2']:+.3f}, y_true 정렬={'OK' if align_ok else 'mismatch'})")
        print()

    print("해석: 충실도 ✓ = 저장된 아티팩트가 '학습 완료된 바로 그 모델' → 재적합 없이 로드만으로")
    print("      real 예측 가능. 충실도 ✗ = 아티팩트가 학습 기록과 다름(feature/split 불일치 조사 필요).")


if __name__ == "__main__":
    main()
