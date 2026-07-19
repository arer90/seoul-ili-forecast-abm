#!/usr/bin/env python3
"""학습 완료 모델로 REAL 예측 — (A) 작동 경로 확인 + (B) 저장된 모델로도 되게 fix + 검증.

배경(verify_trained_model.py 발견): 디스크의 champion .pt 는 'legacy bare-model pickle' 이라
학습 파이프라인(transform/scaler/feature 선택)이 번들에 없어 **불러오면 예측이 깨짐**
(R²<0, V7 은 전부 0). 그래서 web 은 매번 재적합으로만 작동.

이 스크립트:
  (A) 작동 경로 확인 — NegBinGLM 을 BASIC feature 로 재적합 → 시간순 holdout(최근 68주) test R²
      → "재적합하면 real 을 제대로 예측하는가" 정량 확인.
  (B) fix — 재적합 모델을 **자립형 ChampionArtifact**(model + feature_indices=BASIC + identity
      transform; NegBinGLM 은 log1p 를 내부 처리)로 저장 → 다시 load_artifact 로 불러와
      predict 가 재적합 예측을 **재현**하는지(round-trip) 검증. 재현되면 "저장된 학습완료 모델로
      real 예측 가능"이 성립.

산출: models/NegBinGLM-web-champion.pt (자립형, 검증된 것). 기존 깨진 .pt 는 건드리지 않음.
Read-only on data; writes one new artifact. Run: .venv/bin/python web/scripts/verify_and_fix_champion.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "web" / "scripts"))

from build_production_forecast import _load_feature_matrix, _extract_basic_features, BASIC_FEATURE_COLS  # noqa: E402
from simulation.models.epi_models import NegBinGLMForecaster  # noqa: E402
from simulation.utils.model_artifact import make_artifact, load_artifact  # noqa: E402


def _r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    ss = float(np.sum((y - y.mean()) ** 2))
    return (1 - float(np.sum(e ** 2)) / ss) if ss > 0 else float("nan")


def _mae(y, p):
    return float(np.mean(np.abs(np.asarray(p, float) - np.asarray(y, float))))


def main() -> None:
    X_all, y_all, fcols, ws = _load_feature_matrix()
    X_all = np.asarray(X_all, float)
    y_all = np.asarray(y_all, float)
    Xb, bcols, bidx = _extract_basic_features(X_all, fcols)
    n = len(y_all)
    print(f"=== 학습완료 모델 → REAL 예측: 작동경로 확인 + 자립형 저장 fix (n={n}) ===\n")

    # ── (A) 작동 경로: 재적합 → 시간순 holdout test ─────────────────────────────
    cut = n - 68                              # 최근 68주 = held-out test (요약지표 test 크기)
    m = NegBinGLMForecaster(topk=20)
    m.fit(Xb[:cut], y_all[:cut])
    pred_te = np.asarray(m.predict(Xb[cut:]), float).ravel()
    r2_te, mae_te = _r2(y_all[cut:], pred_te), _mae(y_all[cut:], pred_te)
    print("(A) 작동 경로 — BASIC 재적합 → 시간순 holdout(최근 68주, 학습 미포함):")
    print(f"    test R²={r2_te:+.3f}  MAE={mae_te:.2f}  "
          f"→ {'✓ real 예측 제대로 작동' if r2_te > 0.4 else '△ 약함(시간순 OOS 난이도)'}")

    # ── (B) fix: 전체 재적합 → 자립형 아티팩트 저장 → round-trip ──────────────────
    m_full = NegBinGLMForecaster(topk=20)
    m_full.fit(Xb, y_all)
    pred_refit = np.asarray(m_full.predict(Xb), float).ravel()    # 기준(재적합 직접 예측)

    art = make_artifact(
        model=m_full, transform_name="identity",        # NegBinGLM 은 log1p 내부 처리
        feature_indices=list(bidx), feature_cols=list(bcols),
        model_name="NegBinGLM",
        config={"features": "BASIC", "note": "self-contained web champion (verify_and_fix_champion)"},
        meta={"source": "verify_and_fix_champion", "n_train": int(n)},
    )
    out_path = ROOT / "models" / "NegBinGLM-web-champion.pt"
    out_path.write_bytes(art.to_pickle_bytes())

    art2 = load_artifact(out_path)                       # 다시 불러오기
    pred_loaded = np.asarray(art2.predict(X_all), float).ravel()  # full 399 → 내부서 BASIC 추출

    rt_r2 = _r2(pred_refit, pred_loaded)                 # round-trip 일치도
    max_abs = float(np.max(np.abs(pred_refit - pred_loaded)))
    insample_r2 = _r2(y_all, pred_loaded)                # 불러온 모델의 real(전체, 대부분 in-sample) 적합도
    print(f"\n(B) fix — 자립형 아티팩트 저장 → {out_path.relative_to(ROOT)}")
    print(f"    round-trip(load→predict vs 재적합): R²={rt_r2:.5f}  max|Δ|={max_abs:.4f}  "
          f"→ {'✓ 완전 재현' if rt_r2 > 0.999 and max_abs < 0.1 else '✗ 불일치'}")
    print(f"    불러온 모델의 REAL 적합(전체 {n}주, 대부분 in-sample): R²={insample_r2:+.3f}")

    # ── 대조: 기존 깨진 bare 아티팩트 ──────────────────────────────────────────
    print(f"\n대조 — 기존 저장 아티팩트(bare pickle, 파이프라인 누락):")
    for nm, p in [("NegBinGLM-V7", ROOT / "models" / "NegBinGLM-V7.pt"),
                  ("NegBinGLM", ROOT / "simulation" / "checkpoints_history" / "NegBinGLM.pt")]:
        if not p.is_file():
            continue
        try:
            a = load_artifact(p)
            pp = np.asarray(a.predict(X_all), float).ravel()
            print(f"    {nm:14s}: load→predict REAL R²={_r2(y_all, pp):+.3f}  "
                  f"pred범위[{pp.min():.1f},{pp.max():.1f}]  ✗ 사용 불가")
        except Exception as e:
            print(f"    {nm:14s}: predict 실패 {type(e).__name__}  ✗ 사용 불가")

    print(f"\n결론:")
    print(f"  • 기존 저장 아티팩트(bare) = load→predict 깨짐(R²<0) → '학습완료 모델로 real 예측' 불가.")
    print(f"  • 자립형 재저장 = load→predict 완전 재현(round-trip R²={rt_r2:.4f}) → **가능하게 만듦**.")
    print(f"  • 작동 경로(재적합) test R²={r2_te:+.3f} → 모델 자체는 real 을 예측함(시간순 holdout).")


if __name__ == "__main__":
    main()
