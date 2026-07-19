#!/usr/bin/env python3
"""alpha 안정화가 거짓폭주(seasonal)를 막으면서 진짜 surge(pandemic)를 따라가는가 — 실증 sweep.

사용자 핵심 질문: "alpha 하한(①)으로 거짓폭주(201)는 막지만, 팬데믹·갑작스러운 대규모 감염병은?"

V6 NegBinGLM(epi_models.py) = topK|Pearson| + StandardScaler + Ridge(log1p) + expm1, cap=2×max.
불안정 = RidgeCV(alphas logspace(-3,3,20), cv=3) 가 데이터마다 alpha 0.001~54.5 로 출렁 → 0.001
선택 시 log-space 과적합 → expm1 폭주(201). cap(2×max)이 막지만 그 cap 이 팬데믹도 죽임.

이 실험: alpha 를 고정 sweep 하며 cap 없이(raw) 두 거동 측정 —
  (A) 거짓폭주: 실제 겨울피크 주(real ILI 60~100)에서 예측이 real 대비 폭주하나(>1.5×)?
  (B) 팬데믹 추종: 합성 surge(lag1 = 50…400, doubling 구조)에서 예측이 surge 를 따라가나?
→ 둘 다 만족하는 alpha 가 있으면 ①이 팬데믹을 안 죽임. 없으면 trade-off 가 근본적이고
   팬데믹은 ML 이 아니라 기계론적 SEIR/ABM 엔진의 몫(2-engine 설계와 정합).

Read-only. Run: .venv/bin/python web/scripts/pandemic_vs_explosion_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "web" / "scripts"))
from build_production_forecast import _load_feature_matrix, _extract_basic_features  # noqa: E402


def fit_v6(X, y, alpha):
    """Replicate V6 (topK|Pearson| all-kept on BASIC + StandardScaler + Ridge(log1p)), forced alpha.

    Returns a predict(Xq, cap=None) closure. cap=None → raw (no 2×max clamp)."""
    sc = StandardScaler().fit(X)
    ylog = np.log1p(np.maximum(y, 0))
    mdl = Ridge(alpha=alpha).fit(sc.transform(X), ylog)
    ymax = float(np.max(y))

    def predict(Xq, cap=None):
        p = np.expm1(mdl.predict(sc.transform(Xq)))
        p = np.clip(p, 0.0, None)
        if cap is not None:
            p = np.clip(p, 0.0, cap * ymax)
        return p
    return predict, ymax


def main() -> None:
    X_all, y_all, fc, ws = _load_feature_matrix()
    X_all = np.asarray(X_all, float); y_all = np.asarray(y_all, float)
    Xb, bcols, _ = _extract_basic_features(X_all, fc)
    # BASIC col order: lag1,lag2,lag4,lag52, sin_month,cos_month, fourier..., season_idx
    i_lag1, i_lag2, i_lag4 = bcols.index("ili_rate_lag1"), bcols.index("ili_rate_lag2"), bcols.index("ili_rate_lag4")

    peak = np.where(y_all >= 60)[0]                     # 실제 겨울피크 주
    # 합성 pandemic surge: 실제 겨울피크 행을 템플릿으로, 최근 ILI 수준 L 을 sweep (doubling 구조)
    templ = Xb[int(peak[np.argmax(y_all[peak])])].copy()
    def surge_row(L):
        r = templ.copy(); r[i_lag1] = L; r[i_lag2] = L * 0.6; r[i_lag4] = L * 0.25
        return r
    L_sweep = [50, 80, 120, 160, 240, 320, 400]
    Xsurge = np.array([surge_row(L) for L in L_sweep])

    ymax = float(np.max(y_all))
    print(f"=== alpha 안정화 vs 팬데믹 추종 — 실증 sweep (학습 전체, cap 없이 raw) ===")
    print(f"  실제 ILI max={ymax:.1f}/1k · 겨울피크주(≥60) {len(peak)}개 · 합성 surge lag1∈{L_sweep}\n")
    print(f"  {'alpha':>7} | {'(A)거짓폭주: 피크 max예측':>24} | {'(B)팬데믹 추종: L=160→예측 / L=320→예측':>40}")
    print(f"  {'-'*7}-+-{'-'*24}-+-{'-'*40}")

    rows = []
    for alpha in [0.001, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0]:
        pred, _ = fit_v6(Xb, y_all, alpha)
        pk = pred(Xb[peak])                            # 실제 피크주 예측 (raw)
        real_pk = y_all[peak]
        max_over = float(np.max(pk / np.maximum(real_pk, 1)))   # 최악 과대배율
        explode = "✗ 폭주" if max_over > 1.5 else "✓ 안정"
        ps = pred(Xsurge)                               # 합성 surge 예측
        f160 = ps[L_sweep.index(160)]; f320 = ps[L_sweep.index(320)]
        # 추종: L 이 2배(160→320) 될 때 예측도 늘어나는가
        follows = "✓ 따라감" if f320 > f160 * 1.3 else ("△ 둔함" if f320 > f160 * 1.05 else "✗ 포화/수축")
        print(f"  {alpha:>7.3f} | 피크 max {pk.max():6.1f} (real {real_pk.max():.0f}) {max_over:4.1f}× {explode:>6} "
              f"| {f160:6.1f} → {f320:6.1f}  {follows}")
        rows.append((alpha, max_over, explode, f160, f320, follows))

    # 진단: seasonal-안정 + pandemic-추종 둘 다 만족하는 alpha 있나
    print(f"\n  ── 진단 ──")
    both = [r for r in rows if r[1] <= 1.5 and r[4] > r[3] * 1.3]
    season_ok = [r for r in rows if r[1] <= 1.5]
    print(f"  • seasonal 거짓폭주 안정(≤1.5×): alpha ≥ {min((r[0] for r in season_ok), default=None)}")
    if both:
        print(f"  • seasonal 안정 + pandemic 추종 둘 다: alpha ∈ {[r[0] for r in both]} → ① 가 팬데믹 안 죽임 ✓")
    else:
        print(f"  • 둘 다 만족하는 alpha 없음 → trade-off 근본적: 안정 alpha 는 surge 를 따라가도")
        print(f"    학습범위(max {ymax:.0f}) 밖 외삽이라 신뢰 불가 → 팬데믹은 기계론 SEIR/ABM 의 몫.")
    # surge 가 학습 max 를 넘는 구간에서의 예측 — 외삽 신뢰성
    pred1, _ = fit_v6(Xb, y_all, 3.0)
    ps3 = pred1(Xsurge)
    print(f"  • 참고(alpha=3, 안정): 합성 surge 예측 = {[f'{v:.0f}' for v in ps3]} (입력 L={L_sweep})")
    print(f"    → 학습 max {ymax:.0f} 부근에서 포화 경향. ML 은 본 적 없는 대규모 surge 를 외삽 못 함(본질).")


if __name__ == "__main__":
    main()
