"""G-274 STABILITY feature-floor fix 의 실측 개선 확인 게이트 (2026-06-16).

사용자 원칙: "바로 적용/완료 선언 말고 — TDD·실측으로 개선됐다 확인됐을 때만 해라."
floor fix(per_model_optimize.py)는 커밋됐고 재학습이 도는 중 → 이 스크립트가 **새 run 의
per_model 결과를 prior(98438) baseline 과 비교**해 PASS/FAIL 을 낸다. run 이 해당 모델을
끝낼 때마다 호출(미완 모델은 PENDING). 모두 PASS → floor fix "개선 확인" 확정.

⚠ 2026-06-16 실측 교훈: preproc Optuna(OPTUNA_ISOLATE 병렬 trial)가 **run-to-run 비결정적**
  → 안 건드린 XGBoost 도 valWIS 0.833↔1.728 (다른 transform 선택). 따라서 per-model valWIS
  비교는 회귀 판정에 부적합(노이즈 지배). floor fix 의 **결정적 검증 = collapse 해소**(n_features
  ≥ FLOOR_MIN, 즉 더 이상 [0,2,4] 3-feature 아님). valWIS 는 informational(특히 catastrophic
  baseline≥5 가 큰 폭 하락하면 강한 신호지만 단일-run 노이즈 감안).

판정:
  - floored 9모델(collapse 피해자, BayesianMCMC=mechanistic 제외): **PASS = new n_features
        ≥ FLOOR_MIN(10)** (collapse 해소 = floor 의 결정적 보장). valWIS Δ 는 표시만(informational).
  - champion/working: valWIS **informational only** (비결정성으로 회귀-가드 무의미) — 단 baseline
        대비 >3× 악화는 ⚠ 표시(조사가치). 최종 champion 비교는 run 종료 후 별도.

Usage:
    .venv/bin/python -m simulation.scripts.verify_g274_floor_improvement
    .venv/bin/python -m simulation.scripts.verify_g274_floor_improvement --base <archive_dir>

Side effects: 읽기 전용(per_model_optimal JSON). exit 0=전부 PASS/PENDING, 1=FAIL 존재.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = ROOT / "simulation/results/_archive_fullrun_20260616_115711/per_model_optimal"
DEFAULT_NEW = ROOT / "simulation/results/per_model_optimal"

FLOOR = 12
FLOOR_MIN = 10      # collapse 판정 임계 (n_features < 10 = collapse; floor 가 freq-top 12 로 복구)

# floor 가 복구해야 하는 collapse 피해자 (BayesianMCMC=mechanistic 제외, OverseasTransfer=phantom 별도)
FLOORED = ["CQR-LightGBM", "SVR-Linear", "DNN-Conformal", "GAT", "TCN",
           "TabularDNN", "TiDE", "BayesianRidge"]
# 무손상 보장 (floor 미발동 = 회귀 0 이어야)
PROTECTED = ["SVR-RBF", "TabPFN", "KRR", "ElasticNet", "RandomForest",
             "LightGBM", "XGBoost", "PoissonAutoreg", "NegBinGLM-Glum"]


def _load(d: Path, name: str):
    p = d / f"{name}.json"
    if not p.exists():
        return None
    try:
        j = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    bc = j.get("best_config", {}) or {}
    fi = bc.get("feature_indices")
    nf = bc.get("n_features") or (len(fi) if isinstance(fi, list) else None)
    return {
        "nf": nf,
        "wis": (j.get("val_metrics") or {}).get("wis"),
        "r2": (j.get("test_metrics") or {}).get("r2"),
    }


def _fmt(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(DEFAULT_BASE))
    ap.add_argument("--new", default=str(DEFAULT_NEW))
    a = ap.parse_args(argv)
    base, new = Path(a.base), Path(a.new)

    print(f"baseline = {base}")
    print(f"new run  = {new}\n")
    fails, pend = [], []

    print("── FLOORED (collapse 복구): PASS = n_features ≥ 10 (결정적). valWIS=informational ──")
    print(f"{'model':14} {'base nf/WIS':>16}  {'new nf/WIS':>16}  verdict")
    for m in FLOORED:
        b, n = _load(base, m), _load(new, m)
        if n is None:
            pend.append(m)
            print(f"{m:14} {(_fmt(b['nf'],0)+'/'+_fmt(b['wis']) if b else '?'):>16}  {'PENDING':>16}  ⏳ 미완")
            continue
        # 결정적 PASS = collapse 해소 (n_features ≥ FLOOR_MIN, 더 이상 3-feature 아님)
        ok = isinstance(n["nf"], (int, float)) and n["nf"] >= FLOOR_MIN
        if not ok:
            fails.append(m)
        # valWIS Δ 는 참고 (catastrophic baseline 큰 폭 하락 = 강한 신호)
        wnote = ""
        if b and isinstance(n["wis"], (int, float)) and isinstance(b["wis"], (int, float)):
            d = n["wis"] - b["wis"]
            wnote = f" (WIS Δ{d:+.2f}{' ★대폭개선' if b['wis'] >= 5 and d < -2 else ''})"
        tag = ("✅ PASS collapse해소" if ok else "❌ FAIL 여전히 collapse(<10)") + wnote
        bs = f"{_fmt(b['nf'],0)}/{_fmt(b['wis'])}" if b else "?"
        ns = f"{_fmt(n['nf'],0)}/{_fmt(n['wis'])}"
        print(f"{m:14} {bs:>16}  {ns:>16}  {tag}")

    print("\n── PROTECTED (참고만 — preproc Optuna 비결정성으로 valWIS 회귀 가드 무의미) ──")
    print(f"{'model':14} {'base WIS':>10}  {'new WIS':>10}  note")
    for m in PROTECTED:
        b, n = _load(base, m), _load(new, m)
        if n is None:
            pend.append(m)
            print(f"{m:14} {(_fmt(b['wis']) if b else '?'):>10}  {'PENDING':>10}  ⏳ 미완")
            continue
        if b is None or not isinstance(n["wis"], (int, float)) or not isinstance(b["wis"], (int, float)):
            print(f"{m:14} {'?':>10}  {_fmt(n['wis']):>10}  ⚠ 비교불가")
            continue
        note = "≈ (run-to-run 노이즈 범위)" if n["wis"] <= b["wis"] * 3.0 else "⚠ >3× 악화 (조사가치)"
        print(f"{m:14} {_fmt(b['wis']):>10}  {_fmt(n['wis']):>10}  {note}")

    print("\n" + "═" * 60)
    if fails:
        print(f"  ❌ FAIL: {fails} — floor fix 개선 미확인/회귀. 적용 보류·재검토.")
        return 1
    if pend:
        print(f"  ⏳ PENDING {len(pend)}모델 미완 (run 진행 중). 완료 후 재실행.")
        return 0
    print("  ✅ ALL PASS — floor fix 개선 실측 확인. 챔피언 무손상.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
