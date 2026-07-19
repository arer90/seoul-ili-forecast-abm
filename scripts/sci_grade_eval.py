"""SCI-grade 표준 분석 — protocol-flip + DM·BH-FDR pairwise (read-only).

5-검토(wf_e73384a7) 로드맵 우선순위 4 + 방법론 발견 §2 의 정량 산출.
완료된 per_model_optimal/*.json + results/csv/predictions_*.csv 만 읽음 (라이브 run 무영향).

산출:
  ① Protocol-flip: Spearman(hold-out R², rolling R²) + sign-flip 수 + 양방향 표
  ② DM(Diebold-Mariano, HLN 소표본보정) 챔피언 vs 전 모델, test-slab 점예측 기준
  ③ BH-FDR 다중비교 보정 (SSOT = fdr_bh)
  → docs/sci_eval/ 에 CSV + 콘솔 요약. run 종료 후 full 재실행하면 53 전체.

run: KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/python -m scripts.sci_grade_eval   (또는 python scripts/sci_grade_eval.py)
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path("simulation/results")
OUT = Path("docs/sci_eval"); OUT.mkdir(parents=True, exist_ok=True)


def _load_models() -> dict:
    """per_model_optimal JSON → {model: {test_r2, real_r2, test_wis}}."""
    out = {}
    for jf in glob.glob(str(ROOT / "per_model_optimal" / "*.json")):
        m = os.path.basename(jf)[:-5]
        try:
            d = json.load(open(jf))
        except Exception:
            continue
        tm = d.get("test_metrics") or {}; rm = d.get("real_metrics") or {}
        out[m] = {
            "test_r2": tm.get("r2"), "real_r2": rm.get("r2"),
            "test_wis": tm.get("wis") or d.get("wis"),
        }
    return out


def _load_test_pred(model: str):
    """predictions_<model>.csv 의 test split → (y_true, y_pred) 정렬배열 or None."""
    pf = ROOT / "csv" / f"predictions_{model.replace(' ', '_').replace('/', '_')}.csv"
    if not pf.exists():
        return None
    yt, yp = [], []
    for r in csv.DictReader(open(pf)):
        if r.get("split") == "test":
            try:
                yt.append(float(r["y_true"])); yp.append(float(r["y_pred"]))
            except (KeyError, ValueError):
                pass
    if len(yt) < 10:
        return None
    return np.asarray(yt), np.asarray(yp)


def dm_test(y, p1, p2, h: int = 1, loss: str = "se"):
    """Diebold-Mariano (Harvey-Leybourne-Newbold 1997 소표본 보정).

    H0: 두 모델 예측오차 동등. dm>0 이면 p1 이 p2 보다 나쁨(loss 큼).
    Returns (dm_hln, two_sided_p). loss='se'(squared) | 'ae'(absolute).
    """
    from scipy.stats import t as tdist
    e1 = y - p1; e2 = y - p2
    d = (e1 ** 2 - e2 ** 2) if loss == "se" else (np.abs(e1) - np.abs(e2))
    n = len(d); dbar = float(d.mean())
    # h=1 → autocov 0차만. (다중-horizon DM 은 post-run rolling slab 에서)
    gamma0 = float(((d - dbar) ** 2).mean())
    if gamma0 <= 0 or n < 3:
        return float("nan"), float("nan")
    dm = dbar / math.sqrt(gamma0 / n)
    hln = math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * hln
    p = 2 * (1 - tdist.cdf(abs(dm_hln), df=n - 1))
    return float(dm_hln), float(p)


def main():
    from scipy.stats import spearmanr
    models = _load_models()
    print(f"[sci-eval] 완료 모델 {len(models)} 개 분석 (run 진행 중이면 부분; 종료 후 full)")

    # ── ① Protocol-flip ──
    pair = [(m, v["test_r2"], v["real_r2"]) for m, v in models.items()
            if isinstance(v["test_r2"], (int, float)) and isinstance(v["real_r2"], (int, float))]
    if len(pair) >= 5:
        t_arr = np.array([p[1] for p in pair]); r_arr = np.array([p[2] for p in pair])
        rho, pval = spearmanr(t_arr, r_arr)
        flips = [(m, t, r) for m, t, r in pair if (t >= 0) != (r >= 0)]
        up = [(m, t, r) for m, t, r in flips if t < 0 <= r]    # hold-out 음 → rolling 양
        down = [(m, t, r) for m, t, r in flips if t >= 0 > r]  # hold-out 양 → rolling 음
        print("\n═══ ① Protocol-flip (hold-out vs rolling R²) ═══")
        print(f"  Spearman ρ = {rho:.3f} (p={pval:.4f}), n={len(pair)}")
        print(f"  부호역전(sign-flip): {len(flips)}/{len(pair)}  (flip-up {len(up)}, flip-down {len(down)})")
        print(f"  flip-up(고전TS류, hold-out음→rolling양): {[m for m,_,_ in up]}")
        print(f"  flip-down(kernel/spline류, hold-out양→rolling음): {[m for m,_,_ in down]}")
        with open(OUT / "protocol_flip.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["model", "test_r2", "real_r2", "flip", "direction"])
            for m, t, r in sorted(pair, key=lambda x: -(x[1] or 0)):
                fl = (t >= 0) != (r >= 0)
                d = "up" if (t < 0 <= r) else ("down" if (t >= 0 > r) else "")
                w.writerow([m, f"{t:.4f}", f"{r:.4f}", fl, d])
        print(f"  → {OUT/'protocol_flip.csv'}")

    # ── ②③ DM + BH-FDR (챔피언 = 최저 test_wis, test-slab 점예측 기준) ──
    cand = [(m, v["test_wis"]) for m, v in models.items() if isinstance(v["test_wis"], (int, float))]
    champ = min(cand, key=lambda x: x[1])[0] if cand else None
    cp = _load_test_pred(champ) if champ else None
    print(f"\n═══ ②③ DM(HLN) + BH-FDR — 챔피언(최저 test_WIS) = {champ} ═══")
    if cp is None:
        print("  챔피언 예측 CSV 없음 — skip (run 진행 중)"); return
    y, p_champ = cp
    rows = []
    for m in models:
        if m == champ:
            continue
        other = _load_test_pred(m)
        if other is None or len(other[0]) != len(y):
            continue
        _, po = other
        dm, p = dm_test(y, p_champ, po, h=1, loss="se")   # dm<0 이면 챔피언이 나음
        rows.append((m, dm, p))
    if rows:
        from simulation.analytics.multiple_testing import adjust_pvalues
        ps = [r[2] for r in rows]
        try:
            adj = adjust_pvalues(ps, method="fdr_bh")
            adj = list(adj["adjusted"]) if isinstance(adj, dict) else list(adj)
        except Exception:
            adj = ps  # fallback
        merged = [(rows[i][0], rows[i][1], rows[i][2], adj[i]) for i in range(len(rows))]
        print(f"  {'vs 모델':16s} {'DM(HLN)':>9s} {'p':>8s} {'p_BHFDR':>9s} {'챔피언 우위?':>12s}")
        print("  " + "-" * 60)
        out_rows = []; n_sig = 0
        for m, dm, p, pa in sorted(merged, key=lambda z: z[1]):
            is_sig = (pa < 0.05 and dm < 0)
            n_sig += 1 if is_sig else 0
            sig = "✓ 유의" if is_sig else ("(역)" if dm > 0 else "ns")
            print(f"  {m:16s} {dm:9.2f} {p:8.4f} {pa:9.4f} {sig:>12s}")
            out_rows.append([m, f"{dm:.3f}", f"{p:.4f}", f"{pa:.4f}", sig])
        with open(OUT / f"dm_bhfdr_vs_{champ}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["vs_model", "DM_HLN", "p", "p_bhfdr", "champ_better"])
            w.writerows(out_rows)
        print(f"  → 챔피언이 BH-FDR p<0.05 로 유의하게 우월: {n_sig}/{len(rows)} 모델")
        print(f"  → {OUT/f'dm_bhfdr_vs_{champ}.csv'}")
    print("\n[주의] DM 은 현재 hold-out(test-slab) 점예측 기준 = horizon-confounded slab.")
    print("       protocol-matched rolling-origin DM 은 run 종료 후 multi-horizon 배선과 함께(로드맵 P2).")


if __name__ == "__main__":
    main()
