"""gen_supplementary_figures.py — 전 모델 supplementary 학습 그래프 생성 (2026-06-26).

석사논문 supplementary 용. 각 모델당 2-panel figure:
  좌(학습곡선): DL=epoch loss(dl_epoch/lightning_epoch), non-DL=preproc/HP trial WIS 궤적.
  우(예측 fit): test 슬랩 y_true vs y_pred(=학습 데이터 적합).

입력: simulation/results/training_history/<model>_pooled.csv (record_type·step·metric·value)
      + simulation/results/csv/predictions_<model>.csv (split·y_true·y_pred).
출력: simulation/results/figures/supp_<model>.png (model-agnostic = 전 모델 루프).

Usage: .venv/bin/python -m simulation.scripts.gen_supplementary_figures
Returns: 생성 개수(print). Side effects: figures/*.png 작성. (모델 로드 없음 = 가벼움.)
"""
from __future__ import annotations

import glob
import os
import warnings

warnings.filterwarnings("ignore")


def _train_curve(df):
    """학습곡선 (steps, values, ylabel, kind). DL epoch loss 우선, 없으면 optuna trial WIS."""
    for rt, lbl in (("dl_epoch", "epoch"), ("lightning_epoch", "epoch")):
        sub = df[(df["record_type"] == rt) & (df["metric_name"] == "loss")]
        if len(sub) >= 3:
            sub = sub.sort_values("step")
            return sub["step"].values, sub["value"].values, "training loss", lbl
    # non-DL: preproc/HP trial WIS
    sub = df[(df["record_type"] == "optuna_trial") & (df["metric_name"] == "WIS")]
    if len(sub) >= 2:
        sub = sub.sort_values("step")
        return sub["step"].values, sub["value"].values, "OOF WIS", "trial"
    return None


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # 한글 폰트 (macOS AppleGothic → Linux NanumGothic fallback)
    for _f in ("AppleGothic", "NanumGothic", "Malgun Gothic", "DejaVu Sans"):
        try:
            from matplotlib import font_manager
            if any(_f == f.name for f in font_manager.fontManager.ttflist):
                plt.rcParams["font.family"] = _f
                break
        except Exception:
            pass
    plt.rcParams["axes.unicode_minus"] = False

    outdir = "simulation/results/figures"
    os.makedirs(outdir, exist_ok=True)
    hist_dir = "simulation/results/training_history"
    n_made = 0
    models = sorted(os.path.basename(p)[:-len("_pooled.csv")]
                    for p in glob.glob(f"{hist_dir}/*_pooled.csv"))
    for m in models:
        hp = f"{hist_dir}/{m}_pooled.csv"
        pp = f"simulation/results/csv/predictions_{m}.csv"
        try:
            fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
            # 좌: 학습곡선
            ax = axes[0]
            tc = None
            if os.path.exists(hp):
                tc = _train_curve(pd.read_csv(hp))
            if tc is not None:
                steps, vals, ylab, kind = tc
                ax.plot(steps, vals, marker="o" if kind == "trial" else None,
                        ms=3, lw=1.4, color="#185FA5")
                if kind == "trial":
                    best = np.argmin(vals)
                    ax.scatter([steps[best]], [vals[best]], color="#A32D2D", zorder=5, s=40,
                               label=f"best (trial {int(steps[best])})")
                    ax.legend(fontsize=8, frameon=False)
                ax.set_xlabel(kind); ax.set_ylabel(ylab)
                ax.set_title(f"{m} — 학습곡선 ({'epoch loss' if kind=='epoch' else 'preproc/HP trial'})",
                             fontsize=10)
            else:
                ax.text(0.5, 0.5, "학습곡선 데이터 없음\n(closed-form / 비반복 학습)",
                        ha="center", va="center", fontsize=9, color="#5F5E5A")
                ax.set_title(f"{m} — 학습곡선", fontsize=10)
            # 우: 예측 fit
            ax = axes[1]
            if os.path.exists(pp):
                d = pd.read_csv(pp)
                t = d[d["split"] == "test"].reset_index(drop=True)
                if len(t):
                    x = np.arange(len(t))
                    ax.plot(x, t["y_true"], lw=1.6, color="#2C2C2A", label="실측 ILI")
                    ax.plot(x, t["y_pred"], lw=1.4, color="#D85A30", ls="--", label="예측")
                    r2 = 1 - np.sum((t["y_true"]-t["y_pred"])**2) / np.sum((t["y_true"]-t["y_true"].mean())**2)
                    ax.set_title(f"{m} — test 예측 (rolling 1-step, R²={r2:.3f})", fontsize=10)
                    ax.set_xlabel("test week"); ax.set_ylabel("ILI rate")
                    ax.legend(fontsize=8, frameon=False)
            else:
                ax.text(0.5, 0.5, "예측 데이터 없음", ha="center", va="center", fontsize=9)
            plt.tight_layout()
            fig.savefig(f"{outdir}/supp_{m}.png", dpi=110, bbox_inches="tight")
            plt.close(fig)
            n_made += 1
        except Exception as e:
            print(f"  {m}: skip ({type(e).__name__}: {str(e)[:60]})", flush=True)
    print(f"\n  {n_made}/{len(models)} supplementary figures → {outdir}/supp_*.png", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
