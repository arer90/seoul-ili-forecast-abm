"""fig_forecast_eval.py — forecast-evaluation figure 2종 (FluSight 스타일, 2026-06-26).

석사논문/SCI Figure 용. 실 데이터(per_model_eval CSV + predictions CSV)만 소비, 가짜/합성 0:

  Fig 1 — WIS decomposition 누적막대:
      simulation/results/per_model_eval/per_model_metrics.csv 의 WIS 분해 3성분
      (wis_sharpness=구간폭 / wis_underpred=과소예측 패널티 / wis_overpred=과대예측 패널티,
      합=wis_total_decomp) 을 rank_wis 상위 N 모델에 대해 누적막대로. WIS 낮을수록 우수.

  Fig 2 — forecast-vs-truth fan chart (챔피언 FusedEpi, fallback NegBinGLM):
      predictions_<champion>.csv 의 test 68주 y_true vs y_pred 점예측 +
      adaptive online conformal PI(adaptive_conformal.online_conformal_bounds)로
      0.5/0.8/0.95 3중 fan. PI 는 *점예측·과거관측만* 쓰는 leak-free rolling 구간(모델-유래).

엄수: 실 데이터만 · matplotlib Agg + 한글폰트(AppleGothic→NanumGothic fallback) ·
      데이터 없으면 정직히 skip+로그(가짜 생성 X) · 모델-유래 PI 는 제목/주석에 명시 ·
      출력 figures/<name>.png dpi=120 bbox_inches=tight · 결정성(정렬·고정 alpha).

Usage:
    .venv/bin/python -m simulation.scripts.fig_forecast_eval

Returns:
    생성된 PNG 경로 list (print). 0개면 정직히 skip 로그만.

Side effects:
    simulation/results/figures/fig_wis_decomposition.png · fig_forecast_fanchart_<champion>.png 작성.
    DB read-only(현재 미사용 — 산출 CSV 로 충분), 모델 로드 없음(가벼움).

Performance: O(n_models + n_test·K), 수 초.
"""
from __future__ import annotations

import csv
import os

# 결정성 (정렬/alpha 고정으로 충분하나 명시)
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RESULTS = os.path.join(_REPO, "simulation", "results")
_CSV_DIR = os.path.join(_RESULTS, "csv")
_FIG_DIR = os.path.join(_RESULTS, "figures")
_METRICS = os.path.join(_RESULTS, "per_model_eval", "per_model_metrics.csv")

# fan chart 3중 PI level (FluSight 스타일) → online_conformal_bounds alpha (miscoverage)
_PI_LEVELS = (0.50, 0.80, 0.95)
_PI_ALPHAS = (0.50, 0.20, 0.05)  # 1 - level
_CHAMPION_PREFERENCE = ("FusedEpi", "NegBinGLM")
_TOP_N_BARS = 10  # WIS 누적막대 상위 모델 수


def _set_korean_font(plt) -> None:
    """한글 폰트 설정 (macOS AppleGothic → Linux NanumGothic → fallback)."""
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False


def _to_float(x):
    """str→float, 비수치/공백/nan → None."""
    try:
        v = float(x)
        return v if v == v else None  # NaN 제거
    except (TypeError, ValueError):
        return None


def _load_wis_decomposition():
    """per_model_metrics.csv 에서 WIS 분해 3성분 로드 (rank_wis 오름차순).

    Returns:
        list[dict]: {model, sharpness, underpred, overpred, total, wis} —
        3성분 모두 유효(non-nan)한 모델만, rank_wis(없으면 wis) 오름차순 정렬.
        파일 부재/유효 모델 0개 → 빈 list (caller 가 skip).
    """
    if not os.path.exists(_METRICS):
        return []
    out = []
    with open(_METRICS, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sh = _to_float(r.get("wis_sharpness"))
            un = _to_float(r.get("wis_underpred"))
            ov = _to_float(r.get("wis_overpred"))
            if sh is None or un is None or ov is None:
                continue  # nan 모델(예: TiRex) 정직 제외
            tot = _to_float(r.get("wis_total_decomp"))
            out.append({
                "model": (r.get("model") or "").strip(),
                "sharpness": sh, "underpred": un, "overpred": ov,
                "total": tot if tot is not None else (sh + un + ov),
                "wis": _to_float(r.get("wis")),
                "rank": _to_float(r.get("rank_wis")),
            })

    def _key(d):
        return (d["rank"] if d["rank"] is not None else 1e9, d["total"])

    out.sort(key=_key)
    return out


def _load_predictions(model: str):
    """predictions_<model>.csv 의 test 슬랩 (idx, y_true, y_pred) 로드.

    Returns:
        (y_true, y_pred) float list (idx 오름차순), 또는 (None, None) — 파일부재/test행 0.
    """
    path = os.path.join(_CSV_DIR, f"predictions_{model}.csv")
    if not os.path.exists(path):
        return None, None
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r.get("split") or "").strip().lower() != "test":
                continue
            yt = _to_float(r.get("y_true"))
            yp = _to_float(r.get("y_pred"))
            ix = _to_float(r.get("idx"))
            if yt is None or yp is None:
                continue
            rows.append((ix if ix is not None else len(rows), yt, yp))
    if not rows:
        return None, None
    rows.sort(key=lambda t: t[0])
    return [t[1] for t in rows], [t[2] for t in rows]


def _make_wis_decomposition_fig(plt, np) -> str | None:
    """Fig 1 — 상위 N 모델 WIS 분해 누적 가로막대. Returns: 저장경로 또는 None(skip)."""
    data = _load_wis_decomposition()
    if not data:
        print("[skip] WIS decomposition: per_model_metrics.csv 부재 또는 유효 모델 0개")
        return None
    top = data[:_TOP_N_BARS]
    models = [d["model"] for d in top]
    sh = np.array([d["sharpness"] for d in top])
    un = np.array([d["underpred"] for d in top])
    ov = np.array([d["overpred"] for d in top])

    # 가로막대 = 위가 best(rank 1). y 위치 역순으로 두어 1위가 맨 위.
    y = np.arange(len(top))[::-1]
    fig, ax = plt.subplots(figsize=(9.5, max(3.5, 0.55 * len(top) + 1.5)))
    c_sh, c_un, c_ov = "#4C72B0", "#DD8452", "#C44E52"
    ax.barh(y, sh, color=c_sh, label="Sharpness (interval width)", edgecolor="white", linewidth=0.4)
    ax.barh(y, un, left=sh, color=c_un, label="Underprediction (penalty)",
            edgecolor="white", linewidth=0.4)
    ax.barh(y, ov, left=sh + un, color=c_ov, label="Overprediction (penalty)",
            edgecolor="white", linewidth=0.4)

    totals = sh + un + ov
    for yi, t in zip(y, totals):
        ax.text(t + totals.max() * 0.01, yi, f"{t:.2f}", va="center", ha="left",
                fontsize=8.5, color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=9)
    ax.set_xlabel("WIS (Weighted Interval Score) — lower is better", fontsize=10)
    ax.set_title("WIS decomposition (top %d models, hold-out test 68 weeks, model-derived)\n"
                 "WIS = Sharpness + Underprediction + Overprediction" % len(top),
                 fontsize=11.5, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_xlim(left=0)
    ax.margins(y=0.02)
    fig.tight_layout()

    out = os.path.join(_FIG_DIR, "fig_wis_decomposition.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def _make_fanchart_fig(plt, np) -> str | None:
    """Fig 2 — 챔피언 forecast-vs-truth fan chart + adaptive conformal 3중 PI.

    Returns: 저장경로 또는 None(skip). PI 는 모델-유래(점예측+과거관측 online conformal).
    """
    from simulation.analytics.adaptive_conformal import online_conformal_bounds

    champ = None
    y_true = y_pred = None
    for cand in _CHAMPION_PREFERENCE:
        y_true, y_pred = _load_predictions(cand)
        if y_true is not None:
            champ = cand
            break
    if champ is None:
        print("[skip] fan chart: predictions_{FusedEpi,NegBinGLM}.csv 모두 부재")
        return None

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    n = len(yt)
    if n < 4:
        print(f"[skip] fan chart: {champ} test 슬랩 n={n} (<4)")
        return None

    # adaptive online conformal — 점예측 + 과거관측만(leak-free), in-sample 잔차 불요.
    bounds = online_conformal_bounds(yp, yt, alphas=_PI_ALPHAS)

    x = np.arange(1, n + 1)  # test week index (1-based)
    fig, ax = plt.subplots(figsize=(11, 5.2))

    # fan: 넓은(0.95)→좁은(0.50) 순서로 그려 겹침 → 짙어짐
    fan_colors = {0.05: "#cfe0f5", 0.20: "#9dc0e8", 0.50: "#5a91cf"}
    for level, a in sorted(zip(_PI_LEVELS, _PI_ALPHAS), reverse=True):
        if a not in bounds:
            continue
        lo, hi = bounds[a]
        ax.fill_between(x, np.asarray(lo), np.asarray(hi), color=fan_colors[a],
                        alpha=0.9, linewidth=0, label=f"{int(level * 100)}% PI (adaptive conformal)")

    ax.plot(x, yp, color="#08306b", lw=1.8, label=f"{champ} point forecast (y_pred)", zorder=5)
    ax.plot(x, yt, color="#b30000", lw=0, marker="o", ms=4.0,
            label="observed ILI (y_true)", zorder=6)

    # 커버리지 주석 (모델-유래 PI 검증) — 95% PI 실측 피복
    if 0.05 in bounds:
        lo95, hi95 = bounds[0.05]
        cov95 = float(np.mean((yt >= np.asarray(lo95)) & (yt <= np.asarray(hi95))))
        ax.text(0.012, 0.965,
                f"95% PI empirical coverage = {cov95:.0%} (n={n})",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999999", alpha=0.85))

    ax.set_xlabel("hold-out test week (1-step rolling)", fontsize=10)
    ax.set_ylabel("ILI rate (Seoul)", fontsize=10)
    ax.set_title(f"Champion {champ} — forecast-vs-truth fan chart (hold-out test {n} weeks)\n"
                 "shaded = adaptive online-conformal prediction interval (model-derived, not observed; "
                 "point forecast + past observations, leak-free)",
                 fontsize=11.5, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8.5, ncol=2, framealpha=0.9,
              bbox_to_anchor=(0.0, 0.92))
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    out = os.path.join(_FIG_DIR, f"fig_forecast_fanchart_{champ}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> list:
    """figure 2종 생성. Returns: 생성된 PNG 경로 list (size>0 검증)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    _set_korean_font(plt)
    os.makedirs(_FIG_DIR, exist_ok=True)

    made = []
    for fn in (_make_wis_decomposition_fig, _make_fanchart_fig):
        path = fn(plt, np)
        if path and os.path.exists(path) and os.path.getsize(path) > 0:
            made.append(path)
            print(f"[ok] {path} ({os.path.getsize(path):,} bytes)")
        elif path:
            print(f"[fail] {path} — 0 byte 또는 미생성")

    if not made:
        print("[done] 생성된 figure 0개 (데이터 부재로 정직히 skip)")
    else:
        print(f"[done] {len(made)} figure 생성:")
        for p in made:
            print(f"   {p}")
    return made


if __name__ == "__main__":
    main()
