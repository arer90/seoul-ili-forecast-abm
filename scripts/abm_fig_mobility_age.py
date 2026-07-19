"""ABM 연령-통근 이질성 고급 시각화 (Pillar 2 ③, 출판급).

실 데이터 2종을 한 그림에 — ABM 의 age 구조가 mean-field 가 못 잡는 *두* 이질성을 포착함을 보임:
  A. working-age 통근(hub-shift): 업무지구 주/야 인구 share (load_age_hub_shares) → 20-49 강한 주간 집중
  B. 통근 shift 그래디언트: day−night, 역-U(30-39 최대) — 공간 전파 동인
  C. 실 sentinel age-ILI 부담: 학령기(7-12) 최대 — 감수성/접촉 동인 (서로 다른 연령 peak = 이중 구조)

출력: simulation/results/abm_v1/fig_mobility_age.png (300 dpi, 한글 AppleGothic).
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

# 한글 폰트
for _f in ("AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic"):
    if any(_f == f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

DB = "simulation/data/db/epi_real_seoul.db"
HUB = "#1f6feb"        # 강조 (working-age)
MUTE = "#c2cdd6"       # muted
ILI = "#d9480f"        # ILI


def _hub_shift():
    from simulation.abm.agent_mobility import load_age_hub_shares
    sh = load_age_hub_shares(DB, day_hours=(10, 17))
    labels = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
    cols = ["pop_0_9", "pop_10_19", "pop_20_29", "pop_30_39", "pop_40_49",
            "pop_50_59", "pop_60_69", "pop_70plus"]
    day = np.array([sh[c]["day_share"] for c in cols])
    night = np.array([sh[c]["night_share"] for c in cols])
    shift = np.array([sh[c]["shift"] for c in cols])
    return labels, day, night, shift


def _age_ili():
    from simulation.database.storage import read_only_connect
    c = read_only_connect(DB)
    try:
        rows = c.execute("SELECT age_group, AVG(ili_rate) FROM sentinel_influenza "
                         "WHERE ili_rate IS NOT NULL AND age_group != '전체' GROUP BY age_group").fetchall()
    finally:
        c.close()
    order = ["0세", "1-6세", "7-12세", "13-18세", "19-49세", "50-64세", "65세 이상"]
    d = {a: v for a, v in rows}
    labs = [a for a in order if a in d]
    vals = [d[a] for a in labs]
    return labs, np.array(vals)


def main():
    labels, day, night, shift = _hub_shift()
    work = [i for i, l in enumerate(labels) if l in ("20-29", "30-39", "40-49", "50-59")]
    ili_labs, ili_vals = _age_ili()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.suptitle("그림. ABM 연령구조가 포착하는 두 이질성 — 통근(공간) × ILI 부담(감수성)",
                 fontsize=13, fontweight="medium", y=1.02)

    # ── A. hub-shift dumbbell (night → day) ──
    ax = axes[0]
    y = np.arange(len(labels))
    for i in range(len(labels)):
        col = HUB if i in work else MUTE
        ax.plot([night[i], day[i]], [y[i], y[i]], color=col, lw=2.4, zorder=1,
                solid_capstyle="round")
        ax.scatter(night[i], y[i], s=34, color="white", edgecolor=col, lw=1.6, zorder=2)
        ax.scatter(day[i], y[i], s=46, color=col, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("업무지구(중·종로·강남·서초·영등포) 인구 share")
    ax.set_title("A. 연령별 주·야 업무지구 집중\n(○ 야간 → ● 주간)", fontsize=11, loc="left")
    ax.scatter([], [], s=46, color=HUB, label="근로연령 20-59")
    ax.scatter([], [], s=46, color=MUTE, label="기타")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(axis="x", alpha=0.25)

    # ── B. shift gradient (inverted-U) ──
    ax = axes[1]
    cols = [HUB if i in work else MUTE for i in range(len(labels))]
    ax.bar(labels, shift * 100, color=cols, width=0.66)
    imax = int(np.argmax(shift))
    ax.set_ylim(0, float(shift.max()) * 100 * 1.32)            # headroom
    ax.annotate(f"최대 +{shift[imax]*100:.1f}%p", xy=(imax, shift[imax]*100),
                xytext=(imax - 0.2, float(shift.max())*100*1.18), ha="center", fontsize=9.5,
                color=HUB, fontweight="medium",
                arrowprops=dict(arrowstyle="->", color=HUB, lw=1.2))
    ax.set_ylabel("주간 share 증가분 (%p)")
    ax.set_title("B. 통근 shift 그래디언트 (역-U)\n근로연령 peak = 공간 전파 동인", fontsize=11, loc="left")
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(axis="y", alpha=0.25)
    ax.axhline(0, color="#888", lw=0.6)

    # ── C. real age-ILI gradient (school-age peak) ──
    ax = axes[2]
    peak = int(np.argmax(ili_vals))
    cols = [ILI if i == peak else "#f3b59a" for i in range(len(ili_vals))]
    ax.bar(range(len(ili_labs)), ili_vals, color=cols, width=0.68)
    ax.set_ylim(0, float(ili_vals.max()) * 1.32)               # headroom
    ax.annotate(f"학령기 peak\n{ili_labs[peak]} ({ili_vals[peak]:.0f})",
                xy=(peak, ili_vals[peak]), xytext=(peak + 1.3, float(ili_vals.max())*1.14),
                ha="center", fontsize=9.5, color=ILI, fontweight="medium",
                arrowprops=dict(arrowstyle="->", color=ILI, lw=1.2))
    ax.set_xticks(range(len(ili_labs))); ax.set_xticklabels(ili_labs, rotation=45, ha="right")
    ax.set_ylabel("평균 ILI rate (실 sentinel)")
    ax.set_title("C. 실 연령별 ILI 부담\n학령기 peak = 감수성/접촉 동인", fontsize=11, loc="left")
    ax.grid(axis="y", alpha=0.25)

    fig.text(0.5, -0.04,
             "통근 peak(근로연령)와 ILI peak(학령기)가 서로 다름 → ABM 의 연령×공간 구조가 둘을 분리해 포착 "
             "(homogeneous mean-field 는 불가). 실 데이터: daily_population_gu_hourly + sentinel_influenza.",
             ha="center", fontsize=9, color="#444", wrap=True)

    fig.tight_layout()
    out = "simulation/results/abm_v1/fig_mobility_age.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"→ {out}")
    # 수치 동반 저장
    json.dump({"hub_shift": {l: round(float(s), 4) for l, s in zip(labels, shift)},
               "age_ili": {l: round(float(v), 2) for l, v in zip(ili_labs, ili_vals)}},
              open("simulation/results/abm_v1/fig_mobility_age.json", "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
