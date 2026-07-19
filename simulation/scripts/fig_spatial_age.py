"""공간(자치구) + 연령×주 그림 2종 생성기 — 실 데이터 전용.

서울 감염병 시뮬레이션 논문/웹용 보조 그림 2종을 **실측·실산출 데이터만** 사용해
PNG 로 렌더링한다. 합성/가짜 데이터를 절대 만들지 않으며, 데이터가 없으면 정직히
skip + 로그 후 다음 그림으로 넘어간다. 모델-유래(관측 아님) 데이터는 제목/주석에
명시한다.

생성 그림:
    ① fig_spatial_gu_ranking.png
        자치구(25구)별 신호 ranking 막대(높은→낮은).
        SSOT = web/public/aggregates/latest-choropleth.json
        (KDCA sentinel ILI 는 전국 단위만 존재 → 구별 실측 ILI 부재. gu-weights.json
        ladder tier-3 = uniform 으로 구별 분배 안 함이 정직한 기본값.) 따라서 이 그림은
        **관측 per-gu ILI 가 아니라** "구별 법정감염병(2급) 부담"이라는 가장 가까운
        per-gu 실측 신호를 ranking 한다. 제목/주석에 명시.

    ② fig_age_week_heatmap.png
        연령군 × 주(week) ILI rate 히트맵 (x=주, y=연령군, 색=rate).
        SSOT = DB table sentinel_influenza (실측 표본감시, 7 연령군 × 주).
        연령별 유행 시기차를 시각화.

실행:
    .venv/bin/python -m simulation.scripts.fig_spatial_age

출력:
    simulation/results/figures/fig_spatial_gu_ranking.png
    simulation/results/figures/fig_age_week_heatmap.png
    (dpi=120, bbox_inches="tight")

설계 규율:
    - 실 데이터만 (DB read-only sqlite3 / 기존 JSON). 가짜/합성 금지.
    - matplotlib Agg backend + 한글폰트 (AppleGothic → NanumGothic fallback).
    - 결정성: 정렬·쿼리 ORDER BY 명시, 임의 seed 없음.
    - 데이터 없으면 skip + 로그 (가짜 생성 금지).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless 렌더링 (디스플레이 비종속)

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# 경로 SSOT (단일 DB / 단일 코드 루트 — ENGINEERING_PRINCIPLES.md §4 KISS)
# ---------------------------------------------------------------------------
_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parents[2]  # simulation/scripts/fig_spatial_age.py → repo root
DB_PATH = PROJECT_ROOT / "simulation" / "data" / "db" / "epi_real_seoul.db"
CHOROPLETH_JSON = PROJECT_ROOT / "web" / "public" / "aggregates" / "latest-choropleth.json"
GU_WEIGHTS_JSON = PROJECT_ROOT / "web" / "public" / "aggregates" / "gu-weights.json"
FIG_DIR = PROJECT_ROOT / "simulation" / "results" / "figures"

# Seoul 25 districts: Korean gu name -> romanized English (figure tick labels;
# DB values stay Korean for matching).
_GU_ENG: dict[str, str] = {
    "종로구": "Jongno", "중구": "Jung", "용산구": "Yongsan", "성동구": "Seongdong",
    "광진구": "Gwangjin", "동대문구": "Dongdaemun", "중랑구": "Jungnang",
    "성북구": "Seongbuk", "강북구": "Gangbuk", "도봉구": "Dobong", "노원구": "Nowon",
    "은평구": "Eunpyeong", "서대문구": "Seodaemun", "마포구": "Mapo",
    "양천구": "Yangcheon", "강서구": "Gangseo", "구로구": "Guro",
    "금천구": "Geumcheon", "영등포구": "Yeongdeungpo", "동작구": "Dongjak",
    "관악구": "Gwanak", "서초구": "Seocho", "강남구": "Gangnam", "송파구": "Songpa",
    "강동구": "Gangdong",
}


def _gu_eng(name: str) -> str:
    """Romanize a Seoul district name (append '-gu'); pass through if unknown."""
    base = _GU_ENG.get(str(name))
    return f"{base}-gu" if base else str(name)


def _setup_korean_font() -> str:
    """matplotlib 전역 한글폰트 설정 (AppleGothic → NanumGothic fallback).

    Returns:
        실제 적용된 폰트 패밀리 이름 (str). 한글폰트 미발견 시 "DejaVu Sans"
        (한글 깨짐 경고 로그 출력).

    Side effects: plt.rcParams["font.family"], ["axes.unicode_minus"] 전역 변경.
    """
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False  # 음수 부호 깨짐 방지
    print(f"[font] 적용 폰트 = DejaVu Sans")
    return "DejaVu Sans"


def _confirm_png(path: Path) -> bool:
    """PNG 가 실제로 생성되었고 비어있지 않은지 검증.

    Args:
        path: 검증할 PNG 절대경로.

    Returns:
        존재 + size>0 이면 True, 아니면 False (로그 출력).
    """
    if path.exists() and path.stat().st_size > 0:
        print(f"[OK]   {path}  ({path.stat().st_size:,} bytes)")
        return True
    print(f"[FAIL] {path}  (미생성 또는 0 bytes)")
    return False


# ---------------------------------------------------------------------------
# ① 자치구 ranking 막대
# ---------------------------------------------------------------------------
def make_gu_ranking() -> Path | None:
    """자치구별 per-gu 실측 신호 ranking 막대 그림 생성.

    데이터 SSOT = web/public/aggregates/latest-choropleth.json
    (KDCA sentinel ILI = 전국 단위 → 구별 실측 ILI 부재; 가장 가까운 per-gu 실측
    신호 = 법정감염병 2급 부담 cases). gu-weights.json ladder 의 정직성 단계도
    제목 주석에 반영. 관측 per-gu ILI 인 척 하지 않는다.

    Returns:
        생성된 PNG 경로. 데이터 없으면 None (skip + 로그).

    Side effects: FIG_DIR 에 PNG write.
    """
    if not CHOROPLETH_JSON.exists():
        print(f"[SKIP] ① gu ranking — 데이터 없음: {CHOROPLETH_JSON}")
        return None

    payload = json.loads(CHOROPLETH_JSON.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    metric = payload.get("metric", "per-gu signal")
    disclaimer = payload.get("disclaimer", "")
    # '서울시' 등 집계행 제거 + cases 결측 제거
    clean = [
        (r["gu_nm"], r["cases"])
        for r in rows
        if r.get("gu_nm") and r.get("gu_nm") != "서울시" and r.get("cases") is not None
    ]
    if not clean:
        print("[SKIP] ① gu ranking — 유효 per-gu 행 없음")
        return None

    # 결정성: cases 내림차순, 동률은 gu_nm 사전순 (안정 정렬)
    clean.sort(key=lambda kv: (-kv[1], kv[0]))
    gu_names = [c[0] for c in clean]
    cases = [c[1] for c in clean]

    fig, ax = plt.subplots(figsize=(11, 8))
    y_pos = np.arange(len(gu_names))
    # 색: 높은 값=진한 빨강 → 낮은 값=연한, 순위 인지용 (관측 신호 강도)
    norm = np.array(cases, dtype=float)
    norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-9)
    colors = plt.cm.Reds(0.30 + 0.65 * norm)
    bars = ax.barh(y_pos, cases, color=colors, edgecolor="#7f1d1d", linewidth=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([_gu_eng(g) for g in gu_names], fontsize=10)
    ax.invert_yaxis()  # 1위(최댓값)를 맨 위로
    ax.set_xlabel("Group-2 notifiable disease case count (cases)", fontsize=11)

    for bar, v in zip(bars, cases):
        ax.text(
            bar.get_width() + max(cases) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{v:,}",
            va="center",
            fontsize=8,
            color="#374151",
        )

    ax.set_title(
        "Seoul per-district observed-signal ranking (high→low)\n"
        "[not model-derived = observed] metric: " + metric,
        fontsize=12,
        pad=12,
    )
    # 정직성 주석: 이것이 관측 ILI 가 아님을 명시
    note = (
        "Note: KDCA sentinel ILI exists only at the national level, so there is no per-district observed ILI.\n"
        "This figure ranks the closest per-district observed signal, the 'Group-2 notifiable disease burden', and\n"
        "is not the ILI rate. (gu-weights.json: per-district ILI allocation is not statistically significant, treated uniformly)"
    )
    if disclaimer:
        note += f"\nsource note: {disclaimer}"
    fig.text(
        0.01,
        -0.02,
        note,
        fontsize=8,
        color="#6b7280",
        ha="left",
        va="top",
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_spatial_gu_ranking.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# ② 연령 × 주 ILI rate 히트맵
# ---------------------------------------------------------------------------
# 연령군 표시 순서 (어림→고령). 실 DB 의 7개 연령군과 정확히 일치해야 함.
_AGE_ORDER = ["0세", "1-6세", "7-12세", "13-18세", "19-49세", "50-64세", "65세 이상"]
# 표시용(y축 라벨) 영문 매핑 — _AGE_ORDER 는 DB age_group 값과 일치해야 하므로 변경 금지.
_AGE_DISPLAY = {
    "0세": "0 yr",
    "1-6세": "1-6 yr",
    "7-12세": "7-12 yr",
    "13-18세": "13-18 yr",
    "19-49세": "19-49 yr",
    "50-64세": "50-64 yr",
    "65세 이상": "65+ yr",
}


def make_age_week_heatmap() -> Path | None:
    """연령군 × 주(week) ILI rate 히트맵 생성 (실측 표본감시).

    데이터 SSOT = DB table sentinel_influenza (read-only). x=주(시즌 시작 36주 기준
    week_seq), y=연령군, 색=ili_rate. 여러 시즌이 있으면 가장 데이터가 완전한
    (주 수 최대) 단일 시즌을 결정적으로 선택해 시기차를 명확히 보여준다.

    Returns:
        생성된 PNG 경로. 데이터 없으면 None (skip + 로그).

    Side effects: FIG_DIR 에 PNG write. DB 는 read-only 로만 연다.
    """
    if not DB_PATH.exists():
        print(f"[SKIP] ② age×week heatmap — DB 없음: {DB_PATH}")
        return None

    # read-only 연결 (DB 무수정 보장; G-116/G-117 safe helper)
    from simulation.database import read_only_connect
    con = read_only_connect(str(DB_PATH))
    try:
        cur = con.cursor()
        cur.execute("SELECT COUNT(*) FROM sentinel_influenza")
        if cur.fetchone()[0] == 0:
            print("[SKIP] ② age×week heatmap — sentinel_influenza 비어있음")
            return None

        # 결정적 시즌 선택: 주 수(week_seq distinct) 최대 → 동률 시 최신 시즌
        cur.execute(
            """
            SELECT season_start, COUNT(DISTINCT week_seq) AS nweeks
            FROM sentinel_influenza
            GROUP BY season_start
            ORDER BY nweeks DESC, season_start DESC
            """
        )
        season_rows = cur.fetchall()
        if not season_rows:
            print("[SKIP] ② age×week heatmap — 시즌 없음")
            return None
        season = season_rows[0][0]

        # 해당 시즌의 (age_group, week_seq, week_label, ili_rate) 결정적 정렬 추출
        cur.execute(
            """
            SELECT age_group, week_seq, week_label, ili_rate
            FROM sentinel_influenza
            WHERE season_start = ? AND ili_rate IS NOT NULL
            ORDER BY week_seq ASC, age_group ASC
            """,
            (season,),
        )
        data = cur.fetchall()
    finally:
        con.close()

    if not data:
        print(f"[SKIP] ② age×week heatmap — 시즌 {season} 유효 행 없음")
        return None

    # DB 실 연령군과 표시 순서 정합 (DB 에 있는 것만, 표시순서대로)
    db_ages = {row[0] for row in data}
    ages = [a for a in _AGE_ORDER if a in db_ages]
    # _AGE_ORDER 에 없는 미지 연령군이 있으면 뒤에 정렬해 붙임 (정직성)
    for a in sorted(db_ages):
        if a not in ages:
            ages.append(a)

    week_seqs = sorted({row[1] for row in data})
    seq_to_idx = {s: i for i, s in enumerate(week_seqs)}
    age_to_idx = {a: i for i, a in enumerate(ages)}
    label_by_seq = {row[1]: row[2] for row in data}  # week_seq → week_label

    grid = np.full((len(ages), len(week_seqs)), np.nan, dtype=float)
    for age_group, week_seq, _label, rate in data:
        grid[age_to_idx[age_group], seq_to_idx[week_seq]] = rate

    fig, ax = plt.subplots(figsize=(13, 5.5))
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="#f3f4f6")  # 결측 = 연회색 (가짜로 채우지 않음)
    im = ax.imshow(grid, aspect="auto", cmap=cmap, origin="upper", interpolation="nearest")

    ax.set_yticks(np.arange(len(ages)))
    ax.set_yticklabels([_AGE_DISPLAY.get(a, a) for a in ages], fontsize=10)
    ax.set_ylabel("Age group", fontsize=11)

    # x 축: 시즌 시작(36주) 기준 약 8개 라벨만 표시 (가독성)
    n_ticks = min(10, len(week_seqs))
    tick_pos = np.linspace(0, len(week_seqs) - 1, n_ticks).astype(int)
    ax.set_xticks(tick_pos)

    def _wk_label_en(seq: int) -> str:
        # DB week_label is e.g. "36주" (Korean suffix); strip "주" for English tick.
        raw = label_by_seq.get(seq, str(seq))
        return str(raw).replace("주", "").strip() or str(seq)

    ax.set_xticklabels(
        [_wk_label_en(week_seqs[i]) for i in tick_pos],
        fontsize=9,
        rotation=0,
    )
    ax.set_xlabel(f"ISO week — {season} season (starts at week 36)", fontsize=11)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("ILI rate (sentinel surveillance, observed)", fontsize=10)

    ax.set_title(
        f"Age group × week ILI rate heatmap — {season} season [observed sentinel surveillance sentinel_influenza]\n"
        "Age-specific differences in epidemic timing and intensity (light=low, dark=high)",
        fontsize=12,
        pad=10,
    )
    fig.text(
        0.01,
        -0.04,
        "Data source: DB sentinel_influenza (KDCA influenza-like illness sentinel surveillance, observed). "
        "Gray cells = missing (unobserved, not fabricated).",
        fontsize=8,
        color="#6b7280",
        ha="left",
        va="top",
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_age_week_heatmap.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    """그림 2종 생성 + PNG size>0 검증. 데이터 없는 그림은 정직히 skip.

    Side effects: 한글폰트 전역 설정, FIG_DIR 에 PNG write, stdout 로그.
    """
    print("=" * 70)
    print("fig_spatial_age — 공간(자치구) + 연령×주 그림 2종 (실 데이터 전용)")
    print("=" * 70)
    _setup_korean_font()

    results: list[tuple[str, Path | None]] = []
    print("\n[①] 자치구 per-gu 신호 ranking ...")
    results.append(("fig_spatial_gu_ranking", make_gu_ranking()))
    print("\n[②] 연령군 × 주 ILI 히트맵 ...")
    results.append(("fig_age_week_heatmap", make_age_week_heatmap()))

    print("\n" + "=" * 70)
    print("검증:")
    n_ok = 0
    for name, path in results:
        if path is None:
            print(f"[SKIP] {name} — 데이터 부재로 미생성 (정직 skip)")
        elif _confirm_png(path):
            n_ok += 1
    print(f"\n생성 완료: {n_ok}/{len(results)} PNG (size>0)")
    print("=" * 70)


if __name__ == "__main__":
    main()
