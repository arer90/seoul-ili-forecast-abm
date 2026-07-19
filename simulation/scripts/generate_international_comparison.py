"""
simulation/scripts/generate_international_comparison.py
========================================================
국제 ILI 데이터 비교 분석 — 모델 평가 보조

목적:
  - 한국(KR) 모델 예측 결과를 미국(US), 일본(JP), 유럽 5개국(DE/FR/GB/NL/SE)
    실제 ILI 추이와 비교 → 계절성 패턴 · 피크 시기 일치도 분석
  - 국가별 ILI positivity 기반 상관계수 + 리드-래그 분석
  - HTML + CSV 보고서 출력

사용:
  python -m simulation.scripts.generate_international_comparison [--output-dir DIR]

출력:
  simulation/results/international_comparison.html
  simulation/results/international_comparison.csv

데이터 소스 (overseas_ili 테이블):
  US: CDC ILINet (2015-2025) + WHO FluNet (2015-2026)
  JP: Japan JIHS (2025-2026) + WHO FluNet (2015-2026)
  DE/FR/GB/NL/SE: WHO FluNet (2010-2026) — positivity proxy
  KR: WHO FluNet (2015-2026) — positivity proxy
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from simulation.database import safe_connect

log = logging.getLogger(__name__)

# ─── 국가 메타데이터 ────────────────────────────────────────────────────────
COUNTRY_META = {
    "US": {"label": "미국 (CDC ILINet)",    "region": "Americas",  "color": "#1f77b4"},
    "JP": {"label": "일본 (JIHS/FluNet)",   "region": "WPR",       "color": "#d62728"},
    "KR": {"label": "한국 (WHO FluNet)",    "region": "WPR",       "color": "#2ca02c"},
    "DE": {"label": "독일 (WHO FluNet)",    "region": "EUR",       "color": "#ff7f0e"},
    "FR": {"label": "프랑스 (WHO FluNet)",  "region": "EUR",       "color": "#9467bd"},
    "GB": {"label": "영국·잉글랜드 (WHO)", "region": "EUR",       "color": "#8c564b"},
    "NL": {"label": "네덜란드 (WHO FluNet)","region": "EUR",       "color": "#e377c2"},
    "SE": {"label": "스웨덴 (WHO FluNet)",  "region": "EUR",       "color": "#7f7f7f"},
}
TARGET_COUNTRIES = list(COUNTRY_META.keys())
ANALYSIS_START_YEAR = 2015


# ─── 데이터 로드 ────────────────────────────────────────────────────────────
def _load_overseas_ili(db_path: str, start_year: int = ANALYSIS_START_YEAR) -> dict:
    """overseas_ili 테이블에서 국가별 주간 ILI positivity 로드.

    Args:
        db_path: epi_real_seoul.db 경로
        start_year: 분석 시작 연도 (기본 2015)

    Returns:
        {country: [(year, week_no, ili_rate), ...]} dict
    """
    con = safe_connect(db_path)
    cur = con.cursor()

    data = {}
    for country in TARGET_COUNTRIES:
        # US는 CDC ILINet 우선 (실제 ILI %)
        if country == "US":
            cur.execute('''
                SELECT year, week_no, ili_rate FROM overseas_ili
                WHERE country = ? AND source = 'cdc_ilinet' AND year >= ?
                  AND ili_rate IS NOT NULL AND ili_rate > 0
                ORDER BY year, week_no
            ''', (country, start_year))
        else:
            # 그 외: who_flunet positivity (INF/spec)
            cur.execute('''
                SELECT year, week_no, ili_rate FROM overseas_ili
                WHERE country = ? AND source = 'who_flunet' AND year >= ?
                  AND ili_rate IS NOT NULL AND ili_rate > 0
                ORDER BY year, week_no
            ''', (country, start_year))
        rows = cur.fetchall()
        if rows:
            data[country] = rows
            log.info(f"  {country}: {len(rows)}주 로드 ({rows[0][0]}W{rows[0][1]} ~ {rows[-1][0]}W{rows[-1][1]})")
        else:
            log.warning(f"  {country}: 데이터 없음")

    con.close()
    return data


def _rows_to_array(rows: list, normalize: bool = True) -> tuple[list, list]:
    """(year, week, value) 리스트 → (time_labels, values) 변환.

    Args:
        rows: [(year, week_no, value), ...]
        normalize: True이면 0-1 min-max 정규화 (국가 간 scale 차이 보정)

    Returns:
        (time_labels, values) where time_labels = ['YYYY-WXX', ...]
    """
    labels = [f"{r[0]}-W{r[1]:02d}" for r in rows]
    vals = np.array([r[2] for r in rows], dtype=float)
    if normalize and vals.std() > 0:
        vals = (vals - vals.min()) / (vals.max() - vals.min())
    return labels, vals


# ─── 상관 분석 ──────────────────────────────────────────────────────────────
def _compute_correlations(data: dict) -> list[dict]:
    """KR vs 각 국가 간 Pearson 상관 + 최적 lag (±8주 탐색).

    Returns:
        list of {country, n_overlap, pearson_r, pearson_p, best_lag_weeks, lagged_r}
    """
    from scipy.stats import pearsonr

    if "KR" not in data:
        return []

    kr_rows = data["KR"]
    kr_map = {(r[0], r[1]): r[2] for r in kr_rows}

    results = []
    for country, rows in data.items():
        if country == "KR":
            continue
        oc_map = {(r[0], r[1]): r[2] for r in rows}

        # 공통 기간 정렬
        common_keys = sorted(set(kr_map) & set(oc_map))
        if len(common_keys) < 20:
            continue

        kr_vals = np.array([kr_map[k] for k in common_keys])
        oc_vals = np.array([oc_map[k] for k in common_keys])

        # 동시 상관
        try:
            r, p = pearsonr(kr_vals, oc_vals)
        except Exception:
            r, p = np.nan, np.nan

        # 최적 lag (±8주)
        best_lag, best_r = 0, r
        for lag in range(-8, 9):
            if lag == 0:
                continue
            if lag > 0:
                kr_s = kr_vals[lag:]
                oc_s = oc_vals[:-lag]
            else:
                kr_s = kr_vals[:lag]
                oc_s = oc_vals[-lag:]
            if len(kr_s) < 20:
                continue
            try:
                lr, _ = pearsonr(kr_s, oc_s)
                if abs(lr) > abs(best_r):
                    best_r = lr
                    best_lag = lag
            except Exception:
                pass

        results.append({
            "country": country,
            "label": COUNTRY_META[country]["label"],
            "region": COUNTRY_META[country]["region"],
            "n_overlap": len(common_keys),
            "pearson_r": round(r, 4),
            "pearson_p": round(p, 6),
            "best_lag_weeks": best_lag,
            "lagged_r": round(best_r, 4),
        })
        log.info(f"  KR~{country}: r={r:.3f}, best_lag={best_lag}w, lagged_r={best_r:.3f}")

    return sorted(results, key=lambda x: -abs(x["pearson_r"]))


# ─── 피크 시기 분석 ─────────────────────────────────────────────────────────
def _peak_analysis(data: dict) -> list[dict]:
    """국가별 연도별 피크 주차 추출 + KR 피크와의 차이.

    Returns:
        list of {country, season, peak_week, kr_peak_week, diff_weeks}
    """
    # 계절(시즌) 단위: 7월~이듬해 6월
    def get_season(year: int, week: int) -> str:
        if week >= 27:
            return f"{year}/{year+1}"
        else:
            return f"{year-1}/{year}"

    season_data: dict[str, dict[str, list]] = {}
    for country, rows in data.items():
        season_data[country] = {}
        for year, week_no, val in rows:
            season = get_season(year, week_no)
            season_data[country].setdefault(season, []).append((week_no, val))

    # 시즌별 피크 주차
    peak_by_country: dict[str, dict[str, int]] = {}
    for country, seasons in season_data.items():
        peak_by_country[country] = {}
        for season, pairs in seasons.items():
            if len(pairs) < 8:
                continue
            peak_week = max(pairs, key=lambda x: x[1])[0]
            peak_by_country[country][season] = peak_week

    kr_peaks = peak_by_country.get("KR", {})
    results = []
    for country in TARGET_COUNTRIES:
        if country == "KR":
            continue
        oc_peaks = peak_by_country.get(country, {})
        for season in sorted(set(kr_peaks) & set(oc_peaks)):
            diff = oc_peaks[season] - kr_peaks[season]
            results.append({
                "country": country,
                "label": COUNTRY_META[country]["label"],
                "season": season,
                "kr_peak_week": kr_peaks[season],
                "country_peak_week": oc_peaks[season],
                "diff_weeks": diff,
            })

    return results


# ─── HTML 보고서 생성 ────────────────────────────────────────────────────────
def _build_html(corr_results: list, peak_results: list, data: dict) -> str:
    """HTML 보고서 생성."""

    # 상관 테이블
    corr_rows_html = ""
    for r in corr_results:
        sig = "**" if r["pearson_p"] < 0.01 else ("*" if r["pearson_p"] < 0.05 else "")
        corr_rows_html += f"""
        <tr>
          <td>{r['label']}</td>
          <td>{r['region']}</td>
          <td>{r['n_overlap']}</td>
          <td class="{'good' if abs(r['pearson_r']) >= 0.5 else 'warn'}">{r['pearson_r']:.3f}{sig}</td>
          <td>{r['pearson_p']:.4f}</td>
          <td>{r['best_lag_weeks']:+d}주</td>
          <td>{r['lagged_r']:.3f}</td>
        </tr>"""

    # 피크 분석 요약 (최근 5시즌)
    recent_seasons = sorted({p["season"] for p in peak_results}, reverse=True)[:5]
    peak_html = ""
    for season in sorted(recent_seasons):
        season_row = ""
        for r in peak_results:
            if r["season"] == season:
                color = "good" if abs(r["diff_weeks"]) <= 2 else ("warn" if abs(r["diff_weeks"]) <= 4 else "bad")
                season_row += f"<td class='{color}'>{r['country']} W{r['country_peak_week']} ({r['diff_weeks']:+d})</td>"
        kr_peak = next((r["kr_peak_week"] for r in peak_results if r["season"] == season), "N/A")
        peak_html += f"<tr><td>{season}</td><td>KR W{kr_peak}</td>{season_row}</tr>"

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>국제 ILI 비교 분석</title>
<style>
  body {{ font-family: 'Malgun Gothic', sans-serif; margin: 20px; background: #f8f9fa; }}
  h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
  h2 {{ color: #34495e; margin-top: 30px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  th {{ background: #3498db; color: white; padding: 10px; text-align: left; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #eee; }}
  tr:hover {{ background: #f0f4f8; }}
  .good {{ color: #27ae60; font-weight: bold; }}
  .warn {{ color: #f39c12; font-weight: bold; }}
  .bad  {{ color: #e74c3c; font-weight: bold; }}
  .note {{ background: #eaf4fb; border-left: 4px solid #3498db; padding: 12px; margin: 10px 0; border-radius: 4px; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
  .card {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; }}
  .card .val {{ font-size: 2em; font-weight: bold; color: #3498db; }}
  .card .lbl {{ color: #7f8c8d; font-size: 0.9em; }}
</style>
</head>
<body>

<h1>🌍 국제 ILI 비교 분석 보고서</h1>
<p class="note">
  <strong>목적:</strong> 한국(KR) ILI positivity를 미국(US), 일본(JP), 유럽 5개국(DE/FR/GB/NL/SE)과 비교 분석.<br>
  <strong>데이터:</strong> overseas_ili 테이블 (CDC ILINet + WHO FluNet + Japan JIHS) | 분석 기간: {ANALYSIS_START_YEAR}~2026<br>
  <strong>주의:</strong> 국가별 ILI 정의 상이 (US = CLI%, KR/EU = influenza positivity) → 상대적 계절 패턴 비교만 유효
</p>

<div class="summary-grid">
  <div class="card"><div class="val">{len(TARGET_COUNTRIES)}</div><div class="lbl">비교 국가 수</div></div>
  <div class="card"><div class="val">{len([r for r in corr_results if abs(r['pearson_r'])>=0.5])}</div><div class="lbl">r≥0.5 국가 (KR과 강한 상관)</div></div>
  <div class="card"><div class="val">{len([r for r in corr_results if r['best_lag_weeks']==0])}</div><div class="lbl">동시 피크 국가 (lag=0)</div></div>
  <div class="card"><div class="val">{len(recent_seasons)}</div><div class="lbl">분석 시즌 수</div></div>
</div>

<h2>1. KR ILI vs 국제 ILI — Pearson 상관 분석 (2015-2026)</h2>
<p class="note">* p&lt;0.05, ** p&lt;0.01 | best_lag = KR 대비 해당 국가 리드(+)/래그(-) 주 수</p>
<table>
  <tr><th>국가</th><th>지역</th><th>공통 주수</th><th>Pearson r</th><th>p-value</th><th>최적 lag</th><th>lag 보정 r</th></tr>
  {corr_rows_html}
</table>

<h2>2. 연도별 피크 주차 비교 (최근 5시즌)</h2>
<p class="note">괄호 안 = KR 피크 대비 차이 (주). <span class="good">±2주 이내</span> / <span class="warn">±3-4주</span> / <span class="bad">±5주 이상</span></p>
<table>
  <tr><th>시즌</th><th>KR 피크</th><th colspan="7">국가별 피크 (KR 대비)</th></tr>
  {peak_html}
</table>

<h2>3. 데이터 가용성 요약</h2>
<table>
  <tr><th>국가</th><th>소스</th><th>행 수</th><th>지표</th></tr>
  {''.join(f"<tr><td>{COUNTRY_META.get(c, {}).get('label', c)}</td><td>{'CDC ILINet' if c == 'US' else 'WHO FluNet' + ('+JIHS' if c == 'JP' else '')}</td><td>{len(data.get(c, []))}주</td><td>{'ILI %(CLI)' if c == 'US' else 'Influenza positivity'}</td></tr>" for c in TARGET_COUNTRIES if c in data)}
</table>

<p style="color:#aaa;font-size:0.8em;margin-top:30px;">
  생성: simulation/scripts/generate_international_comparison.py |
  데이터: overseas_ili (epi_real_seoul.db) |
  문의: R9 (per_model_optimize) 평가 chain 연계
</p>
</body>
</html>"""


# ─── CSV 출력 ────────────────────────────────────────────────────────────────
def _write_csv(corr_results: list, peak_results: list, output_dir: Path) -> None:
    import csv

    # 상관 CSV
    corr_path = output_dir / "international_comparison_correlation.csv"
    with open(corr_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(corr_results[0].keys()) if corr_results else [])
        w.writeheader()
        w.writerows(corr_results)
    log.info(f"  상관 CSV → {corr_path}")

    # 피크 CSV
    peak_path = output_dir / "international_comparison_peaks.csv"
    with open(peak_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(peak_results[0].keys()) if peak_results else [])
        w.writeheader()
        w.writerows(peak_results)
    log.info(f"  피크 CSV → {peak_path}")


# ─── 진입점 ─────────────────────────────────────────────────────────────────
def main(db_path: Optional[str] = None, output_dir: Optional[str] = None) -> None:
    """국제 ILI 비교 분석 보고서 생성.

    Args:
        db_path: DB 경로 (None이면 프로젝트 기본 경로)
        output_dir: 출력 디렉토리 (None이면 simulation/results/)

    Side effects:
        simulation/results/international_comparison.html 생성
        simulation/results/international_comparison_correlation.csv 생성
        simulation/results/international_comparison_peaks.csv 생성
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # 경로 설정
    root = Path(__file__).parent.parent.parent
    db_path = db_path or str(root / "simulation/data/db/epi_real_seoul.db")
    from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
    out_dir = Path(output_dir) if output_dir else get_results_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== 국제 ILI 비교 분석 시작 ===")
    log.info(f"DB: {db_path}")

    # 1. 데이터 로드
    log.info("1. overseas_ili 데이터 로드")
    data = _load_overseas_ili(db_path)

    if "KR" not in data:
        log.error("KR 데이터 없음 — 분석 불가")
        return

    # 2. 상관 분석
    log.info("2. KR vs 국제 Pearson 상관 분석")
    try:
        corr_results = _compute_correlations(data)
    except ImportError:
        log.warning("scipy 없음 — 상관 분석 skip")
        corr_results = []

    # 3. 피크 분석
    log.info("3. 연도별 피크 주차 분석")
    peak_results = _peak_analysis(data)

    # 4. HTML 생성
    html = _build_html(corr_results, peak_results, data)
    html_path = out_dir / "international_comparison.html"
    html_path.write_text(html, encoding="utf-8")
    log.info(f"HTML → {html_path}")

    # 5. CSV 출력
    if corr_results and peak_results:
        _write_csv(corr_results, peak_results, out_dir)

    log.info("=== 완료 ===")
    print(f"\n✓ 보고서: {html_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="국제 ILI 비교 분석")
    parser.add_argument("--db", default=None, help="DB 경로")
    parser.add_argument("--output-dir", default=None, help="출력 디렉토리")
    args = parser.parse_args()
    main(db_path=args.db, output_dir=args.output_dir)
