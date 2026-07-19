"""
simulation/tests/test_r3_4_flunet_signal.py
============================================
R3-4 간이 테스트 — FluNet subtype share 가 Seoul ILI 에 *여분* 신호를 갖는지 검증.

질문: H1N1/H3N2/B share + positivity 가 계절·lag1 로 이미 설명되는 부분을 뺀
잔차(residual)에 상관이 있는가? 있다면 R3-4 full retrain (≈E급 4-5h) 가치 있음,
없다면 skip 가능.

게이트:
  - |r(raw)| ≥ 0.15 → 강한 marginal 신호
  - |r(residual)| ≥ 0.10 → full retrain 권장
  - |r(residual)| < 0.05 → 과잉, skip

접근: CPU-only, E 와 GPU 안 부딪힘. DB 는 safe_connect (WAL read concurrent).
실행: .venv/Scripts/python.exe -m simulation.tests.test_r3_4_flunet_signal
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "db" / "epi_real_seoul.db"


def _load_kr_flunet_weekly() -> pd.DataFrame:
    """KR FluNet → 주별 subtype share + positivity."""
    con = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql_query(
        """
        SELECT sdate, spec_processed,
               inf_a_h1n1pdm09, inf_a_h3, inf_a_notsubtyped, inf_a,
               inf_b_victoria, inf_b_yamagata, inf_b, inf_total
        FROM who_flunet
        WHERE country = 'Republic of Korea' AND sdate >= '2018-01-01'
        ORDER BY sdate
        """,
        con,
    )
    con.close()
    if df.empty:
        return df
    df["sdate"] = pd.to_datetime(df["sdate"])
    df["week_start"] = df["sdate"] - pd.to_timedelta(df["sdate"].dt.weekday, unit="d")
    for c in ["spec_processed", "inf_a_h1n1pdm09", "inf_a_h3", "inf_a_notsubtyped",
              "inf_a", "inf_b_victoria", "inf_b_yamagata", "inf_b", "inf_total"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    # share / positivity
    denom_total = df["inf_total"].replace(0, np.nan)
    denom_spec = df["spec_processed"].replace(0, np.nan)
    df["h3n2_share"] = (df["inf_a_h3"] / denom_total).fillna(0.0)
    df["h1n1_share"] = (df["inf_a_h1n1pdm09"] / denom_total).fillna(0.0)
    df["b_share"] = (df["inf_b"] / denom_total).fillna(0.0)
    df["positivity"] = (df["inf_total"] / denom_spec).fillna(0.0)
    return df[["week_start", "h3n2_share", "h1n1_share", "b_share", "positivity"]]


def _load_seoul_ili_weekly() -> pd.DataFrame:
    """Seoul sentinel ILI 주별 rate (all-age 평균)."""
    from datetime import date, timedelta

    con = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql_query(
        """
        SELECT season_start, week_seq, ili_rate
        FROM sentinel_influenza
        WHERE age_group = '전체' OR age_group IS NULL OR age_group = ''
        ORDER BY season_start, week_seq
        """,
        con,
    )
    # 전체 평균이 없으면 모든 연령 평균으로 대체
    if df.empty:
        con2 = sqlite3.connect(str(DB_PATH))
        df = pd.read_sql_query(
            "SELECT season_start, week_seq, ili_rate FROM sentinel_influenza",
            con2,
        )
        con2.close()
        df = df.groupby(["season_start", "week_seq"], as_index=False)["ili_rate"].mean()
    con.close()

    def _week_to_date(season_start: int, week_seq: int) -> pd.Timestamp:
        # 서울 influenza season: season_start (year) 의 36주차 시작 (ISO week 36 = 9월 초)
        base = date(int(season_start), 9, 1)
        # 9/1 이 속한 ISO 주 월요일
        base = base - timedelta(days=base.weekday())
        return pd.Timestamp(base + timedelta(weeks=int(week_seq) - 1))

    df["week_start"] = [
        _week_to_date(int(r.season_start), int(r.week_seq)) for r in df.itertuples()
    ]
    return df[["week_start", "ili_rate"]].groupby("week_start", as_index=False)["ili_rate"].mean()


def _seasonal_lag_residual(ili: np.ndarray) -> np.ndarray:
    """y_t - (a·sin + b·cos + c·y_{t-1} + d) OLS residual."""
    n = len(ili)
    t = np.arange(n)
    sin52 = np.sin(2 * np.pi * t / 52.0)
    cos52 = np.cos(2 * np.pi * t / 52.0)
    lag1 = np.concatenate([[ili[0]], ili[:-1]])
    X = np.column_stack([sin52, cos52, lag1, np.ones(n)])
    # Solve OLS
    beta, *_ = np.linalg.lstsq(X, ili, rcond=None)
    resid = ili - X @ beta
    return resid


def _corr_report(name: str, x: np.ndarray, y: np.ndarray) -> tuple[str, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 20:
        return (name, np.nan)
    r = float(np.corrcoef(x[mask], y[mask])[0, 1])
    return (name, r)


def main() -> int:
    print("=" * 60)
    print("R3-4 FluNet subtype signal test")
    print("=" * 60)

    flunet = _load_kr_flunet_weekly()
    ili = _load_seoul_ili_weekly()
    if flunet.empty or ili.empty:
        print("[FAIL] data load empty")
        return 2

    merged = pd.merge(ili, flunet, on="week_start", how="inner").sort_values("week_start")
    print(f"merged weeks: {len(merged)}  "
          f"({merged['week_start'].min().date()} ~ {merged['week_start'].max().date()})")
    if len(merged) < 50:
        print("[FAIL] insufficient overlap")
        return 2

    y = merged["ili_rate"].values
    resid = _seasonal_lag_residual(y)

    print(f"\n-- RAW corr (|r| >= 0.15 = strong) --")
    feats = ["h3n2_share", "h1n1_share", "b_share", "positivity"]
    raw_rs = {}
    for f in feats:
        name, r = _corr_report(f, merged[f].values, y)
        raw_rs[f] = r
        print(f"  r(ili, {f:12s}) = {r:+.3f}")

    print(f"\n-- RESIDUAL corr (after sin/cos + lag1) (|r| >= 0.10 = worth full retrain) --")
    res_rs = {}
    for f in feats:
        name, r = _corr_report(f, merged[f].values, resid)
        res_rs[f] = r
        print(f"  r(resid, {f:12s}) = {r:+.3f}")

    # Incremental R² via ridge: baseline (sin,cos,lag1) vs + subtype features
    from sklearn.linear_model import RidgeCV

    n = len(merged)
    split = int(n * 0.75)
    t = np.arange(n)
    sin52 = np.sin(2 * np.pi * t / 52.0)
    cos52 = np.cos(2 * np.pi * t / 52.0)
    lag1 = np.concatenate([[y[0]], y[:-1]])
    X_base = np.column_stack([sin52, cos52, lag1])
    X_full = np.column_stack([X_base, merged[feats].values])

    m1 = RidgeCV().fit(X_base[:split], y[:split])
    m2 = RidgeCV().fit(X_full[:split], y[:split])

    def _r2(yt, yp):
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2) + 1e-12
        return 1.0 - ss_res / ss_tot

    p1 = m1.predict(X_base[split:])
    p2 = m2.predict(X_full[split:])
    r2_base = _r2(y[split:], p1)
    r2_full = _r2(y[split:], p2)

    print(f"\n-- Hold-out R2 (last {n-split} weeks) --")
    print(f"  baseline (sin+cos+lag1):         R2 = {r2_base:+.4f}")
    print(f"  baseline + FluNet 4 features:    R2 = {r2_full:+.4f}")
    print(f"  dR2 (FluNet marginal):           {r2_full - r2_base:+.4f}")

    # Gate 판정
    max_resid_r = max(abs(v) for v in res_rs.values())
    dr2 = r2_full - r2_base
    print(f"\n=== Gate ===")
    print(f"  max |r_residual|  = {max_resid_r:.3f}  (gate 0.10)")
    print(f"  dR2 marginal      = {dr2:+.4f}  (gate +0.02)")
    if max_resid_r >= 0.10 and dr2 >= 0.02:
        print(f"  VERDICT: R3-4 FULL RETRAIN WORTH IT")
        return 0
    elif max_resid_r >= 0.05 or dr2 >= 0.01:
        print(f"  VERDICT: MARGINAL -- E 결과 보고 결정")
        return 0
    else:
        print(f"  VERDICT: SKIP -- 신호 미약, full retrain 불필요")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
