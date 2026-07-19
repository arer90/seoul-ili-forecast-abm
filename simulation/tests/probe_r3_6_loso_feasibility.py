"""
simulation/tests/test_r3_6_loso_feasibility.py
===============================================
R3-6 간이 테스트 — 2022-23 season LOSO 가 일반 WF-CV 와 *달라지는지* 검증.

질문: last-fold WF-CV (≈ 자동으로 2024-25 예측) 과 LOSO 2022-23 (train 에서
2022-23 제거 → 그 season 예측) 의 metric 이 유의미하게 다른가?

다르다면 (예: RMSE 격차 > 20%): 논문 robustness 섹션용으로 full LOSO 가치 있음.
비슷하다면: 현재 WF-CV 만으로 충분.

경량: ElasticNet + RandomForest + BayesianRidge 3 모델 (CPU-only, DL skip).
    E (GPU) 와 충돌 없음.

실행: .venv/Scripts/python.exe -m simulation.tests.test_r3_6_loso_feasibility
"""
from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge, ElasticNetCV

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "db" / "epi_real_seoul.db"


def _load_seoul_ili() -> pd.DataFrame:
    con = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql_query(
        "SELECT season_start, week_seq, ili_rate FROM sentinel_influenza",
        con,
    )
    con.close()
    df = df.groupby(["season_start", "week_seq"], as_index=False)["ili_rate"].mean()

    def _to_date(s: int, w: int) -> pd.Timestamp:
        base = date(int(s), 9, 1)
        base = base - timedelta(days=base.weekday())
        return pd.Timestamp(base + timedelta(weeks=int(w) - 1))

    df["week_start"] = [_to_date(int(r.season_start), int(r.week_seq))
                        for r in df.itertuples()]
    return df.sort_values("week_start").reset_index(drop=True)[
        ["week_start", "season_start", "ili_rate"]
    ]


def _load_weekly_weather() -> pd.DataFrame:
    con = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql_query(
        """
        SELECT obs_date, ta_avg, hm_avg
        FROM weather_historical
        WHERE stn_id = 108
        ORDER BY obs_date
        """,
        con,
    )
    con.close()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["obs_date"], format="%Y%m%d")
    df["week_start"] = df["date"] - pd.to_timedelta(df["date"].dt.weekday, unit="d")
    weekly = df.groupby("week_start", as_index=False).agg(
        temp_avg=("ta_avg", "mean"), humidity=("hm_avg", "mean")
    )
    return weekly


def _build_features(ili: pd.DataFrame, wx: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(ili, wx, on="week_start", how="left")
    df["temp_avg"] = df["temp_avg"].interpolate().bfill().ffill()
    df["humidity"] = df["humidity"].interpolate().bfill().ffill()
    df["lag1"] = df["ili_rate"].shift(1)
    df["lag2"] = df["ili_rate"].shift(2)
    df["lag4"] = df["ili_rate"].shift(4)
    df["rmean4"] = df["ili_rate"].rolling(4, min_periods=1).mean().shift(1)
    df["rstd4"] = df["ili_rate"].rolling(4, min_periods=1).std().shift(1).fillna(0)
    t = np.arange(len(df))
    df["sin52"] = np.sin(2 * np.pi * t / 52.0)
    df["cos52"] = np.cos(2 * np.pi * t / 52.0)
    return df.dropna(subset=["lag1", "lag2", "lag4"]).reset_index(drop=True)


def _metrics(y: np.ndarray, yp: np.ndarray) -> dict:
    mask = np.isfinite(y) & np.isfinite(yp)
    y, yp = y[mask], yp[mask]
    if len(y) < 3:
        return {"R2": np.nan, "RMSE": np.nan, "peak_ratio": np.nan}
    ss_res = float(np.sum((y - yp) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
    return {
        "R2": 1.0 - ss_res / ss_tot,
        "RMSE": float(np.sqrt(ss_res / len(y))),
        "peak_ratio": float(yp.max() / max(y.max(), 1e-6)),
    }


def _fit_predict(model, X_tr, y_tr, X_te):
    model.fit(X_tr, y_tr)
    return model.predict(X_te)


def main() -> int:
    print("=" * 60)
    print("R3-6 LOSO 2022-23 feasibility test")
    print("=" * 60)

    ili = _load_seoul_ili()
    wx = _load_weekly_weather()
    if ili.empty or wx.empty:
        print("[FAIL] data empty")
        return 2

    df = _build_features(ili, wx)
    print(f"full feature matrix: n={len(df)} weeks "
          f"({df['week_start'].min().date()} ~ {df['week_start'].max().date()})")

    feat_cols = ["lag1", "lag2", "lag4", "rmean4", "rstd4",
                 "temp_avg", "humidity", "sin52", "cos52"]
    X = df[feat_cols].values.astype(float)
    y = df["ili_rate"].values.astype(float)
    ws = df["week_start"].values

    # 2022-23 season: 2022-09-01 ~ 2023-08-31
    season_22_23_mask = (
        (df["week_start"] >= pd.Timestamp("2022-09-01"))
        & (df["week_start"] < pd.Timestamp("2023-09-01"))
    ).values
    n_in = int(season_22_23_mask.sum())
    if n_in < 20:
        print(f"[FAIL] 2022-23 season has only {n_in} weeks")
        return 2

    # LAST-FOLD WF-CV 모사: 데이터의 마지막 15% 를 test
    n = len(df)
    last_fold_test_idx = np.arange(int(n * 0.85), n)
    last_fold_train_idx = np.arange(0, int(n * 0.85))

    # LOSO 2022-23: 2022-23 제외 → train, 2022-23 → test
    loso_test_idx = np.where(season_22_23_mask)[0]
    loso_train_idx = np.where(~season_22_23_mask)[0]

    models = {
        "ElasticNetCV": lambda: ElasticNetCV(cv=5, max_iter=10_000, random_state=42),
        "BayesianRidge": lambda: BayesianRidge(),
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=200, max_depth=8, n_jobs=1, random_state=42
        ),
    }

    print(f"\n-- Split sizes --")
    print(f"  LAST-FOLD WF-CV:  train={len(last_fold_train_idx)}  test={len(last_fold_test_idx)}"
          f" ({df['week_start'].iloc[last_fold_test_idx[0]].date()} ~ "
          f"{df['week_start'].iloc[last_fold_test_idx[-1]].date()})")
    print(f"  LOSO 2022-23:     train={len(loso_train_idx)}  test={len(loso_test_idx)}"
          f" ({df['week_start'].iloc[loso_test_idx[0]].date()} ~ "
          f"{df['week_start'].iloc[loso_test_idx[-1]].date()})")

    results = []
    for name, factory in models.items():
        # WF-CV last fold
        m1 = factory()
        p1 = _fit_predict(m1, X[last_fold_train_idx], y[last_fold_train_idx],
                          X[last_fold_test_idx])
        met1 = _metrics(y[last_fold_test_idx], p1)
        # LOSO 2022-23
        m2 = factory()
        p2 = _fit_predict(m2, X[loso_train_idx], y[loso_train_idx],
                          X[loso_test_idx])
        met2 = _metrics(y[loso_test_idx], p2)

        results.append({
            "model": name,
            "wf_R2": met1["R2"], "wf_RMSE": met1["RMSE"], "wf_peak": met1["peak_ratio"],
            "loso_R2": met2["R2"], "loso_RMSE": met2["RMSE"], "loso_peak": met2["peak_ratio"],
            "rmse_delta_pct": (met2["RMSE"] - met1["RMSE"]) / max(met1["RMSE"], 1e-6) * 100,
        })

    print(f"\n-- Metrics (last-fold WF-CV vs LOSO 2022-23) --")
    print(f"  {'model':15s} {'WF R2':>8} {'LOSO R2':>8} "
          f"{'WF RMSE':>9} {'LOSO RMSE':>10} {'dRMSE%':>9}")
    for r in results:
        print(f"  {r['model']:15s} {r['wf_R2']:+8.3f} {r['loso_R2']:+8.3f} "
              f"{r['wf_RMSE']:9.3f} {r['loso_RMSE']:10.3f} {r['rmse_delta_pct']:+9.1f}")

    # 판정
    max_abs_delta = max(abs(r["rmse_delta_pct"]) for r in results)
    max_r2_gap = max(abs(r["wf_R2"] - r["loso_R2"]) for r in results)
    print(f"\n=== Gate ===")
    print(f"  max |dRMSE%| = {max_abs_delta:6.1f}   (gate 20%)")
    print(f"  max |dR2|    = {max_r2_gap:.3f}   (gate 0.15)")
    if max_abs_delta >= 20 or max_r2_gap >= 0.15:
        print(f"  VERDICT: LOSO 결과 상이 -- R3-6 FULL RETRAIN 가치 있음")
        return 0
    else:
        print(f"  VERDICT: WF-CV 와 유사 -- R3-6 SKIP 가능 (논문에는 WF-CV 로 충분)")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
