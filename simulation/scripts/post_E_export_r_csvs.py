"""
post_E_export_r_csvs.py
=======================
E 학습 이후 R verification 스크립트 (simulation/r_verification/*.R) 가 필요로
하는 CSV 입력을 생성한다.

출력 (simulation/results/post_E/):
  - ili_series.csv       (week_start, ili_rate)
  - model_predictions.csv (week_start, y_true, <model1>, <model2>, ...)
  - model_residuals.csv  (week_start, <model1>, <model2>, ...)  # y_true - y_pred
  - pi_samples.csv       (week_start, model, quantile, value)  # R7(intervals) conformal PI -> long quantile format
  - rt_seir_v2.csv       (week_start, rt_eff)
  - npi_window.csv       (event, iso_date)

의존성:
  - simulation/results/csv/predictions_<model>.csv (test split 만 사용)
  - simulation/results/checkpoints/checkpoint_phase6.json (Conformal PI bounds)
  - simulation/results/phase4_baseline_sidecar.pkl (SEIR-V2-Forced 객체에서 rt 추출)
  - DB epi.weekly_disease (week_start 역산 용)
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from simulation.config_global import Z95, Z90  # SSOT (2026-05-28)

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "simulation" / "results"
CSV_DIR = RES / "csv"
OUT_DIR = RES / "post_E"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NPI_START_ISO = "2020-03-02"
NPI_END_ISO = "2022-12-26"


def _load_dates(n: int) -> np.ndarray:
    """KDCA sentinel_influenza 의 주차 시작일 시퀀스를 길이 n 만큼 반환.

    feature_engine 의 _load_sentinel_ili 가 (season_start, week_seq) →
    cal_date 로 변환한 뒤 week_start = cal_date - weekday(cal_date) 로 계산.
    학습 파이프라인과 동일 경로를 재사용해 정렬 보장.
    """
    from simulation.models.feature_engine.loaders import _load_sentinel_ili
    from simulation.database.config import DB_PATH
    import polars as pl
    df = _load_sentinel_ili(str(DB_PATH))
    # cal_date is pl.Date; week_start = cal_date - weekday offset
    df_pd = df.select(
        (pl.col("cal_date").cast(pl.Datetime("us"))
         - pl.duration(days=pl.col("cal_date").dt.weekday())).alias("week_start")
    ).to_pandas()
    dates = pd.to_datetime(df_pd["week_start"]).sort_values().to_numpy()
    if len(dates) < n:
        raise RuntimeError(f"sentinel_influenza {len(dates)} < required {n}")
    return dates[-n:]


def _export_ili_series(dates: np.ndarray, y_true: np.ndarray) -> Path:
    df = pd.DataFrame({"week_start": dates, "ili_rate": y_true})
    p = OUT_DIR / "ili_series.csv"
    df.to_csv(p, index=False)
    return p


def _collect_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    """predictions_<model>.csv 들을 wide 로 결합.

    Returns:
      preds_wide: (n_test, 1+n_models)   week_start, y_true, <model1>, ...
      resid_wide: (n_test, 1+n_models)   week_start, <model1>, ...
    """
    csvs = sorted(CSV_DIR.glob("predictions_*.csv"))
    if not csvs:
        raise RuntimeError(f"predictions_*.csv not found in {CSV_DIR}")

    y_true_ref: Optional[np.ndarray] = None
    preds: dict[str, np.ndarray] = {}
    for c in csvs:
        df = pd.read_csv(c)
        test = df[df["split"] == "test"].sort_values("idx")
        if len(test) == 0:
            continue
        yt = test["y_true"].to_numpy(dtype=float)
        yp = test["y_pred"].to_numpy(dtype=float)
        if y_true_ref is None:
            y_true_ref = yt
        elif len(yt) != len(y_true_ref) or not np.allclose(yt, y_true_ref, atol=1e-6, equal_nan=True):
            # 길이는 맞추고, 값 불일치는 경고만
            if len(yt) == len(y_true_ref):
                pass
            else:
                continue
        name = c.stem.replace("predictions_", "")
        preds[name] = yp

    if y_true_ref is None:
        raise RuntimeError("no test split found in any predictions CSV")

    n = len(y_true_ref)
    dates = _load_dates(n)

    preds_wide = pd.DataFrame({"week_start": dates, "y_true": y_true_ref})
    resid_wide = pd.DataFrame({"week_start": dates})
    for name, yp in preds.items():
        preds_wide[name] = yp
        resid_wide[name] = y_true_ref - yp

    return preds_wide, resid_wide


def _export_pi_samples(dates_test: np.ndarray, y_true: np.ndarray) -> tuple[Optional[Path], Optional[Path]]:
    """R7(intervals) symmetric conformal quantile → 두 포맷:
      - pi_samples.csv      : (week_start, model, quantile, value) long format
      - pi_samples_wide.csv : (week_start, model, y_true, q025, q500, q975) wide (scoringutils 04.R)

    R7(intervals) 에 없는 모델은 test-residual std 를 sigma 로 써서 z*sigma 로 복원 (Gaussian 근사).
    """
    p6 = RES / "checkpoints" / "checkpoint_phase6.json"
    phase6_q: dict[str, float] = {}
    if p6.exists():
        try:
            data = json.loads(p6.read_text(encoding="utf-8"))
            pi_results = (data.get("data", {}) or {}).get("pi_results", {}) or {}
            for m, entry in pi_results.items():
                if isinstance(entry, dict):
                    q = (entry.get("conformal", {}) or {}).get("quantile")
                    if q is not None:
                        phase6_q[m] = float(q)
        except Exception:
            pass

    rows_long: list[dict] = []
    rows_wide: list[dict] = []
    n = len(y_true)

    for csv in sorted(CSV_DIR.glob("predictions_*.csv")):
        model = csv.stem.replace("predictions_", "")
        df_pred = pd.read_csv(csv)
        test = df_pred[df_pred["split"] == "test"].sort_values("idx")
        if len(test) != n:
            continue
        yp = test["y_pred"].to_numpy(dtype=float)
        resid = y_true - yp

        if model in phase6_q:
            q95 = phase6_q[model]
            source = "phase6_conformal"
        else:
            sigma = float(np.std(resid, ddof=1)) if resid.size > 1 else 1.0
            q95 = Z95 * sigma
            source = "residual_gaussian"
        q90 = q95 * (Z90 / Z95)

        bands = {
            0.025: yp - q95, 0.05: yp - q90,
            0.5: yp,
            0.95: yp + q90, 0.975: yp + q95,
        }
        for q, vals in bands.items():
            for i in range(n):
                rows_long.append({
                    "week_start": dates_test[i], "model": model,
                    "quantile": q, "value": float(vals[i]),
                    "source": source,
                })
        for i in range(n):
            rows_wide.append({
                "week_start": dates_test[i], "model": model,
                "y_true": float(y_true[i]),
                "q025": float(yp[i] - q95), "q500": float(yp[i]),
                "q975": float(yp[i] + q95),
                "q050": float(yp[i] - q90), "q950": float(yp[i] + q90),
                "source": source,
            })

    if not rows_long:
        return None, None

    p_long = OUT_DIR / "pi_samples.csv"
    p_wide = OUT_DIR / "pi_samples_wide.csv"
    pd.DataFrame(rows_long).to_csv(p_long, index=False)
    pd.DataFrame(rows_wide).to_csv(p_wide, index=False)
    return p_long, p_wide


def _export_predictions_long(dates_test: np.ndarray, y_true: np.ndarray) -> Optional[Path]:
    """03_dm_test_canonical.R 용 long format: model, week_start, y_true, y_pred, regime."""
    rows: list[dict] = []
    n = len(y_true)
    # Regime: assign by date (pre-covid < 2020-03-02 < during < 2022-12-26 < post)
    npi_start = np.datetime64("2020-03-02")
    npi_end = np.datetime64("2022-12-26")

    for csv in sorted(CSV_DIR.glob("predictions_*.csv")):
        model = csv.stem.replace("predictions_", "")
        df_pred = pd.read_csv(csv)
        test = df_pred[df_pred["split"] == "test"].sort_values("idx")
        if len(test) != n:
            continue
        yp = test["y_pred"].to_numpy(dtype=float)
        for i in range(n):
            d = np.datetime64(pd.Timestamp(dates_test[i]).date())
            if d < npi_start:
                rg = "pre"
            elif d <= npi_end:
                rg = "during"
            else:
                rg = "post"
            rows.append({
                "model": model, "week_start": dates_test[i],
                "y_true": float(y_true[i]), "y_pred": float(yp[i]),
                "regime": rg,
            })

    if not rows:
        return None
    p = OUT_DIR / "model_predictions_long.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def _export_rt_seir_v2(dates: np.ndarray) -> Optional[Path]:
    """SEIR-V2-Forced.rt_effective_trajectory() 를 full period 로 재계산."""
    sc = RES / "phase4_baseline_sidecar.pkl"
    if not sc.exists():
        return None
    try:
        with sc.open("rb") as f:
            d = pickle.load(f)
        seir_entry = (d.get("individual_results") or {}).get("SEIR-V2-Forced")
        if seir_entry is None:
            return None
    except Exception:
        return None

    # SEIR-V2 model pickle 에서 파라미터 복원
    model_pkl = RES / "models_pt" / "SEIR-V2-Forced.pt"
    if not model_pkl.exists():
        model_pkl = ROOT / "models" / "SEIR-V2-Forced.pt"
    if not model_pkl.exists():
        return None
    try:
        with model_pkl.open("rb") as f:
            model = pickle.load(f)
        # Older pickles saved before __init__ set _npi_*_idx can miss these
        # attrs; repopulate from fallback window indices.
        if not hasattr(model, "_npi_start_idx") or not hasattr(model, "_npi_end_idx"):
            model._npi_start_idx, model._npi_end_idx = 114, 260  # fallback
        if not hasattr(model, "_train_len"):
            model._train_len = len(dates)
        rt = model.rt_effective_trajectory(t_start=0.0, t_end=float(len(dates) - 1))
        if rt is None:
            return None
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("rt_seir_v2 export failed: %s", e)
        return None

    n = min(len(dates), len(rt))
    df = pd.DataFrame({"week_start": dates[:n], "rt_eff": rt[:n]})
    p = OUT_DIR / "rt_seir_v2.csv"
    df.to_csv(p, index=False)
    return p


def _export_npi_window() -> Path:
    df = pd.DataFrame([
        {"event": "npi_start", "iso_date": NPI_START_ISO},
        {"event": "npi_end", "iso_date": NPI_END_ISO},
    ])
    p = OUT_DIR / "npi_window.csv"
    df.to_csv(p, index=False)
    return p


def main() -> int:
    outputs: dict[str, str] = {}

    preds_wide, resid_wide = _collect_predictions()
    p_pred = OUT_DIR / "model_predictions.csv"
    p_resid = OUT_DIR / "model_residuals.csv"
    preds_wide.to_csv(p_pred, index=False)
    resid_wide.to_csv(p_resid, index=False)
    outputs["model_predictions.csv"] = str(p_pred)
    outputs["model_residuals.csv"] = str(p_resid)

    # ILI series — FULL period (for ITS 07 which needs NPI window 2020-03~2022-12)
    dates_test = preds_wide["week_start"].to_numpy()
    y_true = preds_wide["y_true"].to_numpy()
    from simulation.models.feature_engine.loaders import _load_sentinel_ili
    from simulation.database.config import DB_PATH
    import polars as pl
    _df_full = _load_sentinel_ili(str(DB_PATH))
    _df_full_pd = _df_full.select(
        (pl.col("cal_date").cast(pl.Datetime("us"))
         - pl.duration(days=pl.col("cal_date").dt.weekday())).alias("week_start"),
        pl.col("ili_rate"),
    ).to_pandas().sort_values("week_start")
    p_ili = OUT_DIR / "ili_series.csv"
    _df_full_pd.to_csv(p_ili, index=False)
    outputs["ili_series.csv"] = str(p_ili)

    # PI samples (long + wide formats)
    p_pi, p_pi_wide = _export_pi_samples(dates_test, y_true)
    if p_pi:
        outputs["pi_samples.csv"] = str(p_pi)
    if p_pi_wide:
        outputs["pi_samples_wide.csv"] = str(p_pi_wide)

    # Long-format predictions (for 03_dm_test_canonical.R)
    p_long = _export_predictions_long(dates_test, y_true)
    if p_long:
        outputs["model_predictions_long.csv"] = str(p_long)

    # SEIR-V2 Rt trajectory (full period)
    # Need full series dates including train. Reuse feature_engine loader.
    from simulation.models.feature_engine.loaders import _load_sentinel_ili
    from simulation.database.config import DB_PATH
    import polars as pl
    _df_ili = _load_sentinel_ili(str(DB_PATH))
    dates_all = pd.to_datetime(
        _df_ili.select(
            (pl.col("cal_date").cast(pl.Datetime("us"))
             - pl.duration(days=pl.col("cal_date").dt.weekday())).alias("week_start")
        ).to_pandas()["week_start"]
    ).sort_values().to_numpy()
    p_rt = _export_rt_seir_v2(dates_all)
    if p_rt:
        outputs["rt_seir_v2.csv"] = str(p_rt)

    # NPI window (static)
    p_npi = _export_npi_window()
    outputs["npi_window.csv"] = str(p_npi)

    expected = {"ili_series.csv", "model_predictions.csv", "model_residuals.csv",
                "pi_samples.csv", "pi_samples_wide.csv", "model_predictions_long.csv",
                "rt_seir_v2.csv", "npi_window.csv"}
    print(f"\n=== post_E R-verification CSVs exported ({len(outputs)}/{len(expected)}) ===")
    for k, v in outputs.items():
        print(f"  [OK] {k:<28s} -> {v}")
    missing = expected - set(outputs.keys())
    for m in missing:
        print(f"  [SKIP] {m} (input dependency missing)")

    # Manifest for downstream R scripts
    manifest = OUT_DIR / "manifest.json"
    manifest.write_text(
        json.dumps({
            "outputs": outputs,
            "missing": sorted(missing),
            "n_test": int(len(y_true)),
            "n_models_pred": int(len(preds_wide.columns) - 2),  # - week_start, y_true
            "npi_start": NPI_START_ISO,
            "npi_end": NPI_END_ISO,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"\n  [manifest] {manifest}")
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
