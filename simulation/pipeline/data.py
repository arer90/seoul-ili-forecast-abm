"""
R1: 데이터 로드 + 피처 엔지니어링 + Leakage 검사
======================================================
- DB에서 데이터 로드
- 267+ 피처 생성 (build_enriched_features → Polars DataFrame)
- Parquet 캐시 (DB 미변경 시 재사용)
- Data Leakage 검사 (상관 > threshold 피처 자동 제거/경고)
"""
import logging
import os
import time
import numpy as np
import polars as pl
from pathlib import Path
from typing import Tuple, List, Optional

log = logging.getLogger(__name__)


def compute_split_indices(n: int, config) -> tuple[int, int, int]:
    """Compute (n_train, n_val, n_test) for the in-sample window of size n.

    Single source of truth for R1 (data) / R2 (baseline) / R7 (intervals) —
    keeps the 4-way HWP split consistent across phases.

    Priority order:
      1. New HWP fields (`in_sample_test_ratio` / `in_sample_val_ratio`) if
         set on config.split — splits 80% train_pool : 20% test, then carves
         val as last 10% of train_pool.
      2. Legacy ratio fields (`train_ratio` / `val_ratio` / `test_ratio`).
    """
    has_hwp = (hasattr(config.split, "in_sample_test_ratio")
               and getattr(config.split, "in_sample_test_ratio", None) is not None)
    if has_hwp:
        import math
        test_ratio = float(config.split.in_sample_test_ratio)
        val_ratio  = float(getattr(config.split, "in_sample_val_ratio", 0.10))
        # HWP §3 정합: ceil for test → with n=337, test=68 (matches HWP "약 68건").
        # train_pool = 269 (HWP "약 269건"), val carved from pool.
        n_test  = math.ceil(n * test_ratio)
        pool    = n - n_test
        n_val   = int(round(pool * val_ratio))
        n_train = pool - n_val
    else:
        n_train = int(n * float(config.split.train_ratio))
        n_val   = (int(n * float(config.split.val_ratio))
                   if getattr(config.split, "use_validation", True) else 0)
        n_test  = n - n_train - n_val
    return n_train, n_val, n_test


from simulation.utils.resource_tracker import track_resources


def apply_covid_sensitivity_mode(X_all, y_all, dates, feature_cols, real_X, mode):
    """COVID-era 3-way sensitivity transform on the in-sample matrix (C2/M7).

    Implements the leave-the-pandemic-out + NPI-era-covariate robustness the
    flu-surveillance reviewer literature expects for the 2020-03→2022-12 NPI-
    suppressed window (it breaks ILI stationarity):
      - ``"include"`` (default): no change.
      - ``"exclude"``: drop the COVID-era weeks from train/eval — the leave-out
        sensitivity (show conclusions hold without the structural break).
      - ``"indicator"``: append a binary ``covid_era_indicator`` covariate so
        models attribute the suppression structurally rather than memorising it
        as seasonality; the real slab (post-2026) gets all-zeros so column counts
        stay aligned for the rolling-origin vstack.

    Args:
        X_all: (n, p) in-sample design matrix.
        y_all: (n,) target.
        dates: (n,) ``datetime64`` array, or None.
        feature_cols: column-name list (a copy is returned under "indicator").
        real_X: (m, p) real-slab matrix or None (gets a 0-column under "indicator").
        mode: ``"include" | "exclude" | "indicator"``.

    Returns:
        ``(X_all, y_all, dates, feature_cols, real_X)`` after the transform;
        inputs unchanged for "include", ``dates is None``, or no COVID weeks.

    Side effects: logs the action. No disk/DB. Never raises (logs + returns
        inputs on error).
    """
    if mode == "include" or dates is None:
        return X_all, y_all, dates, feature_cols, real_X
    try:
        date_arr = (dates if dates.dtype.kind == "M"
                    else np.array(dates, dtype="datetime64[D]"))
        covid_start = np.datetime64("2020-03-01")
        covid_end = np.datetime64("2022-12-31")
        covid_mask = (date_arr >= covid_start) & (date_arr <= covid_end)
        n_covid = int(covid_mask.sum())
        if mode == "exclude" and n_covid > 0:
            keep = ~covid_mask
            X_all = X_all[keep]
            y_all = y_all[keep]
            dates = date_arr[keep]
            log.info(f"  [COVID/{mode}] dropped {n_covid} weeks "
                     f"({covid_start} → {covid_end})")
        elif mode == "indicator" and n_covid > 0:
            covid_col = covid_mask.astype(np.float64).reshape(-1, 1)
            X_all = np.hstack([X_all, covid_col])
            feature_cols = list(feature_cols) + ["covid_era_indicator"]
            if real_X is not None:
                real_X = np.hstack([real_X, np.zeros((len(real_X), 1), dtype=np.float64)])
            log.info(f"  [COVID/{mode}] added covid_era_indicator covariate "
                     f"({n_covid} positive weeks); real slab gets 0s")
    except Exception as e:
        log.warning(f"  [COVID] sensitivity mode={mode} failed: {e}")
    return X_all, y_all, dates, feature_cols, real_X


@track_resources("phase1")
def run_data(config) -> dict:
    """R1 (data) 실행. 반환: {X_all, y_all, feature_cols, n, metadata, resource_tracker}.

    2026-05-28 (사용자 명시): @track_resources 로 자원/시간 기록 result 에 자동 첨부.
    """
    from simulation.models.feature_engine.builder import build_enriched_features

    log.info("  [1-1] 데이터 로드 + 피처 엔지니어링")
    t0 = time.time()
    cache_dir = Path(config.data.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "feature_cache.parquet"
    db_path = Path(config.data.db_path)

    # 캐시 확인
    dates: Optional[np.ndarray] = None
    if (config.data.use_fe_cache and cache_path.exists()
            and db_path.exists()
            and cache_path.stat().st_mtime > db_path.stat().st_mtime):
        log.info(f"  FE 캐시 사용: {cache_path}")
        df = pl.read_parquet(cache_path)
        y_col = "ili_rate" if "ili_rate" in df.columns else df.columns[0]
        # F1: week_start 는 타겟/피처가 아니다. 피처 열에서 빼고 dates 로 추출.
        feature_cols = [c for c in df.columns if c not in (y_col, "week_start")]
        X_all = df.select(feature_cols).to_numpy().astype(np.float64)
        y_all = df[y_col].to_numpy().astype(np.float64)
        if "week_start" in df.columns:
            dates = df["week_start"].to_numpy()
    else:
        log.info(f"  DB 경로: {db_path}")
        # Feature flags from config (FeatureConfig)
        fe_kwargs = {}
        if hasattr(config, 'features'):
            from dataclasses import asdict
            import inspect
            fe_kwargs = asdict(config.features)
            # Filter to only params the builder accepts
            valid = set(inspect.signature(build_enriched_features).parameters.keys())
            fe_kwargs = {k: v for k, v in fe_kwargs.items() if k in valid}
        result = build_enriched_features(str(db_path), **fe_kwargs)

        # build_enriched_features returns (feat_df: pl.DataFrame, meta: dict)
        if isinstance(result, tuple) and len(result) == 2:
            feat_df, meta = result
            y_col = "ili_rate" if "ili_rate" in feat_df.columns else feat_df.columns[0]
            # fix: drop non-numeric columns (e.g. rt_road congestion
            # labels like '여유'/'보통'/'혼잡') before coercing to float.
            # Upstream feature_engine sometimes leaks categorical strings
            # into the output — filter by polars dtype so numpy astype
            # doesn't explode on the first row.
            numeric_dtypes = (
                pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                pl.Float32, pl.Float64, pl.Boolean,
            )
            schema = feat_df.schema
            dropped_str = [c for c in feat_df.columns
                           if c != y_col and schema[c] not in numeric_dtypes]
            if dropped_str:
                log.warning(
                    f"  [R1] 비숫자 컬럼 {len(dropped_str)}개 제외: "
                    f"{dropped_str[:5]}{'...' if len(dropped_str) > 5 else ''}"
                )
            feature_cols = [c for c in feat_df.columns
                            if c != y_col and schema[c] in numeric_dtypes]
            X_all = feat_df.select(feature_cols).to_numpy().astype(np.float64)
            y_all = feat_df[y_col].to_numpy().astype(np.float64)
            # F1: builder 가 meta["dates"] 로 week_start 를 넘긴다.
            dates = meta.get("dates") if isinstance(meta, dict) else None
            if dates is not None and len(dates) != len(y_all):
                log.warning(
                    f"  [R1] meta dates 길이({len(dates)}) ≠ y_all({len(y_all)}) — 폐기"
                )
                dates = None
        else:
            # Fallback: (X_all, y_all, feature_cols) tuple
            X_all, y_all, feature_cols = result
        # Polars parquet 캐시 저장
        if config.data.use_fe_cache:
            try:
                cache_dict = {"ili_rate": y_all}
                cache_dict.update({col: X_all[:, i] for i, col in enumerate(feature_cols)})
                if dates is not None:
                    cache_dict["week_start"] = dates  # F1
                df_save = pl.DataFrame(cache_dict)
                df_save.write_parquet(cache_path)
                log.info(f"  FE 캐시 저장: {cache_path}")
            except Exception as e:
                log.warning(f"  캐시 저장 실패: {e}")

    # G-265c (nested blocked-CV, 2026-06-13): MPH_DATA_END_WEEK 설정 시 데이터를 시간순 앞 N주로 절단
    #   → 전체 파이프라인(슬랩·split·3-stage)이 짧은 확장윈도우로 일관 작동. nested 의 outer fold 가
    #   동일 파이프라인을 [:end_k] 로 K 번 실행하게 하는 enabler. 미설정 시 전체(기본, 무변화).
    #   ⚑ 양 분기 수렴점에서 단일 적용 — full feature_cache 는 위에서 이미 저장됐으므로 절단은 메모리상
    #     배열만(캐시 무오염). argsort(dates) 로 시간순 보장(이미 정렬돼 있으면 idempotent = [:N]).
    _de = os.environ.get("MPH_DATA_END_WEEK", "").strip()
    if _de.isdigit() and 0 < int(_de) < len(y_all):
        _n = int(_de)
        if dates is not None and len(dates) == len(y_all):
            _keep = np.argsort(dates)[:_n]          # 가장 이른 N주 인덱스(날짜 오름차순) → 시간순 정렬 보장
            X_all = X_all[_keep]; y_all = y_all[_keep]; dates = dates[_keep]
        else:
            X_all = X_all[:_n]; y_all = y_all[:_n]
        log.info(f"  [nested CV] MPH_DATA_END_WEEK={_de} → 시간순 앞 {_de}주 절단 "
                 f"(확장윈도우 outer fold; full 캐시 보존, 메모리상만)")

    n_full = len(y_all)
    log.info(f"  데이터: {n_full}행, {len(feature_cols)}개 피처")
    log.info(f"  FE 소요: {time.time()-t0:.1f}s")

    # ── REAL slab (4-way split, HWP §3): in-sample = first
    #    `paper_cutoff_week` weeks (default 337 per HWP). The remaining
    #    weeks become the "real" forecast slab — truly out-of-sample, never
    #    seen during R2 (baseline)/R4 (WF-CV)/R7 (intervals). P1 (real_forecaster)
    #    refits the final ensemble on the entire in-sample window and predicts
    #    the real slab to report forecasting performance.
    real_X = real_y = real_dates = None
    real_start = n_full
    real_weeks = 0

    # Resolve in-sample end. Date overrides week-count when both are set.
    in_sample_end = getattr(config.split, "in_sample_end", None)
    paper_cutoff = getattr(config.split, "paper_cutoff_week", None)
    if in_sample_end and dates is not None:
        try:
            cutoff = np.datetime64(str(in_sample_end))
            date_arr = (dates if dates.dtype.kind == "M"
                        else np.array(dates, dtype="datetime64[D]"))
            post_mask = date_arr > cutoff
            if bool(post_mask.any()):
                real_start = int(np.argmax(post_mask))
                real_weeks = n_full - real_start
                log.info(
                    f"  [REAL] post-cutoff forecast slab 예약: 마지막 {real_weeks}주 "
                    f"(date > {in_sample_end}, idx {real_start}:{n_full})"
                )
        except Exception as e:
            log.warning(f"  [REAL] in_sample_end={in_sample_end} 파싱 실패: {e}")
    elif paper_cutoff is not None and paper_cutoff < n_full:
        real_start = int(paper_cutoff)
        real_weeks = n_full - real_start
        log.info(
            f"  [REAL] HWP paper_cutoff_week={paper_cutoff} 까지 in-sample, "
            f"이후 {real_weeks}주 = real forecast slab "
            f"(idx {real_start}:{n_full}) — 학습/WF-CV/test 금지 구역, real_forecaster(P1)·inference 전용"
        )

    # Carve real slab off and persist for P1 (real_forecaster)
    if real_weeks > 0:
        real_X = X_all[real_start:].copy()
        real_y = y_all[real_start:].copy()
        if dates is not None:
            date_arr = (dates if dates.dtype.kind == "M"
                        else np.array(dates, dtype="datetime64[D]"))
            real_dates = date_arr[real_start:].copy()
        # Truncate in-sample arrays so all downstream phases operate on
        # the in-sample window only.
        X_all = X_all[:real_start]
        y_all = y_all[:real_start]
        if dates is not None:
            dates = dates[:real_start]

    # ── in_sample_start (optional, advanced): drop weeks before this date.
    in_sample_start = getattr(config.split, "in_sample_start", None)
    if in_sample_start and dates is not None:
        try:
            floor = np.datetime64(str(in_sample_start))
            date_arr = (dates if dates.dtype.kind == "M"
                        else np.array(dates, dtype="datetime64[D]"))
            keep_mask = date_arr >= floor
            n_dropped = int((~keep_mask).sum())
            if n_dropped > 0:
                log.info(
                    f"  [REAL] in_sample_start={in_sample_start} 이전 "
                    f"{n_dropped}주 drop"
                )
                X_all = X_all[keep_mask]
                y_all = y_all[keep_mask]
                dates = date_arr[keep_mask]
        except Exception as e:
            log.warning(
                f"  [REAL] in_sample_start={in_sample_start} 파싱 실패: {e}"
            )

    # ── COVID-era 3-way sensitivity (S2-E)
    #    "include":   keep 2020-03 → 2022-12 weeks as-is (legacy default)
    #    "exclude":   drop NPI-suppressed period from training
    #    "indicator": include + add binary covid_era covariate
    # C2/M7: extracted to the deep helper apply_covid_sensitivity_mode (testable).
    covid_mode = str(getattr(config.split, "covid_inclusion_mode", "include"))
    X_all, y_all, dates, feature_cols, real_X = apply_covid_sensitivity_mode(
        X_all, y_all, dates, feature_cols, real_X, covid_mode)

    # n now reflects the IN-SAMPLE window
    n = len(y_all)
    if real_weeks > 0 or in_sample_start:
        log.info(
            f"  in-sample = {n}주, real = {real_weeks}주 "
            f"(HWP 4-way split: train+val+test | real)"
        )

    # S0-1: carve off conformal holdout from the TAIL so R2 (baseline)/R7
    # (intervals) never see it. holdout_start = first index of the holdout slab.
    # Note: holdout is now WITHIN in-sample (after real-slab removal).
    holdout_weeks = int(getattr(config.split, "conformal_holdout_weeks", 0) or 0)
    holdout_start = n - holdout_weeks if holdout_weeks > 0 else n
    if holdout_weeks > 0:
        log.info(
            f"  [S0-1] Conformal holdout 예약: 마지막 {holdout_weeks}주 "
            f"of in-sample (index {holdout_start}:{n}) — 학습/WF-CV 금지 구역 (conformal PI 전용)"
        )

    # S1-1 guardrail: warn if the FE cache was written before the most
    # recent DB change, which would mean rolling-window features could
    # reflect post-cache DB updates outside the cache window. This is
    # a lightweight runtime sanity check until the full fold-wise
    # rolling audit lands.
    try:
        if cache_path.exists():
            cmt = cache_path.stat().st_mtime
            dmt = db_path.stat().st_mtime if db_path.exists() else 0
            if dmt > cmt:
                log.warning(
                    "  [S1-1] FE cache mtime < DB mtime -- rolling features "
                    "may use stale context. Rerun with --no-cache after DB updates."
                )
    except Exception:
        pass

    # [1-2] Data Leakage 검사
    log.info("  [1-2] Data Leakage 검사")
    removed_features = []
    n_train = int(n * config.split.train_ratio)
    X_train_check = X_all[:n_train]
    y_train_check = y_all[:n_train]

    suspect_indices = []
    # E-1: lag1-derived interaction 피처 (humid_ili, subway_ili, bus_ili,
    # school_ili, age_mixing_ili 등 _add_interaction_features 에서 ili_rate_lag1 ×
    # 외부피처로 causal 하게 정의된 피처) 는 lag1 자기상관(0.85~0.95)으로 인해
    # 이름에 "lag/diff" 가 없지만 타겟 프록시가 된다. leakage 가 아니므로 whitelist.
    _LEAKAGE_WHITELIST_SUFFIXES = ("_ili",)
    for i, col_name in enumerate(feature_cols):
        if col_name.endswith(_LEAKAGE_WHITELIST_SUFFIXES):
            continue  # E-1: causal interaction, not leakage
        if "lag" not in col_name.lower() and "diff" not in col_name.lower():
            col_std = float(np.std(X_train_check[:, i]))
            if col_std < 1e-10:
                continue  # E-3: std=0 이면 corrcoef 정의 불가
            corr = abs(np.corrcoef(X_train_check[:, i], y_train_check)[0, 1])
            if not np.isnan(corr) and corr >= config.data.leakage_corr_threshold:
                if config.data.leakage_action == "remove":
                    suspect_indices.append(i)
                    removed_features.append((col_name, corr))
                    log.warning(f"  [LEAKAGE 제거] {col_name}: 상관={corr:.4f}")
                elif config.data.leakage_action == "warn":
                    log.warning(f"  [LEAKAGE 경고] {col_name}: 상관={corr:.4f}")
    if suspect_indices and config.data.leakage_action == "remove":
        keep = [i for i in range(len(feature_cols)) if i not in suspect_indices]
        X_all = X_all[:, keep]
        feature_cols = [feature_cols[i] for i in keep]
        log.info(f"  {len(suspect_indices)}개 피처 제거 후 {len(feature_cols)}개 피처")

    # G-121: 최종 NaN/Inf 방어 — builder.py 에서 fill_nan+fill_null 을 했지만
    # transform/loader 경유 경로 중 누락이 있을 수 있으므로 numpy 단에서 한 번 더 잡는다.
    from .sanitize import sanitize_numpy
    X_all = sanitize_numpy(X_all, feature_cols, fill_value=0.0, label="G-121 R1")
    y_all = sanitize_numpy(y_all, label="G-121 R1 target")

    # float32 변환 (메모리 절약)
    if config.memory.use_float32:
        X_all = X_all.astype(np.float32)
        log.info("  float32 변환 완료")

    # ── HWP 4-way split indices (within in-sample n):
    #    [   train    |  val  |  test  | (conformal_holdout) ]
    #     0        n_train  pool_end  hold_start            n
    n_train_in, n_val_in, n_test_in = compute_split_indices(n, config)
    pool_end = n_train_in + n_val_in   # train+val pool ends here = test_start

    log.info(
        f"  [SPLIT] HWP 4-way: train={n_train_in}, val={n_val_in}, "
        f"test={n_test_in}, real={real_weeks}  (in-sample n={n})"
    )

    # ── audit Stage 1.1 (#13.c, 2026-05-27) — viral positivity expose ──
    # WHO FluNet KR 의 flu_positivity column 이 X_all 안에 있음 (builder.py:902-910).
    # KDCA epidemic threshold 산출 (Kang SK, Son WS, Kim BI 2024 doi:10.3346/jkms.2024.39.e40 PMID 38288541)
    # 위해 train pool slice 만 추출. caller (R10 per_model_eval/R11 shap/R12 comprehensive) 가 compute_full_metrics
    # 의 viral_positivity_train arg 로 전달.
    _viral_pos_train = None
    if "flu_positivity" in feature_cols:
        _flu_idx = feature_cols.index("flu_positivity")
        _viral_pos_train = X_all[:pool_end, _flu_idx].astype(np.float64).copy()

    return {
        "X_all": X_all,
        "y_all": y_all,
        "feature_cols": list(feature_cols),
        "n": n,
        "n_features": len(feature_cols),
        "removed_leakage": removed_features,
        "holdout_start": int(holdout_start),     # S0-1 conformal
        "holdout_weeks": int(holdout_weeks),     # S0-1 conformal
        "dates": dates,                          # F1 — in-sample only
        "elapsed": time.time() - t0,
        # 4-way split (HWP §3)
        "n_train": int(n_train_in),
        "n_val":   int(n_val_in),
        "n_test":  int(n_test_in),
        "pool_end": int(pool_end),               # train+val ends here
        "test_start": int(pool_end),             # alias (= n_train+n_val)
        "real_start_global": int(real_start),    # idx in original (pre-truncation)
        "real_weeks": int(real_weeks),
        "real_X": real_X,
        "real_y": real_y,
        "real_dates": real_dates,
        # audit Stage 1.1 (Task #13.c) — KDCA threshold 산출 input
        "viral_positivity_train": _viral_pos_train,
    }


# back-compat aliases (2026-06-02 semantic rename — 옛 run_phaseN)
run_phase1 = run_data
