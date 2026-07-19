# R Verification — 논문 방어용 canonical 통계 검증

**목적**: Python pipeline 의 결과를 표준 R 패키지로 교차검증. 논문 reviewer 의
"왜 Python 구현을 믿어야 하나?" 질문에 대한 정량적 답.

**실행 환경**: R ≥ 4.3. Python 학습과 독립 — post-E CSV 만 읽음.

## 1. Install (1회)

```r
install.packages(c(
 "tseries", # ADF/KPSS
 "FinTS", # ArchTest
 "forecast", # dm.test, tbats, ets, Box.test
 "scoringutils", # WIS / interval score / PIT
 "scoringRules", # CRPS sample
 "MASS", # glm.nb
 "AER", # dispersiontest
 "EpiEstim", # Rt estimation (Cori 2013)
 "segmented" # ITS breakpoint regression
))
```

## 2. 입력 파일 (post-E 에서 생성 필요)

모두 `simulation/results/post_E/` 하위 CSV:

| 파일 | 컬럼 | 스크립트 |
|------|------|----------|
| `ili_series.csv` | `week_start, ili_rate` | 01, 05, 06, 07, 08 |
| `model_residuals.csv` | `model, week_start, residual` | 02 |
| `model_predictions.csv` | `model, week_start, y_true, y_pred, regime` | 03 |
| `pi_samples.csv` | `model, week_start, y_true, q025, q500, q975` (optional `q100...q900` quantiles) | 04 |
| `rt_seir_v2.csv` | `week_start, rt_eff` (from `SEIRV2ForcedForecaster.rt_effective_trajectory()`) | 06 |
| `npi_window.csv` | `event, iso_date` (rows: `npi_start`=2020-03-02, `npi_end`=2022-12-26) | 07 |

> `post_E_comprehensive_eval.py` 가 이 CSV 들을 dump 하도록 구현 필요 (R3-1 의 일부).

## 3. 전체 실행

```bash
cd simulation/r_verification
Rscript run_all.R
```

**백그라운드 실행** (Linux/macOS/Git-Bash):
```bash
nohup Rscript run_all.R > r_verification.log 2>&1 &
```

**Windows PowerShell**:
```powershell
Start-Process Rscript -ArgumentList "run_all.R" -RedirectStandardOutput "r_verification.log" -NoNewWindow
```

## 4. 개별 실행

```bash
Rscript 01_stationarity.R
Rscript 02_residual_diag.R
# ...
```

기본 입력/출력 경로는 각 스크립트 상단에 명시. 커스텀 경로는 인자로:
```bash
Rscript 01_stationarity.R <input.csv> <output.csv>
```

## 5. 출력

모두 `simulation/r_verification/results/`:
- `01_stationarity.csv` — ADF, KPSS 결과표
- `02_residual_diag.csv` — 모델별 Ljung-Box/ARCH 통계량
- `03_dm_canonical.csv` — forecast::dm.test p-value (regime 별)
- `04_wis_crps_pit.csv` + `04_pit_histogram.pdf`
- `05_nb_dispersion.csv` — α 추정 + Cameron-Trivedi p-value
- `06_rt_epiestim.csv` + `06_rt_overlay.pdf` — Cori Rt + SEIR-V2 overlay
- `07_its_segmented.csv` — breakpoint 회귀 slope/intercept
- `08_tbats_ets.csv` — univariate baseline metrics vs Python 모델

## 6. 왜 각 스크립트가 필요한가 (paper argument)

| 스크립트 | 논문에서 인용할 문장 |
|---------|---------------------|
| 01 | "Seoul ILI 주별 시계열은 ADF p=<value> / KPSS p=<value> 로 비정상성(non-stationary)을 확인" |
| 02 | "모델별 residual 의 Ljung-Box p > 0.05 (자기상관 없음) + ArchTest 검정으로 이분산 확인" |
| 03 | "Python DM 구현을 `forecast::dm.test` (Diebold & Mariano 1995) 로 재현, 모든 regime 에서 결과 일치" |
| 04 | "WIS 와 CRPS 를 `scoringutils` / `scoringRules` (Bracher 2021) 표준 구현으로 산출" |
| 05 | "MASS::glm.nb 로 NB dispersion α 추정, AER::dispersiontest p<0.001 로 Poisson 기각" |
| 06 | "`EpiEstim::estimate_R` (Cori 2013, SI mean=2.6d) 의 Rt 와 SEIR-V2 β(t)/γ·S/N 궤적이 post-COVID 구간에서 중첩" |
| 07 | "`segmented::segmented` ITS 로 2020-03-02 NPI 도입 시점의 slope/level 변화를 정량화" |
| 08 | "TBATS / ETS univariate baseline 대비 앙상블 RMSE 개선율 X%" |
