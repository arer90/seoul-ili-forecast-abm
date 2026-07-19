# Statistical Audit — 예측 + 시뮬레이션 통합 통계 검증

> **Generated**: 2026-07-14 23:29:44
> **Standards**: TRIPOD+AI 2024, EPIFORGE 2020, PROBAST
> **Tests**: Fisher z, Diebold-Mariano, Bootstrap CI, Hansen MCS, Mondrian PICP

---

## 3. TRIPOD+AI 2024 체크리스트

| 항목 | 본 audit 충족 |
|------|:-:|
| 모든 metric 95% CI 보고 | ✅ Fisher z (R²), Bootstrap (RMSE/MAE/MAPE) |
| Pairwise model comparison p-value | ✅ Diebold-Mariano (R6 dm_test + audit) |
| PI calibration 정량 | ✅ PICP@95 + Mondrian per-group |
| Best 모델 선정 통계적 정당성 | ✅ Hansen MCS (모델 confidence set) |
| Epidemiological validity | ✅ Rt CI + EVS 11 components |
| Intervention effect significance | ✅ Wilcoxon paired test |

## 4. 종합 판정

- **예측 모델**: champion = best-WIS; 0/0 DM 유의 (baseline 대비 우수)
- **시뮬레이션**: 0/0 epi-valid (≥4 components)

> champion = 순수 best-WIS (4-criteria/g175 제거 2026-06-05). R²/MAPE/WIS/PICP 는
> 개별 metric 으로 보고; DM 유의 모델은 baseline 대비 통계 우수로 §결과 보고 권장.