# 비교표 — 컬러 / 흑백 버전

참조 논문 스타일(그룹 헤더 + Δ 개선행 + bold-best + ✗)의 비교표 2종.

| 파일 | 표 | 비고 |
|------|----|----|
| `Table_C1c_metric_comparison_color.png` | Table C.1c (Metric) | **논문 docx에 현재 임베드됨** (Appendix C) |
| `Table_C1c_metric_comparison_BW.png` | Table C.1c (Metric) | 흑백 인쇄용 대체본 |
| `Table_J8_aria_comparison_color.png` | Table J.8 (ARIA) | **논문 docx에 현재 임베드됨** (Appendix J) |
| `Table_J8_aria_comparison_BW.png` | Table J.8 (ARIA) | 흑백 인쇄용 대체본 |

- **현재 논문**: 컬러 버전이 본문에 들어가 있음 (디지털 제출용).
- **흑백 버전**: 색상 대신 교차 회색 음영(그룹 구분) + 굵기(best/Ours) + ±부호(Δ 방향). 흑백 인쇄 시 정보 손실 없음.
- **흑백으로 교체하려면**: 알려주시면 docx의 임베드 이미지(image48/49) + 네이티브 표(Table C.1c/J.8)를 흑백으로 swap합니다 (수치·구조 동일).

생성: 데이터=`simulation/results/per_model_eval/per_model_metrics.csv` (metric), `simulation/results/aria_grounding*.json` (ARIA). 모든 수치 verbatim.
