#!/usr/bin/env bash
# 전체 DB → web aggregate 재생성 (NegBinGLM 기준, 끊이지 않게).
#
# 사용자 요구: champion 은 그대로, NegBinGLM 을 기준으로 DB 에서 web 까지 전부 다시 — 끊김 없이.
# 설계: 각 단계가 독립 graceful + per-step 타임아웃(perl alarm). 한 빌더가 실패/행(hang)해도
#       전체가 멈추지 않고 마지막 산출을 유지한 채 다음 단계로 진행. 종료 시 OK/FAIL/TIMEOUT 요약.
#
# Web forecast 는 .pt 를 안 읽고 build_production_forecast 가 DB 에서 NegBinGLM 을 전체 refit →
# 미래 1-step 예측(source='production-refit-forecast'). champion 재선정/재학습 없음.
#
# Usage:  bash web/scripts/rebuild_all_web.sh           # 전체
#         STEP_TIMEOUT=600 bash web/scripts/rebuild_all_web.sh   # 단계 타임아웃 조정
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY="${PY:-.venv/bin/python}"
TMO="${STEP_TIMEOUT:-300}"
LOG="web/scripts/.rebuild_all_web.log"
: > "$LOG"
OK=0; FAIL=0; TMOUT=0; FAILED_STEPS=""

# per-step 타임아웃 실행(macOS: timeout 없음 → perl alarm). graceful: 실패해도 계속.
run_step() {
  local label="$1"; shift
  printf "  ▶ %-32s " "$label"
  local t0=$(date +%s)
  perl -e 'alarm shift @ARGV; exec @ARGV or exit 127' "$TMO" "$@" >>"$LOG" 2>&1
  local rc=$?
  local dt=$(( $(date +%s) - t0 ))
  if [ $rc -eq 0 ]; then echo "✓ OK (${dt}s)"; OK=$((OK+1))
  elif [ $rc -eq 142 ] || [ $rc -eq 124 ]; then echo "⏱ TIMEOUT(${TMO}s) — 마지막 산출 유지"; TMOUT=$((TMOUT+1)); FAILED_STEPS="$FAILED_STEPS $label(TMO)"
  else echo "✗ FAIL rc=$rc (${dt}s) — 마지막 산출 유지"; FAIL=$((FAIL+1)); FAILED_STEPS="$FAILED_STEPS $label(rc$rc)"; fi
}

echo "=== 전체 DB→web 재생성 시작 $(date -u +%Y-%m-%dT%H:%MZ) (step timeout=${TMO}s) ==="

echo "[1/5] 정적 지오 레이어"
run_step "seoul_boundary"   $PY web/scripts/build_seoul_boundary.py
run_step "subway_lines"     $PY web/scripts/build_subway_lines.py
run_step "subway_stations"  $PY web/scripts/build_subway_stations.py
run_step "bus_routes"       $PY web/scripts/build_bus_routes.py
run_step "bus_stops"        $PY web/scripts/build_bus_stops.py
run_step "schools"          $PY web/scripts/build_schools.py

echo "[2/5] DB 기반 데이터 레이어"
run_step "disease_vax"      $PY web/scripts/build_disease_vax.py
run_step "multi_disease"    $PY web/scripts/build_multi_disease.py
run_step "agent_trips"      $PY web/scripts/build_agent_trips.py
run_step "gu_weights"       $PY web/scripts/build_gu_weights.py
# trained_models = 학습 산출(per_model_metrics.csv) 의존 — 재학습 없으면 FAIL 후 마지막 trained-models.json(53) 유지(정상)
run_step "trained_models"   $PY web/scripts/build_trained_models.py
run_step "backtest"         $PY web/scripts/build_backtest.py
run_step "model_metrics_full" $PY web/scripts/build_model_metrics_full.py
run_step "aria_wiki"        $PY web/scripts/build_aria_wiki.py
run_step "age_seir"         $PY web/scripts/build_age_seir.py

echo "[3/5] 환경·실시간(네트워크 graceful)"
run_step "weather"          $PY web/scripts/build_weather.py
run_step "air_env"          $PY web/scripts/build_air_env.py
run_step "realtime_poi"     $PY web/scripts/build_realtime_poi.py
run_step "live_overlays"    $PY web/scripts/build_live_overlays.py

echo "[4/5] 외부신호 → 모드게이트 → NegBinGLM forecast 체인(순서 중요)"
run_step "external_risk"    $PY web/scripts/build_external_risk.py
run_step "resolve_mode"     $PY web/scripts/resolve_mode.py
run_step "airport_arrivals" $PY web/scripts/build_airport_arrivals.py
run_step "gu_weights(재)"   $PY web/scripts/build_gu_weights.py
run_step "production_fcst"  $PY web/scripts/build_production_forecast.py
run_step "seir_metapop_init" $PY simulation/scripts/export_seir_metapop_init.py
run_step "multi_model_seir360" $PY web/scripts/_build_multi_model_forecast.py
run_step "spatialize_seir360" $PY web/scripts/spatialize_seir_forecast.py
run_step "horizon_reliab"   $PY web/scripts/horizon_reliability.py
run_step "horizon_forecast" $PY web/scripts/build_horizon_forecast.py

echo "[5/5] 검증 게이트 (dual-file sync + jsdom + 사실 TDD)"
run_step "sync_app"         node web/scripts/sync_app.mjs
run_step "smoke_render"     node web/scripts/smoke_render.mjs

echo ""
echo "=== 재생성 완료: ${OK} OK / ${FAIL} FAIL / ${TMOUT} TIMEOUT ==="
[ -n "$FAILED_STEPS" ] && echo "    문제 단계:$FAILED_STEPS (각각 마지막 산출 유지 — 전체 중단 X)"
echo "    상세 로그: $LOG"
# 핵심 forecast 산출이 갱신됐는지 한 줄 확인
$PY -c "import json,datetime;d=json.load(open('web/public/aggregates/ili-forecast.json'));print('    ili-forecast:',d.get('source'),d.get('model'),'city=',round(d.get('city_forecast',0),2),'gen=',d.get('generated_at','')[:16])" 2>/dev/null || true
exit 0
