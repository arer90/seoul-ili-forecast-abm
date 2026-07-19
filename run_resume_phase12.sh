#!/usr/bin/env bash
# Deprecation alias for run_pipeline.sh (2026-05-28).
#
# 사용자 명시 (2026-05-28): "run_resume_phase12 이름 때문에 매번 헷갈림 — 이름 바꿔줘".
# 새 이름 = run_pipeline.sh (전체 pipeline: train + eval + real_eval + world).
#
# 본 file = backward-compat wrapper. 모든 caller 자동 redirect.
# 단계적 deprecation: 다음 sprint 에 caller 갱신 후 본 file 제거.

echo "⚠ [DEPRECATION] run_resume_phase12.sh → run_pipeline.sh 로 변경됨 (2026-05-28)."
echo "  새 사용: bash run_pipeline.sh $*"
echo "  본 alias 는 backward-compat — 자동 redirect 진행."
echo ""

exec bash "$(dirname "$0")/run_pipeline.sh" "$@"
