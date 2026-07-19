#!/usr/bin/env python3
"""L1 모드 게이트 — external-risk + 최근 ILI → SEASONAL / WATCH / PANDEMIC (hysteresis).

설계 SSOT: docs/PANDEMIC_MODE_DESIGN_20260610.md (적대검증 반영). 핵심 규칙:
  - PANDEMIC = KDCA 위기경보 경계(3)↑ **단독 hard 트리거** (FP≈0, 권위) → 즉시 전환.
  - WATCH = DON 호흡기-novel 확증 OR GDELT 뉴스 spike OR trajectory surge(3주↑) — 소프트신호.
    기계론 엔진 병렬계산 + 경보만, nowcast 화면 유지. 2주 연속 진입(깜빡임 격리).
  - SEASONAL = 평시. 외부신호·기계론 엔진 호출 0.
  - hysteresis: 진입 = WATCH 2주(PANDEMIC-via-KDCA 는 즉시) / 해제 = 4주 연속 하향.
정직성: "뉴스가 로컬 ILI 보다 선행"은 본 저장소 데이터로 미검증 → 주장 아닌 **라우팅 근거**로만.
  surge(trajectory)는 1차 아닌 2차(WATCH) 안전망. PANDEMIC 비가역 라우팅은 KDCA-only 로 격리.

Read-only(write mode-state.json). Run: .venv/bin/python web/scripts/resolve_mode.py
"""
from __future__ import annotations

import json
import logging
import datetime
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AGG = ROOT / "web" / "public" / "aggregates"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "web" / "scripts"))
logging.disable(logging.INFO)

MODE_NAMES = {0: "SEASONAL", 1: "WATCH", 2: "PANDEMIC"}
ENTER_WATCH_WEEKS = 2          # 소프트 신호 WATCH 진입
DEESCALATE_WEEKS = 4           # 하향 해제


def routing_manifest(mode_ord: int) -> dict:
    """모드 → web/ARIA 가 읽는 명시 라우팅 계약 (codex/gemini: dead-end 해소).

    primary = 권위 forecast 아티팩트. PANDEMIC 은 ML 외삽 불가라 기계론(seir-forecast-360)이 primary.
    """
    if mode_ord >= 2:
        return {"primary": "seir-forecast-360.json", "secondary": "abm-scenarios.json",
                "fallback": "ili-forecast.json", "alert_level": "pandemic",
                "alert": "대규모 감염 국면(PANDEMIC) — 기계론 SEIR/ABM 우선 표시, ML nowcast 는 참고용(외삽 불가)"}
    if mode_ord == 1:
        return {"primary": "ili-forecast.json", "secondary": "seir-forecast-360.json", "fallback": None,
                "alert_level": "watch",
                "alert": "외부 위험신호 상승(WATCH) — ML nowcast 유지 + 기계론 SEIR/ABM 시나리오 병렬 가동·경보 주시"}
    return {"primary": "ili-forecast.json", "secondary": None, "fallback": None,
            "alert_level": "none", "alert": ""}


def _detect_surge(recent_ili) -> bool:
    """trajectory 2차 신호: 최근 4주 중 3주 연속 상승 + 3주간 ≥50% 성장."""
    r = np.asarray(recent_ili, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 4:
        return False
    rising = all(r[-i] > r[-i - 1] for i in range(1, 4))
    growth = r[-1] / max(r[-4], 1e-6)
    return bool(rising and growth >= 1.5)


def resolve_mode_raw(external_risk: dict, recent_ili) -> dict:
    """external-risk + 최근 ILI → raw 모드 결정(hysteresis 적용 전).

    Args:
        external_risk: external-risk.json dict (kdca_alert_level, summary{...}).
        recent_ili: 최근 관측 ILI 시퀀스(oldest→newest).

    Returns:
        {raw_ord(0/1/2), raw_mode, reason, kdca_hard(bool), signals{...}}.

    Side effects: none.
    """
    kdca = int(external_risk.get("kdca_alert_level", 0))
    s = external_risk.get("summary", {})
    novel = bool(s.get("respiratory_novel_confirmed", False))
    spike = bool(s.get("news_spike", False))
    stale = bool(s.get("kdca_stale", False))      # 권위 신호 stale → fail-safe WATCH
    # 입국 유입압 = 팬데믹 seeding **context** 전용. 독립 WATCH 트리거 아님(정상시 관광 회복으로
    # 고분산·정반대 계절성이라 단독 발화 시 FP — 조사 결론). 이미 켜진 경보의 유입 근거만 부가.
    arrivals_pressure = bool(s.get("arrivals_pressure_high", False))
    surge = _detect_surge(recent_ili)
    kdca_hard = kdca >= 3

    if kdca_hard:
        raw, reason = 2, f"KDCA 위기경보 {kdca}(경계↑) — hard PANDEMIC 트리거(권위)"
    elif novel or spike or surge or stale:
        why = [w for w, ok in (("DON novel확증", novel), ("뉴스 spike", spike),
                               ("trajectory surge", surge),
                               ("KDCA 신호 stale(검증필요)", stale)) if ok]
        raw, reason = 1, "WATCH 소프트신호: " + ", ".join(why) + " — 기계론 병렬+경보, nowcast 유지"
    else:
        raw, reason = 0, "평시 — seasonal nowcast 단독"
    # 신종 경보가 이미 켜졌고 입국 유입압도 높으면 seeding 근거 부가(트리거는 안 바꿈).
    if raw >= 1 and arrivals_pressure:
        reason += " (+ 입국 유입압↑ — importation seeding 가중)"
    return {"raw_ord": raw, "raw_mode": MODE_NAMES[raw], "reason": reason,
            "kdca_hard": bool(kdca_hard),
            "signals": {"kdca": kdca, "don_novel": novel, "news_spike": spike,
                        "surge": surge, "kdca_stale": stale, "arrivals_pressure": arrivals_pressure}}


def _as_date(d) -> datetime.date:
    if isinstance(d, datetime.date):
        return d
    return datetime.date.fromisoformat(str(d)[:10])


def apply_hysteresis(prev_state: dict, raw_ord: int, kdca_hard: bool,
                     current_date: "datetime.date | str") -> dict:
    """깜빡임 격리 (date-based): PANDEMIC-via-KDCA 즉시, WATCH 진입 2주, 하향 해제 4주.

    안전(codex/gemini): 경과 '주'를 **호출 횟수가 아니라 날짜 차이**로 계산 → daily/주간 어떤
    cron 빈도에도 '2주' 가 2회 호출로 붕괴하지 않음.

    Args:
        prev_state: 직전 mode-state {mode_ord, pending_ord, pending_since(ISO)}.
        raw_ord: 이번 raw 결정 ordinal.
        kdca_hard: KDCA 경계↑ 여부(즉시 PANDEMIC).
        current_date: 이번 실행 날짜(외부신호 generated_at 또는 실행일).

    Returns:
        {mode_ord, mode, pending_ord, pending_since, pending_weeks, switched(bool)}.
    """
    prev = int(prev_state.get("mode_ord", 0))
    pending = int(prev_state.get("pending_ord", prev))
    cur = _as_date(current_date)
    since_raw = prev_state.get("pending_since")
    if raw_ord != pending or not since_raw:
        pending = raw_ord
        since = cur
    else:
        try:
            since = _as_date(since_raw)
        except Exception:
            since = cur
    weeks_pending = max(1, (cur - since).days // 7 + 1)    # 경과 주(날짜 기반)

    new = prev
    if kdca_hard and raw_ord == 2:
        new = 2                                  # 즉시 PANDEMIC
    elif pending > prev and weeks_pending >= ENTER_WATCH_WEEKS:
        new = pending                            # 상향(WATCH) 2주
    elif pending < prev and weeks_pending >= DEESCALATE_WEEKS:
        new = pending                            # 하향 4주
    return {"mode_ord": new, "mode": MODE_NAMES[new], "pending_ord": pending,
            "pending_since": since.isoformat(), "pending_weeks": weeks_pending,
            "switched": bool(new != prev)}


def main() -> int:
    er_path = AGG / "external-risk.json"
    if not er_path.is_file():
        print("external-risk.json 없음 — build_external_risk.py 먼저 실행"); return 1
    er = json.loads(er_path.read_text(encoding="utf-8"))

    # 최근 ILI (feature matrix)
    try:
        from build_production_forecast import _load_feature_matrix
        _X, y_all, _fc, _ws = _load_feature_matrix()
        recent = list(np.asarray(y_all, float)[-4:])
    except Exception as e:
        recent = []
        print(f"  (최근 ILI 로드 실패 {type(e).__name__} — surge 미평가)")

    prev = {}
    st_path = AGG / "mode-state.json"
    if st_path.is_file():
        try:
            prev = json.loads(st_path.read_text(encoding="utf-8"))
        except Exception:
            prev = {}

    # 현재 날짜 = 외부신호 generated_at(동적) 또는 실행일
    try:
        cur_date = _as_date(er.get("generated_at") or datetime.date.today().isoformat())
    except Exception:
        cur_date = datetime.date.today()
    raw = resolve_mode_raw(er, recent)
    hy = apply_hysteresis(prev, raw["raw_ord"], raw["kdca_hard"], cur_date)

    state = {
        "generated_at": er.get("generated_at", ""),
        "mode": hy["mode"], "mode_ord": hy["mode_ord"],
        "raw_mode": raw["raw_mode"], "raw_ord": raw["raw_ord"],
        "reason": raw["reason"],
        "switched": hy["switched"],
        "pending_ord": hy["pending_ord"], "pending_since": hy["pending_since"],
        "pending_weeks": hy["pending_weeks"],
        "signals": raw["signals"],
        "recent_ili": [round(float(v), 1) for v in recent],
        "routing": ("seasonal nowcast 단독" if hy["mode_ord"] == 0 else
                    "seasonal nowcast 표시 + 기계론 SEIR/ABM 병렬계산 + 경보" if hy["mode_ord"] == 1 else
                    "ML nowcast 중단 → 기계론 SEIR/ABM + ARIA 경보 라우팅"),
        # web/ARIA 가 읽는 명시 라우팅 계약(codex/gemini: dead-end 해소). primary=권위 forecast.
        "routing_manifest": routing_manifest(hy["mode_ord"]),
        "note": "KDCA 경계(3)↑만 hard PANDEMIC(즉시). WATCH=소프트신호 2주진입/4주해제. "
                "surge=2차 안전망(1차 아님). 상세 docs/PANDEMIC_MODE_DESIGN.",
    }
    st_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"=== mode-state.json ===")
    print(f"  raw={raw['raw_mode']} → 유효모드={hy['mode']}"
          + (" (전환!)" if hy['switched'] else "") + f"  pending={MODE_NAMES[hy['pending_ord']]}×{hy['pending_weeks']}주")
    print(f"  근거: {raw['reason']}")
    print(f"  신호: {raw['signals']}  최근ILI={state['recent_ili']}")
    print(f"  라우팅: {state['routing']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
