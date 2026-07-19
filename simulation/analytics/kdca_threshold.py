"""KDCA 공식 epidemic threshold (Kang SK, Son WS, Kim BI 2024) — audit Stage 1.1.

KDCA 공식 정의 (peer-reviewed, 국가 단위 단일값):
    Threshold = mean(ILI rate during non-epidemic periods, past 3 seasons) + 2 * SD
    non-epidemic period = viral positivity < 2% for ≥ 2 consecutive weeks

⚠ Scope limitation (external audit 2026-05-27 critical finding):
    KDCA 공식 threshold 는 **국가단위 단일값** 만 산출. 22 region/시·도 단위
    threshold 는 KDCA 공식 산출/공표하지 X (KDCA 2025-26 관리지침 + 보도자료).
    Caller 가 region-stratified alert metric 사용 시 외부 표준 부재 — paper
    cross-region 비교는 forecast-only metric (WIS, MAE, PICP) 으로 한정 권장.

References (peer-reviewed, DOI/PMID 우선):
    - Kang SK, Son WS, Kim BI (2024)
      "Application of the Time Derivative (TD) Method for Early Alert of
       Influenza Epidemics"
      J Korean Med Sci 39(4):e40. doi:10.3346/jkms.2024.39.e40
      PMID 38288541
      Methods verbatim: "The national influenza epidemic threshold is established
      each season using the epidemic formula. The epidemic formula for establishing
      the national influenza threshold utilizes the formula provided by the U.S. CDC."
      "The Epidemic Threshold = (Mean of the ILI Rates during Non-Epidemic Periods
       over the Previous 3 Years) + (2 × S.D.)"
      "Non-epidemic period: A period when the influenza viral detection rate is
       less than 2% and lasts for more than 2 weeks."

    - Lee J, Huh S, Seo H (2024)
      "Influenza surveillance in Korea: history and current status"
      Ewha Med J 47(2):e24. doi:10.12771/emj.2024.e24
      ("Korea uses data from the past 3 seasons; U.S. uses the past 2 seasons")

    - U.S. CDC ancestor methodology (1.64 SD cyclic regression):
      Serfling RE (1963) Public Health Rep 78(6):494-506, PMID 19316455

KDCA 공시 임계 cross-verification (season-by-season, season auto-detect 필요):
    - 2023-24 시즌: 6.5 cases/1,000  (KDCA bulletin)
    - 2024-25 시즌: 8.6 cases/1,000  (KDCA 2025-06-13 보도자료, "'24-'25 절기")
    - 2025-26 시즌: 9.1 cases/1,000  (KDCA 2025-10-17 보도자료, "'25~'26절기")

Audit history:
    - Audit #1 (2026-05-27): q70 → KDCA mean+2SD 변경 (audit Stage 1.1).
    - Audit #2 (2026-05-27, external critical re-audit):
      (i) 1차 인용 misattribution 수정: Park HJ → Kang SK, Son WS, Kim BI.
      (ii) hardcoded 8.6 (2024-25 only) → season dict + auto-detect.
      (iii) region-stratified application 의 external reference 부재 명시.

D-5 gray-box contract:
    - 모든 public function: NaN-safe, raise X (single point of failure 회피)
    - viral_positivity 없으면 fallback (q70-lowest-70% proxy) 사용 + method
      반환 키 "kdca" / "fallback_q70" 명시
    - test: simulation/tests/test_kdca_threshold.py
"""
from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = [
    "compute_kdca_epidemic_threshold",
    "identify_non_epidemic_weeks",
    "detect_current_season_threshold",
    "_detect_current_season",
    "get_kdca_season_reference",
    "WEEKS_PER_SEASON",
    "DEFAULT_N_SEASONS",
    "DEFAULT_POSITIVITY_THRESHOLD",
    "DEFAULT_MIN_CONSEC_WEEKS",
    "KDCA_THRESHOLDS_BY_SEASON",
    "KDCA_LATEST_KNOWN_THRESHOLD",
    "KDCA_DEFAULT_THRESHOLD_2024_25",  # backward-compat (deprecated)
    "KDCA_DEFAULT_THRESHOLD_2025_26",  # backward-compat (deprecated)
]

WEEKS_PER_SEASON: int = 52
DEFAULT_N_SEASONS: int = 3  # KDCA = 3 (Kang et al. 2024); U.S. CDC = 2
DEFAULT_POSITIVITY_THRESHOLD: float = 0.02  # KDCA 2% (Kang et al. 2024)
DEFAULT_MIN_CONSEC_WEEKS: int = 2

# KDCA 공시 임계 (paper cross-verification target, season auto-detect 대상)
# audit 2026-05-27: hardcoded 8.6 (2024-25 only) → season dict + auto-detect.
# Source: KDCA Influenza weekly surveillance bulletin (kdca.go.kr/bbs/eng/192/...).
KDCA_THRESHOLDS_BY_SEASON: dict[str, float] = {
    "2023-24": 6.5,   # KDCA bulletin
    "2024-25": 8.6,   # KDCA 2025-06-13 보도자료 "'24-'25 절기 인플루엔자 유행기준: 8.6명"
    "2025-26": 9.1,   # KDCA 2025-10-17 보도자료 "'25~'26절기 인플루엔자 유행기준(9.1명)"
    # 신규 시즌 추가 시 본 dict update 필요 (KDCA 발표 후 6-12주 lag).
}

# Backward-compat constants (deprecated — use detect_current_season_threshold() instead)
KDCA_DEFAULT_THRESHOLD_2024_25: float = KDCA_THRESHOLDS_BY_SEASON["2024-25"]
KDCA_DEFAULT_THRESHOLD_2025_26: float = KDCA_THRESHOLDS_BY_SEASON["2025-26"]

# Fallback when season unknown (use the most recent known)
KDCA_LATEST_KNOWN_THRESHOLD: float = KDCA_THRESHOLDS_BY_SEASON[
    sorted(KDCA_THRESHOLDS_BY_SEASON.keys())[-1]
]


def _detect_current_season(date: Optional["object"] = None) -> str:
    """Auto-detect current ILI season (KR convention: Sep-Aug).

    Season N-(N+1) covers Sep N month through Aug (N+1).
    - If current month ∈ [9, 12]: season starts current year (current-next)
    - If current month ∈ [1, 8]: season started previous year (prev-current)

    Args:
        date: optional datetime; default = today.

    Returns:
        str — "YYYY-YY" format, e.g. "2025-26".
    """
    if date is None:
        from datetime import datetime
        date = datetime.now()
    y = date.year
    m = date.month
    if m >= 9:
        return f"{y}-{(y + 1) % 100:02d}"
    else:
        return f"{y - 1}-{y % 100:02d}"


def detect_current_season_threshold(
    *, date: Optional["object"] = None, fallback: Optional[float] = None,
) -> dict:
    """Season auto-detect + KDCA published threshold lookup.

    audit 2026-05-27 fix: hardcoded fallback 8.6 (2024-25 only) → season-aware.

    Returns:
        dict {
            "season": str (e.g. "2025-26"),
            "threshold": float (KDCA published) or fallback or NaN,
            "source": "kdca_published" | "fallback_arg" | "fallback_latest_known" | "unknown",
        }
    """
    season = _detect_current_season(date)
    if season in KDCA_THRESHOLDS_BY_SEASON:
        return {"season": season, "threshold": KDCA_THRESHOLDS_BY_SEASON[season],
                "source": "kdca_published"}
    if fallback is not None and np.isfinite(fallback):
        return {"season": season, "threshold": float(fallback), "source": "fallback_arg"}
    # final fallback: latest known
    return {"season": season, "threshold": KDCA_LATEST_KNOWN_THRESHOLD,
            "source": "fallback_latest_known"}


def identify_non_epidemic_weeks(
    viral_positivity: np.ndarray,
    *,
    positivity_threshold: float = DEFAULT_POSITIVITY_THRESHOLD,
    min_consec_weeks: int = DEFAULT_MIN_CONSEC_WEEKS,
) -> np.ndarray:
    """KDCA non-epidemic period mask (weekly).

    KDCA 정의: viral positivity < 2% for ≥ 2 consecutive weeks.

    Args:
        viral_positivity: weekly viral positivity (n,) ∈ [0, 1] (예: 0.015 = 1.5%)
                          NaN 허용 — NaN 주는 non-epidemic 으로 간주 (보수적).
        positivity_threshold: 0.02 (= 2%, KDCA 표준)
        min_consec_weeks: 2 (KDCA 표준)

    Returns:
        bool mask (n,) — True = 그 주가 non-epidemic period 의 일부.

    Performance: O(n) time, O(n) memory.
    Side effects: 없음 (pure function).
    Caller responsibility: viral_positivity 는 0-1 scale (백분율 아님).
    """
    if viral_positivity is None or len(viral_positivity) == 0:
        return np.zeros(0, dtype=bool)

    pos = np.asarray(viral_positivity, dtype=np.float64)
    n = len(pos)

    # NaN-safe: NaN → non-epidemic 으로 (보수적, threshold 산출에 더 많은 weeks)
    is_low = np.where(np.isnan(pos), True, pos < positivity_threshold)

    if min_consec_weeks <= 1:
        return is_low

    # ≥ min_consec_weeks 연속 low positivity 만 non-epidemic 으로 간주
    mask = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if is_low[i]:
            j = i
            while j < n and is_low[j]:
                j += 1
            run_len = j - i
            if run_len >= min_consec_weeks:
                mask[i:j] = True
            i = j
        else:
            i += 1

    return mask


def compute_kdca_epidemic_threshold(
    y_train_pool: np.ndarray,
    *,
    viral_positivity_train: Optional[np.ndarray] = None,
    n_seasons: int = DEFAULT_N_SEASONS,
    positivity_threshold: float = DEFAULT_POSITIVITY_THRESHOLD,
    min_consec_weeks: int = DEFAULT_MIN_CONSEC_WEEKS,
    fallback_q: float = 0.70,
) -> dict:
    """KDCA 공식 epidemic threshold (audit Stage 1.1, Task #13).

    Threshold = mean(ILI during non-epidemic, past N seasons) + 2 * SD

    Kang SK, Son WS, Kim BI (2024) J Korean Med Sci 39(4):e40, doi:10.3346/jkms.2024.39.e40 (PMID 38288541):
        "The Epidemic Threshold = (Mean of the ILI Rates during Non-Epidemic
         Periods over the Previous 3 Years) + (2 × S.D.) — Non-epidemic period:
         A period when the influenza viral detection rate is less than 2% and
         lasts for more than 2 weeks."

    Args:
        y_train_pool: weekly ILI rate (n,) per 1,000 (training pool only,
                      leakage-free — test 정보 포함 X).
        viral_positivity_train: weekly viral positivity (n,) ∈ [0, 1] (예: 0.015).
                                None 시 fallback (q70-lowest proxy) 사용.
        n_seasons: 과거 시즌 수 (default 3 = KDCA 표준).
        positivity_threshold: 0.02 = 2% (KDCA 표준).
        min_consec_weeks: 2 (KDCA 표준).
        fallback_q: viral_positivity 없을 때 lowest q (default 0.70 = q70-lowest)
                    의 proxy 산출 — sensitivity analysis 보조.

    Returns:
        dict {
            "threshold": float,         # primary KDCA threshold (mean+2SD on non-epi)
            "threshold_q70": float,     # secondary sensitivity (q70 lowest-30% mean+2SD)
            "n_nonepi_weeks": int,      # non-epidemic period 의 주 수 used
            "method": str,              # "kdca" if viral_positivity provided, "fallback_q70" else
            "mean_nonepi": float,
            "sd_nonepi": float,
            "n_seasons_used": int,
            "reference": "Kang SK, Son WS, Kim BI (2024) J Korean Med Sci 39(4):e40, doi:10.3346/jkms.2024.39.e40 (PMID 38288541)",
        }

    Raises:
        절대 raise X — fail 시 NaN 반환 (NaN-safe).

    Performance: O(n) time + O(n_seasons * 52) memory.
    Side effects: 없음 (pure function).
    Caller responsibility:
        - y_train_pool 의 NaN 허용 (제외됨)
        - viral_positivity_train 은 0-1 scale (백분율 아님). 제공 시 KDCA primary 사용.
        - 미제공 시 fallback_q70 — paper 인용 시 "primary alert threshold 는 KDCA
          mean+2SD; viral positivity 데이터 부재 region 에서는 q70 proxy" 명시.
    """
    out = {
        "threshold": float("nan"),
        "threshold_q70": float("nan"),
        "n_nonepi_weeks": 0,
        "method": "fallback_q70",
        "mean_nonepi": float("nan"),
        "sd_nonepi": float("nan"),
        "n_seasons_used": 0,
        "reference": "Kang SK, Son WS, Kim BI (2024) J Korean Med Sci 39(4):e40, doi:10.3346/jkms.2024.39.e40 (PMID 38288541)",
    }

    if y_train_pool is None:
        return out
    y = np.asarray(y_train_pool, dtype=np.float64)
    n = len(y)
    if n == 0:
        return out

    # 과거 N season window cap — 너무 짧으면 모두 사용
    window_max = n_seasons * WEEKS_PER_SEASON
    y_window = y[-window_max:] if n > window_max else y
    out["n_seasons_used"] = int(min(n_seasons, int(np.ceil(len(y_window) / WEEKS_PER_SEASON))))

    # secondary fallback (always computed for sensitivity analysis)
    try:
        finite = y_window[np.isfinite(y_window)]
        if len(finite) >= 4:
            cutoff_q = float(np.quantile(finite, fallback_q))
            nonepi_proxy = finite[finite < cutoff_q]
            if len(nonepi_proxy) >= 2:
                out["threshold_q70"] = float(
                    np.mean(nonepi_proxy) + 2.0 * np.std(nonepi_proxy, ddof=1)
                )
    except Exception:
        pass

    # primary KDCA threshold
    if viral_positivity_train is not None and len(viral_positivity_train) == n:
        pos_window = np.asarray(viral_positivity_train, dtype=np.float64)[-window_max:] \
            if n > window_max else np.asarray(viral_positivity_train, dtype=np.float64)

        non_epi_mask = identify_non_epidemic_weeks(
            pos_window,
            positivity_threshold=positivity_threshold,
            min_consec_weeks=min_consec_weeks,
        )

        # ILI value 가 finite 한 non-epidemic week 만 사용
        usable = non_epi_mask & np.isfinite(y_window)
        n_nonepi = int(usable.sum())
        out["n_nonepi_weeks"] = n_nonepi

        if n_nonepi >= 4:  # 최소 4 weeks 필요 (SD 의미)
            non_epi_y = y_window[usable]
            mean_v = float(np.mean(non_epi_y))
            sd_v = float(np.std(non_epi_y, ddof=1))
            out["mean_nonepi"] = mean_v
            out["sd_nonepi"] = sd_v
            out["threshold"] = mean_v + 2.0 * sd_v
            out["method"] = "kdca"
            return out

    # fallback path (no viral positivity, or insufficient non-epi weeks)
    out["threshold"] = out["threshold_q70"]
    out["method"] = "fallback_q70"
    return out


def get_kdca_season_reference(season: str) -> Optional[float]:
    """KDCA 공시 epidemic threshold (paper cross-verification).

    Args:
        season: "2024-25" or "2025-26" etc.

    Returns:
        float (cases per 1,000) or None if unknown.

    Reference (audit Caveat 2 — paper PDF 원본 직접 확인 권장):
        - 2024-25: KDCA Influenza weekly surveillance report Week 5 (Jan 26 - Feb 1, 2025)
        - 2025-26: KDCA Week 49 (Nov 30 - Dec 6, 2025), 원문: "2025-2026 season
                   epidemic threshold: 9.1 cases (/1,000)"
    """
    table = {
        "2024-25": KDCA_DEFAULT_THRESHOLD_2024_25,  # 8.6
        "2025-26": KDCA_DEFAULT_THRESHOLD_2025_26,  # 9.1
    }
    return table.get(season)
