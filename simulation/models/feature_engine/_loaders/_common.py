"""Shared loader infrastructure — encoding-safe congestion-text → numeric mapping.

Sprint β Item 5 full migration (Gemini analysis 2026-05-26):
Used by realtime domain (`_load_rt_population`, `_load_rt_spatial_aggregation`,
`_load_rt_temporal_patterns`) and dict-merged inside `_load_rt_population_forecast`.

Handles both population-endpoint ("여유"/"보통"/"약간 붐빔"/"붐빔") and
road-endpoint ("원활"/"약간혼잡"/"혼잡"/"심각혼잡") variants. Uses Unicode
codepoints (`\\uXXXX` literals) as fallback to survive `.py` file encoding
corruption — DO NOT auto-format to literal Korean characters (R4 risk in
Gemini split plan).
"""
from __future__ import annotations


_CONG_MAP_PRIMARY = {"여유": 1.0, "보통": 2.0, "약간 붐빔": 3.0, "붐빔": 4.0}
_CONG_MAP_ROAD    = {"원활": 1.5, "약간혼잡": 3.0, "혼잡": 4.0, "심각혼잡": 5.0}

# Unicode codepoint fallback (survives if .py file encoding is corrupted).
# Preserved as `\\uXXXX` literals — DO NOT auto-format to literal Korean (R4
# Gemini plan: was added specifically to survive past encoding incidents).
_CONG_MAP_UNICODE = {
    "\uc5ec\uc720": 1.0,                              # 여유
    "\ubcf4\ud1b5": 2.0,                              # 보통
    "\uc57d\uac04 \ubd90\ube54": 3.0,               # 약간 붐빔
    "\ubd90\ube54": 4.0,                              # 붐빔
    "\uc6d0\ud65c": 1.5,                              # 원활
    "\uc57d\uac04\ud63c\uc7a1": 3.0,                # 약간혼잡
    "\ud63c\uc7a1": 4.0,                              # 혼잡
    "\uc2ec\uac01\ud63c\uc7a1": 5.0,                # 심각혼잡
}


def _safe_congestion_score(text, default=None):
    """Map Korean congestion text to numeric score, encoding-safe.

    Args:
        text: Korean congestion label or None.
        default: returned when text is None or unmappable.

    Returns:
        Float score (1.0–5.0) or `default`.
    """
    if text is None:
        return default
    t = str(text).strip()
    # Try primary maps first
    if t in _CONG_MAP_PRIMARY:
        return _CONG_MAP_PRIMARY[t]
    if t in _CONG_MAP_ROAD:
        return _CONG_MAP_ROAD[t]
    # Try with spaces removed
    t_ns = t.replace(" ", "")
    for m in (_CONG_MAP_PRIMARY, _CONG_MAP_ROAD):
        for k, v in m.items():
            if k.replace(" ", "") == t_ns:
                return v
    # Unicode fallback
    if t in _CONG_MAP_UNICODE:
        return _CONG_MAP_UNICODE[t]
    return default


__all__ = [
    "_CONG_MAP_PRIMARY", "_CONG_MAP_ROAD", "_CONG_MAP_UNICODE",
    "_safe_congestion_score",
]
