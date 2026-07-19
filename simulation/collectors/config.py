# simulation/collectors/config.py
# Shim so legacy collectors can do: from ..config import KEYS, SEOUL_BASE, ...
# Re-exports everything from simulation.database.config + collector-specific constants

from simulation.database.config import (
    KEYS, DB_PATH, API_MAX_RETRY as MAX_RETRY, RETRY_WAIT,
    SEOUL_GU_KR, _PKG_ROOT,
)

# --- Collector-specific constants ---
SEOUL_BASE = "http://openapi.seoul.go.kr:8088"
TIMEOUT = 60

# S-DOT district code mapping (gu name -> code)
SDOT_CGG_MAP = {
    "jongno": "11010", "jung": "11020", "yongsan": "11030",
    "seongdong": "11040", "gwangjin": "11050", "dongdaemun": "11060",
    "jungnang": "11070", "seongbuk": "11080", "gangbuk": "11090",
    "dobong": "11100", "nowon": "11110", "eunpyeong": "11120",
    "seodaemun": "11130", "mapo": "11140", "yangcheon": "11150",
    "gangseo": "11160", "guro": "11170", "geumcheon": "11180",
    "yeongdeungpo": "11190", "dongjak": "11200", "gwanak": "11210",
    "seocho": "11220", "gangnam": "11230", "songpa": "11240",
    "gangdong": "11250",    # Korean names
    "\uc885\ub85c\uad6c": "11010", "\uc911\uad6c": "11020",
    "\uc6a9\uc0b0\uad6c": "11030", "\uc131\ub3d9\uad6c": "11040",
    "\uad11\uc9c4\uad6c": "11050", "\ub3d9\ub300\ubb38\uad6c": "11060",
    "\uc911\ub791\uad6c": "11070", "\uc131\ubd81\uad6c": "11080",
    "\uac15\ubd81\uad6c": "11090", "\ub3c4\ubd09\uad6c": "11100",
    "\ub178\uc6d0\uad6c": "11110", "\uc740\ud3c9\uad6c": "11120",
    "\uc11c\ub300\ubb38\uad6c": "11130", "\ub9c8\ud3ec\uad6c": "11140",
    "\uc591\ucc9c\uad6c": "11150", "\uac15\uc11c\uad6c": "11160",
    "\uad6c\ub85c\uad6c": "11170", "\uae08\ucc9c\uad6c": "11180",
    "\uc601\ub4f1\ud3ec\uad6c": "11190", "\ub3d9\uc791\uad6c": "11200",
    "\uad00\uc545\uad6c": "11210", "\uc11c\ucd08\uad6c": "11220",
    "\uac15\ub0a8\uad6c": "11230", "\uc1a1\ud30c\uad6c": "11240",
    "\uac15\ub3d9\uad6c": "11250",
}