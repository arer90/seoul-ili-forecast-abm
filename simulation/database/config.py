# simulation.db.config -- Paths, API keys, constants
# All paths are relative to simulation/ package root
import logging
import os
import shutil
from pathlib import Path
from urllib.parse import quote

from simulation.config_global import GLOBAL  # SSOT (2026-05-28)

log = logging.getLogger(__name__)

# simulation/ package root
_PKG_ROOT = Path(__file__).resolve().parent.parent  # simulation/

# Data directories (inside simulation/)
DATA_DIR = _PKG_ROOT / "data"
DB_DIR = DATA_DIR / "db"
COLLECT_DIR = DATA_DIR / "collected"

# DB path — overridable via EPI_DB_PATH env var so the Railway MCP image
# can point at the smaller simulation/data/db_lite/epi_lite.db (~500 MB)
# while local dev keeps the full simulation/data/db/epi_real_seoul.db
# (~12 GB). Falls back to the original path when no env var set.
DB_PATH = os.environ.get(
    "EPI_DB_PATH",
    str(DB_DIR / "epi_real_seoul.db"),
)

# Ensure directories exist
DB_DIR.mkdir(parents=True, exist_ok=True)
COLLECT_DIR.mkdir(parents=True, exist_ok=True)

# --- Temp staging (2026-04-17) --------------------------------------
# Policy requested by user:
#   1. If the primary disk has >= MIN_STAGING_FREE_GB free, stage locally.
#   2. Otherwise try the configured fallback (MPH_STAGING env, else portable
#      project-local _staging — SSOT GLOBAL.paths.staging).
#   3. If the fallback is also tight, *WARN* and ask the user for a path.
# Nothing here silently switches disks; every fallback emits a loud log
# message so the collector caller can surface it to the user.
MIN_STAGING_FREE_GB = 10  # minimum free space to stage locally
_PRIMARY_STAGING = DATA_DIR / "staging"
# SSOT (2026-05-28): 이전 r"E:\MPH_staging" Windows 하드코딩 제거 → GLOBAL.paths.staging
# (env MPH_STAGING override, no-env default = project_root/simulation/_staging, portable #1).
_FALLBACK_STAGING = GLOBAL.paths.staging


def _free_gb(path: Path) -> float:
    r"""Return free space (GB) on the filesystem containing *path*.

    Returns -1 if the path's drive does not exist yet (e.g. E:\ absent).
    """
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if not probe.exists():
        return -1.0
    try:
        return shutil.disk_usage(probe).free / (1024 ** 3)
    except (FileNotFoundError, OSError):
        return -1.0


def resolve_staging_dir(required_gb: float = MIN_STAGING_FREE_GB) -> Path:
    r"""Pick the best writable staging dir based on free disk space.

    Order of preference:
      1. ``simulation/data/staging/`` (primary) — if free space >= required.
      2. ``E:\MPH_staging`` or ``$MPH_STAGING`` fallback — if primary tight.
      3. Raises ``RuntimeError`` with a user-actionable message if both fail.
    """
    primary_free = _free_gb(_PRIMARY_STAGING.parent)
    if primary_free >= required_gb:
        _PRIMARY_STAGING.mkdir(parents=True, exist_ok=True)
        return _PRIMARY_STAGING

    log.warning(
        "Primary staging disk has only %.1f GB free (need %.1f GB). "
        "Falling back to %s.",
        primary_free, required_gb, _FALLBACK_STAGING,
    )
    fallback_free = _free_gb(_FALLBACK_STAGING)
    if fallback_free >= required_gb:
        _FALLBACK_STAGING.mkdir(parents=True, exist_ok=True)
        return _FALLBACK_STAGING

    raise RuntimeError(
        f"Not enough free disk for staging.\n"
        f"  Primary   {_PRIMARY_STAGING.parent}: {primary_free:.1f} GB free\n"
        f"  Fallback  {_FALLBACK_STAGING}:       "
        f"{fallback_free if fallback_free >= 0 else 'drive not found'}\n"
        f"Set MPH_STAGING to a directory with at least "
        f"{required_gb:.0f} GB free, e.g. MPH_STAGING=D:\\\\mph_tmp"
    )


# Lazily evaluated — collectors call ``resolve_staging_dir()`` when they
# actually need temp space, so importing this module never blows up on a
# missing E: drive.
STAGING_DIR_DEFAULT = _PRIMARY_STAGING   # informational only; use the resolver

# --- API keys ---
# (2026-04-17): support the real-world api_key.txt format the user keeps.
# Lines may use "=" or ":" or tab as separator; labels may be Korean.
# Map canonical short names (seoul_subway, data_go_kr, kosis, ...) to whatever
# the collectors reference in KEYS[...].
_KEY_LABEL_MAP = {
    # Seoul \uc5f4\ub9b0\ub370\uc774\ud130\uad11\uc7a5 \uacc4\uc5f4
    "\uc77c\ubc18\uc778\uc99d\ud0a4(\uc778\uad6c?/\uc0c1\uc810?)": "seoul_general",   # 1st occurrence
    "\uc77c\ubc18\uc778\uc99d\ud0a4(\uc778\uad6c/\uc0c1\uc810)":   "seoul_general",
    "\uc9c0\ud558\ucca0\uc778\uc99d\ud0a4":                        "seoul_subway",
    "\uc9c0\ud558\ucca0\uc2e4\uc2dc\uac04 \uc778\uc99d\ud0a4":     "seoul_subway_rt",
    "\uc77c\ubc18\uc778\uc99d\ud0a4(\ub300\uae30)":                "seoul_air",
    # KOSIS / NEIS / \uacf5\uacf5\ub370\uc774\ud130\ud3ec\ud138 / \uae30\uc0c1\uccad
    "KOSIS \uacf5\uc720\uc11c\ube44\uc2a4 \uc0ac\uc6a9\uc790 \uc778\uc99d\ud0a4":     "kosis",
    "\ub098\uc774\uc2a4 \uad50\uc721\uc815\ubcf4 \uac1c\ubc29 \ud3ec\ud138(NEIS) \uc778\uc99d\ud0a4": "neis",
    "\uacf5\uacf5\ub370\uc774\ud130\ud3ec\ud138 \uc11c\ube44\uc2a4\ud0a4":            "data_go_kr",
    "\uae30\uc0c1\uccad_\uc0dd\ud65c\uae30\uc0c1\uc9c0\uc218 \uc870\ud68c\uc11c\ube44\uc2a4":    "data_go_kr",
    "\uae30\uc0c1\uccad api\ud5c8\ube0c":                           "kma_hub",
    "\uae30\uc0c1\uccad \uc778\uc99d\ud0a4":                        "kma_hub",
    "\ud55c\uad6d\ucc9c\ubb38\uc5f0\uad6c\uc6d0_\ud2b9\uc77c \uc815\ubcf4 \uc778\uc99d\ud0a4": "astro_holiday",
    "\uc9c8\ubcd1\uad00\ub9ac\uccad_\uc804\uc218\uc2e0\uace0 \uac10\uc5fc\ubcd1 \ubc1c\uc0dd\ud604\ud669": "kdca_weekly",
}


def _split_kv(line: str):
    """Split a single line into (label, value).

    Priority: ':' (Korean colon-labelled entries) → '\t' → '='.
    Critical: ':' must come before '=' because several base64 keys end with
    '==' padding, so partition('=') would slice *inside* the value and blank
    out the label. (kosis key hits this.)
    """
    for sep in (":", "\t", "="):
        if sep in line:
            k, _, v = line.partition(sep)
            return k.strip(), v.strip()
    return None, None


KEYS: dict[str, str] = {}
_key_file = DATA_DIR / "api_key.txt"
if _key_file.exists():
    try:
        with open(_key_file, "r", encoding="utf-8") as f:
            _seen_general = False
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                label, value = _split_kv(line)
                if not label or not value:
                    continue
                # Strip trailing '=' padding from base64 (KOSIS key ends in '=').
                canonical = _KEY_LABEL_MAP.get(label)
                if canonical == "seoul_general":
                    # file has two "\uc77c\ubc18\uc778\uc99d\ud0a4(\uc778\uad6c?/\uc0c1\uc810?)" entries — 2nd becomes seoul_general2
                    if _seen_general:
                        KEYS["seoul_general2"] = value
                        continue
                    _seen_general = True
                if canonical:
                    KEYS[canonical] = value
                else:
                    # Preserve unknown labels under normalized key (strip
                    # spaces). Skip numeric-only labels like '번호'/'1'..'6'
                    # which are just tab-separated sample IDs from the
                    # employment API section, not real credentials.
                    if label.isdigit() or label in {"번호"}:
                        continue
                    KEYS[label.replace(" ", "_")] = value
    except Exception:
        pass

# Environment overrides (CLI / CI / devops precedence).
for _env_key, _canonical in [
    ("SEOUL_KEY", "seoul_general"),
    ("SEOUL_SUBWAY", "seoul_subway"),
    ("SEOUL_AIR", "seoul_air"),
    ("DATA_GO_KR", "data_go_kr"),
    ("KMA_HUB", "kma_hub"),
    ("KOSIS", "kosis"),
    ("NEIS", "neis"),
]:
    _val = os.environ.get(_env_key)
    if _val:
        KEYS[_canonical] = _val

API_MAX_RETRY = 4
RETRY_WAIT = 3.0

# Minimum key length worth masking. Shorter values (or a leftover placeholder)
# would match unrelated substrings and mangle the log line instead of protecting it.
_REDACT_MIN_LEN = 8
_REDACT_MARK = "***REDACTED***"


def redact_secrets(text: object) -> str:
    """Mask every configured API key occurring anywhere inside ``text``.

    Korean government APIs authenticate by URL rather than by header: the Seoul
    Open API key sits in the request *path* and data.go.kr takes ``serviceKey``
    as a query parameter. Any log line carrying a request URL or a parameter
    dict therefore embeds a live credential unless it is masked first.

    Both the raw and the percent-encoded form of each key are masked, so this
    works whether the caller logs the URL it built itself or the fully encoded
    ``response.url`` that ``requests`` produced.

    Args:
        text: Anything renderable — usually a URL string or a params dict.
            Coerced with ``str()``.

    Returns:
        ``str(text)`` with every known key value replaced by ``***REDACTED***``.
        Values shorter than 8 characters are left untouched.

    Performance: O(len(KEYS) x len(text)); KEYS holds on the order of 20 entries.
    Side effects: none — pure function, reads the module-level KEYS dict.
    Caller responsibility: apply to any log or exception message that may embed
        a request URL, request parameters, or a response body echoing either.
    """
    s = str(text)
    for value in set(KEYS.values()):
        if not value or len(value) < _REDACT_MIN_LEN:
            continue
        s = s.replace(value, _REDACT_MARK)
        encoded = quote(value, safe="")
        if encoded != value:
            s = s.replace(encoded, _REDACT_MARK)
    return s

# Seoul district codes
SEOUL_GU_CODES = {
    "jongno-gu": "11010", "jung-gu": "11020", "yongsan-gu": "11030",
    "seongdong-gu": "11040", "gwangjin-gu": "11050", "dongdaemun-gu": "11060",
    "jungnang-gu": "11070", "seongbuk-gu": "11080", "gangbuk-gu": "11090",
    "dobong-gu": "11100", "nowon-gu": "11110", "eunpyeong-gu": "11120",
    "seodaemun-gu": "11130", "mapo-gu": "11140", "yangcheon-gu": "11150",
    "gangseo-gu": "11160", "guro-gu": "11170", "geumcheon-gu": "11180",
    "yeongdeungpo-gu": "11190", "dongjak-gu": "11200", "gwanak-gu": "11210",
    "seocho-gu": "11220", "gangnam-gu": "11230", "songpa-gu": "11240",
    "gangdong-gu": "11250",
}
# Korean name mapping (for backward compatibility)
SEOUL_GU_KR = {
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

# Stage 5: canonical 25-gu ordering used by the metapop simulator
# (populations + mobility rows/cols align to this index). Kept as a
# module-level list so ``simulation.sim.io`` and friends can import it
# without re-deriving. Preserve insertion order of ``SEOUL_GU_KR``.
SEOUL_GU_ORDERED: list[str] = list(SEOUL_GU_KR.keys())