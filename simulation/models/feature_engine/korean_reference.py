#!/usr/bin/env python3
"""
Korean ↔ English Reference for MPH Infection Simulation Pipeline
================================================================
Centralised mapping of all Korean string constants used in the pipeline.
Import this module instead of hardcoding Korean strings in loaders/builder.

Usage:
    from korean_reference import CONGESTION_MAP, POI_NAMES_KO_EN, get_congestion_score
"""

# ──────────────────────────────────────────────────────────────
# 1. Congestion Level Mapping  (혼잡도)
# ──────────────────────────────────────────────────────────────
# The Seoul citydata API returns congestion as Korean text.
# Two different naming conventions appear depending on endpoint:
#   - Population endpoint: 여유 / 보통 / 약간 붐빔 / 붐빔
#   - Road/general endpoint: 여유 / 원활 / 약간혼잡 / 혼잡 / 심각혼잡
#
# We normalise both variants to a 1-4 (or 1-5) numeric scale.

CONGESTION_MAP = {
    # Population endpoint congestion levels
    "여유":       1.0,   # comfortable / spacious
    "보통":       2.0,   # normal / moderate
    "약간 붐빔":  3.0,   # slightly crowded  (note: space between 약간 and 붐빔)
    "붐빔":       4.0,   # crowded
    # Road/general endpoint congestion levels
    "원활":       1.5,   # smooth flow (≈ between comfortable and normal)
    "약간혼잡":   3.0,   # slightly congested (no space)
    "혼잡":       4.0,   # congested
    "심각혼잡":   5.0,   # severely congested
}

# English labels for each level (for reporting / plots)
CONGESTION_LABELS = {
    1.0: "Comfortable",
    1.5: "Smooth",
    2.0: "Normal",
    3.0: "Slightly Crowded",
    4.0: "Crowded",
    5.0: "Severely Crowded",
}

def get_congestion_score(korean_text: str, default=None) -> float | None:
    """Safe lookup: strips whitespace, tries exact match, then fuzzy."""
    if korean_text is None:
        return default
    t = korean_text.strip()
    if t in CONGESTION_MAP:
        return CONGESTION_MAP[t]
    # Fuzzy: remove all spaces and try again
    t_nospace = t.replace(" ", "")
    for k, v in CONGESTION_MAP.items():
        if k.replace(" ", "") == t_nospace:
            return v
    return default


# ──────────────────────────────────────────────────────────────
# 2. Seoul 79 POI Names  (서울시 실시간 도시데이터 79개 POI)
# ──────────────────────────────────────────────────────────────
# Format: { "POI_CODE": {"ko": "한글명", "en": "English Name", "category": "..."} }

POI_NAMES_KO_EN = {
    "POI001": {"ko": "동대문 관광특구",          "en": "Dongdaemun Tourist Zone",           "category": "tourist_zone"},
    "POI002": {"ko": "명동 관광특구",            "en": "Myeongdong Tourist Zone",            "category": "tourist_zone"},
    "POI003": {"ko": "이태원 관광특구",          "en": "Itaewon Tourist Zone",               "category": "tourist_zone"},
    "POI004": {"ko": "잠실 관광특구",            "en": "Jamsil Tourist Zone",                "category": "tourist_zone"},
    "POI005": {"ko": "종로·청계 관광특구",       "en": "Jongno-Cheongye Tourist Zone",       "category": "tourist_zone"},
    "POI006": {"ko": "경복궁",                   "en": "Gyeongbokgung Palace",               "category": "heritage"},
    "POI007": {"ko": "광화문·덕수궁",            "en": "Gwanghwamun-Deoksugung",             "category": "heritage"},
    "POI008": {"ko": "보신각",                   "en": "Bosingak Pavilion",                  "category": "heritage"},
    "POI009": {"ko": "서울 암사동 유적",         "en": "Seoul Amsa-dong Prehistoric Site",    "category": "heritage"},
    "POI010": {"ko": "창덕궁·종묘",              "en": "Changdeokgung-Jongmyo",              "category": "heritage"},
    "POI011": {"ko": "가산디지털단지역",         "en": "Gasan Digital Complex Stn",           "category": "station"},
    "POI012": {"ko": "건대입구역",               "en": "Konkuk Univ. Stn",                   "category": "station"},
    "POI013": {"ko": "고덕역",                   "en": "Godeok Stn",                         "category": "station"},
    "POI014": {"ko": "고속터미널역",             "en": "Express Bus Terminal Stn",            "category": "station"},
    "POI015": {"ko": "교대역",                   "en": "Seoul Nat'l Univ. of Education Stn", "category": "station"},
    "POI016": {"ko": "동대문역",                 "en": "Dongdaemun Stn",                     "category": "station"},
    "POI017": {"ko": "동대문역사문화공원",       "en": "Dongdaemun History & Culture Park",   "category": "station"},
    "POI018": {"ko": "동묘앞역",                 "en": "Dongmyo Stn",                        "category": "station"},
    "POI019": {"ko": "동작역",                   "en": "Dongjak Stn",                        "category": "station"},
    "POI020": {"ko": "명동역",                   "en": "Myeongdong Stn",                     "category": "station"},
    "POI021": {"ko": "명동",                     "en": "Myeongdong Area",                    "category": "commercial"},
    "POI022": {"ko": "삼각지역",                 "en": "Samgakji Stn",                       "category": "station"},
    "POI023": {"ko": "삼성역",                   "en": "Samsung Stn (COEX)",                 "category": "station"},
    "POI024": {"ko": "삼청동",                   "en": "Samcheong-dong",                     "category": "commercial"},
    "POI025": {"ko": "서대문역",                 "en": "Seodaemun Stn",                      "category": "station"},
    "POI026": {"ko": "서울역",                   "en": "Seoul Station",                      "category": "station"},
    "POI027": {"ko": "성신여대입구역",           "en": "Sungshin Women's Univ. Stn",         "category": "station"},
    "POI028": {"ko": "세종대로",                 "en": "Sejong-daero",                       "category": "commercial"},
    "POI029": {"ko": "신도림역",                 "en": "Sindorim Stn",                       "category": "station"},
    "POI030": {"ko": "신림역",                   "en": "Sillim Stn",                         "category": "station"},
    "POI031": {"ko": "압구정역",                 "en": "Apgujeong Stn",                      "category": "station"},
    "POI032": {"ko": "양재역",                   "en": "Yangjae Stn",                        "category": "station"},
    "POI033": {"ko": "역삼역",                   "en": "Yeoksam Stn",                        "category": "station"},
    "POI034": {"ko": "왕십리역",                 "en": "Wangsimni Stn",                      "category": "station"},
    "POI035": {"ko": "용산역",                   "en": "Yongsan Stn",                        "category": "station"},
    "POI036": {"ko": "외대앞",                   "en": "Hankuk Univ. of Foreign Studies",    "category": "station"},
    "POI037": {"ko": "을지로",                   "en": "Euljiro",                            "category": "commercial"},
    "POI038": {"ko": "을지로3가역",              "en": "Euljiro 3-ga Stn",                   "category": "station"},
    "POI039": {"ko": "이태원역",                 "en": "Itaewon Stn",                        "category": "station"},
    "POI040": {"ko": "잠실역",                   "en": "Jamsil Stn",                         "category": "station"},
    "POI041": {"ko": "천호역",                   "en": "Cheonho Stn",                        "category": "station"},
    "POI042": {"ko": "충정로역",                 "en": "Chungjeongno Stn",                   "category": "station"},
    "POI043": {"ko": "합정역",                   "en": "Hapjeong Stn",                       "category": "station"},
    "POI044": {"ko": "혼화역",                   "en": "Honhwa Stn",                         "category": "station"},
    "POI045": {"ko": "홍대입구역",               "en": "Hongik Univ. Stn",                   "category": "station"},
    "POI046": {"ko": "회기역",                   "en": "Hoegi Stn",                          "category": "station"},
    "POI047": {"ko": "가로수길",                 "en": "Garosu-gil",                         "category": "commercial"},
    "POI048": {"ko": "김포공항",                 "en": "Gimpo Airport",                      "category": "transport"},
    "POI049": {"ko": "북촌한옥마을",             "en": "Bukchon Hanok Village",              "category": "heritage"},
    "POI050": {"ko": "서촌",                     "en": "Seochon",                            "category": "commercial"},
    "POI051": {"ko": "한강공원",                 "en": "Hangang Park",                       "category": "park"},
    "POI052": {"ko": "영등포 타임스퀘어",        "en": "Yeongdeungpo Times Square",          "category": "commercial"},
    "POI053": {"ko": "창동 신경제 중심지",       "en": "Changdong New Economy Hub",          "category": "commercial"},
    "POI054": {"ko": "청담동 명품거리",          "en": "Cheongdam Luxury Street",            "category": "commercial"},
    "POI055": {"ko": "DMC(디지털미디어시티)",     "en": "DMC (Digital Media City)",           "category": "commercial"},
    "POI056": {"ko": "DDP(동대문디자인플라자)",   "en": "DDP (Dongdaemun Design Plaza)",      "category": "heritage"},
    "POI057": {"ko": "남산공원",                 "en": "Namsan Park",                        "category": "park"},
    "POI058": {"ko": "노들섬",                   "en": "Nodeul Island",                      "category": "park"},
    "POI059": {"ko": "한강공원 여의도",          "en": "Hangang Park Yeouido",               "category": "park"},
    "POI060": {"ko": "북서울꿈의숲",             "en": "Bukseoul Dream Forest",              "category": "park"},
    "POI061": {"ko": "서울대공원",               "en": "Seoul Grand Park",                   "category": "park"},
    "POI062": {"ko": "서울숲공원",               "en": "Seoul Forest",                       "category": "park"},
    "POI063": {"ko": "아차산",                   "en": "Achasan Mountain",                   "category": "park"},
    "POI064": {"ko": "어린이대공원",             "en": "Children's Grand Park",              "category": "park"},
    "POI065": {"ko": "월드컵공원",               "en": "World Cup Park",                     "category": "park"},
    "POI066": {"ko": "잠실종합운동장",           "en": "Jamsil Sports Complex",              "category": "sports"},
    "POI067": {"ko": "청계산",                   "en": "Cheonggyesan Mountain",              "category": "park"},
    "POI068": {"ko": "노량진",                   "en": "Noryangjin",                         "category": "commercial"},
    "POI069": {"ko": "여의도",                   "en": "Yeouido",                            "category": "commercial"},
    "POI070": {"ko": "이태원 앤틱가구거리",      "en": "Itaewon Antique Furniture St",       "category": "commercial"},
    "POI071": {"ko": "고척돔",                   "en": "Gocheok Sky Dome",                  "category": "sports"},
    "POI072": {"ko": "서울식물원",               "en": "Seoul Botanic Park",                 "category": "park"},
    "POI073": {"ko": "성수카페거리",             "en": "Seongsu Cafe Street",                "category": "commercial"},
    "POI074": {"ko": "수유리 먹자골목",          "en": "Suyu-ri Food Alley",                 "category": "commercial"},
    "POI075": {"ko": "쌍림동 가구거리",          "en": "Ssangnim-dong Furniture St",         "category": "commercial"},
    "POI076": {"ko": "압구정로데오거리",         "en": "Apgujeong Rodeo Street",             "category": "commercial"},
    "POI077": {"ko": "연남동",                   "en": "Yeonnam-dong",                       "category": "commercial"},
    "POI078": {"ko": "용리단길",                 "en": "Yongridan-gil",                      "category": "commercial"},
    "POI079": {"ko": "해방촌·경리단길",          "en": "Haebangchon-Gyeongridan-gil",        "category": "commercial"},
}


# ──────────────────────────────────────────────────────────────
# 3. Reverse Lookups
# ──────────────────────────────────────────────────────────────

# Korean name → POI code
KO_TO_CODE = {v["ko"]: k for k, v in POI_NAMES_KO_EN.items()}

# Korean name → English name
KO_TO_EN = {v["ko"]: v["en"] for v in POI_NAMES_KO_EN.values()}

# English name → Korean name
EN_TO_KO = {v["en"]: v["ko"] for v in POI_NAMES_KO_EN.values()}


# ──────────────────────────────────────────────────────────────
# 4. POI Category Metadata
# ──────────────────────────────────────────────────────────────

POI_CATEGORIES = {
    "tourist_zone": {"ko": "관광특구",     "en": "Tourist Zone"},
    "heritage":     {"ko": "문화유산",     "en": "Heritage Site"},
    "station":      {"ko": "역/교통",      "en": "Station / Transit"},
    "commercial":   {"ko": "상업지구",     "en": "Commercial District"},
    "park":         {"ko": "공원/자연",    "en": "Park / Nature"},
    "transport":    {"ko": "교통시설",     "en": "Transport Facility"},
    "sports":       {"ko": "스포츠시설",   "en": "Sports Facility"},
}


# ──────────────────────────────────────────────────────────────
# 5. Other Korean Constants Used in Pipeline
# ──────────────────────────────────────────────────────────────

# Road traffic message levels (from citydata API)
ROAD_TRAFFIC_MSG = {
    "원활":     "Smooth",
    "서행":     "Slow",
    "정체":     "Congested",
}

# Day-of-week names used in temporal features
WEEKDAY_NAMES_KO = ["월", "화", "수", "목", "금", "토", "일"]
WEEKDAY_NAMES_EN = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Subway congestion mapping (if used)
SUBWAY_CONGESTION = {
    "여유":       1.0,
    "보통":       2.0,
    "주의":       3.0,   # caution
    "혼잡":       4.0,   # congested
}


# ──────────────────────────────────────────────────────────────
# 6. Utility: Safe Encoding Helpers
# ──────────────────────────────────────────────────────────────

def safe_area_name(name: str) -> str:
    """Ensure area name is properly encoded UTF-8 string."""
    if isinstance(name, bytes):
        # SSOT (2026-05-28): 수동 인코딩 ladder → decode_bytes_safe.
        # 기존 순서/never-raise 계약 보존 (display string → replace fallback).
        from simulation.utils.safe_io import decode_bytes_safe
        return decode_bytes_safe(
            name, encodings=("utf-8", "euc-kr", "cp949"), fallback_errors="replace"
        )
    return str(name)


def poi_code_from_name(korean_name: str) -> str | None:
    """Look up POI code from Korean area name (fuzzy match)."""
    name = safe_area_name(korean_name).strip()
    if name in KO_TO_CODE:
        return KO_TO_CODE[name]
    # Try removing middle dots (·) and spaces for fuzzy match
    clean = name.replace("·", "").replace(" ", "")
    for ko, code in KO_TO_CODE.items():
        if ko.replace("·", "").replace(" ", "") == clean:
            return code
    return None


if __name__ == "__main__":
    print(f"Total POIs: {len(POI_NAMES_KO_EN)}")
    print(f"Congestion levels: {len(CONGESTION_MAP)}")
    print(f"\nCongestion Map:")
    for ko, score in CONGESTION_MAP.items():
        en = CONGESTION_LABELS.get(score, "?")
        print(f"  {ko:10s} → {score:.1f} ({en})")
    print(f"\nPOI Categories:")
    for cat, info in POI_CATEGORIES.items():
        count = sum(1 for v in POI_NAMES_KO_EN.values() if v["category"] == cat)
        print(f"  {cat:15s}: {count:2d} POIs ({info['ko']} / {info['en']})")
    print(f"\nSample POIs:")
    for code in ["POI001", "POI026", "POI045", "POI057", "POI069"]:
        p = POI_NAMES_KO_EN[code]
        print(f"  {code}: {p['ko']:20s} → {p['en']}")
