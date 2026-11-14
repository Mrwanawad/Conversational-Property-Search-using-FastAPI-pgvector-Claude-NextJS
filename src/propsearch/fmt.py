"""Bilingual number / label formatting shared by responses and relaxation explanations."""

from __future__ import annotations

from propsearch.data.catalog import AREAS, LANDMARKS

TYPE_LABELS = {
    "apartment": ("apartment", "شقة"),
    "duplex": ("duplex", "دوبلكس"),
    "penthouse": ("penthouse", "بنتهاوس"),
    "studio": ("studio", "ستوديو"),
    "townhouse": ("townhouse", "تاون هاوس"),
    "twin_house": ("twin house", "توين هاوس"),
    "villa": ("villa", "فيلا"),
    "chalet": ("chalet", "شاليه"),
}
AMENITY_LABELS = {
    "garden": ("garden", "جنينة"),
    "roof": ("roof", "روف"),
    "private_pool": ("private pool", "حمام سباحة خاص"),
    "sea_view": ("sea view", "فيو بحر"),
    "maids_room": ("maid's room", "غرفة دادة"),
    "swimming_pool": ("pool", "حمام سباحة"),
    "clubhouse": ("clubhouse", "كلوب هاوس"),
    "gym": ("gym", "جيم"),
    "kids_area": ("kids area", "منطقة أطفال"),
    "crystal_lagoon": ("lagoon", "لاجون"),
    "beach_access": ("beach access", "على البحر"),
    "golf": ("golf", "جولف"),
    "security": ("security", "أمن"),
    "parking": ("parking", "جراج"),
}
FINISH_LABELS = {
    "fully_finished": ("fully finished", "متشطب بالكامل"),
    "semi_finished": ("semi-finished", "نص تشطيب"),
    "core_shell": ("core & shell", "طوب أحمر"),
}


def egp(v: float, lang: str = "en") -> str:
    if v >= 1_000_000:
        m = v / 1_000_000
        s = f"{m:.2f}".rstrip("0").rstrip(".")
        return f"{s}M EGP" if lang == "en" else f"{s} مليون جنيه"
    k = round(v / 1_000)
    return f"{k:,}K EGP" if lang == "en" else f"{k:,} ألف جنيه"


def area_name(slug: str, lang: str = "en") -> str:
    a = AREAS[slug]
    return a.name_en if lang == "en" else a.name_ar


def landmark_name(slug: str, lang: str = "en") -> str:
    lm = LANDMARKS[slug]
    return lm.name_en if lang == "en" else lm.name_ar


def type_name(t: str, lang: str = "en") -> str:
    return TYPE_LABELS[t][0 if lang == "en" else 1]


def amenity_name(a: str, lang: str = "en") -> str:
    return AMENITY_LABELS[a][0 if lang == "en" else 1]


def finish_name(f: str, lang: str = "en") -> str:
    return FINISH_LABELS[f][0 if lang == "en" else 1]
