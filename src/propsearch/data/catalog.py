"""Static geography: areas, area groups and landmarks.

Area names and landmark coordinates are real Egyptian places. Compounds, developers and units
are synthetic (see scripts/generate_inventory.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Area:
    slug: str
    name_en: str
    name_ar: str
    lat: float
    lng: float
    price_per_m2: tuple[int, int]  # synthetic 2026 EGP range for primary-market units
    neighbors: tuple[str, ...] = field(default_factory=tuple)
    coastal: bool = False


AREAS: dict[str, Area] = {
    a.slug: a
    for a in [
        Area(
            "new_cairo",
            "New Cairo",
            "القاهرة الجديدة",
            30.0300,
            31.4700,
            (50_000, 85_000),
            ("mostakbal_city", "shorouk", "new_capital"),
        ),
        Area(
            "mostakbal_city",
            "Mostakbal City",
            "مدينة المستقبل",
            30.1150,
            31.6050,
            (45_000, 66_000),
            ("new_cairo", "shorouk", "new_capital"),
        ),
        Area(
            "new_capital",
            "New Capital",
            "العاصمة الإدارية",
            30.0200,
            31.7300,
            (42_000, 62_000),
            ("mostakbal_city", "new_cairo"),
        ),
        Area(
            "shorouk",
            "El Shorouk",
            "الشروق",
            30.1400,
            31.6200,
            (35_000, 52_000),
            ("mostakbal_city", "new_cairo"),
        ),
        Area(
            "sheikh_zayed",
            "Sheikh Zayed",
            "الشيخ زايد",
            30.0450,
            30.9800,
            (60_000, 92_000),
            ("new_zayed", "october"),
        ),
        Area(
            "new_zayed",
            "New Zayed",
            "زايد الجديدة",
            30.1150,
            30.9200,
            (45_000, 66_000),
            ("sheikh_zayed", "october"),
        ),
        Area(
            "october",
            "6th of October",
            "السادس من أكتوبر",
            29.9600,
            30.9300,
            (35_000, 55_000),
            ("sheikh_zayed", "new_zayed"),
        ),
        Area("north_coast", "North Coast", "الساحل الشمالي", 30.9500, 28.7500, (80_000, 150_000), (), True),
        Area("ain_sokhna", "Ain Sokhna", "العين السخنة", 29.6000, 32.3200, (50_000, 85_000), (), True),
    ]
}

AREA_GROUPS: dict[str, tuple[str, ...]] = {
    "east_cairo": ("new_cairo", "mostakbal_city", "shorouk", "new_capital"),
    "west_cairo": ("sheikh_zayed", "new_zayed", "october"),
}


@dataclass(frozen=True)
class Landmark:
    slug: str
    name_en: str
    name_ar: str
    lat: float
    lng: float
    area: str | None = None  # area implied when the user names only the landmark


LANDMARKS: dict[str, Landmark] = {
    lm.slug: lm
    for lm in [
        Landmark(
            "auc", "AUC (American University in Cairo)", "الجامعة الأمريكية", 30.0196, 31.4999, "new_cairo"
        ),
        Landmark("cairo_airport", "Cairo Airport", "مطار القاهرة", 30.1219, 31.4056),
        Landmark("smart_village", "Smart Village", "القرية الذكية", 30.0717, 31.0170),
        Landmark("mall_of_arabia", "Mall of Arabia", "مول العرب", 30.0072, 30.9733, "october"),
        Landmark("mall_of_egypt", "Mall of Egypt", "مول مصر", 29.9727, 31.0170, "october"),
        Landmark("gem", "Grand Egyptian Museum", "المتحف المصري الكبير", 29.9946, 31.1186),
        Landmark(
            "capital_cbd", "New Capital CBD", "حي المال والأعمال بالعاصمة", 30.0080, 31.6980, "new_capital"
        ),
        Landmark("new_alamein", "New Alamein", "العلمين الجديدة", 30.8250, 28.9550, "north_coast"),
    ]
}


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))
