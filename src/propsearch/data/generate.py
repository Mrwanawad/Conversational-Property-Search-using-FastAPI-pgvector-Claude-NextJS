"""Deterministic synthetic inventory: 14 developers, 45 compounds, ~1,400 units with bilingual descriptions.

Areas and landmarks are real places; every developer, compound, unit, price and plan is invented.
Each unit also gets hidden "soft tags" (quiet, near schools, natural light...) that are written into
its descriptions as paraphrases. The tags are never stored in the database: evals use them as
relevance labels, so the retriever has to find "quiet" in "tucked away from the main road".
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta

from propsearch.data.catalog import AREAS, LANDMARKS

SEED = 20261101
NOW = datetime(2026, 11, 1, 12, 0, 0)

DEVELOPERS = [
    ("orchid", "Orchid Developments", "أوركيد للتطوير العقاري", 0.91),
    ("basalt", "Basalt Real Estate", "بازلت العقارية", 0.84),
    ("nile_crest", "Nile Crest Developments", "نايل كريست للتطوير", 0.88),
    ("sandline", "Sandline Properties", "ساندلاين العقارية", 0.79),
    ("cedar_group", "Cedar Group", "مجموعة سيدار", 0.93),
    ("tamarind", "Tamarind Development", "تامارند للتطوير", 0.81),
    ("azure_shore", "Azure Shore Resorts", "أزور شور للمنتجعات", 0.86),
    ("granite_key", "Granite Key", "جرانيت كي العقارية", 0.9),
    ("lumen", "Lumen Developments", "لومن للتطوير", 0.76),
    ("saffron", "Saffron Estates", "سافرون إستيتس", 0.87),
    ("halfa", "Halfa Urban", "حلفا للتعمير", 0.82),
    ("qamar", "Qamar Developments", "قمر للتطوير العقاري", 0.74),
    ("wadi_stone", "Wadi Stone", "وادي ستون", 0.89),
    ("sidra", "Sidra Holding", "سدرة القابضة", 0.85),
]

# slug, name_en, name_ar, area, developer, profile
COMPOUNDS = [
    ("cedar_walk", "Cedar Walk", "سيدار ووك", "new_cairo", "cedar_group", "mixed"),
    ("aster_gardens", "Aster Gardens", "أستر جاردنز", "new_cairo", "orchid", "mixed"),
    ("juniper_heights", "Juniper Heights", "جونيبر هايتس", "new_cairo", "granite_key", "apartments"),
    ("saffron_courts", "Saffron Courts", "سافرون كورتس", "new_cairo", "saffron", "mixed"),
    ("myrtle_park", "Myrtle Park", "ميرتل بارك", "new_cairo", "nile_crest", "houses"),
    ("indigo_lane", "Indigo Lane", "إنديجو لين", "new_cairo", "basalt", "apartments"),
    ("quartz_residence", "Quartz Residence", "كوارتز ريزيدنس", "new_cairo", "wadi_stone", "mixed"),
    ("sycamore_grove", "Sycamore Grove", "سيكامور جروف", "mostakbal_city", "cedar_group", "mixed"),
    ("halfa_village", "Halfa Village", "قرية حلفا", "mostakbal_city", "halfa", "mixed"),
    ("wren_court", "Wren Court", "رين كورت", "mostakbal_city", "lumen", "apartments"),
    ("amber_terraces", "Amber Terraces", "أمبر تيراسيز", "mostakbal_city", "sidra", "houses"),
    ("lumen_park", "Lumen Park", "لومن بارك", "new_capital", "lumen", "apartments"),
    ("tamarind_residence", "Tamarind Residence", "تامارند ريزيدنس", "new_capital", "tamarind", "apartments"),
    ("obsidian_towers", "Obsidian Towers", "أوبسيديان تاورز", "new_capital", "qamar", "apartments"),
    ("sidra_gardens", "Sidra Gardens", "حدائق سدرة", "new_capital", "sidra", "mixed"),
    ("capital_loom", "Capital Loom", "كابيتال لوم", "new_capital", "basalt", "mixed"),
    ("basalt_quarter", "Basalt Quarter", "حي بازلت", "new_capital", "basalt", "apartments"),
    ("olive_terraces", "Olive Terraces", "أوليف تيراسيز", "shorouk", "halfa", "mixed"),
    ("palmyra_walk", "Palmyra Walk", "بالميرا ووك", "shorouk", "qamar", "apartments"),
    ("nakhla_residence", "Nakhla Residence", "نخلة ريزيدنس", "shorouk", "orchid", "mixed"),
    ("granite_gate", "Granite Gate", "جرانيت جيت", "sheikh_zayed", "granite_key", "mixed"),
    ("linden_square", "Linden Square", "ليندن سكوير", "sheikh_zayed", "saffron", "apartments"),
    ("acacia_row", "Acacia Row", "أكاشيا رو", "sheikh_zayed", "nile_crest", "houses"),
    ("sandstone_villas", "Sandstone Villas", "ساندستون فيلاز", "sheikh_zayed", "wadi_stone", "villas"),
    ("marigold_park", "Marigold Park", "ماريجولد بارك", "sheikh_zayed", "cedar_group", "mixed"),
    ("willow_crescent", "Willow Crescent", "ويلو كريسنت", "sheikh_zayed", "orchid", "mixed"),
    ("hazel_fields", "Hazel Fields", "هازل فيلدز", "new_zayed", "sandline", "mixed"),
    ("rowan_gardens", "Rowan Gardens", "روان جاردنز", "new_zayed", "tamarind", "houses"),
    ("cypress_point", "Cypress Point", "سايبرس بوينت", "new_zayed", "granite_key", "mixed"),
    ("ivory_walk", "Ivory Walk", "آيفوري ووك", "new_zayed", "lumen", "apartments"),
    ("fig_tree", "Fig Tree Residence", "فيج تري ريزيدنس", "october", "halfa", "apartments"),
    ("clover_park", "Clover Park", "كلوفر بارك", "october", "sandline", "mixed"),
    ("topaz_compound", "Topaz Compound", "كمبوند توباز", "october", "qamar", "apartments"),
    ("laurel_gardens", "Laurel Gardens", "لوريل جاردنز", "october", "nile_crest", "mixed"),
    ("jasmine_court", "Jasmine Court", "ياسمين كورت", "october", "sidra", "mixed"),
    ("azure_bay", "Azure Shore Bay", "أزور شور باي", "north_coast", "azure_shore", "coastal"),
    ("coral_sands", "Coral Sands", "كورال ساندز", "north_coast", "sandline", "coastal"),
    ("salt_lagoon", "Salt Lagoon", "سولت لاجون", "north_coast", "azure_shore", "coastal"),
    ("driftwood_bay", "Driftwood Bay", "دريفتوود باي", "north_coast", "wadi_stone", "coastal"),
    ("sea_lavender", "Sea Lavender", "سي لافندر", "north_coast", "saffron", "coastal"),
    ("alamein_pearl", "Alamein Pearl", "لؤلؤة العلمين", "north_coast", "nile_crest", "coastal"),
    ("red_reef", "Red Reef", "ريد ريف", "ain_sokhna", "azure_shore", "coastal"),
    ("sokhna_crest", "Sokhna Crest", "سخنة كريست", "ain_sokhna", "tamarind", "coastal"),
    ("galala_steps", "Galala Steps", "جلالة ستيبس", "ain_sokhna", "qamar", "coastal"),
    ("sapphire_cove", "Sapphire Cove", "سافاير كوف", "ain_sokhna", "orchid", "coastal"),
]

TYPE_MIX = {
    "apartments": {"apartment": 70, "duplex": 8, "penthouse": 8, "studio": 14},
    "mixed": {
        "apartment": 50,
        "duplex": 8,
        "penthouse": 5,
        "studio": 5,
        "townhouse": 12,
        "twin_house": 10,
        "villa": 10,
    },
    "houses": {"townhouse": 40, "twin_house": 30, "villa": 30},
    "villas": {"villa": 80, "twin_house": 20},
    "coastal": {"chalet": 68, "twin_house": 12, "townhouse": 10, "villa": 10},
}
TYPE_LABEL = {
    "apartment": ("Apartment", "شقة"),
    "duplex": ("Duplex", "دوبلكس"),
    "penthouse": ("Penthouse", "بنتهاوس"),
    "studio": ("Studio", "ستوديو"),
    "townhouse": ("Townhouse", "تاون هاوس"),
    "twin_house": ("Twin House", "توين هاوس"),
    "villa": ("Villa", "فيلا"),
    "chalet": ("Chalet", "شاليه"),
}
TYPE_PRICE_MULT = {
    "apartment": 1.0,
    "duplex": 0.97,
    "penthouse": 1.07,
    "studio": 1.06,
    "townhouse": 1.0,
    "twin_house": 1.05,
    "villa": 1.12,
    "chalet": 1.0,
}
FINISH_MULT = {"fully_finished": 1.0, "semi_finished": 0.93, "core_shell": 0.86}

# soft tag -> (EN paraphrases, AR paraphrases). Deliberately varied wording so lexical match is not enough.
SOFT_TEXT: dict[str, tuple[list[str], list[str]]] = {
    "quiet": (
        [
            "Tucked away on a low-traffic inner street, far from the main road noise.",
            "A peaceful pocket of the compound with very little through traffic.",
            "Calm surroundings, away from the commercial strip.",
        ],
        [
            "في شارع داخلي هادي بعيد عن دوشة الطريق الرئيسي.",
            "مكان هادئ جدا داخل الكمبوند وبعيد عن الزحمة.",
            "منطقة ساكنة بعيدة عن المحلات والحركة.",
        ],
    ),
    "family": (
        [
            "Designed for families, with play areas in sight of the living room.",
            "A family-oriented cluster where neighbours are mostly young families.",
            "Kids can walk safely to the play zones without crossing car roads.",
        ],
        [
            "مناسبة جدا للعائلات وفي العاب اطفال قصاد الريسبشن.",
            "كلستر عائلي وأغلب السكان عائلات صغيرة.",
            "الأطفال يقدروا يتمشوا لمناطق الألعاب من غير ما يعدوا شوارع عربيات.",
        ],
    ),
    "schools": (
        [
            "A few minutes' drive to several international schools.",
            "Close to the area's school belt, ideal for the daily school run.",
            "Two international schools within 5 minutes.",
        ],
        [
            "دقايق بالعربية من مدارس انترناشونال كتير.",
            "قريبة من حزام المدارس في المنطقة ومريحة لمشوار المدرسة اليومي.",
            "مدرستين دولي على بعد 5 دقايق.",
        ],
    ),
    "hospital": (
        [
            "Near a major private hospital and several clinics.",
            "Medical centre and a 24-hour hospital within a short drive.",
        ],
        ["قريبة من مستشفى خاص كبير وعيادات.", "مركز طبي ومستشفى 24 ساعة على مسافة قريبة."],
    ),
    "light": (
        [
            "Double-aspect layout that gets sun most of the day.",
            "Large windows and a south-facing living room flooded with daylight.",
            "Bright interiors thanks to floor-to-ceiling glazing.",
        ],
        [
            "الوحدة بحريه وقبلي والشمس داخلة معظم اليوم.",
            "شبابيك كبيرة وريسبشن قبلي منور جدا بضوء النهار.",
            "إضاءة طبيعية ممتازة بسبب الواجهات الزجاج.",
        ],
    ),
    "view": (
        [
            "Unobstructed open view over the central landscape.",
            "Overlooks the main green with no buildings in front.",
            "Wide open panorama from the main terrace.",
        ],
        [
            "فيو مفتوح على اللاندسكيب الرئيسي ومفيش مباني قدامها.",
            "اطلالة مفتوحة على المساحات الخضراء.",
            "بانوراما واسعة من التراس الرئيسي.",
        ],
    ),
    "privacy": (
        [
            "Not overlooked by neighbours, with a private entrance.",
            "Secluded plot at the end of a cul-de-sac.",
        ],
        ["مفيش جيران كاشفين عليها ولها مدخل خاص.", "قطعة في آخر شارع مقفول وخصوصية عالية."],
    ),
    "open_kitchen": (
        [
            "Open-plan kitchen flowing into the living area.",
            "American-style kitchen with an island facing the reception.",
        ],
        ["مطبخ أمريكاني مفتوح على الريسبشن.", "مطبخ مفتوح بجزيرة في نص المساحة."],
    ),
    "smart_home": (
        [
            "Pre-wired for home automation: smart locks, lighting and AC control.",
            "Comes with a smart-home package controlled from your phone.",
        ],
        ["متجهزة سمارت هوم: إضاءة وتكييفات وأقفال من الموبايل.", "باكدج منزل ذكي تتحكم فيه من الموبايل."],
    ),
    "pets": (
        [
            "Pet-friendly building with a dog run nearby.",
            "Pets are welcome; there is a fenced pet park in the cluster.",
        ],
        [
            "المبنى مسموح فيه بالحيوانات الأليفة وفي مكان مخصص للكلاب.",
            "الحيوانات الأليفة مرحب بيها وفي بارك مقفول ليهم.",
        ],
    ),
    "corner": (
        ["Corner unit with windows on three sides.", "Sits on the corner of the block, so extra frontage."],
        ["وحدة ناصية بشبابيك على تلات اتجاهات.", "على ناصية البلوك وليها واجهة إضافية."],
    ),
    "high_floor": (
        ["On an upper floor with views over the rooftops.", "High floor, above the tree line."],
        ["في دور عالي وفوق مستوى الشجر.", "دور مرتفع وبيطل على أسطح الكمبوند."],
    ),
    "ground_floor": (
        ["Ground-floor unit with direct step-out access.", "Ground level, no stairs or lifts needed."],
        ["دور أرضي وليها خروج مباشر.", "في الدور الأرضي ومش محتاج سلم أو أسانسير."],
    ),
    "investment": (
        [
            "Strong rental demand in this cluster; popular with tenants.",
            "Good resale and rental potential given the location.",
            "Holiday-let friendly with an on-site rental management office.",
        ],
        [
            "طلب عالي على الإيجار في الكلستر ده.",
            "فرصة كويسة لإعادة البيع والإيجار بسبب الموقع.",
            "مناسبة للإيجار الموسمي وفي مكتب إدارة إيجارات في الكمبوند.",
        ],
    ),
    "walkable": (
        [
            "Walk to the cafés, pharmacy and supermarket in the commercial hub.",
            "Everything daily is a short walk away.",
        ],
        ["الكافيهات والصيدلية والسوبر ماركت على المشي.", "كل الخدمات اليومية مسافة مشي قصيرة."],
    ),
    "metro": (
        ["Near the planned monorail station.", "A short drive to the LRT station."],
        ["قريبة من محطة المونوريل.", "مسافة قصيرة من محطة القطار الكهربائي."],
    ),
    "green": (
        [
            "Surrounded by landscaped gardens and tree-lined walkways.",
            "Over 80% of the compound is green space.",
        ],
        ["حواليها حدائق ولاندسكيب وممشى بالشجر.", "أكتر من 80% من الكمبوند مساحات خضرا."],
    ),
    "modern": (
        ["Contemporary architecture with clean lines.", "Modern elevation and minimalist design."],
        ["تصميم معماري مودرن وبسيط.", "واجهة حديثة وتصميم مينيمال."],
    ),
    "storage": (
        ["Comes with a private storage room.", "Includes a storeroom in the basement."],
        ["معاها غرفة تخزين خاصة.", "معاها مخزن في البدروم."],
    ),
    "sunset": (
        ["West-facing terrace for sunset views over the water.", "Watch the sunset from the terrace."],
        ["تراس غربي على منظر الغروب على المية.", "تقدر تتفرج على الغروب من التراس."],
    ),
}
AREA_CODE = {
    "new_cairo": "NC",
    "mostakbal_city": "MC",
    "new_capital": "CA",
    "shorouk": "SH",
    "sheikh_zayed": "SZ",
    "new_zayed": "NZ",
    "october": "OC",
    "north_coast": "NO",
    "ain_sokhna": "SO",
}
INLAND_ONLY = {"schools", "hospital", "metro", "walkable"}
COASTAL_ONLY = {"sunset"}


@dataclass
class PlanRow:
    name: str
    down_payment_pct: float
    years: int
    cash_discount_pct: float = 0.0


@dataclass
class CompoundRow:
    slug: str
    name_en: str
    name_ar: str
    area: str
    developer: str
    lat: float
    lng: float
    amenities: list[str]
    plans: list[PlanRow]
    delivered: bool


@dataclass
class UnitRow:
    code: str
    compound: str
    area: str
    unit_type: str
    bedrooms: int
    bathrooms: int
    bua_m2: int
    garden_m2: int
    floor: int | None
    finishing: str
    price: int
    delivery_year: int
    is_ready: bool
    status: str
    amenities: list[str]
    description_en: str
    description_ar: str
    updated_at: str
    soft_tags: list[str] = field(default_factory=list)  # eval-only labels, never written to the DB


def _pick(rng: random.Random, weights: dict[str, int]) -> str:
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def _plans(rng: random.Random, delivered: bool, area: str) -> list[PlanRow]:
    if delivered:
        pool = [PlanRow("Ready 3Y", 25, 3), PlanRow("Ready 4Y", 35, 4), PlanRow("Ready 2Y", 50, 2)]
        plans = rng.sample(pool, 2)
    else:
        pool = [
            PlanRow("5% / 10Y", 5, 10),
            PlanRow("10% / 8Y", 10, 8),
            PlanRow("10% / 7Y", 10, 7),
            PlanRow("15% / 6Y", 15, 6),
            PlanRow("20% / 5Y", 20, 5),
            PlanRow("0% / 6Y", 0, 6),
        ]
        if area == "new_capital":
            pool.append(PlanRow("5% / 12Y", 5, 12))
        plans = rng.sample(pool, rng.choice([2, 3]))
    plans.append(PlanRow("Cash", 100, 0, float(rng.choice([10, 15, 20, 25]))))
    return plans


def _size(rng: random.Random, t: str) -> tuple[int, int]:
    if t == "studio":
        return 0, rng.randint(45, 75)
    if t == "apartment":
        b = _pick(rng, {"1": 10, "2": 30, "3": 45, "4": 15})
        bua = {"1": (70, 95), "2": (100, 140), "3": (130, 185), "4": (190, 250)}[b]
        return int(b), rng.randint(*bua)
    if t == "duplex":
        b = rng.choice([3, 4, 4, 5])
        return b, rng.randint(200, 320)
    if t == "penthouse":
        b = rng.choice([3, 3, 4])
        return b, rng.randint(180, 300)
    if t == "townhouse":
        b = rng.choice([3, 4, 4])
        return b, rng.randint(180, 260)
    if t == "twin_house":
        b = rng.choice([4, 4, 5])
        return b, rng.randint(230, 330)
    if t == "villa":
        b = rng.choice([4, 5, 5, 6, 7])
        return b, rng.randint(300, 600)
    b = _pick(rng, {"1": 25, "2": 40, "3": 30, "4": 5})  # chalet
    return int(b), {
        "1": rng.randint(65, 90),
        "2": rng.randint(90, 130),
        "3": rng.randint(125, 175),
        "4": rng.randint(160, 210),
    }[b]


def _fmt_money_en(v: float) -> str:
    return f"{v / 1e6:.2f}M EGP" if v >= 1e6 else f"{v / 1e3:.0f}K EGP"


def generate(seed: int = SEED) -> tuple[list[dict], list[CompoundRow], list[UnitRow]]:
    rng = random.Random(seed)
    developers = [{"slug": s, "name_en": en, "name_ar": ar, "on_time_rate": r} for s, en, ar, r in DEVELOPERS]
    compounds: list[CompoundRow] = []
    units: list[UnitRow] = []
    for slug, name_en, name_ar, area_slug, dev, profile in COMPOUNDS:
        area = AREAS[area_slug]
        if slug == "alamein_pearl":
            base = LANDMARKS["new_alamein"]
            lat, lng = base.lat + rng.uniform(-0.02, 0.02), base.lng + rng.uniform(-0.03, 0.03)
        elif area.coastal:
            lat, lng = area.lat + rng.uniform(-0.05, 0.05), area.lng + rng.uniform(-0.25, 0.25)
        else:
            lat, lng = area.lat + rng.uniform(-0.035, 0.035), area.lng + rng.uniform(-0.04, 0.04)
        delivered = rng.random() < 0.3
        amen = {"security", "parking"}
        for a, p in [
            ("swimming_pool", 0.8),
            ("clubhouse", 0.7),
            ("gym", 0.75),
            ("kids_area", 0.8),
            ("crystal_lagoon", 0.6 if area.coastal else 0.15),
            ("golf", 0.08),
        ]:
            if rng.random() < p:
                amen.add(a)
        if area.coastal and rng.random() < 0.7:
            amen.add("beach_access")
        plans = _plans(rng, delivered, area_slug)
        ppm2 = rng.uniform(*area.price_per_m2)
        compounds.append(
            CompoundRow(
                slug,
                name_en,
                name_ar,
                area_slug,
                dev,
                round(lat, 5),
                round(lng, 5),
                sorted(amen),
                plans,
                delivered,
            )
        )
        n_units = rng.randint(22, 42)
        words = name_en.upper().split()
        prefix = (words[0][:2] + (words[1][0] if len(words) > 1 else words[0][2])) if words else "XXX"
        for k in range(n_units):
            t = _pick(rng, TYPE_MIX[profile])
            bedrooms, bua = _size(rng, t)
            house = t in ("townhouse", "twin_house", "villa")
            floor = None if house else (rng.randint(0, 3) if t == "chalet" else rng.randint(0, 9))
            if t == "penthouse":
                floor = rng.randint(6, 10)
            garden = 0
            if house:
                garden = rng.randint(80, 600 if t == "villa" else 250)
            elif floor == 0 and rng.random() < 0.6:
                garden = rng.randint(30, 150)
            u_amen = set(amen)
            if garden:
                u_amen.add("garden")
            if t == "penthouse" or (t in ("villa", "duplex") and rng.random() < 0.4):
                u_amen.add("roof")
            if (t == "villa" and rng.random() < 0.25) or (t == "chalet" and rng.random() < 0.05):
                u_amen.add("private_pool")
            if area.coastal and rng.random() < 0.35:
                u_amen.add("sea_view")
            if bedrooms >= 3 and rng.random() < 0.35:
                u_amen.add("maids_room")
            if house:
                finishing = _pick(rng, {"core_shell": 45, "semi_finished": 35, "fully_finished": 20})
            elif area.coastal:
                finishing = _pick(rng, {"fully_finished": 75, "semi_finished": 25})
            else:
                finishing = _pick(rng, {"fully_finished": 45, "semi_finished": 40, "core_shell": 15})
            if delivered:
                delivery_year, ready = rng.choice([2024, 2025, 2026]), True
            else:
                delivery_year, ready = rng.choice([2027, 2027, 2028, 2028, 2029, 2030, 2031]), False
            price = bua * ppm2 * TYPE_PRICE_MULT[t] * FINISH_MULT[finishing] * rng.uniform(0.93, 1.08)
            price *= 1.08 if ready else 1.0
            price *= 1.12 if "sea_view" in u_amen else 1.0
            price += garden * ppm2 * 0.15
            price = int(round(price, -4))
            status = _pick(rng, {"available": 82, "reserved": 12, "sold": 6})
            updated = NOW - timedelta(days=rng.randint(0, 150), hours=rng.randint(0, 23))

            candidates = [
                s
                for s in SOFT_TEXT
                if not (area.coastal and s in INLAND_ONLY)
                and not (not area.coastal and s in COASTAL_ONLY)
                and not (s == "ground_floor" and floor != 0)
                and not (s == "high_floor" and (floor or 0) < 6)
            ]
            tags = rng.sample(candidates, rng.randint(2, 4))
            code = f"{AREA_CODE[area_slug]}-{prefix}-{t[0].upper()}{k + 101:03d}"
            te, ta = TYPE_LABEL[t]
            bed_en = "studio" if t == "studio" else f"{bedrooms}-bedroom"
            bed_ar = (
                "" if t == "studio" else {1: " غرفة واحدة", 2: " غرفتين"}.get(bedrooms, f" {bedrooms} غرف")
            )
            floor_en = (
                "" if floor is None else (" on the ground floor" if floor == 0 else f" on floor {floor}")
            )
            fin_en = {
                "fully_finished": "Delivered fully finished.",
                "semi_finished": "Delivered semi-finished.",
                "core_shell": "Delivered core and shell.",
            }[finishing]
            fin_ar = {
                "fully_finished": "تسليم متشطب بالكامل.",
                "semi_finished": "تسليم نص تشطيب.",
                "core_shell": "تسليم طوب أحمر.",
            }[finishing]
            deliv_en = "Ready to move now." if ready else f"Delivery {delivery_year}."
            deliv_ar = "استلام فوري." if ready else f"استلام {delivery_year}."
            extra_en = [rng.choice(SOFT_TEXT[s][0]) for s in tags]
            extra_ar = [rng.choice(SOFT_TEXT[s][1]) for s in tags]
            if garden:
                extra_en.append(f"Private garden of {garden} m2.")
                extra_ar.append(f"جنينة خاصة {garden} متر.")
            if "sea_view" in u_amen:
                extra_en.append("Direct sea view.")
                extra_ar.append("فيو مباشر على البحر.")
            desc_en = (
                f"{bed_en.capitalize()} {te.lower()} of {bua} m2{floor_en} in {name_en}, {area.name_en}. "
                + " ".join(extra_en)
                + f" {fin_en} {deliv_en}"
            )
            desc_ar = (
                f"{ta}{bed_ar} مساحة {bua} متر في {name_ar}، {area.name_ar}. "
                + " ".join(extra_ar)
                + f" {fin_ar} {deliv_ar}"
            )
            units.append(
                UnitRow(
                    code=code,
                    compound=slug,
                    area=area_slug,
                    unit_type=t,
                    bedrooms=bedrooms,
                    bathrooms=max(1, bedrooms - (0 if house else 1) + (1 if bedrooms >= 4 else 0)),
                    bua_m2=bua,
                    garden_m2=garden,
                    floor=floor,
                    finishing=finishing,
                    price=price,
                    delivery_year=delivery_year,
                    is_ready=ready,
                    status=status,
                    amenities=sorted(u_amen),
                    description_en=desc_en,
                    description_ar=desc_ar,
                    updated_at=updated.isoformat(),
                    soft_tags=tags,
                )
            )
    return developers, compounds, units


def manifest(units: list[UnitRow]) -> dict:
    payload = json.dumps([asdict(u) for u in units], sort_keys=True, ensure_ascii=False).encode()
    return {"seed": SEED, "units": len(units), "sha256": hashlib.sha256(payload).hexdigest()}


def soft_tags_by_code(seed: int = SEED) -> dict[str, list[str]]:
    return {u.code: u.soft_tags for u in generate(seed)[2]}
