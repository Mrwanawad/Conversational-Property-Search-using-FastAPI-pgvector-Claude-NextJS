"""Deterministic query parser over the concept tags from normalize.tag().

Used offline (tests, CI, demo without an API key), as the fallback when the LLM call fails,
and as the baseline the Claude parser is measured against in evals/.
"""

from __future__ import annotations

from dataclasses import dataclass

from propsearch.data.catalog import AREA_GROUPS, AREAS, LANDMARKS, haversine_km
from propsearch.domain import Ambiguity, ParsedQuery, SearchFilters
from propsearch.normalize import Match, TaggedQuery, tag

SOFT_PHRASES = {
    "soft:quiet": "quiet calm area",
    "soft:family": "family friendly",
    "soft:schools": "close to schools",
    "soft:hospital": "close to a hospital",
    "soft:light": "natural light bright",
    "soft:view": "open view",
    "soft:privacy": "privacy",
    "soft:open_kitchen": "open kitchen",
    "soft:smart_home": "smart home",
    "soft:pets": "pet friendly",
    "soft:corner": "corner unit",
    "soft:high_floor": "high floor",
    "soft:ground_floor": "ground floor",
    "soft:investment": "good for rental investment",
    "soft:walkable": "walking distance to services",
    "soft:metro": "near monorail station",
    "soft:green": "green landscape",
    "soft:modern": "modern design",
    "soft:storage": "storage room",
    "soft:sunset": "sunset view",
}

MONEY_CONTEXT = {"kw:under", "kw:over", "kw:around", "kw:between", "kw:from"}
DEFAULT_LANDMARK_RADIUS_KM = 12.0
CURRENT_YEAR = 2026


@dataclass
class Atom:
    kind: str  # "num" | "c" | "w"
    value: float | str
    magnitude: str = ""  # "" | "million" | "thousand" | "percent" | "meter"
    pos: int = 0
    used: bool = False
    surface: str = ""


# Short forms that only mean "million" after a number: bare "mn" is Franco for "from" ("2orayeba mn").
_MILLION_ABBR = {"m", "mn", "mil", "mill", "mio"}


def _atoms(tq: TaggedQuery) -> list[Atom]:
    raw: list[Atom] = []
    for it in tq.items:
        if isinstance(it, Match):
            nums = [c for c in it.concepts if c.startswith("num:")]
            duals = [c for c in it.concepts if c.startswith("dual:")]
            if nums:
                raw.append(Atom("num", float(nums[0].split(":")[1])))
            elif duals:
                raw.append(Atom("num", 2.0))
                kind = duals[0].split(":")[1]
                raw.append(
                    Atom(
                        "c",
                        {
                            "million": "unit:million",
                            "years": "kw:years",
                            "bedrooms": "kw:bedrooms",
                            "thousand": "unit:thousand",
                        }[kind],
                    )
                )
            else:
                for c in it.concepts:
                    raw.append(Atom("c", c, surface=it.surface))
        else:
            tok = it
            if tok.replace(".", "", 1).isdigit():
                raw.append(Atom("num", float(tok)))
            elif tok == "+":
                raw.append(Atom("c", "kw:plus"))
            else:
                raw.append(Atom("w", tok))
    # fold magnitudes into numbers: "8 مليون ونص", "نص مليون", "800 الف", "10 %", "مليون و 200 الف"
    out: list[Atom] = []
    i = 0
    while i < len(raw):
        a = raw[i]
        nxt = raw[i + 1] if i + 1 < len(raw) else None
        if a.kind == "c" and a.value == "unit:half" and nxt and nxt.value == "unit:million":
            out.append(Atom("num", 500_000, "million"))
            i += 2
            continue
        if (
            a.kind == "c"
            and a.value == "unit:million"
            and a.surface not in _MILLION_ABBR
            and (not out or out[-1].kind != "num" or out[-1].magnitude)
        ):
            out.append(Atom("num", 1_000_000, "million"))
            i += 1
            i = _absorb_fraction(raw, i, out[-1])
            continue
        if a.kind == "num" and nxt and nxt.kind == "c":
            unit = nxt.value
            if unit == "unit:million":
                if a.value > 60 and raw[i + 1 : i + 2] and _surface_is_m(tq, a, nxt):
                    out.append(Atom("num", a.value, "meter"))
                else:
                    out.append(Atom("num", a.value * 1_000_000, "million"))
                i += 2
                i = _absorb_fraction(raw, i, out[-1])
                continue
            if unit == "unit:thousand":
                out.append(Atom("num", a.value * 1_000, "thousand"))
                i += 2
                continue
            if unit == "unit:percent":
                out.append(Atom("num", a.value, "percent"))
                i += 2
                continue
        out.append(a)
        i += 1
    # "مليون و 200 الف" -> 1.2M
    merged: list[Atom] = []
    for a in out:
        if (
            a.kind == "num"
            and a.magnitude == "thousand"
            and len(merged) >= 2
            and merged[-1].kind == "c"
            and merged[-1].value == "kw:to"
            and merged[-2].kind == "num"
            and merged[-2].magnitude == "million"
            and a.value < 1_000_000
        ):
            merged.pop()
            merged[-1].value += a.value
            continue
        merged.append(a)
    for idx, a in enumerate(merged):
        a.pos = idx
    return merged


def _surface_is_m(tq: TaggedQuery, num: Atom, unit: Atom) -> bool:
    """'120m' / '120 m' with a number above 60 is meters, not millions."""
    return any(isinstance(it, Match) and it.surface in {"m"} for it in tq.items)


def _absorb_fraction(raw: list[Atom], i: int, money: Atom) -> int:
    if i < len(raw) and raw[i].kind == "c" and raw[i].value in ("unit:half", "unit:quarter"):
        base = 1_000_000 if money.magnitude == "million" else 1_000
        money.value += base * (0.5 if raw[i].value == "unit:half" else 0.25)
        return i + 1
    return i


def _near(atoms: list[Atom], pos: int, concepts: set[str], before: int = 3, after: int = 2) -> bool:
    lo, hi = max(0, pos - before), min(len(atoms), pos + after + 1)
    return any(a.kind == "c" and a.value in concepts for a in atoms[lo:hi] if a.pos != pos)


def _prev_concept(atoms: list[Atom], pos: int, concepts: set[str], window: int = 3) -> str | None:
    for a in reversed(atoms[max(0, pos - window) : pos]):
        if a.kind == "c" and a.value in concepts:
            return str(a.value)
    return None


def _next_is(atoms: list[Atom], pos: int, concept: str, skip: int = 1) -> bool:
    return any(a.kind == "c" and a.value == concept for a in atoms[pos + 1 : pos + 1 + skip])


def _money(a: Atom, context: str | None) -> float | None:
    """Value in EGP, or None if this number is not money."""
    if a.magnitude in ("million", "thousand"):
        return a.value
    if a.magnitude:
        return None
    if a.value >= 100_000:
        return a.value
    if context and a.value <= 100:
        return a.value * 1_000_000  # "تحت 8" in a budget context = 8M
    if context and 100 < a.value < 1000:
        return a.value * 1_000
    return None


def parse_turn(tq: TaggedQuery) -> tuple[SearchFilters, list[str], dict]:
    """Extract this turn's own constraints (no session merge). Returns filters, soft concepts, flags."""
    atoms = _atoms(tq)
    f: dict = {"areas": [], "unit_types": [], "finishing": [], "amenities": []}
    flags: dict = {
        "refine": set(),
        "reset": False,
        "ambiguous_zayed": False,
        "sort": None,
        "drop_amenities": set(),
        "drop_ready": False,
    }
    soft: list[str] = []

    concepts = [str(a.value) for a in atoms if a.kind == "c"]
    negated = -1  # index of the last negation cue; it applies to the next amenity / ready concept
    for i, c in enumerate(concepts):
        if c == "neg":
            negated = i
            continue
        if negated >= 0 and i - negated <= 3 and (c.startswith("amenity:") or c == "ready"):
            negated = -1
            if c == "ready":
                flags["drop_ready"] = True
            else:
                flags["drop_amenities"].add(c.split(":", 1)[1])
            continue
        if c.startswith("area:"):
            f["areas"].append(c.split(":", 1)[1])
        elif c.startswith("group:"):
            f["areas"].extend(AREA_GROUPS[c.split(":", 1)[1]])
        elif c == "ambiguous:zayed":
            flags["ambiguous_zayed"] = True
        elif c.startswith("type:"):
            f["unit_types"].append(c.split(":", 1)[1])
        elif c.startswith("finish:"):
            f["finishing"].append(c.split(":", 1)[1])
        elif c.startswith("amenity:"):
            f["amenities"].append(c.split(":", 1)[1])
        elif c == "ready":
            f["ready_to_move"] = True
        elif c.startswith("landmark:"):
            slug = c.split(":", 1)[1]
            lm = LANDMARKS[slug]
            prev = concepts[i - 1] if i > 0 else ""
            if prev == "kw:in" and lm.area:
                f["areas"].append(lm.area)
            f["near_landmark"] = slug
            f["max_distance_km"] = DEFAULT_LANDMARK_RADIUS_KM
        elif c.startswith("refine:"):
            flags["refine"].add(c.split(":", 1)[1])
        elif c == "reset":
            flags["reset"] = True
        elif c.startswith("soft:"):
            soft.append(c)

    nums = [a for a in atoms if a.kind == "num"]
    for a in nums:
        if a.used:
            continue
        p = a.pos
        # range: "من 5 ل 8 مليون", "5-8M", "between 5 and 8 million", "3-4 غرف"
        partner = None
        if (
            p + 2 < len(atoms)
            and atoms[p + 1].kind == "c"
            and atoms[p + 1].value == "kw:to"
            and atoms[p + 2].kind == "num"
        ):
            partner = atoms[p + 2]
        if a.magnitude == "percent":
            if _near(atoms, p, {"kw:down_payment"}, before=3, after=3):
                f["down_payment_max_pct"] = a.value
            a.used = True
            continue
        if (
            a.magnitude == "meter"
            or _next_is(atoms, p, "kw:sqm")
            or (partner and _next_is(atoms, p + 2, "kw:sqm"))
        ):
            lo, hi = a.value, (partner.value if partner else None)
            ctx = _prev_concept(atoms, p, MONEY_CONTEXT)
            if hi:
                f["bua_min"], f["bua_max"] = int(lo), int(hi)
                partner.used = True
            elif ctx == "kw:under":
                f["bua_max"] = int(lo)
            elif ctx == "kw:over":
                f["bua_min"] = int(lo)
            else:
                f["bua_min"], f["bua_max"] = int(lo * 0.85), int(lo * 1.2)
            a.used = True
            continue
        if (
            _next_is(atoms, p, "kw:bedrooms")
            or (partner and _next_is(atoms, p + 2, "kw:bedrooms"))
            or (_prev_concept(atoms, p, {"kw:bedrooms"}, window=1) and a.value <= 7 and not a.magnitude)
        ):
            if partner and partner.value <= 10:
                f["bedrooms_min"], f["bedrooms_max"] = int(a.value), int(partner.value)
                partner.used = True
            elif _next_is(atoms, p, "kw:plus") or _prev_concept(atoms, p, {"kw:over"}):
                f["bedrooms_min"] = int(a.value)
            else:
                f["bedrooms_min"] = f["bedrooms_max"] = int(a.value)
            a.used = True
            continue
        if _next_is(atoms, p, "kw:years") and a.value <= 15 and not a.magnitude:
            if _prev_concept(atoms, p, {"kw:delivery"}, window=3):
                f["delivery_year_max"] = CURRENT_YEAR + int(a.value)
            else:
                f["installment_years_min"] = int(a.value)
            a.used = True
            continue
        if 2024 <= a.value <= 2040 and not a.magnitude:
            f["delivery_year_max"] = int(a.value)
            a.used = True
            continue
        if not a.magnitude and a.value <= 15 and _prev_concept(atoms, p, {"kw:installments"}, window=2):
            f["installment_years_min"] = int(a.value)
            a.used = True
            continue
        ctx = _prev_concept(atoms, p, MONEY_CONTEXT | {"kw:down_payment", "kw:monthly"}, window=3)
        money = _money(a, ctx)
        if money is None:
            continue
        a.used = True
        if _prev_concept(atoms, p, {"kw:down_payment"}, window=3) or _next_is(atoms, p, "kw:down_payment"):
            f["down_payment_max_amount"] = int(money)
            continue
        if _near(atoms, p, {"kw:monthly"}, before=3, after=2):
            f["monthly_installment_max"] = int(money)
            continue
        if partner:
            hi = _money(partner, "kw:between")
            if partner.magnitude in ("million", "thousand") and not a.magnitude:
                money = a.value * (1_000_000 if partner.magnitude == "million" else 1_000)
            if hi:
                f["budget_min"], f["budget_max"] = int(money), int(hi)
                partner.used = True
                continue
        if ctx == "kw:over" or ctx == "kw:from":
            f["budget_min"] = int(money)
        elif ctx == "kw:around":
            f["budget_min"], f["budget_max"] = int(money * 0.85), int(money * 1.1)
        else:
            f["budget_max"] = int(money)

    if "sort:price_asc" in concepts:
        flags["sort"] = "price_asc"
    return SearchFilters(**f), soft, flags


class RuleParser:
    name = "rules"

    def parse(
        self, text: str, previous: ParsedQuery | None = None, context: dict | None = None
    ) -> ParsedQuery:
        tq = tag(text)
        return merge_turn(tq, *parse_turn(tq), previous=previous, context=context or {}, parser=self.name)


def merge_turn(
    tq: TaggedQuery,
    turn: SearchFilters,
    soft_concepts: list[str],
    flags: dict,
    previous: ParsedQuery | None,
    context: dict,
    parser: str,
) -> ParsedQuery:
    """Apply this turn on top of the session state (refinements like 'cheaper', 'closer to AUC')."""
    prev = previous if previous and not flags["reset"] else None
    base = prev.filters.model_dump() if prev else SearchFilters().model_dump()
    new = turn.model_dump(exclude_defaults=True)
    is_new_search = False
    if prev:
        type_changed = bool(turn.unit_types) and set(turn.unit_types) != set(prev.filters.unit_types)
        area_changed = (bool(turn.areas) and set(turn.areas) != set(prev.filters.areas)) or flags[
            "ambiguous_zayed"
        ]
        primary = sum(
            bool(x)
            for x in (
                turn.areas or flags["ambiguous_zayed"],
                turn.unit_types,
                turn.budget_max or turn.budget_min,
                turn.bedrooms_min is not None,
            )
        )
        is_new_search = (type_changed and area_changed) or (area_changed and primary >= 3)
    if is_new_search:
        base = SearchFilters().model_dump()
    base.update(new)
    # a landmark carried over from earlier turns no longer applies once the user moves far away
    if prev and not is_new_search and turn.areas and not turn.near_landmark and base.get("near_landmark"):
        lm = LANDMARKS[base["near_landmark"]]
        if all(haversine_km(lm.lat, lm.lng, AREAS[a].lat, AREAS[a].lng) > 25 for a in turn.areas):
            base["near_landmark"] = None
            base["max_distance_km"] = None
    # amenities accumulate across refinements ("and with a garden")
    if prev and not is_new_search and turn.amenities:
        base["amenities"] = sorted(set(prev.filters.amenities) | set(turn.amenities))
    # "not necessarily on the beach" / "مش لازم استلام فوري" removes a constraint instead of adding it
    if flags["drop_amenities"]:
        base["amenities"] = sorted(set(base["amenities"]) - flags["drop_amenities"])
    if flags["drop_ready"]:
        base["ready_to_move"] = None

    ref_price = context.get("median_price")
    ref_bua = context.get("median_bua")
    refine = flags["refine"]
    if "cheaper" in refine:
        anchor = (
            min(x for x in (base.get("budget_max"), ref_price) if x)
            if (base.get("budget_max") or ref_price)
            else None
        )
        if anchor:
            base["budget_max"] = int(anchor * 0.9)
            if base.get("budget_min") and base["budget_min"] > base["budget_max"]:
                base["budget_min"] = None
    if "pricier" in refine and ref_price:
        base["budget_min"] = int(ref_price * 1.1)
        if base.get("budget_max") and base["budget_max"] < base["budget_min"]:
            base["budget_max"] = None
    if "bigger" in refine and ref_bua:
        base["bua_min"] = int(ref_bua * 1.15)
        if base.get("bua_max") and base["bua_max"] < base["bua_min"]:
            base["bua_max"] = None
        if base.get("bedrooms_max") is not None:
            base["bedrooms_max"] = None
    if "smaller" in refine and ref_bua:
        base["bua_max"] = int(ref_bua * 0.85)
        base["bua_min"] = None

    filters = SearchFilters(**base)
    prev_soft = prev.soft_preferences if prev and not is_new_search else ""
    soft_text = "; ".join(SOFT_PHRASES[c] for c in dict.fromkeys(soft_concepts))
    soft = "; ".join(x for x in (prev_soft, soft_text) if x)
    soft = "; ".join(dict.fromkeys(s.strip() for s in soft.split(";") if s.strip()))

    ambiguities = []
    if flags["ambiguous_zayed"] and not {"sheikh_zayed", "new_zayed"} & set(filters.areas):
        ambiguities.append(Ambiguity(field="areas", surface="zayed", options=["sheikh_zayed", "new_zayed"]))

    if flags["reset"]:
        intent = "reset"
    elif "more" in refine and turn.is_empty() and not soft_concepts:
        intent = "more"
    elif prev and not is_new_search:
        intent = "refine"
    else:
        intent = "search"

    known = sum(1 for it in tq.items if isinstance(it, Match))
    unknown = len(tq.unknown_terms)
    confidence = round(known / (known + unknown), 3) if known + unknown else 0.0
    sort = flags.get("sort") or (prev.sort if prev and not is_new_search else "relevance")
    return ParsedQuery(
        filters=filters,
        soft_preferences=soft,
        sort=sort,
        intent=intent,
        ambiguities=ambiguities,
        parser=parser,
        confidence=confidence,
        unknown_terms=tq.unknown_terms,
    )
