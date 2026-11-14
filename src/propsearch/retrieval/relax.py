"""Zero-result fallback: find the smallest relaxation of the filters that returns units, and explain it.

Also reports which constraints are *binding* (relaxing them alone unlocks units). That list is logged
per query and aggregated on /insights as unmet demand.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from propsearch.data.catalog import AREAS
from propsearch.domain import Relaxation, SearchFilters, UnitType
from propsearch.fmt import amenity_name, area_name, egp, finish_name, type_name
from propsearch.retrieval import sql

TYPE_SIBLINGS = {
    "townhouse": ["twin_house"],
    "twin_house": ["townhouse", "villa"],
    "villa": ["twin_house"],
    "apartment": ["duplex"],
    "duplex": ["apartment", "penthouse"],
    "penthouse": ["duplex"],
    "studio": ["apartment"],
    "chalet": ["twin_house"],
}
MAX_STEPS = 3


@dataclass
class Option:
    cost: int
    field: str
    filters: SearchFilters
    relax: Relaxation


def _rel(field: str, a: str, b: str, en: str, ar: str) -> Relaxation:
    return Relaxation(field=field, from_value=a, to_value=b, en=en, ar=ar, unlocked=0)


def options(f: SearchFilters) -> list[Option]:
    out: list[Option] = []

    def add(cost: int, field: str, rel: Relaxation, **changes) -> None:
        out.append(Option(cost, field, f.model_copy(update=changes), rel))

    if f.budget_max:
        for pct, cost in ((10, 1), (20, 2), (35, 4)):
            nb = int(f.budget_max * (1 + pct / 100))
            add(
                cost,
                "budget_max",
                _rel(
                    "budget_max",
                    egp(f.budget_max),
                    egp(nb),
                    f"raised the budget {pct}% to {egp(nb)}",
                    f"زودت الميزانية {pct}% لـ {egp(nb, 'ar')}",
                ),
                budget_max=nb,
            )
    if f.budget_min:
        nb = int(f.budget_min * 0.8)
        add(
            1,
            "budget_min",
            _rel(
                "budget_min",
                egp(f.budget_min),
                egp(nb),
                f"lowered the minimum price to {egp(nb)}",
                f"نزلت أقل سعر لـ {egp(nb, 'ar')}",
            ),
            budget_min=nb,
        )
    for a in f.amenities:
        rest = [x for x in f.amenities if x != a]
        add(
            2,
            "amenities",
            _rel(
                "amenities",
                a.value,
                "",
                f"dropped the {amenity_name(a.value)} requirement",
                f"شلت شرط الـ{amenity_name(a.value, 'ar')}",
            ),
            amenities=rest,
        )
    if f.installment_years_min:
        for d, cost in ((1, 1), (2, 2), (3, 3)):
            y = f.installment_years_min - d
            if y >= 1:
                add(
                    cost,
                    "installment_years_min",
                    _rel(
                        "installment_years_min",
                        str(f.installment_years_min),
                        str(y),
                        f"accepted a {y}-year plan instead of {f.installment_years_min}",
                        f"قبلت تقسيط {y} سنين بدل {f.installment_years_min}",
                    ),
                    installment_years_min=y,
                )
    if f.down_payment_max_pct is not None:
        for d, cost in ((5, 1), (10, 2), (15, 3)):
            p = f.down_payment_max_pct + d
            add(
                cost,
                "down_payment_max_pct",
                _rel(
                    "down_payment_max_pct",
                    f"{f.down_payment_max_pct:g}%",
                    f"{p:g}%",
                    f"allowed a down payment up to {p:g}%",
                    f"سمحت بمقدم لحد {p:g}%",
                ),
                down_payment_max_pct=p,
            )
    if f.down_payment_max_amount:
        nb = int(f.down_payment_max_amount * 1.3)
        add(
            2,
            "down_payment_max_amount",
            _rel(
                "down_payment_max_amount",
                egp(f.down_payment_max_amount),
                egp(nb),
                f"allowed a down payment up to {egp(nb)}",
                f"سمحت بمقدم لحد {egp(nb, 'ar')}",
            ),
            down_payment_max_amount=nb,
        )
    if f.monthly_installment_max:
        for pct, cost in ((15, 2), (30, 3)):
            nb = int(f.monthly_installment_max * (1 + pct / 100))
            add(
                cost,
                "monthly_installment_max",
                _rel(
                    "monthly_installment_max",
                    egp(f.monthly_installment_max),
                    egp(nb),
                    f"allowed monthly installments up to {egp(nb)}",
                    f"سمحت بقسط شهري لحد {egp(nb, 'ar')}",
                ),
                monthly_installment_max=nb,
            )
    if f.ready_to_move:
        add(
            2,
            "ready_to_move",
            _rel(
                "ready_to_move", "ready", "2027", "included units delivering by 2027", "ضفت وحدات استلام 2027"
            ),
            ready_to_move=None,
            delivery_year_max=2027,
        )
    if f.delivery_year_max:
        for d, cost in ((1, 1), (2, 2)):
            y = f.delivery_year_max + d
            add(
                cost,
                "delivery_year_max",
                _rel(
                    "delivery_year_max",
                    str(f.delivery_year_max),
                    str(y),
                    f"extended delivery to {y}",
                    f"مديت الاستلام لـ {y}",
                ),
                delivery_year_max=y,
            )
    if f.bedrooms_min is not None and f.bedrooms_min == f.bedrooms_max:
        n = f.bedrooms_min
        add(
            2,
            "bedrooms",
            _rel(
                "bedrooms",
                str(n),
                f"{n}-{n + 1}",
                f"included {n + 1}-bedroom units",
                f"ضفت وحدات {n + 1} غرف",
            ),
            bedrooms_max=n + 1,
        )
        if n > 1:
            add(
                2,
                "bedrooms",
                _rel(
                    "bedrooms",
                    str(n),
                    f"{n - 1}-{n}",
                    f"included {n - 1}-bedroom units",
                    f"ضفت وحدات {n - 1} غرف",
                ),
                bedrooms_min=n - 1,
            )
    if f.bua_min:
        nb = int(f.bua_min * 0.85)
        add(
            1,
            "bua_min",
            _rel("bua_min", str(f.bua_min), str(nb), f"accepted from {nb} m2", f"قبلت من {nb} متر"),
            bua_min=nb,
        )
    if f.bua_max:
        nb = int(f.bua_max * 1.15)
        add(
            1,
            "bua_max",
            _rel("bua_max", str(f.bua_max), str(nb), f"accepted up to {nb} m2", f"قبلت لحد {nb} متر"),
            bua_max=nb,
        )
    if f.finishing:
        add(
            2,
            "finishing",
            _rel(
                "finishing",
                ",".join(x.value for x in f.finishing),
                "any",
                f"included any finishing (not only {finish_name(f.finishing[0].value)})",
                f"ضفت أي تشطيب (مش بس {finish_name(f.finishing[0].value, 'ar')})",
            ),
            finishing=[],
        )
    if f.areas:
        extra = sorted({n for a in f.areas for n in AREAS[a].neighbors} - set(f.areas))
        if extra:
            names_en = ", ".join(area_name(a) for a in extra)
            names_ar = "، ".join(area_name(a, "ar") for a in extra)
            add(
                3,
                "areas",
                _rel(
                    "areas",
                    ",".join(f.areas),
                    ",".join(extra),
                    f"added nearby areas: {names_en}",
                    f"ضفت مناطق قريبة: {names_ar}",
                ),
                areas=sorted(set(f.areas) | set(extra)),
            )
    if f.unit_types:
        sib = sorted(
            {s for t in f.unit_types for s in TYPE_SIBLINGS[t.value]} - {t.value for t in f.unit_types}
        )
        if sib:
            add(
                3,
                "unit_types",
                _rel(
                    "unit_types",
                    ",".join(t.value for t in f.unit_types),
                    ",".join(sib),
                    "also included " + ", ".join(type_name(s) for s in sib),
                    "ضفت كمان " + "، ".join(type_name(s, "ar") for s in sib),
                ),
                unit_types=[*f.unit_types, *(UnitType(s) for s in sib)],
            )
    if f.near_landmark and f.max_distance_km:
        r = f.max_distance_km * 2
        add(
            2,
            "max_distance_km",
            _rel(
                "max_distance_km",
                f"{f.max_distance_km:g}",
                f"{r:g}",
                f"widened the search radius to {r:g} km",
                f"وسعت نطاق البحث لـ {r:g} كم",
            ),
            max_distance_km=r,
        )
    return out


def relax(session: Session, f: SearchFilters) -> tuple[SearchFilters, list[Relaxation], list[dict]]:
    """Returns (relaxed filters, applied relaxations, binding constraints)."""
    opts = options(f)
    counted = [(o, sql.count(session, o.filters)) for o in opts]
    binding: dict[str, int] = {}
    for o, n in counted:
        if n > 0:
            binding[o.field] = max(binding.get(o.field, 0), n)
    binding_list = [{"field": k, "unlocks": v} for k, v in sorted(binding.items(), key=lambda kv: -kv[1])]
    winners = [(o, n) for o, n in counted if n > 0]
    if winners:
        o, n = min(winners, key=lambda x: (x[0].cost, -x[1]))
        return o.filters, [o.relax.model_copy(update={"unlocked": n})], binding_list

    # nothing single works: relax step by step, cheapest field first, each field at most once
    current, applied, used = f, [], set()
    for _ in range(MAX_STEPS):
        opts = [o for o in options(current) if o.field not in used]
        if not opts:
            break
        o = min(opts, key=lambda x: x.cost)
        used.add(o.field)
        current = o.filters
        n = sql.count(session, current)
        applied.append(o.relax.model_copy(update={"unlocked": n}))
        if n > 0:
            return current, applied, binding_list
        winners = [(x, sql.count(session, x.filters)) for x in options(current) if x.field not in used]
        winners = [(x, c) for x, c in winners if c > 0]
        if winners:
            x, c = min(winners, key=lambda w: (w[0].cost, -w[1]))
            applied.append(x.relax.model_copy(update={"unlocked": c}))
            return x.filters, applied, binding_list
    return f, [], binding_list
