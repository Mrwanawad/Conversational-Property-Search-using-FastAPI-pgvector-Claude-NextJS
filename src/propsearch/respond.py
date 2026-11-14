"""Grounded replies. Every number shown comes from a DB field or arithmetic on DB fields.

Reasons and the reply message are deterministic templates. An optional Claude narrative
(settings.llm_summary) is accepted only if every number in it can be traced to the facts it was
given; otherwise the template is kept.
"""

from __future__ import annotations

import json
import logging
import re

import numpy as np

from propsearch.data.catalog import AREAS
from propsearch.db import Unit
from propsearch.domain import (
    Clarification,
    ParsedQuery,
    PlanSummary,
    Reason,
    Relaxation,
    SearchFilters,
    UnitHit,
)
from propsearch.fmt import amenity_name, area_name, egp, finish_name, landmark_name, type_name
from propsearch.retrieval.ranker import Scored

log = logging.getLogger(__name__)
_SENT = re.compile(r"(?<=[.!؟?])\s+")


def best_plan(unit: Unit, f: SearchFilters) -> PlanSummary | None:
    """Plan that satisfies the user's plan constraints with the lowest monthly; else the longest plan."""
    plans = []
    for p in unit.compound.plans:
        if p.years <= 0:
            continue
        dp = unit.price * p.down_payment_pct / 100
        monthly = (unit.price - dp) / (p.years * 12)
        ok = (
            (f.installment_years_min is None or p.years >= f.installment_years_min)
            and (f.down_payment_max_pct is None or p.down_payment_pct <= f.down_payment_max_pct)
            and (f.down_payment_max_amount is None or dp <= f.down_payment_max_amount)
            and (f.monthly_installment_max is None or monthly <= f.monthly_installment_max)
        )
        plans.append((ok, p, int(round(dp, -3)), int(round(monthly, -2))))
    if not plans:
        return None
    ok = [x for x in plans if x[0]]
    _, p, dp, monthly = min(ok, key=lambda x: x[3]) if ok else max(plans, key=lambda x: x[1].years)
    return PlanSummary(
        name=p.name, down_payment_pct=p.down_payment_pct, down_payment=dp, years=p.years, monthly=monthly
    )


def cash_discount(unit: Unit) -> float:
    return max((p.cash_discount_pct for p in unit.compound.plans if p.years == 0), default=0.0)


def evidence_sentence(unit: Unit, soft: str, lang: str, scorer) -> str | None:
    """Description sentence that best supports the soft preference (quoted verbatim, so grounded)."""
    if not soft or scorer is None:
        return None
    # Score the English sentences (same language as the soft text); quote the parallel Arabic sentence
    # when replying in Arabic. Descriptions are written sentence-aligned.
    en = _SENT.split(unit.description_en)[1:]  # skip the generic first sentence
    ar = _SENT.split(unit.description_ar)[1:]
    if not en:
        return None
    scores = scorer(soft, en)
    i = int(np.argmax(scores))
    if scores[i] <= 0:
        return None
    return ar[i] if lang == "ar" and len(ar) == len(en) else en[i]


def reasons_for(
    unit: Unit,
    s: Scored,
    pq: ParsedQuery,
    plan: PlanSummary | None,
    evidence: str | None,
    asked: SearchFilters | None = None,
) -> list[Reason]:
    f = pq.filters
    asked = asked or f  # what the user asked for, before any relaxation
    out: list[Reason] = []
    if f.areas or f.compounds:
        out.append(
            Reason(
                kind="match",
                en=f"In {unit.compound.name_en}, {area_name(unit.area)}",
                ar=f"في {unit.compound.name_ar}، {area_name(unit.area, 'ar')}",
            )
        )
    if f.bedrooms_min is not None or f.bedrooms_max is not None:
        out.append(Reason(kind="match", en=f"{unit.bedrooms} bedrooms", ar=f"{unit.bedrooms} غرف"))
    if asked.budget_max and unit.price > asked.budget_max:
        over = unit.price - asked.budget_max
        out.append(
            Reason(
                kind="relaxed",
                en=f"{egp(unit.price)}, {egp(over)} over your {egp(asked.budget_max)} budget",
                ar=f"{egp(unit.price, 'ar')}، أعلى من ميزانيتك بـ {egp(over, 'ar')}",
            )
        )
    elif f.budget_max:
        gap = asked.budget_max - unit.price if asked.budget_max else f.budget_max - unit.price
        budget = asked.budget_max or f.budget_max
        en = (
            f"{egp(unit.price)}, {egp(gap)} under your {egp(budget)} budget"
            if gap >= 10_000
            else f"{egp(unit.price)}, at your {egp(budget)} budget"
        )
        ar = (
            f"{egp(unit.price, 'ar')}، أقل من ميزانيتك بـ {egp(gap, 'ar')}"
            if gap >= 10_000
            else f"{egp(unit.price, 'ar')}، في حدود ميزانيتك"
        )
        out.append(Reason(kind="match", en=en, ar=ar))
    plan_filter = any(
        v is not None
        for v in (
            f.installment_years_min,
            f.down_payment_max_pct,
            f.down_payment_max_amount,
            f.monthly_installment_max,
        )
    )
    if plan:
        en = (
            f"{plan.down_payment_pct:g}% down ({egp(plan.down_payment)}), {plan.years} years, "
            f"about {egp(plan.monthly)}/month"
        )
        ar = (
            f"مقدم {plan.down_payment_pct:g}% ({egp(plan.down_payment, 'ar')})، {plan.years} سنين، "
            f"حوالي {egp(plan.monthly, 'ar')} في الشهر"
        )
        out.append(Reason(kind="match" if plan_filter else "info", en=en, ar=ar))
    if f.ready_to_move:
        out.append(Reason(kind="match", en="Ready to move", ar="استلام فوري"))
    elif f.delivery_year_max:
        out.append(
            Reason(kind="match", en=f"Delivery {unit.delivery_year}", ar=f"استلام {unit.delivery_year}")
        )
    for a in f.amenities:
        extra_en = f" ({unit.garden_m2} m2)" if a.value == "garden" and unit.garden_m2 else ""
        extra_ar = f" ({unit.garden_m2} متر)" if a.value == "garden" and unit.garden_m2 else ""
        out.append(
            Reason(
                kind="match",
                en=f"Has {amenity_name(a.value)}{extra_en}",
                ar=f"فيها {amenity_name(a.value, 'ar')}{extra_ar}",
            )
        )
    if f.finishing:
        out.append(
            Reason(
                kind="match",
                en=finish_name(unit.finishing).capitalize(),
                ar=finish_name(unit.finishing, "ar"),
            )
        )
    if f.near_landmark and s.distance_km is not None:
        out.append(
            Reason(
                kind="near",
                en=f"{s.distance_km:.1f} km from {landmark_name(f.near_landmark)}",
                ar=f"{s.distance_km:.1f} كم من {landmark_name(f.near_landmark, 'ar')}",
            )
        )
    if evidence:
        out.append(Reason(kind="match", en=f'"{evidence}"', ar=f'"{evidence}"'))
    if s.parts.get("value", 0) >= 0.6:
        out.append(
            Reason(
                kind="info",
                en="Priced below the area's typical price per m2",
                ar="سعر المتر أقل من متوسط المنطقة",
            )
        )
    return out


def to_hit(
    unit: Unit, s: Scored, pq: ParsedQuery, lang: str, scorer, asked: SearchFilters | None = None
) -> UnitHit:
    plan = best_plan(unit, pq.filters)
    evidence = evidence_sentence(unit, pq.soft_preferences, lang, scorer)
    te, ta = type_name(unit.unit_type), type_name(unit.unit_type, "ar")
    beds_en = "Studio" if unit.unit_type == "studio" else f"{unit.bedrooms}BR {te}"
    beds_ar = ta if unit.unit_type == "studio" else f"{ta} {unit.bedrooms} غرف"
    return UnitHit(
        id=unit.id,
        code=unit.code,
        title_en=f"{beds_en} in {unit.compound.name_en}",
        title_ar=f"{beds_ar} في {unit.compound.name_ar}",
        compound=unit.compound.name_en,
        compound_ar=unit.compound.name_ar,
        developer=unit.compound.developer.name_en,
        area=unit.area,
        area_en=area_name(unit.area),
        area_ar=area_name(unit.area, "ar"),
        unit_type=unit.unit_type,
        bedrooms=unit.bedrooms,
        bathrooms=unit.bathrooms,
        bua_m2=unit.bua_m2,
        garden_m2=unit.garden_m2,
        floor=unit.floor,
        finishing=unit.finishing,
        price=unit.price,
        price_per_m2=int(unit.price / unit.bua_m2),
        delivery="ready" if unit.is_ready else str(unit.delivery_year),
        status=unit.status,
        amenities=list(unit.amenities),
        best_plan=plan,
        distance_km=s.distance_km,
        score=s.score,
        reasons=reasons_for(unit, s, pq, plan, evidence, asked),
        description_en=unit.description_en,
        description_ar=unit.description_ar,
    )


def describe_filters(f: SearchFilters, soft: str, lang: str) -> str:
    """Human summary of the active filters, e.g. '3-bedroom apartments in New Cairo under 8M EGP'."""
    ar = lang == "ar"
    parts: list[str] = []
    types = [type_name(t.value, lang) for t in f.unit_types]
    head = " / ".join(types) if types else ("وحدات" if ar else "units")
    if f.bedrooms_min is not None:
        b = (
            f"{f.bedrooms_min}"
            if f.bedrooms_min == f.bedrooms_max
            else f"{f.bedrooms_min}-{f.bedrooms_max}"
            if f.bedrooms_max
            else f"{f.bedrooms_min}+"
        )
        head = f"{head} {b} غرف" if ar else f"{b}-bedroom {head}"
    parts.append(head)
    if f.areas:
        sep = " و" if ar else " or "
        parts.append(("في " if ar else "in ") + sep.join(area_name(a, lang) for a in f.areas))
    if f.budget_min and f.budget_max:
        parts.append(
            f"من {egp(f.budget_min, 'ar')} لـ {egp(f.budget_max, 'ar')}"
            if ar
            else f"between {egp(f.budget_min)} and {egp(f.budget_max)}"
        )
    elif f.budget_max:
        parts.append(f"تحت {egp(f.budget_max, 'ar')}" if ar else f"under {egp(f.budget_max)}")
    elif f.budget_min:
        parts.append(f"فوق {egp(f.budget_min, 'ar')}" if ar else f"over {egp(f.budget_min)}")
    if f.installment_years_min:
        parts.append(
            f"تقسيط {f.installment_years_min}+ سنين" if ar else f"{f.installment_years_min}+ year plans"
        )
    if f.down_payment_max_pct is not None:
        parts.append(
            f"مقدم لحد {f.down_payment_max_pct:g}%" if ar else f"up to {f.down_payment_max_pct:g}% down"
        )
    if f.down_payment_max_amount:
        parts.append(
            f"مقدم لحد {egp(f.down_payment_max_amount, 'ar')}"
            if ar
            else f"down payment up to {egp(f.down_payment_max_amount)}"
        )
    if f.monthly_installment_max:
        parts.append(
            f"قسط لحد {egp(f.monthly_installment_max, 'ar')} شهريا"
            if ar
            else f"monthly up to {egp(f.monthly_installment_max)}"
        )
    if f.ready_to_move:
        parts.append("استلام فوري" if ar else "ready to move")
    elif f.delivery_year_max:
        parts.append(f"استلام لحد {f.delivery_year_max}" if ar else f"delivery by {f.delivery_year_max}")
    if f.amenities:
        sep = " و" if ar else " and "
        parts.append(
            ("فيها " if ar else "with ") + sep.join(amenity_name(a.value, lang) for a in f.amenities)
        )
    if f.finishing:
        parts.append(" / ".join(finish_name(x.value, lang) for x in f.finishing))
    if f.near_landmark:
        parts.append(
            f"قريب من {landmark_name(f.near_landmark, 'ar')}"
            if ar
            else f"near {landmark_name(f.near_landmark)}"
        )
    if soft:
        parts.append(f"({soft})")
    return "، ".join(parts) if ar else ", ".join(parts)


def units_en(n: int) -> str:
    return "1 unit" if n == 1 else f"{n} units"


def units_ar(n: int) -> str:
    if n == 1:
        return "وحدة واحدة"
    if n == 2:
        return "وحدتين"
    return f"{n} وحدات" if 3 <= n <= 10 else f"{n} وحدة"


def template_message(
    pq: ParsedQuery,
    total: int,
    shown: int,
    relaxations: list[Relaxation],
    lang: str,
    original: SearchFilters | None = None,
    price_floor: int | None = None,
) -> str:
    ar = lang == "ar"
    msg = _template_message(pq, total, shown, relaxations, lang, original)
    if total == 0 and price_floor:
        msg += (
            f" أرخص وحدة بنفس المواصفات بتبدأ من {egp(price_floor, 'ar')}."
            if ar
            else f" The cheapest unit with these specs starts at {egp(price_floor)}."
        )
    return msg


def _template_message(
    pq: ParsedQuery,
    total: int,
    shown: int,
    relaxations: list[Relaxation],
    lang: str,
    original: SearchFilters | None,
) -> str:
    ar = lang == "ar"
    if relaxations:
        what = describe_filters(original or pq.filters, "", lang)
        steps = "، ثم ".join(r.ar for r in relaxations) if ar else ", then ".join(r.en for r in relaxations)
        if total == 0:
            return (
                f"مفيش وحدات متاحة لـ {what} حتى بعد ما {steps}. جرب منطقة أو ميزانية تانية."
                if ar
                else f"No units match {what}, even after I {steps}. Try another area or budget."
            )
        return (
            f"مفيش تطابق كامل لـ {what}. {steps} ولقيت {units_ar(total)}. دول أفضل {shown}:"
            if ar
            else f"No exact matches for {what}, so I {steps} and found {units_en(total)}. Top {shown}:"
        )
    what = describe_filters(pq.filters, pq.soft_preferences, lang)
    if total == 0:
        return f"مفيش وحدات متاحة لـ {what}." if ar else f"No units match {what}."
    return (
        f"لقيت {units_ar(total)}: {what}. دول أفضل {shown}:"
        if ar
        else f"Found {units_en(total)}: {what}. Top {shown}:"
    )


def suggestions(pq: ParsedQuery, total: int, shown: int, lang: str) -> list[dict]:
    ar = lang == "ar"
    f = pq.filters
    out = []
    if total:
        out.append({"label": "أرخص شوية" if ar else "Cheaper", "value": "ارخص شوية" if ar else "cheaper"})
    if not f.ready_to_move:
        out.append(
            {
                "label": "استلام فوري بس" if ar else "Ready to move only",
                "value": "استلام فوري" if ar else "ready to move",
            }
        )
    if "garden" not in f.amenities and not any(t.value in ("chalet", "studio") for t in f.unit_types):
        out.append(
            {"label": "فيها جنينة" if ar else "With a garden", "value": "بجنينة" if ar else "with garden"}
        )
    if f.areas == ["new_cairo"] and f.near_landmark != "auc":
        out.append(
            {
                "label": "أقرب للجامعة الأمريكية" if ar else "Closer to AUC",
                "value": "اقرب للجامعة الامريكية" if ar else "closer to AUC",
            }
        )
    if total > shown:
        out.append({"label": "اختيارات تانية" if ar else "More options", "value": "غيرهم" if ar else "more"})
    out.append({"label": "بحث جديد" if ar else "New search", "value": "بحث جديد" if ar else "new search"})
    return out


def clarification_for(pq: ParsedQuery, area_counts: dict[str, int]) -> Clarification | None:
    if pq.ambiguities:
        amb = pq.ambiguities[0]
        opts = [
            {"label_en": AREAS[o].name_en, "label_ar": AREAS[o].name_ar, "value": AREAS[o].name_en}
            for o in amb.options
            if o in AREAS
        ]
        names_en = " or ".join(o["label_en"] for o in opts)
        names_ar = " ولا ".join(o["label_ar"] for o in opts)
        return Clarification(
            question_en=f'"{amb.surface}" could mean {names_en}. Which one?',
            question_ar=f"تقصد {names_ar}؟",
            options=opts,
        )
    f = pq.filters
    has_location = bool(f.areas or f.compounds or f.near_landmark or f.unit_code)
    if (
        pq.intent == "search"
        and not has_location
        and not f.budget_max
        and not f.budget_min
        and not pq.soft_preferences
        and not f.monthly_installment_max
    ):
        top = sorted(area_counts, key=area_counts.get, reverse=True)[:5]
        opts = [
            {"label_en": AREAS[a].name_en, "label_ar": AREAS[a].name_ar, "value": AREAS[a].name_en}
            for a in top
        ]
        return Clarification(
            question_en="Which area are you looking in, and roughly what budget?",
            question_ar="بتدور في أنهي منطقة، وميزانيتك في حدود كام؟",
            options=opts,
        )
    return None


# ---------------------------------------------------------------- optional LLM narrative

_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def _allowed_numbers(facts: list[dict]) -> set[str]:
    allowed: set[str] = set()

    def add(v) -> None:
        if isinstance(v, bool) or v is None:
            return
        if isinstance(v, (int, float)):
            allowed.add(f"{v:g}")
            allowed.add(str(int(v)))
            for div in (1_000_000, 1_000):
                if v >= div:
                    q = v / div
                    for nd in (0, 1, 2):
                        allowed.add(f"{q:.{nd}f}".rstrip("0").rstrip(".") if nd else f"{q:.0f}")
        elif isinstance(v, str):
            allowed.update(_NUM.findall(v))
        elif isinstance(v, dict):
            for x in v.values():
                add(x)
        elif isinstance(v, list):
            for x in v:
                add(x)

    add(facts)
    return allowed


def grounded(text: str, facts: list[dict]) -> bool:
    allowed = _allowed_numbers(facts) | {str(i) for i in range(0, 11)}
    return all(n.replace(",", "") in allowed or n in allowed for n in _NUM.findall(text))


def llm_narrative(client, model: str, query: str, hits: list[UnitHit], lang: str) -> tuple[str | None, dict]:
    facts = [
        {
            "code": h.code,
            "title": h.title_en,
            "price_egp": h.price,
            "bua_m2": h.bua_m2,
            "bedrooms": h.bedrooms,
            "delivery": h.delivery,
            "plan": h.best_plan.model_dump() if h.best_plan else None,
            "reasons": [r.en for r in h.reasons],
        }
        for h in hits[:3]
    ]
    prompt = (
        f"User query: {query}\nTop units (the ONLY facts you may use):\n"
        f"{json.dumps(facts, ensure_ascii=False)}\n\n"
        f"Write 2 short sentences in {'Egyptian Arabic' if lang == 'ar' else 'English'} recommending which "
        "unit to look at first and why. Use only numbers present in the facts. No preamble."
    )
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=1500,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
        return (text if text and grounded(text, facts) else None), usage
    except Exception as e:
        log.warning("narrative failed: %s", e)
        return None, {}
