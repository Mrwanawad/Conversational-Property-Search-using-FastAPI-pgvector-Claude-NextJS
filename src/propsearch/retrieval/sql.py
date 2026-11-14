"""Hard filters -> SQL. Every SearchFilters field is one predicate; payment-plan fields must hold on the same plan."""

from __future__ import annotations

from sqlalchemy import Select, and_, exists, func, select
from sqlalchemy.orm import Session

from propsearch.data.catalog import LANDMARKS, haversine_km
from propsearch.db import Compound, PaymentPlan, Unit
from propsearch.domain import SearchFilters

CURRENT_YEAR = 2026


_COORDS: dict[int, tuple[float, float]] = {}


def compound_coords(session: Session) -> dict[int, tuple[float, float]]:
    """Compound coordinates are static inventory data; load once per process."""
    if not _COORDS:
        for cid, lat, lng in session.execute(select(Compound.id, Compound.lat, Compound.lng)):
            _COORDS[cid] = (lat, lng)
    return _COORDS


def compounds_within(session: Session, landmark: str, radius_km: float) -> list[int]:
    lm = LANDMARKS[landmark]
    return [
        cid
        for cid, (lat, lng) in compound_coords(session).items()
        if haversine_km(lm.lat, lm.lng, lat, lng) <= radius_km
    ]


def plan_predicates(f: SearchFilters) -> list:
    preds = []
    if f.installment_years_min is not None:
        preds.append(PaymentPlan.years >= f.installment_years_min)
    if f.down_payment_max_pct is not None:
        preds.append(PaymentPlan.down_payment_pct <= f.down_payment_max_pct)
    if f.down_payment_max_amount is not None:
        preds.append(Unit.price * PaymentPlan.down_payment_pct / 100.0 <= f.down_payment_max_amount)
    if f.monthly_installment_max is not None:
        preds.append(
            Unit.price
            * (1 - PaymentPlan.down_payment_pct / 100.0)
            / (func.greatest(PaymentPlan.years, 1) * 12)
            <= f.monthly_installment_max
        )
    if preds:
        preds.append(PaymentPlan.years > 0)
    return preds


def where_clauses(session: Session, f: SearchFilters) -> list:
    c = [Unit.status != "sold"]
    if f.areas:
        c.append(Unit.area.in_(f.areas))
    if f.compounds:
        c.append(Unit.compound_id.in_(select(Compound.id).where(Compound.slug.in_(f.compounds))))
    if f.unit_types:
        c.append(Unit.unit_type.in_([t.value for t in f.unit_types]))
    if f.bedrooms_min is not None:
        c.append(Unit.bedrooms >= f.bedrooms_min)
    if f.bedrooms_max is not None:
        c.append(Unit.bedrooms <= f.bedrooms_max)
    if f.budget_min is not None:
        c.append(Unit.price >= f.budget_min)
    if f.budget_max is not None:
        c.append(Unit.price <= f.budget_max)
    if f.bua_min is not None:
        c.append(Unit.bua_m2 >= f.bua_min)
    if f.bua_max is not None:
        c.append(Unit.bua_m2 <= f.bua_max)
    if f.ready_to_move:
        c.append(Unit.is_ready.is_(True))
    if f.delivery_year_max is not None:
        c.append(Unit.delivery_year <= f.delivery_year_max)
    if f.finishing:
        c.append(Unit.finishing.in_([x.value for x in f.finishing]))
    if f.amenities:
        c.append(Unit.amenities.contains([a.value for a in f.amenities]))
    if f.near_landmark and f.max_distance_km:
        c.append(Unit.compound_id.in_(compounds_within(session, f.near_landmark, f.max_distance_km) or [-1]))
    if f.unit_code:
        c.append(func.upper(Unit.code) == f.unit_code.upper())
    pp = plan_predicates(f)
    if pp:
        c.append(exists().where(and_(PaymentPlan.compound_id == Unit.compound_id, *pp)))
    return c


def candidate_query(session: Session, f: SearchFilters) -> Select:
    return select(Unit.id).where(*where_clauses(session, f))


def candidate_ids(session: Session, f: SearchFilters, limit: int = 2000) -> list[int]:
    return list(session.scalars(candidate_query(session, f).order_by(Unit.id).limit(limit)))


def min_price(session: Session, f: SearchFilters) -> int | None:
    """Cheapest unit matching every filter except the budget: the price floor shown when the budget is too low."""
    nb = f.model_copy(update={"budget_min": None, "budget_max": None})
    return session.execute(select(func.min(Unit.price)).where(*where_clauses(session, nb))).scalar_one()


def count(session: Session, f: SearchFilters) -> int:
    return session.execute(
        select(func.count()).select_from(Unit).where(*where_clauses(session, f))
    ).scalar_one()
