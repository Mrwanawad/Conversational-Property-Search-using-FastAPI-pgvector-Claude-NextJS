"""End-to-end turns on a real Postgres + pgvector with the seeded synthetic inventory."""

from __future__ import annotations

import anthropic
import httpx
import pytest
from sqlalchemy import select

from propsearch.db import QueryLog, Unit
from propsearch.domain import SearchFilters
from propsearch.retrieval import sql
from propsearch.retrieval.relax import relax


def test_seeded_inventory(Session):
    with Session() as db:
        assert db.scalar(select(Unit.id).limit(1)) is not None


def test_every_hit_satisfies_hard_filters(engine, Session):
    r = engine.turn("3 غرف في التجمع تحت 8 مليون تقسيط 8 سنين")
    assert r.total_matches > 0 and r.units
    with Session() as db:
        for h in r.units:
            u = db.get(Unit, h.id)
            assert u.area == "new_cairo" and u.bedrooms == 3 and u.price <= 8_000_000
            assert u.status != "sold"


def test_answer_is_grounded_in_db_fields(engine, Session):
    """Prices, sizes and codes in the response come from the database, never from a model."""
    r = engine.turn("twin house in October with a garden, delivery 2027")
    with Session() as db:
        for h in r.units:
            u = db.get(Unit, h.id)
            assert (h.code, h.price, h.bua_m2, h.bedrooms) == (u.code, u.price, u.bua_m2, u.bedrooms)
            assert "garden" in u.amenities and u.delivery_year <= 2027


def test_reply_language_follows_query(engine):
    assert engine.turn("villa in October under 30 million").reply_language == "en"
    assert engine.turn("فيلا في أكتوبر تحت 30 مليون").reply_language == "ar"
    assert engine.turn("3ayez villa f october").reply_language == "ar"


def test_ambiguous_area_returns_clarification(engine):
    r = engine.turn("villa in Zayed, ready to move, garden")
    assert r.clarification is not None and not r.units
    r2 = engine.turn("Sheikh Zayed", r.session_id)
    assert r2.clarification is None
    assert r2.parsed.filters.areas == ["sheikh_zayed"] and r2.parsed.filters.ready_to_move


def test_cheaper_refinement_lowers_prices(engine):
    r1 = engine.turn("apartment in New Cairo")
    r2 = engine.turn("cheaper", r1.session_id)
    assert r2.parsed.intent == "refine"
    assert r2.parsed.filters.budget_max is not None
    assert all(h.price <= r2.parsed.filters.budget_max for h in r2.units)


def test_more_pages_without_repeats(engine):
    r1 = engine.turn("apartment in New Cairo")
    r2 = engine.turn("more", r1.session_id)
    assert r2.units and not {h.id for h in r1.units} & {h.id for h in r2.units}


def test_zero_result_is_relaxed_with_explanation(engine):
    r = engine.turn("penthouse in New Cairo with private pool under 9 million ready to move")
    if r.total_matches and not r.relaxations:
        pytest.skip("synthetic inventory happens to have an exact match")
    assert r.relaxations, r.message
    # steps apply in order; only the last one has to unlock units
    assert r.relaxations[-1].unlocked > 0 and r.total_matches > 0 and r.units


def test_impossible_budget_reports_price_floor(engine, Session):
    r = engine.turn("villa in New Cairo under 2 million")
    assert r.total_matches == 0
    with Session() as db:
        floor = sql.min_price(db, SearchFilters(areas=["new_cairo"], unit_types=["villa"]))
    assert floor and floor > 2_000_000
    assert "cheapest" in r.message.lower()


def test_relax_prefers_cheapest_change(Session):
    base = SearchFilters(areas=["new_cairo"], unit_types=["apartment"])
    with Session() as db:
        floor = sql.min_price(db, base)
        f = base.model_copy(update={"budget_max": int(floor * 0.95)})
        assert sql.count(db, f) == 0
        relaxed, applied, binding = relax(db, f)
    # a 10% budget raise is the cheapest single change that unlocks units
    assert len(applied) == 1 and applied[0].field == "budget_max" and applied[0].unlocked > 0
    assert relaxed.budget_max >= floor and relaxed.areas == ["new_cairo"]
    assert "budget_max" in [b["field"] for b in binding]


def test_turn_is_logged(engine, Session):
    r = engine.turn("chalet in the North Coast sea view")
    with Session() as db:
        ql = db.get(QueryLog, r.turn_id)
    assert ql.session_id == r.session_id and ql.total_matches == r.total_matches
    assert ql.latency_ms > 0


class _FailingMessages:
    def create(self, **_):
        raise anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))


class _FailingClient:
    class beta:
        messages = _FailingMessages()


def test_claude_parser_falls_back_to_rules(settings, Session):
    from propsearch.pipeline import SearchEngine

    eng = SearchEngine(settings.model_copy(update={"parser": "anthropic"}), Session, client=_FailingClient())
    r = eng.turn("3 غرف في التجمع تحت 8 مليون")
    assert r.parsed.parser == "rules_fallback"
    assert r.parsed.filters.areas == ["new_cairo"] and r.units


def test_relaxed_budget_reason_compares_with_the_asked_budget(engine):
    # Found during a browser pilot: after a +10% relaxation, a unit above the user's budget was
    # described as "under your budget" because the reason used the relaxed number.
    r = engine.turn("3ayez chalet fel sa7el 3ala el ba7r, 2 owad, ta7t 10 mlyon")
    asked = r.parsed.filters.budget_max
    over = [h for h in r.units if h.price > asked]
    for h in over:
        assert any(x.kind == "relaxed" and "over your" in x.en for x in h.reasons)
        assert not any("under your" in x.en for x in h.reasons)
