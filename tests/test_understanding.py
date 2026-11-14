"""Query understanding without a database: language detection, normalisation and the rule parser."""

from __future__ import annotations

import pytest

from propsearch.domain import ParsedQuery, SearchFilters
from propsearch.normalize import detect_language, normalize_arabic
from propsearch.understanding.rules import RuleParser

P = RuleParser()


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("3 غرف في التجمع تحت 8 مليون", "ar"),
        ("villa in Zayed, ready to move, garden", "en"),
        ("3ayez sha2a fel tagamo3 ta7t 7 mlyon", "franco"),
    ],
)
def test_detect_language(text, lang):
    assert detect_language(text) == lang


def test_normalize_arabic_folds_letter_variants():
    assert normalize_arabic("أحمد") == normalize_arabic("احمد")
    assert normalize_arabic("مدينة") == normalize_arabic("مدينه")


def test_arabic_query_to_filters():
    f = P.parse("3 غرف في التجمع تحت 8 مليون تقسيط 8 سنين").filters
    assert f.areas == ["new_cairo"]
    assert (f.bedrooms_min, f.bedrooms_max) == (3, 3)
    assert f.budget_max == 8_000_000
    assert f.installment_years_min == 8


def test_franco_query_to_filters():
    f = P.parse("3ayez sha2a fel tagamo3 ta7t 7 mlyon w mo2adam 10%").filters
    assert f.areas == ["new_cairo"]
    assert [t.value for t in f.unit_types] == ["apartment"]
    assert f.budget_max == 7_000_000
    assert f.down_payment_max_pct == 10


def test_english_query_with_landmark_and_monthly():
    pq = P.parse("apartment near AUC, quiet, good for families, monthly under 70k")
    assert pq.filters.near_landmark == "auc"
    assert pq.filters.monthly_installment_max == 70_000
    assert "quiet" in pq.soft_preferences


def test_ambiguous_zayed_asks():
    pq = P.parse("villa in Zayed, ready to move, garden")
    assert pq.ambiguities and set(pq.ambiguities[0].options) == {"sheikh_zayed", "new_zayed"}
    assert pq.filters.ready_to_move is True
    assert [a.value for a in pq.filters.amenities] == ["garden"]


def test_refinement_keeps_state_and_lowers_budget():
    first = P.parse("apartment in New Cairo under 8 million")
    second = P.parse("cheaper", first, {"median_price": 7_000_000})
    assert second.intent == "refine"
    assert second.filters.areas == ["new_cairo"]
    assert second.filters.budget_max == 6_300_000


def test_amenities_accumulate_across_turns():
    first = P.parse("villa in October")
    second = P.parse("with a garden", first)
    assert [a.value for a in second.filters.amenities] == ["garden"]
    assert second.filters.areas == first.filters.areas


def test_reset_clears_filters():
    first = P.parse("villa in October under 20 million")
    pq = P.parse("new search", first)
    assert pq.intent == "reset"
    assert pq.filters == SearchFilters()


def test_unknown_area_is_rejected_by_schema():
    with pytest.raises(ValueError):
        SearchFilters(areas=["atlantis"])


def test_parsed_query_round_trips_json():
    pq = P.parse("twin house in October with a garden, delivery 2027")
    assert ParsedQuery.model_validate(pq.model_dump(mode="json")) == pq
