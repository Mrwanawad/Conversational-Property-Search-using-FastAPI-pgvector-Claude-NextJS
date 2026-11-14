"""OpenAI-compatible parser (Groq / OpenRouter) and the rules-first gate, with HTTP mocked."""

from __future__ import annotations

import json

import httpx

from propsearch.understanding.llm import TOOL_NAME, GatedParser, OpenAICompatParser
from propsearch.understanding.rules import RuleParser

COMPOUNDS = [("quartz_residence", "Quartz Residence", "كوارتز ريزيدنس", "new_cairo")]


def _tool_args(**over) -> dict:
    args = {
        "areas": ["new_cairo"],
        "compounds": [],
        "unit_types": ["apartment"],
        "bedrooms_min": 2,
        "bedrooms_max": 2,
        "budget_min": None,
        "budget_max": 6_000_000,
        "bua_min": None,
        "bua_max": None,
        "down_payment_max_pct": None,
        "down_payment_max_amount": None,
        "installment_years_min": None,
        "monthly_installment_max": None,
        "ready_to_move": None,
        "delivery_year_max": None,
        "finishing": [],
        "amenities": [],
        "near_landmark": None,
        "max_distance_km": None,
        "unit_code": None,
        "soft_preferences": "",
        "sort": "relevance",
        "intent": "search",
        "ambiguous_areas": [],
    }
    args.update(over)
    return args


def _parser(handler, calls: list | None = None) -> OpenAICompatParser:
    def wrapped(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(json.loads(request.content))
        return handler(request)

    http = httpx.Client(transport=httpx.MockTransport(wrapped))
    return OpenAICompatParser(
        "https://api.groq.com/openai/v1", "k", "openai/gpt-oss-120b", COMPOUNDS, "low", http=http
    )


def _ok(args: dict) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "tool_calls": [{"function": {"name": TOOL_NAME, "arguments": json.dumps(args)}}]
                    }
                }
            ],
            "usage": {"prompt_tokens": 3000, "completion_tokens": 200},
        },
    )


def test_tool_call_becomes_validated_filters():
    calls: list = []
    p = _parser(lambda r: _ok(_tool_args()), calls)
    pq = p.parse("two-bed flat in the Fifth Settlement, max 6M")
    assert pq.parser == "llm"
    assert pq.filters.areas == ["new_cairo"]
    assert pq.filters.bedrooms_min == pq.filters.bedrooms_max == 2
    assert pq.filters.budget_max == 6_000_000
    body = calls[0]
    assert body["tool_choice"]["function"]["name"] == TOOL_NAME
    assert body["reasoning_effort"] == "low"
    assert p.last_usage.cost_usd > 0


def test_unknown_slugs_are_dropped():
    p = _parser(lambda r: _ok(_tool_args(areas=["new_cairo", "atlantis"], near_landmark="nowhere")))
    pq = p.parse("flat in new cairo")
    assert pq.filters.areas == ["new_cairo"]
    assert pq.filters.near_landmark is None


def test_http_error_falls_back_to_rules():
    p = _parser(lambda r: httpx.Response(429, json={"error": {"message": "rate limited"}}))
    pq = p.parse("3 bedroom apartment in New Cairo under 8 million")
    assert pq.parser == "rules_fallback"
    assert pq.filters.budget_max == 8_000_000
    assert "429" in p.last_usage.error


def test_invalid_arguments_fall_back_to_rules():
    p = _parser(lambda r: _ok(_tool_args(bedrooms_min=99)))
    assert p.parse("3 bedroom apartment in New Cairo").parser == "rules_fallback"


def test_gate_skips_llm_when_rules_understand_everything():
    calls: list = []
    gated = GatedParser(RuleParser(), _parser(lambda r: _ok(_tool_args()), calls))
    pq = gated.parse("3 bedroom apartment in New Cairo under 8 million")
    assert pq.parser == "rules"
    assert calls == []


def test_gate_calls_llm_on_unknown_terms():
    calls: list = []
    gated = GatedParser(RuleParser(), _parser(lambda r: _ok(_tool_args(ready_to_move=True)), calls))
    pq = gated.parse("standalone villa in 6 October, keys now, budget 25 million tops")
    assert len(calls) == 1
    assert pq.parser == "llm"
    assert pq.filters.ready_to_move is True
