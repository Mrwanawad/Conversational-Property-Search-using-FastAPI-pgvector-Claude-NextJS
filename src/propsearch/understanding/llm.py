"""LLM tool calling -> Pydantic filters (Claude, or any OpenAI-compatible API such as Groq / OpenRouter).

The model sees the normalised query, the dictionary concepts the normaliser found, the previous
session filters and the catalog (areas, landmarks, compounds). It must call `set_search_filters`
with the FULL updated state. The tool schema is strict; the result is validated again with
Pydantic and unknown slugs are dropped. Any failure falls back to the rule parser.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

import anthropic
import httpx
from pydantic import ValidationError

from propsearch.data.catalog import AREA_GROUPS, AREAS, LANDMARKS
from propsearch.domain import Ambiguity, Amenity, Finishing, ParsedQuery, SearchFilters, UnitType
from propsearch.normalize import tag
from propsearch.understanding.rules import RuleParser

log = logging.getLogger(__name__)

# USD per million tokens (input, output). Claude: first-party API. Groq: list prices when this was written.
PRICING = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
}
TOOL_NAME = "set_search_filters"


def _nullable(t: str, **extra) -> dict:
    return {"type": [t, "null"], **extra}


def tool_schema(compound_slugs: list[str]) -> dict:
    props = {
        "areas": {"type": "array", "items": {"type": "string", "enum": sorted(AREAS)}},
        "compounds": {"type": "array", "items": {"type": "string", "enum": sorted(compound_slugs)}},
        "unit_types": {"type": "array", "items": {"type": "string", "enum": [t.value for t in UnitType]}},
        "bedrooms_min": _nullable("integer"),
        "bedrooms_max": _nullable("integer"),
        "budget_min": _nullable("integer", description="EGP, full number (8 million = 8000000)"),
        "budget_max": _nullable("integer", description="EGP, full number"),
        "bua_min": _nullable("integer", description="built-up area m2"),
        "bua_max": _nullable("integer"),
        "down_payment_max_pct": _nullable("number", description="0-100"),
        "down_payment_max_amount": _nullable("integer", description="EGP"),
        "installment_years_min": _nullable("integer"),
        "monthly_installment_max": _nullable("integer", description="EGP per month"),
        "ready_to_move": _nullable("boolean"),
        "delivery_year_max": _nullable("integer"),
        "finishing": {"type": "array", "items": {"type": "string", "enum": [f.value for f in Finishing]}},
        "amenities": {"type": "array", "items": {"type": "string", "enum": [a.value for a in Amenity]}},
        "near_landmark": _nullable("string", enum=[*sorted(LANDMARKS), None]),
        "max_distance_km": _nullable("number"),
        "unit_code": _nullable("string"),
        "soft_preferences": {
            "type": "string",
            "description": "Short English phrase list of wishes that are not hard filters "
            "(quiet, near schools, natural light, investment...). Empty string if none.",
        },
        "sort": {
            "type": "string",
            "enum": ["relevance", "price_asc", "price_desc", "price_per_m2_asc", "delivery_asc"],
        },
        "intent": {"type": "string", "enum": ["search", "refine", "reset", "more", "out_of_scope"]},
        "ambiguous_areas": {
            "type": "array",
            "description": "Set only when the user named a place that maps to several areas and context does "
            "not decide (e.g. 'Zayed' = sheikh_zayed or new_zayed). List the candidate slugs.",
            "items": {"type": "string", "enum": sorted(AREAS)},
        },
    }
    return {
        "name": TOOL_NAME,
        "description": "Record the complete, updated property-search state for this conversation turn.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": props,
            "required": list(props),
            "additionalProperties": False,
        },
    }


def system_prompt(compounds: list[tuple[str, str, str, str]]) -> str:
    areas = "\n".join(f"- {a.slug}: {a.name_en} / {a.name_ar}" for a in AREAS.values())
    groups = "\n".join(f"- {g}: {', '.join(v)}" for g, v in AREA_GROUPS.items())
    landmarks = "\n".join(f"- {lm.slug}: {lm.name_en} / {lm.name_ar}" for lm in LANDMARKS.values())
    comps = "\n".join(f"- {s}: {en} / {ar} ({area})" for s, en, ar, area in compounds)
    return f"""You turn Egyptian property-search messages into structured filters for a real-estate inventory.
Users write Egyptian Arabic, MSA, Franco-Arabic (Arabic in Latin letters with digits: 3=ع, 7=ح, 2=ء/ق, 5=خ)
and English, often mixed in one message.

Call the {TOOL_NAME} tool exactly once with the FULL state after applying the new message to the previous
state. Keep previous constraints the user did not change. On "new search" / "بحث جديد" start from empty.

Conventions:
- Money is EGP. "8 مليون", "8M", "8 mlyon" = 8000000. A bare small number in a budget context
  ("تحت 8", "budget 8") means millions. "مليون ونص" = 1500000. "نص مليون" = 500000. "800 الف" = 800000.
- "تحت/under/ta7t X" -> budget_max. "فوق/over/at least X" -> budget_min. "حوالي/around X" -> 0.85X..1.1X.
  A bare budget with no qualifier -> budget_max.
- "N غرف / N bedrooms / N owad / NBR" -> bedrooms_min = bedrooms_max = N. "اوضتين/غرفتين" = 2. "N+" -> min only.
- "تقسيط N سنين / N years installments" -> installment_years_min = N.
- "مقدم X%" -> down_payment_max_pct. "مقدم مليون" -> down_payment_max_amount. "قسط شهري X" -> monthly_installment_max.
- "استلام فوري / ready to move / جاهز" -> ready_to_move = true. "استلام 2027" -> delivery_year_max = 2027.
- "near/قريب من/جنب <landmark>" -> near_landmark, max_distance_km 12 unless the user gives a distance.
  "in <landmark city>" (e.g. في العلمين) -> also set its area.
- Refinements use the previous results context: "cheaper/ارخص" -> budget_max = 0.9 x min(previous budget_max,
  median price shown); "bigger/اكبر" -> bua_min = 1.15 x median built-up area shown; "more/غيرهم" -> intent "more".
- Hard filters only for things the inventory has as fields. Wishes like quiet, family friendly, near schools,
  natural light, view, privacy, investment/rental, smart home, pet friendly -> soft_preferences (English).
- intent: "search" for a fresh request, "refine" when modifying the previous state, "reset", "more",
  "out_of_scope" for messages that are not property search.
- Never invent values the user did not imply. Unknown -> null / empty list.

Areas (slug: English / Arabic):
{areas}
Area groups (expand to member slugs): east/شرق القاهرة, west/غرب القاهرة
{groups}
Landmarks:
{landmarks}
Compounds:
{comps}"""


@dataclass
class LlmUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    model: str = ""
    error: str | None = None
    extra: dict = field(default_factory=dict)


def _clean(raw: dict) -> dict:
    """Drop values the schema allows but the domain rejects (e.g. a null in an enum list)."""
    out = dict(raw)
    out["areas"] = [a for a in raw.get("areas") or [] if a in AREAS]
    if out.get("near_landmark") not in LANDMARKS:
        out["near_landmark"] = None
        out["max_distance_km"] = None
    return out


def to_parsed(raw: dict, text: str, parser: str) -> ParsedQuery:
    """Tool arguments -> validated ParsedQuery. Raises ValidationError / TypeError on bad output."""
    raw = _clean(dict(raw))
    soft = raw.pop("soft_preferences", "") or ""
    sort = raw.pop("sort", "relevance") or "relevance"
    intent = raw.pop("intent", "search") or "search"
    amb = raw.pop("ambiguous_areas", []) or []
    filters = SearchFilters(**{k: v for k, v in raw.items() if v is not None})
    return ParsedQuery(
        filters=filters,
        soft_preferences=soft.strip(),
        sort=sort,
        intent=intent,
        ambiguities=[Ambiguity(field="areas", surface=text, options=amb)] if len(amb) > 1 else [],
        parser=parser,
        confidence=1.0,
        unknown_terms=tag(text).unknown_terms,
    )


def state_message(text: str, previous: ParsedQuery | None, context: dict) -> str:
    tq = tag(text)
    state = {
        "previous_filters": previous.filters.model_dump(mode="json", exclude_defaults=True)
        if previous
        else {},
        "previous_soft_preferences": previous.soft_preferences if previous else "",
        "previous_results": context,
        "normalized_text": tq.normalized,
        "dictionary_concepts": tq.concepts(),
        "detected_language": tq.language,
    }
    return f"<state>{json.dumps(state, ensure_ascii=False)}</state>\n<message>{text}</message>"


class ClaudeParser:
    name = "claude"

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str,
        compounds: list[tuple[str, str, str, str]],
        effort: str = "low",
        fallback: RuleParser | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.effort = effort
        self.compound_slugs = [c[0] for c in compounds]
        self.tool = tool_schema(self.compound_slugs)
        self.system = system_prompt(compounds)
        self.fallback = fallback or RuleParser()
        self.last_usage = LlmUsage()

    def _messages(self, text: str, previous: ParsedQuery | None, context: dict) -> list[dict]:
        return [{"role": "user", "content": state_message(text, previous, context)}]

    def parse(
        self, text: str, previous: ParsedQuery | None = None, context: dict | None = None
    ) -> ParsedQuery:
        context = context or {}
        usage = LlmUsage(model=self.model)
        t = time.perf_counter()
        try:
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=4000,
                system=[{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}],
                tools=[self.tool],
                tool_choice={"type": "auto"},
                messages=self._messages(text, previous, context),
                output_config={"effort": self.effort},
                betas=["server-side-fallback-2026-07-01"],
                extra_body={"fallbacks": "default"},
            )
            usage.latency_ms = (time.perf_counter() - t) * 1000
            usage.input_tokens = (
                (resp.usage.input_tokens or 0)
                + (resp.usage.cache_read_input_tokens or 0)
                + (resp.usage.cache_creation_input_tokens or 0)
            )
            usage.output_tokens = resp.usage.output_tokens or 0
            p_in, p_out = PRICING.get(self.model, (0.0, 0.0))
            usage.cost_usd = round(
                (
                    (resp.usage.input_tokens or 0)
                    + 1.25 * (resp.usage.cache_creation_input_tokens or 0)
                    + 0.1 * (resp.usage.cache_read_input_tokens or 0)
                )
                * p_in
                / 1e6
                + usage.output_tokens * p_out / 1e6,
                6,
            )
            if resp.stop_reason == "refusal":
                raise ValueError("refusal")
            block = next((b for b in resp.content if b.type == "tool_use" and b.name == TOOL_NAME), None)
            if block is None:
                raise ValueError(f"no tool call (stop_reason={resp.stop_reason})")
            pq = to_parsed(block.input, text, self.name)
            self.last_usage = usage
            return pq
        except (
            anthropic.APIStatusError,
            anthropic.APIConnectionError,
            ValidationError,
            ValueError,
            KeyError,
            TypeError,
        ) as e:
            usage.error = f"{type(e).__name__}: {e}"[:300]
            usage.latency_ms = (time.perf_counter() - t) * 1000
            log.warning("claude parser failed, using rules: %s", usage.error)
            pq = self.fallback.parse(text, previous, context)
            pq.parser = "rules_fallback"
            self.last_usage = usage
            return pq


class OpenAICompatParser:
    """Same schema, prompt and fallback as ClaudeParser, over the OpenAI chat-completions API.

    Works with Groq (https://api.groq.com/openai/v1), OpenRouter (https://openrouter.ai/api/v1) and
    any other OpenAI-compatible endpoint. The tool call is forced; the arguments are validated with
    Pydantic exactly like the Claude path, and any failure falls back to the rule parser.
    """

    name = "llm"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        compounds: list[tuple[str, str, str, str]],
        reasoning_effort: str | None = "low",
        timeout_s: float = 15.0,
        fallback: RuleParser | None = None,
        http: httpx.Client | None = None,
        max_wait_s: float = 0.0,
    ) -> None:
        self.max_wait_s = max_wait_s  # >0 only for offline evals: wait out 429s instead of falling back
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.reasoning_effort = reasoning_effort or None
        self.http = http or httpx.Client(timeout=timeout_s, headers={"Authorization": f"Bearer {api_key}"})
        schema = tool_schema([c[0] for c in compounds])
        self.tool = {
            "type": "function",
            "function": {
                "name": TOOL_NAME,
                "description": schema["description"],
                "parameters": schema["input_schema"],
            },
        }
        self.system = system_prompt(compounds)
        self.fallback = fallback or RuleParser()
        self.last_usage = LlmUsage()

    def parse(
        self, text: str, previous: ParsedQuery | None = None, context: dict | None = None
    ) -> ParsedQuery:
        context = context or {}
        usage = LlmUsage(model=self.model)
        t = time.perf_counter()
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 2000,
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": state_message(text, previous, context)},
            ],
            "tools": [self.tool],
            "tool_choice": {"type": "function", "function": {"name": TOOL_NAME}},
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        try:
            resp = self.http.post(self.url, json=body)
            waited = 0.0
            while resp.status_code == 429 and waited < self.max_wait_s:
                pause = min(float(resp.headers.get("retry-after", 5)), 60.0) + 0.5
                time.sleep(pause)
                waited += pause
                t = time.perf_counter()
                resp = self.http.post(self.url, json=body)
            usage.latency_ms = (time.perf_counter() - t) * 1000
            if resp.status_code >= 400:
                raise ValueError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
            u = data.get("usage") or {}
            usage.input_tokens = u.get("prompt_tokens", 0) or 0
            usage.output_tokens = u.get("completion_tokens", 0) or 0
            p_in, p_out = PRICING.get(self.model, (0.0, 0.0))
            usage.cost_usd = round((usage.input_tokens * p_in + usage.output_tokens * p_out) / 1e6, 6)
            calls = data["choices"][0]["message"].get("tool_calls") or []
            call = next((c for c in calls if c["function"]["name"] == TOOL_NAME), None)
            if call is None:
                raise ValueError("no tool call")
            pq = to_parsed(json.loads(call["function"]["arguments"]), text, self.name)
            self.last_usage = usage
            return pq
        except (httpx.HTTPError, ValidationError, ValueError, KeyError, TypeError, IndexError) as e:
            usage.error = f"{type(e).__name__}: {e}"[:300]
            usage.latency_ms = (time.perf_counter() - t) * 1000
            log.warning("llm parser failed, using rules: %s", usage.error)
            pq = self.fallback.parse(text, previous, context)
            pq.parser = "rules_fallback"
            self.last_usage = usage
            return pq


class GatedParser:
    """Rules first; the LLM only sees messages the rules could not fully read (unknown terms).

    Most traffic stays deterministic and ~10 ms, and LLM calls (latency, cost, free-tier quota) are
    spent where the dictionary has a gap. The LLM parser still falls back to rules on any failure.
    """

    name = "gated"

    def __init__(self, rules: RuleParser, llm) -> None:
        self.rules = rules
        self.llm = llm
        self.model = llm.model
        self.last_usage = LlmUsage()

    def parse(
        self, text: str, previous: ParsedQuery | None = None, context: dict | None = None
    ) -> ParsedQuery:
        pq = self.rules.parse(text, previous, context)
        if not pq.unknown_terms:
            self.last_usage = LlmUsage()
            return pq
        out = self.llm.parse(text, previous, context)
        self.last_usage = self.llm.last_usage
        return out
